from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import jev_gateway.service as service_module
from jev_gateway.api import create_app
from jev_gateway.config import Adapter, Server, Settings, Upstream
from jev_gateway.schema import SystemOneRequest
from jev_gateway.service import Gateway, RequestError


def _settings(
    *,
    temperature: float = 1.0,
    double_round_robin: bool = False,
    callsigns: list[str] | None = None,
    server: Server | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Settings:
    return Settings(
        upstream=Upstream(
            base_url="https://upstream.invalid",
            model="upstream-model",
            api_key_env="UPSTREAM_TEST_KEY",
            timeout=1.0,
            top_logprobs=20,
            extra_body=extra_body or {},
        ),
        adapter=Adapter(
            temperature=temperature,
            double_round_robin=double_round_robin,
            callsigns=[] if callsigns is None else callsigns,
        ),
        server=server or Server(),
    )


def _payload(questions: dict[str, Any], state: Any = "evidence") -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {"model": "jev-latest", "state": state, "questions": questions}
    )


def _tokens_from_request(request: httpx.Request) -> list[str]:
    body = json.loads(request.content)
    user = body["messages"][-1]["content"]
    match = re.search(r"Output exactly one of: ([^.]+)\.", user)
    assert match
    return [item.strip() for item in match.group(1).split(",")]


def _completion(
    tokens: list[str],
    probabilities: list[float] | None = None,
    *,
    usage: dict[str, Any] | None = None,
) -> httpx.Response:
    probabilities = probabilities or [1 / len(tokens)] * len(tokens)
    entries = [
        {"token": token, "logprob": math.log(probability)}
        for token, probability in zip(tokens, probabilities, strict=True)
    ]
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {
                                **entries[0],
                                "top_logprobs": entries,
                            }
                        ]
                    },
                }
            ],
            "usage": usage
            or {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    )


def _probability_transport(
    seen: list[httpx.Request] | None = None,
    probabilities: list[float] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        tokens = _tokens_from_request(request)
        supplied = probabilities if probabilities and len(probabilities) == len(tokens) else None
        return _completion(tokens, supplied)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_gateway_returns_all_official_shapes_and_preserves_usage(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "upstream-secret")
    seen: list[httpx.Request] = []
    client = httpx.AsyncClient(transport=_probability_transport(seen))
    gateway = Gateway(_settings(), client)
    payload = _payload(
        {
            "truth-check": {
                "type": "noul",
                "instructions": {"ask": "true?"},
                "criteria": {"true": "supported", "false": "unsupported"},
            },
            "choice-id": {
                "type": "choice",
                "instructions": ["choose"],
                "criteria": {"private-key-a": {"label": "Alpha"}, "private-key-b": "Beta"},
            },
            "score-id": {
                "type": "score",
                "instructions": "score",
                "criteria": ["low", {"level": "high"}],
            },
        },
        state={"nested": [1, None, True]},
    )

    response, diagnostics = await gateway.evaluate(payload, "request-1")

    assert isinstance(response["answers"]["truth-check"]["noul"], float)
    assert response["answers"]["choice-id"]["choice"] in {"private-key-a", "private-key-b"}
    score = response["answers"]["score-id"]
    assert score["score"] == pytest.approx(0.5)
    assert score["legend"] == {"0": "low", "1": '{"level":"high"}'}
    assert response["usage"] == {"input_tokens": 21, "output_tokens": 3}
    assert diagnostics["usage"]["prompt_tokens_details"]["cached_tokens"] == 9
    assert diagnostics["request_id"] == "request-1"
    assert diagnostics["branches"][1]["messages"] == gateway.plan(payload)[1]["messages"]
    assert diagnostics["branches"][1]["mapping"]
    assert diagnostics["branches"][1]["raw_probabilities"]
    assert diagnostics["branches"][1]["label_logprobs"]
    assert diagnostics["temperature"] == 1.0
    assert diagnostics["aggregation_method"] == "single"
    assert diagnostics["config"] == gateway.settings.model_dump(mode="json")
    assert len(seen) == 3
    await client.aclose()




@pytest.mark.asyncio
async def test_temperature_is_applied_once_before_aggregation(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    client = httpx.AsyncClient(
        transport=_probability_transport(probabilities=[0.9, 0.1])
    )
    gateway = Gateway(_settings(temperature=2.0), client)
    payload = _payload(
        {"q": {"type": "choice", "instructions": "pick", "criteria": {"x": "X", "y": "Y"}}}
    )

    response, _ = await gateway.evaluate(payload)

    assert response["answers"]["q"]["probabilities"] == pytest.approx({"x": 0.75, "y": 0.25})
    await client.aclose()


@pytest.mark.asyncio
async def test_incomplete_top_k_sets_zero_and_emits_diagnostic_warning(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")

    def handler(request: httpx.Request) -> httpx.Response:
        tokens = _tokens_from_request(request)
        return _completion(tokens[:1], [0.6])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = Gateway(_settings(), client)
    payload = _payload(
        {"q": {"type": "choice", "instructions": "pick", "criteria": {"x": "X", "y": "Y"}}}
    )

    response, diagnostics = await gateway.evaluate(payload)

    assert response["answers"]["q"]["probabilities"] == {"x": 1.0, "y": 0.0}
    branch = diagnostics["branches"][0]
    assert branch["missing_labels"] == ["y"]
    assert branch["observed_label_mass"] == pytest.approx(0.6)
    assert branch["label_mass_upper_bound"] <= 1
    assert "incomplete_labels" in {warning["code"] for warning in branch["warnings"]}
    await client.aclose()


@pytest.mark.asyncio
async def test_round_robin_plans_directional_pairs_and_aggregates_usage(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    seen: list[httpx.Request] = []
    client = httpx.AsyncClient(transport=_probability_transport(seen, [0.75, 0.25]))
    gateway = Gateway(_settings(double_round_robin=True), client)
    payload = _payload(
        {
            "q": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"a": "A option", "b": "B option", "c": "C option"},
            }
        }
    )

    plan = gateway.plan(payload)
    response, diagnostics = await gateway.evaluate(payload)

    assert len(plan) == 6
    assert {tuple(branch["pair"]) for branch in plan} == {
        ("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("b", "c"), ("c", "b")
    }
    assert len(seen) == 6
    assert response["usage"] == {"input_tokens": 42, "output_tokens": 6}
    assert diagnostics["usage"]["prompt_tokens_details"]["cached_tokens"] == 18
    assert sum(response["answers"]["q"]["probabilities"].values()) == pytest.approx(1)
    await client.aclose()


@pytest.mark.asyncio
async def test_all_questions_are_validated_before_any_upstream_call(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    settings = _settings(callsigns=["A", "B"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = Gateway(settings, client)
    payload = _payload(
        {
            "valid": {"type": "noul", "instructions": "ok"},
            "invalid-later": {
                "type": "choice",
                "instructions": "too many",
                "criteria": {"a": "A", "b": "B", "c": "C"},
            },
        }
    )

    with pytest.raises(RequestError) as caught:
        await gateway.evaluate(payload)

    assert caught.value.status == 422
    assert calls == 0
    await client.aclose()


def test_call_cap_is_rejected_before_expanding_messages(monkeypatch) -> None:
    settings = _settings(
        double_round_robin=True,
        server=Server(max_calls_per_request=10),
    )
    payload = _payload(
        {
            "q": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {str(index): f"option {index}" for index in range(4)},
            }
        },
        state="large state would otherwise be copied into every branch",
    )
    monkeypatch.setattr(
        service_module,
        "build_plan",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not expand")),
    )

    with pytest.raises(RequestError) as caught:
        Gateway(settings).plan(payload)

    assert caught.value.status == 422




def test_api_auth_key_separation_headers_models_limits_and_health(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "upstream-secret")
    monkeypatch.setenv("DOWNSTREAM_TEST_KEY", "downstream-secret")
    seen: list[httpx.Request] = []
    upstream = httpx.AsyncClient(transport=_probability_transport(seen, [0.8, 0.2]))
    settings = _settings(server=Server(api_key_env="DOWNSTREAM_TEST_KEY"))
    app = create_app(settings, client=upstream)

    with TestClient(app) as local:
        assert local.get("/health/live").json() == {"status": "ok", "scope": "process"}
        health = local.get("/health")
        assert health.status_code == 200
        assert health.json()["scope"] == "configuration"
        assert local.get("/v1/models").status_code == 401

        headers = {
            "Authorization": "Bearer downstream-secret",
            "x-typesafe-request-id": "client-request-id",
        }
        models = local.get("/v1/models", headers=headers).json()
        assert models["object"] == "list"
        assert {row["name"] for row in models["models"]} == {"jev-latest", "upstream-model"}
        assert {row["id"] for row in models["data"]} == {"jev-latest", "upstream-model"}
        limits = local.get("/v1/limits", headers=headers).json()
        assert limits == {
            "max_answers_per_question": 26,
            "max_questions": 64,
            "max_body_bytes": 2_097_152,
            "max_calls_per_request": 256,
            "max_concurrent_requests": 16,
            "max_concurrent_calls": 8,
            "max_images": 8,
            "max_image_bytes": 12_582_912,
            "max_total_image_bytes": 33_554_432,
            "max_image_pixels": 16_000_000,
        }
        result = local.post(
            "/v1/systemone",
            headers=headers,
            json={
                "model": "jev-latest",
                "state": "state",
                "questions": {"q": {"type": "noul", "instructions": "true?"}},
            },
        )

    assert result.status_code == 200
    assert result.headers["x-typesafe-request-id"] == "client-request-id"
    assert result.headers["x-jev-config-id"] == settings.config_id
    assert result.json()["answers"]["q"]["noul"] == pytest.approx(0.8)
    assert seen[0].headers["authorization"] == "Bearer upstream-secret"
    sent = json.loads(seen[0].content)
    assert sent["model"] == "upstream-model"
    assert sent["max_tokens"] == 1
    assert sent["temperature"] == 1
    assert sent["logprobs"] is True
    assert sent["stream"] is False
    logged = capsys.readouterr().err
    assert "evaluation.completed" in logged
    assert "upstream-secret" not in logged
    assert "state" not in logged


def test_stderr_includes_warning_codes_without_trace(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion(_tokens_from_request(request)[:1], [0.6])

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_settings(), client=upstream)
    with TestClient(app) as local:
        result = local.post(
            "/v1/systemone",
            json={
                "model": "jev-latest",
                "state": "TRACE-MUST-STAY-PRIVATE",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"x": "X", "y": "Y"},
                    }
                },
            },
        )

    assert result.status_code == 200
    event = json.loads(capsys.readouterr().err.strip())
    assert "incomplete_labels" in event["branches"][0]["warning_codes"]
    assert "TRACE-MUST-STAY-PRIVATE" not in json.dumps(event)


def test_body_limit_and_sanitized_validation_error_have_request_id(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    upstream = httpx.AsyncClient(transport=_probability_transport())
    app = create_app(_settings(server=Server(max_body_bytes=80)), client=upstream)

    with TestClient(app) as local:
        too_large = local.post(
            "/v1/systemone",
            content=b"x" * 81,
            headers={"content-type": "application/json"},
        )
        invalid = local.post(
            "/v1/systemone",
            json={"model": "jev-latest", "state": "TOP-SECRET", "questions": {}},
        )

    assert too_large.status_code == 413
    assert too_large.headers["x-typesafe-request-id"]
    assert invalid.status_code == 422
    assert invalid.headers["x-typesafe-request-id"]
    assert "TOP-SECRET" not in invalid.text
    detail = invalid.json()["error"]["details"][0]
    assert set(detail) == {"loc", "msg", "type"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, 429), (401, 502), (500, 503), (400, 502)],
)
def test_upstream_errors_are_safe(monkeypatch, status: int, expected: int) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    secret = "PRIVATE-UPSTREAM-BODY"
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text=secret))
    )
    app = create_app(_settings(), client=upstream)

    with TestClient(app) as local:
        result = local.post(
            "/v1/systemone",
            json={
                "model": "jev-latest",
                "state": "PRIVATE-STATE",
                "questions": {"q": {"type": "noul", "instructions": "ask"}},
            },
        )

    assert result.status_code == expected
    assert secret not in result.text
    assert "PRIVATE-STATE" not in result.text
    assert result.headers["x-typesafe-request-id"]
    if expected in {429, 503, 529}:
        assert result.headers["retry-after"] == "1"


@pytest.mark.asyncio
async def test_timeout_and_missing_usage_are_explicit_502_or_504(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("PRIVATE timeout detail", request=request)

    timeout_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
    gateway = Gateway(_settings(), timeout_client)
    payload = _payload({"q": {"type": "noul", "instructions": "ask"}})
    with pytest.raises(RequestError) as timeout_error:
        await gateway.evaluate(payload)
    assert timeout_error.value.status == 504
    assert "PRIVATE" not in timeout_error.value.message
    await timeout_client.aclose()

    def missing_usage(request: httpx.Request) -> httpx.Response:
        response = _completion(_tokens_from_request(request))
        content = response.json()
        content.pop("usage")
        return httpx.Response(200, json=content)

    usage_client = httpx.AsyncClient(transport=httpx.MockTransport(missing_usage))
    with pytest.raises(RequestError) as usage_error:
        await Gateway(_settings(), usage_client).evaluate(payload)
    assert usage_error.value.status == 502
    await usage_client.aclose()


@pytest.mark.asyncio
async def test_one_branch_failure_cancels_siblings(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        rendered = body["messages"][-1]["content"]
        if "FAIL-BRANCH" in rendered:
            await slow_started.wait()
            raise httpx.ReadTimeout("timed out", request=request)
        slow_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise
        return _completion(_tokens_from_request(request))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = Gateway(_settings(), client)
    payload = _payload(
        {
            "fail": {"type": "noul", "instructions": "FAIL-BRANCH"},
            "slow": {"type": "noul", "instructions": "SLOW-BRANCH"},
        }
    )

    with pytest.raises(RequestError) as caught:
        await gateway.evaluate(payload)

    assert caught.value.status == 504
    assert slow_cancelled.is_set()
    await client.aclose()


@pytest.mark.asyncio
async def test_concurrent_request_overflow_is_529(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return _completion(_tokens_from_request(request))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = _settings(server=Server(max_concurrent_requests=1))
    gateway = Gateway(settings, client)
    payload = _payload({"q": {"type": "noul", "instructions": "ask"}})
    first = asyncio.create_task(gateway.evaluate(payload))
    await entered.wait()

    with pytest.raises(RequestError) as overflow:
        await gateway.evaluate(payload)

    assert overflow.value.status == 529
    release.set()
    await first
    await client.aclose()


def test_non_ascii_configured_downstream_key_does_not_crash_auth(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    monkeypatch.setenv("DOWNSTREAM_TEST_KEY", "密钥")
    upstream = httpx.AsyncClient(transport=_probability_transport())
    app = create_app(
        _settings(server=Server(api_key_env="DOWNSTREAM_TEST_KEY")), client=upstream
    )

    with TestClient(app) as local:
        result = local.get(
            "/v1/models", headers={"authorization": "Bearer wrong-ascii-token"}
        )

    assert result.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/evaluate"])
async def test_client_disconnect_cancels_inflight_evaluation(monkeypatch, path: str) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _completion(_tokens_from_request(request))

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_settings(), client=upstream)
    payload = {
        "model": "jev-latest",
        "state": "state",
        "questions": {"q": {"type": "noul", "instructions": "ask"}},
    }
    body = json.dumps(payload if path == "/v1/systemone" else {
        "request": payload, "execution": {},
    }).encode()
    first_receive = True

    async def receive() -> dict[str, Any]:
        nonlocal first_receive
        if first_receive:
            first_receive = False
            return {"type": "http.request", "body": body, "more_body": False}
        await entered.wait()
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=2)

    assert cancelled.is_set()
    await upstream.aclose()
