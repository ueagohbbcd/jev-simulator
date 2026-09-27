"""The extended endpoint is a per-request inference plan over one deployment."""
from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from jev_gateway.api import create_app
from jev_gateway.config import Adapter, InferenceConfig, Prompt, Server, Settings, Upstream
from jev_gateway.service import Gateway


def base_settings(*, server: Server | None = None) -> Settings:
    return Settings(
        upstream=Upstream(
            base_url="https://upstream.invalid", model="deployment-model",
            api_key_env="UPSTREAM_TEST_KEY", top_logprobs=12,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        adapter=Adapter(temperature=3, double_round_robin=True,
                        callsigns=["service-left", "service-right"]),
        prompt=Prompt(system="SERVICE PROMPT {{output}}"),
        server=server or Server(),
    )


def noul_request() -> dict:
    return {"model": "jev-latest", "state": "PRIVATE-STATE",
            "questions": {"q": {"type": "noul", "instructions": "ask"}}}


def token_completion() -> httpx.Response:
    entries = [{"token": label, "logprob": math.log(probability)}
               for label, probability in [("Yes", .8), ("No", .2)]]
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "length", "logprobs": {
            "content": [{**entries[0], "top_logprobs": entries}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    })


def reported_execution() -> dict:
    return {"adapter": {"mode": "reported_probability", "temperature": 2},
            "prompt": {"system": "Return only a JSON probability object."},
            "generation": {"temperature": .25, "max_tokens": 44}}


def test_dry_run_uses_code_defaults_and_never_calls_upstream(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    cfg = base_settings()
    calls = []
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: calls.append(request) or token_completion()))
    app = create_app(cfg, client=upstream)
    choice = {"model": "jev-latest", "state": "s", "questions": {
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"x": "X", "y": "Y"}}}}
    with TestClient(app) as local:
        result = local.post("/v1/evaluate", json={
            "request": choice, "execution": {}, "dry_run": True,
        })
    assert result.status_code == 200
    body = result.json()
    assert body["dry_run"] is True
    assert body["config_id"] == result.headers["x-jev-config-id"]
    assert body["config_id"] != cfg.config_id
    assert body["execution"]["adapter"] == {
        "mode": "token_logprobs", "temperature": 1.0,
        "double_round_robin": False, "callsigns": [],
    }
    assert body["execution"]["generation"] is None
    assert body["plan"]["request_count"] == 1
    assert body["plan"]["requests"][0]["mapping"] == {"A": "x", "B": "y"}
    assert "SERVICE PROMPT" not in body["plan"]["requests"][0]["messages"][0]["content"]
    assert body["warnings"] == [] and calls == []
    assert app.state.settings is cfg
    assert app.state.gateway.settings is cfg


def test_dry_run_warnings_and_effective_inference_config(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    app = create_app(base_settings(), client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("dry run called upstream"))))
    execution = {"adapter": {"mode": "reported_probability",
                             "double_round_robin": True, "callsigns": ["left", "right"]},
                 "prompt": {"system": "Return JSON probabilities."},
                 "diagnostics": {"low_mass_threshold": .8}}
    with TestClient(app) as local:
        result = local.post("/v1/evaluate", json={
            "request": noul_request(), "execution": execution, "dry_run": True,
        })
    assert result.status_code == 200
    body = result.json()
    assert body["execution"]["generation"] == {"temperature": 0, "max_tokens": 1024}
    assert body["plan"]["request_count"] == 1
    assert body["plan"]["requests"][0]["mapping"] == {"Yes": "true", "No": "false"}
    assert {warning["code"] for warning in body["warnings"]} == {
        "round_robin_unused", "callsigns_unused", "low_mass_threshold_unused"}
    assert all(set(warning) == {"code", "field", "message"} for warning in body["warnings"])
    assert "PRIVATE-STATE" not in json.dumps(body["warnings"])


def test_execute_uses_deployment_upstream_and_preserves_compatibility(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "PRIVATE-KEY")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer PRIVATE-KEY"
        assert body["model"] == "deployment-model"
        assert body["thinking"] == {"type": "disabled"}
        if "response_format" in body:
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"Yes":0.8,"No":0.2}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5},
            })
        return token_completion()

    cfg = base_settings()
    app = create_app(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as local:
        extended = local.post("/v1/evaluate", json={
            "request": noul_request(), "execution": reported_execution(),
        })
        compatible = local.post("/v1/systemone", json=noul_request())
    assert extended.status_code == compatible.status_code == 200
    body = extended.json()
    assert body["dry_run"] is False
    assert body["config_id"] == extended.headers["x-jev-config-id"]
    assert body["config_id"] != compatible.headers["x-jev-config-id"] == cfg.config_id
    assert body["result"]["answers"]["q"]["noul"] == pytest.approx(2 / 3)
    assert body["result"]["usage"] == {"input_tokens": 7, "output_tokens": 5}
    assert set(body) == {"dry_run", "config_id", "execution", "warnings", "result"}
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["temperature"] == .25 and seen[0]["max_tokens"] == 44
    assert "logprobs" not in seen[0]
    assert seen[1]["logprobs"] is True and seen[1]["top_logprobs"] == 12
    assert "PRIVATE-STATE" not in capsys.readouterr().err
    assert app.state.settings is cfg


@pytest.mark.parametrize("execution", [
    None,
    {"upstream": {"model": "unauthorized"}},
    {"server": {"max_calls_per_request": 999}},
    {"profile": "other"},
    {"overrides": {"adapter": {"temperature": 2}}},
    {"adapter": {"unknown": True}},
    {"adapter": {"mode": "reported_probability"}},
    {"generation": {"max_tokens": 2}},
])
def test_invalid_execution_is_sanitized_422_without_network(monkeypatch, execution) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    app = create_app(base_settings(), client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("invalid execution called upstream"))))
    body = {"request": noul_request(), "dry_run": True}
    if execution is not None:
        body["execution"] = execution
    with TestClient(app) as local:
        result = local.post("/v1/evaluate", json=body)
    assert result.status_code == 422
    assert result.headers["x-jev-config-id"] == app.state.settings.config_id
    assert result.json()["error"]["details"]
    assert "PRIVATE-STATE" not in result.text
    if execution == {"adapter": {"mode": "reported_probability"}}:
        assert "prompt.system" in result.text
    if execution == {"generation": {"max_tokens": 2}}:
        assert "generation" in result.text


def test_failure_after_resolution_uses_effective_header_and_call_cap(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    cfg = base_settings(server=Server(max_calls_per_request=1))
    app = create_app(cfg, client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("over-budget request called upstream"))))
    execution = {"adapter": {"double_round_robin": True}}
    effective_id = Gateway(cfg).with_execution(
        InferenceConfig.model_validate(execution)).settings.config_id
    request = {"model": "jev-latest", "state": "s", "questions": {
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"x": "X", "y": "Y"}}}}
    with TestClient(app) as local:
        result = local.post("/v1/evaluate", json={
            "request": request, "execution": execution, "dry_run": True,
        })
    assert result.status_code == 422
    assert result.headers["x-jev-config-id"] == effective_id != cfg.config_id
    assert "too many upstream calls" in result.text.lower()


@pytest.mark.asyncio
async def test_extended_and_compatible_share_request_capacity(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return token_completion()

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(base_settings(server=Server(max_concurrent_requests=1)), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://testserver") as local:
        first = asyncio.create_task(local.post("/v1/systemone", json=noul_request()))
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            blocked = await asyncio.wait_for(local.post("/v1/evaluate", json={
                "request": noul_request(), "execution": {},
            }), timeout=2)
            assert blocked.status_code == 529
            assert blocked.headers["x-jev-config-id"] != app.state.settings.config_id
        finally:
            release.set()
        assert (await asyncio.wait_for(first, timeout=2)).status_code == 200
    await upstream.aclose()
