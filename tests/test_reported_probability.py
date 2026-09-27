"""The numerical readout protocol uses mocked Chat Completions only."""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from jev_gateway.api import create_app
from jev_gateway.config import Adapter, Generation, Prompt, Settings, Upstream
from jev_gateway.core import CoreError, parse_reported_response
from jev_gateway.runtime import GatewayRuntime
from jev_gateway.schema import SystemOneRequest
from jev_gateway.service import Gateway


PROMPT = Prompt(
    system="Return a JSON distribution for the output labels.",
    user="{{state}}\n{{instructions}}\n{{options}}\n{{output}}",
)


def settings(**adapter_overrides) -> Settings:
    return Settings(
        upstream=Upstream(base_url="https://upstream.invalid", model="mock-model",
                          api_key_env="UPSTREAM_TEST_KEY"),
        adapter=Adapter(mode="reported_probability", **adapter_overrides),
        prompt=PROMPT,
    )


def completion(content: str, *, finish: str = "stop", refusal=None,
               usage: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "mock-model",
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "refusal": refusal}}],
        "usage": usage or {"prompt_tokens": 7, "completion_tokens": 5,
                           "total_tokens": 12},
    })


def request_labels(request: httpx.Request) -> tuple[dict, str, list[str]]:
    body = json.loads(request.content)
    user = body["messages"][1]["content"]
    match = re.search(r"keys: (.*?)\. Each value", user)
    assert match is not None
    return body, user, json.loads("[" + match.group(1) + "]")


def test_reported_config_defaults_explicit_prompt_and_generation_rules() -> None:
    default = Settings()
    assert default.adapter.mode == "token_logprobs"
    assert default.generation is None
    assert Settings.model_validate(default.model_dump()).generation is None
    reported = settings()
    assert reported.generation == Generation()
    assert Settings.model_validate(reported.model_dump()).generation == Generation()
    with pytest.raises(ValidationError, match="explicit, customized"):
        Settings(adapter={"mode": "reported_probability"})
    with pytest.raises(ValidationError, match="explicit, customized"):
        Settings(adapter={"mode": "reported_probability"},
                 prompt={"system": Prompt().system})
    with pytest.raises(ValidationError, match="generation settings require"):
        Settings(generation={})
    for generation in ({"temperature": -0.1}, {"temperature": 2.1},
                       {"temperature": float("nan")}, {"max_tokens": 0},
                       {"max_tokens": True}):
        with pytest.raises(ValidationError):
            Settings(adapter={"mode": "reported_probability"},
                     prompt=PROMPT, generation=generation)


def test_reported_http_noul_choice_score_and_private_diagnostics(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "private-key")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body, user, labels = request_labels(request)
        seen.append(body)
        assert labels == (["Yes", "No"] if "N-question" in user else ["alpha", "fox"])
        if "N-question" in user:
            values = [.8, .2]
        elif "C-question" in user:
            values = [.9, .1]
        else:
            values = [.25, .75]
        return completion(json.dumps(dict(zip(labels, values, strict=True))))

    cfg = settings(temperature=2, callsigns=["alpha", "fox"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(cfg, client=client)
    with TestClient(app) as local:
        result = local.post("/v1/systemone", json={
            "model": "jev-latest", "state": "PRIVATE-STATE",
            "questions": {
                "n": {"type": "noul", "instructions": "N-question"},
                "c": {"type": "choice", "instructions": "C-question",
                      "criteria": {"x": "X", "y": "Y"}},
                "s": {"type": "score", "instructions": "S-question",
                      "criteria": ["low", "high"]},
            },
        })
    assert result.status_code == 200
    data = result.json()
    assert data["answers"]["n"]["noul"] == pytest.approx(2 / 3)
    assert data["answers"]["c"]["probabilities"] == pytest.approx({"x": .75, "y": .25})
    assert data["answers"]["s"]["score"] == pytest.approx(3**.5 / (1 + 3**.5))
    assert data["usage"] == {"input_tokens": 21, "output_tokens": 15}
    assert len(seen) == 3
    for body in seen:
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] == 1024 and body["temperature"] == 0.0
        assert "logprobs" not in body and "top_logprobs" not in body
    event = json.loads(capsys.readouterr().err.strip())
    assert event["adapter_mode"] == "reported_probability"
    assert "observed_label_mass" not in event["branches"][0]
    assert "PRIVATE-STATE" not in json.dumps(event)


@pytest.mark.asyncio
async def test_reported_round_robin_maps_each_direction_and_keeps_zero(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body, _, labels = request_labels(request)
        seen.append(body)
        assert labels == ["left", "right"]
        return completion('{"left":0.8,"right":0.2}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = settings(double_round_robin=True, callsigns=["left", "right"])
    result, diagnostics = await Gateway(cfg, client).evaluate(SystemOneRequest.model_validate({
        "model": "jev-latest", "state": "s", "questions": {
            "c": {"type": "choice", "instructions": "pick",
                  "criteria": {"first": "F", "second": "S"}},
        },
    }))
    assert len(seen) == 2
    assert result["answers"]["c"]["probabilities"] == pytest.approx(
        {"first": .5, "second": .5})
    forward, reverse = diagnostics["branches"]
    assert forward["mapping"] == {"left": "first", "right": "second"}
    assert reverse["mapping"] == {"left": "second", "right": "first"}
    assert forward["raw_probabilities"] == {"first": .8, "second": .2}
    assert reverse["raw_probabilities"] == {"second": .8, "first": .2}
    assert forward["reported_label_probabilities"] == {"left": .8, "right": .2}
    assert forward["raw_output"] == '{"left":0.8,"right":0.2}'
    assert "observed_label_mass" not in forward
    assert result["usage"] == {"input_tokens": 14, "output_tokens": 10}
    zero = parse_reported_response(completion('{"left":0,"right":1}').json(),
                                   {"left": "first", "right": "second"}, 2)
    assert zero["probabilities"] == {"first": 0.0, "second": 1.0}
    await client.aclose()


@pytest.mark.parametrize("content,finish,refusal", [
    ('{"Yes":0.5,"Yes":0.5,"No":0}', "stop", None),
    ('{"Yes":true,"No":0}', "stop", None),
    ('{"Yes":NaN,"No":0}', "stop", None),
    ('{"Yes":1e309,"No":0}', "stop", None),
    ('{"Yes":' + '9' * 400 + ',"No":0}', "stop", None),
    ('{"Yes":-0.1,"No":1.1}', "stop", None),
    ('{"Yes":0.8,"No":0.1}', "stop", None),
    ('{"Yes":1,"No":0,"extra":0}', "stop", None),
    ('{"Yes":1}', "stop", None),
    ('[0.8,0.2]', "stop", None),
    ('{"Yes":0.8,"No":', "stop", None),
    ('{"Yes":0.8,"No":0.2}', "length", None),
    ('{"Yes":0.8,"No":0.2}', "stop", "refused"),
])
def test_invalid_reported_completions_fail_closed(content, finish, refusal) -> None:
    with pytest.raises(CoreError):
        parse_reported_response(completion(content, finish=finish, refusal=refusal).json(),
                                {"Yes": "true", "No": "false"}, 1)


def test_tiny_sum_drift_is_normalized_but_larger_drift_is_rejected() -> None:
    parsed = parse_reported_response(
        completion('{"Yes":0.5,"No":0.5000005}').json(),
        {"Yes": "true", "No": "false"}, 1)
    assert parsed["reported_label_probabilities"]["No"] == .5000005
    assert sum(parsed["raw_probabilities"].values()) == pytest.approx(1)
    with pytest.raises(CoreError, match="sum to one"):
        parse_reported_response(completion('{"Yes":0.5,"No":0.500002}').json(),
                                {"Yes": "true", "No": "false"}, 1)


def test_http_malformed_report_is_safe_502_and_does_not_retry(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return completion('PRIVATE-BAD-OUTPUT')

    app = create_app(settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as local:
        result = local.post("/v1/systemone", json={"model": "jev-latest",
            "state": "PRIVATE-STATE", "questions": {
                "q": {"type": "noul", "instructions": "ask"}}})
    assert result.status_code == 502 and len(seen) == 1
    assert "PRIVATE" not in result.text + capsys.readouterr().err


@pytest.mark.asyncio
async def test_reload_can_switch_readout_modes_without_changing_answer_shape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "key")
    token_path = tmp_path / "token.toml"
    reported_path = tmp_path / "reported.toml"
    token_path.write_text('[upstream]\nbase_url="https://upstream.invalid"\n'
                          'model="mock-model"\napi_key_env="UPSTREAM_TEST_KEY"\n',
                          encoding="utf-8")
    reported_path.write_text(token_path.read_text(encoding="utf-8") +
        '[adapter]\nmode="reported_probability"\n'
        '[prompt]\nsystem="Return JSON probabilities."\n', encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("response_format"):
            return completion('{"Yes":0.8,"No":0.2}')
        assert body["logprobs"] is True
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
            "logprobs": {"content": [{"token": "Yes", "logprob": -0.2231435513,
                "top_logprobs": [{"token": "Yes", "logprob": -0.2231435513},
                                 {"token": "No", "logprob": -1.6094379124}]}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from jev_gateway.config import load_settings
    runtime = GatewayRuntime(load_settings(token_path), client=client, config_path=token_path)
    payload = SystemOneRequest.model_validate({"model": "jev-latest", "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "ask"}}})
    before, _ = await runtime.snapshot.gateway.evaluate(payload)
    assert runtime.status()["adapter_mode"] == "token_logprobs"
    await runtime.reload(reported_path)
    after, _ = await runtime.snapshot.gateway.evaluate(payload)
    assert runtime.status()["adapter_mode"] == "reported_probability"
    assert before["answers"]["q"]["noul"] == pytest.approx(.8)
    assert after["answers"]["q"]["noul"] == pytest.approx(.8)
    await runtime.reload(token_path)
    assert runtime.status()["adapter_mode"] == "token_logprobs"
    await client.aclose()
