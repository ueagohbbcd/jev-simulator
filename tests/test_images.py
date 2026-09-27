"""Inline image validation and multimodal Chat Completions forwarding."""
from __future__ import annotations

import base64
import io
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
from pydantic import ValidationError

import jev_gateway.images as image_module
import jev_gateway.service as service_module
from jev_gateway.api import create_app
from jev_gateway.config import Server, Settings, Upstream
from jev_gateway.images import ImageValidationError, normalize_images
from jev_gateway.schema import ImageInput, SystemOneRequest


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def picture(fmt: str, *, size=(2, 3), color="red") -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


def configured(*, supports_images=True, server: Server | None = None) -> Settings:
    return Settings(
        upstream=Upstream(base_url="https://upstream.invalid", model="mock-model",
                          api_key_env="IMAGE_TEST_KEY", supports_images=supports_images),
        server=server or Server(),
    )


def payload(images: list[dict] | None = None) -> dict:
    return {"model": "jev-latest", "state": "PRIVATE-STATE",
            "questions": {"n": {"type": "noul", "instructions": "Is it red?"}},
            "images": images or []}


def token_completion() -> httpx.Response:
    entries = [{"token": label, "logprob": math.log(probability)}
               for label, probability in [("Yes", .8), ("No", .2)]]
    return httpx.Response(200, json={"choices": [{"finish_reason": "length",
        "logprobs": {"content": [{**entries[0], "top_logprobs": entries}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}})


def reported_completion() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
        "message": {"content": '{"Yes":0.8,"No":0.2}'}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 9}})


def test_both_http_routes_forward_ordered_normalized_images_in_both_modes(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("IMAGE_TEST_KEY", "PRIVATE-KEY")
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("secret", "META-SHOULD-DISAPPEAR")
    png_buffer = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(png_buffer, format="PNG", pnginfo=png_info)
    jpeg_buffer = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6  # Rotate a 2x3 JPEG to 3x2 during normalization.
    Image.new("RGB", (2, 3), "blue").save(jpeg_buffer, format="JPEG", exif=exif)
    inputs = [
        {"data": "data:image/png;base64," + encoded(png_buffer.getvalue())},
        {"type": "image/jpeg", "data": encoded(jpeg_buffer.getvalue())},
    ]
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer PRIVATE-KEY"
        return reported_completion() if "response_format" in body else token_completion()

    app = create_app(configured(), client=httpx.AsyncClient(
        transport=httpx.MockTransport(handler)))
    with TestClient(app) as local:
        legacy = local.post("/v1/systemone", json=payload(inputs))
        extended = local.post("/v1/evaluate", json={"request": payload(inputs),
            "execution": {"adapter": {"mode": "reported_probability"},
                          "prompt": {"system": "Return JSON probabilities."}}})
    assert legacy.status_code == extended.status_code == 200
    assert legacy.json()["answers"]["n"]["noul"] == pytest.approx(.8)
    assert extended.json()["result"]["answers"]["n"]["noul"] == pytest.approx(.8)
    assert len(seen) == 2
    for body in seen:
        messages = body["messages"]
        assert isinstance(messages[0]["content"], str)
        blocks = messages[1]["content"]
        assert [block["type"] for block in blocks] == ["image_url", "image_url", "text"]
        assert "PRIVATE-STATE" in blocks[2]["text"]
        for index, size in enumerate([(2, 3), (3, 2)]):
            url = blocks[index]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as normalized:
                normalized.load()
                assert normalized.format == "PNG"
                assert normalized.size == size
                assert normalized.info == {}
    assert "META-SHOULD-DISAPPEAR" not in json.dumps(seen)
    assert encoded(png_buffer.getvalue()) not in capsys.readouterr().err


def test_round_robin_dry_run_reuses_images_once_and_preserves_order(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_TEST_KEY", "key")
    image_data = {"data": "data:image/png;base64," + encoded(picture("PNG"))}
    calls = 0
    original = normalize_images

    def counted(items):
        nonlocal calls
        calls += 1
        return original(items)

    monkeypatch.setattr(service_module, "normalize_images", counted)
    app = create_app(configured(), client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: pytest.fail("dry run called upstream"))))
    request = {"model": "jev-latest", "state": "state", "images": [image_data, image_data],
               "questions": {"c": {"type": "choice", "instructions": "choose",
                   "criteria": {"a": "A", "b": "B", "c": "C"}}}}
    with TestClient(app) as local:
        result = local.post("/v1/evaluate", json={"request": request,
            "execution": {"adapter": {"double_round_robin": True}}, "dry_run": True})
    assert result.status_code == 200
    assert calls == 1
    plan = result.json()["plan"]
    assert plan["request_count"] == 6
    image_blocks = [branch["messages"][1]["content"][:2] for branch in plan["requests"]]
    assert all(blocks == image_blocks[0] for blocks in image_blocks)
    assert [block["type"] for block in image_blocks[0]] == ["image_url", "image_url"]
    assert plan["requests"][0]["messages"][1]["content"][2]["type"] == "text"


@pytest.mark.parametrize("fmt,mime", [
    ("JPEG", "image/jpeg"), ("PNG", "image/png"),
    ("WEBP", "image/webp"), ("GIF", "image/gif"),
])
def test_supported_static_formats_normalize_to_png(fmt, mime) -> None:
    url = normalize_images([ImageInput(type=mime, data=encoded(picture(fmt)))])[0]
    assert url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        image.load()
        assert image.format == "PNG" and image.size == (2, 3)


@pytest.mark.parametrize("item", [
    {"data": "https://example.invalid/image.png"},
    {"type": "image/png", "data": "C:\\private\\image.png"},
    {"type": "image/png", "data": "abc!"},
    {"type": "image/png", "data": "YW Jj"},
    {"type": "image/jpeg", "data": "data:image/png;base64," + encoded(picture("PNG"))},
    {"type": "image/jpeg", "data": encoded(picture("PNG"))},
    {"type": "image/png", "data": encoded(picture("PNG")[:-12])},
])
def test_invalid_or_mismatched_images_fail_before_upstream(monkeypatch, item) -> None:
    monkeypatch.setenv("IMAGE_TEST_KEY", "key")
    app = create_app(configured(), client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: pytest.fail("invalid image called upstream"))))
    with TestClient(app) as local:
        result = local.post("/v1/systemone", json=payload([item]))
    assert result.status_code == 422
    assert item["data"] not in result.text


def test_animated_image_and_pixel_count_are_rejected() -> None:
    animation = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(animation, format="GIF", save_all=True,
        append_images=[Image.new("RGB", (2, 2), "blue")], duration=100, loop=0)
    with pytest.raises(ImageValidationError, match="Animated"):
        normalize_images([ImageInput(type="image/gif", data=encoded(animation.getvalue()))])

    large = picture("PNG", size=(4001, 4000), color="white")
    with pytest.raises(ImageValidationError, match="pixel"):
        normalize_images([ImageInput(type="image/png", data=encoded(large))])


def test_decoded_and_normalized_size_limits_apply_before_expensive_work(monkeypatch) -> None:
    monkeypatch.setattr(image_module, "MAX_IMAGE_BYTES", 4)
    with pytest.raises(ImageValidationError, match="decoded limit"):
        normalize_images([ImageInput(type="image/png", data=encoded(b"xxxxx"))])

    monkeypatch.setattr(image_module, "MAX_IMAGE_BYTES", 12 * 1024 * 1024)
    monkeypatch.setattr(image_module, "MAX_TOTAL_IMAGE_BYTES", 6)
    monkeypatch.setattr(image_module, "_normalize_png", lambda *_: pytest.fail("normalized before total check"))
    with pytest.raises(ImageValidationError, match="decoded total"):
        normalize_images([ImageInput(type="image/png", data=encoded(b"xxxx")),
                          ImageInput(type="image/png", data=encoded(b"yyyy"))])

    monkeypatch.undo()
    with pytest.raises(ImageValidationError, match="Normalized images exceed"):
        image_module._normalize_png("image/png", picture("PNG"), remaining=1)


def test_capability_count_and_body_limits_reject_without_upstream(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_TEST_KEY", "key")
    bad = {"data": "data:image/png;base64,NOT-BASE64"}
    unavailable = create_app(configured(supports_images=False), client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("unsupported image called upstream"))))
    with TestClient(unavailable) as local:
        no_capability = local.post("/v1/evaluate", json={"request": payload([bad]),
            "execution": {}, "dry_run": True})
    assert no_capability.status_code == 422
    assert "does not support images" in no_capability.text
    assert "NOT-BASE64" not in no_capability.text

    image_data = {"type": "image/png", "data": encoded(picture("PNG"))}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload([image_data] * 9))

    capped = create_app(configured(server=Server(max_body_bytes=100)), client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("oversize body called upstream"))))
    with TestClient(capped) as local:
        too_large = local.post("/v1/systemone", json=payload([image_data]))
    assert too_large.status_code == 413
