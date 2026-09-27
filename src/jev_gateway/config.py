"""One validated TOML document, no executable configuration or hidden overlays."""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Upstream(Strict):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: PositiveFloat = 60.0
    top_logprobs: int = Field(default=20, ge=1, le=20)
    supports_images: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        return value.rstrip("/")

    @field_validator("model", "api_key_env")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("extra_body")
    @classmethod
    def extension_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {"model", "messages", "stream", "n", "max_tokens", "max_completion_tokens", "temperature", "logprobs", "top_logprobs", "tools", "tool_choice", "response_format", "logit_bias", "top_p", "stop", "frequency_penalty", "presence_penalty"}
        if reserved & value.keys():
            raise ValueError(f"extra_body cannot override protocol fields: {sorted(reserved & value.keys())}")
        json.dumps(value, allow_nan=False)
        return value


class Adapter(Strict):
    mode: Literal["token_logprobs", "reported_probability"] = "token_logprobs"
    temperature: PositiveFloat = 1.0
    double_round_robin: bool = False
    callsigns: list[str] = Field(default_factory=list, max_length=255)

    @field_validator("callsigns")
    @classmethod
    def labels(cls, value: list[str]) -> list[str]:
        if value and len(value) < 2:
            raise ValueError("callsigns must be empty or have at least two tokens")
        if len(set(value)) != len(value):
            raise ValueError("callsigns must be unique")
        if any(not v or v.strip() != v or any(c.isspace() for c in v) or any(ord(c) < 32 for c in v) for v in value):
            raise ValueError("callsigns must be nonempty tokens without whitespace or control characters")
        return value


class Prompt(Strict):
    system: str = (
        "Answer the question using the evidence. "
        "Reply with one label, without whitespace."
    )
    user: str = "Evidence:\n{{state}}\n\nQuestion:\n{{instructions}}\n\nOptions:\n{{options}}\n\n{{output}}"

    @model_validator(mode="after")
    def placeholders(self):
        allowed = {"state", "instructions", "options", "output"}
        text = self.system + "\n" + self.user
        matches = list(re.finditer(r"\{\{\s*([a-z_]+)\s*\}\}", text))
        slots = {m[1] for m in matches}
        residue = re.sub(r"\{\{\s*([a-z_]+)\s*\}\}", "", text)
        if slots != allowed or "{{" in residue or "}}" in residue:
            raise ValueError("templates must use state, instructions, options, output; unknown or malformed placeholders are not allowed")
        return self


class Generation(Strict):
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, gt=0)


class Server(Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key_env: str | None = None
    max_questions: int = Field(default=64, ge=1, le=1024)
    max_body_bytes: int = Field(default=2_097_152, ge=1)
    max_calls_per_request: int = Field(default=256, ge=1)
    max_concurrent_requests: int = Field(default=16, ge=1)
    max_concurrent_calls: int = Field(default=8, ge=1)
    request_timeout: PositiveFloat = 120.0

    @field_validator("api_key_env")
    @classmethod
    def valid_key_env(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("omit api_key_env to disable local authentication")
        return value


class Diagnostics(Strict):
    low_mass_threshold: float = Field(default=0.99, ge=0, le=1, allow_inf_nan=False)


def _compatible_readout(adapter: Adapter, prompt: Prompt, generation: Generation | None) -> Generation | None:
    if adapter.mode == "token_logprobs":
        if generation is not None:
            raise ValueError("generation settings require adapter.mode = reported_probability")
        return None
    if (
        "system" not in prompt.model_fields_set
        or prompt.system == Prompt.model_fields["system"].default
    ):
        raise ValueError("reported_probability requires an explicit, customized prompt.system")
    return generation or Generation()


class InferenceConfig(Strict):
    """A complete request-level inference configuration with code defaults."""

    adapter: Adapter = Field(default_factory=Adapter)
    prompt: Prompt = Field(default_factory=Prompt)
    generation: Generation | None = None
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @model_validator(mode="after")
    def compatible_readout(self):
        self.generation = _compatible_readout(self.adapter, self.prompt, self.generation)
        return self


class Settings(Strict):
    upstream: Upstream = Field(default_factory=Upstream)
    adapter: Adapter = Field(default_factory=Adapter)
    prompt: Prompt = Field(default_factory=Prompt)
    generation: Generation | None = None
    server: Server = Field(default_factory=Server)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @model_validator(mode="after")
    def compatible_readout(self):
        self.generation = _compatible_readout(self.adapter, self.prompt, self.generation)
        return self

    @property
    def config_id(self) -> str:
        encoded = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def max_answers(self) -> int:
        return len(self.adapter.callsigns) if self.adapter.callsigns else 26


def load_settings(path: str | Path) -> Settings:
    with Path(path).open("rb") as source:
        return Settings.model_validate(tomllib.load(source))
