"""Network orchestration for the Jev-compatible decision adapter."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx

from .config import InferenceConfig, Settings
from .core import CoreError, aggregate, build_plan, parse_reported_response, parse_response
from .images import ImageValidationError, normalize_images
from .schema import SystemOneRequest


class RequestError(Exception):
    """A safe, public-facing service error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GatewayCapacity:
    """Process-wide admission and call limits shared by reloaded gateways."""

    def __init__(self, settings: Settings) -> None:
        self.call_slots = asyncio.Semaphore(settings.server.max_concurrent_calls)
        self.admission_lock = threading.Lock()
        self.active_requests = 0


def _sum_usage(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Recursively sum numeric usage counters while preserving provider detail."""
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            target[key] = target.get(key, 0) + value
        elif isinstance(value, Mapping):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                _sum_usage(child, value)


class Gateway:
    """Validate, plan, execute and aggregate one SystemOne request."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        *,
        capacity: GatewayCapacity | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._capacity = capacity or GatewayCapacity(settings)

    async def startup(self) -> None:
        """Validate runtime secrets and create the owned HTTP client."""
        self._upstream_key()
        if self._client is None:
            self._client = httpx.AsyncClient(trust_env=False)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _upstream_key(self) -> str:
        key = os.environ.get(self.settings.upstream.api_key_env)
        if not key:
            raise RequestError(503, "Upstream API key is not configured")
        return key

    def with_execution(self, execution: InferenceConfig) -> Gateway:
        """Use request inference settings with this snapshot's deployment resources."""
        effective = Settings(
            upstream=self.settings.upstream,
            adapter=execution.adapter,
            prompt=execution.prompt,
            generation=execution.generation,
            server=self.settings.server,
            diagnostics=execution.diagnostics,
        )
        return Gateway(effective, self._client, capacity=self._capacity)

    def plan(self, payload: SystemOneRequest) -> list[dict[str, Any]]:
        """Build every branch after checking all deployment limits."""
        allowed_models = {"jev-latest", self.settings.upstream.model}
        if payload.model not in allowed_models:
            raise RequestError(422, "Unsupported model")
        if len(payload.questions) > self.settings.server.max_questions:
            raise RequestError(422, "Too many questions")

        capacity = self.settings.max_answers
        planned_calls = 0
        for question in payload.questions.values():
            if question.type == "noul":
                answers = 2
            elif question.type == "choice":
                answers = len(question.criteria)
            else:
                answers = len(question.criteria)
            if answers > capacity:
                raise RequestError(422, "A question has too many answers")
            if self.settings.adapter.double_round_robin and question.type in {"choice", "score"}:
                planned_calls += answers * (answers - 1)
            else:
                planned_calls += 1
        if planned_calls > self.settings.server.max_calls_per_request:
            raise RequestError(422, "Request requires too many upstream calls")

        if payload.images and not self.settings.upstream.supports_images:
            raise RequestError(422, "Configured upstream does not support images")
        try:
            image_urls = normalize_images(payload.images) if payload.images else None
        except ImageValidationError as exc:
            raise RequestError(422, str(exc)) from exc

        branches = build_plan(
            payload.model_dump(mode="json", exclude={"images"}),
            self.settings.model_dump(mode="json"),
            image_urls=image_urls,
        )
        if len(branches) != planned_calls:
            raise CoreError("planner returned an unexpected number of branches")
        return branches

    async def evaluate(
        self, payload: SystemOneRequest, request_id: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute a fully validated plan with bounded, fail-fast concurrency."""
        request_id = request_id or uuid.uuid4().hex
        with self._capacity.admission_lock:
            if self._capacity.active_requests >= self.settings.server.max_concurrent_requests:
                raise RequestError(529, "Too many concurrent requests")
            self._capacity.active_requests += 1

        started = time.perf_counter()
        try:
            try:
                branches = self.plan(payload)
            except CoreError as exc:
                raise RequestError(502, "Failed to build evaluation plan") from exc
            key = self._upstream_key()
            if self._client is None:
                self._client = httpx.AsyncClient(trust_env=False)

            try:
                async with asyncio.timeout(self.settings.server.request_timeout):
                    runs = await self._run_all(branches, key)
            except TimeoutError as exc:
                raise RequestError(504, "Evaluation timed out") from exc

            answers: dict[str, Any] = {}
            try:
                for question_id, question in payload.questions.items():
                    question_runs = [run for run in runs if run["question_id"] == question_id]
                    answers[question_id] = aggregate(
                        question.model_dump(mode="json"),
                        question_runs,
                        self.settings.adapter.double_round_robin,
                    )
            except CoreError as exc:
                raise RequestError(502, "Failed to aggregate upstream response") from exc

            usage: dict[str, Any] = {}
            for run in runs:
                _sum_usage(usage, run["usage"])

            response = {
                "model": payload.model,
                "answers": answers,
                "usage": {
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                },
            }
            diagnostics = {
                "schema_version": 1,
                "request_id": request_id,
                "config_id": self.settings.config_id,
                "adapter_mode": self.settings.adapter.mode,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "temperature": self.settings.adapter.temperature,
                "aggregation_method": (
                    "double_round_robin"
                    if self.settings.adapter.double_round_robin
                    else "single"
                ),
                "config": self.settings.model_dump(mode="json"),
                "usage": usage,
                "branches": [self._branch_diagnostics(run) for run in runs],
            }
            return response, diagnostics
        finally:
            with self._capacity.admission_lock:
                self._capacity.active_requests -= 1

    async def _run_all(
        self, branches: list[dict[str, Any]], key: str
    ) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self._run_branch(branch, key)) for branch in branches]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _run_branch(self, branch: dict[str, Any], key: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.upstream.model,
            "messages": branch["messages"],
            "stream": False,
        }
        if self.settings.adapter.mode == "reported_probability":
            assert self.settings.generation is not None
            body.update({
                "max_tokens": self.settings.generation.max_tokens,
                "temperature": self.settings.generation.temperature,
                "response_format": {"type": "json_object"},
            })
        else:
            body.update({
                "max_tokens": 1,
                "temperature": 1,
                "logprobs": True,
                "top_logprobs": self.settings.upstream.top_logprobs,
            })
        body.update(self.settings.upstream.extra_body)
        assert self._client is not None
        try:
            async with self._capacity.call_slots:
                response = await self._client.post(
                    self.settings.upstream.base_url + "/chat/completions",
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.settings.upstream.timeout,
                )
        except httpx.TimeoutException as exc:
            raise RequestError(504, "Upstream request timed out") from exc
        except httpx.RequestError as exc:
            raise RequestError(503, "Upstream service is unavailable") from exc

        if response.status_code == 429:
            raise RequestError(429, "Upstream rate limit exceeded")
        if response.status_code in {401, 403}:
            raise RequestError(502, "Upstream authentication failed")
        if response.status_code >= 500:
            raise RequestError(503, "Upstream service is unavailable")
        if response.status_code < 200 or response.status_code >= 300:
            raise RequestError(502, "Upstream rejected the request")
        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise RequestError(502, "Upstream returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RequestError(502, "Upstream returned an invalid response")

        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise RequestError(502, "Upstream response omitted usage")
        for field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RequestError(502, "Upstream response has invalid usage")

        try:
            if self.settings.adapter.mode == "reported_probability":
                parsed = parse_reported_response(
                    data, branch["mapping"], self.settings.adapter.temperature
                )
            else:
                parsed = parse_response(
                    data,
                    branch["mapping"],
                    self.settings.adapter.temperature,
                    self.settings.diagnostics.low_mass_threshold,
                )
        except CoreError as exc:
            raise RequestError(502, "Upstream response could not be interpreted") from exc
        return {**branch, **parsed, "adapter_mode": self.settings.adapter.mode, "usage": usage}

    @staticmethod
    def _branch_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
        # The caller may explicitly persist this sensitive trace. The HTTP log
        # selects a small metadata-only subset and never emits these messages.
        return dict(run)
