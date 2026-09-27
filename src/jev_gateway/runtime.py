"""Live configuration snapshots and the optional stdin control channel."""
from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx
from pydantic import ValidationError

from .config import Settings, load_settings
from .service import Gateway, GatewayCapacity, RequestError


@dataclass(frozen=True)
class RuntimeSnapshot:
    settings: Settings
    gateway: Gateway
    config_path: Path | None


class ReloadError(Exception):
    """A sanitized control-channel error safe to print to stdout."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        fields: list[str] | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fields = fields
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.fields:
            result["fields"] = self.fields
        if self.details:
            result["details"] = self.details
        return result


class GatewayRuntime:
    """Own one client/capacity pool and atomically publish immutable snapshots."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        config_path: str | Path | None = None,
        startup_cwd: str | Path | None = None,
    ) -> None:
        self.startup_cwd = Path(startup_cwd or Path.cwd()).resolve()
        config_path = (
            self._resolve_path(config_path) if config_path is not None else None
        )
        self._capacity = GatewayCapacity(settings)
        self._owner_gateway = Gateway(settings, client, capacity=self._capacity)
        self._snapshot = RuntimeSnapshot(settings, self._owner_gateway, config_path)
        self._app: Any = None

    @property
    def snapshot(self) -> RuntimeSnapshot:
        return self._snapshot

    @property
    def config_path(self) -> Path | None:
        return self._snapshot.config_path

    def bind_app(self, app: Any) -> None:
        self._app = app
        self._sync_app_state()

    async def startup(self) -> None:
        await self._owner_gateway.startup()

    async def aclose(self) -> None:
        await self._owner_gateway.aclose()

    def status(self) -> dict[str, Any]:
        settings = self._snapshot.settings
        return {
            "config_path": (
                str(self._snapshot.config_path)
                if self._snapshot.config_path is not None
                else None
            ),
            "config_id": settings.config_id,
            "mode": "round_robin" if settings.adapter.double_round_robin else "single",
            "adapter_mode": settings.adapter.mode,
            "temperature": settings.adapter.temperature,
        }

    async def reload(self, path: str | Path | None = None) -> RuntimeSnapshot:
        candidate_path = self.config_path if path is None else self._resolve_path(path)
        if candidate_path is None:
            raise ReloadError("no_active_config", "No active configuration path")

        try:
            candidate = load_settings(candidate_path)
        except FileNotFoundError as exc:
            raise ReloadError("config_not_found", "Configuration file was not found") from exc
        except PermissionError as exc:
            raise ReloadError("config_unreadable", "Configuration file is not readable") from exc
        except tomllib.TOMLDecodeError as exc:
            location = {
                name: value
                for name in ("lineno", "colno")
                if isinstance((value := getattr(exc, name, None)), int)
            }
            raise ReloadError(
                "invalid_toml",
                "Configuration is not valid TOML",
                details=[location] if location else None,
            ) from exc
        except ValidationError as exc:
            details = [
                {
                    "field": ".".join(str(part) for part in error["loc"]),
                    "message": str(error["msg"]),
                }
                for error in exc.errors(include_input=False, include_context=False)
            ]
            raise ReloadError(
                "validation_failed",
                "Configuration validation failed",
                details=details,
            ) from exc
        except OSError as exc:
            raise ReloadError("config_unreadable", "Configuration file could not be read") from exc

        current = self._snapshot.settings
        changed_server_fields = sorted(
            name
            for name in type(current.server).model_fields
            if getattr(candidate.server, name) != getattr(current.server, name)
        )
        if changed_server_fields:
            raise ReloadError(
                "restart_required",
                "Server configuration changes require a restart",
                fields=changed_server_fields,
            )
        if not os.environ.get(candidate.upstream.api_key_env):
            raise ReloadError(
                "missing_upstream_api_key",
                "Configured upstream API key environment variable is missing",
            )

        gateway = Gateway(
            candidate,
            self._owner_gateway._client,
            capacity=self._capacity,
        )
        snapshot = RuntimeSnapshot(candidate, gateway, candidate_path)
        self._snapshot = snapshot
        self._sync_app_state()
        return snapshot

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.startup_cwd / candidate
        return candidate.resolve()

    def _sync_app_state(self) -> None:
        if self._app is not None:
            self._app.state.settings = self._snapshot.settings
            self._app.state.gateway = self._snapshot.gateway
            self._app.state.runtime = self


_RELOAD = re.compile(r"^reload(?:\s+(.*))?$")


def _reload_argument(line: str) -> tuple[bool, str | None]:
    match = _RELOAD.fullmatch(line)
    if match is None:
        return False, None
    value = (match.group(1) or "").strip()
    if not value:
        return True, None
    if value[0] in {'"', "'"}:
        if len(value) < 2 or value[-1] != value[0]:
            raise ReloadError("invalid_command", "Reload path has an unmatched quote")
        value = value[1:-1]
        if not value:
            raise ReloadError("invalid_command", "Reload path is empty")
    return True, value


class StdinControl:
    """Read UTF-8 command lines with a daemon raw-fd reader."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        input_stream: IO[str],
        output_stream: IO[str],
    ) -> None:
        self.runtime = runtime
        self.input_stream = input_stream
        self.output_stream = output_stream
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = threading.Event()

    def start(self) -> bool:
        try:
            fd = self.input_stream.fileno()
        except (AttributeError, OSError, ValueError):
            return False
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run_commands())
        target = self._read_console if os.name == "nt" and os.isatty(fd) else self._read_fd
        threading.Thread(
            target=target,
            args=(fd,),
            name="jev-stdin-control",
            daemon=True,
        ).start()
        return True

    async def stop(self) -> None:
        self._stopped.set()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    def _read_fd(self, fd: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        try:
            while not self._stopped.is_set():
                chunk = os.read(fd, 4096)
                if not chunk:
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        self._submit(pending.rstrip("\r"))
                    return
                pending += decoder.decode(chunk)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    self._submit(line.rstrip("\r"))
        except OSError:
            return

    def _read_console(self, fd: int) -> None:
        """Use the Windows wide console API so non-ASCII paths survive code pages."""
        import ctypes
        import msvcrt
        from ctypes import wintypes

        read_console = ctypes.windll.kernel32.ReadConsoleW
        read_console.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        read_console.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        pending = ""
        while not self._stopped.is_set():
            buffer = ctypes.create_unicode_buffer(2048)
            count = wintypes.DWORD()
            ok = read_console(
                handle, buffer, len(buffer) - 1, ctypes.byref(count), None
            )
            if not ok or count.value == 0:
                if pending:
                    self._submit(pending.rstrip("\r"))
                return
            text = buffer[: count.value]
            eof = "\x1a" in text
            pending += text.split("\x1a", 1)[0]
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self._submit(line.rstrip("\r"))
            if eof:
                if pending:
                    self._submit(pending.rstrip("\r"))
                return

    def _submit(self, line: str) -> None:
        if self._stopped.is_set() or self._loop is None or self._queue is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, line)
        except RuntimeError:
            return

    async def _run_commands(self) -> None:
        assert self._queue is not None
        while True:
            line = (await self._queue.get()).strip()
            if not line:
                continue
            await self._dispatch(line)

    async def _dispatch(self, line: str) -> None:
        command = "unknown"
        error: ReloadError | None = None
        try:
            if line == "status":
                command = "status"
            else:
                if line == "reload" or re.match(r"^reload\s", line):
                    command = "reload"
                is_reload, path = _reload_argument(line)
                if not is_reload:
                    raise ReloadError("unknown_command", "Unknown control command")
                await self.runtime.reload(path)
        except ReloadError as exc:
            error = exc
        except RequestError:
            error = ReloadError("reload_failed", "Configuration reload failed")
        except Exception:
            error = ReloadError("reload_failed", "Configuration reload failed")

        result = {"command": command, "ok": error is None, **self.runtime.status()}
        if error is not None:
            result["error"] = error.as_dict()
        self.output_stream.write(
            json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        self.output_stream.flush()
