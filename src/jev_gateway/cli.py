"""Inspect and run the same gateway used by the HTTP service."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from .config import load_settings
from .schema import SystemOneRequest


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _payload(path: str) -> SystemOneRequest:
    return SystemOneRequest.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))


async def _evaluate(settings, payload, diagnostics_path):
    from .service import Gateway

    async with httpx.AsyncClient(trust_env=False) as client:
        result, diagnostics = await Gateway(settings, client).evaluate(payload)
    if diagnostics_path:
        target = Path(diagnostics_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json(diagnostics) + "\n", encoding="utf-8")
    print(_json(result))


def main(argv=None) -> int:
    # JSON files/pipes stay UTF-8 on Windows regardless of the console code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Jev-compatible decisions over Chat Completions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("check", "Validate configuration without network access"),
        ("preview", "Render every planned request without calling the model"),
        ("evaluate", "Call the upstream and print the compatible response"),
        ("serve", "Start the HTTP gateway"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default="config.toml", help="TOML configuration path")
        if name in {"preview", "evaluate"}:
            command.add_argument("--request", required=True, help="Jev JSON request file")
        if name == "evaluate":
            command.add_argument("--diagnostics", metavar="PATH", help="Opt-in full trace file (may contain sensitive input)")
    args = parser.parse_args(argv)
    startup_cwd = Path.cwd().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = startup_cwd / config_path
    config_path = config_path.resolve()
    try:
        settings = load_settings(config_path)
        if args.command == "check":
            print(_json({"status": "ok", "config_id": settings.config_id,
                         "max_answers_per_question": settings.max_answers,
                         "adapter_mode": settings.adapter.mode,
                         "generation": settings.generation.model_dump() if settings.generation else None,
                         "temperature": settings.adapter.temperature,
                         "double_round_robin": settings.adapter.double_round_robin}))
        elif args.command == "preview":
            from .service import Gateway
            plan = Gateway(settings).plan(_payload(args.request))
            print(_json({"config_id": settings.config_id, "adapter_mode": settings.adapter.mode,
                         "generation": settings.generation.model_dump() if settings.generation else None,
                         "request_count": len(plan), "requests": plan}))
        elif args.command == "evaluate":
            asyncio.run(_evaluate(settings, _payload(args.request), args.diagnostics))
        else:
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(
                settings,
                config_path=config_path,
                startup_cwd=startup_cwd,
                control_stdin=True,
            ), host=settings.server.host, port=settings.server.port,
                        access_log=False, log_level="warning")
        return 0
    except ValidationError as exc:
        # Pydantic's default rendering includes rejected input. Keep input private.
        errors = [{"loc": error["loc"], "message": error["msg"]}
                  for error in exc.errors(include_input=False, include_context=False)]
        print(_json({"error": "validation_failed", "details": errors}), file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(_json({"error": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        from .service import RequestError
        if isinstance(exc, RequestError):
            print(_json({"error": str(exc), "status": exc.status}), file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
