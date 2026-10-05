"""Command line: `kvserve serve [--engine-flags]`."""

from __future__ import annotations

import argparse
import dataclasses
import logging

from kvserve.config import EngineConfig


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(EngineConfig):
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            kind = {"num_kv_blocks": int}.get(f.name, type(default))
            parser.add_argument(flag, type=kind, default=default)


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvserve")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    _add_engine_args(serve)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    names = {f.name for f in dataclasses.fields(EngineConfig)}
    config = EngineConfig(**{k: v for k, v in vars(args).items() if k in names})

    import uvicorn

    from kvserve.server import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
