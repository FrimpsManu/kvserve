"""Command line: `kvserve serve [--engine-flags]`."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import socket
import sys

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


def _port_taken(port: int) -> bool:
    """True if something already accepts connections on localhost:port.

    Binding 0.0.0.0 can succeed even when another process holds 127.0.0.1 on the same
    port (macOS allows it), and then http://localhost:port silently reaches that other
    process. Check by connecting rather than binding.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvserve")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--engine-mode",
        choices=["process", "thread"],
        default="process",
        help="run the engine in its own process (default) or on a thread of the server process",
    )
    _add_engine_args(serve)
    args = parser.parse_args()

    if _port_taken(args.port):
        sys.exit(f"kvserve: port {args.port} is already in use on localhost; pick another with --port")

    logging.basicConfig(level=logging.WARNING)
    names = {f.name for f in dataclasses.fields(EngineConfig)}
    config = EngineConfig(**{k: v for k, v in vars(args).items() if k in names})

    import uvicorn

    from kvserve.server import create_app

    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    app = create_app(config, url=f"http://{display_host}:{args.port}", engine_mode=args.engine_mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
