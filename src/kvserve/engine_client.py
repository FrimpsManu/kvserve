"""API-server side of the two-process architecture (see engine_core.py).

The server process keeps HTTP, tokenization, detokenization, SSE and telemetry; the
engine process only schedules and runs the model. They no longer share a GIL, so
per-token streaming work for hundreds of connections cannot stall the step loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import multiprocessing
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio
from transformers import AutoTokenizer

from kvserve.config import EngineConfig, resolve_model_path
from kvserve.engine import RequestOutput, StepStats
from kvserve.engine_core import run_engine_core
from kvserve.sequence import FinishReason, SamplingParams
from kvserve.telemetry import RequestTiming, Telemetry


class EngineDeadError(RuntimeError):
    pass


@dataclass
class _Stream:
    queue: asyncio.Queue
    timing: RequestTiming = field(default_factory=lambda: RequestTiming(time.perf_counter()))
    tokens: list[int] = field(default_factory=list)


class ProcessEngineClient:
    mode = "process"

    def __init__(self, config: EngineConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(config.model))
        self.telemetry = Telemetry()
        self.info: dict[str, Any] = {}
        self._streams: dict[str, _Stream] = {}
        self._ids = itertools.count()
        self._error: str | None = None
        self._tasks: list[asyncio.Task] = []

        self._ipc_dir = Path(tempfile.mkdtemp(prefix="kvserve-"))
        input_addr = f"ipc://{self._ipc_dir}/input"
        output_addr = f"ipc://{self._ipc_dir}/output"
        self._ctx = zmq.asyncio.Context()
        self._to_engine = self._ctx.socket(zmq.PUSH)
        self._to_engine.bind(input_addr)
        self._from_engine = self._ctx.socket(zmq.PULL)
        self._from_engine.bind(output_addr)

        # 'spawn': CUDA cannot be re-initialised in a forked child.
        mp = multiprocessing.get_context("spawn")
        self._proc = mp.Process(
            target=run_engine_core, args=(config, input_addr, output_addr), name="kvserve-engine-core", daemon=True
        )
        self._proc.start()

    @classmethod
    async def create(cls, config: EngineConfig, startup_timeout_s: float = 900.0) -> ProcessEngineClient:
        client = cls(config)
        await client._wait_ready(startup_timeout_s)
        client._tasks = [asyncio.create_task(client._pump()), asyncio.create_task(client._watchdog())]
        return client

    async def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            if await self._from_engine.poll(timeout=500):
                msg = await self._from_engine.recv_pyobj()
                if msg[0] == "ready":
                    self.info = msg[1]
                    return
                if msg[0] == "error":
                    raise EngineDeadError(f"engine process failed to start:\n{msg[1]}")
            if not self._proc.is_alive():
                raise EngineDeadError(f"engine process exited during startup (code {self._proc.exitcode})")
            if time.monotonic() > deadline:
                raise EngineDeadError(f"engine process not ready after {timeout_s:.0f}s")

    @property
    def healthy(self) -> bool:
        return self._proc.is_alive() and self._error is None

    async def generate(self, prompt: str | list[int], params: SamplingParams) -> AsyncIterator[RequestOutput]:
        if not self.healthy:
            raise EngineDeadError(self._error or "engine process is not running")
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens; max_model_len is {self.config.max_model_len}")

        request_id = f"cmpl-{next(self._ids)}"
        stream = _Stream(asyncio.Queue())
        self._streams[request_id] = stream
        await self._to_engine.send_pyobj(("add", request_id, token_ids, params))
        finished = False
        try:
            while True:
                item = await stream.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished and self._proc.is_alive():  # client went away: free its KV blocks
                await self._to_engine.send_pyobj(("abort", request_id))

    async def _pump(self) -> None:
        """Route per-step messages from the engine to request streams and telemetry."""
        while True:
            msg = await self._from_engine.recv_pyobj()
            kind = msg[0]
            now = time.perf_counter()
            if kind == "step":
                _, packed, report = msg
                for request_id, new_ids, finished, reason, num_prompt, num_cached in packed:
                    stream = self._streams.get(request_id)
                    tokens = stream.tokens if stream else list(new_ids)
                    if stream:
                        tokens.extend(new_ids)
                    out = RequestOutput(
                        request_id, new_ids, list(tokens), finished,
                        FinishReason(reason) if reason else None, num_prompt, num_cached,
                    )  # fmt: skip
                    self.telemetry.on_output(out, stream.timing if stream else None, now)
                    if stream:
                        stream.queue.put_nowait(out)
                self._apply_report(report)
            elif kind == "gauges":
                self._apply_report(msg[1])
            elif kind == "reject":
                stream = self._streams.get(msg[1])
                if stream:
                    stream.queue.put_nowait(ValueError(msg[2]))
            elif kind == "error":
                self._fail(f"engine process crashed:\n{msg[1]}")
                return

    def _apply_report(self, report: dict[str, Any]) -> None:
        if "step" in report:
            self.telemetry.on_step(StepStats(*report["step"]))
        self.telemetry.on_step_paths(report["step_paths"])
        self.telemetry.on_gauges(report["running"], report["waiting"], report["kv_usage"])

    async def _watchdog(self) -> None:
        """Fail in-flight requests if the engine process dies without reporting why."""
        while self._proc.is_alive():
            await asyncio.sleep(1.0)
        if self._error is None:
            self._fail(f"engine process exited unexpectedly (code {self._proc.exitcode})")

    def _fail(self, reason: str) -> None:
        self._error = reason
        for stream in list(self._streams.values()):
            stream.queue.put_nowait(EngineDeadError(reason))

    def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._proc.is_alive():
            with contextlib.suppress(zmq.ZMQError):
                self._to_engine.send_pyobj(("shutdown",), flags=zmq.NOBLOCK)
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
        self._to_engine.close(linger=0)
        self._from_engine.close(linger=0)
        self._ctx.term()
        shutil.rmtree(self._ipc_dir, ignore_errors=True)
