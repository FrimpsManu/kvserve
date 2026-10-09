"""Engine front ends for the async server.

Two implementations share one interface (`EngineClient`):

- `ThreadEngineClient` runs LLMEngine on a dedicated thread in the server process.
  Simple, and GPU work never blocks the event loop, but the HTTP layer and the engine
  share one GIL, so per-token streaming work competes with the step loop.
- `ProcessEngineClient` (engine_client.py) runs the engine in its own process.

Both report the same telemetry and stream `RequestOutput`s to per-request asyncio queues.
"""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from kvserve.config import EngineConfig
from kvserve.engine import LLMEngine, RequestOutput
from kvserve.sequence import SamplingParams
from kvserve.telemetry import RequestTiming, Telemetry


def engine_info(engine: LLMEngine) -> dict[str, Any]:
    """Static facts about a running engine, for the banner and /stats."""
    runner = engine.runner
    return {
        "model": engine.config.model,
        "device": engine.config.device,
        "dtype": str(runner.dtype).removeprefix("torch."),
        "attention": runner.attn_backend.__name__,
        "cuda_graphs": runner.graphs.max_batch if runner.graphs else None,
        "piecewise_graphs": runner.piecewise.max_tokens if runner.piecewise else None,
        "kv_blocks": runner.num_kv_blocks,
        "block_size": engine.config.block_size,
        "kv_capacity_tokens": runner.num_kv_blocks * engine.config.block_size,
        "speculative": _describe_speculation(engine.config),
    }


def _describe_speculation(c: EngineConfig) -> str | None:
    if c.speculative_method == "none":
        return None
    source = f"draft model {c.draft_model}" if c.speculative_method == "draft" else "ngram lookup"
    return f"{source}, {c.num_speculative_tokens} tokens per step"


class EngineClient(Protocol):
    config: EngineConfig
    tokenizer: Any
    telemetry: Telemetry
    info: dict[str, Any]
    mode: str

    @property
    def healthy(self) -> bool: ...

    def generate(self, prompt: str | list[int], params: SamplingParams) -> AsyncIterator[RequestOutput]: ...

    def shutdown(self) -> None: ...


@dataclass
class _Stream:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    timing: RequestTiming = field(default_factory=lambda: RequestTiming(time.perf_counter()))


class ThreadEngineClient:
    """LLMEngine on a dedicated thread of the server process.

    Requests and aborts reach the engine thread through a command queue; tokens flow
    back to each request's asyncio.Queue via `call_soon_threadsafe`.
    """

    mode = "thread"

    def __init__(self, config: EngineConfig):
        self.engine = LLMEngine(config)
        self.tokenizer = self.engine.tokenizer
        self.config = config
        self.info = engine_info(self.engine)
        self.telemetry = Telemetry()
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._streams: dict[str, _Stream] = {}
        self._ids = itertools.count()
        self._stopped = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="kvserve-engine", daemon=True)
        self._thread.start()

    @property
    def healthy(self) -> bool:
        return self._thread.is_alive() and self._error is None

    async def generate(self, prompt: str | list[int], params: SamplingParams) -> AsyncIterator[RequestOutput]:
        if not self.healthy:
            raise RuntimeError("engine is not running") from self._error
        request_id = f"cmpl-{next(self._ids)}"
        stream = _Stream(asyncio.get_running_loop(), asyncio.Queue())
        self._streams[request_id] = stream
        self._commands.put(("add", request_id, prompt, params))
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
            if not finished:  # client went away or errored: free its KV blocks
                self._commands.put(("abort", request_id))

    def shutdown(self) -> None:
        self._stopped.set()
        self._commands.put(("noop",))
        self._thread.join(timeout=10)

    # ---- engine thread -------------------------------------------------------------

    def _run(self) -> None:
        eng = self.engine
        try:
            while not self._stopped.is_set():
                # Block when idle; otherwise drain whatever arrived since the last step.
                if not eng.has_unfinished():
                    self._handle(self._commands.get())
                while not self._commands.empty():
                    self._handle(self._commands.get_nowait())
                self._update_gauges()  # adds/aborts change queue sizes even when idle
                if not eng.has_unfinished():
                    continue
                outputs = eng.step()
                self.telemetry.on_step(eng.last_step)
                self.telemetry.on_step_paths(eng.runner.step_paths)
                self._update_gauges()
                for out in outputs:
                    self._deliver(out)
        except BaseException as e:  # surface engine crashes to every waiting client
            self._error = e
            for stream in list(self._streams.values()):
                stream.loop.call_soon_threadsafe(stream.queue.put_nowait, e)
            raise

    def _handle(self, cmd: tuple) -> None:
        kind = cmd[0]
        if kind == "add":
            _, request_id, prompt, params = cmd
            try:
                self.engine.add_request(prompt, params, request_id)
            except ValueError as e:
                stream = self._streams.get(request_id)
                if stream:
                    stream.loop.call_soon_threadsafe(stream.queue.put_nowait, e)
        elif kind == "abort":
            self.engine.abort(cmd[1])

    def _deliver(self, out: RequestOutput) -> None:
        stream = self._streams.get(out.request_id)
        self.telemetry.on_output(out, stream.timing if stream else None, time.perf_counter())
        if stream is not None:
            # Snapshot the token list: the engine keeps appending to it.
            out.output_token_ids = list(out.output_token_ids)
            stream.loop.call_soon_threadsafe(stream.queue.put_nowait, out)

    def _update_gauges(self) -> None:
        sched = self.engine.scheduler
        self.telemetry.on_gauges(len(sched.running), len(sched.waiting), self.engine.kv.usage)
