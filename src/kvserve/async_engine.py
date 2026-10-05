"""Asyncio front end for LLMEngine.

The engine is single-threaded by design: one dedicated thread owns it and runs the
step loop, so GPU work never blocks the event loop and engine state needs no locks.
Requests and aborts reach that thread through a command queue; tokens flow back to
each request's asyncio.Queue via `call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from kvserve import metrics
from kvserve.config import EngineConfig
from kvserve.engine import LLMEngine, RequestOutput
from kvserve.sequence import SamplingParams


@dataclass
class _Stream:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    start: float
    first_token: float | None = None


class AsyncLLMEngine:
    def __init__(self, config: EngineConfig):
        self.engine = LLMEngine(config)
        self.tokenizer = self.engine.tokenizer
        self.config = config
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
        stream = _Stream(asyncio.get_running_loop(), asyncio.Queue(), time.perf_counter())
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
                if not eng.has_unfinished():
                    continue
                outputs = eng.step()
                self._record_step()
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
        now = time.perf_counter()
        if stream is not None and stream.first_token is None:
            stream.first_token = now
            metrics.ttft_seconds.observe(now - stream.start)
            metrics.prompt_tokens_total.inc(out.num_prompt_tokens)
            metrics.cached_prompt_tokens_total.inc(out.num_cached_tokens)
        metrics.generation_tokens_total.inc(len(out.new_token_ids))
        if out.finished:
            metrics.requests_total.labels(out.finish_reason.value if out.finish_reason else "unknown").inc()
            if stream is not None:
                metrics.e2e_seconds.observe(now - stream.start)
                n = len(out.output_token_ids)
                if n > 1 and stream.first_token is not None:
                    metrics.tpot_seconds.observe((now - stream.first_token) / (n - 1))
        if stream is not None:
            # Snapshot the token list: the engine keeps appending to it.
            out.output_token_ids = list(out.output_token_ids)
            stream.loop.call_soon_threadsafe(stream.queue.put_nowait, out)

    def _record_step(self) -> None:
        stats = self.engine.last_step
        sched = self.engine.scheduler
        if stats is not None:
            metrics.step_seconds.observe(stats.duration_s)
            metrics.step_tokens.observe(stats.num_tokens)
            if stats.num_preempted:
                metrics.preemptions_total.inc(stats.num_preempted)
        metrics.running_requests.set(len(sched.running))
        metrics.waiting_requests.set(len(sched.waiting))
        metrics.kv_cache_usage.set(self.engine.kv.usage)
