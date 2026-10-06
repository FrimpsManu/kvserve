"""Request and engine telemetry shared by both engine front ends.

Per-request latency (TTFT, TPOT, end-to-end) is measured where requests enter and
leave the server; per-step numbers come from the engine. `Telemetry` turns both into
Prometheus metrics and the rolling `LiveStats` behind /stats.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from kvserve import metrics
from kvserve.engine import RequestOutput, StepStats


class LiveStats:
    """Rolling counters for the demo dashboard (/stats).

    Updated from whichever thread delivers outputs, read by HTTP handlers; a lock keeps
    snapshots consistent. Throughput is measured over the last WINDOW_S seconds.
    """

    WINDOW_S = 3.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, int]] = deque()
        self.generated_tokens = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.running = 0
        self.waiting = 0
        self.kv_usage = 0.0
        self.last_step_tokens = 0
        self.last_step_ms = 0.0

    def add_prompt(self, num_tokens: int, num_cached: int) -> None:
        with self._lock:
            self.prompt_tokens += num_tokens
            self.cached_prompt_tokens += num_cached

    def add_tokens(self, now: float, n: int) -> None:
        with self._lock:
            self.generated_tokens += n
            self._recent.append((now, n))

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            while self._recent and now - self._recent[0][0] > self.WINDOW_S:
                self._recent.popleft()
            recent = sum(n for _, n in self._recent)
            return {
                "tokens_per_s": recent / self.WINDOW_S,
                "generated_tokens": self.generated_tokens,
                "requests": self.requests,
                "running": self.running,
                "waiting": self.waiting,
                "kv_usage": self.kv_usage,
                "prompt_tokens": self.prompt_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "prefix_hit_rate": self.cached_prompt_tokens / self.prompt_tokens if self.prompt_tokens else 0.0,
                "last_step_tokens": self.last_step_tokens,
                "last_step_ms": self.last_step_ms,
            }


@dataclass
class RequestTiming:
    start: float
    first_token: float | None = None


class Telemetry:
    def __init__(self) -> None:
        self.stats = LiveStats()
        self.step_paths: dict[str, int] = {}

    def on_output(self, out: RequestOutput, timing: RequestTiming | None, now: float) -> None:
        if timing is not None and timing.first_token is None:
            timing.first_token = now
            metrics.ttft_seconds.observe(now - timing.start)
            metrics.prompt_tokens_total.inc(out.num_prompt_tokens)
            metrics.cached_prompt_tokens_total.inc(out.num_cached_tokens)
            self.stats.add_prompt(out.num_prompt_tokens, out.num_cached_tokens)
        metrics.generation_tokens_total.inc(len(out.new_token_ids))
        self.stats.add_tokens(now, len(out.new_token_ids))
        if not out.finished:
            return
        self.stats.requests += 1
        metrics.requests_total.labels(out.finish_reason.value if out.finish_reason else "unknown").inc()
        if timing is not None:
            metrics.e2e_seconds.observe(now - timing.start)
            n = len(out.output_token_ids)
            if n > 1 and timing.first_token is not None:
                metrics.tpot_seconds.observe((now - timing.first_token) / (n - 1))

    def on_step(self, step: StepStats | None) -> None:
        if step is None:
            return
        metrics.step_seconds.observe(step.duration_s)
        metrics.step_tokens.observe(step.num_tokens)
        if step.num_preempted:
            metrics.preemptions_total.inc(step.num_preempted)
        self.stats.last_step_tokens, self.stats.last_step_ms = step.num_tokens, step.duration_s * 1000

    def on_step_paths(self, paths: dict[str, int]) -> None:
        """Cumulative per-path step counts from the engine; exported as counter increments."""
        for path, count in paths.items():
            delta = count - self.step_paths.get(path, 0)
            if delta > 0:
                metrics.steps_total.labels(path).inc(delta)
        self.step_paths = dict(paths)

    def on_gauges(self, running: int, waiting: int, kv_usage: float) -> None:
        metrics.running_requests.set(running)
        metrics.waiting_requests.set(waiting)
        metrics.kv_cache_usage.set(kv_usage)
        self.stats.running, self.stats.waiting, self.stats.kv_usage = running, waiting, kv_usage
