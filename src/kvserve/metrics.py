"""Prometheus metrics. Names follow the `kvserve_` prefix and Prometheus unit conventions."""

from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

requests_total = Counter("kvserve_requests_total", "Finished requests", ["finish_reason"])
prompt_tokens_total = Counter("kvserve_prompt_tokens_total", "Prompt tokens received")
cached_prompt_tokens_total = Counter("kvserve_cached_prompt_tokens_total", "Prompt tokens served from prefix cache")
generation_tokens_total = Counter("kvserve_generation_tokens_total", "Tokens generated")
steps_total = Counter("kvserve_steps_total", "Engine steps by execution path", ["path"])
preemptions_total = Counter("kvserve_preemptions_total", "Sequences preempted for lack of KV blocks")

ttft_seconds = Histogram("kvserve_time_to_first_token_seconds", "Time to first token", buckets=_LATENCY_BUCKETS)
tpot_seconds = Histogram(
    "kvserve_time_per_output_token_seconds", "Mean inter-token latency per request", buckets=_LATENCY_BUCKETS
)
e2e_seconds = Histogram("kvserve_e2e_request_latency_seconds", "End-to-end request latency", buckets=_LATENCY_BUCKETS)
step_seconds = Histogram("kvserve_step_duration_seconds", "Engine step duration", buckets=_LATENCY_BUCKETS)
step_tokens = Histogram(
    "kvserve_step_tokens", "Tokens processed per engine step", buckets=(1, 4, 16, 64, 256, 512, 1024, 2048, 4096)
)

running_requests = Gauge("kvserve_running_requests", "Sequences in the running batch")
waiting_requests = Gauge("kvserve_waiting_requests", "Sequences waiting for admission")
kv_cache_usage = Gauge("kvserve_kv_cache_usage_ratio", "Fraction of KV blocks in use")
