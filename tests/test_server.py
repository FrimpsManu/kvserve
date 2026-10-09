"""HTTP layer, in both engine modes: demo page, /stats, streaming, usage, metrics."""

import json

import pytest
from fastapi.testclient import TestClient

from kvserve import EngineConfig
from kvserve.server import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", params=["process", "thread"])
def client(request):
    app = create_app(EngineConfig(device="cpu", num_kv_blocks=256), engine_mode=request.param)
    with TestClient(app) as c:
        yield c


def sse_events(resp) -> list[dict]:
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "kvserve" in r.text and "/stats" in r.text


def test_stats_shape(client):
    s = client.get("/stats").json()
    for key in ("tokens_per_s", "running", "waiting", "kv_usage", "prefix_hit_rate", "info", "engine_mode"):
        assert key in s
    assert s["info"]["kv_capacity_tokens"] == 256 * 16


def test_chat_stream_reports_usage_and_prefix_cache(client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " * 8},  # spans several KV blocks
        {"role": "user", "content": "Say hi."},
    ]
    body = {"messages": messages, "max_tokens": 5, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}  # fmt: skip

    usages, texts = [], []
    for _ in range(2):
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            events = sse_events(r)
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-2]["choices"][0]["finish_reason"] == "length"
        usages.append(events[-1]["usage"])
        texts.append("".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"]))

    assert usages[0]["completion_tokens"] == 5
    assert usages[1]["prompt_tokens_details"]["cached_tokens"] > 0  # same prompt, reused blocks
    assert texts[0] == texts[1] and texts[0]  # greedy: identical text both times

    s = client.get("/stats").json()
    assert s["requests"] >= 2 and s["generated_tokens"] >= 10 and s["prefix_hit_rate"] > 0
    assert s["running"] == 0


def test_non_streaming_completion(client):
    r = client.post("/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 4, "temperature": 0})
    body = r.json()
    assert r.status_code == 200 and body["usage"]["completion_tokens"] == 4
    assert body["choices"][0]["text"]


def test_metrics_include_step_paths(client):
    client.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 3, "temperature": 0})
    text = client.get("/metrics").text
    assert 'kvserve_steps_total{path="eager"}' in text  # CPU runs eagerly
    assert "kvserve_time_to_first_token_seconds_count" in text


def test_prompt_too_long_is_400(client):
    r = client.post("/v1/completions", json={"prompt": list(range(5000)), "max_tokens": 1})
    assert r.status_code == 400


def test_engine_process_crash_is_reported():
    """If the engine process dies, health turns 503 and requests fail instead of hanging."""
    app = create_app(EngineConfig(device="cpu", num_kv_blocks=64), engine_mode="process")
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 200
        engine = c.app.state.engine
        engine._proc.kill()
        engine._proc.join(timeout=10)
        assert c.get("/health").status_code == 503
        r = c.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 2})
        assert r.status_code >= 500


@pytest.mark.parametrize("engine_mode", ["process", "thread"])
def test_speculative_stream_matches_plain_and_reports_acceptance(engine_mode):
    body = {"prompt": "Repeat exactly: one two three four five six. one two three four", "max_tokens": 24,
            "temperature": 0, "stream": True}  # fmt: skip
    texts = []
    for spec in ("none", "ngram"):
        config = EngineConfig(device="cpu", num_kv_blocks=256, speculative_method=spec)
        app = create_app(config, engine_mode=engine_mode)
        with TestClient(app) as c, c.stream("POST", "/v1/completions", json=body) as r:
            texts.append("".join(e["choices"][0]["text"] for e in sse_events(r)))
            stats = c.get("/stats").json()
    assert texts[0] == texts[1] and texts[0]
    assert stats["info"]["speculative"] == "ngram lookup, 4 tokens per step"
    assert 0 < stats["spec_accepted_tokens"] <= stats["spec_draft_tokens"]
