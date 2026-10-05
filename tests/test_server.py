"""HTTP layer: demo page, /stats, OpenAI-compatible streaming and usage reporting."""

import json

import pytest
from fastapi.testclient import TestClient

from kvserve import EngineConfig
from kvserve.server import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    app = create_app(EngineConfig(device="cpu", num_kv_blocks=256))
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
    for key in ("tokens_per_s", "running", "waiting", "kv_usage", "prefix_hit_rate", "info"):
        assert key in s
    assert s["info"]["kv_capacity_tokens"] == 256 * 16


def test_chat_stream_reports_usage_and_prefix_cache(client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " * 8},  # spans several KV blocks
        {"role": "user", "content": "Say hi."},
    ]
    body = {"messages": messages, "max_tokens": 5, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}  # fmt: skip

    usages = []
    for _ in range(2):
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            events = sse_events(r)
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-2]["choices"][0]["finish_reason"] == "length"
        usages.append(events[-1]["usage"])

    assert usages[0]["completion_tokens"] == 5
    assert usages[1]["prompt_tokens_details"]["cached_tokens"] > 0  # same prompt, reused blocks

    s = client.get("/stats").json()
    assert s["requests"] >= 2 and s["generated_tokens"] >= 10 and s["prefix_hit_rate"] > 0
    assert s["running"] == 0


def test_prompt_too_long_is_400(client):
    r = client.post("/v1/completions", json={"prompt": list(range(5000)), "max_tokens": 1})
    assert r.status_code == 400
