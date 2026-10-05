"""Incremental detokenization must reproduce the full decode exactly."""

import random

import pytest

from kvserve.detokenizer import IncrementalDetokenizer

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from kvserve.config import DEFAULT_MODEL, resolve_model_path

    return AutoTokenizer.from_pretrained(resolve_model_path(DEFAULT_MODEL))


TEXTS = [
    "Hello, world! The quick brown fox jumps over the lazy dog.",
    "Emoji 🚀🔥 and accents: café, naïve, jalapeño.",
    "中文测试：大型语言模型推理。日本語のテキストも。",
    "def f(x):\n    return x ** 2  # code with    spaces\n",
    "Mixed 🤖 текст with ünïcödé and 数字 123.",
]


def stream(tokenizer, ids: list[int]) -> str:
    detok = IncrementalDetokenizer(tokenizer)
    return "".join(detok.delta(ids[: i + 1], final=(i == len(ids) - 1)) for i in range(len(ids)))


@pytest.mark.parametrize("text", TEXTS)
def test_incremental_matches_full_decode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert stream(tokenizer, ids) == tokenizer.decode(ids, skip_special_tokens=True)


def test_random_token_sequences(tokenizer):
    rng = random.Random(0)
    for _ in range(50):
        ids = [rng.randrange(0, 128000) for _ in range(rng.randrange(1, 40))]
        assert stream(tokenizer, ids) == tokenizer.decode(ids, skip_special_tokens=True)


def test_never_emits_replacement_char_mid_stream(tokenizer):
    ids = tokenizer.encode("🚀🔥🤖", add_special_tokens=False)
    detok = IncrementalDetokenizer(tokenizer)
    for i in range(len(ids) - 1):
        assert "�" not in detok.delta(ids[: i + 1], final=False)


def test_work_per_token_is_bounded(tokenizer):
    """Each call decodes a short window, not the whole output."""
    calls: list[int] = []

    class Spy:
        def decode(self, ids, **kw):
            calls.append(len(ids))
            return tokenizer.decode(ids, **kw)

    ids = tokenizer.encode("word " * 400, add_special_tokens=False)
    detok = IncrementalDetokenizer(Spy())
    for i in range(len(ids)):
        detok.delta(ids[: i + 1], final=False)
    assert max(calls) <= 8, max(calls)
