"""Streaming detokenization."""

from __future__ import annotations

from typing import Any


class IncrementalDetokenizer:
    """Turns a growing token list into text deltas in O(1) work per token.

    Decoding the whole output on every token is O(n) per token, O(n^2) per request,
    and runs once per token for every open stream. Instead, decode only a short window:
    `prefix_offset..read_offset` is text already emitted, kept as context so tokenizers
    that merge across token boundaries decode correctly, and `read_offset..` is new.
    A trailing U+FFFD means an incomplete UTF-8 sequence, so wait for more tokens.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0

    def delta(self, token_ids: list[int], final: bool) -> str:
        decode = self.tokenizer.decode
        prefix = decode(token_ids[self.prefix_offset : self.read_offset], skip_special_tokens=True)
        text = decode(token_ids[self.prefix_offset :], skip_special_tokens=True)
        if len(text) <= len(prefix) or (text.endswith("\ufffd") and not final):
            return ""
        self.prefix_offset, self.read_offset = self.read_offset, len(token_ids)
        return text[len(prefix) :]
