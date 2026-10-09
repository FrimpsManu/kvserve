"""Speculative decoding: proposers that guess the next few tokens of decoding sequences.

The engine asks a proposer for up to `num_speculative_tokens` draft tokens per decoding
sequence before scheduling. The scheduler appends them to the sequence's one pending
token, the target model scores all of them in a single forward pass (the same
multi-token path chunked prefill uses), and `sampler.rejection_sample` keeps the
longest prefix the target agrees with plus one token of its own. A step therefore
emits between 1 and k + 1 tokens per sequence for one target forward pass.

K/V written for rejected drafts sits past `num_computed_tokens` and is simply
overwritten later, so the KV cache and prefix cache need no rollback.
"""

from __future__ import annotations

import numpy as np

from kvserve.sequence import Sequence


def max_draft_tokens(seq: Sequence, k: int, max_model_len: int) -> int:
    """Drafts worth proposing: never more than the tokens the sequence may still emit.

    A step emits accepted drafts plus one token, so leave room for that token.
    """
    remaining = min(seq.params.max_tokens - len(seq.output_token_ids), max_model_len - seq.num_tokens)
    return max(0, min(k, remaining - 1))


def ngram_lookup(tokens: np.ndarray, k: int, n_max: int, n_min: int) -> list[int]:
    """Prompt-lookup decoding: find the last n tokens earlier in the context, propose what followed.

    Tries the longest n first and takes the most recent earlier occurrence. Works when
    the output copies from its context: summaries, RAG answers, code edits, extraction.
    """
    length = len(tokens)
    for n in range(min(n_max, length - 1), n_min - 1, -1):
        suffix = tokens[length - n :]
        # Windows starting at 0..length-n-1 end before the suffix's own position, so each
        # match has at least one token following it.
        windows = np.lib.stride_tricks.sliding_window_view(tokens[: length - 1], n)
        hits = np.flatnonzero((windows == suffix).all(axis=1))
        if hits.size:
            start = int(hits[-1]) + n
            return tokens[start : start + k].tolist()
    return []


class NgramProposer:
    """Drafts from the sequence's own context; no draft model, no extra GPU work."""

    def __init__(self, k: int, max_model_len: int, n_max: int = 4, n_min: int = 2):
        self.k, self.max_model_len, self.n_max, self.n_min = k, max_model_len, n_max, n_min

    def propose(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            k = max_draft_tokens(seq, self.k, self.max_model_len)
            seq.spec_token_ids = ngram_lookup(np.asarray(seq.token_ids), k, self.n_max, self.n_min) if k else []
