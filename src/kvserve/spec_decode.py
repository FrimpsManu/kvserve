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
import torch

from kvserve.sampler import probs_from_logits
from kvserve.sequence import SamplingParams, Sequence


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


class DraftModelProposer:
    """Drafts with a small model that shares the target's tokenizer (e.g. Llama-3.2-1B for 8B).

    The draft model has its own K/V pool but indexes it with the target's block tables:
    the same block size and token positions mean the target's allocator manages both, so
    there is no second allocator and preemption or freeing covers the draft too. Each
    sequence records how many of its tokens have draft K/V (`num_draft_computed`).

    Per step: one catch-up pass feeds the draft every token it has not seen yet (the
    whole prompt after admission, then usually the 1-2 tokens emitted by the last
    verification) and samples d_1; then k - 1 single-token passes sample d_2..d_k,
    replayed as CUDA graphs on GPU. Draft K/V for rejected drafts is discarded the same
    way as the target's: `num_draft_computed` is clamped to the verified length.
    """

    def __init__(self, config, target, kv) -> None:  # EngineConfig, ModelRunner, KVCacheManager
        from kvserve.config import ModelConfig, resolve_model_path
        from kvserve.cuda_graph import DecodeGraphRunner
        from kvserve.model import load_model

        self.k = config.num_speculative_tokens
        self.max_model_len = config.max_model_len
        self.kv = kv
        self.device, self.backend = target.device, target.attn_backend
        path = resolve_model_path(config.draft_model)
        mc = ModelConfig.from_dir(path)
        if mc.vocab_size != target.model_config.vocab_size:
            raise ValueError(
                f"draft model vocab ({mc.vocab_size}) differs from the target's ({target.model_config.vocab_size})"
            )
        self.model = load_model(path, mc, config.device, target.dtype, config.max_model_len, target.attn_backend)
        # Same block count as the target plus the scratch block, at the draft's (smaller) shape.
        self.kv_caches = torch.zeros(
            mc.num_layers, 2, target.num_kv_blocks + 1, config.block_size, mc.num_kv_heads, mc.head_dim,
            dtype=target.dtype, device=self.device,
        )  # fmt: skip
        self.generator = torch.Generator().manual_seed(config.seed + 1)
        self.graphs = None
        if target.graphs is not None:  # CUDA + Triton with graphs enabled
            self.graphs = DecodeGraphRunner(
                self.model,
                self.kv_caches,
                scratch_block=target.scratch_block,
                block_size=config.block_size,
                max_batch=target.graphs.max_batch,
                max_blocks_per_seq=-(-config.max_model_len // config.block_size),
            )
            self.graphs.capture()

    def _forward(self, input_ids, positions, slots, tables, seq_lens, query_lens) -> torch.Tensor:
        from kvserve.model_runner import build_metadata

        if self.graphs is not None and max(query_lens) == 1 and len(input_ids) <= self.graphs.max_batch:
            return self.graphs.run(input_ids, positions, slots, tables, seq_lens)
        meta = build_metadata(self.device, self.backend, tables, slots, seq_lens, query_lens)
        ids, pos = torch.tensor(input_ids, device=self.device), torch.tensor(positions, device=self.device)
        return self.model(ids, pos, self.kv_caches, meta)

    def _sample(self, hidden: torch.Tensor, params: list[SamplingParams]) -> tuple[list[int], torch.Tensor | None]:
        """Draft tokens, plus the distributions they came from when any row samples randomly."""
        logits = self.model.compute_logits(hidden)
        greedy = logits.argmax(dim=-1)
        if all(p.temperature == 0 for p in params):
            return greedy.tolist(), None
        probs = probs_from_logits(logits, params)
        sampled = torch.multinomial(probs.cpu(), 1, generator=self.generator).squeeze(1).to(self.device)
        is_greedy = torch.tensor([p.temperature == 0 for p in params], device=self.device)
        return torch.where(is_greedy, greedy, sampled).tolist(), probs

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence]) -> None:
        active: list[tuple[Sequence, int]] = []
        for seq in seqs:
            seq.spec_token_ids, seq.spec_draft_probs = [], None
            k = max_draft_tokens(seq, self.k, self.max_model_len)
            # Reserve target slots for the pending token + k drafts now: the draft writes
            # its own K/V at those positions before the scheduler runs.
            if k and self.kv.allocate_slots(seq, 1 + k):
                active.append((seq, k))
        if not active:
            return

        # Catch-up pass: every token without draft K/V, through the pending token.
        input_ids, positions, slots, seq_lens, query_lens = [], [], [], [], []
        for seq, _ in active:
            start, end = min(seq.num_draft_computed, seq.num_computed_tokens), seq.num_tokens
            input_ids += seq.token_ids[start:end]
            positions += range(start, end)
            slots += self.kv.slots(seq, start, end - start)
            seq_lens.append(end)
            query_lens.append(end - start)
        tables = [seq.block_table for seq, _ in active]
        hidden = self._forward(input_ids, positions, slots, tables, seq_lens, query_lens)
        last_rows = torch.tensor(query_lens, device=self.device).cumsum(0) - 1
        tokens, probs = self._sample(hidden[last_rows], [seq.params for seq, _ in active])
        drafts = [[t] for t in tokens]
        draft_probs = [[probs[i]] if probs is not None else None for i in range(len(active))]

        # k - 1 decode passes over the sequences that still want more drafts.
        for j in range(1, max(k for _, k in active)):
            batch = [i for i, (_, k) in enumerate(active) if k > j]
            seqs_j = [active[i][0] for i in batch]
            pos = [s.num_tokens - 1 + j for s in seqs_j]
            hidden = self._forward(
                [drafts[i][-1] for i in batch],
                pos,
                [self.kv.slot(s, p) for s, p in zip(seqs_j, pos, strict=True)],
                [s.block_table for s in seqs_j],
                [p + 1 for p in pos],
                [1] * len(batch),
            )
            tokens, probs = self._sample(hidden, [s.params for s in seqs_j])
            for row, i in enumerate(batch):
                drafts[i].append(tokens[row])
                if draft_probs[i] is not None:
                    draft_probs[i].append(probs[row])

        for i, (seq, k) in enumerate(active):
            seq.spec_token_ids = drafts[i]
            seq.spec_draft_probs = torch.stack(draft_probs[i]) if draft_probs[i] is not None else None
            seq.num_draft_computed = seq.num_tokens + k - 1  # d_k itself was never fed back
