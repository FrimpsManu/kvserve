"""Batched token sampling: greedy, temperature, top-k and top-p, per request.

`sample` draws one token per row. `rejection_sample` verifies speculative draft tokens
against the target model's distribution (see its docstring).
"""

from __future__ import annotations

import torch

from kvserve.sequence import SamplingParams


def probs_from_logits(logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
    """The distribution each row samples from: temperature, then top-k, then top-p.

    logits: [N, vocab] float32. Greedy rows (temperature 0) get temperature 1; callers
    take their argmax instead of using these probabilities.
    """
    device = logits.device
    temps = torch.tensor([p.temperature or 1.0 for p in params], device=device)
    logits = logits / temps[:, None]

    top_ks = [p.top_k for p in params]
    if any(k > 0 for k in top_ks):
        vocab = logits.shape[-1]
        k = torch.tensor([k if k > 0 else vocab for k in top_ks], device=device)
        kth = logits.sort(dim=-1, descending=True).values.gather(1, (k - 1)[:, None])
        logits = logits.masked_fill(logits < kth, float("-inf"))

    top_ps = torch.tensor([p.top_p for p in params], device=device)
    if (top_ps < 1).any():
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop tokens once the mass *before* them already reaches top_p (keeps >= 1 token).
        drop = (probs.cumsum(dim=-1) - probs) >= top_ps[:, None]
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(1, order, sorted_logits)

    return logits.softmax(dim=-1)


def _multinomial(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    # MPS has no generator-aware multinomial; sample on CPU there for reproducibility.
    sampled = torch.multinomial(probs.cpu() if generator is not None else probs, 1, generator=generator)
    return sampled.squeeze(1).to(probs.device)


def sample(logits: torch.Tensor, params: list[SamplingParams], generator: torch.Generator | None) -> list[int]:
    """logits: [N, vocab] float32. Returns one token id per row."""
    greedy = logits.argmax(dim=-1)
    if all(p.temperature == 0 for p in params):
        return greedy.tolist()
    sampled = _multinomial(probs_from_logits(logits, params), generator)
    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=logits.device)
    return torch.where(is_greedy, greedy, sampled).tolist()


def rejection_sample(
    logits: torch.Tensor,
    draft_token_ids: list[list[int]],
    params: list[SamplingParams],
    generator: torch.Generator | None,
    draft_probs: list[torch.Tensor | None] | None = None,
) -> list[list[int]]:
    """Speculative-decoding verification (Leviathan et al. 2023; Chen et al. 2023).

    Sequence i proposed draft tokens d_1..d_k. The target model scored all of them in one
    forward pass, giving k + 1 rows of logits: row j is the target's distribution p_j
    for the token after d_1..d_j (row 0 follows the last real token). Walk the drafts in
    order and accept d_j with probability min(1, p_j(d_j) / q_j(d_j)), where q_j is the
    distribution the draft was sampled from. On the first rejection, sample a replacement
    from norm(max(p_j - q_j, 0)) and stop; if every draft is accepted, sample a bonus
    token from p_k. The emitted tokens are then distributed exactly as if the target had
    sampled them one at a time, so speculation changes speed, never outputs.

    Greedy rows accept d_j iff it is the target's argmax, and emit the argmax on a
    mismatch, which is token-for-token identical to plain greedy decoding.

    logits: [sum(k_i + 1), vocab] float32, sequences' rows concatenated in order.
    draft_probs[i]: [k_i, vocab] draft distributions, or None when the drafts were
    proposed deterministically (n-gram lookup), i.e. q_j is one-hot on d_j.
    Returns, per sequence, the accepted drafts followed by one sampled token
    (between 1 and k_i + 1 tokens).
    """
    device = logits.device
    draft_probs = draft_probs or [None] * len(params)
    row_params = [p for p, d in zip(params, draft_token_ids, strict=True) for _ in range(len(d) + 1)]
    greedy = logits.argmax(dim=-1).tolist()
    any_random = any(p.temperature > 0 for p in params)
    probs = probs_from_logits(logits, row_params) if any_random else None
    if any_random:
        # One uniform per draft position, drawn up front so the walk below needs no sync.
        total = sum(len(d) for d in draft_token_ids)
        uniforms = torch.rand(total, generator=generator).tolist()

    out: list[list[int]] = []
    row = 0  # first row of the current sequence
    u = 0  # next uniform
    for i, (drafts, p) in enumerate(zip(draft_token_ids, params, strict=True)):
        k = len(drafts)
        accepted: list[int] = []
        if p.temperature == 0:
            for j, d in enumerate(drafts):
                if d != greedy[row + j]:
                    break
                accepted.append(d)
            out.append(accepted + [greedy[row + len(accepted)]])
            row += k + 1
            continue

        assert probs is not None
        q = draft_probs[i]
        idx = torch.arange(k, device=device)
        d_t = torch.tensor(drafts, device=device, dtype=torch.long)
        p_d = probs[row + idx, d_t].tolist() if k else []
        q_d = q[idx, d_t].tolist() if q is not None and k else [1.0] * k
        rejected_at = None
        for j, d in enumerate(drafts):
            if q_d[j] > 0 and uniforms[u + j] < min(1.0, p_d[j] / q_d[j]):
                accepted.append(d)
            else:
                rejected_at = j
                break
        u += k

        if rejected_at is None:  # every draft accepted: bonus token from the last row
            token = _multinomial(probs[row + k][None], generator).item()
        else:
            j = rejected_at
            if q is None:
                residual = probs[row + j].clone()
                residual[drafts[j]] = 0
            else:
                residual = (probs[row + j] - q[j]).clamp_(min=0)
            if residual.sum() <= 0:  # p == q exactly: any token from p is correct
                residual = probs[row + j]
            token = _multinomial((residual / residual.sum())[None], generator).item()
        out.append(accepted + [token])
        row += k + 1
    return out
