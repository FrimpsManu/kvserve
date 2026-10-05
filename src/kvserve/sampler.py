"""Batched token sampling: greedy, temperature, top-k and top-p, per request."""

from __future__ import annotations

import torch

from kvserve.sequence import SamplingParams


def sample(logits: torch.Tensor, params: list[SamplingParams], generator: torch.Generator | None) -> list[int]:
    """logits: [N, vocab] float32. Returns one token id per row."""
    greedy = logits.argmax(dim=-1)
    if all(p.temperature == 0 for p in params):
        return greedy.tolist()

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

    probs = logits.softmax(dim=-1)
    # MPS has no generator-aware multinomial; sample on CPU there for reproducibility.
    sampled = torch.multinomial(probs.cpu() if generator is not None else probs, 1, generator=generator)
    sampled = sampled.squeeze(1).to(device)
    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=device)
    return torch.where(is_greedy, greedy, sampled).tolist()
