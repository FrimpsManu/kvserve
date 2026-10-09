"""Sampling and speculative-decoding verification, checked statistically on a tiny vocab."""

import torch

from kvserve.sampler import probs_from_logits, rejection_sample, sample
from kvserve.sequence import SamplingParams

GREEDY = SamplingParams(temperature=0)
RANDOM = SamplingParams(temperature=1.0)


def one_hot_logits(tokens: list[int], vocab: int = 8) -> torch.Tensor:
    logits = torch.zeros(len(tokens), vocab)
    logits[torch.arange(len(tokens)), torch.tensor(tokens)] = 10.0
    return logits


def test_greedy_accepts_matching_prefix_then_corrects():
    # Target argmax per row: 3, 5, 1, 7. Drafts agree on the first two only.
    logits = one_hot_logits([3, 5, 1, 7])
    assert rejection_sample(logits, [[3, 5, 2]], [GREEDY], None) == [[3, 5, 1]]
    assert rejection_sample(logits, [[3, 5, 1]], [GREEDY], None) == [[3, 5, 1, 7]]  # all accepted + bonus
    assert rejection_sample(logits[:1], [[]], [GREEDY], None) == [[3]]  # no drafts = plain greedy


def test_batch_mixes_sequences_with_different_draft_counts():
    logits = one_hot_logits([3, 5, 1, 2, 4])  # seq A: 3 rows (2 drafts), seq B: 2 rows (1 draft)
    out = rejection_sample(logits, [[3, 0], [2]], [GREEDY, GREEDY], None)
    assert out == [[3, 5], [2, 4]]


def _first_token_freqs(target: torch.Tensor, draft_q: torch.Tensor | None, trials: int, seed: int) -> torch.Tensor:
    vocab = target.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits = target.log().expand(trials * 2, vocab).contiguous()  # 2 rows each: one draft + bonus
    if draft_q is None:  # deterministic n-gram style draft, always token 0
        drafts = [[0]] * trials
        probs = None
    else:
        drafts = [[t] for t in torch.multinomial(draft_q, trials, replacement=True, generator=gen).tolist()]
        probs = [draft_q[None]] * trials
    out = rejection_sample(logits, drafts, [RANDOM] * trials, gen, probs)
    first = torch.tensor([o[0] for o in out])
    return torch.bincount(first, minlength=vocab).float() / trials


def test_rejection_sampling_preserves_target_distribution_with_draft_model():
    target = torch.tensor([0.1, 0.4, 0.05, 0.3, 0.15])
    draft = torch.tensor([0.3, 0.1, 0.3, 0.2, 0.1])  # deliberately a poor draft
    freqs = _first_token_freqs(target, draft, trials=20000, seed=0)
    assert torch.allclose(freqs, target, atol=0.015), freqs


def test_rejection_sampling_preserves_target_distribution_with_ngram_drafts():
    target = torch.tensor([0.25, 0.4, 0.05, 0.3])
    freqs = _first_token_freqs(target, None, trials=20000, seed=1)
    assert torch.allclose(freqs, target, atol=0.015), freqs


def test_probs_respect_top_k_and_top_p():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    top_k = probs_from_logits(logits, [SamplingParams(top_k=2)])
    assert (top_k[0, 2:] == 0).all() and torch.isclose(top_k.sum(), torch.tensor(1.0))
    top_p = probs_from_logits(logits, [SamplingParams(top_p=0.5)])
    assert top_p[0, 0] == 1.0


def test_sample_greedy_and_seeded():
    logits = torch.randn(4, 16)
    assert sample(logits, [GREEDY] * 4, None) == logits.argmax(-1).tolist()
    a = sample(logits, [RANDOM] * 4, torch.Generator().manual_seed(3))
    b = sample(logits, [RANDOM] * 4, torch.Generator().manual_seed(3))
    assert a == b
