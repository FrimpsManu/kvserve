"""Llama decoder written for a flat, paged-KV batch layout.

Differences from the HF reference: no padding (tokens of all sequences are packed),
fused QKV and gate/up projections (fewer, larger GEMMs), and K/V are read from and
written to the shared paged pool instead of a per-sequence cache.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from kvserve.attention import AttentionMetadata, TorchAttention
from kvserve.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


def rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    dim = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    scaling = cfg.rope_scaling or {}
    if scaling.get("rope_type", scaling.get("type")) != "llama3":
        return inv_freq
    # Llama 3 frequency-dependent scaling for long context.
    factor = scaling["factor"]
    low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
    old_ctx = scaling["original_max_position_embeddings"]
    low_wavelen, high_wavelen = old_ctx / low, old_ctx / high
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int):
        super().__init__()
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), rope_inv_freq(cfg))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].to(q.dtype)[:, None]  # [T, 1, D]
        sin = self.sin[positions].to(q.dtype)[:, None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: type[TorchAttention]):
        super().__init__()
        self.backend = backend
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_size = cfg.num_heads * cfg.head_dim
        self.kv_size = cfg.num_kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        self.scale = self.head_dim**-0.5

    def project(
        self, x: torch.Tensor, positions: torch.Tensor, rope: RotaryEmbedding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV projection + RoPE: [T, hidden] -> q [T, H, D], k and v [T, H_kv, D]."""
        n = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(n, self.num_heads, self.head_dim)
        k = k.view(n, self.num_kv_heads, self.head_dim)
        v = v.view(n, self.num_kv_heads, self.head_dim)
        q, k = rope(q, k, positions)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        meta: AttentionMetadata,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write this step's K/V into the paged cache and attend over it: [T, H, D]."""
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        self.backend.write_kv(k_cache, v_cache, k, v, meta.slot_mapping)
        return self.backend.forward(q, k_cache, v_cache, meta, self.scale, out=out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: type[TorchAttention]):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, backend)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int, backend: type[TorchAttention] = TorchAttention):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, backend) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(cfg, max_positions)

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        """Returns final hidden states [T, hidden]. kv_caches: [layers, 2, blocks, bs, kv_heads, D]."""
        x, q, k, v = self.piece(0, input_ids, positions, None, None)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn.attend(q, k, v, kv_caches[i], meta)
            x, q, k, v = self.piece(i + 1, input_ids, positions, x, attn)
        return x

    def piece(
        self,
        i: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        x: torch.Tensor | None,
        attn: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Everything between attention call i-1 and attention call i, for i in 0..num_layers.

        Piece i finishes layer i-1 (output projection, residual, MLP) and starts layer i
        (norm, QKV projection, RoPE). Pieces are row-wise and never touch the KV cache,
        which is what makes them safe to capture as CUDA graphs with padded rows while
        attention runs eagerly in between (see cuda_graph.PiecewiseGraphRunner).

        Returns (x, q, k, v); for the last piece, (final hidden states, None, None, None).
        """
        if i == 0:
            x = self.embed_tokens(input_ids)
        else:
            assert x is not None and attn is not None
            prev = self.layers[i - 1]
            x = x + prev.self_attn.o_proj(attn.flatten(1))
            x = x + prev.mlp(prev.post_attention_layernorm(x))
        if i == len(self.layers):
            return self.norm(x), None, None, None
        layer = self.layers[i]
        q, k, v = layer.self_attn.project(layer.input_layernorm(x), positions, self.rope)
        return x, q, k, v

    @torch.inference_mode()
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()

    def load_weights(self, path: Path) -> None:
        params = dict(self.named_parameters())
        fused = {  # HF name suffix -> (our fused param suffix, shard index)
            "q_proj": ("qkv_proj", 0),
            "k_proj": ("qkv_proj", 1),
            "v_proj": ("qkv_proj", 2),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        }
        loaded: set[str] = set()
        for file in sorted(path.glob("*.safetensors")):
            with safe_open(file, framework="pt") as f:
                for hf_name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    name = hf_name.removeprefix("model.")
                    tensor = f.get_tensor(hf_name)
                    module, _, leaf = name.rpartition(".")
                    parent, _, proj = module.rpartition(".")
                    if proj in fused:
                        target, shard = fused[proj]
                        name = f"{parent}.{target}.{leaf}"
                        param = params[name]
                        offset, size = self._shard_range(target, shard)
                        param.data[offset : offset + size].copy_(tensor)
                    else:
                        if name == "lm_head.weight" and self.cfg.tie_word_embeddings:
                            continue
                        params[name].data.copy_(tensor)
                    loaded.add(name)
        missing = set(params) - loaded - ({"lm_head.weight"} if self.cfg.tie_word_embeddings else set())
        if missing:
            raise ValueError(f"weights missing from checkpoint: {sorted(missing)[:5]}")

    def _shard_range(self, target: str, shard: int) -> tuple[int, int]:
        c = self.cfg
        if target == "qkv_proj":
            q, kv = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
            return [(0, q), (q, kv), (q + kv, kv)][shard]
        return (shard * c.intermediate_size, c.intermediate_size)


def load_model(
    path: Path,
    cfg: ModelConfig,
    device: str,
    dtype: torch.dtype,
    max_positions: int,
    backend: type[TorchAttention] = TorchAttention,
) -> LlamaForCausalLM:
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg, max_positions, backend)
    model = model.to_empty(device=device).to(dtype)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    model.load_weights(path)
    # Built after the dtype cast so the cos/sin tables stay fp32.
    model.rope = RotaryEmbedding(cfg, max_positions).to(device)
    return model.eval()
