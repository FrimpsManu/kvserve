"""Model and engine configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def resolve_model_path(model: str) -> Path:
    """Return a local directory for `model`, downloading from the HF Hub if needed."""
    if os.path.isdir(model):
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_dir(cls, path: Path) -> ModelConfig:
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("model_type") != "llama":
            raise ValueError(f"unsupported model_type {cfg.get('model_type')!r}; only llama is supported")

        eos: set[int] = set()
        for source in (cfg, _read_json(path / "generation_config.json")):
            ids = source.get("eos_token_id")
            if isinstance(ids, int):
                eos.add(ids)
            elif isinstance(ids, list):
                eos.update(ids)

        num_heads = cfg["num_attention_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=cfg.get("num_key_value_heads", num_heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg.get("rope_theta", 10000.0),
            rope_scaling=cfg.get("rope_scaling"),
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            eos_token_ids=tuple(sorted(eos)),
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class EngineConfig:
    model: str = DEFAULT_MODEL
    device: str = field(default_factory=default_device)
    dtype: str = "auto"  # float32 on cpu, bfloat16 elsewhere
    block_size: int = 16  # tokens per KV block
    kv_cache_memory_gb: float = 2.0  # memory reserved for the paged KV pool
    num_kv_blocks: int | None = None  # overrides kv_cache_memory_gb when set
    max_num_seqs: int = 64  # max sequences in one batch
    max_num_batched_tokens: int = 2048  # per-step token budget (chunked prefill)
    max_model_len: int = 4096  # prompt + output cap per sequence
    enable_prefix_caching: bool = True
    attention_backend: str = "auto"  # auto | torch | triton
    enable_cuda_graphs: bool = True  # decode steps; needs CUDA + the triton backend
    max_graph_batch_size: int = 128  # largest decode batch captured (capped at max_num_seqs)
    seed: int = 0

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "auto":
            return torch.float32 if self.device == "cpu" else torch.bfloat16
        return _DTYPES[self.dtype]
