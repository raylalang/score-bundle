"""A from-scratch decoder-only Transformer (NumPy) — forward pass + sampling.

This is the *legible* implementation: token + positional embeddings, pre-LayerNorm
blocks with causal multi-head self-attention and a GELU MLP, weight-tied LM head.
It is not trained (no autograd); it exists to (a) make the architecture explicit,
(b) provide per-note embeddings for the Phase-1 integration, and (c) be unit-testable
anywhere.  The trainable twin is ``model_torch.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 512
    mlp_ratio: int = 4

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


def init_params(config: GPTConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    d = config.d_model
    scale = 0.02
    p: Dict[str, np.ndarray] = {
        "wte": rng.normal(0, scale, (config.vocab_size, d)),
        "wpe": rng.normal(0, scale, (config.block_size, d)),
        "ln_f_g": np.ones(d),
        "ln_f_b": np.zeros(d),
    }
    for i in range(config.n_layer):
        p[f"h{i}.ln1_g"] = np.ones(d)
        p[f"h{i}.ln1_b"] = np.zeros(d)
        p[f"h{i}.attn_qkv"] = rng.normal(0, scale, (d, 3 * d))
        p[f"h{i}.attn_proj"] = rng.normal(0, scale, (d, d))
        p[f"h{i}.ln2_g"] = np.ones(d)
        p[f"h{i}.ln2_b"] = np.zeros(d)
        p[f"h{i}.mlp_fc"] = rng.normal(0, scale, (d, config.mlp_ratio * d))
        p[f"h{i}.mlp_proj"] = rng.normal(0, scale, (config.mlp_ratio * d, d))
    return p


def layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _attention(x: np.ndarray, p: Dict[str, np.ndarray], prefix: str, cfg: GPTConfig) -> np.ndarray:
    T, d = x.shape
    qkv = x @ p[f"{prefix}.attn_qkv"]              # (T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)            # each (T, d)
    H, dh = cfg.n_head, cfg.d_head
    q = q.reshape(T, H, dh).transpose(1, 0, 2)     # (H, T, dh)
    k = k.reshape(T, H, dh).transpose(1, 0, 2)
    v = v.reshape(T, H, dh).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh)  # (H, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # causal
    scores = np.where(mask[None], -1e30, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ v                                  # (H, T, dh)
    out = out.transpose(1, 0, 2).reshape(T, d)      # (T, d)
    return out @ p[f"{prefix}.attn_proj"]


def forward(
    params: Dict[str, np.ndarray], idx: np.ndarray, cfg: GPTConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on a single sequence ``idx`` of shape (T,).

    Returns (logits (T, vocab), hidden (T, d_model)) where ``hidden`` is the
    pre-head representation (after the final LayerNorm).
    """
    idx = np.asarray(idx, dtype=int)
    T = idx.shape[0]
    if T > cfg.block_size:
        raise ValueError(f"sequence length {T} exceeds block_size {cfg.block_size}")
    x = params["wte"][idx] + params["wpe"][:T]
    for i in range(cfg.n_layer):
        pre = layernorm(x, params[f"h{i}.ln1_g"], params[f"h{i}.ln1_b"])
        x = x + _attention(pre, params, f"h{i}", cfg)
        pre = layernorm(x, params[f"h{i}.ln2_g"], params[f"h{i}.ln2_b"])
        h = gelu(pre @ params[f"h{i}.mlp_fc"]) @ params[f"h{i}.mlp_proj"]
        x = x + h
    hidden = layernorm(x, params["ln_f_g"], params["ln_f_b"])
    logits = hidden @ params["wte"].T   # weight-tied head
    return logits, hidden


def generate(
    params: Dict[str, np.ndarray],
    cfg: GPTConfig,
    idx: np.ndarray,
    max_new_tokens: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Autoregressively sample ``max_new_tokens`` continuing ``idx`` (1-D)."""
    idx = list(np.asarray(idx, dtype=int))
    for _ in range(max_new_tokens):
        context = np.array(idx[-cfg.block_size :])
        logits, _ = forward(params, context, cfg)
        logits = logits[-1] / max(temperature, 1e-6)
        if top_k is not None:
            thresh = np.sort(logits)[-top_k]
            logits = np.where(logits < thresh, -1e30, logits)
        probs = softmax(logits)
        idx.append(int(rng.choice(len(probs), p=probs)))
    return np.array(idx)
