"""Trainable PyTorch twin of the from-scratch Transformer (the model we pretrain).

Same architecture as ``model_numpy`` (pre-LN decoder blocks, causal MHA, GELU MLP,
weight-tied head).  Requires PyTorch:  ``pip install score-bundle[torch]`` or
``pip install torch``.  Kept dependency-light so the rest of the package imports
without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

try:  # optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch
    _HAS_TORCH = False


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.1


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "model_torch requires PyTorch. Install with `pip install torch` "
            "(or `pip install score-bundle[torch]`). The NumPy model in "
            "score_bundle.lm.model_numpy needs no extra dependencies."
        )


if _HAS_TORCH:

    class Block(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True
            )
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x

    class MusicGPT(nn.Module):
        """Decoder-only Transformer over music tokens."""

        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.tok.weight  # weight tying

        def forward(self, idx, return_hidden: bool = False):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok(idx) + self.pos(pos)[None])
            for blk in self.blocks:
                x = blk(x)
            hidden = self.ln_f(x)
            logits = self.head(hidden)
            return (logits, hidden) if return_hidden else logits


def build_model(cfg: "GPTConfig"):
    """Construct a trainable MusicGPT (raises if torch is unavailable)."""
    _require_torch()
    return MusicGPT(cfg)


def train_lm(
    model,
    batches: Iterator[Tuple["object", "object"]],
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    log_every: int = 50,
):
    """Minimal next-token training loop. ``batches`` yields (x, y) int tensors/arrays."""
    _require_torch()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses = []
    for step, (x, y) in enumerate(batches):
        x = torch.as_tensor(x, dtype=torch.long)
        y = torch.as_tensor(y, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"step {step:5d} | loss {float(loss):.4f}")
    return losses
