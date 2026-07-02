"""The Phase-0 music language model — a from-scratch decoder-only Transformer (PyTorch).

Built from the ground up: hand-written causal self-attention (q/k/v ``nn.Linear`` +
masked softmax — no ``nn.MultiheadAttention`` and no ``transformers`` library), a GELU
MLP, pre-LayerNorm residual blocks, and a weight-tied LM head. Every part is inspectable,
which is the point (interpretability/uncertainty claims, attention hooks, etc.).

Requires PyTorch (``pip install -e '.[train]'``). The import is guarded so the rest of the
package still imports without torch — but the LM itself is torch-based, so training and the
LM tests need it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

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
    causal: bool = True  # False = bidirectional encoder (Stage-2 masked objective)


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "The Phase-0 music LM is PyTorch-based. Install with "
            "`pip install -e '.[train]'` (or `pip install torch`)."
        )


if _HAS_TORCH:

    class CausalSelfAttention(nn.Module):
        """Multi-head self-attention, written by hand.

        Causal (autoregressive mask) by default; with ``cfg.causal=False`` it is fully
        bidirectional — the Stage-2 masked-objective encoder.  ``getattr`` keeps old
        pickled ``GPTConfig`` checkpoints (which predate the flag) loading as causal.
        """

        def __init__(self, cfg: GPTConfig):
            super().__init__()
            assert cfg.d_model % cfg.n_head == 0
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.attn_drop = nn.Dropout(cfg.dropout)
            self.resid_drop = nn.Dropout(cfg.dropout)
            self.n_head = cfg.n_head
            self.d_head = cfg.d_model // cfg.n_head
            self.causal = bool(getattr(cfg, "causal", True))

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(C, dim=2)
            q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, h, T, dh)
            k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B, h, T, T)
            if self.causal:
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
                att = att.masked_fill(mask, float("-inf"))
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v                                                  # (B, h, T, dh)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_drop(self.proj(y))

    class MLP(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.fc = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model)
            self.proj = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):
            return self.drop(self.proj(F.gelu(self.fc(x))))

    class Block(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
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

        def _backbone(self, idx):
            B, T = idx.shape
            assert T <= self.cfg.block_size, f"sequence {T} exceeds block_size {self.cfg.block_size}"
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok(idx) + self.pos(pos)[None])
            for blk in self.blocks:
                x = blk(x)
            return self.ln_f(x)  # pre-head hidden states

        def forward(self, idx, targets=None):
            hidden = self._backbone(idx)
            logits = self.head(hidden)
            if targets is None:
                return logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,  # masked-objective targets mark unused positions -100
            )
            return logits, loss

        @torch.no_grad()
        def embed(self, idx):
            """Per-position hidden states (B, T, d_model) — used for note embeddings."""
            return self._backbone(idx)

        @torch.no_grad()
        def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size :]
                logits = self.forward(idx_cond)[:, -1, :] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
            return idx


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
    """Minimal next-token training loop. ``batches`` yields (x, y) int arrays/tensors."""
    _require_torch()
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses = []
    for step, (x, y) in enumerate(batches):
        x = torch.as_tensor(x, dtype=torch.long, device=device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            ppl = math.exp(min(float(loss), 20.0))
            print(f"step {step:5d} | loss {float(loss):.4f} | ppl {ppl:.2f}")
    return losses
