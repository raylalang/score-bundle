"""Pretraining utilities for the Phase-0 music LM (PyTorch).

Reusable building blocks for next-token pretraining of :class:`MusicGPT` on tokenized
MIDI streams: stream construction with on-disk caching, a cosine-with-warmup schedule,
held-out perplexity evaluation, an epoch loop with checkpointing, and per-epoch sampling.
``scripts/train_lm.py`` is a thin CLI over these.  Everything here is import-guarded on
torch (the Phase-1 numpy core never imports this module).

Notation: validation **perplexity** = exp(mean next-token cross-entropy) is the headline
metric; it must drop as the LM learns expressive piano structure (Phase-0 gate, design §5).
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .tokenizer import MidiTokenizer, NoteEvent
from . import data as _data


@dataclass
class TrainConfig:
    # data
    block_size: int = 512
    batch_size: int = 32
    max_notes: Optional[int] = None       # cap notes per piece (None = whole piece)
    train_limit: Optional[int] = None     # cap number of train pieces
    val_limit: Optional[int] = None
    # optimization
    epochs: int = 5
    steps_per_epoch: int = 500
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # eval / logging
    eval_batches: int = 50
    sample_notes: int = 40
    seed: int = 0
    # io
    out_dir: str = "checkpoints"
    cache_dir: Optional[str] = None


def _stream_cache_key(tok: MidiTokenizer, split: str, limit, max_notes) -> str:
    sig = (
        tok.grid, tok.max_shift_steps, tok.max_dur_steps, tok.pitch_min,
        tok.pitch_max, tok.n_vel_bins, split, limit, max_notes,
    )
    return hashlib.md5(repr(sig).encode()).hexdigest()[:16]


def build_stream(
    root: str,
    tok: MidiTokenizer,
    split: str,
    limit: Optional[int] = None,
    max_notes: Optional[int] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Tokenize a MAESTRO split into one packed int64 stream (cached to ``cache_dir``)."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _stream_cache_key(tok, split, limit, max_notes)
        cache_path = os.path.join(cache_dir, f"maestro_{split}_{key}.npy")
        if os.path.exists(cache_path):
            if verbose:
                print(f"[stream] load cached {cache_path}")
            return np.load(cache_path)

    seqs: List[List[int]] = []
    n = 0
    for notes in _data.iter_maestro_note_streams(
        root, split=split, limit=limit, max_notes=max_notes
    ):
        seqs.append(tok.encode(notes))
        n += 1
        if verbose and n % 50 == 0:
            print(f"[stream] {split}: tokenized {n} pieces")
    if not seqs:
        raise RuntimeError(f"no MAESTRO pieces found for split={split!r} under {root!r}")
    stream = _data.pack_tokens(seqs)
    if verbose:
        print(f"[stream] {split}: {n} pieces -> {len(stream):,} tokens")
    if cache_path:
        np.save(cache_path, stream)
    return stream


def cosine_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Linear warmup then cosine decay from ``lr`` to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    prog = min(max(prog, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * prog))


def evaluate(model, stream: np.ndarray, cfg: TrainConfig, n_batches: int, seed: int = 0) -> float:
    """Mean next-token cross-entropy over ``n_batches`` random windows (no grad)."""
    import torch

    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in _data.lm_batches(stream, cfg.block_size, cfg.batch_size, rng, n_batches):
            x = torch.as_tensor(x, dtype=torch.long, device=device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            _, loss = model(x, targets=y)
            losses.append(float(loss))
    if was_training:
        model.train()
    return float(np.mean(losses))


def sample_continuation(
    model, tok: MidiTokenizer, prompt_notes: List[NoteEvent], max_new_tokens: int = 40,
    top_k: int = 16, temperature: float = 1.0,
) -> List[NoteEvent]:
    """Sample a continuation from a note prompt and decode it back to notes."""
    import torch

    device = next(model.parameters()).device
    prompt = tok.encode(prompt_notes, add_bos_eos=False)
    # keep within block size, leaving room to generate
    prompt = prompt[-(model.cfg.block_size - max_new_tokens):] or prompt[:1]
    idx = torch.as_tensor(prompt, dtype=torch.long, device=device)[None]
    out = model.generate(idx, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)
    return tok.decode(out[0].tolist())


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_ppl: List[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_path: Optional[str] = None


def train(
    model,
    train_stream: np.ndarray,
    val_stream: Optional[np.ndarray],
    cfg: TrainConfig,
    tok: Optional[MidiTokenizer] = None,
    sample_prompt: Optional[List[NoteEvent]] = None,
    verbose: bool = True,
) -> History:
    """Pretrain ``model`` with next-token CE; per-epoch val ppl, checkpoint, and sample.

    Returns a :class:`History` of train/val loss and validation perplexity.  Checkpoints
    the best-val model to ``cfg.out_dir/best.pt``.
    """
    import torch

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm optional
        def tqdm(it, **kw):
            return it

    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    rng = np.random.default_rng(cfg.seed)
    hist = History()
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        batches = _data.lm_batches(train_stream, cfg.block_size, cfg.batch_size, rng, cfg.steps_per_epoch)
        run = 0.0
        bar = tqdm(batches, total=cfg.steps_per_epoch, desc=f"epoch {epoch+1}/{cfg.epochs}", disable=not verbose)
        for x, y in bar:
            lr = cosine_lr(step, cfg, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x = torch.as_tensor(x, dtype=torch.long, device=device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            _, loss = model(x, targets=y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            lv = loss.item()
            run += lv
            step += 1
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{lv:.3f}", lr=f"{lr:.1e}")
        train_loss = run / cfg.steps_per_epoch
        hist.train_loss.append(train_loss)

        if val_stream is not None:
            vloss = evaluate(model, val_stream, cfg, cfg.eval_batches, seed=cfg.seed + 1)
            vppl = math.exp(min(vloss, 20.0))
            hist.val_loss.append(vloss)
            hist.val_ppl.append(vppl)
            if verbose:
                print(f"epoch {epoch+1}: train_loss {train_loss:.4f} | val_loss {vloss:.4f} | val_ppl {vppl:.2f}")
            if vloss < hist.best_val and cfg.out_dir:
                hist.best_val = vloss
                hist.best_path = os.path.join(cfg.out_dir, "best.pt")
                torch.save({"model": model.state_dict(), "cfg": model.cfg, "epoch": epoch, "val_loss": vloss}, hist.best_path)
                if verbose:
                    print(f"  ↳ checkpoint {hist.best_path} (val_loss {vloss:.4f})")
        elif verbose:
            print(f"epoch {epoch+1}: train_loss {train_loss:.4f}")

        if tok is not None and sample_prompt is not None and verbose:
            notes = sample_continuation(model, tok, sample_prompt, max_new_tokens=cfg.sample_notes * 4)
            print(f"  ↳ sampled {len(notes)} notes (e.g. pitches {[n.pitch for n in notes[:8]]})")

    return hist
