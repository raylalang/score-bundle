"""Stage 2 — masked, score-conditioned pretraining aligned with the Phase-1 read-out.

Why this exists (the restart branch's declared next step): the Stage-1 leak-free fix
reads a *causal next-token* model at the pre-velocity token — a state that was trained
to predict the note's velocity, but only from leftward context, and never explicitly
trained to expose timing/articulation.  Stage 2 replaces the objective with the task we
actually pose at inference time:

    given the score-like tokens of ALL notes (both directions) and the *observed*
    notes' velocities, predict each hidden note's velocity.

Concretely: same tokenizer, same 4-token note groups ``[TIME_SHIFT, PITCH, DURATION,
VELOCITY]``; a **bidirectional** transformer (``GPTConfig(causal=False)``); one extra
``[MASK]`` embedding id appended after the tokenizer vocabulary (the tokenizer itself
is untouched); per training window an observed fraction ``rho ~ U(frac_range)`` is
drawn and each velocity token is independently replaced by ``[MASK]`` with probability
``1 - rho``; cross-entropy on the original velocity bins at masked positions only.
This is amortized conditional inference p(v_hidden | score, v_observed) at every
masking rate — exactly the Phase-1 imputation family, and mask-aware by construction
(a masked note's state *cannot* see its own velocity: the token is not in the input).

The Phase-1 read-out has two variants: :func:`masked_note_embeddings_long` (read each
note at its velocity position under the eval mask — leak-free for *hidden* notes only,
since an observed note's state sits at its real velocity token) and
:func:`masked_note_embeddings_loo` (leave-one-out: every note read at a ``[MASK]``\\ ed
own-velocity position — leak-free for all notes, the variant the ridge head and the EB
noise fit require).  Import-guarded on torch like the rest of the LM package.
"""
from __future__ import annotations

import math
import os
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .tokenizer import MidiTokenizer, NoteEvent
from . import data as _data
from .features import note_velocity_positions
from .train import TrainConfig, History, cosine_lr

IGNORE_INDEX = -100


def mask_token_id(tokenizer: MidiTokenizer) -> int:
    """The ``[MASK]`` id — one past the tokenizer vocabulary (tokenizer untouched)."""
    return tokenizer.vocab_size


def masked_vocab_size(tokenizer: MidiTokenizer) -> int:
    """Model vocab = tokenizer vocab + the ``[MASK]`` embedding."""
    return tokenizer.vocab_size + 1


def velocity_token_mask(tokenizer: MidiTokenizer, x: np.ndarray) -> np.ndarray:
    """Boolean array marking VELOCITY tokens in ``x`` (identified by id range)."""
    x = np.asarray(x)
    return (x >= tokenizer.vel_base) & (x < tokenizer.vel_base + tokenizer.n_vel_bins)


def mask_velocities(
    x: np.ndarray,
    tokenizer: MidiTokenizer,
    rng: np.random.Generator,
    observed_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace velocity tokens by ``[MASK]`` w.p. ``1 - observed_frac``; build targets.

    ``x`` is (T,) or (B, T) int tokens.  Returns ``(x_masked, targets)`` where targets
    equal the original token at masked velocity positions and ``IGNORE_INDEX``
    everywhere else (the loss is computed only where the model must infer).
    """
    x = np.asarray(x, dtype=np.int64)
    is_vel = velocity_token_mask(tokenizer, x)
    hide = is_vel & (rng.random(x.shape) >= observed_frac)
    # never a completely un-supervised window: hide at least one velocity if any exist
    if not hide.any() and is_vel.any():
        flat = np.flatnonzero(is_vel.reshape(-1))
        pick = flat[rng.integers(len(flat))]
        hide.reshape(-1)[pick] = True
    targets = np.where(hide, x, IGNORE_INDEX)
    x_masked = np.where(hide, mask_token_id(tokenizer), x)
    return x_masked, targets


def masked_batches(
    stream: np.ndarray,
    tokenizer: MidiTokenizer,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
    n_batches: int,
    frac_range: Tuple[float, float] = (0.1, 0.9),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(x_masked, targets)`` windows; one observed-fraction draw per row.

    Sampling the rate row-wise trains a single amortized model across the whole
    Phase-1 masking family instead of one fixed rate.
    """
    hi = len(stream) - block_size
    if hi <= 0:
        raise ValueError("stream shorter than block_size")
    lo_f, hi_f = frac_range
    for _ in range(n_batches):
        ix = rng.integers(0, hi, size=batch_size)
        xs, ts = [], []
        for i in ix:
            frac = float(rng.uniform(lo_f, hi_f))
            xm, tg = mask_velocities(stream[i : i + block_size], tokenizer, rng, frac)
            xs.append(xm)
            ts.append(tg)
        yield np.stack(xs), np.stack(ts)


def evaluate_masked(
    model,
    stream: np.ndarray,
    tokenizer: MidiTokenizer,
    cfg: TrainConfig,
    n_batches: int,
    observed_frac: float = 0.6,
    seed: int = 0,
) -> float:
    """Mean masked-velocity cross-entropy at a fixed rate (the Phase-1 setting)."""
    import torch

    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for x, t in masked_batches(stream, tokenizer, cfg.block_size, cfg.batch_size,
                                   rng, n_batches, frac_range=(observed_frac, observed_frac)):
            x = torch.as_tensor(x, dtype=torch.long, device=device)
            t = torch.as_tensor(t, dtype=torch.long, device=device)
            _, loss = model(x, targets=t)
            losses.append(float(loss))
    if was_training:
        model.train()
    return float(np.mean(losses))


def train_masked(
    model,
    train_stream: np.ndarray,
    val_stream: Optional[np.ndarray],
    cfg: TrainConfig,
    tokenizer: MidiTokenizer,
    frac_range: Tuple[float, float] = (0.1, 0.9),
    verbose: bool = True,
) -> History:
    """Masked-velocity pretraining loop; mirrors :func:`score_bundle.lm.train.train`.

    Val metric / checkpoint criterion: masked CE at the fixed 60%-observed rate (the
    published Phase-1 protocol), so "best" means best at the deployment condition.
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
        batches = masked_batches(train_stream, tokenizer, cfg.block_size,
                                 cfg.batch_size, rng, cfg.steps_per_epoch, frac_range)
        run = 0.0
        bar = tqdm(batches, total=cfg.steps_per_epoch,
                   desc=f"epoch {epoch+1}/{cfg.epochs}", disable=not verbose)
        for x, t in bar:
            lr = cosine_lr(step, cfg, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x = torch.as_tensor(x, dtype=torch.long, device=device)
            t = torch.as_tensor(t, dtype=torch.long, device=device)
            _, loss = model(x, targets=t)
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
            vloss = evaluate_masked(model, val_stream, tokenizer, cfg,
                                    cfg.eval_batches, seed=cfg.seed + 1)
            vppl = math.exp(min(vloss, 20.0))
            hist.val_loss.append(vloss)
            hist.val_ppl.append(vppl)  # exp(masked CE): velocity-bin perplexity
            if verbose:
                print(f"epoch {epoch+1}: train_mce {train_loss:.4f} | "
                      f"val_mce@0.6 {vloss:.4f} | vel-bin ppl {vppl:.2f}", flush=True)
            if vloss < hist.best_val and cfg.out_dir:
                hist.best_val = vloss
                hist.best_path = os.path.join(cfg.out_dir, "best.pt")
                torch.save({"model": model.state_dict(), "cfg": model.cfg,
                            "epoch": epoch, "val_loss": vloss,
                            "objective": "masked_velocity",
                            "mask_id": mask_token_id(tokenizer)}, hist.best_path)
                if verbose:
                    print(f"  -> checkpoint {hist.best_path} (val_mce {vloss:.4f})", flush=True)
        elif verbose:
            print(f"epoch {epoch+1}: train_mce {train_loss:.4f}", flush=True)
    return hist


# --------------------------------------------------------------- Phase-1 read-out
def masked_note_embeddings_long(
    model,
    tokenizer: MidiTokenizer,
    notes: Sequence[NoteEvent],
    observed: np.ndarray,
    return_vel_logits: bool = False,
):
    """Per-note embeddings under the Stage-2 read-out (mask-aware by construction).

    Encodes ``notes`` without BOS/EOS (fixed 4-token stride), replaces the velocity
    token of every note with ``observed[i] == False`` by ``[MASK]``, runs the
    bidirectional model on block-sized windows and reads each note's hidden state at
    its velocity position.  A hidden note's state cannot see its own velocity (the
    token is absent from the input) — but an OBSERVED note's state is read at its real
    velocity token and does contain its own velocity; for anything fit on observed
    rows (ridge head, EB noise) use :func:`masked_note_embeddings_loo` instead.

    Returns ``(n_notes, d_model)``; with ``return_vel_logits=True`` also returns the
    ``(n_notes, n_vel_bins)`` velocity-bin logits at each note's position (the model's
    own direct prediction, used by :func:`direct_velocity_mean`).
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    observed = np.asarray(observed, dtype=bool)
    if len(observed) != len(notes):
        raise ValueError("observed mask must have one entry per note")
    block = model.cfg.block_size
    win = max(4, (block // 4) * 4)  # whole notes per window
    toks = np.asarray(tokenizer.encode(list(notes), add_bos_eos=False), dtype=np.int64)
    vel_pos = np.asarray(note_velocity_positions(tokenizer, toks.tolist()), dtype=int)
    if len(vel_pos) != len(notes):
        raise ValueError("token stream does not have one velocity token per note")
    x = toks.copy()
    x[vel_pos[~observed]] = mask_token_id(tokenizer)

    embs, logits_all = [], []
    with torch.no_grad():
        for s in range(0, len(x), win):
            chunk = x[s : s + win]
            idx = torch.as_tensor(chunk, dtype=torch.long, device=device)[None]
            hidden = model.embed(idx)[0]
            in_win = (vel_pos >= s) & (vel_pos < s + len(chunk))
            pos = vel_pos[in_win] - s
            embs.append(hidden[pos].detach().cpu().numpy())
            if return_vel_logits:
                logits = model.head(hidden[pos])
                vl = logits[:, tokenizer.vel_base : tokenizer.vel_base + tokenizer.n_vel_bins]
                logits_all.append(vl.detach().cpu().numpy())
    H = (np.concatenate(embs, axis=0) if embs
         else np.zeros((0, model.cfg.d_model)))
    if not return_vel_logits:
        return H
    VL = (np.concatenate(logits_all, axis=0) if logits_all
          else np.zeros((0, tokenizer.n_vel_bins)))
    return H, VL


def masked_note_embeddings_loo(
    model,
    tokenizer: MidiTokenizer,
    notes: Sequence[NoteEvent],
    observed: np.ndarray,
    return_vel_logits: bool = False,
    batch_size: int = 64,
):
    """Leave-one-out Stage-2 read-out: EVERY note is read at a ``[MASK]`` token.

    :func:`masked_note_embeddings_long` is leak-free only for *hidden* notes — an
    observed note's state is read at its real velocity token, which (bidirectionally)
    contains the note's own velocity.  A ridge head fit on such rows partly learns to
    decode the target, and the EB noise fit (which uses observed-note residuals) then
    collapses — the classic leak signature, resurfacing inside Stage 2.

    Here the base input carries observed notes' velocities and ``[MASK]`` at hidden
    notes (the strict eval condition), and each note is read from a variant of its
    window in which additionally its OWN velocity token is ``[MASK]``\\ ed (a no-op for
    hidden notes).  Conditioning = all *other* observed notes in the window; no note's
    embedding can contain its own performed velocity.  This matches the training
    distribution (the model always predicts at a ``[MASK]`` given the others) and
    restores, for observed notes, exactly the guarantee the Stage-1 pre-velocity
    read-out gives by construction.  One forward per ``batch_size`` variants.

    Returns ``(n_notes, d_model)``; with ``return_vel_logits=True`` also the
    ``(n_notes, n_vel_bins)`` velocity-bin logits at each note's masked position.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    observed = np.asarray(observed, dtype=bool)
    if len(observed) != len(notes):
        raise ValueError("observed mask must have one entry per note")
    block = model.cfg.block_size
    win = max(4, (block // 4) * 4)  # whole notes per window
    toks = np.asarray(tokenizer.encode(list(notes), add_bos_eos=False), dtype=np.int64)
    vel_pos = np.asarray(note_velocity_positions(tokenizer, toks.tolist()), dtype=int)
    if len(vel_pos) != len(notes):
        raise ValueError("token stream does not have one velocity token per note")
    mask_id = mask_token_id(tokenizer)
    x = toks.copy()
    x[vel_pos[~observed]] = mask_id

    H = np.zeros((len(notes), model.cfg.d_model))
    VL = np.zeros((len(notes), tokenizer.n_vel_bins)) if return_vel_logits else None
    with torch.no_grad():
        for s in range(0, len(x), win):
            chunk = x[s : s + win]
            in_win = np.flatnonzero((vel_pos >= s) & (vel_pos < s + len(chunk)))
            if in_win.size == 0:
                continue
            pos = vel_pos[in_win] - s
            variants = np.repeat(chunk[None, :], len(in_win), axis=0)
            variants[np.arange(len(in_win)), pos] = mask_id  # own velocity out
            for b in range(0, len(in_win), batch_size):
                rows = np.arange(b, min(b + batch_size, len(in_win)))
                idx = torch.as_tensor(variants[rows], dtype=torch.long, device=device)
                hidden = model.embed(idx)
                sel = hidden[torch.arange(len(rows), device=device),
                             torch.as_tensor(pos[rows], device=device)]
                H[in_win[rows]] = sel.detach().cpu().numpy()
                if return_vel_logits:
                    logits = model.head(sel)
                    vl = logits[:, tokenizer.vel_base : tokenizer.vel_base + tokenizer.n_vel_bins]
                    VL[in_win[rows]] = vl.detach().cpu().numpy()
    if not return_vel_logits:
        return H
    return H, VL


def direct_velocity_mean(
    tokenizer: MidiTokenizer,
    vel_logits: np.ndarray,
    velocities: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """The model's own velocity prediction mapped to Phase-1 ``v`` units.

    ``v`` is per-piece-centered normalized velocity, so the centering offset is
    estimated from the *observed* notes only (all a real system would have).  EVERY
    note — observed included — gets the softmax expectation over bin centers, never
    its true value: this is a *prior mean*, and substituting the observation at
    observed notes makes their residuals exactly zero, which collapses the EB noise
    fit (``noise_floor_frac`` scales the floor by the observed residual variance —
    5% of zero is zero).  Use with logits from a leave-one-out read-out
    (:func:`masked_note_embeddings_loo`) so no note's logits saw its own velocity.
    """
    velocities = np.asarray(velocities, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    z = vel_logits - vel_logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    centers = (np.arange(tokenizer.n_vel_bins) + 0.5) * 128.0 / tokenizer.n_vel_bins
    exp_vel = p @ centers
    offset = float(np.mean(velocities[observed] / 127.0)) if observed.any() else 0.5
    return exp_vel / 127.0 - offset
