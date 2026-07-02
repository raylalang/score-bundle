"""Bridge Phase 0 -> Phase 1: per-note LM embeddings as a learned prior mean.

The hidden state at each note's VELOCITY token is a per-note embedding ``h_i`` aligned
with the score graph's nodes.  A small ridge head maps ``h_i -> mu_i`` (a prediction of
the expressive variables); the Phase-1 graph prior then models the residual ``y - mu``
with calibrated structured uncertainty:

    y = mu_LM(h) + xi,     xi ~ N(0, Q_G^{-1}).

This turns the Phase-1 comparison into the stronger question: does a score-graph residual
prior on top of a learned LM mean beat the LM mean (and the hand-built baselines) on
recovery *and* calibration?

``note_embeddings`` needs the PyTorch model; ``fit_prior_mean`` is pure NumPy.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .tokenizer import MidiTokenizer


def note_velocity_positions(tokenizer: MidiTokenizer, tokens: List[int]) -> List[int]:
    """Indices of each note's VELOCITY token (the *post*-velocity read-out position).

    This is the historical read-out. In a causal model the hidden state here has already
    attended to the note's own performed velocity, so an embedding read here **leaks** the
    velocity target into a "score-conditioned" prior mean (see the 2026-07-02 correction in
    ``docs/phase1_calibration_results.md``). Correct for *reconstruction* / z-init (it does
    encode the note's realized features); wrong for the *predictive* prior mean ``mu_LM``.
    Use :func:`note_score_positions` for the leak-free predictive read-out.
    """
    return [i for i, tok in enumerate(tokens) if tokenizer.token_type(tok)[0] == "velocity"]


def note_score_positions(tokenizer: MidiTokenizer, tokens: List[int]) -> List[int]:
    """Indices of each note's DURATION token — the leak-free, pre-velocity read-out.

    In the fixed 4-token group ``[TIME_SHIFT, PITCH, DURATION, VELOCITY]`` the DURATION token
    sits immediately before VELOCITY, so its causal hidden state is **structurally blind** to
    the note's own performed velocity, yet under next-token pretraining it is exactly the
    state trained to *predict* that velocity from score + left context. Reading here removes
    the velocity leak with no retraining. (TIME_SHIFT/DURATION carry *score* onset/duration in
    the Phase-1 bridge, not the tau/log r targets, so only velocity ever leaked.)

    Robust to the ``add_bos_eos=False`` stride used by :func:`note_embeddings_long`: each
    velocity position minus one is that note's duration token.
    """
    return [i - 1 for i in note_velocity_positions(tokenizer, tokens)]


_READOUTS = {
    "velocity": note_velocity_positions,   # post-velocity (leaks v; reconstruction/z-init)
    "pre_velocity": note_score_positions,  # leak-free predictive read-out
    "score": note_score_positions,         # alias
    "duration": note_score_positions,      # alias
}


def _readout_positions(readout: str, tokenizer: MidiTokenizer, tokens: List[int]) -> List[int]:
    try:
        fn = _READOUTS[readout]
    except KeyError:
        raise ValueError(
            f"unknown readout {readout!r}; expected one of {sorted(_READOUTS)}"
        )
    return fn(tokenizer, tokens)


def note_embeddings(
    model, tokenizer: MidiTokenizer, tokens: List[int], readout: str = "velocity"
) -> np.ndarray:
    """Run the (PyTorch) model and return per-note embeddings, shape (n_notes, d_model).

    ``model`` is a ``score_bundle.lm.model_torch.MusicGPT``.  ``tokens`` must fit in the
    model's ``block_size``; for long pieces use :func:`note_embeddings_long`.  ``readout``
    selects the per-note position: ``"velocity"`` (default, historical; leaks velocity) or
    ``"pre_velocity"``/``"score"``/``"duration"`` (leak-free predictive read-out).
    """
    import torch  # local import keeps this module importable without torch

    model.eval()
    device = next(model.parameters()).device
    idx = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]  # (1, T)
    hidden = model.embed(idx)[0]                                          # (T, d)
    pos = _readout_positions(readout, tokenizer, tokens)
    return hidden[pos].detach().cpu().numpy()


def note_embeddings_long(
    model, tokenizer: MidiTokenizer, notes, readout: str = "velocity"
) -> np.ndarray:
    """Per-note embeddings for a whole piece, windowed to the model's ``block_size``.

    Encodes ``notes`` (a sequence of :class:`~score_bundle.lm.tokenizer.NoteEvent`) without
    BOS/EOS so the stride is exactly 4 tokens/note, then runs the model on consecutive
    block-sized windows (split at note boundaries) and concatenates the per-note read-outs.
    Returns shape ``(len(notes), d_model)``.  Windows start fresh, so notes near a window's
    start see less left context — acceptable for a per-note feature.  ``readout`` selects the
    per-note position (see :func:`note_embeddings`); default ``"velocity"`` preserves the
    historical behaviour, ``"pre_velocity"`` is the leak-free read-out.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    win = max(4, (block // 4) * 4)  # whole notes per window
    toks = tokenizer.encode(notes, add_bos_eos=False)
    embs = []
    with torch.no_grad():
        for s in range(0, len(toks), win):
            chunk = toks[s : s + win]
            idx = torch.as_tensor(chunk, dtype=torch.long, device=device)[None]
            hidden = model.embed(idx)[0]
            pos = _readout_positions(readout, tokenizer, chunk)
            embs.append(hidden[pos].detach().cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, model.cfg.d_model))


def fit_prior_mean(
    embeddings: np.ndarray, y_targets: np.ndarray, l2: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge map h_i -> mu_i. Returns (mu predictions, weight matrix).

    ``y_targets`` is (n_notes,) or (n_notes, k).  Demonstrates the learned prior mean;
    in practice the head is trained on a held-out split (see :func:`fit_prior_mean_head`
    / :func:`apply_prior_mean`).
    """
    H = np.asarray(embeddings, dtype=float)
    Y = np.asarray(y_targets, dtype=float)
    single = Y.ndim == 1
    if single:
        Y = Y[:, None]
    Hb = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)  # bias column
    A = Hb.T @ Hb + l2 * np.eye(Hb.shape[1])
    W = np.linalg.solve(A, Hb.T @ Y)
    mu = Hb @ W
    return (mu[:, 0] if single else mu), W


def fit_prior_mean_head(
    embeddings: np.ndarray, y_targets: np.ndarray, l2: float = 1.0
) -> np.ndarray:
    """Fit only the ridge head h_i -> mu_i and return its weight matrix ``W``.

    Use on a *train* split, then :func:`apply_prior_mean` on held-out embeddings to get an
    out-of-sample learned prior mean mu_LM (the honest setting for the Phase-1 comparison).
    """
    _, W = fit_prior_mean(embeddings, y_targets, l2=l2)
    return W


def apply_prior_mean(embeddings: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply a fitted head ``W`` (from :func:`fit_prior_mean_head`) to new embeddings.

    Returns mu of shape (n_notes,) if ``W`` has one column else (n_notes, k).
    """
    H = np.asarray(embeddings, dtype=float)
    Hb = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
    mu = Hb @ np.asarray(W, dtype=float)
    return mu[:, 0] if mu.shape[1] == 1 else mu
