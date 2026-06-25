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
    """Indices of each note's VELOCITY token (the per-note read-out position)."""
    return [i for i, tok in enumerate(tokens) if tokenizer.token_type(tok)[0] == "velocity"]


def note_embeddings(model, tokenizer: MidiTokenizer, tokens: List[int]) -> np.ndarray:
    """Run the (PyTorch) model and return per-note embeddings, shape (n_notes, d_model).

    ``model`` is a ``score_bundle.lm.model_torch.MusicGPT``.
    """
    import torch  # local import keeps this module importable without torch

    model.eval()
    device = next(model.parameters()).device
    idx = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]  # (1, T)
    hidden = model.embed(idx)[0]                                          # (T, d)
    pos = note_velocity_positions(tokenizer, tokens)
    return hidden[pos].detach().cpu().numpy()


def fit_prior_mean(
    embeddings: np.ndarray, y_targets: np.ndarray, l2: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge map h_i -> mu_i. Returns (mu predictions, weight matrix).

    ``y_targets`` is (n_notes,) or (n_notes, k).  Demonstrates the learned prior mean;
    in practice the head is trained jointly / on a held-out split.
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
