"""Bridge Phase 0 -> Phase 1: per-note LM embeddings as a learned prior mean.

After pretraining, the hidden state at each note's VELOCITY token is a per-note
embedding ``h_i`` aligned with the score graph's nodes.  A small ridge head maps
``h_i -> mu_i`` (a prediction of the expressive variables); the Phase-1 graph prior
then models the residual ``y - mu`` with calibrated structured uncertainty:

    y = mu_LM(h) + xi,     xi ~ N(0, Q_G^{-1}).

This turns the Phase-1 comparison into the stronger question: does a score-graph
residual prior on top of a learned LM mean beat the LM mean (and the hand-built
baselines) on recovery *and* calibration?
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import model_numpy as mnp
from .tokenizer import MidiTokenizer, BOS


def note_velocity_positions(tokenizer: MidiTokenizer, tokens: List[int]) -> List[int]:
    """Indices of each note's VELOCITY token (the per-note read-out position)."""
    positions = []
    for i, tok in enumerate(tokens):
        name, _ = tokenizer.token_type(tok)
        if name == "velocity":
            positions.append(i)
    return positions


def note_embeddings(
    params: Dict[str, np.ndarray],
    cfg: "mnp.GPTConfig",
    tokenizer: MidiTokenizer,
    tokens: List[int],
) -> np.ndarray:
    """Run the NumPy model and return per-note embeddings, shape (n_notes, d_model)."""
    _, hidden = mnp.forward(params, np.asarray(tokens, dtype=int), cfg)
    pos = note_velocity_positions(tokenizer, tokens)
    return hidden[pos]


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
