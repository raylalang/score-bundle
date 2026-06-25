"""Synthetic data for the recovery / imputation experiments.

Generates a random score, draws a performance field from a graph prior with known
hyperparameters, and adds observation noise.  Because the ground truth is known,
this is the cleanest test of whether the posterior recovers the latents and whether
its credible intervals are calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import build_adjacency, laplacian
from .model import GraphGaussianField
from .prior import laplacian_precision
from .score import Score


@dataclass
class SyntheticDataset:
    score: Score
    L: np.ndarray
    field: GraphGaussianField
    y_true: np.ndarray
    y_obs: np.ndarray
    noise_var: float


def random_score(n: int, rng: np.random.Generator, n_voices: int = 1) -> Score:
    """A random monophonic-ish score: pitch random walk, monotone onsets."""
    steps = rng.integers(-3, 4, size=n)
    pitch = 60 + np.cumsum(steps)
    pitch = np.clip(pitch, 36, 96)
    durations = rng.choice([0.5, 1.0, 1.0, 2.0], size=n)
    onset = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    voice = rng.integers(0, n_voices, size=n)
    return Score.from_arrays(pitch, onset, durations, voice)


def make_synthetic(
    rng: np.random.Generator,
    n: int = 60,
    lam: float = 0.5,
    eta: float = 3.0,
    noise_var: float = 0.05,
    ell_b: float = 2.0,
    ell_p: float = 4.0,
) -> SyntheticDataset:
    """Sample a single-channel performance field from a known graph prior."""
    score = random_score(n, rng)
    W = build_adjacency(score, ell_b=ell_b, ell_p=ell_p)
    L = laplacian(W)
    Q = laplacian_precision(L, lam=lam, eta=eta)
    field = GraphGaussianField(Q)
    y_true = field.sample(rng)
    y_obs = y_true + rng.normal(scale=np.sqrt(noise_var), size=n)
    return SyntheticDataset(score, L, field, y_true, y_obs, noise_var)


def random_mask(n: int, rng: np.random.Generator, observed_frac: float = 0.7) -> np.ndarray:
    """Boolean mask with ``observed_frac`` of nodes observed (rest held out)."""
    mask = rng.random(n) < observed_frac
    if mask.all():          # ensure at least one held-out node
        mask[rng.integers(n)] = False
    if not mask.any():      # ensure at least one observed node
        mask[rng.integers(n)] = True
    return mask
