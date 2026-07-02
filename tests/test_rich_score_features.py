"""Tests for the rich score-only feature baseline (`baselines.rich_score_features`).

The point of this representation is to be the honest rival to the LM embedding under
the identical cross-piece head protocol, so the tests pin: shape/determinism, strict
score-only-ness (permutation equivariance, no dependence on anything but the support),
and that a cross-piece head on these features actually learns a score-driven signal.
"""
from __future__ import annotations

import numpy as np

from score_bundle.baselines import rich_score_features
from score_bundle.lm.features import apply_prior_mean, fit_prior_mean_head
from score_bundle.score import Score


def _random_score(rng: np.random.Generator, n: int = 60) -> Score:
    onset = np.cumsum(rng.choice([0.0, 0.25, 0.5, 1.0], size=n, p=[0.15, 0.35, 0.3, 0.2]))
    pitch = np.clip(60 + np.cumsum(rng.integers(-4, 5, size=n)), 30, 100)
    duration = rng.choice([0.25, 0.5, 1.0, 2.0], size=n)
    voice = rng.integers(0, 2, size=n)
    return Score.from_arrays(pitch, onset, duration, voice)


def test_shape_finite_deterministic():
    rng = np.random.default_rng(0)
    s = _random_score(rng)
    X1 = rich_score_features(s)
    X2 = rich_score_features(s)
    assert X1.shape[0] == len(s) and X1.shape[1] >= 20
    assert np.isfinite(X1).all()
    np.testing.assert_array_equal(X1, X2)
    Z = rich_score_features(s, rff_dim=32)
    assert Z.shape == (len(s), X1.shape[1] + 32)
    assert np.isfinite(Z).all()


def test_permutation_equivariant():
    """Features belong to notes, not array slots — score-only and order-robust."""
    rng = np.random.default_rng(1)
    s = _random_score(rng)
    # perturb onsets so every note is unique (no chord-rank ties under permutation)
    onset = s.onset + rng.uniform(0, 1e-4, size=len(s))
    pitch, dur, voice = s.pitch, s.duration, s.voice
    s1 = Score.from_arrays(pitch, onset, dur, voice)
    perm = rng.permutation(len(s1))
    s2 = Score.from_arrays(pitch[perm], onset[perm], dur[perm], voice[perm])
    X1 = rich_score_features(s1)
    X2 = rich_score_features(s2)
    np.testing.assert_allclose(X2, X1[perm], atol=1e-10)


def test_tiny_scores_do_not_crash():
    for n in (1, 2, 3):
        s = Score.from_arrays(np.full(n, 60.0), np.arange(n, dtype=float),
                              np.ones(n), np.zeros(n, dtype=int))
        X = rich_score_features(s, rff_dim=8)
        assert X.shape[0] == n and np.isfinite(X).all()


def test_cross_piece_head_learns_metrical_signal():
    """A head fit on head pieces must transfer a score-driven target to new pieces."""
    rng = np.random.default_rng(2)

    def target(s: Score) -> np.ndarray:
        # downbeat accent + pitch tilt: a caricature of expressive dynamics
        return 0.5 * np.cos(2 * np.pi * np.mod(s.onset, 1.0)) + 0.02 * (s.pitch - 60)

    head = [_random_score(rng) for _ in range(8)]
    test = [_random_score(rng) for _ in range(4)]
    H = np.concatenate([rich_score_features(s) for s in head])
    Yh = np.concatenate([target(s) for s in head])[:, None]
    W = fit_prior_mean_head(H, Yh, l2=1.0)
    errs, base = [], []
    for s in test:
        mu = apply_prior_mean(rich_score_features(s), W)
        yt = target(s)
        errs.append(np.sqrt(np.mean((mu - yt) ** 2)))
        base.append(np.sqrt(np.mean(yt ** 2)))
    assert np.mean(errs) < 0.35 * np.mean(base)
