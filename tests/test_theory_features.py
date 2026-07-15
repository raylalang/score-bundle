"""Tests for the music-theory feature block (`baselines._theory_block`).

Pins: shape/determinism/finiteness, strict score-only-ness (default off; the
25 base columns are byte-identical with and without the flag), and musical
sanity — the K-S key finder recovers C major on a C-major scale, in-scale
flags fire correctly, metrical weight ranks downbeats above off-beats, and an
exactly repeated motif gets a higher repetition count than fresh material.
"""
from __future__ import annotations

import numpy as np

from score_bundle.baselines import _theory_block, rich_score_features
from score_bundle.score import Score


def _random_score(rng: np.random.Generator, n: int = 60) -> Score:
    onset = np.cumsum(rng.choice([0.0, 0.25, 0.5, 1.0], size=n,
                                 p=[0.15, 0.35, 0.3, 0.2]))
    pitch = np.clip(60 + np.cumsum(rng.integers(-4, 5, size=n)), 30, 100)
    duration = rng.choice([0.25, 0.5, 1.0, 2.0], size=n)
    voice = rng.integers(0, 2, size=n)
    return Score.from_arrays(pitch, onset, duration, voice)


def test_shape_finite_deterministic_and_default_off():
    rng = np.random.default_rng(0)
    s = _random_score(rng)
    base = rich_score_features(s)
    X1 = rich_score_features(s, theory=True)
    X2 = rich_score_features(s, theory=True)
    assert X1.shape == (len(s), base.shape[1] + 14)
    assert np.isfinite(X1).all()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(X1[:, : base.shape[1]], base)  # base untouched


def test_key_and_scale_degree_on_c_major():
    # two octaves of the C-major scale, quarter notes, one voice
    degs = np.array([0, 2, 4, 5, 7, 9, 11])
    pitch = np.concatenate([60 + degs, 72 + degs])
    n = len(pitch)
    T = _theory_block(pitch.astype(float), np.arange(n, dtype=float),
                      np.ones(n), np.zeros(n))
    key_clarity, mode_major, in_scale = T[:, 0], T[:, 1], T[:, 4]
    assert (mode_major == 1.0).mean() > 0.8          # detected as major
    assert (in_scale == 1.0).all()                   # every note diatonic
    assert (key_clarity > 0.5).all()
    # a chromatic intruder is flagged out-of-scale
    pitch2 = np.concatenate([pitch, [61.0]])         # C# against C major
    T2 = _theory_block(pitch2, np.arange(n + 1, dtype=float),
                       np.ones(n + 1), np.zeros(n + 1))
    assert T2[-1, 4] == 0.0


def test_metrical_weight_ranks_downbeats():
    onsets = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    T = _theory_block(np.full(6, 60.0), onsets, np.ones(6), np.zeros(6))
    w = T[:, 6]
    assert w[0] == w[5] and w[0] > w[3] > w[2] > w[1]


def test_repetition_counts_repeated_motif():
    # motif (5 notes, distinctive contour) stated three times, then fresh notes
    motif = np.array([60, 64, 62, 67, 65], dtype=float)
    pitch = np.concatenate([motif, motif, motif,
                            np.array([50, 71, 53, 68, 49], dtype=float)])
    n = len(pitch)
    T = _theory_block(pitch, np.arange(n, dtype=float), np.ones(n), np.zeros(n))
    rep = T[:, 11]
    assert rep[:15].mean() > rep[15:].mean()
    assert rep[0] > 0.0                              # the motif is seen again


def test_verticals_dissonance_and_bass():
    # one consonant triad and one cluster, same piece
    pitch = np.array([60, 64, 67, 70, 71, 72], dtype=float)
    onset = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    T = _theory_block(pitch, onset, np.ones(6), np.zeros(6))
    dissonance, is_bass = T[:, 7], T[:, 8]
    assert dissonance[:3].mean() < dissonance[3:].mean()
    assert is_bass[0] == 1.0 and is_bass[3] == 1.0 and is_bass.sum() == 2.0


def test_degenerate_inputs():
    # single note, and a single chord with identical onsets
    T1 = _theory_block(np.array([60.0]), np.array([0.0]),
                       np.array([1.0]), np.array([0.0]))
    assert T1.shape == (1, 14) and np.isfinite(T1).all()
    T2 = _theory_block(np.array([60.0, 64.0, 67.0]), np.zeros(3),
                       np.ones(3), np.zeros(3))
    assert T2.shape == (3, 14) and np.isfinite(T2).all()
