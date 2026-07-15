"""Tests for the specified Phase-2 vibrato estimator (`fit_vibrato_note`).

Pins: recovery of known (c, gamma, f) from noisy synthetic curves; the
centre-vs-mean distinction on fractional cycles (the thesis's explicit point);
the identifiability rule for short notes; and rough calibration of the
reported variances (z-scores of repeated estimates approximately unit-scale).
"""
from __future__ import annotations

import numpy as np

from score_bundle.phase2.intonation import fit_vibrato_note


def _curve(rng, c=12.0, gamma=28.0, f=5.5, dur=1.2, sr=100.0, noise=5.0,
           delta=0.03):
    t = np.arange(0.0, dur, 1.0 / sr)
    x = c + gamma * np.sin(2 * np.pi * f * (t - delta)) + rng.normal(0, noise, t.size)
    return t, x


def test_recovers_known_parameters():
    rng = np.random.default_rng(0)
    t, x = _curve(rng)
    out = fit_vibrato_note(t, x)
    assert out["vibrato_identifiable"]
    assert abs(out["c"] - 12.0) < 2.0
    assert abs(out["gamma"] - 28.0) < 3.0
    assert abs(out["f"] - 5.5) < 0.15


def test_centre_is_not_the_mean_on_fractional_cycles():
    rng = np.random.default_rng(1)
    # 1.75 cycles: the sample mean is biased away from the true centre
    t, x = _curve(rng, c=0.0, gamma=30.0, f=5.0, dur=0.35, noise=1.0, delta=0.0)
    out = fit_vibrato_note(t, x, min_cycles=1.5)
    assert abs(x.mean()) > 3.0            # the naive mean is visibly off
    assert abs(out["c"]) < 2.0            # the joint fit recovers the centre


def test_short_note_flagged_unidentifiable():
    rng = np.random.default_rng(2)
    t, x = _curve(rng, dur=0.15)          # < 1 cycle at any grid rate
    out = fit_vibrato_note(t, x)
    assert not out["vibrato_identifiable"]
    assert np.isfinite(out["c"]) and np.isfinite(out["var_c"])


def test_variances_roughly_calibrated():
    rng = np.random.default_rng(3)
    zs_c, zs_g = [], []
    for _ in range(60):
        t, x = _curve(rng, dur=1.5, noise=6.0)
        out = fit_vibrato_note(t, x)
        if not out["vibrato_identifiable"]:
            continue
        zs_c.append((out["c"] - 12.0) / np.sqrt(out["var_c"]))
        zs_g.append((out["gamma"] - 28.0) / np.sqrt(out["var_gamma"]))
    zc, zg = np.array(zs_c), np.array(zs_g)
    # unit-ish scale, not wildly over/under-confident
    assert 0.5 < zc.std() < 2.0
    assert 0.5 < zg.std() < 2.0
    assert abs(zc.mean()) < 0.6 and abs(zg.mean()) < 0.6
