"""Tests for the onset-anchored tau channel (phase2.warp).

Pins: the LOO property of eq:localwarp (a note never enters its own tempo
line — tau is the true residual, not trivially zero); recovery of injected
timing deviations under a drifting tempo; matching (exact and DTW with an
insertion); and the predictive-variance noise row scaling with the local
scatter.
"""
from __future__ import annotations

import numpy as np

from score_bundle.phase2.warp import (local_loo_warp,
                                      match_score_to_performance, note_tau)


def _track(rng, n=60, drift=0.02, noise=0.02):
    b = np.arange(n) * 0.5
    tempo = 1.0 + drift * np.sin(np.linspace(0, 3, n))
    t = np.cumsum(0.5 * tempo)
    dev = rng.normal(0, noise, n)
    return b, t + dev, dev


def test_loo_recovers_injected_deviation():
    rng = np.random.default_rng(0)
    b, t, dev = _track(rng, noise=0.0)
    t2 = t.copy()
    t2[30] += 0.08                      # one late note
    pred, _ = local_loo_warp(b, t2)
    tau = t2 - pred
    assert abs(tau[30] - 0.08) < 0.02   # deviation survives LOO
    assert np.nanmedian(np.abs(tau[np.arange(len(tau)) != 30])) < 0.02


def test_loo_not_trivially_zero():
    rng = np.random.default_rng(1)
    b, t, dev = _track(rng, noise=0.03)
    pred, _ = local_loo_warp(b, t)
    tau = t - pred
    # correlated with the injected deviations, but not identical to zero
    keep = np.isfinite(tau)
    assert np.corrcoef(tau[keep], dev[keep])[0, 1] > 0.7
    assert np.std(tau[keep]) > 0.01


def test_predictive_variance_scales_with_scatter():
    rng = np.random.default_rng(2)
    b, t_lo, _ = _track(rng, noise=0.005)
    b2, t_hi, _ = _track(rng, noise=0.05)
    _, v_lo = local_loo_warp(b, t_lo)
    _, v_hi = local_loo_warp(b2, t_hi)
    assert np.nanmedian(v_hi) > 5 * np.nanmedian(v_lo)


def test_exact_match_and_dtw_with_insertion():
    sp = np.array([60, 62, 64, 65, 67, 69, 71, 72] * 3)
    si, pi, method = match_score_to_performance(sp, sp)
    assert method == "exact" and si.size == sp.size
    # performance has one inserted (extra) note
    pp = np.insert(sp, 5, 99)
    si, pi, method = match_score_to_performance(sp, pp)
    assert method == "dtw"
    assert si.size >= 0.9 * sp.size
    assert np.all(sp[si] == pp[pi])


def test_note_tau_shapes_and_missing():
    rng = np.random.default_rng(3)
    b, t, _ = _track(rng, n=40)
    sp = np.tile([60, 64, 67, 72], 10)
    out = note_tau(b, sp, t, sp)
    assert out["method"] == "exact"
    assert out["tau"].shape == (40,)
    inner = out["tau"][2:-2]
    assert np.isfinite(inner).all()
    assert np.isfinite(out["var"][2:-2]).all()
    # a garbage pitch sequence must fail to match, all cells missing
    out2 = note_tau(b, sp, t, np.full(40, 30))
    assert out2["method"] == "failed"
    assert not np.isfinite(out2["tau"]).any()
