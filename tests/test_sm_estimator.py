"""Tests for the SM-GP within-note estimator (phase2.sm_estimator).

Pins: the closed-form c-marginalized evidence against numeric integration;
recovery of (c, f, gamma) on synthetic vibrato with drift; the Fact-2
amplitude mapping gamma = sqrt(2 w1); graceful (wide, not refused)
behaviour on uninformative notes; predictive calibration on model-true
data; determinism.  numpy-only.
"""
from __future__ import annotations

import numpy as np

from score_bundle.phase2.sm_estimator import (fit_sm_note, sm_kernel,
                                              sm_log_evidence, sm_predict)


def _synth(rng, n=140, dur=1.4, c=12.0, gamma=18.0, f=5.5,
           slope=8.0, noise=2.0):
    t = np.linspace(0.0, dur, n)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    y = (c + gamma * np.sin(2 * np.pi * f * t + phi)
         + slope * (t - t.mean()) + noise * rng.standard_normal(n))
    return t, y


def test_evidence_matches_numeric_c_integration():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1.0, 25))
    y = rng.standard_normal(25) * 3.0 + 5.0
    p = np.array([np.log(4.0), 5.0, np.log(0.3), np.log(2.0),
                  np.log(0.1), np.log(1.5)])
    got = sm_log_evidence(t, y, p)

    # brute force: integrate N(y; c 1, K) over a wide c grid
    tau = t[:, None] - t[None, :]
    K = sm_kernel(tau, 4.0, 5.0, 0.3, 2.0, 0.1)
    K[np.diag_indices_from(K)] = 4.0 + 2.0 + 1.5 + 1e-10 * 7.5
    sign, logdet = np.linalg.slogdet(K)
    Ki = np.linalg.inv(K)
    cs = np.linspace(-40, 50, 20001)
    logs = np.array([
        -0.5 * len(t) * np.log(2 * np.pi) - 0.5 * logdet
        - 0.5 * (y - c) @ Ki @ (y - c) for c in cs])
    m = logs.max()
    numeric = m + np.log(np.trapezoid(np.exp(logs - m), cs))
    assert abs(got - numeric) < 1e-6


def test_recovers_synthetic_vibrato_with_drift():
    rng = np.random.default_rng(1)
    errs_f, errs_g, errs_c = [], [], []
    for _ in range(8):
        t, y = _synth(rng)
        out = fit_sm_note(t, y)
        errs_f.append(abs(out["f"] - 5.5))
        errs_g.append(abs(out["gamma"] - 18.0) / 18.0)
        errs_c.append(abs(out["c"] - 12.0))
    assert np.median(errs_f) < 0.25          # Hz
    assert np.median(errs_g) < 0.25          # relative
    assert np.median(errs_c) < 2.0           # cents
    assert out["var_c"] > 0 and np.isfinite(out["var_c"])


def test_fact2_amplitude_mapping():
    # pure random-phase sinusoid + small noise: gamma_hat = sqrt(2 w1) ~ a
    rng = np.random.default_rng(2)
    t = np.linspace(0, 2.0, 220)
    y = 25.0 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
    y = y + 1.0 * rng.standard_normal(t.size)
    out = fit_sm_note(t, y)
    assert abs(out["f"] - 6.0) < 0.1
    assert abs(out["gamma"] - 25.0) / 25.0 < 0.15


def test_weak_note_is_wide_not_refused():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 0.12, 10)             # far too short for 5 Hz
    y = 4.0 * rng.standard_normal(10) + 7.0
    out = fit_sm_note(t, y)
    assert np.isfinite(out["c"])             # centre always delivered
    # rate is unidentifiable here: variance must be large or flagged wide
    assert out["wide"] or out["var_f"] > 1.0


def test_prediction_covers_model_true_curve():
    rng = np.random.default_rng(4)
    hits, total = 0, 0
    for _ in range(6):
        t, y = _synth(rng, n=160, dur=1.6)
        fit = fit_sm_note(t, y)
        hold = np.arange(t.size) % 4 == 0    # held-out frames
        refit = fit_sm_note(t[~hold], y[~hold])
        if refit["params"] is None:
            continue
        m, v = sm_predict(t[~hold], y[~hold], refit["params"], t[hold])
        z = (y[hold] - m) / np.sqrt(v)
        hits += int(np.sum(np.abs(z) <= 1.6449))
        total += int(z.size)
    assert 0.75 <= hits / total <= 1.0       # ~90% nominal, loose band


def test_deterministic():
    rng = np.random.default_rng(5)
    t, y = _synth(rng)
    a = fit_sm_note(t, y)
    b = fit_sm_note(t, y)
    assert a["c"] == b["c"] and a["f"] == b["f"] and a["gamma"] == b["gamma"]
