"""End-to-end Phase-2 front end: extract_f0 (pyin) -> cents -> NLLS estimator.

Synthesizes a monophonic tone with known vibrato and checks the full chain
recovers the ground truth within tolerance.  Skips cleanly when librosa is
absent (optional dependency; the numpy core never imports it).
"""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from score_bundle.phase2.intonation import (cents_from_f0, extract_f0,
                                            fit_vibrato_note)

SR = 22050
F_REF = 440.0


def _vibrato_tone(rng, semitone=9, c_cents=15.0, gamma=35.0, rate=5.5,
                  delay=0.2, dur=1.6, snr_db=30.0):
    """Additive-synthesis tone whose instantaneous pitch follows eq:vibrato."""
    t = np.arange(int(dur * SR)) / SR
    cents = np.full_like(t, c_cents)
    on = t >= delay
    cents[on] += gamma * np.sin(2 * np.pi * rate * (t[on] - delay))
    f_inst = F_REF * 2.0 ** (semitone / 12.0) * 2.0 ** (cents / 1200.0)
    phase = 2 * np.pi * np.cumsum(f_inst) / SR
    x = np.sin(phase) + 0.3 * np.sin(2 * phase)   # fundamental + one overtone
    noise = rng.standard_normal(t.size)
    x = x + noise * np.sqrt(np.var(x) / (10 ** (snr_db / 10)))
    return x.astype(float)


def test_extract_f0_shapes_and_voicing():
    rng = np.random.default_rng(0)
    x = _vibrato_tone(rng)
    out = extract_f0(x, SR)
    assert set(out) == {"t", "f0", "voiced", "prob"}
    n = out["t"].size
    assert out["f0"].shape == (n,) and out["voiced"].shape == (n,)
    assert out["voiced"].mean() > 0.8          # a clean tone is mostly voiced
    v = out["f0"][out["voiced"]]
    assert np.all(np.isfinite(v))
    # hop grid is 10 ms
    assert np.allclose(np.diff(out["t"]), 0.010, atol=1e-4)


def test_full_chain_recovers_known_vibrato():
    rng = np.random.default_rng(1)
    c_true, gamma_true, rate_true, delay = 15.0, 35.0, 5.5, 0.2
    x = _vibrato_tone(rng, c_cents=c_true, gamma=gamma_true, rate=rate_true,
                      delay=delay)
    out = extract_f0(x, SR)
    voiced = out["voiced"] & (out["t"] >= delay + 0.05) & (out["t"] <= 1.55)
    cents = cents_from_f0(out["f0"][voiced], F_REF, 9)
    fit = fit_vibrato_note(out["t"][voiced] - delay, cents)
    assert fit["vibrato_identifiable"]
    assert abs(fit["f"] - rate_true) < 0.15
    assert abs(fit["gamma"] - gamma_true) < 5.0
    assert abs(fit["c"] - c_true) < 5.0


def test_unvoiced_noise_is_flagged():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(SR)               # 1 s of white noise
    out = extract_f0(x, SR)
    assert out["voiced"].mean() < 0.3
