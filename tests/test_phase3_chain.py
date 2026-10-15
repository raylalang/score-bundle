"""Tests for the chain / transition-region boundary model (phase3/curve.py).

Torch-gated no-ops like the other phase-3 curve tests; all synthetic.
Pins: the support design equals the full design where supports coincide;
a model-true two-note chain recovers the boundary and both centres from
a perturbed init; window masses are sane; the ordering hinge activates.
"""
from __future__ import annotations

import numpy as np

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def test_design_support_matches_full():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import (chunked_design_torch,
                                           design_support_torch)

    rng = np.random.default_rng(0)
    t = np.arange(800) / 2000.0
    cents = rng.normal(0.0, 5.0, t.size)
    t_t = torch.as_tensor(t, dtype=torch.float64)
    c_t = torch.as_tensor(cents, dtype=torch.float64)
    full = chunked_design_torch(c_t, t_t, 60.0, 3, 2)
    sup = design_support_torch(c_t, t_t, 60.0, 0, t.size, 3, 2)
    assert np.allclose(sup.numpy(), full.numpy(), atol=1e-12)
    # padded variant: rows outside [ia, ib) are exactly zero
    ia, ib = 100, 500
    part = design_support_torch(c_t[ia:ib], t_t[ia:ib], 60.0, ia, t.size,
                                3, 2)
    assert part.shape == full.shape
    assert float(part[:ia].abs().max()) == 0.0
    assert float(part[ib:].abs().max()) == 0.0


def _two_note_audio(rng, sr=4000, dur=1.2, b_true=0.6, width_true=0.02,
                    midis=(60.0, 64.0), n_harm=3, n_chunk=2, amp_var=10.0,
                    noise_frac=0.08):
    """Model-true chain audio: two notes, sigmoid-edged crossfade."""
    from score_bundle.phase3.curve import design_support_torch

    t = np.arange(int(dur * sr)) / sr
    m = t.size
    t_t = torch.as_tensor(t, dtype=torch.float64)
    cents = [8.0 * np.sin(2 * np.pi * 5.0 * t) + 5.0,
             6.0 * np.sin(2 * np.pi * 5.5 * t) - 4.0]
    b_lo, b_hi = b_true - width_true / 2, b_true + width_true / 2
    x = np.zeros(m)
    softness = 0.010
    for k, (midi, ce) in enumerate(zip(midis, cents)):
        Phi = design_support_torch(
            torch.as_tensor(ce, dtype=torch.float64), t_t, midi, 0, m,
            n_harm, n_chunk).numpy()
        w = np.ones(m)
        if k == 1:
            w *= 1.0 / (1.0 + np.exp(-(t - b_lo) / softness))
        else:
            w *= 1.0 / (1.0 + np.exp(-(b_hi - t) / softness))
        a = rng.normal(0.0, np.sqrt(amp_var), Phi.shape[1])
        x += (Phi * w[:, None]) @ a
    x += rng.normal(0.0, noise_frac * x.std(), m)
    return x, t, cents, (b_lo, b_hi)


def test_chain_model_true_recovery():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import CurvePrior, fit_chain

    rng = np.random.default_rng(1)
    b_true, width_true = 0.6, 0.02
    x, t, cents, (b_lo_true, b_hi_true) = _two_note_audio(
        rng, b_true=b_true, width_true=width_true)
    prior = CurvePrior(knots_per_s=12.0, j_min=4, j_max=10)
    fit = fit_chain(x, t, (60.0, 64.0), [b_true + 0.04], prior=prior,
                    n_harm=3, n_chunk=2, n_steps=300, support_margin=0.15)

    # boundary: the fade-in edge recovers within ~15 ms from a 40 ms start
    assert abs(fit.b_lo[0] - b_lo_true) < 0.015, fit.b_lo
    assert fit.widths[0] < 0.06, fit.widths
    # centres: both notes' c within 2.5 cents of their true curve mean
    # over their own region
    for k, sl in ((0, t < b_lo_true), (1, t > b_hi_true)):
        mean_true = cents[k][sl].mean()
        mean_fit = fit.curve(k, t[sl]).mean()
        assert abs(mean_fit - mean_true) < 2.5, (k, mean_fit, mean_true)
    # window masses: both notes solidly present
    assert fit.window_mass.min() > 0.6, fit.window_mass
    assert np.isfinite(fit.neg_log_joint)
    assert fit.noise_var > 0 and fit.noise_var_init > 0


def test_fixed_width_arm_freezes_width():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import CurvePrior, fit_chain

    rng = np.random.default_rng(2)
    x, t, _, _ = _two_note_audio(rng)
    prior = CurvePrior(knots_per_s=12.0, j_min=4, j_max=10)
    fit = fit_chain(x, t, (60.0, 64.0), [0.64], prior=prior, n_harm=3,
                    n_chunk=2, n_steps=120, free_width=False,
                    fixed_width=0.010)
    assert np.allclose(fit.widths, 0.010, atol=1e-12)


def test_ordering_hinge_activates():
    if not HAS_TORCH:
        return
    # three very short notes with boundaries initialized so close that an
    # unpenalized fade-out would overrun the next fade-in; the fit must
    # keep b_hi[0] below b_lo[1] + margin (window mass of the middle note
    # stays > 0.3 rather than being swallowed).
    from score_bundle.phase3.curve import CurvePrior, fit_chain

    rng = np.random.default_rng(3)
    sr, dur = 4000, 1.2
    t = np.arange(int(dur * sr)) / sr
    x = rng.normal(0.0, 1.0, t.size)  # pure noise: likelihood indifferent
    prior = CurvePrior(knots_per_s=8.0, j_min=4, j_max=6)
    fit = fit_chain(x, t, (60.0, 64.0, 67.0), [0.42, 0.52], prior=prior,
                    n_harm=2, n_chunk=2, n_steps=80, order_margin=0.05)
    assert fit.b_hi[0] <= fit.b_lo[1] - 0.05 + 0.005, (fit.b_hi, fit.b_lo)
    assert fit.window_mass[1] > 0.3, fit.window_mass


def test_sample_mask_equals_truncated_likelihood():
    if not HAS_TORCH:
        return
    # masking rows must equal fitting on the kept samples only: the
    # reference noise and the fit on (x, mask) match a fit on a chain
    # whose masked note is pure (unmodelled) noise in the unmasked arm
    # only through the mask; here we pin the mechanical identity that
    # chain_reference with a mask ignores the masked span entirely.
    from score_bundle.phase3.curve import chain_reference

    rng = np.random.default_rng(5)
    x, t, _, _ = _two_note_audio(rng)
    mask = np.ones(t.size, dtype=bool)
    sl = slice(int(0.55 * 4000), int(1.2 * 4000))
    mask[sl] = False
    x_perturbed = x.copy()
    x_perturbed[sl] = rng.normal(0.0, 10.0, sl.stop - sl.start)
    c_a, nv_a = chain_reference(x, t, (60.0, 64.0), [0.6],
                                sample_mask=mask, n_harm=3, n_chunk=2)
    c_b, nv_b = chain_reference(x_perturbed, t, (60.0, 64.0), [0.6],
                                sample_mask=mask, n_harm=3, n_chunk=2)
    assert np.allclose(c_a, c_b) and abs(nv_a - nv_b) < 1e-12
    # masked note 2 falls back to the prior-mean initializer
    assert c_b[1] == 0.0


def test_c_prior_pulls_masked_note():
    if not HAS_TORCH:
        return
    # two-note model-true chain, note 2's audio masked out: with a
    # strongly coupled c prior the hidden note's constant is pulled
    # toward note 1's (informative); with an independent prior it stays
    # at the prior mean (0). Boundaries fixed throughout.
    from score_bundle.phase3.curve import CurvePrior, fit_chain

    rng = np.random.default_rng(6)
    b_true = 0.6

    # build audio with constant offsets +18 and +15 cents (correlated)
    from score_bundle.phase3.curve import design_support_torch
    sr, dur = 4000, 1.2
    t = np.arange(int(dur * sr)) / sr
    m = t.size
    x = np.zeros(m)
    for k, (midi, cval) in enumerate(((60.0, 18.0), (64.0, 15.0))):
        ce = np.full(m, cval)
        Phi = design_support_torch(
            torch.as_tensor(ce, dtype=torch.float64),
            torch.as_tensor(t, dtype=torch.float64), midi, 0, m, 3,
            2).numpy()
        w = (1.0 / (1.0 + np.exp(-(b_true - t) / 0.01)) if k == 0
             else 1.0 / (1.0 + np.exp(-(t - b_true) / 0.01)))
        a = rng.normal(0.0, np.sqrt(10.0), Phi.shape[1])
        x += (Phi * w[:, None]) @ a
    x += rng.normal(0.0, 0.05 * x.std(), m)

    mask = np.ones(m, dtype=bool)
    # note 2 unobserved; the mask extends 50 ms before the boundary so
    # the note's sigmoid window leak carries no evidence (the same guard
    # the experiment uses -- without it the leak pulls the "hidden" c)
    mask[int((b_true - 0.05) * sr):] = False
    prior = CurvePrior(knots_per_s=8.0, j_min=4, j_max=6)
    var_c = 20.0 ** 2
    k_ind = np.eye(2) * var_c
    k_cpl = var_c * np.array([[1.0, 0.95], [0.95, 1.0]])
    common = dict(prior=prior, n_harm=3, n_chunk=2, n_steps=250,
                  sample_mask=mask, fix_boundaries=True)
    fit_i = fit_chain(x, t, (60.0, 64.0), [b_true], c_prior_cov=k_ind,
                      **common)
    fit_g = fit_chain(x, t, (60.0, 64.0), [b_true], c_prior_cov=k_cpl,
                      **common)
    # independent: hidden c stays near the prior mean
    assert abs(fit_i.c[1]) < 6.0, fit_i.c
    # coupled: hidden c pulled most of the way to the observed note's c
    assert fit_g.c[1] > 9.0, fit_g.c
    # observed note barely differs between arms
    assert abs(fit_i.c[0] - fit_g.c[0]) < 3.0
    # boundaries stayed exactly at init
    assert np.allclose(fit_g.b_lo, [b_true], atol=1e-12)
