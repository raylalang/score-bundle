"""Tests for the differentiable Phase-3 curve module (phase3/curve.py).

These require torch; when it is absent each test returns early (a no-op)
so the numpy-only suite still runs.  Everything is synthetic — no dataset
access.  The pins: torch mirrors equal their numpy references (kernel,
design, collapsed likelihood), the likelihood gradient survives gradcheck,
the full Laplace matches a finite-difference Hessian, and MAP recovery on
model-true audio lands within tolerance with honest bands.
"""
from __future__ import annotations

import numpy as np

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def _np_chunked_design(f0, t, n_harm, n_chunk):
    """Numpy reference: the development study's chunked design."""
    from score_bundle.phase3.synth import harmonic_design_matrix

    base = harmonic_design_matrix(f0, t, n_harm)
    edges = np.linspace(0, t.size, n_chunk + 1).astype(int)
    cols = []
    for q in range(n_chunk):
        blk = np.zeros_like(base)
        blk[edges[q]:edges[q + 1]] = base[edges[q]:edges[q + 1]]
        cols.append(blk)
    return np.concatenate(cols, axis=1)


def _synthetic_note(rng, dur=0.3, sr=4000, midi=60.0, n_harm=4, n_chunk=2,
                    amp_var=10.0, cents_fn=None, noise_frac=0.1):
    """Model-true synthetic audio; returns (x, t, cents, noise_sd)."""
    t = np.arange(int(dur * sr)) / sr
    cents = (10.0 * np.sin(2 * np.pi * 5.0 * t) + 3.0
             if cents_fn is None else cents_fn(t))
    f0 = 440.0 * 2.0 ** ((midi - 69.0 + cents / 100.0) / 12.0)
    Phi = _np_chunked_design(f0, t, n_harm, n_chunk)
    a = rng.normal(0.0, np.sqrt(amp_var), Phi.shape[1])
    signal = Phi @ a
    noise_sd = noise_frac * float(signal.std())
    x = signal + rng.normal(0.0, noise_sd, signal.size)
    return x, t, cents, noise_sd


def test_sm_kernel_torch_matches_numpy():
    if not HAS_TORCH:
        return
    from score_bundle.phase2.sm_estimator import sm_kernel
    from score_bundle.phase3.curve import sm_kernel_torch

    rng = np.random.default_rng(0)
    tau = rng.normal(0.0, 0.3, (7, 7))
    w1, mu1, v1, w2, v2 = 30.0, 5.5, 0.4, 60.0, 0.3
    ref = sm_kernel(tau, w1, mu1, v1, w2, v2)
    got = sm_kernel_torch(torch.as_tensor(tau, dtype=torch.float64),
                          w1, mu1, v1, w2, v2).numpy()
    assert np.allclose(got, ref, atol=1e-12)


def test_design_and_loglik_match_numpy():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import (chunked_design_torch,
                                           collapsed_loglik_torch)
    from score_bundle.phase3.waveform_model import (collapsed_loglik,
                                                    collapsed_loglik_lowrank)

    rng = np.random.default_rng(1)
    n_harm, n_chunk, amp_var, midi = 4, 2, 10.0, 60.0
    x, t, cents, noise_sd = _synthetic_note(rng, n_harm=n_harm,
                                            n_chunk=n_chunk, midi=midi)
    f0 = 440.0 * 2.0 ** ((midi - 69.0 + cents / 100.0) / 12.0)
    Phi_np = _np_chunked_design(f0, t, n_harm, n_chunk)
    Phi_t = chunked_design_torch(
        torch.as_tensor(cents, dtype=torch.float64),
        torch.as_tensor(t, dtype=torch.float64), midi, n_harm, n_chunk)
    assert np.allclose(Phi_t.numpy(), Phi_np, atol=1e-9)

    nv = noise_sd ** 2
    Sigma_a = np.eye(Phi_np.shape[1]) * amp_var
    ll_lowrank = collapsed_loglik_lowrank(x, Phi_np, Sigma_a, noise_var=nv)
    ll_dense = collapsed_loglik(x, Phi_np, Sigma_a, noise_var=nv)
    ll_torch = float(collapsed_loglik_torch(
        torch.as_tensor(x, dtype=torch.float64), Phi_t, nv, amp_var))
    assert abs(ll_torch - ll_lowrank) < 1e-6 * abs(ll_lowrank)
    assert abs(ll_torch - ll_dense) < 1e-6 * abs(ll_dense)


def test_loglik_gradcheck():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import (chunked_design_torch,
                                           collapsed_loglik_torch)

    rng = np.random.default_rng(2)
    x, t, cents, noise_sd = _synthetic_note(rng, dur=0.25, sr=480,
                                            n_harm=2, n_chunk=2)
    x_t = torch.as_tensor(x, dtype=torch.float64)
    t_t = torch.as_tensor(t, dtype=torch.float64)
    nv = noise_sd ** 2

    def f(cents_t):
        Phi = chunked_design_torch(cents_t, t_t, 60.0, 2, 2)
        return collapsed_loglik_torch(x_t, Phi, nv, 10.0)

    cents0 = torch.as_tensor(cents, dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (cents0,), eps=1e-6, atol=1e-4,
                                    rtol=1e-3)


def test_laplace_matches_finite_differences():
    if not HAS_TORCH:
        return
    from score_bundle.phase3.curve import (CurvePrior, fit_note, knot_gram,
                                           make_neg_log_joint)

    rng = np.random.default_rng(3)
    prior = CurvePrior(knots_per_s=10.0, j_min=4, j_max=6)
    x, t, _, _ = _synthetic_note(rng, dur=0.4, sr=2000, n_harm=3, n_chunk=2)
    fit = fit_note(x, t, 60.0, prior, n_harm=3, n_chunk=2, n_steps=150)

    knots = torch.as_tensor(fit.knots, dtype=torch.float64)
    _, chol_k = knot_gram(knots, prior)
    nlj = make_neg_log_joint(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(t, dtype=torch.float64), 60.0, knots, chol_k,
        fit.noise_var, 10.0, 3, 2)
    theta = np.concatenate([[fit.c], fit.u])
    d = theta.size
    h = 1e-4
    H_fd = np.zeros((d, d))

    def val(v):
        return float(nlj(torch.as_tensor(v, dtype=torch.float64)))

    f0 = val(theta)
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d)
            ej = np.zeros(d)
            ei[i] = h
            ej[j] = h
            if i == j:
                H_fd[i, i] = (val(theta + ei) - 2 * f0 + val(theta - ei)) / h ** 2
            else:
                H_fd[i, j] = H_fd[j, i] = (
                    val(theta + ei + ej) - val(theta + ei - ej)
                    - val(theta - ei + ej) + val(theta - ei - ej)
                ) / (4 * h ** 2)

    # Pin 1: the autograd Hessian equals the finite-difference one.
    H_auto = torch.autograd.functional.hessian(
        nlj, torch.as_tensor(theta, dtype=torch.float64)).numpy()
    scale = np.abs(H_fd).max()
    assert np.allclose(H_auto, H_fd, rtol=5e-3, atol=2e-4 * scale)

    # Pin 2: the curve bands agree.  The raw (c, u) covariance is NOT
    # comparable entrywise: the likelihood is exactly flat along
    # (c + delta, u - delta), so both Hessians are near-singular there and
    # their inverses disagree in that direction — but curve() bands are
    # invariant to it (the interp weights contract it to zero).
    from dataclasses import replace

    cov_fd = np.linalg.inv(H_fd)
    tq = np.linspace(t[0], t[-1], 50)
    _, sd_auto = fit.curve(tq)
    _, sd_fd = replace(fit, cov=cov_fd).curve(tq)
    assert fit.laplace_pd
    assert np.allclose(sd_auto, sd_fd, rtol=2e-2, atol=1e-3)

    # Pin 3: evidence_fixed_map at the fit's own hyperparameters
    # reproduces CurveFit.log_evidence (the coordinate-ascent base point).
    from score_bundle.phase3.curve import (chunked_design_torch,
                                           collapsed_loglik_torch,
                                           evidence_fixed_map, interp_knots,
                                           loglik_hessian)

    x_t = torch.as_tensor(x, dtype=torch.float64)
    t_t = torch.as_tensor(t, dtype=torch.float64)
    u_t = torch.as_tensor(fit.u, dtype=torch.float64)
    cents = fit.c + interp_knots(u_t, knots, t_t)
    Phi = chunked_design_torch(cents, t_t, 60.0, 3, 2)
    loglik_map = float(collapsed_loglik_torch(x_t, Phi, fit.noise_var, 10.0))
    H_lik = torch.as_tensor(
        loglik_hessian(x, t, 60.0, fit, n_harm=3, n_chunk=2),
        dtype=torch.float64)
    ev = float(evidence_fixed_map(loglik_map, u_t, knots, H_lik, prior))
    assert abs(ev - fit.log_evidence) < 1e-6 * abs(fit.log_evidence), (
        ev, fit.log_evidence)


def test_map_recovery_model_true():
    if not HAS_TORCH:
        return
    from score_bundle.phase2.sm_estimator import sm_kernel
    from score_bundle.phase3.curve import CurvePrior, fit_note

    rng = np.random.default_rng(4)
    prior = CurvePrior(knots_per_s=12.0, j_min=4, j_max=8)
    dur, sr, midi, c_true = 0.5, 4000, 60.0, 12.0

    # prior-sampled curve on the fit's own knot grid (model-true)
    j = prior.n_knots(dur - 1.0 / sr)
    knots = np.linspace(0.0, dur - 1.0 / sr, j)
    K = sm_kernel(knots[:, None] - knots[None, :], prior.w1, prior.mu1,
                  prior.v1, prior.w2, prior.v2)
    K[np.diag_indices_from(K)] += 1e-6 * (prior.w1 + prior.w2)
    u_true = np.linalg.cholesky(K) @ rng.normal(size=j)

    def cents_fn(t):
        return c_true + np.interp(t, knots, u_true)

    x, t, cents_true, _ = _synthetic_note(
        rng, dur=dur, sr=sr, midi=midi, n_harm=4, n_chunk=2,
        cents_fn=cents_fn, noise_frac=0.1)
    fit = fit_note(x, t, midi, prior, n_harm=4, n_chunk=2, n_steps=300)

    tq = np.linspace(t[0], t[-1], 200)
    mean, sd = fit.curve(tq)
    true_q = cents_fn(tq)
    rmse = float(np.sqrt(np.mean((mean - true_q) ** 2)))
    cover = float(np.mean(np.abs(mean - true_q) <= 1.645 * sd))
    assert rmse < 2.0, rmse
    assert cover >= 0.5, cover
    assert np.isfinite(fit.log_evidence)
