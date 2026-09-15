"""Tests for the exploration-week adoption capabilities (all opt-in).

Pins: (1) with ``noise_corr`` unset or rho=0 the correlated-noise paths are
BIT-IDENTICAL to the published diagonal paths; (2) the correlated
observation predictive equals brute-force Gaussian conditioning; (3) the
evidence prefers rho > 0 on synthetically correlated noise; (4) fit_t_em
down-weights planted outliers and reduces to ~the plain fit on clean
data; (5) the t predictive is variance-matched and tail-robust; (6) the
completion guard flags planted extrapolation and stays quiet on
in-distribution data.  numpy-only.
"""
from __future__ import annotations

import numpy as np

from score_bundle.downstream import completion_guard
from score_bundle.gp import MultiOutputGraphGP
from score_bundle.metrics import gaussian_nll, student_t_nll


def _toy_gp(n=24, k=3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W
    nu, U = np.linalg.eigh(L)
    X = rng.standard_normal((n, 4))
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X],
                            n_channels=k)
    Y = rng.standard_normal((n, k))
    mask2d = rng.random((n, k)) > 0.35
    x = gp.x0() + 0.1 * rng.standard_normal(gp.x0().size)
    return gp, Y, mask2d, x


def test_rho_zero_and_unset_are_bit_identical():
    gp, Y, mask2d, x = _toy_gp()
    lml_default = gp.log_marginal_likelihood(Y, mask2d, x)
    m0, s0 = gp.posterior(Y, mask2d, x)
    gp.noise_corr = {"channel": 0, "rho": 0.0}
    assert gp.log_marginal_likelihood(Y, mask2d, x) == lml_default
    m1, s1 = gp.posterior(Y, mask2d, x)
    assert np.array_equal(m0, m1) and np.array_equal(s0, s1)


def test_posterior_observations_matches_brute_force():
    gp, Y, mask2d, x = _toy_gp(n=12, seed=1)
    gp.noise_corr = {"channel": 1, "rho": 0.6}
    p = gp.unpack(x)
    allidx = np.arange(gp.N)
    every = np.arange(gp.k * gp.N)
    T = gp._blocks(p, allidx, allidx) + gp._noise_cov_obs(p, every)
    obs = gp._cell_obs(mask2d)
    y = np.concatenate([Y[:, c] for c in range(gp.k)])[obs]
    # brute force: condition the joint normal on the observed cells
    T_oo = T[np.ix_(obs, obs)]
    m_b = T[:, obs] @ np.linalg.solve(T_oo, y)
    v_b = np.diag(T) - np.einsum(
        "ij,ji->i", T[:, obs], np.linalg.solve(T_oo, T[:, obs].T))
    M, S = gp.posterior_observations(Y, mask2d, x)
    assert np.allclose(M, m_b.reshape(gp.k, gp.N).T, atol=1e-9)
    assert np.allclose(S ** 2, np.clip(v_b, 1e-12, None
                                       ).reshape(gp.k, gp.N).T, atol=1e-8)


def test_evidence_prefers_true_rho_on_correlated_noise():
    rng = np.random.default_rng(2)
    n, k, rho_true = 40, 2, 0.7
    nu, U = np.linalg.eigh(np.eye(n))          # trivial graph
    gp = MultiOutputGraphGP(nu, U, kernel="none", features=[], n_channels=k)
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    R = rho_true ** lag
    e0 = np.linalg.cholesky(R + 1e-9 * np.eye(n)) @ rng.standard_normal(n)
    Y = np.stack([e0, rng.standard_normal(n)], axis=1)
    mask2d = np.ones((n, k), dtype=bool)
    x = gp.x0()
    x[-k:] = 0.0                                # noise var ~ 1
    gp.noise_corr = {"channel": 0, "rho": 0.0}
    lml0 = gp.log_marginal_likelihood(Y, mask2d, x)
    gp.noise_corr = {"channel": 0, "rho": rho_true}
    lml1 = gp.log_marginal_likelihood(Y, mask2d, x)
    assert lml1 > lml0 + 1.0


def test_fit_t_em_downweights_planted_outliers():
    rng = np.random.default_rng(3)
    n = 60
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W
    nu_e, U = np.linalg.eigh(L)
    X = rng.standard_normal((n, 2))             # low capacity on purpose
    gp = MultiOutputGraphGP(nu_e, U, kernel="additive", features=[X],
                            n_channels=3)
    Y = rng.standard_normal((n, 3))
    mask2d = rng.random((n, 3)) > 0.2
    bad = np.where(mask2d[:, 0])[0][:2]
    Y[bad, 0] += 20.0                           # planted observed outliers
    # oracle noise: the toy evidence would otherwise absorb the outliers
    # into the channel noise (the real pipeline's floor prevents that)
    x_hat, info = gp.fit_t_em(Y, mask2d, channel=0, rounds=3,
                              noise_fixed=np.ones(3), maxiter=60)
    scale = gp.noise_scale
    others = np.setdiff1d(np.where(mask2d[:, 0])[0], bad)
    assert scale[bad, 0].min() > 3.0            # outliers inflated hard
    assert np.median(scale[others, 0]) < 2.0
    assert np.isfinite(gp.log_marginal_likelihood(Y, mask2d, x_hat))


def test_fit_t_em_near_plain_on_clean_data():
    gp, Y, mask2d, _ = _toy_gp(n=26, seed=4)
    floor = 0.05 * np.array([Y[mask2d[:, c], c].var() for c in range(3)])
    x_plain, _ = gp.fit(Y, mask2d, noise_floor=floor, maxiter=60)
    m_plain, _ = gp.posterior(Y, mask2d, x_plain)
    gp2 = MultiOutputGraphGP(gp.nu, gp.U, kernel="additive",
                             features=gp.features, n_channels=3)
    x_em, _ = gp2.fit_t_em(Y, mask2d, channel=0, noise_floor=floor,
                           maxiter=60)
    m_em, _ = gp2.posterior(Y, mask2d, x_em)
    held = ~mask2d
    assert np.abs(m_plain[held] - m_em[held]).mean() < 0.15


def test_student_t_nll_variance_matched_and_tail_robust():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(4000)
    m = np.zeros(4000)
    s = np.ones(4000)
    g = gaussian_nll(y, m, s)
    t = student_t_nll(y, m, s, nu=5.0)
    assert abs(t - g) < 0.1                     # near-Gaussian on clean data
    y_out = y.copy()
    y_out[:4] = 40.0                            # a few many-sigma outliers
    assert gaussian_nll(y_out, m, s) - g > 0.5
    assert student_t_nll(y_out, m, s, nu=5.0) - t < 0.05


def test_completion_guard_flags_extrapolation_only():
    rng = np.random.default_rng(6)
    n, d, k = 120, 15, 3
    X = rng.standard_normal((n, d))
    X[100:] += 8.0                              # a far-out held-out block
    mask = np.ones(n, dtype=bool)
    mask[80:] = False                           # observed = in-distribution
    pred = rng.standard_normal((n, k))
    head = pred + 0.1 * rng.standard_normal((n, k))
    head_sd = np.ones(k)
    pred[110, 0] = 50.0                         # one catastrophic cell
    note_flags, cell_flags = completion_guard(X, mask, pred, head, head_sd)
    assert note_flags[100:].mean() > 0.9        # extrapolated block caught
    assert note_flags[80:100].mean() < 0.2      # in-distribution held-out ok
    assert cell_flags[110, 0]                   # the catastrophe caught
    assert cell_flags.mean() < 0.05
