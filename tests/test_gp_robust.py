"""PROTOTYPE tests — Student-t tau noise (gp_robust). NumPy-only."""
import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.gp import MultiOutputGraphGP
from score_bundle.gp_robust import (WeightedNoiseGP, fit_robust_tau,
                                    t_predictive_nll)
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.score import Note, Score


def _setup(n=30, seed=0):
    rng = np.random.default_rng(seed)
    notes = [Note(pitch=60 + int(rng.integers(-6, 7)), onset=float(i) * 0.5,
                  duration=0.5) for i in range(n)]
    L = laplacian(build_adjacency(Score(notes)))
    nu, U = np.linalg.eigh(L)
    Y = rng.standard_normal((n, 3)) * np.array([0.1, 0.8, 0.1])
    mask = ie.random_mask(n, rng, observed_frac=0.6)
    return nu, U, Y, mask, rng


def test_unit_scale_reduces_to_plain_gp():
    nu, U, Y, mask, _ = _setup()
    gp_w = WeightedNoiseGP(nu, U, kernel="additive")
    gp_p = MultiOutputGraphGP(nu, U, kernel="additive")
    x = gp_p.x0(); x[0] = 0.3; x[1:4] = 0.1; x[-3:] = np.log(0.05)
    assert abs(gp_w.log_marginal_likelihood(Y, mask, x)
               - gp_p.log_marginal_likelihood(Y, mask, x)) < 1e-10
    Mw, Sw = gp_w.posterior(Y, mask, x)
    Mp, Sp = gp_p.posterior(Y, mask, x)
    np.testing.assert_allclose(Mw, Mp, atol=1e-10)
    np.testing.assert_allclose(Sw, Sp, atol=1e-10)


def test_em_downweights_observed_tau_outliers():
    nu, U, Y, mask, rng = _setup(seed=1)
    obs = np.where(mask)[0]
    Y2 = Y.copy()
    bad = obs[:2]
    Y2[bad, 0] += 3.0  # gross tau outliers among the OBSERVED notes
    _, _, w = fit_robust_tau(nu, U, Y2, mask, nu_t=4.0, em_iters=2,
                             noise_floor=np.full(3, 1e-4), maxiter=40)
    w_bad = w[np.isin(obs, bad)]
    w_good = w[~np.isin(obs, bad)]
    assert w_bad.max() < np.median(w_good)  # outliers clearly downweighted


def test_t_predictive_bounds_outlier_penalty():
    # a 10-sigma point: Gaussian NLL ~ 50+, t(4) NLL stays modest
    y, m, s = np.array([10.0]), np.array([0.0]), np.array([1.0])
    g = 0.5 * (np.log(2 * np.pi) + 0.0 + 100.0)
    t = float(t_predictive_nll(y, m, s, nu_t=4.0)[0])
    assert t < 10.0 < g
    # near the center the two are close (within ~0.3 nats)
    y0 = np.array([0.5])
    g0 = 0.5 * (np.log(2 * np.pi) + 0.25)
    t0 = float(t_predictive_nll(y0, m, s, nu_t=4.0)[0])
    assert abs(t0 - g0) < 0.3
