"""Tests for per-(note, channel) cell masks in `gp.MultiOutputGraphGP`.

The Phase-2 missingness case: a short note may supply intonation but no
vibrato parameters.  Pins: (1) a 2-D mask whose rows are constant reproduces
the published 1-D path bitwise-close; (2) evidence and posterior under an
arbitrary cell mask equal brute-force Gaussian identities computed
independently from the full covariance; (3) heteroscedastic per-cell noise
(`noise_scale`) is honored exactly on the cell path.
"""
from __future__ import annotations

import numpy as np

from score_bundle.gp import MultiOutputGraphGP
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.score import Score

_LOG2PI = float(np.log(2.0 * np.pi))


def _setup(n=40, k=3, seed=0, features=True, noise_scale=None):
    rng = np.random.default_rng(seed)
    onset = np.cumsum(rng.choice([0.25, 0.5, 1.0], size=n))
    pitch = np.clip(60 + np.cumsum(rng.integers(-3, 4, size=n)), 40, 90)
    score = Score.from_arrays(pitch, onset, rng.choice([0.5, 1.0], size=n),
                              np.zeros(n, dtype=int))
    nu, U = np.linalg.eigh(laplacian(build_adjacency(score)))
    feats = [np.concatenate([rng.normal(size=(n, 4)), np.ones((n, 1))], axis=1)] \
        if features else None
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                            n_channels=k)
    if noise_scale is not None:
        gp.noise_scale = noise_scale
    x = gp.x0() + 0.1 * rng.normal(size=gp.n_params())
    Y = rng.normal(size=(n, k))
    return gp, x, Y, rng


def _brute(gp, x, Y, mask2d):
    """Independent dense-Gaussian reference: evidence + conditional moments."""
    p = gp.unpack(x)
    allidx = np.arange(gp.N)
    C = gp._blocks(p, allidx, allidx)
    scale = getattr(gp, "noise_scale", None)
    nv = np.concatenate([
        p["noise"][c] * (np.asarray(scale, dtype=float)[:, c]
                         if scale is not None else np.ones(gp.N))
        for c in range(gp.k)])
    obs = np.concatenate([c * gp.N + np.where(mask2d[:, c])[0]
                          for c in range(gp.k)])
    K = C[np.ix_(obs, obs)] + np.diag(nv[obs])
    y = np.concatenate([Y[:, c] for c in range(gp.k)])[obs]
    sign, logdet = np.linalg.slogdet(K)
    lml = -0.5 * (y @ np.linalg.solve(K, y) + logdet + y.size * _LOG2PI)
    m = C[:, obs] @ np.linalg.solve(K, y)
    V = C - C[:, obs] @ np.linalg.solve(K, C[obs, :])
    return (float(lml), m.reshape(gp.k, gp.N).T,
            np.sqrt(np.clip(np.diag(V), 0, None)).reshape(gp.k, gp.N).T)


def test_rowconstant_cellmask_equals_note_mask():
    gp, x, Y, rng = _setup()
    mask1d = rng.random(gp.N) < 0.6
    mask2d = np.repeat(mask1d[:, None], gp.k, axis=1)
    l1 = gp.log_marginal_likelihood(Y, mask1d, x)
    l2 = gp.log_marginal_likelihood(Y, mask2d, x)
    assert abs(l1 - l2) < 1e-8
    m1, s1 = gp.posterior(Y, mask1d, x)
    m2, s2 = gp.posterior(Y, mask2d, x)
    np.testing.assert_allclose(m1, m2, atol=1e-8)
    np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_arbitrary_cellmask_matches_bruteforce():
    gp, x, Y, rng = _setup(seed=1)
    mask2d = rng.random((gp.N, gp.k)) < 0.55
    mask2d[0, :] = True                       # at least one full note
    lml = gp.log_marginal_likelihood(Y, mask2d, x)
    m, s = gp.posterior(Y, mask2d, x)
    lml_b, m_b, s_b = _brute(gp, x, Y, mask2d)
    assert abs(lml - lml_b) < 1e-8
    np.testing.assert_allclose(m, m_b, atol=1e-8)
    np.testing.assert_allclose(s, s_b, atol=1e-6)


def test_heteroscedastic_cell_noise_honored():
    rng0 = np.random.default_rng(3)
    scale = np.exp(rng0.normal(scale=0.7, size=(40, 3)))
    gp, x, Y, rng = _setup(seed=3, noise_scale=scale)
    mask2d = rng.random((gp.N, gp.k)) < 0.6
    lml = gp.log_marginal_likelihood(Y, mask2d, x)
    m, s = gp.posterior(Y, mask2d, x)
    lml_b, m_b, s_b = _brute(gp, x, Y, mask2d)
    assert abs(lml - lml_b) < 1e-8
    np.testing.assert_allclose(m, m_b, atol=1e-8)
    np.testing.assert_allclose(s, s_b, atol=1e-6)
    # and the scale genuinely matters
    del gp.noise_scale
    assert abs(gp.log_marginal_likelihood(Y, mask2d, x) - lml) > 1e-3


def test_fit_runs_under_cellmask_and_missing_cells_get_wider_bands():
    gp, x, Y, rng = _setup(seed=5)
    mask2d = rng.random((gp.N, gp.k)) < 0.7
    Ynan = Y.copy()
    Ynan[~mask2d] = np.nan                    # missing cells never read
    floor = 0.05 * np.array([float(np.var(Y[mask2d[:, c], c]))
                             for c in range(3)])
    x_hat, info = gp.fit(Ynan, mask2d, noise_floor=floor, maxiter=60)
    assert np.isfinite(info["nll"])
    m, s = gp.posterior(Ynan, mask2d, x_hat)
    assert np.isfinite(m).all() and np.isfinite(s).all()
    # a hidden cell is less pinned than an observed one, on average
    assert s[~mask2d].mean() > s[mask2d].mean()
