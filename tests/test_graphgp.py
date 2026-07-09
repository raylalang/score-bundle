"""GP-first model (gp.MultiOutputGraphGP): the orthodoxy reformulation's contracts.

  1. SPECIAL CASE: with B diagonal and no feature kernel, the joint marginal
     likelihood equals the sum of the three per-channel SpectralGaussianField
     margliks under the matched (lam, eta) reparameterization — the published
     model is nested inside the GP-first model, exactly.
  2. MEAN-AS-KERNEL: a linear feature kernel equals the explicitly marginalized
     Bayesian linear mean N(0, K + X S X^T) — folding the mean into the kernel is
     the same model.
  3. ICM posterior: cross-channel coupling actually transfers information (a
     correlated channel improves prediction of a held-out one vs B diagonal).
  4. Zero-leak: corrupting held-out targets leaves fit + predictions bitwise
     unchanged (same contract as the kernel sweep).

NumPy-only.
"""
import numpy as np
import pytest

from score_bundle import imputation_eval as ie
from score_bundle.gp import MultiOutputGraphGP, chol_to_B, shape_cov
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.model import SpectralGaussianField
from score_bundle.prior import spectral_covariance
from score_bundle.score import Note, Score


def _setup(n=24, seed=0):
    rng = np.random.default_rng(seed)
    notes = [Note(pitch=60 + int(rng.integers(-6, 7)), onset=float(i) * 0.5,
                  duration=0.5) for i in range(n)]
    L = laplacian(build_adjacency(Score(notes)))
    nu, U = np.linalg.eigh(L)
    Y = rng.standard_normal((n, 3)) * np.array([0.2, 0.9, 0.1])
    mask = ie.random_mask(n, rng, observed_frac=0.6)
    return nu, U, Y, mask, rng


def test_bdiag_marglik_equals_sum_of_scalar_fields():
    nu, U, Y, mask, _ = _setup()
    gp = MultiOutputGraphGP(nu, U, kernel="additive")
    # params: shape s, B = diag(b), noise nv (all fixed, no fitting)
    s, b, nv = 0.7, np.array([0.5, 2.0, 0.08]), np.array([0.02, 0.1, 0.01])
    x = gp.x0()
    x[0] = np.log(s)
    x[1:4] = 0.5 * np.log(b)          # log-Cholesky diag of diag(b)
    x[4:7] = 0.0                       # off-diagonals zero
    x[-3:] = np.log(nv)
    lml_joint = gp.log_marginal_likelihood(Y, mask, x)

    # per-channel: b_c * 1/(1+s*nu) = 1/(lam + eta*nu) with lam=1/b_c, eta=s/b_c
    lml_sum = 0.0
    for c in range(3):
        K = spectral_covariance(U, nu, "additive", (1.0 / b[c], s / b[c]))
        f = SpectralGaussianField(K)
        lml_sum += f.log_marginal_likelihood(Y[:, c], nv[c], mask=mask)
    assert abs(lml_joint - lml_sum) < 1e-8


def test_bdiag_posterior_equals_scalar_fields():
    nu, U, Y, mask, _ = _setup(seed=1)
    gp = MultiOutputGraphGP(nu, U, kernel="matern2")
    s, b, nv = 1.3, np.array([0.4, 1.5, 0.05]), np.array([0.03, 0.08, 0.005])
    x = gp.x0()
    x[0] = np.log(s); x[1:4] = 0.5 * np.log(b); x[4:7] = 0.0; x[-3:] = np.log(nv)
    M, S = gp.posterior(Y, mask, x)
    for c in range(3):
        # b_c * (s/(s+nu))^2 = sigma_g^2 (kappa^2+nu)^-2 with kappa^2=s, sigma_g^2=b_c s^2
        K = spectral_covariance(U, nu, "matern2", (np.sqrt(b[c]) * s, np.sqrt(s)))
        m, sd = SpectralGaussianField(K).posterior(Y[:, c], nv[c], mask=mask)
        np.testing.assert_allclose(M[:, c], m, atol=1e-8)
        np.testing.assert_allclose(S[:, c], sd, atol=1e-8)


def test_linear_kernel_is_marginalized_linear_mean():
    nu, U, Y, mask, rng = _setup(seed=2)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 4)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X])
    s, b, c_scale, nv = 0.5, np.array([1.0, 1.0, 1.0]), 0.6, np.array([0.05] * 3)
    x = gp.x0()
    x[0] = np.log(s); x[1:4] = 0.5 * np.log(b); x[4:7] = 0.0
    x[7:10] = np.log(c_scale)          # feature scale, all channels
    x[-3:] = np.log(nv)
    lml_joint = gp.log_marginal_likelihood(Y, mask, x)

    # explicit marginalization per channel: y ~ N(0, b K_G + c X X^T + nv I)
    lml_expl = 0.0
    Kg = shape_cov(nu, U, "additive", s)
    obs = np.where(mask)[0]
    for c in range(3):
        C = b[c] * Kg[np.ix_(obs, obs)] + c_scale * X[obs] @ X[obs].T \
            + nv[c] * np.eye(obs.size)
        r = Y[obs, c]
        sign, logdet = np.linalg.slogdet(C)
        lml_expl += -0.5 * (r @ np.linalg.solve(C, r) + logdet
                            + obs.size * np.log(2 * np.pi))
    assert abs(lml_joint - lml_expl) < 1e-8


def test_icm_coupling_transfers_information_across_channels():
    """Three channels are noisy copies of ONE latent field. A near-rank-1
    coregionalization matrix (B ~ all-ones) lets the GP average the three noisy
    copies at observed notes before extrapolating to held-out notes; B = I cannot.
    The coupled posterior must beat the diagonal one on the held-out notes."""
    nu, U, _, _, rng = _setup(seed=3)
    n = nu.size
    g = 1.0 / (1.0 + nu)  # additive shape eigenvalues at s=1
    f = U @ (np.sqrt(g) * rng.standard_normal(n))
    Y = np.stack([f, f, f], axis=1) + 0.05 * rng.standard_normal((n, 3))
    mask = ie.random_mask(n, rng, observed_frac=0.5)

    gp = MultiOutputGraphGP(nu, U, kernel="additive")

    def params(coupled):
        x = gp.x0()
        x[0] = 0.0
        if coupled:
            # Cholesky ~ [e^0; e^-6; e^-6] diag with L21=L31=1, L32=0 -> B ~ all-ones
            x[1:4] = np.array([0.0, -6.0, -6.0])
            x[4:7] = np.array([1.0, 1.0, 0.0])
        else:
            x[1:4] = 0.0
            x[4:7] = 0.0
        x[-3:] = np.log(0.05 ** 2)
        return x

    M_ind, _ = gp.posterior(Y, mask, params(False))
    M_icm, _ = gp.posterior(Y, mask, params(True))
    held = ~mask
    err_ind = np.sqrt(np.mean((M_ind[held] - Y[held]) ** 2))
    err_icm = np.sqrt(np.mean((M_icm[held] - Y[held]) ** 2))
    assert err_icm < err_ind


def test_fit_runs_and_improves_marglik():
    nu, U, Y, mask, _ = _setup(seed=4)
    gp = MultiOutputGraphGP(nu, U, kernel="additive")
    x_hat, info = gp.fit(Y, mask, noise_floor=np.full(3, 1e-4), maxiter=60)
    assert np.isfinite(info["nll"])
    assert info["nll"] <= -gp.log_marginal_likelihood(Y, mask, gp.x0()) + 1e-9
    M, S = gp.posterior(Y, mask, x_hat)
    assert np.all(np.isfinite(M)) and np.all(np.isfinite(S))


def test_heldout_targets_cannot_influence_gp_predictions():
    nu, U, Y, mask, rng = _setup(seed=5)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="matern1", features=[X])
    held = ~mask

    def run(Yin):
        x_hat, info = gp.fit(Yin, mask, noise_floor=np.full(3, 1e-4), maxiter=40)
        M, S = gp.posterior(Yin, mask, x_hat)
        return x_hat, M[held], S[held]

    Y2 = Y.copy()
    Y2[held] = 1e6
    xa, Ma, Sa = run(Y)
    xb, Mb, Sb = run(Y2)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(Ma, Mb)
    np.testing.assert_array_equal(Sa, Sb)


def test_chol_to_B_is_psd_and_roundtrips_diag():
    theta = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2])
    B = chol_to_B(theta)
    assert np.allclose(B, B.T)
    assert np.all(np.linalg.eigvalsh(B) > 0)
    theta_diag = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
    Bd = chol_to_B(theta_diag)
    np.testing.assert_allclose(Bd, np.diag(np.exp(2 * theta_diag[:3])))
