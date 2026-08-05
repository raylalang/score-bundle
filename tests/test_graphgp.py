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


def test_loo_predictive_matches_bruteforce_conditional():
    nu, U, Y, _, _ = _setup(seed=6, n=12)
    gp = MultiOutputGraphGP(nu, U, kernel="additive")
    x = gp.x0(); x[0] = 0.2; x[1:4] = 0.1; x[4:7] = 0.3; x[-3:] = np.log(0.05)
    loo_m, loo_v = gp.loo_predictive(Y, x)
    # brute force: for each (note, channel) index, condition on all the others
    p = gp.unpack(x)
    allidx = np.arange(gp.N)
    C = gp._blocks(p, allidx, allidx)
    for c in range(3):
        C[c * gp.N:(c + 1) * gp.N, c * gp.N:(c + 1) * gp.N] += p["noise"][c] * np.eye(gp.N)
    y = np.concatenate([Y[:, c] for c in range(3)])
    n = y.size
    for i in (0, 7, 20, n - 1):
        rest = np.array([j for j in range(n) if j != i])
        Cro = C[np.ix_(rest, rest)]
        m_i = C[i, rest] @ np.linalg.solve(Cro, y[rest])
        v_i = C[i, i] - C[i, rest] @ np.linalg.solve(Cro, C[rest, i])
        ci, ni = divmod(i, gp.N)
        assert abs(loo_m[ni, ci] - m_i) < 1e-8
        assert abs(loo_v[ni, ci] - v_i) < 1e-8


def test_fit_with_fixed_noise_pins_noise_exactly():
    nu, U, Y, mask, _ = _setup(seed=7)
    gp = MultiOutputGraphGP(nu, U, kernel="additive")
    nf = np.array([0.04, 0.09, 0.01])
    x_hat, info = gp.fit(Y, mask, noise_fixed=nf, maxiter=40)
    np.testing.assert_allclose(gp.unpack(x_hat)["noise"], nf, rtol=1e-12)


def test_fit_guarded_healthy_is_noop():
    nu, U, Y, mask, _ = _setup(seed=8)
    gp = MultiOutputGraphGP(nu, U, kernel="additive")
    floor = np.full(3, 1e-4)
    x_g, info_g = gp.fit_guarded(Y, mask, noise_floor=floor, maxiter=60,
                                 rng=np.random.default_rng(0))
    x_u, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=60)
    assert info_g["guard"] == "marglik"
    np.testing.assert_array_equal(x_g, x_u)


def test_fit_guarded_impossible_screen_bottoms_out_bounded():
    nu, U, Y, mask, rng = _setup(seed=9)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="matern2", features=[X])
    x_c, info = gp.fit_guarded(Y, mask, noise_floor=np.full(3, 1e-4),
                               guard_factor=1e-9, nll_margin=-1e9,
                               maxiter=40, rng=np.random.default_rng(0))
    assert info["guard"] == "conservative"
    M, S = gp.posterior(Y, mask, x_c)
    held = ~mask
    # conservative contract: ~mean prediction with honest scale, decoupled graph
    for c in range(3):
        scale = float(np.std(Y[mask, c]))
        assert np.sqrt(np.mean((Y[held, c] - M[held, c]) ** 2)) < 3.0 * scale
        assert np.all(np.isfinite(S[:, c]))
    # decoupled: posterior at held-out notes stays ~the prior (no coupling pull)
    assert float(np.max(np.abs(M[held]))) < 3.0 * float(np.max(np.abs(Y[mask].mean(axis=0)))) + 1e-6


def test_posterior_components_sum_and_match_explicit_construction():
    """m splits exactly into graph + per-feature terms, each equal to the
    explicitly constructed E[f_c | y_obs] = K_ao^(c) (K_oo + noise)^-1 y."""
    nu, U, Y, mask, rng = _setup(seed=10)
    n = nu.size
    X1 = np.concatenate([rng.standard_normal((n, 4)), np.ones((n, 1))], axis=1)
    X2 = rng.standard_normal((n, 2))
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X1, X2])
    x_hat, _ = gp.fit(Y, mask, noise_floor=np.full(3, 1e-4), maxiter=40)
    M, _ = gp.posterior(Y, mask, x_hat)
    comps = gp.posterior_components(Y, mask, x_hat)
    assert set(comps) == {"graph", "feat_0", "feat_1", "total"}
    np.testing.assert_allclose(comps["total"], M, atol=1e-8)
    np.testing.assert_allclose(
        comps["graph"] + comps["feat_0"] + comps["feat_1"], M, atol=1e-8)

    # independent brute-force construction (no _blocks / only involved)
    p = gp.unpack(x_hat)
    obs = np.where(mask)[0]
    n_o = obs.size
    Kg = shape_cov(nu, U, "additive", p["s"])
    K_oo = np.zeros((3 * n_o, 3 * n_o))
    for a in range(3):
        for b in range(3):
            blk = p["B"][a, b] * Kg[np.ix_(obs, obs)]
            if a == b:
                blk = blk + p["feature_scales"][0][a] * X1[obs] @ X1[obs].T \
                    + p["feature_scales"][1][a] * X2[obs] @ X2[obs].T \
                    + p["noise"][a] * np.eye(n_o)
            K_oo[a * n_o:(a + 1) * n_o, b * n_o:(b + 1) * n_o] = blk
    y = np.concatenate([Y[obs, c] for c in range(3)])
    w = np.linalg.solve(K_oo, y)
    graph_ao = np.zeros((3 * n, 3 * n_o))
    for a in range(3):
        for b in range(3):
            graph_ao[a * n:(a + 1) * n, b * n_o:(b + 1) * n_o] = \
                p["B"][a, b] * Kg[:, obs]
    np.testing.assert_allclose(comps["graph"],
                               (graph_ao @ w).reshape(3, n).T, atol=1e-8)
    for f_idx, (name, X) in enumerate((("feat_0", X1), ("feat_1", X2))):
        feat_ao = np.zeros((3 * n, 3 * n_o))
        for a in range(3):
            feat_ao[a * n:(a + 1) * n, a * n_o:(a + 1) * n_o] = \
                p["feature_scales"][f_idx][a] * X @ X[obs].T
        np.testing.assert_allclose(comps[name],
                                   (feat_ao @ w).reshape(3, n).T, atol=1e-8)


def test_posterior_components_cell_mask_sum():
    nu, U, Y, _, rng = _setup(seed=11)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="matern1", features=[X])
    mask2d = rng.random((n, 3)) < 0.6
    x = gp.x0()
    x[0] = 0.1; x[1:4] = 0.2; x[4:7] = 0.1
    x[7:10] = np.log(0.5); x[-3:] = np.log(0.05)
    M, _ = gp.posterior(Y, mask2d, x)
    comps = gp.posterior_components(Y, mask2d, x)
    np.testing.assert_allclose(comps["total"], M, atol=1e-8)
    np.testing.assert_allclose(comps["graph"] + comps["feat_0"], M, atol=1e-8)


def test_posterior_components_zero_leak():
    nu, U, Y, mask, rng = _setup(seed=12)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X])
    x = gp.x0()
    x[0] = 0.3; x[1:4] = 0.1; x[7:10] = np.log(0.4); x[-3:] = np.log(0.05)
    held = ~mask
    Y2 = Y.copy()
    Y2[held] = 1e6
    ca = gp.posterior_components(Y, mask, x)
    cb = gp.posterior_components(Y2, mask, x)
    for name in ca:
        np.testing.assert_array_equal(ca[name], cb[name])


def test_posterior_component_cov_matches_bruteforce():
    """Component variances/cross-covariances equal explicit Gaussian
    conditioning, and they sum to the latent posterior variance exactly."""
    nu, U, Y, mask, rng = _setup(seed=13)
    n = nu.size
    X1 = np.concatenate([rng.standard_normal((n, 4)), np.ones((n, 1))], axis=1)
    X2 = rng.standard_normal((n, 2))
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X1, X2])
    x_hat, _ = gp.fit(Y, mask, noise_floor=np.full(3, 1e-4), maxiter=40)
    cov = gp.posterior_component_cov(Y, mask, x_hat)
    _, S = gp.posterior(Y, mask, x_hat)

    # total-variance identity
    tot = cov["var_graph"] + cov["var_feat_0"] + cov["var_feat_1"] \
        + 2 * (cov["cov_graph_feat_0"] + cov["cov_graph_feat_1"]
               + cov["cov_feat_0_feat_1"])
    np.testing.assert_allclose(tot, S ** 2, atol=1e-8)

    # brute force: explicit per-component covariances (independent of _blocks)
    p = gp.unpack(x_hat)
    obs = np.where(mask)[0]
    n_o = obs.size
    Kg = shape_cov(nu, U, "additive", p["s"])

    def comp_mats(sel):
        K_aa = np.zeros((3 * n, 3 * n))
        for a in range(3):
            for b in range(3):
                blk = np.zeros((n, n))
                if sel == "graph":
                    blk = p["B"][a, b] * Kg
                elif a == b:
                    f, X = sel
                    blk = p["feature_scales"][f][a] * X @ X.T
                K_aa[a * n:(a + 1) * n, b * n:(b + 1) * n] = blk
        return K_aa

    mats = {"graph": comp_mats("graph"), "feat_0": comp_mats((0, X1)),
            "feat_1": comp_mats((1, X2))}
    idx_o = np.concatenate([c * n + obs for c in range(3)])
    K_oo = sum(mats.values())[np.ix_(idx_o, idx_o)]
    for c in range(3):
        K_oo[c * n_o:(c + 1) * n_o, c * n_o:(c + 1) * n_o] += \
            p["noise"][c] * np.eye(n_o)
    Kinv = np.linalg.inv(K_oo)
    for a in ("graph", "feat_0", "feat_1"):
        for b in ("graph", "feat_0", "feat_1"):
            expected = float(a == b) * mats[a] \
                - mats[a][:, idx_o] @ Kinv @ mats[b][idx_o, :]
            d = np.diag(expected)
            got = cov[f"var_{a}"] if a == b else cov.get(
                f"cov_{a}_{b}", cov.get(f"cov_{b}_{a}"))
            np.testing.assert_allclose(got, d.reshape(3, n).T, atol=1e-8)


def test_posterior_component_cov_cell_mask_identity():
    nu, U, Y, _, rng = _setup(seed=14)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="matern1", features=[X])
    mask2d = rng.random((n, 3)) < 0.6
    x = gp.x0()
    x[0] = 0.1; x[1:4] = 0.2; x[4:7] = 0.1
    x[7:10] = np.log(0.5); x[-3:] = np.log(0.05)
    _, S = gp.posterior(Y, mask2d, x)
    cov = gp.posterior_component_cov(Y, mask2d, x)
    tot = cov["var_graph"] + cov["var_feat_0"] + 2 * cov["cov_graph_feat_0"]
    np.testing.assert_allclose(tot, S ** 2, atol=1e-8)


def test_fit_b_diagonal_pins_offdiagonals_to_zero():
    nu, U, Y, mask, rng = _setup(seed=15)
    n = nu.size
    X = np.concatenate([rng.standard_normal((n, 3)), np.ones((n, 1))], axis=1)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=[X])
    x_hat, _ = gp.fit(Y, mask, noise_floor=np.full(3, 1e-4), maxiter=40,
                      b_diagonal=True)
    B = gp.unpack(x_hat)["B"]
    np.testing.assert_array_equal(B - np.diag(np.diag(B)), np.zeros((3, 3)))
    assert np.all(np.diag(B) > 0)


def test_chol_to_B_is_psd_and_roundtrips_diag():
    theta = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2])
    B = chol_to_B(theta)
    assert np.allclose(B, B.T)
    assert np.all(np.linalg.eigvalsh(B) > 0)
    theta_diag = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
    Bd = chol_to_B(theta_diag)
    np.testing.assert_allclose(Bd, np.diag(np.exp(2 * theta_diag[:3])))
