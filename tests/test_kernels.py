"""Spectral kernel framework (kernel comparison, docs/kernel_comparison_experiment.md).

Pins the mechanics the sweep relies on:

  1. the spectral additive kernel is the *same model* as the precision-form additive
     path (posterior mean/std and marginal likelihood agree to numerical precision);
  2. every registered kernel yields a finite, PSD covariance and a working posterior,
     including the diffusion kernel at extreme t (whose precision form overflows);
  3. the spectral EB fit and its guard mirror the additive ladder (healthy fit ->
     "marglik" and identical to unguarded; impossible screen -> bounded conservative
     fallback);
  4. the music-theory adjacencies order pitch relations tonally (octave/fifth close,
     semitone far) and place chord/voice-leading edges only where defined.

NumPy-only (Phase-1 core).
"""
import numpy as np
import pytest

from score_bundle import imputation_eval as ie
from score_bundle.graph import (build_adjacency, build_adjacency_harmonic,
                                build_adjacency_tonal, fifths_distance, laplacian)
from score_bundle.model import (GraphGaussianField, SpectralGaussianField,
                                fit_spectral_field, fit_spectral_field_guarded)
from score_bundle.prior import (SPECTRAL_KERNELS, laplacian_precision,
                                spectral_cov_eigs, spectral_covariance)
from score_bundle.score import Note, Score


def _score(n=30, rng=None):
    rng = rng or np.random.default_rng(0)
    notes = [Note(pitch=60 + int(rng.integers(-7, 8)), onset=float(i) * 0.5,
                  duration=0.5) for i in range(n)]
    return Score(notes)


def _setup(n=30, seed=0):
    rng = np.random.default_rng(seed)
    score = _score(n, rng)
    L = laplacian(build_adjacency(score))
    Q = laplacian_precision(L, lam=0.7, eta=1.5)
    y = GraphGaussianField(Q).sample(rng) + 0.1 * rng.standard_normal(n)
    mask = ie.random_mask(n, rng, observed_frac=0.6)
    return score, L, y, mask


# --------------------------------------------------------------------------- equivalence
def test_spectral_additive_matches_precision_form():
    _, L, y, mask = _setup()
    lam, eta, nv = 0.7, 1.5, 0.05
    nu, U = np.linalg.eigh(L)

    f_prec = GraphGaussianField(laplacian_precision(L, lam=lam, eta=eta))
    f_spec = SpectralGaussianField(spectral_covariance(U, nu, "additive", (lam, eta)))

    m_p, s_p = f_prec.posterior(y, nv, mask=mask)
    m_s, s_s = f_spec.posterior(y, nv, mask=mask)
    np.testing.assert_allclose(m_s, m_p, atol=1e-8)
    np.testing.assert_allclose(s_s, s_p, atol=1e-8)

    lml_p = f_prec.log_marginal_likelihood(y, nv, mask=mask)
    lml_s = f_spec.log_marginal_likelihood(y, nv, mask=mask)
    assert abs(lml_p - lml_s) < 1e-6


def test_independent_kernel_is_diagonal():
    _, L, _, _ = _setup()
    nu, U = np.linalg.eigh(L)
    K = spectral_covariance(U, nu, "independent", (2.0,))
    np.testing.assert_allclose(K, 0.5 * np.eye(L.shape[0]), atol=1e-10)


# --------------------------------------------------------------------------- registry
@pytest.mark.parametrize("kernel", sorted(SPECTRAL_KERNELS))
def test_every_kernel_gives_psd_covariance_and_finite_posterior(kernel):
    _, L, y, mask = _setup()
    nu, U = np.linalg.eigh(L)
    params = np.exp(np.asarray(SPECTRAL_KERNELS[kernel].x0))
    g = spectral_cov_eigs(nu, kernel, params)
    assert np.all(np.isfinite(g)) and np.all(g > 0)
    K = spectral_covariance(U, nu, kernel, params)
    field = SpectralGaussianField(K)
    m, s = field.posterior(y, 0.05, mask=mask)
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(s))


def test_diffusion_extreme_t_stays_finite():
    """The diffusion *precision* exp(t nu) overflows at large t; the covariance
    form must stay finite and the posterior must fall back to ~the prior mean."""
    _, L, y, mask = _setup()
    nu, U = np.linalg.eigh(L)
    K = spectral_covariance(U, nu, "diffusion", (1.0, 1e4))
    field = SpectralGaussianField(K, mean=0.0)
    m, s = field.posterior(y, 0.05, mask=mask)
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(s))
    lml = field.log_marginal_likelihood(y, 0.05, mask=mask)
    assert np.isfinite(lml)


# --------------------------------------------------------------------------- fits
def test_fit_spectral_field_additive_recovers_reasonable_field():
    _, L, y, mask = _setup(seed=3)
    field, hp = fit_spectral_field(L, y, kernel="additive", mask=mask,
                                   noise_floor=0.01)
    assert hp["kernel"] == "additive"
    assert set(("lam", "eta", "noise_var")) <= set(hp)
    assert hp["noise_var"] >= 0.01
    m, s = field.posterior(y, hp["noise_var"], mask=mask)
    held = ~mask
    # the graph fit must beat predicting zero on data sampled with coupling
    assert np.sqrt(np.mean((y[held] - m[held]) ** 2)) < np.sqrt(np.mean(y[held] ** 2))


def test_fit_spectral_field_reuses_precomputed_eig():
    _, L, y, mask = _setup(seed=4)
    eig = np.linalg.eigh(L)
    f1, hp1 = fit_spectral_field(L, y, kernel="matern2", mask=mask)
    f2, hp2 = fit_spectral_field(None, y, kernel="matern2", mask=mask, eig=eig)
    assert hp1 == hp2
    np.testing.assert_allclose(f1.K, f2.K)


def test_guarded_healthy_matches_unguarded():
    _, L, y, mask = _setup(seed=5)
    f_g, hp_g = fit_spectral_field_guarded(L, y, kernel="matern1", mask=mask,
                                           noise_floor=0.01,
                                           rng=np.random.default_rng(0))
    f_u, hp_u = fit_spectral_field(L, y, kernel="matern1", mask=mask,
                                   noise_floor=0.01)
    assert hp_g["guard"] == "marglik"
    for k in ("sigma_g", "kappa", "noise_var"):
        assert hp_g[k] == hp_u[k]
    m_g, _ = f_g.posterior(y, hp_g["noise_var"], mask=mask)
    m_u, _ = f_u.posterior(y, hp_u["noise_var"], mask=mask)
    np.testing.assert_allclose(m_g, m_u)


def test_guarded_impossible_factor_falls_back_conservative():
    _, L, y, mask = _setup(seed=6)
    field, hp = fit_spectral_field_guarded(L, y, kernel="diffusion", mask=mask,
                                           noise_floor=0.01, guard_factor=1e-9,
                                           rng=np.random.default_rng(0))
    assert hp["guard"] == "conservative"
    m, std = field.posterior(y, hp["noise_var"], mask=mask)
    held = ~mask
    resid_scale = float(np.std(y[mask]))
    assert np.sqrt(np.mean((y[held] - m[held]) ** 2)) < 3.0 * resid_scale
    assert np.all(np.isfinite(std))


@pytest.mark.parametrize("kernel", ["additive", "matern2", "diffusion"])
def test_heldout_targets_cannot_influence_predictions(kernel):
    """Zero-leak contract of the kernel-sweep cell: corrupting y at HELD-OUT nodes
    (arbitrarily large values) must leave the EB fit, the guard path, the posterior
    at held-out nodes, and the predictive std bitwise unchanged — mirroring the
    per-cell block of scripts/eval_kernels.py stage_run (noise floor from observed
    residuals only, guarded fit, predictive-variance floor)."""
    _, L, y, mask = _setup(seed=7)
    mean = 0.1 * np.ones_like(y)
    eig = np.linalg.eigh(L)
    held = ~mask

    def cell(y_in):
        floor = 0.05 * float(np.var((y_in - mean)[mask]))
        field, hp = fit_spectral_field_guarded(
            None, y_in, kernel=kernel, mask=mask, mean=mean, noise_floor=floor,
            rng=np.random.default_rng(0), eig=eig)
        m, std = field.posterior(y_in, hp["noise_var"], mask=mask)
        return m[held], np.sqrt(std[held] ** 2 + hp["noise_var"]), hp

    y_corrupt = y.copy()
    y_corrupt[held] = 1e6 * np.arange(1, held.sum() + 1)
    pred_a, std_a, hp_a = cell(y)
    pred_b, std_b, hp_b = cell(y_corrupt)
    assert hp_a == hp_b
    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(std_a, std_b)


# --------------------------------------------------------------------------- music-theory graphs
def test_fifths_distance_orders_intervals_tonally():
    # unison 0, perfect fifth 1, major second 2, semitone 5, tritone 6
    assert fifths_distance(60, 60) == 0
    assert fifths_distance(60, 67) == 1
    assert fifths_distance(60, 62) == 2
    assert fifths_distance(60, 61) == 5
    assert fifths_distance(60, 66) == 6


def test_tonal_adjacency_octave_and_fifth_closer_than_semitone():
    # three simultaneous pairs, same onset spacing: octave, fifth, semitone
    notes = [Note(60, 0.0, 1.0), Note(72, 0.0, 1.0),   # octave
             Note(60, 4.0, 1.0), Note(67, 4.0, 1.0),   # fifth
             Note(60, 8.0, 1.0), Note(61, 8.0, 1.0)]   # semitone
    W = build_adjacency_tonal(Score(notes))
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
    w_oct, w_fifth, w_semi = W[0, 1], W[2, 3], W[4, 5]
    assert w_oct > w_semi and w_fifth > w_semi
    # the plain combinatorial graph orders them the other way (semitone closest)
    W0 = build_adjacency(Score(notes))
    assert W0[4, 5] > W0[0, 1] and W0[4, 5] > W0[2, 3]


def test_harmonic_adjacency_edge_families():
    notes = [Note(60, 0.0, 1.0), Note(64, 0.0, 1.0),   # chord (same onset)
             Note(65, 1.0, 1.0),                        # step of +1 from note 1
             Note(80, 1.0, 1.0)]                        # leap (no vl edge)
    score = Score(notes)
    base = build_adjacency(score)
    W_chord = build_adjacency_harmonic(score, chord_weight=1.0, vl_weight=0.0)
    assert np.allclose(W_chord, W_chord.T) and np.all(np.diag(W_chord) == 0)
    # chord bonus only on the same-onset pair
    assert np.isclose(W_chord[0, 1] - base[0, 1], 1.0)
    assert np.isclose(W_chord[1, 2], base[1, 2])
    # voice-leading bonus only on the stepwise, different-onset pair
    W_vl = build_adjacency_harmonic(score, chord_weight=0.0, vl_weight=1.0)
    assert np.isclose(W_vl[1, 2] - base[1, 2], 1.0)
    assert np.isclose(W_vl[1, 3], base[1, 3])       # leap: no bonus
    assert np.isclose(W_vl[0, 1], base[0, 1])       # same onset: no vl bonus
    # both weights zero reduces exactly to the combinatorial base
    np.testing.assert_allclose(
        build_adjacency_harmonic(score, chord_weight=0.0, vl_weight=0.0), base
    )
