"""EB guard (`fit_laplacian_field_guarded`): screen + fallback ladder.

The real-world validation is the piece-28 diagnostic rerun (the collapse cannot be
reliably reproduced synthetically — it is a knife-edge (mean, mask) event of the
marglik optimizer; see docs/phase1_calibration_results.md). These tests pin the
mechanics deterministically:

  1. healthy fit -> screen passes, guarded == unguarded, hp["guard"] == "marglik";
  2. an impossible guard_factor forces the ladder -> falls back, predictions stay
     within the mean-only error scale (the guard's contract);
  3. the guard threads through impute_methods without changing shapes or the
     shared rng stream (identical-mask A/B safety).

NumPy-only (Phase-1 core).
"""
import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.model import (GraphGaussianField, fit_laplacian_field,
                                fit_laplacian_field_guarded)
from score_bundle.prior import laplacian_precision
from score_bundle.score import Note, Score


def _chain_score(n=40, rng=None):
    rng = rng or np.random.default_rng(0)
    notes = [Note(pitch=60 + int(rng.integers(-5, 6)), onset=float(i) * 0.5,
                  duration=0.5) for i in range(n)]
    return Score(notes)


def _smooth_sample(L, rng, lam=0.5, eta=2.0):
    Q = laplacian_precision(L, lam=lam, eta=eta)
    return GraphGaussianField(Q).sample(rng)


def test_healthy_fit_passes_screen_and_matches_unguarded():
    rng = np.random.default_rng(1)
    score = _chain_score(40, rng)
    L = laplacian(build_adjacency(score))
    y = _smooth_sample(L, rng) + 0.1 * rng.standard_normal(L.shape[0])
    mask = ie.random_mask(len(y), rng, observed_frac=0.6)

    f_g, hp_g = fit_laplacian_field_guarded(L, y, mask=mask, noise_floor=0.01,
                                            rng=np.random.default_rng(0))
    f_u, hp_u = fit_laplacian_field(L, y, mask=mask, noise_floor=0.01)
    assert hp_g["guard"] == "marglik"
    for k in ("lam", "eta", "noise_var"):
        assert hp_g[k] == hp_u[k]
    m_g, _ = f_g.posterior(y, hp_g["noise_var"], mask=mask)
    m_u, _ = f_u.posterior(y, hp_u["noise_var"], mask=mask)
    np.testing.assert_allclose(m_g, m_u)


def test_impossible_factor_triggers_fallback_and_stays_bounded():
    rng = np.random.default_rng(2)
    score = _chain_score(40, rng)
    L = laplacian(build_adjacency(score))
    y = _smooth_sample(L, rng) + 0.1 * rng.standard_normal(L.shape[0])
    mask = ie.random_mask(len(y), rng, observed_frac=0.6)

    # guard_factor so small every graph fit fails the screen -> ladder bottoms out
    field, hp = fit_laplacian_field_guarded(L, y, mask=mask, noise_floor=0.01,
                                            guard_factor=1e-9,
                                            rng=np.random.default_rng(0))
    assert hp["guard"] == "conservative"
    assert hp["eta"] == 0.0
    m, std = field.posterior(y, hp["noise_var"], mask=mask)
    held = ~mask
    # contract: no-coupling fallback predicts ~the prior mean with honest scale
    resid_scale = float(np.std(y[mask]))
    assert np.sqrt(np.mean((y[held] - m[held]) ** 2)) < 3.0 * resid_scale
    assert np.all(np.isfinite(std))


def test_guard_skips_screen_when_too_few_observed():
    rng = np.random.default_rng(3)
    score = _chain_score(6, rng)
    L = laplacian(build_adjacency(score))
    y = rng.standard_normal(6)
    mask = np.array([True, True, True, False, True, False])
    _, hp = fit_laplacian_field_guarded(L, y, mask=mask, guard_factor=1e-9,
                                        rng=np.random.default_rng(0))
    assert hp["guard"] == "marglik"  # 4 observed < 8: screen skipped, marglik kept


def test_impute_methods_guard_preserves_shapes_and_rng_stream():
    rng = np.random.default_rng(4)
    score = _chain_score(30, rng)
    L = laplacian(build_adjacency(score))
    y = np.stack([_smooth_sample(L, rng) for _ in range(3)], axis=1)
    y += 0.1 * rng.standard_normal(y.shape)
    means = {"zero": np.zeros_like(y)}

    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    mask = ie.random_mask(len(y), rng_a, 0.6)
    _ = ie.random_mask(len(y), rng_b, 0.6)
    out_off = ie.impute_methods(score, y, means, mask, fit_hyper=True, rng=rng_a,
                                noise_floor_frac=0.05, guard=False)
    out_on = ie.impute_methods(score, y, means, mask, fit_hyper=True, rng=rng_b,
                               noise_floor_frac=0.05, guard=True)
    for key in out_off:
        assert out_on[key].pred.shape == out_off[key].pred.shape
    # the shared rng streams must stay in lockstep (identical-mask A/B safety)
    assert rng_a.integers(1 << 30) == rng_b.integers(1 << 30)
