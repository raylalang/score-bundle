"""Tests for the downstream tasks (completion masks, anomaly detection, denoising)."""
import numpy as np
import pytest

from score_bundle.downstream import (
    anomaly_scores,
    auroc,
    average_precision,
    block_mask,
    denoise_channel,
    independent_denoise,
    inject_anomalies,
    loo_predictive,
    prefix_mask,
)
from score_bundle.metrics import coverage, rmse
from score_bundle.model import GraphGaussianField
from score_bundle.synthetic import make_synthetic


# --------------------------------------------------------------------------- masks
def test_prefix_mask_shape_and_contiguity():
    m = prefix_mask(10, 0.3)
    assert m.sum() == 3 and m[:3].all() and not m[3:].any()
    # degenerate fractions still leave >=1 observed and >=1 held out
    assert prefix_mask(10, 0.0).sum() == 1
    assert prefix_mask(10, 1.0).sum() == 9


def test_block_mask_holds_out_one_contiguous_block():
    rng = np.random.default_rng(0)
    for frac in (0.2, 0.6, 0.9):
        m = block_mask(50, rng, observed_frac=frac)
        held = np.where(~m)[0]
        assert held.size == round((1 - frac) * 50)
        assert (np.diff(held) == 1).all()  # contiguous
        assert 0 < m.sum() < 50


# --------------------------------------------------------------------------- anomaly
def test_loo_predictive_matches_brute_force_mask_out():
    rng = np.random.default_rng(1)
    ds = make_synthetic(rng, n=40)
    field = GraphGaussianField(ds.field.Q)
    loo_mean, loo_var = loo_predictive(field, ds.y_obs, ds.noise_var)
    for i in (0, 7, 39):
        mask = np.ones(40, dtype=bool)
        mask[i] = False
        m, std = field.posterior(ds.y_obs, ds.noise_var, mask=mask)
        # predictive of the held-out *observation* = latent variance + noise
        np.testing.assert_allclose(loo_mean[i], m[i], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(loo_var[i], std[i] ** 2 + ds.noise_var, rtol=1e-8)


def test_auroc_and_ap_sanity():
    labels = np.array([0, 0, 0, 1, 1], dtype=bool)
    perfect = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    inverted = -perfect
    assert auroc(labels, perfect) == 1.0
    assert auroc(labels, inverted) == 0.0
    assert average_precision(labels, perfect) == 1.0
    # all-tied scores -> chance
    assert auroc(labels, np.zeros(5)) == pytest.approx(0.5)


def test_inject_anomalies_labels_and_magnitude():
    rng = np.random.default_rng(2)
    y = rng.normal(size=100)
    y_bad, labels = inject_anomalies(y, rng, frac=0.1, scale=3.0)
    assert labels.sum() == 10
    assert np.allclose(y_bad[~labels], y[~labels])
    assert np.all(np.abs(y_bad[labels] - y[labels]) > 2.0 * np.std(y))


def test_graph_loo_beats_unstructured_zscore_on_coupled_data():
    """On data drawn with inter-note coupling, the structured LOO surprise must
    rank injected errors above clean notes better than the residual z-score."""
    rng = np.random.default_rng(3)
    g_scores, z_scores = [], []
    for _ in range(5):
        ds = make_synthetic(rng, n=60, eta=3.0, noise_var=0.02)
        y_bad, labels = inject_anomalies(ds.y_obs, rng, frac=0.08, scale=3.0)
        mean = np.zeros(60)
        g = anomaly_scores(ds.L, y_bad, mean, use_graph=True)
        z = anomaly_scores(ds.L, y_bad, mean, use_graph=False)
        g_scores.append(auroc(labels, g))
        z_scores.append(auroc(labels, z))
    assert np.mean(g_scores) > np.mean(z_scores)
    assert np.mean(g_scores) > 0.85


# --------------------------------------------------------------------------- denoise
def test_independent_denoise_is_wiener_shrinkage():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200) * 2.0
    nv = 0.5
    pred, std = independent_denoise(y, np.zeros(200), nv)
    prior_var = max(np.var(y) - nv, 1e-8)
    w = prior_var / (prior_var + nv)
    np.testing.assert_allclose(pred, w * y)
    np.testing.assert_allclose(std, np.sqrt(w * nv))


def test_graph_denoiser_beats_identity_and_does_not_collapse():
    """Blind graph denoising: better RMSE than the raw observations, and the
    noise-floored EB fit must not collapse the latent intervals to nothing
    (without the floor, coverage hits 0.0 on some seeds).  Blind noise estimation
    on short pieces genuinely under-covers somewhat; exact calibration is asserted
    for the oracle-noise variant below."""
    rng = np.random.default_rng(5)
    r_id, r_graph, covs = [], [], []
    for _ in range(4):
        ds = make_synthetic(rng, n=60, eta=3.0, noise_var=0.0)  # y_obs == y_true
        noise_std = 0.5 * float(np.std(ds.y_true))
        y_noisy = ds.y_true + rng.normal(scale=noise_std, size=60)
        mean = np.zeros(60)
        pred_i, _ = denoise_channel(ds.L, y_noisy, mean, noise_std, "identity")
        pred_g, std_g = denoise_channel(ds.L, y_noisy, mean, noise_std, "graph")
        r_id.append(rmse(ds.y_true, pred_i))
        r_graph.append(rmse(ds.y_true, pred_g))
        covs.append(coverage(ds.y_true, pred_g, std_g, level=0.9))
    assert np.mean(r_graph) < np.mean(r_id)          # shrinkage recovers signal
    assert min(covs) > 0.3 and np.mean(covs) > 0.6   # floored fit: no interval collapse


def test_graph_oracle_noise_variant_is_calibrated():
    rng = np.random.default_rng(5)
    r_id, r_or, covs = [], [], []
    for _ in range(4):
        ds = make_synthetic(rng, n=60, eta=3.0, noise_var=0.0)
        noise_std = 0.5 * float(np.std(ds.y_true))
        y_noisy = ds.y_true + rng.normal(scale=noise_std, size=60)
        pred, std = denoise_channel(ds.L, y_noisy, np.zeros(60), noise_std, "graph-oracle")
        r_id.append(rmse(ds.y_true, y_noisy))
        r_or.append(rmse(ds.y_true, pred))
        covs.append(coverage(ds.y_true, pred, std, level=0.9))
        assert np.all(std > 0)
    assert np.mean(r_or) < np.mean(r_id)
    assert 0.80 <= np.mean(covs) <= 0.99             # known noise -> calibrated intervals


def test_denoise_graph_calib_variant_runs():
    rng = np.random.default_rng(8)
    ds = make_synthetic(rng, n=40, eta=3.0, noise_var=0.0)
    noise_std = 0.5 * float(np.std(ds.y_true))
    y_noisy = ds.y_true + rng.normal(scale=noise_std, size=40)
    pred, std = denoise_channel(ds.L, y_noisy, np.zeros(40), noise_std, "graph-calib")
    assert pred.shape == (40,) and np.all(std >= 0)


def test_denoise_unknown_method_raises():
    with pytest.raises(ValueError):
        denoise_channel(np.eye(3), np.zeros(3), np.zeros(3), 0.1, "nope")


def test_std_rescale_factor_restores_coverage():
    from score_bundle.metrics import std_rescale_factor

    rng = np.random.default_rng(9)
    y = rng.normal(size=5000)
    mean = np.zeros(5000)
    overconfident = np.full(5000, 0.5)  # true std is 1.0
    s = std_rescale_factor(y, mean, overconfident, level=0.9)
    assert s == pytest.approx(2.0, rel=0.05)
    assert coverage(y, mean, s * overconfident, level=0.9) == pytest.approx(0.9, abs=0.01)
    # already-calibrated stds are left (nearly) alone
    s1 = std_rescale_factor(y, mean, np.ones(5000), level=0.9)
    assert s1 == pytest.approx(1.0, rel=0.05)


def test_fit_laplacian_field_respects_noise_floor():
    from score_bundle.model import fit_laplacian_field

    rng = np.random.default_rng(7)
    # smooth noiseless data: the unconstrained EB fit is free to send noise_var -> 0
    ds = make_synthetic(rng, n=40, eta=3.0, noise_var=0.0)
    floor = 0.05 * float(np.var(ds.y_true))
    _, hp_floor = fit_laplacian_field(ds.L, ds.y_true, noise_floor=floor)
    assert hp_floor["noise_var"] >= floor * (1 - 1e-12)
    # default (floor=0) is unchanged behaviour
    _, hp_free = fit_laplacian_field(ds.L, ds.y_true)
    assert hp_free["noise_var"] > 0
