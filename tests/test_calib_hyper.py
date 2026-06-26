"""Calibration-split hyperparameter selection (numpy-only).

``fit_laplacian_field_calib`` picks (lam, eta, noise_var) by held-out NLL on a calibration
subset of the observed nodes, rather than the in-sample marginal likelihood.  These tests
check it runs, returns a usable field, degrades gracefully on tiny inputs, and is selectable
through ``impute_methods`` as an extra graph variant.
"""
import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.model import fit_laplacian_field_calib
from score_bundle.synthetic import make_synthetic


def test_calib_fit_runs_and_predicts():
    rng = np.random.default_rng(0)
    d = make_synthetic(rng, n=120, lam=0.4, eta=3.0, noise_var=0.04)
    L = laplacian(build_adjacency(d.score))
    mask = ie.random_mask(len(d.score), rng, observed_frac=0.6)
    field, hp = fit_laplacian_field_calib(L, d.y_obs, mask=mask, mean=0.0, rng=rng)
    assert hp["lam"] > 0 and hp["eta"] > 0 and hp["noise_var"] > 0
    m, std = field.posterior(d.y_obs, hp["noise_var"], mask=mask)
    assert np.isfinite(m).all() and (std > 0).all()


def test_calib_falls_back_on_tiny_observed_set():
    rng = np.random.default_rng(1)
    d = make_synthetic(rng, n=20, lam=0.5, eta=2.0, noise_var=0.05)
    L = laplacian(build_adjacency(d.score))
    mask = np.zeros(len(d.score), dtype=bool)
    mask[:3] = True  # too few to split -> must fall back, not crash
    field, hp = fit_laplacian_field_calib(L, d.y_obs, mask=mask, mean=0.0, rng=rng)
    assert hp["noise_var"] > 0


def test_calib_variant_selectable_in_impute_methods():
    rng = np.random.default_rng(2)
    d = make_synthetic(rng, n=100, lam=0.5, eta=2.0, noise_var=0.04)
    mask = ie.random_mask(len(d.score), rng, observed_frac=0.6)
    variants = [(False, False, "marglik"), ("on", True, "marglik"), ("calib", True, "calib")]
    cells = ie.impute_methods(
        d.score, d.y_obs, {"zero": np.zeros(len(d.score))}, mask,
        fit_hyper=True, graph_variants=variants, rng=rng,
    )
    assert ("zero", "calib") in cells and ("zero", "on") in cells
    calib = cells[("zero", "calib")]
    assert np.isfinite(calib.pred).all() and (calib.std > 0).all()
