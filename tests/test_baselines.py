import numpy as np

from score_bundle.baselines import (
    independent_field,
    ridge_impute,
    score_features,
    temporal_field,
)
from score_bundle.synthetic import make_synthetic, random_mask


def test_temporal_field_runs():
    rng = np.random.default_rng(0)
    data = make_synthetic(rng, n=40)
    field = temporal_field(data.score, lam=0.5, eta=1.0)
    mean, std = field.posterior(data.y_obs, data.noise_var)
    assert mean.shape == (40,) and np.isfinite(mean).all()
    assert (std > 0).all()


def test_independent_field_predicts_prior_mean_for_masked():
    rng = np.random.default_rng(1)
    data = make_synthetic(rng, n=30)
    mask = random_mask(30, rng, observed_frac=0.5)
    field = independent_field(30, prior_var=1.0)
    mean, _ = field.posterior(data.y_obs, data.noise_var, mask=mask)
    # masked nodes have no coupling -> pulled to the (zero) prior mean
    assert np.allclose(mean[~mask], 0.0, atol=1e-8)


def test_ridge_impute_finite():
    rng = np.random.default_rng(2)
    data = make_synthetic(rng, n=50)
    mask = random_mask(50, rng, observed_frac=0.7)
    pred, sigma = ridge_impute(data.score, data.y_obs, mask)
    assert pred.shape == (50,) and np.isfinite(pred).all()
    assert sigma >= 0
    assert score_features(data.score).shape == (50, 4)
