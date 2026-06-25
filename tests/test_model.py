import numpy as np

from score_bundle import GraphGaussianField, fit_laplacian_field, laplacian_precision
from score_bundle.baselines import independent_field
from score_bundle.metrics import rmse
from score_bundle.synthetic import make_synthetic, random_mask


def test_posterior_recovers_latents_full_observation():
    rng = np.random.default_rng(0)
    data = make_synthetic(rng, n=60, noise_var=0.02)
    mean, std = data.field.posterior(data.y_obs, data.noise_var)
    # with low noise and the true prior, posterior mean tracks the truth
    assert rmse(data.y_true, mean) < 0.2
    assert (std > 0).all()


def test_graph_prior_beats_independent_on_imputation():
    rng = np.random.default_rng(1)
    data = make_synthetic(rng, n=80, lam=0.4, eta=3.0, noise_var=0.02)
    mask = random_mask(80, rng, observed_frac=0.6)
    held = ~mask

    g_mean, _ = data.field.posterior(data.y_obs, data.noise_var, mask=mask)
    indep = independent_field(80, prior_var=float(np.var(data.y_obs[mask])))
    i_mean, _ = indep.posterior(data.y_obs, data.noise_var, mask=mask)

    graph_err = rmse(data.y_true[held], g_mean[held])
    indep_err = rmse(data.y_true[held], i_mean[held])
    assert graph_err < indep_err  # the core hypothesis, by construction


def test_marginal_likelihood_finite_and_learnable():
    rng = np.random.default_rng(2)
    data = make_synthetic(rng, n=50, noise_var=0.05)
    Q0 = laplacian_precision(data.L, lam=1.0, eta=1.0)
    base = GraphGaussianField(Q0)
    lml0 = base.log_marginal_likelihood(data.y_obs, 0.05)
    assert np.isfinite(lml0)

    fitted, hp = fit_laplacian_field(data.L, data.y_obs)
    lml1 = fitted.log_marginal_likelihood(data.y_obs, hp["noise_var"])
    # empirical-Bayes fit should not be worse than the arbitrary starting point
    assert lml1 >= lml0 - 1e-6
    assert hp["lam"] > 0 and hp["eta"] > 0 and hp["noise_var"] > 0
