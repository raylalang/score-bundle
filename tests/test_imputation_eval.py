"""Tests for the held-out imputation comparison harness (numpy-only)."""
import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.synthetic import make_synthetic


def test_graph_residual_helps_zero_mean():
    """On data drawn with inter-note coupling, the graph residual must beat no-graph.

    With a zero mean, 'graph off' predicts 0 everywhere; 'graph on' propagates observed
    neighbours, so its held-out RMSE must be lower.
    """
    rng = np.random.default_rng(0)
    d = make_synthetic(rng, n=120, lam=0.4, eta=3.0, noise_var=0.02)
    mask = ie.random_mask(len(d.score), rng, observed_frac=0.6)

    means = {"zero": np.zeros(len(d.score))}
    cells = ie.impute_methods(d.score, d.y_obs, means, mask, fit_hyper=True)

    off = cells[("zero", False)]
    on = cells[("zero", True)]
    rmse_off = np.sqrt(np.mean((off.y - off.pred) ** 2))
    rmse_on = np.sqrt(np.mean((on.y - on.pred) ** 2))
    assert rmse_on < rmse_off
    assert (on.std > 0).all() and (off.std > 0).all()


def test_graph_predictive_std_includes_observation_noise():
    """The held-out predictive std must exceed the latent posterior std by ~the obs noise.

    The fix for the NLL blow-up: a held-out *observation* y = f + e has variance
    diag(Sigma_y) + noise_var, so even at a perfectly-pinned node the interval can't collapse.
    With a known noise_var, the reported std must be >= sqrt(noise_var) everywhere.
    """
    rng = np.random.default_rng(3)
    d = make_synthetic(rng, n=100, lam=0.5, eta=2.0, noise_var=0.05)
    mask = ie.random_mask(len(d.score), rng, observed_frac=0.6)
    means = {"zero": np.zeros(len(d.score))}
    nv = 0.05
    cells = ie.impute_methods(d.score, d.y_obs, means, mask, fit_hyper=False,
                              lam=0.5, eta=2.0, noise_var=nv)
    on = cells[("zero", True)]
    assert (on.std >= np.sqrt(nv) - 1e-9).all()


def test_multichannel_shapes_and_accumulator():
    rng = np.random.default_rng(1)
    d = make_synthetic(rng, n=80, lam=0.5, eta=2.0, noise_var=0.03)
    n = len(d.score)
    # build a 2-channel target and a ridge mean
    Y = np.stack([d.y_obs, d.y_obs * 0.5 + rng.normal(scale=0.1, size=n)], axis=1)
    mask = ie.random_mask(n, rng, observed_frac=0.65)
    means = {
        "zero": np.zeros((n, 2)),
        "ridge": ie.ridge_mean(d.score, Y, mask),
        "LM": Y + rng.normal(scale=0.2, size=(n, 2)),  # stand-in for an out-of-sample mu_LM
    }
    cells = ie.impute_methods(d.score, Y, means, mask, fit_hyper=False, lam=0.5, eta=2.0)

    held = int((~mask).sum())
    for key, cell in cells.items():
        assert cell.y.shape == (held * 2,)  # two channels concatenated
        assert cell.pred.shape == cell.y.shape and cell.std.shape == cell.y.shape
        assert cell.channel.shape == cell.y.shape
        assert set(np.unique(cell.channel)) == {0, 1}
        assert np.isfinite(cell.pred).all() and (cell.std > 0).all()

    acc = ie.MetricAccumulator()
    acc.add(cells)
    acc.add(cells)  # pooling two "pieces"
    rep = acc.report(level=0.9)
    assert ("LM", True) in rep and "rmse" in rep[("LM", True)]
    table = ie.format_report(rep)
    assert "mean source" in table and "LM" in table

    # per-channel breakdown
    chan = acc.report_by_channel(["c0", "c1"], level=0.9)
    assert ("LM", True, "c0") in chan and ("LM", True, "c1") in chan
    chan_table = ie.format_report_by_channel(chan, ["c0", "c1"])
    assert "[c0]" in chan_table and "[c1]" in chan_table


def test_ridge_mean_is_oracle_better_than_zero():
    rng = np.random.default_rng(2)
    d = make_synthetic(rng, n=100, lam=0.5, eta=2.0, noise_var=0.02)
    mask = ie.random_mask(len(d.score), rng, observed_frac=0.6)
    # an (oracle) informative mean should beat the zero mean without the graph
    means = {"zero": np.zeros(len(d.score)), "oracle": d.y_true}
    cells = ie.impute_methods(d.score, d.y_obs, means, mask, fit_hyper=False)
    zero = cells[("zero", False)]
    oracle = cells[("oracle", False)]
    assert np.sqrt(np.mean((oracle.y - oracle.pred) ** 2)) < np.sqrt(np.mean((zero.y - zero.pred) ** 2))
