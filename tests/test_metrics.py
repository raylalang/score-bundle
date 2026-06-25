import numpy as np

from score_bundle import metrics


def test_coverage_and_pit_on_calibrated_gaussian():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.standard_normal(n)
    mean = np.zeros(n)
    std = np.ones(n)

    cov = metrics.coverage(y, mean, std, level=0.9)
    assert abs(cov - 0.9) < 0.02

    pit = metrics.pit_values(y, mean, std)
    assert abs(pit.mean() - 0.5) < 0.02
    assert metrics.calibration_error(y, mean, std) < 0.05


def test_overconfident_model_is_miscalibrated():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.standard_normal(n)
    mean = np.zeros(n)
    std_good = np.ones(n)
    std_bad = np.full(n, 0.3)  # far too confident
    assert metrics.calibration_error(y, mean, std_bad) > metrics.calibration_error(y, mean, std_good)
    assert metrics.gaussian_nll(y, mean, std_bad) > metrics.gaussian_nll(y, mean, std_good)


def test_error_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 4.0])
    assert abs(metrics.mae(y, p) - (1 / 3)) < 1e-12
    assert abs(metrics.rmse(y, p) - np.sqrt(1 / 3)) < 1e-12
