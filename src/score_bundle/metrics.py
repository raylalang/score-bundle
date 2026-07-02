"""Recovery and calibration metrics.

The central evaluation question is whether the structured prior improves both
*recovery accuracy* and *uncertainty calibration* relative to baselines.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF (vectorized, via erf)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d ** 2)))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Mean negative log-likelihood under N(mean, std^2)."""
    std = np.clip(np.asarray(std, dtype=float), 1e-12, None)
    d = np.asarray(y_true) - np.asarray(mean)
    nll = 0.5 * (np.log(2 * np.pi) + 2 * np.log(std) + (d / std) ** 2)
    return float(np.mean(nll))


def pit_values(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Probability integral transform: should be ~Uniform(0,1) if calibrated."""
    std = np.clip(np.asarray(std, dtype=float), 1e-12, None)
    return _phi((np.asarray(y_true) - np.asarray(mean)) / std)


def coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float = 0.9) -> float:
    """Empirical coverage of central credible intervals at ``level``."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    std = np.clip(np.asarray(std, dtype=float), 1e-12, None)
    inside = np.abs(np.asarray(y_true) - np.asarray(mean)) <= z * std
    return float(np.mean(inside))


def calibration_error(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """KS-style distance between the PIT distribution and Uniform(0,1)."""
    u = np.sort(pit_values(y_true, mean, std))
    n = u.size
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - u)))


def std_rescale_factor(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float = 0.9
) -> float:
    """Multiplicative std factor s so that coverage@``level`` is exact on this data.

    Conformal-style variance scaling: with normalized residuals z = (y - mean)/std,
    ``s = quantile(|z|, level) / z_level`` makes the central ``level`` interval of
    N(mean, (s*std)^2) cover exactly a ``level`` fraction of the given examples.
    Fit s on *calibration* data (never the eval notes), then multiply eval stds by it.
    """
    from statistics import NormalDist

    std = np.clip(np.asarray(std, dtype=float), 1e-12, None)
    z = np.abs((np.asarray(y_true) - np.asarray(mean)) / std)
    z_level = NormalDist().inv_cdf(0.5 + level / 2.0)
    return float(np.quantile(z, level) / z_level)


def evaluate(y_true, mean, std, level: float = 0.9) -> Dict[str, float]:
    """Bundle the common metrics into one dict."""
    return {
        "mae": mae(y_true, mean),
        "rmse": rmse(y_true, mean),
        "nll": gaussian_nll(y_true, mean, std),
        "coverage@%.2f" % level: coverage(y_true, mean, std, level),
        "calibration_error": calibration_error(y_true, mean, std),
    }
