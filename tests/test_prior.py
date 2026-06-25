import numpy as np

from score_bundle import laplacian, build_adjacency, laplacian_precision, matern_precision
from score_bundle.prior import is_positive_definite
from score_bundle.synthetic import random_score


def _L(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return laplacian(build_adjacency(random_score(n, rng)))


def test_laplacian_precision_positive_definite():
    L = _L()
    Q = laplacian_precision(L, lam=0.5, eta=2.0)
    assert is_positive_definite(Q)


def test_matern_alpha1_matches_laplacian_form():
    L = _L()
    kappa = 0.8
    # Matern with alpha=1: (kappa^2 I + L) == additive form with lam=kappa^2, eta=1
    Q_matern = matern_precision(L, kappa=kappa, alpha=1, sigma_g=1.0)
    Q_lap = laplacian_precision(L, lam=kappa ** 2, eta=1.0)
    assert np.allclose(Q_matern, Q_lap)


def test_matern_requires_positive_integer_alpha():
    L = _L()
    for bad in (0, 1.5, -2):
        try:
            matern_precision(L, alpha=bad)
            assert False, "should have raised"
        except ValueError:
            pass
