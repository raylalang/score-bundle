import numpy as np

from score_bundle import build_adjacency, laplacian, chain_adjacency
from score_bundle.graph import degree
from score_bundle.synthetic import random_score


def test_adjacency_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    score = random_score(20, rng)
    W = build_adjacency(score, ell_b=2.0, ell_p=4.0)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)
    assert (W >= 0).all()


def test_laplacian_rows_sum_to_zero():
    rng = np.random.default_rng(1)
    score = random_score(15, rng)
    W = build_adjacency(score)
    L = laplacian(W)
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-8)
    # Laplacian is positive semidefinite
    evals = np.linalg.eigvalsh(L)
    assert evals.min() > -1e-8


def test_knn_sparsify_limits_neighbours():
    rng = np.random.default_rng(2)
    score = random_score(30, rng)
    W = build_adjacency(score, knn=3)
    # symmetrization can add a few, but degree should stay small relative to n
    nnz_per_row = (W > 0).sum(axis=1)
    assert nnz_per_row.max() <= 8


def test_chain_adjacency():
    W = chain_adjacency(n=5)
    assert W.shape == (5, 5)
    assert W[0, 1] == 1 and W[1, 2] == 1
    assert W[0, 2] == 0
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(degree(W)), W.sum(1))
