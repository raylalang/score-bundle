"""A tiny dependency-free Nelder-Mead simplex optimizer.

Used for empirical-Bayes hyperparameter learning when SciPy is not installed.
If SciPy is available, prefer ``scipy.optimize.minimize``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: float = 0.5,
    max_iter: int = 400,
    tol: float = 1e-7,
) -> np.ndarray:
    """Minimize ``f`` starting from ``x0``. Returns the best point found."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += step if x0[i] == 0 else step * abs(x0[i])
        simplex.append(xi)
    simplex = np.array(simplex)
    fvals = np.array([f(x) for x in simplex])

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol and (
            np.max(np.abs(fvals - fvals[0])) < tol
        ):
            break

        centroid = simplex[:-1].mean(axis=0)
        # reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        # expansion
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        # contraction
        xc = centroid + rho * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        # shrink
        simplex = simplex[0] + sigma * (simplex - simplex[0])
        fvals = np.array([f(x) for x in simplex])

    best = np.argmin(fvals)
    return simplex[best]
