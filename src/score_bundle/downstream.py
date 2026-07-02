"""Downstream tasks built on the Phase-1 graph posterior (numpy-only).

Three demonstrations that the score-graph prior earns its keep beyond the Phase-1
random-mask imputation benchmark:

1. **Performance completion / expressive rendering.** Predict the expression of
   unheard notes from a small observed excerpt — a contiguous prefix (the performer
   played the opening) or a contiguous held-out block. Structured *extrapolation*
   rather than random-mask *interpolation*; the cells themselves re-use
   :func:`score_bundle.imputation_eval.impute_methods`, only the masks differ.

2. **Performance-error (anomaly) detection.** Rank notes by leave-one-out predictive
   surprise under the model. A calibrated, structured posterior should separate
   injected errors from clean notes better than an unstructured residual z-score —
   this is the task that directly cashes in calibration.

3. **Transcription denoising.** Observe *every* note through synthetic observation
   noise (as from a noisy AMT transcription) and recover the clean values by
   posterior shrinkage. The latent posterior std ``sqrt(diag(Sigma_y))`` is the
   calibration object (no observation-noise floor: the target is the latent y, not
   a held-out observation).

Everything here is numpy-only (Phase-1 core rules); the LM mean arrives as an array.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .model import GraphGaussianField, fit_laplacian_field
from .optimize import nelder_mead
from .prior import laplacian_precision

_LOG2PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- data
def load_piece_arrays(path: str):
    """Load the named per-piece array cache from ``scripts/extract_asap_arrays.py``.

    Returns ``(head, eval, meta)`` lists of per-piece dicts.  Refuses caches whose
    provenance does not record the MAESTRO contamination filter — downstream evals
    must not silently run on possibly-contaminated pieces.
    """
    import pickle

    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    meta = blob.get("meta", {})
    if not meta.get("contamination_filtered"):
        raise ValueError(
            f"{path} does not record contamination filtering; regenerate it with "
            "scripts/extract_asap_arrays.py (which always applies the MAESTRO filter)."
        )
    return blob["head"], blob["eval"], meta


def piece_score(p: dict):
    """Rebuild the :class:`~score_bundle.score.Score` support from a cached piece dict."""
    from .score import Score

    return Score.from_arrays(p["pitch"], p["onset"], p["duration"], p["voice"])


# --------------------------------------------------------------------------- masks
def prefix_mask(n: int, observed_frac: float) -> np.ndarray:
    """Observe the first ``observed_frac`` of notes (score order); predict the rest.

    The 'performance completion' setting: the performer played the opening excerpt.
    """
    k = int(round(observed_frac * n))
    k = min(max(k, 1), n - 1)
    mask = np.zeros(n, dtype=bool)
    mask[:k] = True
    return mask


def block_mask(n: int, rng: np.random.Generator, observed_frac: float = 0.6) -> np.ndarray:
    """Hold out one contiguous block of ``1 - observed_frac`` at a random position.

    Unlike :func:`score_bundle.imputation_eval.random_mask`, held-out notes have no
    observed interior neighbours, so the graph must extrapolate across the gap.
    """
    n_held = int(round((1.0 - observed_frac) * n))
    n_held = min(max(n_held, 1), n - 1)
    start = int(rng.integers(0, n - n_held + 1))
    mask = np.ones(n, dtype=bool)
    mask[start : start + n_held] = False
    return mask


# --------------------------------------------------------------------------- anomaly
def inject_anomalies(
    y: np.ndarray,
    rng: np.random.Generator,
    frac: float = 0.05,
    scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Corrupt a random ``frac`` of notes with +-``scale`` * channel-std shifts.

    ``y`` is a single channel, shape (N,). Returns ``(y_corrupt, labels)`` with
    ``labels`` True at corrupted notes. Signs are random; magnitude is fixed at
    ``scale`` standard deviations so task difficulty is controlled by one number.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    n_bad = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=n_bad, replace=False)
    sd = float(np.std(y)) or 1.0
    shift = scale * sd * rng.choice([-1.0, 1.0], size=n_bad)
    y_corrupt = y.copy()
    y_corrupt[idx] += shift
    labels = np.zeros(n, dtype=bool)
    labels[idx] = True
    return y_corrupt, labels


def loo_predictive(
    field: GraphGaussianField, y_obs: np.ndarray, noise_var: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out predictive N(mean_i, var_i) of each *observation* given the rest.

    Under the joint observation marginal ``y_obs ~ N(mu, C)`` with
    ``C = Q^{-1} + noise_var I``, the standard LOO identity with ``P = C^{-1}`` and
    ``r = y_obs - mu`` gives

        mean_i = y_obs_i - (P r)_i / P_ii,      var_i = 1 / P_ii.

    ``var_i`` is the predictive variance of the noisy observation (it already
    includes the observation noise), so no extra floor is needed.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    C = field.covariance() + float(noise_var) * np.eye(field.N)
    P = np.linalg.inv(C)
    r = y_obs - field.mean
    Pr = P @ r
    dii = np.clip(np.diag(P), 1e-12, None)
    loo_mean = y_obs - Pr / dii
    loo_var = 1.0 / dii
    return loo_mean, loo_var


def anomaly_scores(
    L: np.ndarray,
    y_obs: np.ndarray,
    mean: np.ndarray,
    use_graph: bool = True,
) -> np.ndarray:
    """Per-note surprise score (higher = more anomalous) for one channel.

    ``use_graph=True``: empirical-Bayes fit of the graph field on the (corrupted)
    channel, then leave-one-out predictive NLL of each note.  ``use_graph=False``:
    the unstructured baseline — NLL under a homoscedastic Gaussian around the mean
    (equivalent ranking to the absolute residual z-score).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if not use_graph:
        r = y_obs - mean
        s2 = max(float(np.var(r)), 1e-12)
        return 0.5 * (_LOG2PI + np.log(s2) + r**2 / s2)
    field, hp = fit_laplacian_field(L, y_obs, mask=None, mean=mean)
    loo_mean, loo_var = loo_predictive(field, y_obs, hp["noise_var"])
    r = y_obs - loo_mean
    return 0.5 * (_LOG2PI + np.log(loo_var) + r**2 / loo_var)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) statistic, tie-aware."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:  # average ranks over ties
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(float)
    cum_hits = np.cumsum(hits)
    precision = cum_hits / np.arange(1, labels.size + 1)
    return float((precision * hits).sum() / labels.sum())


# --------------------------------------------------------------------------- denoise
def independent_denoise(
    y_noisy: np.ndarray, mean: np.ndarray, noise_var: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form independent-prior denoiser with *known* noise level (oracle).

    Without coupling, only prior_var + noise_var is identified from noisy data, so
    the independent baseline must be told the noise level; the prior variance is
    then moment-matched (``max(var(resid) - noise_var, eps)``) and the posterior is
    scalar Wiener shrinkage toward the mean.  Returns (pred, latent posterior std).
    """
    y_noisy = np.asarray(y_noisy, dtype=float)
    mean = np.asarray(mean, dtype=float)
    r = y_noisy - mean
    prior_var = max(float(np.var(r)) - float(noise_var), 1e-8)
    w = prior_var / (prior_var + float(noise_var))
    pred = mean + w * r
    std = np.full(y_noisy.shape, np.sqrt(w * float(noise_var)))
    return pred, std


def _fit_graph_fixed_noise(
    L: np.ndarray, y_obs: np.ndarray, mean: np.ndarray, noise_var: float,
    x0=(0.0, 0.0),
) -> GraphGaussianField:
    """EB fit of (lam, eta) with the observation noise *fixed* (oracle-noise variant)."""
    L = np.asarray(L, dtype=float)

    def neg_lml(logp: np.ndarray) -> float:
        lam, eta = np.exp(logp)
        try:
            Q = laplacian_precision(L, lam=lam, eta=eta)
            field = GraphGaussianField(Q, mean=mean)
            return -field.log_marginal_likelihood(y_obs, noise_var)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12
    best = nelder_mead(neg_lml, np.asarray(x0, dtype=float))
    lam, eta = np.exp(best)
    return GraphGaussianField(laplacian_precision(L, lam=lam, eta=eta), mean=mean)


def denoise_channel(
    L: np.ndarray,
    y_noisy: np.ndarray,
    mean: np.ndarray,
    noise_std: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Denoise one fully-observed noisy channel; returns (latent pred, latent std).

    Methods
    -------
    ``identity``      the observation itself, std = the (oracle) noise level.
    ``independent``   scalar Wiener shrinkage toward the mean, oracle noise.
    ``graph``         GMRF posterior, *blind*: (lam, eta, noise_var) all EB-fit —
                      the graph structure is what makes the noise level identifiable.
                      The fit floors noise_var at 5% of the residual variance
                      (without it the EB fit occasionally collapses noise_var -> 0
                      and the latent intervals shrink to nothing).
    ``graph-oracle``  GMRF posterior with the true noise variance fixed.
    ``graph-calib``   GMRF posterior, blind, but hyperparameters chosen by held-out
                      predictive NLL on a calibration split of the notes (directly
                      optimizes what the eval scores) instead of in-sample marglik.
                      NB: in practice this can *also* collapse noise_var on smooth
                      pieces (its NLL explodes on real ASAP data); kept for
                      completeness, not a recommended default.
    """
    from .model import fit_laplacian_field_calib

    y_noisy = np.asarray(y_noisy, dtype=float)
    mean = np.asarray(mean, dtype=float)
    nv = float(noise_std) ** 2
    if method == "identity":
        return y_noisy, np.full(y_noisy.shape, float(noise_std))
    if method == "independent":
        return independent_denoise(y_noisy, mean, nv)
    if method == "graph":
        floor = 0.05 * float(np.var(y_noisy - mean))
        field, hp = fit_laplacian_field(L, y_noisy, mask=None, mean=mean, noise_floor=floor)
        return field.posterior(y_noisy, hp["noise_var"])
    if method == "graph-calib":
        field, hp = fit_laplacian_field_calib(L, y_noisy, mask=None, mean=mean,
                                              rng=np.random.default_rng(0))
        return field.posterior(y_noisy, hp["noise_var"])
    if method == "graph-oracle":
        field = _fit_graph_fixed_noise(L, y_noisy, mean, nv)
        return field.posterior(y_noisy, nv)
    raise ValueError(f"unknown denoise method {method!r}")
