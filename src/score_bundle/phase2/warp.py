"""Onset-anchored score-time warp and the Phase-2 timing channel (tau).

Implements the measured tau policy of docs/phase2_prereg_design.md (option 1,
adopted on the feasibility evidence in results/tau_feasibility_dev.md): URMP's
note-level onset annotations anchor the warp — no audio aligner — via

  - score-to-performance note matching (exact order match when the pitch
    sequences agree, else a monotone pitch DTW), and
  - the LOCAL leave-one-out tempo line of draft eq:localwarp: each note's
    predicted time comes from a +/-``win``-note linear fit of performed onset
    on score onset that EXCLUDES the note itself, so tau_i = t_i - prediction
    never sees its own onset.

The aligner error enters the GP through the noise row (the prereg policy):
``note_tau`` returns the OLS *predictive* variance of each note's tempo-line
prediction — s^2 (1 + x0^T (X^T X)^{-1} x0) from the neighbour fit — as the
per-note tau observation-noise variance.  Unmatched notes are NaN (missing
cells for the cell-mask GP).  numpy-only, deterministic.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

WIN = 8


def dtw_match(sp: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Monotone alignment of two pitch sequences (small banded DP); returns
    index pairs (i_score, j_perf) for matched notes with equal pitch."""
    sp = np.asarray(sp)
    pp = np.asarray(pp)
    ns, npf = sp.size, pp.size
    cost = np.full((ns + 1, npf + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, ns + 1):
        for j in range(max(1, i - 40), min(npf + 1, i + 40)):
            sub = 0.0 if sp[i - 1] == pp[j - 1] else 1.0
            cost[i, j] = sub + min(cost[i - 1, j - 1], cost[i - 1, j] + 0.1,
                                   cost[i, j - 1] + 0.1)
    pairs = []
    i, j = ns, npf
    while i > 0 and j > 0:
        moves = [(cost[i - 1, j - 1], i - 1, j - 1),
                 (cost[i - 1, j], i - 1, j),
                 (cost[i, j - 1], i, j - 1)]
        _, i2, j2 = min(moves)
        if i2 == i - 1 and j2 == j - 1 and sp[i - 1] == pp[j - 1]:
            pairs.append((i - 1, j - 1))
        i, j = i2, j2
    return np.array(pairs[::-1], dtype=int).reshape(-1, 2)


def match_score_to_performance(score_pitch: np.ndarray,
                               perf_pitch: np.ndarray,
                               min_match: float = 0.8
                               ) -> Tuple[np.ndarray, np.ndarray, str]:
    """(score indices, performance indices, method) or empty arrays when the
    match is too poor (fewer than ``min_match`` of the shorter sequence)."""
    sp = np.asarray(score_pitch)
    pp = np.asarray(perf_pitch)
    if sp.size == pp.size and sp.size and np.mean(sp == pp) > 0.9:
        idx = np.arange(sp.size)
        return idx, idx, "exact"
    pairs = dtw_match(sp, pp)
    if pairs.size == 0 or len(pairs) < min_match * min(sp.size, pp.size):
        return np.array([], int), np.array([], int), "failed"
    return pairs[:, 0], pairs[:, 1], "dtw"


def local_loo_warp(b: np.ndarray, t: np.ndarray, win: int = WIN
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out local tempo line (draft eq:localwarp) over matched pairs.

    Returns per matched note: the predicted performance time, and the OLS
    predictive variance of that prediction (the tau noise row).  The note
    itself never enters its own fit.
    """
    b = np.asarray(b, dtype=float)
    t = np.asarray(t, dtype=float)
    n = b.size
    pred = np.full(n, np.nan)
    pvar = np.full(n, np.inf)
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        idx = np.r_[lo:i, i + 1:hi]
        if idx.size < 3:
            continue
        A = np.stack([b[idx], np.ones(idx.size)], axis=1)
        coef, res, rank, _ = np.linalg.lstsq(A, t[idx], rcond=None)
        if rank < 2:
            continue
        x0 = np.array([b[i], 1.0])
        pred[i] = float(x0 @ coef)
        dof = idx.size - 2
        s2 = float(res[0]) / dof if res.size and dof > 0 else np.nan
        if np.isfinite(s2):
            try:
                lever = float(x0 @ np.linalg.solve(A.T @ A, x0))
            except np.linalg.LinAlgError:
                continue
            pvar[i] = s2 * (1.0 + lever)
    return pred, pvar


def note_tau(score_onset: np.ndarray, score_pitch: np.ndarray,
             perf_onset: np.ndarray, perf_pitch: np.ndarray,
             win: int = WIN) -> dict:
    """The tau channel over the PERFORMED notes of one track.

    Returns dict: ``tau`` (N_perf,) with NaN at unmatched/undetermined notes,
    ``var`` (N_perf,) predictive variances (the observation-noise row),
    ``matched`` boolean, ``method`` ("exact"/"dtw"/"failed").
    """
    si, pi, method = match_score_to_performance(score_pitch, perf_pitch)
    n = np.asarray(perf_onset).size
    tau = np.full(n, np.nan)
    var = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)
    if method == "failed":
        return {"tau": tau, "var": var, "matched": matched, "method": method}
    b = np.asarray(score_onset, dtype=float)[si]
    t = np.asarray(perf_onset, dtype=float)[pi]
    pred, pvar = local_loo_warp(b, t, win=win)
    ok = np.isfinite(pred) & np.isfinite(pvar)
    tau[pi[ok]] = t[ok] - pred[ok]
    var[pi[ok]] = pvar[ok]
    matched[pi] = True
    return {"tau": tau, "var": var, "matched": matched, "method": method}
