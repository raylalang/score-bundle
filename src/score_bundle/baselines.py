"""Baselines for the Phase-1 imputation/recovery task.

- Independent per-note prior   : diagonal precision, no graph coupling.
- Temporal smoothing (AR(1))   : chain graph along score time only.
- Score-feature regression     : predict expression from local score features
                                 (ridge; gradient-boosted trees if scikit-learn
                                 is installed). No inter-note coupling.

The *proposed* model is :class:`score_bundle.model.GraphGaussianField` with the
full score graph; a win requires beating both the independent and temporal-only
baselines on recovery *and* calibration.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .graph import chain_adjacency, laplacian
from .model import GraphGaussianField
from .prior import laplacian_precision
from .score import Score


def independent_field(n: int, prior_var: float = 1.0, mean=0.0) -> GraphGaussianField:
    """Diagonal prior precision (no coupling)."""
    Q = np.eye(n) / float(prior_var)
    return GraphGaussianField(Q, mean=mean)


def temporal_field(
    score: Score, lam: float = 1.0, eta: float = 1.0, mean=0.0
) -> GraphGaussianField:
    """AR(1)-style chain graph along score-time order."""
    order = np.argsort(score.onset)
    W = chain_adjacency(order=order)
    L = laplacian(W)
    Q = laplacian_precision(L, lam=lam, eta=eta)
    return GraphGaussianField(Q, mean=mean)


def score_features(score: Score) -> np.ndarray:
    """Simple local score features: [pitch, onset-in-bar proxy, beat strength, duration]."""
    p = score.pitch
    b = score.onset
    d = score.duration
    onset_frac = b - np.floor(b)            # position within the beat/bar (proxy)
    beat_strength = np.cos(2 * np.pi * onset_frac)  # high on strong beats
    X = np.stack([p, onset_frac, beat_strength, d], axis=1)
    # standardize
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    return (X - mu) / sd


def ridge_impute(
    score: Score,
    y_obs: np.ndarray,
    mask: np.ndarray,
    l2: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Predict masked notes from observed ones via ridge on score features.

    Returns (predictions over all notes, residual std estimated on observed set).
    """
    X = score_features(score)
    Xo, yo = X[mask], np.asarray(y_obs)[mask]
    n_feat = X.shape[1]
    A = Xo.T @ Xo + l2 * np.eye(n_feat)
    w = np.linalg.solve(A, Xo.T @ (yo - yo.mean()))
    b = yo.mean()
    pred = X @ w + b
    resid = yo - (Xo @ w + b)
    sigma = float(np.std(resid)) if resid.size > 1 else 1.0
    return pred, sigma


def gbm_impute(score: Score, y_obs: np.ndarray, mask: np.ndarray):
    """Gradient-boosted-trees baseline (requires scikit-learn)."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("gbm_impute requires scikit-learn (pip install score-bundle[baselines])") from exc

    X = score_features(score)
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X[mask], np.asarray(y_obs)[mask])
    return model.predict(X)


def _window_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of ``x`` over the +-``w``-note window around each position (cumsum trick)."""
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    hi = np.minimum(i + w + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


# Krumhansl–Kessler tonal-hierarchy profiles (probe-tone ratings; Krumhansl 1990).
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_DIATONIC = {0: np.isin(np.arange(12), [0, 2, 4, 5, 7, 9, 11]),   # major
             1: np.isin(np.arange(12), [0, 2, 3, 5, 7, 8, 10])}   # natural minor
_DISSONANT_PC = np.isin(np.arange(12), [1, 2, 6, 10, 11])


def _theory_block(ps: np.ndarray, bs: np.ndarray, ds: np.ndarray, vs: np.ndarray,
                  key_window: int = 16) -> np.ndarray:
    """Music-theory features on score-order arrays; returns (n, 14).

    Columns: key clarity, mode (major=1), scale-degree sin/cos, in-scale flag,
    circle-of-fifths motion of the local key, metrical weight, vertical
    dissonance, bass flag, LBDM boundary salience (IOI / pitch, per voice),
    motif-repetition count (per-voice interval+rhythm 4-grams), within-voice
    pitch z, within-voice position.  Score-only by construction.
    """
    n = len(ps)
    eps = 1e-9
    pc = np.mod(ps.astype(int), 12)

    # --- local key by Krumhansl–Schmuckler over a +-key_window note window ----
    onehot = np.zeros((n, 12))
    onehot[np.arange(n), pc] = np.maximum(ds, 0.25)  # duration-weighted, floored
    c = np.concatenate([np.zeros((1, 12)), np.cumsum(onehot, axis=0)])
    i = np.arange(n)
    lo = np.maximum(i - key_window, 0)
    hi = np.minimum(i + key_window + 1, n)
    hist = c[hi] - c[lo]
    profs = np.stack([np.roll(prof, t)                       # 24 = 12 tonics x 2 modes
                      for prof in (_KK_MAJOR, _KK_MINOR) for t in range(12)])
    hz = hist - hist.mean(axis=1, keepdims=True)
    pz = profs - profs.mean(axis=1, keepdims=True)
    r = (hz @ pz.T) / np.maximum(np.linalg.norm(hz, axis=1, keepdims=True)
                                 * np.linalg.norm(pz, axis=1), eps)
    best = np.argmax(r, axis=1)
    tonic = best % 12
    mode = best // 12                                        # 0 = major, 1 = minor
    key_clarity = r[np.arange(n), best]
    deg = np.mod(pc - tonic, 12)
    deg_sin = np.sin(2 * np.pi * deg / 12.0)
    deg_cos = np.cos(2 * np.pi * deg / 12.0)
    in_scale = np.stack([_DIATONIC[0], _DIATONIC[1]])[mode, deg].astype(float)
    cof = np.mod(7 * tonic, 12)                              # circle-of-fifths index
    dcof = np.abs(np.diff(cof, prepend=cof[:1]))
    fifths_motion = np.minimum(dcof, 12 - dcof) / 6.0

    # --- metrical weight: how many metrical levels the onset sits on ----------
    levels = (0.25, 0.5, 1.0, 2.0, 4.0)
    metric_w = sum((np.abs(bs - lv * np.round(bs / lv)) < 1e-6).astype(float)
                   for lv in levels) / len(levels)

    # --- verticals: dissonance within the same-onset chord + bass flag --------
    dissonance = np.zeros(n)
    is_bass = np.zeros(n)
    _, grp = np.unique(np.round(bs, 6), return_inverse=True)
    for g in range(grp.max() + 1):
        idx = np.flatnonzero(grp == g)
        is_bass[idx[np.argmin(ps[idx])]] = 1.0
        if len(idx) < 2:
            continue
        for j in idx:
            ivl = np.mod(np.abs(pc[idx] - pc[j]), 12)
            dissonance[j] = float(np.mean(_DISSONANT_PC[ivl[idx != j]]))

    # --- per-voice sequences: LBDM salience, repetition, tessitura ------------
    lbdm_ioi = np.zeros(n)
    lbdm_pitch = np.zeros(n)
    repeat_cnt = np.zeros(n)
    v_pitch_z = np.zeros(n)
    v_pos = np.zeros(n)
    for v in np.unique(vs):
        vi = np.flatnonzero(vs == v)
        vp, vb = ps[vi], bs[vi]
        m = len(vi)
        v_pitch_z[vi] = (vp - vp.mean()) / (vp.std() + eps)
        v_pos[vi] = np.arange(m) / max(m - 1, 1)
        if m < 3:
            continue
        ioi = np.maximum(np.diff(vb), eps)                   # m-1 transitions
        ivl = np.abs(np.diff(vp))
        for x, out in ((ioi, lbdm_ioi), (ivl, lbdm_pitch)):
            rch = np.abs(np.diff(x)) / (x[1:] + x[:-1] + eps)   # m-2 change ratios
            sal = np.zeros(m)                                # note k <- boundary before it
            sal[2:] = np.log1p(x[1:]) * rch
            out[vi] = sal
        if m >= 5:                                           # interval+rhythm 4-grams
            q_ivl = np.clip(np.diff(vp).astype(int), -12, 12)
            q_rhy = np.clip(np.round(2 * np.log2(ioi[1:] / ioi[:-1])) / 2, -3, 3)
            grams: dict = {}
            keys = [tuple(q_ivl[k:k + 4]) + tuple(q_rhy[k:k + 3])
                    for k in range(m - 4)]
            for key in keys:
                grams[key] = grams.get(key, 0) + 1
            cnt = np.zeros(m)
            for k, key in enumerate(keys):                   # gram k covers notes k..k+4
                cnt[k:k + 5] = np.maximum(cnt[k:k + 5], grams[key])
            repeat_cnt[vi] = np.log1p(cnt - 1)

    return np.stack(
        [key_clarity, 1.0 - mode.astype(float), deg_sin, deg_cos, in_scale,
         fifths_motion, metric_w, dissonance, is_bass,
         lbdm_ioi, lbdm_pitch, repeat_cnt, v_pitch_z, v_pos],
        axis=1,
    )


def rich_score_features(
    score: Score,
    window: int = 8,
    rff_dim: int = 0,
    rff_seed: int = 0,
    theory: bool = False,
) -> np.ndarray:
    """Rich, score-only per-note features — the honest rival to the LM embedding.

    Everything here is computed from the score support alone (pitch / onset /
    duration / voice, never the performance), so a prior-mean head fit on these
    features under the *identical* cross-piece protocol as the LM head
    (:func:`score_bundle.lm.features.fit_prior_mean_head` on head pieces, applied
    out-of-sample) isolates what the pretrained transformer adds beyond hand-built
    local score structure.  Unlike :func:`score_features` (4 dims, fit per piece on
    the observed notes), these are designed to transfer across pieces.

    Base features (score-time order, mapped back to the input order): absolute and
    piece-relative pitch, local pitch deviation / trend / spread, melodic interval
    and contour, log-duration (absolute / piece-z / local-relative), metrical phase
    at beat, half-bar and bar periods (sin + cos), inter-onset intervals, local note
    density, chord size and within-chord pitch rank, piece position and edge
    proximity, voice.  With ``rff_dim > 0``, appends deterministic random Fourier
    features of the base features (seed ``rff_seed``) so a linear head can fit a
    smooth nonlinear function.

    ``theory=True`` appends the 14 music-theory columns of :func:`_theory_block`
    (local key / scale degree, metrical weight, vertical dissonance, phrase
    boundaries, motif repetition, voice tessitura) — still score-only.  Default
    off: every published result used the 25 base columns.
    """
    n = len(score)
    p = score.pitch.astype(float)
    b = score.onset.astype(float)
    d = score.duration.astype(float)
    v = score.voice.astype(float)
    order = np.lexsort((p, b))
    ps, bs, ds, vs = p[order], b[order], d[order], v[order]

    w = max(1, min(window, n - 1)) if n > 1 else 1
    eps = 1e-9

    def pz(x):  # per-piece z-score
        return (x - x.mean()) / (x.std() + eps)

    # pitch
    pitch_abs = (ps - 60.0) / 24.0
    pitch_z = pz(ps)
    m_p = _window_mean(ps, w)
    pitch_local_dev = (ps - m_p) / 12.0
    m_p2 = _window_mean(ps * ps, w)
    pitch_local_std = np.sqrt(np.maximum(m_p2 - m_p ** 2, 0.0)) / 12.0
    interval_prev = np.diff(ps, prepend=ps[:1]) / 12.0
    contour = np.sign(interval_prev)
    # local pitch-vs-time slope over the window (semitones per beat, squashed)
    m_b = _window_mean(bs, w)
    m_bp = _window_mean(bs * ps, w)
    m_b2 = _window_mean(bs * bs, w)
    slope = (m_bp - m_b * m_p) / (m_b2 - m_b ** 2 + eps)
    slope = np.tanh(slope / 12.0)
    # duration
    dur_log = np.log1p(ds)
    dur_z = pz(dur_log)
    dur_local_rel = dur_log - _window_mean(dur_log, w)
    # metrical phase (beat / half-bar / bar proxies; ASAP onsets are in beats)
    phases = []
    for period in (1.0, 2.0, 4.0):
        ph = 2.0 * np.pi * np.mod(bs, period) / period
        phases += [np.sin(ph), np.cos(ph)]
    # inter-onset intervals and local density
    ioi_prev = np.log1p(np.diff(bs, prepend=bs[:1]))
    ioi_next = np.log1p(np.diff(bs, append=bs[-1:]))
    span = 2.0
    dens = (np.searchsorted(bs, bs + span, side="right")
            - np.searchsorted(bs, bs - span, side="left")) / (2.0 * span)
    dens = np.log1p(dens)
    # chords (identical onsets): size and normalized pitch rank within the chord
    _, grp, counts = np.unique(np.round(bs, 6), return_inverse=True, return_counts=True)
    chord_size = np.log1p(counts[grp].astype(float))
    starts = np.concatenate([[0], np.cumsum(counts)])[grp]
    chord_rank = (np.arange(n) - starts) / np.maximum(counts[grp] - 1.0, 1.0)
    # position in the piece and edge proximity
    pos = np.arange(n) / max(n - 1, 1)
    edge_start = np.exp(-np.arange(n) / 8.0)
    edge_end = np.exp(-(n - 1 - np.arange(n)) / 8.0)
    voice_n = vs / (vs.max() + 1.0)

    X = np.stack(
        [pitch_abs, pitch_z, pitch_local_dev, pitch_local_std, interval_prev,
         contour, slope, dur_log, dur_z, dur_local_rel, *phases,
         ioi_prev, ioi_next, dens, chord_size, chord_rank,
         pos, edge_start, edge_end, voice_n],
        axis=1,
    )
    if theory:
        X = np.concatenate([X, _theory_block(ps, bs, ds, vs)], axis=1)
    if rff_dim > 0:
        rng = np.random.default_rng(rff_seed)
        G = rng.normal(0.0, 1.0, size=(X.shape[1], rff_dim))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=rff_dim)
        Z = np.sqrt(2.0 / rff_dim) * np.cos(X @ G + phase)
        X = np.concatenate([X, Z], axis=1)

    out = np.empty_like(X)
    out[order] = X
    return out
