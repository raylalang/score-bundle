#!/usr/bin/env python
"""The spectral mixture method applied AS INTENDED (dev-only, exploratory).

The supervisor's correction stands (docs/sm_kernel_verification.md): our
estimator-v2 was a hand-structured two-component kernel, not the
Wilson & Adams METHOD. This study runs the method proper on the same
within-note cents curves: Q free components, means free on [0, 15] Hz
(physical cap only -- no role pinning, no coherence bound), weights and
bandwidths learned, initialization from the note's least-squares
periodogram (the uneven-sampling analogue of W&A's empirical-spectrum
initialization), greedy component addition then joint polish.

Committed question (before any run): does the learned Q-component
spectrum (a) describe the curves better than the hand-structured
2-component kernel at matched evidence/curve-level measures, and
(b) REDISCOVER the vibrato structure (a component in 4-9 Hz) without
being told?

The only structural bound kept is mathematical, applied uniformly to
every component: v_q >= 0.025/span^2 (a component flat over the note is
exactly redundant with the flat-prior constant c -- the same degeneracy
documented for the 2-component kernel, not a role assignment).

Stages:
    inspect          one note: fit, print, and plot learned spectrum vs
                     the periodogram (.cache/sm_proper_inspect.png)
    pilot [n=4]      first track timed alone (long-run rule), then n dev
                     tracks; rows -> .cache/sm_proper_pilot.pkl
    report           summary table from the pickle

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_sm_proper.py pilot
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from score_bundle.optimize import nelder_mead  # noqa: E402
from score_bundle.phase2.sm_estimator import (_decimate,  # noqa: E402
                                               fit_sm_note)

F_MAX = 15.0      # physical cap (cents curves carry no structure above)
Q = 5             # free components, W&A-scale for short series
_SENT = 1e12
OUT = ".cache/sm_proper_pilot.pkl"


# ---------------------------------------------------------------- kernel
def kern_q(tau: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Q-component SM kernel; p = [log w_q, mu_q, log v_q]*Q + [log s2]."""
    k = np.zeros_like(np.asarray(tau, float))
    nq = (p.size - 1) // 3
    for q in range(nq):
        w, mu, v = np.exp(p[3 * q]), p[3 * q + 1], np.exp(p[3 * q + 2])
        k += w * np.exp(-2.0 * np.pi ** 2 * v * tau ** 2) \
            * np.cos(2.0 * np.pi * mu * tau)
    return k


def evid_q(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Flat-prior-c marginal log likelihood under kern_q (cf. sm_estimator)."""
    nq = (p.size - 1) // 3
    s2 = float(np.exp(p[-1]))
    tau = t[:, None] - t[None, :]
    K = kern_q(tau, p)
    k0 = float(sum(np.exp(p[3 * q]) for q in range(nq)))
    K[np.diag_indices_from(K)] = k0 + s2 + 1e-10 * (k0 + s2)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -_SENT
    one = np.ones(t.size)
    a1 = np.linalg.solve(L, one)
    ay = np.linalg.solve(L, y)
    a = float(a1 @ a1)
    b = float(a1 @ ay)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    quad = float(ay @ ay) - b * b / a
    val = (-0.5 * (y.size - 1) * np.log(2.0 * np.pi) - 0.5 * logdet
           - 0.5 * quad - 0.5 * np.log(a))
    return val if np.isfinite(val) else -_SENT


def ls_periodogram(t: np.ndarray, r: np.ndarray,
                   grid: np.ndarray) -> np.ndarray:
    """Least-squares amplitude^2 at each grid frequency (uneven sampling)."""
    out = np.empty(grid.size)
    for i, f in enumerate(grid):
        th = 2.0 * np.pi * f * t
        A = np.stack([np.ones(t.size), np.sin(th), np.cos(th)], axis=1)
        beta, *_ = np.linalg.lstsq(A, r, rcond=None)
        out[i] = beta[1] ** 2 + beta[2] ** 2
    return out


def fit_sm_proper(t: np.ndarray, y: np.ndarray, q_max: int = Q,
                  nm_iter: int = 300) -> dict:
    """Greedy component addition + joint polish; returns params/evidence."""
    t, y = _decimate(np.asarray(t, float), np.asarray(y, float), 200)
    vy = max(float(y.var(ddof=1)), 1e-6)
    span = max(float(t.max() - t.min()), 1e-3)
    v_min = 0.025 / span ** 2          # flat-component degeneracy floor
    v_cap, w_cap = 25.0, 50.0 * vy
    s2_0 = float(np.clip(np.median(np.diff(y) ** 2) / 2.0, 1e-3 * vy, vy))
    grid = np.linspace(0.0, F_MAX, 121)

    def neg(p: np.ndarray) -> float:
        nq = (p.size - 1) // 3
        for q in range(nq):
            w, mu, v = np.exp(p[3 * q]), p[3 * q + 1], np.exp(p[3 * q + 2])
            if not (0.0 <= mu <= F_MAX) or not (v_min <= v <= v_cap) \
                    or w > w_cap:
                return _SENT
        s2 = float(np.exp(p[-1]))
        if s2 > 10.0 * vy or s2 < 1e-4 * vy:
            return _SENT
        return -evid_q(t, y, p)

    def post_mean(p: np.ndarray) -> np.ndarray:
        nq = (p.size - 1) // 3
        s2 = float(np.exp(p[-1]))
        tau = t[:, None] - t[None, :]
        K = kern_q(tau, p)
        k0 = float(sum(np.exp(p[3 * q]) for q in range(nq)))
        Kn = K.copy()
        Kn[np.diag_indices_from(Kn)] = k0 + s2 + 1e-10 * (k0 + s2)
        L = np.linalg.cholesky(Kn)
        one = np.ones(t.size)
        a1 = np.linalg.solve(L, one)
        ay = np.linalg.solve(L, y)
        c = float(a1 @ ay) / float(a1 @ a1)
        r2 = np.linalg.solve(L.T, np.linalg.solve(L, y - c))
        return c + K @ r2

    p = np.array([np.log(s2_0)])
    resid = y - y.mean()
    for q in range(q_max):
        pg = ls_periodogram(t, resid, grid)
        f0 = float(grid[int(np.argmax(pg))])
        w0 = float(np.clip(pg.max() / 2.0, 1e-3 * vy, 0.5 * w_cap))
        v0 = float(np.clip(max(0.05, (max(f0, 0.5) / 10.0) ** 2),
                           1.1 * v_min, 0.5 * v_cap))
        comp = np.array([np.log(w0), f0, np.log(v0)])
        p = np.concatenate([p[:-1], comp, p[-1:]])
        p = nelder_mead(neg, p, step=0.3, max_iter=nm_iter)
        if neg(p) >= 0.5 * _SENT:          # fit broke; drop the component
            p = np.concatenate([p[:-4], p[-1:]])
            break
        resid = y - post_mean(p)
    p = nelder_mead(neg, p, step=0.1, max_iter=2 * nm_iter)
    nq = (p.size - 1) // 3
    comps = [(float(np.exp(p[3 * q])), float(p[3 * q + 1]),
              float(np.exp(p[3 * q + 2]))) for q in range(nq)]
    return {"params": p, "evidence": float(-neg(p)), "q": nq,
            "components": comps, "s2": float(np.exp(p[-1]))}


def predict_q(t_fit, y_fit, p, t_new):
    """Posterior mean/var at t_new under kern_q with flat-prior c."""
    nq = (p.size - 1) // 3
    s2 = float(np.exp(p[-1]))
    tau = t_fit[:, None] - t_fit[None, :]
    K = kern_q(tau, p)
    k0 = float(sum(np.exp(p[3 * q]) for q in range(nq)))
    K[np.diag_indices_from(K)] = k0 + s2 + 1e-10 * (k0 + s2)
    L = np.linalg.cholesky(K)
    one = np.linalg.solve(L, np.ones(t_fit.size))
    ay = np.linalg.solve(L, y_fit)
    a = float(one @ one)
    c = float(one @ ay) / a
    Ks = kern_q(t_new[:, None] - t_fit[None, :], p)
    A = np.linalg.solve(L, Ks.T)
    r = np.linalg.solve(L, y_fit - c)
    mean = c + A.T @ r
    var = k0 - np.einsum("ij,ij->j", A, A)
    var = var + (1.0 - A.T @ one) ** 2 / a
    return mean, np.maximum(var, 1e-12) + s2


# ---------------------------------------------------------------- data
def iter_notes(n_tracks: int):
    from eval_phase2_real import dev_unique_tracks
    from make_phase2_intro import note_curves
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    for key in sorted(data)[:n_tracks]:
        d = data[key]
        for i in range(d["onset"].size):
            if d["n_frames"][i] < 20:
                continue
            tt, x, ttg, xg = note_curves(key, i, data, f0s, tracks)
            if tt.size < 20 or ttg.size < 20:
                continue
            yield key, i, bool(d["ident"][i]), tt, x, ttg, xg


def run_note(tt, x, ttg, xg):
    """Both fits on the tracked curve, both scored on the GT frames."""
    row = {}
    t0 = time.time()
    hand = fit_sm_note(tt, x)
    row["t_hand"] = time.time() - t0
    t0 = time.time()
    prop = fit_sm_proper(tt, x)
    row["t_prop"] = time.time() - t0
    row["ev_hand"] = hand["evidence"]
    row["ev_prop"] = prop["evidence"]
    row["components"] = prop["components"]
    tot_w = sum(w for w, _, _ in prop["components"]) or 1.0
    row["vib_found"] = any(4.0 <= mu <= 9.0 and w / tot_w >= 0.05
                           for w, mu, _ in prop["components"])
    from score_bundle.phase2.sm_estimator import sm_predict
    if hand["params"] is not None:
        mh, vh = sm_predict(tt, x, hand["params"], ttg, include_noise=True)
        row["rmse_hand"] = float(np.sqrt(np.mean((mh - xg) ** 2)))
        zh = (xg - mh) / np.sqrt(vh)
        row["nll_hand"] = float(np.mean(0.5 * zh ** 2 + 0.5 * np.log(
            2 * np.pi * vh)))
        row["cov_hand"] = float(np.mean(np.abs(zh) <= 1.645))
    mp_, vp = predict_q(tt, x, prop["params"], ttg)
    row["rmse_prop"] = float(np.sqrt(np.mean((mp_ - xg) ** 2)))
    zp = (xg - mp_) / np.sqrt(vp)
    row["nll_prop"] = float(np.mean(0.5 * zp ** 2 + 0.5 * np.log(
        2 * np.pi * vp)))
    row["cov_prop"] = float(np.mean(np.abs(zp) <= 1.645))
    return row


def stage_inspect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, i, ident, tt, x, ttg, xg in iter_notes(2):
        if ident and tt.size >= 80:
            break
    prop = fit_sm_proper(tt, x)
    print(f"note {key}#{i}: ev={prop['evidence']:.1f}, "
          f"components (w, mu Hz, v): {np.round(prop['components'], 3)}")
    grid = np.linspace(0.0, F_MAX, 301)
    pg = ls_periodogram(tt, x - x.mean(), grid)
    S = np.zeros_like(grid)
    for w, mu, v in prop["components"]:
        S += 0.5 * w * (np.exp(-(grid - mu) ** 2 / (2 * v))
                        + np.exp(-(grid + mu) ** 2 / (2 * v))) \
            / np.sqrt(2 * np.pi * v)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(grid, pg / pg.max(), color="#6B7280",
            label="least-squares periodogram (scaled)")
    ax.plot(grid, S / S.max(), color="#0072B2",
            label="learned spectral density (scaled)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.0)
    for _, mu, _ in prop["components"]:
        ax.axvline(mu, color="#0072B2", ls=":", lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(".cache/sm_proper_inspect.png", dpi=150)
    print("wrote .cache/sm_proper_inspect.png")


def stage_pilot(n_tracks: int = 4):
    rows, t_first = [], None
    t_start = time.time()
    seen_tracks = set()
    for key, i, ident, tt, x, ttg, xg in iter_notes(n_tracks):
        if t_first is None and key not in seen_tracks and len(seen_tracks) == 1:
            t_first = time.time() - t_start
            print(f"[timing] first track: {t_first/60:.1f} min "
                  f"({len(rows)} notes) -> projected "
                  f"{n_tracks * t_first/60:.0f} min for {n_tracks} tracks",
                  flush=True)
        seen_tracks.add(key)
        row = run_note(tt, x, ttg, xg)
        row.update({"key": key, "i": i, "ident": ident, "n": tt.size})
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} notes ({time.time()-t_start:.0f}s)",
                  flush=True)
            pickle.dump(rows, open(OUT, "wb"))
    pickle.dump(rows, open(OUT, "wb"))
    print(f"done: {len(rows)} notes, {(time.time()-t_start)/60:.1f} min")
    stage_report()


def stage_report():
    rows = pickle.load(open(OUT, "rb"))
    ok = [r for r in rows if "rmse_hand" in r]
    ident = [r for r in ok if r["ident"]]
    d_ev = np.array([r["ev_prop"] - r["ev_hand"] for r in ok])
    d_rmse = np.array([r["rmse_prop"] - r["rmse_hand"] for r in ok])
    d_nll = np.array([r["nll_prop"] - r["nll_hand"] for r in ok])
    print(f"n = {len(ok)} notes ({len(ident)} vibrato-identifiable)")
    print(f"evidence  (proper - hand): median {np.median(d_ev):+.2f}, "
          f"proper higher on {np.mean(d_ev > 0):.0%}")
    print(f"GT-frame RMSE delta: median {np.median(d_rmse):+.3f} cents")
    print(f"GT-frame NLL  delta: median {np.median(d_nll):+.3f}")
    print(f"coverage@90: proper {np.mean([r['cov_prop'] for r in ok]):.3f} "
          f"vs hand {np.mean([r['cov_hand'] for r in ok]):.3f}")
    if ident:
        print(f"vibrato band rediscovered (4-9 Hz, >=5% power): "
              f"{np.mean([r['vib_found'] for r in ident]):.0%} "
              f"of identifiable notes")
    print(f"wall: hand {np.median([r['t_hand'] for r in ok]):.1f}s vs "
          f"proper {np.median([r['t_prop'] for r in ok]):.1f}s per note")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "inspect":
        stage_inspect()
    elif cmd == "pilot":
        stage_pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        stage_report()
