#!/usr/bin/env python
"""Phase-3 development study: waveform-likelihood intonation at scale.

Scales scripts/demo_phase3_waveform.py from two notes to hundreds across
instrument families, and tests the design constraint the feasibility demo
measured: does adding the WITHIN-NOTE DRIFT term to the position model
improve accuracy against ground truth and repair the overconfident
posterior widths?

Per note, two variants of the f0-curve model (eq:cents-curve/eq:vibrato,
vibrato scaffold from the Phase-2 estimator where identifiable, flat
otherwise; the vibrato scaffold is the one estimator-dependent ingredient
and is recorded as such):

  flat   c only                       (the demo's model)
  drift  c + slope * (t - midpoint)   (the drift-study term)

Inference is estimator-free: the c grid is centred on the SCORE-nominal
pitch (c = 0), +/-50 cents coarse, 0.03-cent fine refinement; the drift
variant adds a coarse slope grid (+/-40 cents/s) and refines c at the
best slope (the reported posterior is conditional on the profiled slope).
Noise variance is refit at the coarse optimum before the fine pass.

Scored per note against the quasi-truth centre (same NLLS estimator on
URMP's ground-truth curve): absolute error, z-score, coverage@90; the
Phase-2 estimator's error is reported alongside where it exists.

Sharded: `run k/n` writes results/phase3_cells/wave.shard{k}of{n}.pkl;
`report` merges into results/phase3_waveform_dev.md. DEV tracks only.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_waveform_dev.py run 0/8
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_waveform_dev.py report
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

SR = 16000
N_HARM = 8
N_CHUNK = 4
AMP_VAR = 10.0
MAX_NOTES_PER_TRACK = 60
MAX_SEG_S = 2.0                     # cap segment length (cost control)
INSTRUMENTS = ("vn", "vc", "fl", "cl", "sax", "tpt", "tbn")
CELLS_DIR = "results/phase3_cells"
OUT_MD = "results/phase3_waveform_dev.md"
_Z90 = 1.6448536269514722


def f0_curve(t, midi, c, slope=0.0, gamma=0.0, f=0.0, delta=0.0):
    cents = c + slope * (t - t.mean())
    if gamma > 0 and f > 0:
        cents = cents + np.where(
            t >= delta, gamma * np.sin(2 * np.pi * f * (t - delta)), 0.0)
    return 440.0 * 2.0 ** ((midi - 69.0 + cents / 100.0) / 12.0)


def chunked_design(f0, t):
    from score_bundle.phase3.synth import harmonic_design_matrix
    base = harmonic_design_matrix(f0, t, N_HARM)
    edges = np.linspace(0, t.size, N_CHUNK + 1).astype(int)
    cols = []
    for q in range(N_CHUNK):
        blk = np.zeros_like(base)
        blk[edges[q]:edges[q + 1]] = base[edges[q]:edges[q + 1]]
        cols.append(blk)
    return np.concatenate(cols, axis=1)


def loglik(x, t, midi, c, nv, **kw):
    from score_bundle.phase3.waveform_model import collapsed_loglik_lowrank
    Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
    Sigma_a = np.eye(Phi.shape[1]) * AMP_VAR
    return collapsed_loglik_lowrank(x, Phi, Sigma_a, noise_var=nv)


def fit_noise(x, t, midi, c, **kw):
    Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
    beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
    r = x - Phi @ beta
    return float(r @ r / max(x.size - Phi.shape[1], 1))


def posterior_stats(grid, lls):
    w = np.exp(lls - lls.max())
    w /= w.sum()
    mean = float(w @ grid)
    sd = float(np.sqrt(w @ (grid - mean) ** 2))
    return mean, max(sd, 1e-6)


def make_whitener(rho):
    """AR(1) whitening filter: y_t = x_t - rho x_{t-1} (first sample scaled).

    Applied identically to the audio and every column of Phi, this maps the
    AR(1)-noise likelihood to the white-noise one; the Jacobian is constant
    across the c grid (rho fixed per note), so posterior comparisons stand.
    """
    def w(v):
        out = v.copy()
        out[1:] = v[1:] - rho * v[:-1]
        out[0] = v[0] * np.sqrt(max(1.0 - rho * rho, 1e-6))
        return out
    return w


def fit_rho(x, t, midi, c, **kw):
    """Residual lag-1 autocorrelation at the given curve (the AR(1) fit)."""
    Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
    beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
    r = x - Phi @ beta
    rho = float(r[1:] @ r[:-1] / max(r @ r, 1e-12))
    return float(np.clip(rho, 0.0, 0.999))


def infer_c(x, t, midi, drift, scaffold, ar1=False):
    """Score-centred grid inference; returns (mean, sd, slope_hat, rho)."""
    rho = fit_rho(x, t, midi, 0.0, **scaffold) if ar1 else 0.0
    wh = make_whitener(rho) if ar1 else None
    xw = wh(x) if ar1 else x

    def ll(c, nv, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        if wh is not None:
            Phi = np.apply_along_axis(wh, 0, Phi)
        from score_bundle.phase3.waveform_model import \
            collapsed_loglik_lowrank
        Sigma_a = np.eye(Phi.shape[1]) * AMP_VAR
        return collapsed_loglik_lowrank(xw, Phi, Sigma_a, noise_var=nv)

    def noise(c, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        if wh is not None:
            Phi = np.apply_along_axis(wh, 0, Phi)
        beta, *_ = np.linalg.lstsq(Phi, xw, rcond=None)
        r = xw - Phi @ beta
        return float(r @ r / max(xw.size - Phi.shape[1], 1))

    coarse = np.arange(-50.0, 50.0 + 1e-9, 1.0)
    slopes = np.linspace(-40.0, 40.0, 9) if drift else np.array([0.0])
    nv = noise(0.0, **scaffold)
    best = (-np.inf, 0.0, 0.0)
    for s in slopes:
        kw = dict(scaffold, slope=float(s))
        lls = np.array([ll(c, nv, **kw) for c in coarse])
        j = int(np.argmax(lls))
        if lls[j] > best[0]:
            best = (float(lls[j]), float(coarse[j]), float(s))
    _, c0, s_hat = best
    kw = dict(scaffold, slope=s_hat)
    nv = noise(c0, **kw)
    if drift:                        # refine the slope at the coarse c
        sfine = s_hat + np.linspace(-6.0, 6.0, 13)
        lls = np.array([ll(c0, nv, **dict(scaffold, slope=float(s)))
                        for s in sfine])
        s_hat = float(sfine[int(np.argmax(lls))])
        kw = dict(scaffold, slope=s_hat)
    fine = c0 + np.arange(-3.0, 3.0 + 1e-9, 0.03)
    lls = np.array([ll(c, nv, **kw) for c in fine])
    mean, sd = posterior_stats(fine, lls)
    return mean, sd, s_hat, rho


def infer_c_devprior(x, t, midi, scaffold, n_bump=8, dev_sd=5.0):
    """Deviation-prior variant: coarse pass as in `drift`, then a fine pass
    with the collapsed likelihood augmented by marginalized smooth pitch-curve
    deviations.

    Deviations delta(t) live on a zero-mean Hann-bump basis (n_bump windows,
    50% overlap, each de-meaned so a constant shift stays identified as c);
    the waveform Jacobian wrt each coefficient is finite-differenced at the
    per-candidate LS amplitudes and appended to Phi with prior sd `dev_sd`
    cents per coefficient --- marginalized exactly like the amplitudes.
    Returns (mean, sd, slope_hat, 0.0).
    """
    from score_bundle.phase3.waveform_model import collapsed_loglik_lowrank

    # coarse: reuse the white-noise drift search for location only
    def ll_white(c, nv, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        Sigma_a = np.eye(Phi.shape[1]) * AMP_VAR
        return collapsed_loglik_lowrank(x, Phi, Sigma_a, noise_var=nv)

    def noise_at(c, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        r = x - Phi @ beta
        return float(r @ r / max(x.size - Phi.shape[1], 1))

    coarse = np.arange(-50.0, 50.0 + 1e-9, 1.0)
    nv = noise_at(0.0, **scaffold)
    best = (-np.inf, 0.0, 0.0)
    for s in np.linspace(-40.0, 40.0, 9):
        kw = dict(scaffold, slope=float(s))
        lls = np.array([ll_white(c, nv, **kw) for c in coarse])
        j = int(np.argmax(lls))
        if lls[j] > best[0]:
            best = (float(lls[j]), float(coarse[j]), float(s))
    _, c0, s_hat = best
    kw = dict(scaffold, slope=s_hat)

    # zero-mean Hann bump basis over the note
    m = t.size
    centers = np.linspace(0, m - 1, n_bump + 2)[1:-1]
    half = (centers[1] - centers[0]) if n_bump > 1 else m / 2.0
    bumps = []
    idx = np.arange(m)
    for cm in centers:
        b = np.clip(1.0 - np.abs(idx - cm) / half, 0.0, None)
        b = 0.5 - 0.5 * np.cos(np.pi * np.clip(
            1.0 - np.abs(idx - cm) / half, 0.0, 1.0) * 2.0)
        b -= b.mean()
        bumps.append(b)

    def ll_dev(c, nv):
        f0 = f0_curve(t, midi, c, **kw)
        Phi = chunked_design(f0, t)
        a_hat, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        mean_wave = Phi @ a_hat
        eps = 1.0
        J = np.empty((m, len(bumps)))
        base_cents_extra = np.zeros(m)
        for jb, b in enumerate(bumps):
            f0p = f0 * 2.0 ** (eps * b / 1200.0)
            J[:, jb] = (chunked_design(f0p, t) @ a_hat - mean_wave) / eps
        Phi_aug = np.concatenate([Phi, J], axis=1)
        p_amp = Phi.shape[1]
        Sig = np.diag(np.concatenate([np.full(p_amp, AMP_VAR),
                                      np.full(len(bumps), dev_sd ** 2)]))
        return collapsed_loglik_lowrank(x, Phi_aug, Sig, noise_var=nv)

    nv = noise_at(c0, **kw)
    fine = c0 + np.arange(-3.0, 3.0 + 1e-9, 0.06)
    lls = np.array([ll_dev(c, nv) for c in fine])
    mean, sd = posterior_stats(fine, lls)
    return mean, sd, s_hat, 0.0


def eligible_notes(d, notes):
    n = len(d["ident"])
    gt_c = d["est_gt"][:, 0]
    out = [i for i in range(n)
           if np.isfinite(gt_c[i]) and abs(gt_c[i]) <= 150.0
           and notes["duration"][i] >= 0.25]
    rng = np.random.default_rng(0)
    if len(out) > MAX_NOTES_PER_TRACK:
        out = sorted(rng.choice(out, MAX_NOTES_PER_TRACK, replace=False))
    return out


def selected():
    from eval_phase2_real import dev_unique_tracks
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    picks = []
    seen = set()
    for key, d in sorted(data.items()):
        ins = d["instrument"]
        if ins in INSTRUMENTS and ins not in seen:
            seen.add(ins)
            picks.append((key, d, tracks[key]))
    return picks


def stage_run(shard: str) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    from score_bundle.phase2.urmp import read_notes_annotation

    k, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    rows = []
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        idx = eligible_notes(d, notes)
        mine = [i for i in idx if (gidx + idx.index(i)) % nsh == k]
        gidx += len(idx)
        if not mine:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, dtype=float), SR, int(sr48))
        for i in mine:
            t0 = time.time()
            on, du = notes["onset"][i], min(notes["duration"][i], MAX_SEG_S)
            a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
            x = audio[max(a, 0):min(b, audio.size)]
            if x.size < SR // 8:
                continue
            t = np.arange(x.size) / SR
            midi = float(d["midi"][i])
            scaffold = {}
            if d["ident"][i] and np.isfinite(d["est"][i, 1]):
                scaffold = dict(gamma=float(np.exp(d["est"][i, 1])),
                                f=float(np.exp(d["est"][i, 2])),
                                delta=float(d["dvib"][i])
                                if np.isfinite(d["dvib"][i]) else 0.0)
            rec = {"key": key, "i": i, "instr": d["instrument"],
                   "dur": float(du), "n_frames": int(d["n_frames"][i]),
                   "gt_c": float(d["est_gt"][i, 0]),
                   "est_c": float(d["est"][i, 0]),
                   "est_sd": float(np.sqrt(d["var"][i, 0]))
                   if np.isfinite(d["var"][i, 0]) else np.nan}
            for name, drift, ar1 in (("flat", False, False),
                                     ("drift", True, False),
                                     ("ar1", True, True)):
                rec[name] = infer_c(x, t, midi, drift, scaffold, ar1)
            rows.append(rec)
            print(f"{key} note {i} ({d['instrument']}): flat "
                  f"{rec['flat'][0]:+.2f}+/-{rec['flat'][1]:.2f}, drift "
                  f"{rec['drift'][0]:+.2f}+/-{rec['drift'][1]:.2f}, ar1 "
                  f"{rec['ar1'][0]:+.2f}+/-{rec['ar1'][1]:.2f} "
                  f"(rho={rec['ar1'][3]:.3f}) gt {rec['gt_c']:+.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    out = f"{CELLS_DIR}/wave.shard{k}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} notes)")


def stage_report() -> None:
    import glob
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/wave.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards found")
        return

    def stats(variant):
        err = np.array([abs(r[variant][0] - r["gt_c"]) for r in rows])
        z = np.array([(r[variant][0] - r["gt_c"]) / r[variant][1]
                      for r in rows])
        cov = float(np.mean(np.abs(z) <= _Z90))
        return err, z, cov

    lines = ["# Phase 3: waveform-likelihood intonation at scale (DEV)\n",
             f"\n{len(rows)} notes, tracks: "
             + ", ".join(sorted({f"{r['key']}({r['instr']})"
                                 for r in rows})) + "\n",
             "\n| variant | median abs err (cents) | q90 | median sd | "
             "median abs z | cov@90 |\n|---|---|---|---|---|---|\n"]
    for v in ("flat", "drift", "ar1"):
        err, z, cov = stats(v)
        sd = np.array([r[v][1] for r in rows])
        lines.append(f"| {v} | {np.median(err):.2f} | "
                     f"{np.quantile(err, .9):.2f} | {np.median(sd):.3f} | "
                     f"{np.median(np.abs(z)):.1f} | {cov:.2f} |\n")
    est_err = np.array([abs(r["est_c"] - r["gt_c"]) for r in rows
                        if np.isfinite(r["est_c"])])
    lines.append(f"\nestimator |err| median {np.median(est_err):.2f} "
                 f"(n={est_err.size} notes with estimates)\n")
    d_err = np.array([abs(r["drift"][0] - r["gt_c"])
                      - abs(r["flat"][0] - r["gt_c"]) for r in rows])
    lines.append(f"drift-flat paired |err|: median {np.median(d_err):+.3f}, "
                 f"drift better on {np.mean(d_err < 0):.0%} of notes\n")
    rhos = np.array([r["ar1"][3] for r in rows])
    lines.append(f"AR(1) residual rho: median {np.median(rhos):.3f}, "
                 f"q10/q90 {np.quantile(rhos, .1):.3f}/"
                 f"{np.quantile(rhos, .9):.3f}\n")
    lines.append("\nPer family (median abs err flat / drift / estimator):\n")
    for fam, mem in (("strings", ("vn", "va", "vc", "db")),
                     ("winds", ("fl", "cl", "ob", "sax", "bn")),
                     ("brass", ("tpt", "hn", "tbn", "tba"))):
        sub = [r for r in rows if r["instr"] in mem]
        if not sub:
            continue
        ef = np.median([abs(r["flat"][0] - r["gt_c"]) for r in sub])
        ed = np.median([abs(r["drift"][0] - r["gt_c"]) for r in sub])
        ee = np.median([abs(r["est_c"] - r["gt_c"]) for r in sub
                        if np.isfinite(r["est_c"])])
        lines.append(f"- {fam} (n={len(sub)}): {ef:.2f} / {ed:.2f} / "
                     f"{ee:.2f}\n")
    open(OUT_MD, "w").writelines(lines)
    print("".join(lines))
    print(f"wrote {OUT_MD}")


def stage_rundev(shard: str) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    from score_bundle.phase2.urmp import read_notes_annotation

    k, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    rows = []
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        idx = eligible_notes(d, notes)
        mine = [i for i in idx if (gidx + idx.index(i)) % nsh == k]
        gidx += len(idx)
        if not mine:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, dtype=float), SR, int(sr48))
        for i in mine:
            t0 = time.time()
            on, du = notes["onset"][i], min(notes["duration"][i], MAX_SEG_S)
            a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
            x = audio[max(a, 0):min(b, audio.size)]
            if x.size < SR // 8:
                continue
            t = np.arange(x.size) / SR
            scaffold = {}
            if d["ident"][i] and np.isfinite(d["est"][i, 1]):
                scaffold = dict(gamma=float(np.exp(d["est"][i, 1])),
                                f=float(np.exp(d["est"][i, 2])),
                                delta=float(d["dvib"][i])
                                if np.isfinite(d["dvib"][i]) else 0.0)
            r = infer_c_devprior(x, t, float(d["midi"][i]), scaffold)
            rows.append({"key": key, "i": i, "dev8": r,
                         "gt_c": float(d["est_gt"][i, 0])})
            print(f"{key} note {i}: dev8 {r[0]:+.2f}+/-{r[1]:.2f} gt "
                  f"{d['est_gt'][i, 0]:+.2f} [{time.time() - t0:.0f}s]",
                  flush=True)
    out = f"{CELLS_DIR}/wavedev.shard{k}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} notes)")


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "report"
    if verb == "run":
        stage_run(sys.argv[2])
    elif verb == "rundev":
        stage_rundev(sys.argv[2])
    elif verb == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown verb {verb}")
