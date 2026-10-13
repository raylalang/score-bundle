#!/usr/bin/env python
"""Phase-3 overnight dev study: transition-region boundaries on joint chains.

Follow-up to results/phase3_boundary_pilot_dev.md: the gradient boundary
mechanism works but converges to the acoustic transition, 11-87 ms from
the annotated onset — a transition is a region, not a point.  This study
models the region: boundary k = note-(k+1) fade-in edge b_lo + note-k
fade-out edge b_hi = b_lo + width, lognormal width prior, joint gradient
MAP over 3-5-note chains (curve.fit_chain; per-note curves under the
evidence-fitted prior).

Committed questions, stated before the run:
  (a) Does the fade-in edge land closer to the annotated onset than the
      pilot's fixed-softness point boundary (median ~25 ms)?  And how
      often does the annotation fall inside [b_lo, b_hi] (region
      coverage)?
  (b) Does the free width beat the FIXED-width control (10 ms, same
      chains, same perturbations) on edge error — or does the width
      absorb pitch error instead (edge error vs width correlation)?
  (c) Deployment arm: boundary error when initialized from the local
      tempo-line prediction (annotated onset minus tau — score + tempo
      information only), the genuine no-annotation situation.
  (d) Width distribution per instrument family; unison boundaries are
      effectively unidentified by a harmonic likelihood and are reported
      as a separate stratum, never averaged in (octaves flagged too).

Arms per chain: free-width recovery (init = annotated +/-60 ms),
fixed-width recovery (same deltas), free-width deployment (nominal
init).  Shared per-chain reference noise/c-init per protocol
(curve.chain_reference — the pilot's noise-at-perturbed-init confound is
fixed).  Curve scoring: GT-frame RMSE per note with and without a
+/-30 ms guard around annotated boundaries (GT frames are least reliable
exactly there).

Sharded: `run k/n` writes results/phase3_cells/chain.shard{k}of{n}.pkl;
`report` merges into results/phase3_chain_dev.md (tables only; readings
appended by hand — report reruns CLOBBER).  `smoke` times one full fit
for the wall-time projection (the long-run rule).  DEV only; no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_chain_dev.py smoke
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_chain_dev.py run 0/8
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_chain_dev.py report
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_waveform_dev import SR, selected
from eval_phase3_curve_dev import load_hypers

CELLS_DIR = "results/phase3_cells"
OUT_MD = "results/phase3_chain_dev.md"
CHAIN_MIN, CHAIN_MAX = 3, 5
MAX_CHAINS_PER_TRACK = 20
MAX_SPAN_S = 7.0
NOTE_MIN_S, NOTE_MAX_S = 0.25, 2.0
GAP_LO, GAP_HI = -0.05, 0.04
DELTAS_MS = (-60.0, 60.0)
N_STEPS = 500
GUARD_S = 0.03
FAMS = (("strings", ("vn", "va", "vc", "db")),
        ("winds", ("fl", "cl", "ob", "sax", "bn")),
        ("brass", ("tpt", "hn", "tbn", "tba")))


def select_chains(notes):
    """Greedy non-overlapping chains of 3-5 consecutive eligible notes."""
    on = np.asarray(notes["onset"], float)
    du = np.asarray(notes["duration"], float)
    n = on.size
    chains, i = [], 0
    while i < n - CHAIN_MIN + 1 and len(chains) < MAX_CHAINS_PER_TRACK:
        if not (NOTE_MIN_S <= du[i] <= NOTE_MAX_S):
            i += 1
            continue
        idx = [i]
        j = i
        while (len(idx) < CHAIN_MAX and j + 1 < n
               and NOTE_MIN_S <= du[j + 1] <= NOTE_MAX_S
               and GAP_LO <= on[j + 1] - (on[j] + du[j]) <= GAP_HI
               and (on[j + 1] + du[j + 1]) - on[i] <= MAX_SPAN_S):
            idx.append(j + 1)
            j += 1
        if len(idx) >= CHAIN_MIN:
            chains.append(idx)
            i = idx[-1] + 1
        else:
            i += 1
    return chains


def chain_segment(audio, notes, idx):
    on = np.asarray(notes["onset"], float)
    du = np.asarray(notes["duration"], float)
    t0 = float(on[idx[0]])
    t1 = float(on[idx[-1]] + min(du[idx[-1]], NOTE_MAX_S))
    a, b = int(t0 * SR), int(t1 * SR)
    x = audio[max(a, 0):min(b, audio.size)]
    t = np.arange(x.size) / SR
    b_annot = on[idx[1:]] - t0
    return x, t, t0, b_annot


def gt_frames_for_note(key, i, data, f0s, tracks, note_curves, t0):
    """GT frame (segment-time, cents) pairs for note i; None if too few."""
    try:
        _, _, ttg, xg = note_curves(key, i, data, f0s, tracks)
    except Exception:
        return None
    if ttg.size < 10:
        return None
    on_i = float(data[key]["onset"][i])
    return on_i - t0 + ttg, xg


def score_chain(fit, b_annot, midis, gts, cuts):
    """Per-boundary and per-note scores for one arm's fit."""
    k_n = len(midis)
    bres = []
    for j in range(k_n - 1):
        bres.append({
            "edge_ms": abs(fit.b_lo[j] - b_annot[j]) * 1000.0,
            "center_ms": abs(0.5 * (fit.b_lo[j] + fit.b_hi[j])
                             - b_annot[j]) * 1000.0,
            "covered": bool(fit.b_lo[j] <= b_annot[j] <= fit.b_hi[j]),
            "width_ms": fit.widths[j] * 1000.0,
            "interval": float(midis[j + 1] - midis[j]),
        })
    nres = []
    for k in range(k_n):
        rec = {"mass": float(fit.window_mass[k])}
        if gts[k] is not None:
            gt_t, gt_x = gts[k]
            lo, hi = cuts[k], cuts[k + 1]
            sel = (gt_t >= lo) & (gt_t <= hi)
            selg = sel.copy()
            if k > 0:
                selg &= gt_t >= lo + GUARD_S
            if k < k_n - 1:
                selg &= gt_t <= hi - GUARD_S
            for name, s in (("rmse", sel), ("rmse_g", selg)):
                rec[name] = (float(np.sqrt(np.mean(
                    (fit.curve(k, gt_t[s]) - gt_x[s]) ** 2)))
                    if s.sum() >= 5 else np.nan)
        else:
            rec["rmse"] = rec["rmse_g"] = np.nan
        nres.append(rec)
    return bres, nres


def iter_chains():
    import soundfile as sf
    from scipy.signal import resample_poly

    from eval_phase2_real import dev_unique_tracks
    from make_phase2_intro import note_curves
    from score_bundle.phase2.urmp import read_notes_annotation

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    track_map = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        chains = select_chains(notes)
        if not chains:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        tau = d["tau"]
        for idx in chains:
            x, t, t0, b_annot = chain_segment(audio, notes, idx)
            midis = [float(d["midi"][i]) for i in idx]
            b_nom = b_annot.copy()
            for j, i in enumerate(idx[1:]):
                if np.isfinite(tau[i]):
                    b_nom[j] = b_annot[j] - float(tau[i])
            gts = [gt_frames_for_note(key, i, data, f0s, track_map,
                                      note_curves, t0) for i in idx]
            cuts = np.concatenate([[0.0], b_annot, [t[-1]]])
            yield gidx, dict(key=key, idx=idx, instr=d["instrument"],
                             x=x, t=t, midis=midis, b_annot=b_annot,
                             b_nom=b_nom, gts=gts, cuts=cuts)
            gidx += 1


def run_arms(ch, hypers, n_steps=N_STEPS):
    from score_bundle.phase3.curve import chain_reference, fit_chain

    x, t, midis = ch["x"], ch["t"], ch["midis"]
    c_ref, nv_ref = chain_reference(x, t, midis, ch["b_annot"])
    c_nom, nv_nom = chain_reference(x, t, midis, ch["b_nom"],
                                    guard_frac=0.2)
    arms = {}
    for dm in DELTAS_MS:
        for freew in (True, False):
            tag = f"{'free' if freew else 'fixed'}{dm:+.0f}"
            t0_ = time.time()
            fit = fit_chain(x, t, midis, ch["b_annot"] + dm / 1000.0,
                            hypers=hypers, c_init=c_ref,
                            noise_var=nv_ref, free_width=freew,
                            n_steps=n_steps)
            arms[tag] = (fit, time.time() - t0_)
    t0_ = time.time()
    fit = fit_chain(x, t, midis, ch["b_nom"], hypers=hypers, c_init=c_nom,
                    noise_var=nv_nom, free_width=True, n_steps=n_steps)
    arms["nominal"] = (fit, time.time() - t0_)
    return arms


def stage_smoke():
    hypers = load_hypers()
    for gidx, ch in iter_chains():
        from score_bundle.phase3.curve import chain_reference, fit_chain
        k_n = len(ch["midis"])
        print(f"smoke chain: {ch['key']} notes {ch['idx']} "
              f"({ch['instr']}, {k_n} notes, {ch['t'][-1]:.1f}s, "
              f"m={ch['x'].size})", flush=True)
        t0_ = time.time()
        c_ref, nv_ref = chain_reference(ch["x"], ch["t"], ch["midis"],
                                        ch["b_annot"])
        t_ref = time.time() - t0_
        t0_ = time.time()
        fit = fit_chain(ch["x"], ch["t"], ch["midis"],
                        ch["b_annot"] + 0.06, hypers=hypers, c_init=c_ref,
                        noise_var=nv_ref, n_steps=N_STEPS)
        t_fit = time.time() - t0_
        bres, _ = score_chain(fit, ch["b_annot"], ch["midis"], ch["gts"],
                              ch["cuts"])
        n_chains = sum(1 for _ in iter_chains())
        n_fits = n_chains * 5
        proj_h = (n_fits * t_fit + n_chains * 2 * t_ref) / 3600.0
        print(f"reference {t_ref:.0f}s, fit {t_fit:.0f}s "
              f"({N_STEPS} steps); edges "
              + ", ".join(f"{b['edge_ms']:.1f}ms(w{b['width_ms']:.0f})"
                          for b in bres), flush=True)
        print(f"PROJECTION: {n_chains} chains x 5 fits = {n_fits} fits "
              f"-> {proj_h:.1f} h CPU / 8 shards = {proj_h / 8:.1f} h "
              f"wall", flush=True)
        return
    print("no chains found")


def stage_run(shard: str) -> None:
    k_sh, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    hypers = load_hypers()
    rows = []
    for gidx, ch in iter_chains():
        if gidx % nsh != k_sh:
            continue
        arms = run_arms(ch, hypers)
        rec = {"key": ch["key"], "idx": ch["idx"], "instr": ch["instr"],
               "midis": ch["midis"], "span": float(ch["t"][-1]),
               "init_err_nom_ms": list(np.abs(ch["b_nom"] - ch["b_annot"])
                                       * 1000.0)}
        for tag, (fit, wall) in arms.items():
            bres, nres = score_chain(fit, ch["b_annot"], ch["midis"],
                                     ch["gts"], ch["cuts"])
            rec[tag] = {"b": bres, "n": nres, "wall": wall,
                        "b_lo": list(fit.b_lo),
                        "nv0": fit.noise_var_init, "nv": fit.noise_var}
        rows.append(rec)
        eg = rec["free+60"]["b"]
        print(f"{ch['key']} notes {ch['idx']} ({ch['instr']}): free+60 "
              f"edges " + ",".join(f"{b['edge_ms']:.0f}" for b in eg)
              + f" ms, widths "
              + ",".join(f"{b['width_ms']:.0f}" for b in eg)
              + f" [{sum(a[1] for a in arms.values()):.0f}s total]",
              flush=True)
    out = f"{CELLS_DIR}/chain.shard{k_sh}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} chains)")


def _collect(rows, tag, field, unison_sel="exclude"):
    out = []
    for r in rows:
        for b in r[tag]["b"]:
            uni = abs(b["interval"]) < 0.5
            if unison_sel == "exclude" and uni:
                continue
            if unison_sel == "only" and not uni:
                continue
            out.append(b[field])
    return np.asarray(out, float)


def stage_report() -> None:
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/chain.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards found")
        return
    arms = ["free-60", "free+60", "fixed-60", "fixed+60", "nominal"]
    n_b = sum(len(r["free+60"]["b"]) for r in rows)
    n_uni = int(_collect(rows, "free+60", "edge_ms", "only").size)

    lines = [
        "# Phase 3: transition-region boundaries on joint chains (DEV, "
        "exploratory, no claims)\n",
        "\nJoint gradient MAP over 3-5-note chains: per-note curves "
        "(evidence-fitted prior) +\nper-boundary fade-in/fade-out edge "
        "pair (width under a lognormal prior), amplitudes\ncollapsed "
        "exactly (curve.fit_chain).  Committed questions in the script "
        "docstring\n(scripts/eval_phase3_chain_dev.py): (a) fade-in edge "
        "vs the pilot's point boundary +\nregion coverage of the "
        "annotation; (b) free vs FIXED-width control; (c) deployment\n"
        "from score-nominal (tempo-line) inits; (d) per-family widths, "
        "unisons stratified out.\nPilot baseline: median 25 ms "
        "(results/phase3_boundary_pilot_dev.md).\n",
        f"\n{len(rows)} chains, {n_b} boundaries ({n_uni} unison, "
        "excluded from the main rows), tracks: "
        + ", ".join(sorted({f"{r['key']}({r['instr']})" for r in rows}))
        + "\n",
        "\n| arm | median edge err (ms) | q90 | median center err | "
        "region covers annot | median width (ms) |\n|---|---|---|---|"
        "---|---|\n"]
    for tag in arms:
        e = _collect(rows, tag, "edge_ms")
        ce = _collect(rows, tag, "center_ms")
        cov = _collect(rows, tag, "covered")
        w = _collect(rows, tag, "width_ms")
        lines.append(f"| {tag} | {np.median(e):.1f} | "
                     f"{np.quantile(e, .9):.1f} | {np.median(ce):.1f} | "
                     f"{np.mean(cov):.0%} | {np.median(w):.0f} |\n")

    # (b) paired free vs fixed, same chain + delta
    dpair = []
    for r in rows:
        for dm in ("-60", "+60"):
            for bf, bx in zip(r[f"free{dm}"]["b"], r[f"fixed{dm}"]["b"]):
                if abs(bf["interval"]) < 0.5:
                    continue
                dpair.append(bf["edge_ms"] - bx["edge_ms"])
    dpair = np.asarray(dpair)
    lines.append(f"\nfree − fixed paired edge error: median "
                 f"{np.median(dpair):+.1f} ms, free better on "
                 f"{np.mean(dpair < 0):.0%} of boundaries (n="
                 f"{dpair.size})\n")
    e_all = _collect(rows, "free+60", "edge_ms")
    w_all = _collect(rows, "free+60", "width_ms")
    ok = np.isfinite(e_all) & np.isfinite(w_all)
    corr = np.corrcoef(np.log(np.maximum(w_all[ok], 1.0)),
                       e_all[ok])[0, 1]
    lines.append(f"edge error vs log width (free+60): Pearson r = "
                 f"{corr:+.2f}\n")

    # start-independence: spread between the two recovery inits
    spread = []
    for r in rows:
        for j in range(len(r["free+60"]["b_lo"])):
            spread.append(abs(r["free+60"]["b_lo"][j]
                              - r["free-60"]["b_lo"][j]) * 1000.0)
    lines.append(f"recovery spread |b_lo(+60) − b_lo(−60)|: median "
                 f"{np.median(spread):.1f} ms, q90 "
                 f"{np.quantile(spread, .9):.1f} ms\n")

    # (c) deployment
    e_nom = _collect(rows, "nominal", "edge_ms")
    init_nom = np.asarray([v for r in rows
                           for v in r["init_err_nom_ms"]])
    lines.append(f"\ndeployment (score-nominal init): median edge err "
                 f"{np.median(e_nom):.1f} ms (q90 "
                 f"{np.quantile(e_nom, .9):.1f}) from median init err "
                 f"{np.median(init_nom):.1f} ms\n")

    # (d) per family + strata
    lines.append("\nPer family, free+60: median edge err / width "
                 "(non-unison boundaries):\n")
    for fam, mem in FAMS:
        sub = [r for r in rows if r["instr"] in mem]
        if not sub:
            continue
        e = _collect(sub, "free+60", "edge_ms")
        w = _collect(sub, "free+60", "width_ms")
        lines.append(f"- {fam} (n={e.size}): {np.median(e):.1f} ms / "
                     f"{np.median(w):.0f} ms\n")
    e_uni = _collect(rows, "free+60", "edge_ms", "only")
    if e_uni.size:
        lines.append(f"- unison stratum (n={e_uni.size}): median edge "
                     f"err {np.median(e_uni):.1f} ms (expected poor — "
                     "harmonic likelihood carries no unison boundary)\n")

    # curves + dropout + wall
    lines.append("\nCurve GT-frame RMSE (cents), median over notes with "
                 ">=5 frames (guarded ±30 ms at boundaries / "
                 "unguarded):\n")
    for tag in ("free+60", "nominal"):
        rg = np.asarray([n["rmse_g"] for r in rows for n in r[tag]["n"]])
        ru = np.asarray([n["rmse"] for r in rows for n in r[tag]["n"]])
        lines.append(f"- {tag}: "
                     f"{np.nanmedian(rg):.2f} / {np.nanmedian(ru):.2f} "
                     f"(single-note batch reference 3.16)\n")
    mass = np.asarray([n["mass"] for r in rows
                       for n in r["free+60"]["n"]])
    walls = np.asarray([r[tag]["wall"] for r in rows for tag in arms])
    lines.append(f"\nwindow mass: min {mass.min():.2f}, fraction < 0.5 "
                 f"= {np.mean(mass < 0.5):.1%} (dropout detector)\n")
    lines.append(f"wall per fit: median {np.median(walls):.0f} s, q90 "
                 f"{np.quantile(walls, .9):.0f} s\n")
    open(OUT_MD, "w").writelines(lines)
    print("".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "report"
    if verb == "run":
        stage_run(sys.argv[2])
    elif verb == "smoke":
        stage_smoke()
    elif verb == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown verb {verb}")
