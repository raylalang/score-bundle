#!/usr/bin/env python
"""SM-GP estimator vs the NLLS sine fit: the kill-cheap test (DEV ONLY).

The committed development study of docs/gp_everywhere_memo.md /
docs/sm_estimator_note.tex §7: on the vibrato-identifiable development
notes of URMP, fit the spectral-mixture GP (phase2.sm_estimator) and the
incumbent sine estimator (phase2.intonation.fit_vibrato_note) on the
tracked cents curve AND on the ground-truth cents curve of the same note,
then score four measures:

  M1 parameter accuracy (pass/fail): |tracked - own-GT reference| per
     quantity (c, log gamma, log f), paired per note, cluster bootstrap
     over tracks.  Each estimator is scored against ITS OWN GT-curve
     output, so neither is judged by the other's estimand.
  M2 calibration: z = (tracked - reference)/sqrt(var_tr + var_ref);
     coverage@90, mean NLL, PIT KS.  Wide (unidentified-curvature) cells
     are excluded from z and counted.
  M3 refused notes: where the sine fit's rule refuses the tracked curve
     but the GT curve is identifiable, does the SM posterior cover its
     GT reference?
  M4 curve level: both models fitted on tracked frames, scored on the
     ground-truth curve's frames (RMSE / NLL / coverage of the
     predictive band; sine band = fitted curve +- its residual sigma).

Reads the dev caches only; the confirmation split is never touched.
Exploratory, no claims.  Run (shardable) then report:

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_sm_estimator.py run [--shard K/N]
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_sm_estimator.py report
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

Z90 = 1.6448536269514722
QUANT = ("c", "lg", "lf")
QLABEL = {"c": "intonation c (cents)", "lg": "extent log gamma",
          "lf": "rate log f"}


def _note_frames(tsel, f0sel, midi):
    from score_bundle.phase2.intonation import cents_from_f0
    return cents_from_f0(f0sel, 440.0, float(midi - 69))


def _sm_row(out):
    return {"c": out["c"], "lg": float(np.log(max(out["gamma"], 1e-6))),
            "lf": float(np.log(max(out["f"], 1e-6))),
            "vc": out["var_c"], "vlg": out["var_log_gamma"],
            "vlf": out["var_log_f"], "wide": out["wide"],
            "params": out["params"]}


def run(shard_k: int, shard_n: int) -> None:
    from score_bundle.phase2.intonation import fit_vibrato_note
    from score_bundle.phase2.sm_estimator import fit_sm_note, sm_predict
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)
    from eval_phase2_real import dev_unique_tracks

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    keys = sorted(data)[shard_k - 1::shard_n]

    rows, t0 = [], time.time()
    for ki, key in enumerate(keys):
        d, f0c, tr = data[key], f0s[key], tracks[key]
        notes = read_notes_annotation(tr.notes)
        t_gt, f0_gt = read_f0_annotation(tr.f0s)
        # the note_targets frame rule, verbatim (targets.py)
        voiced = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
        floor = np.quantile(f0c["prob"][voiced], 0.2) if voiced.any() else 0.0
        voiced = voiced & (f0c["prob"] >= floor)
        ok_gt = np.isfinite(f0_gt) & (f0_gt > 0)

        for i in range(d["onset"].size):
            on, du = notes["onset"][i], notes["duration"][i]
            midi = d["midi"][i]
            rec = {"key": key, "i": i, "dur": float(du),
                   "instr": d["instrument"],
                   "ident_tr": bool(d["ident"][i]),
                   "ident_gt": bool(d["ident_gt"][i]),
                   "nl_tr": {"c": d["est"][i, 0], "lg": d["est"][i, 1],
                             "lf": d["est"][i, 2], "vc": d["var"][i, 0],
                             "vlg": d["var"][i, 1], "vlf": d["var"][i, 2]},
                   "nl_gt": {"c": d["est_gt"][i, 0], "lg": d["est_gt"][i, 1],
                             "lf": d["est_gt"][i, 2],
                             "vc": d["var_gt"][i, 0],
                             "vlg": d["var_gt"][i, 1],
                             "vlf": d["var_gt"][i, 2]}}
            sel = voiced & (f0c["t"] >= on) & (f0c["t"] < on + du)
            selg = ok_gt & (t_gt >= on) & (t_gt < on + du)
            tt, ttg = f0c["t"][sel] - on, t_gt[selg] - on
            x = _note_frames(tt, f0c["f0"][sel], midi)
            xg = _note_frames(ttg, f0_gt[selg], midi)
            rec["n_tr"], rec["n_gt"] = int(sel.sum()), int(selg.sum())
            if rec["n_tr"] >= 6:
                sm_tr = fit_sm_note(tt, x)
                rec["sm_tr"] = _sm_row(sm_tr)
                # curve-level scoring on the GT frames (M4), both models
                if rec["n_gt"] >= 6 and sm_tr["params"] is not None:
                    m, v = sm_predict(tt, x, sm_tr["params"], ttg)
                    z = (xg - m) / np.sqrt(v)
                    rec["curve_sm"] = {
                        "rmse": float(np.sqrt(np.mean((xg - m) ** 2))),
                        "nll": float(np.mean(0.5 * np.log(2 * np.pi * v)
                                             + 0.5 * z ** 2)),
                        "cov": float(np.mean(np.abs(z) <= Z90))}
                    nl = fit_vibrato_note(tt, x)
                    s2 = max(nl["sse"] / max(nl["n"] - 4, 1), 1e-8)
                    curve = (nl["c"] + nl["gamma"]
                             * np.sin(2 * np.pi * nl["f"]
                                      * (ttg - nl["delta"])))
                    zn = (xg - curve) / np.sqrt(s2)
                    rec["curve_nl"] = {
                        "rmse": float(np.sqrt(np.mean((xg - curve) ** 2))),
                        "nll": float(np.mean(0.5 * np.log(2 * np.pi * s2)
                                             + 0.5 * zn ** 2)),
                        "cov": float(np.mean(np.abs(zn) <= Z90))}
            if rec["n_gt"] >= 6:
                rec["sm_gt"] = _sm_row(fit_sm_note(ttg, xg))
            rows.append(rec)
        done, el = ki + 1, time.time() - t0
        print(f"track {done}/{len(keys)} {key} n={d['onset'].size} "
              f"elapsed {el:.0f}s (proj total {el / done * len(keys):.0f}s)",
              flush=True)

    out = f".cache/sm_dev_shard_{shard_k}_{shard_n}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} notes)")


def _boot_tracks(vals, keys, rng, B=2000):
    """Cluster bootstrap (over tracks) 95% CI of the mean."""
    uk = sorted(set(keys))
    per = {k: [] for k in uk}
    for v, k in zip(vals, keys):
        per[k].append(v)
    means = []
    for _ in range(B):
        pick = rng.choice(len(uk), len(uk), replace=True)
        pool = np.concatenate([per[uk[j]] for j in pick])
        means.append(pool.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def report() -> None:
    rows = []
    for f in sorted(glob.glob(".cache/sm_dev_shard_*.pkl")):
        rows += pickle.load(open(f, "rb"))
    print(f"{len(rows)} note records from "
          f"{len(glob.glob('.cache/sm_dev_shard_*.pkl'))} shard files\n")
    rng = np.random.default_rng(0)

    both = [r for r in rows if r["ident_tr"] and r["ident_gt"]
            and "sm_tr" in r and "sm_gt" in r
            and np.isfinite(r["nl_tr"]["lf"])
            and np.isfinite(r["nl_gt"]["lf"])]
    keys = [r["key"] for r in both]
    n_wide = sum(r["sm_tr"]["wide"] or r["sm_gt"]["wide"] for r in both)
    print(f"=== population: both-identifiable notes n={len(both)} "
          f"({len(set(keys))} tracks; SM wide on {n_wide})\n")

    print("=== M1 parameter accuracy: |tracked - own GT reference| "
          "(SM minus NLLS; negative = SM better) ===")
    for q in QUANT:
        e_sm = np.array([abs(r["sm_tr"][q] - r["sm_gt"][q]) for r in both])
        e_nl = np.array([abs(r["nl_tr"][q] - r["nl_gt"][q]) for r in both])
        dlt = e_sm - e_nl
        lo, hi = _boot_tracks(dlt, keys, rng)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {QLABEL[q]:24s} |err| median SM {np.median(e_sm):7.4f} "
              f"NLLS {np.median(e_nl):7.4f}  paired dmean "
              f"{dlt.mean():+8.4f} [{lo:+.4f},{hi:+.4f}]{star}")

    print("\n=== M2 calibration of the tracked estimate vs its reference "
          "(combined var) ===")
    for q in QUANT:
        for name, est, ref in (("SM  ", "sm_tr", "sm_gt"),
                               ("NLLS", "nl_tr", "nl_gt")):
            s2 = np.array([r[est]["v" + q] + r[ref]["v" + q] for r in both])
            df = np.array([r[est][q] - r[ref][q] for r in both])
            fin = np.isfinite(s2) & (s2 > 0)
            z = df[fin] / np.sqrt(s2[fin])
            nll = float(np.mean(0.5 * np.log(2 * np.pi * s2[fin])
                                + 0.5 * z ** 2))
            u = np.sort(0.5 * (1 + np.vectorize(_phi)(z)))
            ks = float(np.max(np.abs(u - (np.arange(u.size) + 0.5)
                                     / u.size)))
            print(f"  {QLABEL[q]:24s} {name} cov@90 "
                  f"{np.mean(np.abs(z) <= Z90):.3f}  NLL {nll:7.3f}  "
                  f"PIT-KS {ks:.3f}  (n={fin.sum()})")

    print("\n=== M3 notes the sine fit refuses (tracked) but GT "
          "identifiable ===")
    ref3 = [r for r in rows if not r["ident_tr"] and r["ident_gt"]
            and "sm_tr" in r and "sm_gt" in r]
    print(f"  n={len(ref3)}")
    for q in QUANT:
        s2 = np.array([r["sm_tr"]["v" + q] + r["sm_gt"]["v" + q]
                       for r in ref3])
        df = np.array([r["sm_tr"][q] - r["sm_gt"][q] for r in ref3])
        fin = np.isfinite(s2) & (s2 > 0)
        z = df[fin] / np.sqrt(s2[fin])
        scale = np.std([r["sm_gt"][q] for r in ref3])
        print(f"  {QLABEL[q]:24s} cov@90 {np.mean(np.abs(z) <= Z90):.3f} "
              f"(n={fin.sum()}, wide excl {len(ref3) - int(fin.sum())})  "
              f"median|err| {np.median(np.abs(df)):.4f} vs ref scale "
              f"{scale:.4f}")

    print("\n=== M4 curve level: fitted on tracked frames, scored on the "
          "GT curve's frames ===")
    cb = [r for r in both if "curve_sm" in r and "curve_nl" in r]
    ck = [r["key"] for r in cb]
    for met, better in (("rmse", "lower"), ("nll", "lower"),
                        ("cov", "closer to .90")):
        a = np.array([r["curve_sm"][met] for r in cb])
        b = np.array([r["curve_nl"][met] for r in cb])
        dlt = a - b
        lo, hi = _boot_tracks(dlt, ck, rng)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  frame {met:4s} ({better:14s}) SM {np.median(a):7.3f} "
              f"NLLS {np.median(b):7.3f}  paired dmean {dlt.mean():+8.3f} "
              f"[{lo:+.3f},{hi:+.3f}]{star}  (n={len(cb)})")


def _phi(x):
    """Standard normal error-function CDF helper (numpy-only)."""
    from math import erf, sqrt
    return erf(x / sqrt(2.0))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    if sys.argv[1] == "report":
        report()
        return
    shard_k, shard_n = 1, 1
    if "--shard" in sys.argv:
        shard = sys.argv[sys.argv.index("--shard") + 1]
        shard_k, shard_n = (int(v) for v in shard.split("/"))
    run(shard_k, shard_n)


if __name__ == "__main__":
    main()
