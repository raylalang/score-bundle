#!/usr/bin/env python
"""Phase-3 batch dev study: the curve-from-waveform posterior at scale.

Scales results/phase3_curve_proto_dev.md (three notes) to the full
development protocol of eval_phase3_waveform_dev (same 7 one-per-
instrument dev tracks, same eligibility and rng(0) subsample, so the
populations are comparable), with the packaged differentiable model
(score_bundle.phase3.curve: gradient MAP + full joint Laplace).

Committed questions, stated before the run:
  (a) does the waveform curve posterior beat the pYIN frames on
      GT-frame RMSE, per instrument family?
  (b) is curve-level coverage@90 near nominal — the August scalar
      study's estimand-gap caveat re-examined at curve level?  The
      Laplace band is a PARAMETER band; the GT frames carry their own
      measurement noise, so an honest gap here is expected and measured,
      not hidden.

Scoring per note, on the corpus ground-truth frames inside the segment
(never seen by the fit): curve RMSE, band NLL, coverage@90; the pYIN
frames' own RMSE as the tracker anchor; the Phase-2 estimator scalar c
as context.  Two arms: the fixed prototype prior and the evidence-
fitted prior (.cache/phase3_prior_fit.json, scripts/fit_phase3_prior.py).

Sharded: `run k/n` writes results/phase3_cells/curve.shard{k}of{n}.pkl;
`report` merges into results/phase3_curve_dev.md.  DEV only; no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_curve_dev.py run 0/8
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_curve_dev.py report
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_waveform_dev import (MAX_SEG_S, SR, eligible_notes,
                                      selected)

CELLS_DIR = "results/phase3_cells"
OUT_MD = "results/phase3_curve_dev.md"
HYPERS_JSON = ".cache/phase3_prior_fit.json"
MIN_GT_FRAMES = 10
_Z90 = 1.6448536269514722


def load_hypers():
    if not os.path.exists(HYPERS_JSON):
        return None
    return json.load(open(HYPERS_JSON))["hypers"]


def score_curve(fit, gt_t, gt_x):
    mean, sd = fit.curve(gt_t)
    sd = np.maximum(sd, 1e-6)
    z = (mean - gt_x) / sd
    rmse = float(np.sqrt(np.mean((mean - gt_x) ** 2)))
    nll = float(np.mean(0.5 * z ** 2 + np.log(sd)
                        + 0.5 * np.log(2 * np.pi)))
    cov = float(np.mean(np.abs(z) <= _Z90))
    return rmse, nll, cov, float(np.median(sd))


def stage_run(shard: str) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    from eval_phase2_real import dev_unique_tracks
    from make_phase2_intro import note_curves
    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.phase3.curve import fit_note

    k, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    hyp = load_hypers()
    arms = [("fixed", None)] + ([("fitted", hyp)] if hyp else [])
    if not hyp:
        print("NOTE: no fitted hypers at", HYPERS_JSON, "- fixed arm only",
              flush=True)
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    track_map = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    rows = []
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        idx = eligible_notes(d, notes)
        mine = [i for pos, i in enumerate(idx) if (gidx + pos) % nsh == k]
        gidx += len(idx)
        if not mine:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, dtype=float), SR, int(sr48))
        for i in mine:
            on, du = notes["onset"][i], min(notes["duration"][i], MAX_SEG_S)
            a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
            x = audio[max(a, 0):min(b, audio.size)]
            if x.size < SR // 8:
                continue
            t = np.arange(x.size) / SR
            midi = float(d["midi"][i])

            tt, xx, ttg, xg = note_curves(key, i, data, f0s, track_map)
            off = 0.02
            sel_g = (ttg >= off) & (ttg <= off + t[-1])
            gt_t, gt_x = ttg[sel_g] - off, xg[sel_g]
            if gt_t.size < MIN_GT_FRAMES:
                continue
            sel_t = (tt >= off) & (tt <= off + t[-1])
            tr_t, tr_x = tt[sel_t] - off, xx[sel_t]
            pyin_rmse = (float(np.sqrt(np.mean(
                (np.interp(gt_t, tr_t, tr_x) - gt_x) ** 2)))
                if tr_t.size > 1 else np.nan)

            rec = {"key": key, "i": i, "instr": d["instrument"],
                   "dur": float(du), "n_gt": int(gt_t.size),
                   "pyin_rmse": pyin_rmse,
                   "gt_c": float(d["est_gt"][i, 0]),
                   "est_c": float(d["est"][i, 0])}
            msg = []
            for arm, hypers in arms:
                t0 = time.time()
                fit = fit_note(x, t, midi, hypers=hypers)
                rmse, nll, cov, med_sd = score_curve(fit, gt_t, gt_x)
                avg, avg_sd = fit.average()
                rec[arm] = {"rmse": rmse, "nll": nll, "cov": cov,
                            "med_sd": med_sd, "c": fit.c,
                            "avg": avg, "avg_sd": avg_sd,
                            "pd": fit.laplace_pd,
                            "logev": fit.log_evidence,
                            "wall": time.time() - t0}
                msg.append(f"{arm} {rmse:.2f}c cov {cov:.2f} "
                           f"[{rec[arm]['wall']:.0f}s]")
            rows.append(rec)
            print(f"{key} note {i} ({d['instrument']}): "
                  + ", ".join(msg) + f", pyin {pyin_rmse:.2f}c", flush=True)
    out = f"{CELLS_DIR}/curve.shard{k}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} notes)")


def stage_report() -> None:
    import glob
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/curve.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards found")
        return
    arms = [a for a in ("fixed", "fitted") if a in rows[0]]

    lines = [
        "# Phase 3: the curve-from-waveform posterior at scale (DEV, "
        "exploratory, no claims)\n",
        "\nBatch scaling of results/phase3_curve_proto_dev.md under the "
        "development protocol of\neval_phase3_waveform_dev (same tracks, "
        "same eligibility + rng(0) subsample); model =\n"
        "score_bundle.phase3.curve (gradient MAP + full joint Laplace; "
        "docs/ddsp_review.md).\nScored on corpus ground-truth frames the "
        "fit never sees.  Committed questions (stated\nbefore the run, "
        "see the script docstring): (a) curve vs pYIN frames on GT-frame "
        "RMSE per\nfamily; (b) curve-level coverage@90 — parameter band "
        "vs frames that carry their own\nmeasurement noise (the August "
        "estimand gap at curve level).\n",
        f"\n{len(rows)} notes, tracks: "
        + ", ".join(sorted({f"{r['key']}({r['instr']})" for r in rows}))
        + "\n",
        "\n| arm | median RMSE (cents) | q90 | median band sd | "
        "cov@90 | mean NLL | PD fits |\n|---|---|---|---|---|---|---|\n"]
    for a in arms:
        rmse = np.array([r[a]["rmse"] for r in rows])
        sd = np.array([r[a]["med_sd"] for r in rows])
        cov = np.array([r[a]["cov"] for r in rows])
        nll = np.array([r[a]["nll"] for r in rows])
        pd_ = np.array([r[a]["pd"] for r in rows])
        lines.append(f"| {a} | {np.median(rmse):.2f} | "
                     f"{np.quantile(rmse, .9):.2f} | {np.median(sd):.2f} | "
                     f"{np.mean(cov):.2f} | {np.mean(nll):.2f} | "
                     f"{np.mean(pd_):.0%} |\n")
    py = np.array([r["pyin_rmse"] for r in rows])
    ok = np.isfinite(py)
    lines.append(f"| pYIN frames | {np.median(py[ok]):.2f} | "
                 f"{np.quantile(py[ok], .9):.2f} | — | — | — | "
                 f"(n={int(ok.sum())}) |\n")

    d0 = np.array([r["fixed"]["rmse"] - r["pyin_rmse"] for r in rows
                   if np.isfinite(r["pyin_rmse"])])
    lines.append(f"\ncurve(fixed) − pYIN paired RMSE: median "
                 f"{np.median(d0):+.3f} cents, curve better on "
                 f"{np.mean(d0 < 0):.0%} of notes\n")
    if "fitted" in arms:
        d1 = np.array([r["fitted"]["rmse"] - r["fixed"]["rmse"]
                       for r in rows])
        dn = np.array([r["fitted"]["nll"] - r["fixed"]["nll"] for r in rows])
        lines.append(f"fitted − fixed paired: RMSE median "
                     f"{np.median(d1):+.3f} cents (better on "
                     f"{np.mean(d1 < 0):.0%}), NLL median "
                     f"{np.median(dn):+.3f}\n")

    lines.append("\nPer family, median GT-frame RMSE ("
                 + " / ".join(arms) + " / pYIN):\n")
    for fam, mem in (("strings", ("vn", "va", "vc", "db")),
                     ("winds", ("fl", "cl", "ob", "sax", "bn")),
                     ("brass", ("tpt", "hn", "tbn", "tba"))):
        sub = [r for r in rows if r["instr"] in mem]
        if not sub:
            continue
        vals = [f"{np.median([r[a]['rmse'] for r in sub]):.2f}"
                for a in arms]
        pys = [r["pyin_rmse"] for r in sub if np.isfinite(r["pyin_rmse"])]
        lines.append(f"- {fam} (n={len(sub)}): " + " / ".join(vals)
                     + f" / {np.median(pys):.2f}\n")

    walls = np.array([r["fixed"]["wall"] for r in rows])
    lines.append(f"\nwall per note (fixed arm): median {np.median(walls):.0f}"
                 f" s, q90 {np.quantile(walls, .9):.0f} s\n")
    open(OUT_MD, "w").writelines(lines)
    print("".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "report"
    if verb == "run":
        stage_run(sys.argv[2])
    elif verb == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown verb {verb}")
