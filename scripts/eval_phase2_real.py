#!/usr/bin/env python
"""Phase 2 on real audio: the first URMP results (DEVELOPMENT side only).

End-to-end: pyin f0 (extract_f0, cached) -> confidence-filtered per-note NLLS
targets (targets.note_targets) -> heteroscedastic cell-mask graph GP ->
held-out imputation, graph vs no-graph, both axes.  Mirrors the synthetic
pilot (eval_phase2_synthetic.py) with real estimator targets in place of
synthetic truth.  Two scorings per held-out note:

  vs ESTIMATOR targets (primary; the prereg design's honest claim — recovery
     = agreement with the estimator; predictive sd includes the cell noise);
  vs GT-DERIVED targets (quasi-truth: the same NLLS run on URMP's
     ground-truth F0 curve; scored with the latent sd, pilot-style).

Shared recordings across arrangements are deduplicated by the MD5 of the
ground-truth F0 annotation.  Confirmation pieces are refused.

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/eval_phase2_real.py extract
    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/eval_phase2_real.py eval
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/home/ray/Research/data/urmp/Dataset"
CACHE = ".cache/urmp_targets_dev.pkl"
OUT_MD = "results/phase2_real_results.md"
_Z90 = 1.6448536269514722
CH = ["c (cents)", "log gamma", "log f"]
HOLD_FRAC, SEEDS = 0.30, (0, 1)
MIN_NOTES = 30

RANGES = {"vn": (180, 3000), "va": (120, 1500), "vc": (60, 1000),
          "db": (35, 500), "fl": (240, 2500), "ob": (220, 1800),
          "cl": (140, 1800), "bn": (55, 700), "sax": (100, 1200),
          "tpt": (160, 1200), "hn": (85, 800), "tbn": (70, 700),
          "tba": (40, 400)}


def dev_unique_tracks():
    from score_bundle.phase2.splits import urmp_split
    from score_bundle.phase2.urmp import load_urmp_meta

    dev, _ = urmp_split(load_urmp_meta(ROOT))
    seen, out = set(), []
    for p in dev:
        for tr in p.tracks:
            if not (tr.audio and tr.f0s and tr.notes):
                continue
            digest = hashlib.md5(open(tr.f0s, "rb").read()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.append((p, tr))
    return out


def stage_extract() -> None:
    import soundfile as sf

    from score_bundle.phase2.intonation import extract_f0
    from score_bundle.phase2.targets import note_targets
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)

    tracks = dev_unique_tracks()
    print(f"{len(tracks)} unique development tracks", flush=True)
    data = {}
    for p, tr in tracks:
        notes = read_notes_annotation(tr.notes)
        audio, sr = sf.read(tr.audio)
        fmin, fmax = RANGES[tr.instrument]
        py = extract_f0(np.asarray(audio, dtype=float), sr,
                        fmin=float(fmin), fmax=float(fmax))
        tg = note_targets(py["t"], py["f0"], py["voiced"], py["prob"],
                          notes["onset"], notes["duration"],
                          notes["pitch_hz"])
        t_gt, f0_gt = read_f0_annotation(tr.f0s)
        tg_gt = note_targets(t_gt, f0_gt, f0_gt > 0, None,
                             notes["onset"], notes["duration"],
                             notes["pitch_hz"])
        data[(p.index, tr.number)] = {
            "piece": p.index, "name": p.name, "instrument": tr.instrument,
            "onset": notes["onset"], "duration": notes["duration"],
            "midi": tg["midi"], "est": tg["est"], "var": tg["var"],
            "ident": tg["ident"], "n_frames": tg["n_frames"],
            "est_gt": tg_gt["est"], "var_gt": tg_gt["var"],
            "ident_gt": tg_gt["ident"]}
        print(f"  {p.index:02d} {p.name:<12} tr{tr.number} {tr.instrument:<4} "
              f"{notes['onset'].size:4d} notes  ident "
              f"{tg['ident'].mean():.0%}", flush=True)
    os.makedirs(".cache", exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    print(f"wrote {CACHE} ({len(data)} tracks)")


def _fit_systems(score_eig, feats, Yobs, mask, scale, var, rng_seedless):
    from score_bundle.gp import MultiOutputGraphGP

    nu, U = score_eig
    floor = 0.05 * np.array([float(np.var(Yobs[mask[:, c], c]))
                             if mask[:, c].sum() > 2 else 1.0
                             for c in range(3)])
    med_var = np.array([np.median(var[:, c][mask[:, c]])
                        if mask[:, c].any() else 1.0 for c in range(3)])
    fits = {}
    for name, kern, fixed in (("gp", "additive", None),
                              ("gp_asgiven", "additive", med_var),
                              ("nograph", "none", None)):
        g = MultiOutputGraphGP(nu, U, kernel=kern, features=feats,
                               n_channels=3)
        g.noise_scale = scale
        x_hat, _ = g.fit(Yobs, mask, noise_floor=floor, maxiter=200,
                         noise_fixed=fixed)
        m, sd = g.posterior(Yobs, mask, x_hat)
        nv = g.unpack(x_hat)["noise"]
        sd_pred = np.sqrt(sd ** 2 + nv[None, :] * scale)
        fits[name] = (m, sd, sd_pred)
    return fits


def stage_eval() -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    rows = {sys_: {tgt: {c: {"se": [], "cov": [], "n": 0} for c in range(3)}
                   for tgt in ("est", "gt")}
            for sys_ in ("gp", "gp_asgiven", "nograph")}
    per_track = []                      # (key, sys, channel, rmse_vs_est)
    used = 0
    for key, d in sorted(data.items()):
        est, var, ident = d["est"], d["var"], d["ident"]
        n = est.shape[0]
        usable = np.isfinite(est[:, 0])
        if usable.sum() < MIN_NOTES:
            continue
        used += 1
        score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                                  np.zeros(n, dtype=int))
        eig = np.linalg.eigh(laplacian(build_adjacency(score)))
        X = rich_score_features(score, rff_dim=0)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
        scale = np.ones((n, 3))
        for c in range(3):
            v = var[:, c]
            obs_c = np.isfinite(v)
            med = np.median(v[obs_c]) if obs_c.any() else 1.0
            scale[:, c] = np.where(np.isfinite(v),
                                   np.clip(v / max(med, 1e-12), 1e-2, 1e3),
                                   1.0)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + 7 * key[0] + key[1] + seed)
            held = (rng.random(n) < HOLD_FRAC) & usable
            mask = np.zeros((n, 3), dtype=bool)
            mask[:, 0] = usable & ~held
            mask[:, 1] = mask[:, 2] = usable & ~held & ident
            if mask[:, 0].sum() < 15 or held.sum() < 5:
                continue
            Yobs = np.where(mask, np.nan_to_num(est), 0.0)
            fits = _fit_systems(eig, feats, Yobs, mask, scale, var, None)
            for sname, (m, sd, sd_pred) in fits.items():
                for tgt_name, tgt, tvar, s_use in (
                        ("est", est, None, sd_pred),
                        ("gt", d["est_gt"], None, sd)):
                    for c in range(3):
                        cells = held & np.isfinite(tgt[:, c])
                        if c > 0:
                            cells &= ident if tgt_name == "est" \
                                else d["ident_gt"]
                        if cells.sum() < 3:
                            continue
                        err = tgt[cells, c] - m[cells, c]
                        s = s_use[cells, c]
                        r = rows[sname][tgt_name][c]
                        r["se"].extend((err ** 2).tolist())
                        r["cov"].extend(
                            (np.abs(err) <= _Z90 * s).tolist())
                        r["n"] += int(cells.sum())
                        if tgt_name == "est":
                            per_track.append(
                                (key, seed, sname, c,
                                 float(np.sqrt(np.mean(err ** 2)))))

    lines = [f"# Phase 2 on real audio — first URMP dev results "
             f"({used} unique tracks, {len(SEEDS)} seeds, {HOLD_FRAC:.0%} "
             f"of notes hidden)", ""]
    for tgt_name, title in (("est", "vs estimator targets (primary; "
                                    "predictive sd)"),
                            ("gt", "vs ground-truth-derived targets "
                                   "(quasi-truth; latent sd)")):
        lines += [f"## {title}", "",
                  "| system | " + " | ".join(
                      f"{c} RMSE / cov@90" for c in CH) + " |",
                  "|---|---|---|---|"]
        for sname, label in (("gp", "graph GP (learned scale)"),
                             ("gp_asgiven", "graph GP (as-given)"),
                             ("nograph", "no-graph ablation")):
            cells = []
            for c in range(3):
                r = rows[sname][tgt_name][c]
                if not r["se"]:
                    cells.append("--")
                    continue
                cells.append(f"{np.sqrt(np.mean(r['se'])):.3f} / "
                             f"{np.mean(r['cov']):.2f} (n={r['n']})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # paired per-(track,seed) graph-vs-nograph deltas, vs estimator targets
    from eval_graphgp import bootstrap_ci
    rngb = np.random.default_rng(31)
    lines += ["## Paired graph value (gp - nograph), per (track, seed), "
              "vs estimator targets", "",
              "| channel | dRMSE [95% CI] |", "|---|---|"]
    by = {}
    for key, seed, sname, c, rmse in per_track:
        by.setdefault((key, seed, c), {})[sname] = rmse
    for c in range(3):
        d = [v["gp"] - v["nograph"] for (k, s, cc), v in by.items()
             if cc == c and "gp" in v and "nograph" in v]
        mu, lo, hi = bootstrap_ci(np.array(d), B=2000, rng=rngb)
        sig = "*" if (lo > 0) or (hi < 0) else " "
        lines.append(f"| {CH[c]} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig} "
                     f"(n={len(d)}) |")
    table = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open(OUT_MD, "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    {"extract": stage_extract, "eval": stage_eval}[sys.argv[1]]()
