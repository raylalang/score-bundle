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
CH = ["c (cents)", "log gamma", "log f", "loudness ell", "tau (s)",
      "delta_vib (s)"]
GT_COLS = (0, 1, 2, 5)            # GT curves carry no amplitude and no tau
CELLS_PKL = "results/phase2_real_cells.pkl"
TONAL_OUT_MD = "results/phase2_tonal_dev.md"
TONAL_CELLS = "results/phase2_tonal_cells.pkl"
# Circle-of-fifths study (dev, EXPLORATORY per the registration; hypothesis
# stated before the run): if temperament/tuning-reference structure is real,
# notes close on the circle of fifths share a mistuning that semitone
# neighbours do not — the tonal-metric graph (build_adjacency_tonal, the
# metric that HURT on piano expression) should improve held-out intonation
# recovery and/or calibration vs the plain graph. Primary read: channel c,
# as-given variant, paired tonal - plain, 95% CI. Expected on the non-pitch
# channels: the Phase-1 replacement penalty reappears. Either outcome is
# reported; nothing here touches the registered results file or claims.
HOLD_FRAC, SEEDS = 0.30, (0, 1)
MIN_NOTES = 30
# Estimator-failure rule for the intonation channel: |c| > C_MAX cents is an
# octave/semitone slip, not intonation — GT-validated on the dev extraction
# (0/358 notes with |c_pyin| > 150 agree with the GT-derived c within 50c).
# Such cells are marked MISSING, the same epistemic status as unidentifiable
# vibrato.  2.2% of notes, concentrated in low-register strings.
C_MAX = 150.0

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


def stage_loudness() -> None:
    """Append the loudness channel: per-note log RMS from the track audio.

    ell_i = mean over 4 equal chunks of log chunk-RMS, centred per track
    (level is arbitrary per recording); var_ell = chunk variance / 4.
    Cheap audio-only pass; extends the existing cache in place.
    """
    import soundfile as sf

    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    paths = {(p.index, tr.number): tr for p, tr in dev_unique_tracks()}
    for key, d in sorted(data.items()):
        tr = paths[key]
        audio, sr = sf.read(tr.audio)
        audio = np.asarray(audio, dtype=float)
        n = d["onset"].size
        ell = np.full(n, np.nan)
        vell = np.full(n, np.nan)
        for i in range(n):
            a, b = int(d["onset"][i] * sr), int((d["onset"][i]
                                                + d["duration"][i]) * sr)
            seg = audio[max(a, 0):min(b, audio.size)]
            if seg.size < 8:
                continue
            chunks = np.array_split(seg, 4)
            lr = np.array([np.log(np.sqrt(np.mean(c ** 2)) + 1e-8)
                           for c in chunks])
            ell[i], vell[i] = float(lr.mean()), float(lr.var(ddof=1) / 4)
        fin = np.isfinite(ell)
        ell[fin] -= ell[fin].mean()          # per-track centring
        d["ell"], d["var_ell"] = ell, np.maximum(vell, 1e-6)
        print(f"  {key} ell done ({fin.sum()}/{n})", flush=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    print("loudness appended")


def stage_tau() -> None:
    """Append the timing channel: onset-anchored local LOO warp residual.

    The adopted tau policy (prereg option 1, feasibility measured in
    results/tau_feasibility_dev.md): score MIDI matched to the annotated
    performed notes, tau from the +/-8-note leave-one-out tempo line
    (draft eq:localwarp), aligner error folded into the noise row as the
    tempo line's OLS predictive variance.  Extends the cache in place.
    """
    import pretty_midi

    from score_bundle.phase2.targets import hz_to_semitone
    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.phase2.warp import note_tau

    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    paths = {(p.index, tr.number): (p, tr) for p, tr in dev_unique_tracks()}
    n_method = {"exact": 0, "dtw": 0, "failed": 0}
    for key, d in sorted(data.items()):
        p, tr = paths[key]
        try:
            pm = pretty_midi.PrettyMIDI(
                os.path.join(p.folder, os.path.basename(p.score_mid)))
            sc = pm.instruments[tr.number - 1].notes
        except Exception:
            sc = []
        notes = read_notes_annotation(tr.notes)
        if not sc:
            n = notes["onset"].size
            out = {"tau": np.full(n, np.nan), "var": np.full(n, np.nan),
                   "method": "failed"}
        else:
            out = note_tau(np.array([nt.start for nt in sc]),
                           np.array([nt.pitch for nt in sc]),
                           notes["onset"],
                           hz_to_semitone(notes["pitch_hz"]))
        n_method[out["method"]] += 1
        d["tau"], d["var_tau"] = out["tau"], np.maximum(out["var"], 1e-8)
        d["tau_method"] = out["method"]
        fin = np.isfinite(out["tau"])
        print(f"  {key} tau {out['method']:<6} ({fin.sum()}/{fin.size}, "
              f"std {np.nanstd(out['tau']) * 1000 if fin.any() else 0:.0f} ms)",
              flush=True)
    print("match methods:", n_method)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    print("tau appended")


def stage_delta() -> None:
    """Append the vibrato onset-delay channel (delta_vib IN per the
    pre-stated criterion, results/delta_vib_dev.md).

    Values come from the GATED per-note fits already computed by
    scripts/eval_delta_vib.py (the eq:vibrato-exact estimator); the bundle's
    [c, log gamma, log f] keep the validated ungated estimator — the two
    fits coexist per note, documented in the prereg design.  GT-derived
    deltas are stored for the quasi-truth scoring.  Extends the cache.
    """
    with open(".cache/delta_vib_pyin.pkl", "rb") as fh:
        dpy = pickle.load(fh)
    with open(".cache/delta_vib_gt.pkl", "rb") as fh:
        dgt = pickle.load(fh)
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    for key, d in sorted(data.items()):
        n = d["onset"].size
        dv = np.full(n, np.nan)
        vv = np.full(n, np.nan)
        gv = np.full(n, np.nan)
        for i, r in enumerate(dpy[key]["gated_pyin"]):
            if r and r["delta_identifiable"] and np.isfinite(r["var_delta"]):
                dv[i], vv[i] = r["delta"], max(r["var_delta"], 1e-8)
        for i, r in enumerate(dgt[key]["gated_gt"]):
            if r and r["delta_identifiable"]:
                gv[i] = r["delta"]
        d["dvib"], d["var_dvib"], d["dvib_gt"] = dv, vv, gv
        print(f"  {key} delta_vib {np.isfinite(dv).sum()}/{n}", flush=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    print("delta_vib appended")


def _fit_systems(score_eig, feats, Yobs, mask, scale, var, rng_seedless,
                 tonal_eig=None):
    from score_bundle.gp import MultiOutputGraphGP

    k = Yobs.shape[1]
    floor = 0.05 * np.array([float(np.var(Yobs[mask[:, c], c]))
                             if mask[:, c].sum() > 2 else 1.0
                             for c in range(k)])
    med_var = np.array([np.median(var[:, c][mask[:, c] & np.isfinite(var[:, c])])
                        if (mask[:, c] & np.isfinite(var[:, c])).any() else 1.0
                        for c in range(k)])
    systems = [("gp", "additive", None, score_eig),
               ("gp_asgiven", "additive", med_var, score_eig),
               ("nograph", "none", None, score_eig)]
    if tonal_eig is not None:
        systems += [("gp_tonal", "additive", None, tonal_eig),
                    ("gp_tonal_asgiven", "additive", med_var, tonal_eig)]
    fits = {}
    for name, kern, fixed, (nu, U) in systems:
        g = MultiOutputGraphGP(nu, U, kernel=kern, features=feats,
                               n_channels=k)
        g.noise_scale = scale
        x_hat, _ = g.fit(Yobs, mask, noise_floor=floor, maxiter=200,
                         noise_fixed=fixed)
        m, sd = g.posterior(Yobs, mask, x_hat)
        nv = g.unpack(x_hat)["noise"]
        sd_pred = np.sqrt(sd ** 2 + nv[None, :] * scale)
        fits[name] = (m, sd, sd_pred)
    return fits


def stage_eval(tonal: bool = False) -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import (build_adjacency, build_adjacency_tonal,
                                    laplacian)
    from score_bundle.score import Score

    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    n_ch = len(CH)
    sys_labels = [("gp", "graph GP (learned scale)"),
                  ("gp_asgiven", "graph GP (as-given)"),
                  ("nograph", "no-graph ablation")]
    if tonal:
        sys_labels += [("gp_tonal", "tonal graph GP (learned scale)"),
                       ("gp_tonal_asgiven", "tonal graph GP (as-given)")]
    rows = {sys_: {tgt: {c: {"se": [], "cov": [], "nll": [], "n": 0}
                         for c in range(n_ch)}
                   for tgt in ("est", "gt")}
            for sys_, _ in sys_labels}
    per_track = []          # (key, seed, sys, channel, instrument, rmse_vs_est)
    used = 0
    for key, d in sorted(data.items()):
        est = np.concatenate([d["est"], d["ell"][:, None],
                              d["tau"][:, None], d["dvib"][:, None]], axis=1)
        var = np.concatenate([d["var"], d["var_ell"][:, None],
                              d["var_tau"][:, None],
                              d["var_dvib"][:, None]], axis=1)
        gt_full = np.full((est.shape[0], len(CH)), np.nan)
        gt_full[:, :3] = d["est_gt"]
        gt_full[:, 5] = d["dvib_gt"]
        ident = d["ident"]
        n = est.shape[0]
        usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= C_MAX)
        if usable.sum() < MIN_NOTES:
            continue
        used += 1
        score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                                  np.zeros(n, dtype=int))
        eig = np.linalg.eigh(laplacian(build_adjacency(score)))
        eig_t = (np.linalg.eigh(laplacian(build_adjacency_tonal(score)))
                 if tonal else None)
        X = rich_score_features(score, rff_dim=0)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
        scale = np.ones((n, n_ch))
        for c in range(n_ch):
            v = var[:, c]
            obs_c = np.isfinite(v)
            med = np.median(v[obs_c]) if obs_c.any() else 1.0
            scale[:, c] = np.where(np.isfinite(v),
                                   np.clip(v / max(med, 1e-12), 1e-2, 1e3),
                                   1.0)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + 7 * key[0] + key[1] + seed)
            held = (rng.random(n) < HOLD_FRAC) & usable
            mask = np.zeros((n, n_ch), dtype=bool)
            mask[:, 0] = usable & ~held
            mask[:, 1] = mask[:, 2] = usable & ~held & ident
            mask[:, 3] = usable & ~held & np.isfinite(est[:, 3])
            mask[:, 4] = ~held & np.isfinite(est[:, 4])
            mask[:, 5] = ~held & np.isfinite(est[:, 5])
            if mask[:, 0].sum() < 15 or held.sum() < 5:
                continue
            Yobs = np.where(mask, np.nan_to_num(est), 0.0)
            fits = _fit_systems(eig, feats, Yobs, mask, scale, var, None,
                                tonal_eig=eig_t)
            for sname, (m, sd, sd_pred) in fits.items():
                for tgt_name, tgt, s_use, cols in (
                        ("est", est, sd_pred, tuple(range(n_ch))),
                        ("gt", gt_full, sd, GT_COLS)):
                    for c in cols:
                        cells = held & np.isfinite(tgt[:, c])
                        if c == 0:
                            cells &= np.abs(tgt[:, 0]) <= C_MAX
                        elif c < 3:
                            cells &= ident if tgt_name == "est" \
                                else d["ident_gt"]
                        if cells.sum() < 3:
                            continue
                        err = tgt[cells, c] - m[cells, c]
                        s = s_use[cells, c]
                        nll = 0.5 * (np.log(2 * np.pi * s ** 2)
                                     + (err / s) ** 2)
                        r = rows[sname][tgt_name][c]
                        r["se"].extend((err ** 2).tolist())
                        r["cov"].extend(
                            (np.abs(err) <= _Z90 * s).tolist())
                        r["nll"].extend(nll.tolist())
                        r["n"] += int(cells.sum())
                        if tgt_name == "est":
                            per_track.append(
                                (key, seed, sname, c, d["instrument"],
                                 float(np.sqrt(np.mean(err ** 2))),
                                 float(np.mean(nll))))

    if tonal:
        lines = [f"# Circle-of-fifths metric on the Phase-2 bundle — "
                 f"EXPLORATORY dev study "
                 f"({used} unique tracks, {len(SEEDS)} seeds, {HOLD_FRAC:.0%} "
                 f"of notes hidden)", "",
                 "> Hypothesis stated before the run (see the constants block "
                 "of scripts/eval_phase2_real.py, committed first): the tonal "
                 "metric that hurt on piano expression should help on "
                 "intonation if temperament structure is real. Primary read: "
                 "channel c, as-given, paired tonal - plain. Exploratory per "
                 "the registration; the registered results file and claims "
                 "are untouched.", ""]
    else:
        lines = [f"# Phase 2 on real audio — first URMP dev results "
                 f"({used} unique tracks, {len(SEEDS)} seeds, {HOLD_FRAC:.0%} "
                 f"of notes hidden)", ""]
    for tgt_name, title, cols in (("est", "vs estimator targets (primary; "
                                          "predictive sd)", tuple(range(n_ch))),
                                  ("gt", "vs ground-truth-derived targets "
                                         "(quasi-truth; latent sd)", GT_COLS)):
        lines += [f"## {title}", "",
                  "| system | " + " | ".join(
                      f"{CH[c]} RMSE / NLL / cov@90" for c in cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for sname, label in sys_labels:
            cells = []
            for c in cols:
                r = rows[sname][tgt_name][c]
                if not r["se"]:
                    cells.append("--")
                    continue
                cells.append(f"{np.sqrt(np.mean(r['se'])):.3f} / "
                             f"{np.mean(r['nll']):+.2f} / "
                             f"{np.mean(r['cov']):.2f} (n={r['n']})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # medians over (track, seed) — a collapsed fit cannot hide in these
    lines += ["## Median per-(track, seed) RMSE vs estimator targets", "",
              "| system | " + " | ".join(CH) + " |",
              "|---" * (n_ch + 1) + "|"]
    by_sys = {}
    for key, seed, sname, c, inst, rmse, nll in per_track:
        by_sys.setdefault(sname, {}).setdefault(c, []).append(rmse)
    for sname, label in sys_labels:
        cells = [f"{np.median(by_sys[sname].get(c, [np.nan])):.3f}"
                 for c in range(n_ch)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # paired per-(track,seed) graph-vs-nograph deltas, vs estimator targets,
    # for BOTH noise variants (the as-given variant is the declared default;
    # 2026-08-13 audit: its paired contrast was previously unquantified)
    from eval_graphgp import bootstrap_ci
    rngb = np.random.default_rng(31)
    by = {}
    for key, seed, sname, c, inst, rmse, nll in per_track:
        by.setdefault((key, seed, c), {})[sname] = (rmse, nll)
    contrasts = [("gp", "nograph", "graph value (learned scale; gp - nograph)"),
                 ("gp_asgiven", "nograph",
                  "graph value (as-given, the default; gp_asgiven - nograph)")]
    if tonal:
        contrasts += [
            ("gp_tonal_asgiven", "gp_asgiven",
             "TONAL vs PLAIN metric (PRIMARY read; as-given, "
             "gp_tonal_asgiven - gp_asgiven)"),
            ("gp_tonal", "gp",
             "tonal vs plain metric (learned scale; gp_tonal - gp)"),
            ("gp_tonal_asgiven", "nograph",
             "tonal graph value (as-given; gp_tonal_asgiven - nograph)")]
    for sname, base, slabel in contrasts:
        lines += [f"## Paired {slabel}, "
                  "per (track, seed), vs estimator targets", "",
                  "| channel | dRMSE [95% CI] | dNLL [95% CI] |",
                  "|---|---|---|"]
        for c in range(n_ch):
            pairs = [(v[sname], v[base]) for (k, s, cc), v in by.items()
                     if cc == c and sname in v and base in v]
            cols = []
            for m_i in range(2):
                d = np.array([a[m_i] - b[m_i] for a, b in pairs])
                mu, lo, hi = bootstrap_ci(d, B=2000, rng=rngb)
                sig = "*" if (lo > 0) or (hi < 0) else " "
                cols.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")
            lines.append(f"| {CH[c]} | {cols[0]} | {cols[1]} "
                         f"(n={len(pairs)}) |")
        lines.append("")

    # per-instrument-family breakdown of the paired graph value
    from score_bundle.phase2.splits import FAMILIES
    lines += ["", "## Paired graph value by instrument family "
                  "(dRMSE, gp - nograph)", "",
              "| family | " + " | ".join(CH) + " |",
              "|---" * (n_ch + 1) + "|"]
    by_fam = {}
    for key, seed, sname, c, inst, rmse, nll in per_track:
        by_fam.setdefault((FAMILIES[inst], key, seed, c), {})[sname] = rmse
    for fam in ("strings", "wood", "brass"):
        cells = []
        for c in range(n_ch):
            d = [v["gp"] - v["nograph"] for (f, k, s, cc), v in by_fam.items()
                 if f == fam and cc == c and "gp" in v and "nograph" in v]
            if len(d) < 6:
                cells.append("--")
                continue
            mu, lo, hi = bootstrap_ci(np.array(d), B=2000, rng=rngb)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            cells.append(f"{mu:+.3f}{sig} (n={len(d)})")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    out_md = TONAL_OUT_MD if tonal else OUT_MD
    cells_pkl = TONAL_CELLS if tonal else CELLS_PKL
    os.makedirs("results", exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(table + "\n")
    with open(cells_pkl, "wb") as fh:      # raw cells: re-analysis provenance
        pickle.dump({"per_track": per_track, "rows": rows,
                     "channels": CH, "seeds": SEEDS,
                     "hold_frac": HOLD_FRAC}, fh)
    print(table)
    print(f"\nwrote {out_md} and {cells_pkl}")


if __name__ == "__main__":
    {"extract": stage_extract, "loudness": stage_loudness,
     "tau": stage_tau, "delta": stage_delta, "eval": stage_eval,
     "eval-tonal": lambda: stage_eval(tonal=True)}[sys.argv[1]]()
