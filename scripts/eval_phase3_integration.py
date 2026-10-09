#!/usr/bin/env python
"""Phase-3 integration study: the waveform as a 7th bundle channel (DEV).

The design conclusion of the Phase-3 study (results/phase3_waveform_dev.md,
thesis sec:phase3-study) made measurable: the waveform-derived intonation
posterior enters the Phase-2 GP as an ADDITIONAL observation channel whose
noise row is its posterior variance PLUS an empirically calibrated per-note
discrepancy floor. Does that improve held-out intonation inference — and is
the floor what keeps it honest?

Design (dev, exploratory; the 7 study tracks of the Phase-3 study):
- Standard Phase-2 evaluation cell: 6-channel bundle, 30% of notes hidden
  (same rng recipe as eval_phase2_real, so the same held sets), as-given
  noise, additive kernel, seeds (0, 1).
- Fusion condition: a 7th channel holds the waveform posterior mean for
  every note the Phase-3 study covered — INCLUDING held-out notes, which is
  the deployment situation (audio is always available; what is hidden is
  the estimator's cells). Coupling between channel 7 and channel c is
  learned per piece by the evidence (B is 7x7).
- The discrepancy floor is calibrated per (track, seed) from VISIBLE notes
  only: floor^2 = median over non-held notes of (wave - est_c)^2. Held
  notes contribute nothing to the floor.
- Systems: base6 (no wave channel) / wave+floor (the design) / wave-nofloor
  (posterior variance only — the honesty control: the Phase-3 study says
  this must be overconfident).
- Scoring: channel c only, held & estimator-finite cells, vs estimator
  targets (predictive sd; primary) and vs quasi-truth (latent sd). Paired
  per (track, seed), n = 14 pairs; medians, sign counts, and bootstrap CIs
  (B = 2000, wide at this n — exploratory throughout).

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/eval_phase3_integration.py run 0/7
    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/eval_phase3_integration.py report
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

HOLD_FRAC, SEEDS = 0.30, (0, 1)
C_MAX = 150.0
CELLS_DIR = "results/phase3_cells"
OUT_MD = "results/phase3_integration_dev.md"
_Z90 = 1.6448536269514722


def wave_posteriors():
    dev = {}
    for f in sorted(glob.glob(f"{CELLS_DIR}/wavedev.shard*.pkl")):
        for r in pickle.load(open(f, "rb")):
            dev[(tuple(r["key"]), r["i"])] = (r["dev8"][0], r["dev8"][1])
    return dev


def curve_posteriors(arm: str = "fitted"):
    """Realized-average read-out of the curve posterior (2026-10 study,
    eval_phase3_curve_dev cells): the curve-level replacement for the
    scalar dev8 channel.  (mean, sd) per (key, i), sd = exact Laplace sd
    of the time average (CurveFit.average)."""
    dev = {}
    for f in sorted(glob.glob(f"{CELLS_DIR}/curve.shard*.pkl")):
        for r in pickle.load(open(f, "rb")):
            if arm in r and np.isfinite(r[arm]["avg"]):
                dev[(tuple(r["key"]), r["i"])] = (r[arm]["avg"],
                                                  r[arm]["avg_sd"])
    return dev


def fit_three(eig, feats, est7, var7, mask, n):
    from score_bundle.gp import MultiOutputGraphGP

    fits = {}
    for name, cols in (("base6", 6), ("wave_floor", 7), ("wave_nofloor", 7)):
        k = cols
        Y = est7[:, :k]
        V = var7[:, :k].copy()
        msk = mask[:, :k]
        if name == "wave_nofloor":
            V[:, 6] = var7[:, 7]          # posterior variance only
        scale = np.ones((n, k))
        med_var = np.ones(k)
        for c in range(k):
            v = V[:, c]
            obs_c = msk[:, c] & np.isfinite(v)
            med = np.median(v[obs_c]) if obs_c.any() else 1.0
            med_var[c] = med
            scale[:, c] = np.where(np.isfinite(v),
                                   np.clip(v / max(med, 1e-12), 1e-2, 1e3),
                                   1.0)
        floor = 0.05 * np.array([float(np.var(Y[msk[:, c], c]))
                                 if msk[:, c].sum() > 2 else 1.0
                                 for c in range(k)])
        g = MultiOutputGraphGP(eig[0], eig[1], kernel="additive",
                               features=feats, n_channels=k)
        g.noise_scale = scale
        x_hat, _ = g.fit(np.where(msk, np.nan_to_num(Y), 0.0), msk,
                         noise_floor=floor, maxiter=200,
                         noise_fixed=med_var)
        m, sd = g.posterior(np.where(msk, np.nan_to_num(Y), 0.0), msk, x_hat)
        nv = g.unpack(x_hat)["noise"]
        sd_pred = np.sqrt(sd ** 2 + nv[None, :] * scale)
        fits[name] = (m, sd, sd_pred)
    return fits


def stage_run(shard: str, source: str = "wave") -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score
    from eval_phase3_waveform_dev import selected

    k_sh, nsh = (int(v) for v in shard.split("/"))
    wave = wave_posteriors() if source == "wave" else curve_posteriors()
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    frags = []
    for ti, (key, d, _tr) in enumerate(selected()):
        if ti % nsh != k_sh:
            continue
        d = data[key]
        est = np.concatenate([d["est"], d["ell"][:, None],
                              d["tau"][:, None], d["dvib"][:, None]], axis=1)
        var = np.concatenate([d["var"], d["var_ell"][:, None],
                              d["var_tau"][:, None],
                              d["var_dvib"][:, None]], axis=1)
        n = est.shape[0]
        ident = d["ident"]
        usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= C_MAX)
        wv = np.full((n, 2), np.nan)
        for i in range(n):
            if (tuple(key), i) in wave:
                wv[i] = wave[(tuple(key), i)]
        score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                                  np.zeros(n, dtype=int))
        eig = np.linalg.eigh(laplacian(build_adjacency(score)))
        X = rich_score_features(score, rff_dim=0)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
        for seed in SEEDS:
            t0 = time.time()
            rng = np.random.default_rng(1000 + 7 * key[0] + key[1] + seed)
            held = (rng.random(n) < HOLD_FRAC) & usable
            mask7 = np.zeros((n, 8), dtype=bool)   # col 7 unused (var slot)
            mask7[:, 0] = usable & ~held
            mask7[:, 1] = mask7[:, 2] = usable & ~held & ident
            mask7[:, 3] = usable & ~held & np.isfinite(est[:, 3])
            mask7[:, 4] = ~held & np.isfinite(est[:, 4])
            mask7[:, 5] = ~held & np.isfinite(est[:, 5])
            mask7[:, 6] = np.isfinite(wv[:, 0])    # wave: ALL covered notes
            # discrepancy floor from visible notes only
            vis = mask7[:, 6] & mask7[:, 0]
            if vis.sum() < 5:
                continue
            floor2 = float(np.median((wv[vis, 0] - est[vis, 0]) ** 2))
            est7 = np.concatenate([est, wv[:, :1]], axis=1)
            var7 = np.concatenate(
                [var, (wv[:, 1] ** 2 + floor2)[:, None],
                 (wv[:, 1] ** 2)[:, None]], axis=1)  # col7 = nofloor var
            fits = fit_three(eig, feats, est7, var7, mask7[:, :7], n)
            rec = {"key": key, "seed": seed, "floor": np.sqrt(floor2),
                   "n_wave": int(mask7[:, 6].sum())}
            cells = held & np.isfinite(est[:, 0]) & usable
            gt_ok = held & np.isfinite(d["est_gt"][:, 0]) \
                & (np.abs(d["est_gt"][:, 0]) <= C_MAX)
            for name, (m, sd, sd_pred) in fits.items():
                for tgt_name, tgt, s_use, cc in (
                        ("est", est[:, 0], sd_pred[:, 0], cells),
                        ("gt", d["est_gt"][:, 0], sd[:, 0], gt_ok)):
                    err = tgt[cc] - m[cc, 0]
                    s = s_use[cc]
                    rec[f"{name}_{tgt_name}"] = (
                        float(np.sqrt(np.mean(err ** 2))),
                        float(np.mean(0.5 * (np.log(2 * np.pi * s ** 2)
                                             + (err / s) ** 2))),
                        float(np.mean(np.abs(err) <= _Z90 * s)),
                        int(cc.sum()))
            frags.append(rec)
            print(f"{key} seed {seed}: 3 systems in {time.time()-t0:.0f}s "
                  f"(floor {np.sqrt(floor2):.2f} cents, "
                  f"{rec['n_wave']} wave cells)", flush=True)
    tag = "integ" if source == "wave" else "integ_curve"
    out = f"{CELLS_DIR}/{tag}.shard{k_sh}of{nsh}.pkl"
    pickle.dump(frags, open(out, "wb"))
    print(f"wrote {out} ({len(frags)} cells)")


def stage_report() -> None:
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/integ.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards")
        return
    rng = np.random.default_rng(31)

    def paired(sys_a, sys_b, tgt, idx):
        d = np.array([r[f"{sys_a}_{tgt}"][idx] - r[f"{sys_b}_{tgt}"][idx]
                      for r in rows])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        return (f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{star} "
                f"(better on {np.mean(d < 0):.0%})")

    lines = [
        "# Phase 3 integration: the waveform as a 7th bundle channel (DEV)\n",
        "\n**DEV ONLY — EXPLORATORY.** Design + rationale in the script "
        "docstring (`scripts/eval_phase3_integration.py`); thesis context "
        "sec:phase3-study. n = "
        f"{len(rows)} (track, seed) pairs over "
        f"{len({tuple(r['key']) for r in rows})} tracks; discrepancy floor "
        "median "
        f"{np.median([r['floor'] for r in rows]):.2f} cents.\n",
        "\n## Intonation channel c at held-out notes\n",
        "\n| system | vs estimator: RMSE / NLL / cov@90 | vs quasi-truth: "
        "RMSE / NLL / cov@90 |\n|---|---|---|\n"]
    for name in ("base6", "wave_floor", "wave_nofloor"):
        cells = []
        for tgt in ("est", "gt"):
            r_ = np.mean([r[f"{name}_{tgt}"][0] for r in rows])
            nll = np.mean([r[f"{name}_{tgt}"][1] for r in rows])
            cov = np.mean([r[f"{name}_{tgt}"][2] for r in rows])
            cells.append(f"{r_:.3f} / {nll:+.3f} / {cov:.2f}")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |\n")
    lines.append("\n## Paired contrasts (negative favours the first)\n\n")
    for a, b, why in (("wave_floor", "base6", "the integration's value"),
                      ("wave_floor", "wave_nofloor",
                       "the floor's value (calibration)")):
        for tgt in ("est", "gt"):
            for idx, met in ((0, "RMSE"), (1, "NLL")):
                lines.append(f"- {a} vs {b}, {tgt} {met}: "
                             f"{paired(a, b, tgt, idx)}  <!-- {why} -->\n")
    open(OUT_MD, "w").writelines(lines)
    print("".join(lines))
    print(f"wrote {OUT_MD}")


def stage_report_curve() -> None:
    """The curve channel vs the scalar channel, paired per (track, seed).

    Loads the curve-sourced cells (integ_curve) AND the August
    scalar-sourced cells (integ); the hold-out rng recipe is identical,
    so (key, seed) cells pair exactly — pinned by the base6 cross-run
    equality check below.
    """
    cur, old = [], {}
    for f in sorted(glob.glob(f"{CELLS_DIR}/integ_curve.shard*.pkl")):
        cur.extend(pickle.load(open(f, "rb")))
    for f in sorted(glob.glob(f"{CELLS_DIR}/integ.shard*.pkl")):
        for r in pickle.load(open(f, "rb")):
            old[(tuple(r["key"]), r["seed"])] = r
    if not cur or not old:
        print("missing shards (need both integ_curve and integ)")
        return
    pairs = [(r, old[(tuple(r["key"]), r["seed"])]) for r in cur
             if (tuple(r["key"]), r["seed"]) in old]
    b6 = max(abs(r["base6_est"][0] - o["base6_est"][0]) for r, o in pairs)
    rng = np.random.default_rng(31)

    def paired(d):
        d = np.asarray(d)
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        return (f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{star} "
                f"(better on {np.mean(d < 0):.0%})")

    lines = [
        "# Phase 3 integration: the CURVE posterior as the waveform "
        "channel (DEV, exploratory, no claims)\n",
        "\nRe-run of the August integration design "
        "(results/phase3_integration_dev.md) with the\nrealized-average "
        "read-out of the curve posterior (CurveFit.average, fitted-prior "
        "arm,\nexact Laplace sd) replacing the scalar dev8 channel; same "
        "hold-out recipe, so cells\npair exactly across runs (base6 "
        f"cross-run max |RMSE diff| = {b6:.2e}, must be ~0).\n",
        f"\nn = {len(pairs)} (track, seed) pairs; curve-source floor "
        f"median {np.median([r['floor'] for r, _ in pairs]):.2f} cents "
        f"(scalar-source {np.median([o['floor'] for _, o in pairs]):.2f})."
        "\n",
        "\n| system | vs estimator: RMSE / NLL / cov@90 | vs quasi-truth: "
        "RMSE / NLL / cov@90 |\n|---|---|---|\n"]
    named = [("base6", lambda r, o: r), ("scalar wave_floor",
             lambda r, o: o), ("curve_floor", lambda r, o: r),
             ("curve_nofloor", lambda r, o: r)]
    keymap = {"base6": "base6", "scalar wave_floor": "wave_floor",
              "curve_floor": "wave_floor", "curve_nofloor": "wave_nofloor"}
    for name, pick in named:
        cells = []
        for tgt in ("est", "gt"):
            kk = f"{keymap[name]}_{tgt}"
            r_ = np.mean([pick(r, o)[kk][0] for r, o in pairs])
            nll = np.mean([pick(r, o)[kk][1] for r, o in pairs])
            cov = np.mean([pick(r, o)[kk][2] for r, o in pairs])
            cells.append(f"{r_:.3f} / {nll:+.3f} / {cov:.2f}")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |\n")
    lines.append("\n## Paired contrasts (negative favours the first)\n\n")
    for label, da in (
        ("curve_floor vs base6 (the curve channel's value)",
         lambda tgt, i: [r[f"wave_floor_{tgt}"][i] - r[f"base6_{tgt}"][i]
                         for r, _ in pairs]),
        ("curve_floor vs scalar wave_floor (curve vs scalar read-out)",
         lambda tgt, i: [r[f"wave_floor_{tgt}"][i] - o[f"wave_floor_{tgt}"][i]
                         for r, o in pairs]),
        ("curve_floor vs curve_nofloor (the floor's value)",
         lambda tgt, i: [r[f"wave_floor_{tgt}"][i]
                         - r[f"wave_nofloor_{tgt}"][i] for r, _ in pairs]),
    ):
        for tgt in ("est", "gt"):
            for i, met in ((0, "RMSE"), (1, "NLL")):
                lines.append(f"- {label}, {tgt} {met}: "
                             f"{paired(da(tgt, i))}\n")
    out = "results/phase3_curve_integration_dev.md"
    open(out, "w").writelines(lines)
    print("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "report"
    if verb == "run":
        stage_run(sys.argv[2])
    elif verb == "runcurve":
        stage_run(sys.argv[2], source="curve")
    elif verb == "reportcurve":
        stage_report_curve()
    else:
        stage_report()
