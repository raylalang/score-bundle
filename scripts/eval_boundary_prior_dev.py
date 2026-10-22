#!/usr/bin/env python
"""The boundary-prior experiment: does score-informed structure add value
to note-boundary (timing) inference from the waveform where the acoustic
evidence is absent?  (DEV, exploratory, no claims.)  Design per
docs/boundary_prior_proposal.md; the two lessons of the intonation
first span (results/prior_value_v2_tuning_dev.md) are built in: the
hidden element is DROPPED from the design (hard cuts, not just masked
samples) and a free per-chain common-mode scalar sits in EVERY arm.

Three cross-joint priors over the boundary offsets db (ms from the
score-nominal tempo-line boundary b_nom = annotated onset - tau), at
matched marginal variance, identical likelihood / data / optimizer /
steps, sharp edges (fixed width 1 ms, the chain study's winner):

  indep     db ~ N(s 1, sigma_b^2 I)
  temporal  db ~ N(s 1, sigma_b^2 K_chain)        joints in order
  graph     db ~ N(s 1, sigma_b^2 K_score)        joint k = note k+1 (the
                                                  note it starts), score
                                                  graph over those notes
  s ~ N(0, 50^2) ms, the free per-chain offset scalar, marginalized as a
  rank-one term in all arms (absorbs the chain's tempo error and the
  acoustic-vs-musical onset gap).  sigma_b^2 = the track's own variance
  of tau (its population timing scale, in ms^2; not tuned).
  K = (I + L)^{-1} diagonal-normalized for both structured arms.

Two conditions per chain (the chain study's 47 chains, 4 dev tracks;
the HIDDEN joint = the middle joint, which needs a finite tau):
  masked  the hidden joint is removed from the design: its two sigmoid
          edges become hard cuts at b_annot -/+ 100 ms and the samples
          in that interval are masked, so the likelihood is exactly flat
          in db[hidden] (fit_chain(drop_joints=), test-pinned) and its
          posterior is the prior conditional given the observed joints,
          which the audio pins (the chain study's deployment arm);
  clean   all joints free from the nominal inits, full audio --- the
          deployment protocol with a prior over db.
All joints initialize at b_nom (no annotation access in the fit); the
reference c-init / noise come from chain_reference at b_nom with the
deployment guard, shared by the three arms of a chain.

Committed hypotheses, stated before the run:
  H-B1 (ambiguity): on masked cells the structured arms recover the
       hidden boundary better than the independent arm, paired.  The
       independent arm WITH the scalar returns b_nom + the observed
       joints' mean offset; WITHOUT the scalar it would return b_nom
       exactly (error = |tau|, the deadpan row, computed analytically,
       not run).  Falsification: no significant structured-vs-indep gap
       => at chain scale boundary timing carries no note-level
       structure beyond the common offset; reported verbatim.  The
       graph-vs-temporal contrast is expected ns (2-4 joints per chain).
  H-B2 (control): on clean cells the arms tie, and the prior does not
       hurt the deployment error of the chain study (~19.5 ms median).
  H-B3 (calibration): the hidden joint's band is the prior conditional
       sd; its coverage@90 tests the timing prior's honesty.  No target
       beyond "reported".
Strata recorded, not averaged in: unison joints (same pitch across the
joint), per family.  Scoring: hidden-joint |b_hat - b_annot| in ms,
coverage from a joint Laplace over (c, u, db) at the MAP; all joints'
errors stored for the record.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_boundary_prior_dev.py smoke
    bash scripts/run_boundary_prior_eval.sh 8          # sharded + report

THE CHAIN-SCALE PRIOR WIDTH (2026-10-22; env BOUNDPRIOR_SIGMA=chain; prefix
boundprior_chainsig; results/boundary_prior_chainsig_dev.md).  The first
run's masked bands covered at 1.00 with sd 73-95 ms against ~30 ms errors:
sigma_b^2 = the track's full tau variance is a population scale that
overstates within-chain variation.  This rerun sets sigma_b^2 = the median
over the track's chains of the within-chain variance of tau (the scale at
which the prior actually operates), everything else identical.
Committed hypotheses, stated before the run:
  H-B4 (calibration): masked coverage@90 of the hidden joint moves from
       1.00 toward 0.90 for all three arms.
  H-B5 (verdict unchanged): the paired contrasts between arms stay ties;
       the arms-vs-tempo-line gains stay.  Calibration-only rerun; if the
       point verdict changed that would itself be the finding, reported.
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

from eval_phase3_chain_dev import SR, chain_segment, select_chains
from eval_phase3_curve_dev import load_hypers

CELLS_DIR = "results/phase3_cells"
SIGMA_MODE = os.environ.get("BOUNDPRIOR_SIGMA", "track")
PREFIX = "boundprior" if SIGMA_MODE == "track" else "boundprior_chainsig"
OUT_MD = ("results/boundary_prior_dev.md" if SIGMA_MODE == "track"
          else "results/boundary_prior_chainsig_dev.md")
ARMS = ("indep", "temporal", "graph")
N_STEPS = 300
S_SCALE = 1.0
SCALAR_SD_MS = 50.0
HALF_MASK_S = 0.10
FIXED_WIDTH = 0.001
_Z90 = 1.6448536269514722


def joint_cov(arm, midis, onsets, durs, var_b):
    """(K-1, K-1) prior covariance over db (ms^2): joint k <-> note k+1."""
    k_j = len(midis) - 1
    if arm == "indep":
        K = np.eye(k_j)
    else:
        if arm == "temporal":
            A = np.zeros((k_j, k_j))
            for i in range(k_j - 1):
                A[i, i + 1] = A[i + 1, i] = 1.0
        else:
            from score_bundle.graph import build_adjacency
            from score_bundle.score import Score
            sc = Score.from_arrays(np.asarray(midis[1:], float),
                                   np.asarray(onsets[1:], float),
                                   np.asarray(durs[1:], float),
                                   np.zeros(k_j, dtype=int))
            A = build_adjacency(sc)
        L = np.diag(A.sum(1)) - A
        K = np.linalg.inv(np.eye(k_j) + S_SCALE * L)
        d = np.sqrt(np.diag(K))
        K = K / d[:, None] / d[None, :]
    return var_b * K + SCALAR_SD_MS ** 2 * np.ones((k_j, k_j))


def chain_laplace_db(x, t, midis, fit, hypers, b_init, b_prior_cov,
                     sample_mask, drop_joints, n_harm=8, n_chunk=2,
                     amp_var=10.0, softness=0.010):
    """Marginal sd (ms) of each db from the joint (c, u, db) Laplace at
    the MAP, with the priors, the mask and the hard-cut joints included."""
    import torch
    from score_bundle.phase3.curve import (collapsed_loglik_torch,
                                           design_support_torch,
                                           interp_knots, knot_gram,
                                           CurvePrior)
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64)
    t_np = np.asarray(t, float)
    t_t = torch.as_tensor(t_np, dtype=torch.float64)
    m = t_np.size
    k_n = len(midis)
    prior = CurvePrior()
    msk = (torch.as_tensor(np.asarray(sample_mask, bool))
           if sample_mask is not None else None)
    dj = {int(k): (float(lo), float(hi))
          for k, (lo, hi) in (drop_joints or {}).items()}
    chol_b = torch.linalg.cholesky(torch.as_tensor(
        np.asarray(b_prior_cov, float), dtype=torch.float64))
    b_init_t = torch.as_tensor(np.asarray(b_init, float),
                               dtype=torch.float64)
    knots_t, chols, slices, sizes = [], [], [], []
    for k in range(k_n):
        kt = torch.as_tensor(fit.knots[k], dtype=torch.float64)
        _, chol = knot_gram(kt, prior, **(hypers or {}))
        knots_t.append(kt)
        chols.append(chol)
        lo, hi = fit.knots[k][0], fit.knots[k][-1]
        ia = int(np.searchsorted(t_np, max(t_np[0], lo - 0.2)))
        ib = int(np.searchsorted(t_np, min(t_np[-1], hi + 0.2)))
        slices.append((ia, ib))
        sizes.append(1 + fit.knots[k].size)
    n_cu = sum(sizes)

    def nlj(theta):
        db = theta[n_cu:]
        b_lo = b_init_t + db / 1000.0
        b_hi = b_lo + FIXED_WIDTH
        pos, cols = 0, []
        prior_term = 0.5 * (db @ torch.cholesky_solve(
            db.unsqueeze(1), chol_b).squeeze(1))
        for k in range(k_n):
            c_k = theta[pos]
            u_k = theta[pos + 1:pos + sizes[k]]
            pos += sizes[k]
            ia, ib = slices[k]
            ts = t_t[ia:ib]
            cents = c_k + interp_knots(u_k, knots_t[k], ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, n_harm, n_chunk)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                if (k - 1) in dj:
                    w = w * (t_t >= dj[k - 1][1]).to(w.dtype)
                else:
                    w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                if k in dj:
                    w = w * (t_t < dj[k][0]).to(w.dtype)
                else:
                    w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            cols.append(Phi_k * w.unsqueeze(1))
            prior_term = prior_term + 0.5 * (u_k @ torch.cholesky_solve(
                u_k.unsqueeze(1), chols[k]).squeeze(1))
        Phi = torch.cat(cols, dim=1)
        if msk is not None:
            ll = collapsed_loglik_torch(x_t[msk], Phi[msk], fit.noise_var,
                                        amp_var)
        else:
            ll = collapsed_loglik_torch(x_t, Phi, fit.noise_var, amp_var)
        return -ll + prior_term

    db0 = (np.asarray(fit.b_lo, float) - np.asarray(b_init, float)) * 1000.0
    theta0 = torch.as_tensor(np.concatenate(
        [np.concatenate([[fit.c[k]], fit.us[k]]) for k in range(k_n)]
        + [db0]), dtype=torch.float64)
    H = torch.autograd.functional.hessian(nlj, theta0)
    H = 0.5 * (H + H.T)
    try:
        cov = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        ev, evec = torch.linalg.eigh(H)
        ev = ev.clamp(min=1e-8 * float(ev.max().clamp(min=1.0)))
        cov = (evec / ev) @ evec.T
    cov = cov.numpy()
    return np.sqrt(np.maximum(np.diag(cov)[n_cu:], 0.0))


def iter_cells():
    import soundfile as sf
    from scipy.signal import resample_poly
    from eval_phase2_real import dev_unique_tracks  # noqa: F401
    from score_bundle.phase2.urmp import read_notes_annotation
    from eval_phase3_waveform_dev import selected

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        chains = select_chains(notes)
        if not chains:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        tau = np.asarray(d["tau"], float)
        var_b = float(np.nanvar(tau[np.isfinite(tau)])) * 1e6   # ms^2
        if SIGMA_MODE == "chain":
            within = []
            for idx_ in chains:
                tt = tau[idx_]
                tt = tt[np.isfinite(tt)]
                if tt.size >= 2:
                    within.append(float(np.var(tt)))
            if within:
                var_b = float(np.median(within)) * 1e6
            var_b = max(var_b, 4.0 ** 2)      # floor at (4 ms)^2
        for idx in chains:
            k_j = len(idx) - 1
            j = (k_j - 1) // 2                      # hidden = middle joint
            i_hidden = idx[j + 1]                    # the note it starts
            if not np.isfinite(tau[i_hidden]):
                continue
            x, t, t0, b_annot = chain_segment(audio, notes, idx)
            b_nom = b_annot.copy()
            for q, i in enumerate(idx[1:]):
                if np.isfinite(tau[i]):
                    b_nom[q] = b_annot[q] - float(tau[i])
            lo, hi = b_annot[j] - HALF_MASK_S, b_annot[j] + HALF_MASK_S
            mask = ~((t >= lo) & (t < hi))
            midis = [float(d["midi"][i]) for i in idx]
            onsets = [float(notes["onset"][i]) for i in idx]
            durs = [float(notes["duration"][i]) for i in idx]
            unison = abs(midis[j] - midis[j + 1]) < 0.5
            yield gidx, dict(key=key, idx=idx, instr=d["instrument"], x=x,
                             t=t, b_annot=b_annot, b_nom=b_nom, midis=midis,
                             onsets=onsets, durs=durs, var_b=var_b, j=j,
                             tau_ms=float(tau[i_hidden]) * 1000.0,
                             mask=mask, lo=lo, hi=hi, unison=unison)
            gidx += 1


def run_cell(ch, hypers, n_steps=N_STEPS):
    from score_bundle.phase3.curve import chain_reference, fit_chain
    out = {}
    x, t, midis, b_nom = ch["x"], ch["t"], ch["midis"], ch["b_nom"]
    for cond in ("masked", "clean"):
        mask = ch["mask"] if cond == "masked" else None
        dj = {ch["j"]: (ch["lo"], ch["hi"])} if cond == "masked" else None
        c0, nv = chain_reference(x, t, midis, b_nom, guard_frac=0.2,
                                 sample_mask=mask)
        for arm in ARMS:
            K_b = joint_cov(arm, midis, ch["onsets"], ch["durs"], ch["var_b"])
            t0_ = time.time()
            fit = fit_chain(x, t, midis, b_nom, hypers=hypers, c_init=c0,
                            noise_var=nv, b_prior_cov=K_b, sample_mask=mask,
                            drop_joints=dj, free_width=False,
                            fixed_width=FIXED_WIDTH, n_steps=n_steps)
            sds = chain_laplace_db(x, t, midis, fit, hypers, b_nom, K_b,
                                   mask, dj)
            err_all = (np.asarray(fit.b_lo) - ch["b_annot"]) * 1000.0
            j = ch["j"]
            out[f"{cond}_{arm}"] = {
                "err": float(err_all[j]), "sd": float(sds[j]),
                "err_all": err_all.copy(), "sd_all": np.asarray(sds).copy(),
                "db_all": ((np.asarray(fit.b_lo) - b_nom) * 1000.0).copy(),
                "wall": time.time() - t0_}
    return out


def stage_smoke():
    hypers = load_hypers()
    for gidx, ch in iter_cells():
        t0_ = time.time()
        res = run_cell(ch, hypers)
        wall = time.time() - t0_
        n = sum(1 for _ in iter_cells())
        print(f"cell 0: {ch['key']} notes {ch['idx']} hidden joint #{ch['j']} "
              f"(tau {ch['tau_ms']:+.1f} ms = deadpan err, unison={ch['unison']}, "
              f"sigma_b {np.sqrt(ch['var_b']):.0f} ms)")
        for k, v in res.items():
            print(f"  {k}: err {v['err']:+.1f} +/- {v['sd']:.1f} ms "
                  f"(db {np.round(v['db_all'], 1)}) [{v['wall']:.0f}s]")
        print(f"PROJECTION: {n} cells x {wall:.0f}s = "
              f"{n * wall / 3600:.1f} h CPU / 8 shards = "
              f"{n * wall / 8 / 3600:.1f} h wall", flush=True)
        return
    print("no cells")


def stage_run(shard):
    k_sh, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    hypers = load_hypers()
    rows = []
    for gidx, ch in iter_cells():
        if gidx % nsh != k_sh:
            continue
        res = run_cell(ch, hypers)
        rec = {k: ch[k] for k in ("key", "idx", "instr", "j", "tau_ms",
                                  "unison", "var_b")}
        rec.update(res)
        rows.append(rec)
        print(f"{ch['key']} notes {ch['idx']} joint#{ch['j']} tau "
              f"{ch['tau_ms']:+.0f}: masked "
              + " ".join(f"{a}:{res['masked_' + a]['err']:+.0f}" for a in ARMS)
              + f" [{sum(v['wall'] for v in res.values()):.0f}s]", flush=True)
    out = f"{CELLS_DIR}/{PREFIX}.shard{k_sh}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} cells)")


def stage_report():
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/{PREFIX}.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards")
        return
    rng = np.random.default_rng(31)
    main = [r for r in rows if not r["unison"]]

    def err(r, cond, arm):
        return abs(r[f"{cond}_{arm}"]["err"])

    def paired(cond, a, b, sub):
        d = np.array([err(r, cond, a) - err(r, cond, b) for r in sub])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        return (f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{star} "
                f"(better on {np.mean(d < 0):.0%}, n={d.size})")

    lines = [
        "# The boundary-prior experiment: score structure in note-boundary "
        "inference from the waveform ("
        + ("chain-scale prior width; " if SIGMA_MODE == "chain" else "")
        + "DEV, exploratory, no claims)\n",
        "\nThree cross-joint priors over boundary offsets from the tempo-line "
        "boundaries, matched marginal\nvariance, free per-chain offset scalar "
        "in every arm, identical likelihood/data/optimizer, sharp\nedges. "
        "Masked = the hidden (middle) joint removed from the design (hard "
        "cuts +/-100 ms, samples\nmasked); clean = deployment from the "
        "tempo line with full audio. Committed hypotheses and\nfalsification "
        "rule in the script docstring (`scripts/eval_boundary_prior_dev.py`). "
        "Scored against the\nannotated onset. Unison joints are a separate "
        "stratum, excluded from the main rows.\n",
        f"\n{len(rows)} cells (chains), {len(main)} non-unison, tracks: "
        + ", ".join(sorted({f"{r['key']}({r['instr']})" for r in rows})) + "\n",
        "\n| condition | arm | median abs err (ms) | q90 | cov@90 | median sd |"
        "\n|---|---|---|---|---|---|\n"]
    for cond in ("masked", "clean"):
        for arm in ARMS:
            e = np.array([err(r, cond, arm) for r in main])
            z = np.array([r[f"{cond}_{arm}"]["err"]
                          / max(r[f"{cond}_{arm}"]["sd"], 1e-9) for r in main])
            sd = np.array([r[f"{cond}_{arm}"]["sd"] for r in main])
            lines.append(f"| {cond} | {arm} | {np.median(e):.1f} | "
                         f"{np.quantile(e, .9):.1f} | "
                         f"{np.mean(np.abs(z) <= _Z90):.2f} | "
                         f"{np.median(sd):.1f} |\n")
    dp = np.array([abs(r["tau_ms"]) for r in main])
    lines.append(f"| masked | (deadpan = tempo line, analytic indep without "
                 f"the scalar) | {np.median(dp):.1f} | {np.quantile(dp, .9):.1f} "
                 "| --- | --- |\n")
    lines.append("\n## Paired contrasts, masked condition (negative favours "
                 "the first arm)\n\n")
    for a, b in (("graph", "indep"), ("temporal", "indep"), ("graph", "temporal")):
        lines.append(f"- {a} vs {b}: {paired('masked', a, b, main)}\n")
    for a in ARMS:
        d = np.array([err(r, "masked", a) - abs(r["tau_ms"]) for r in main])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        lines.append(f"- {a} vs deadpan (tempo line): {d.mean():+.2f} "
                     f"[{lo:+.2f}, {hi:+.2f}]{star} (better on {np.mean(d < 0):.0%})\n")
    lines.append("\n## Paired contrasts, clean condition (H-B2: expect ties; "
                 "deployment with a prior)\n\n")
    for a, b in (("graph", "indep"), ("temporal", "indep")):
        lines.append(f"- {a} vs {b}: {paired('clean', a, b, main)}\n")
    uni = [r for r in rows if r["unison"]]
    if len(uni) >= 3:
        lines.append(f"\n## Unison stratum (n={len(uni)}), masked median abs err: "
                     + ", ".join(f"{a} {np.median([err(r, 'masked', a) for r in uni]):.1f}"
                                 for a in ARMS) + "\n")
    walls = [v["wall"] for r in rows for k, v in r.items()
             if isinstance(v, dict) and "wall" in v]
    lines.append(f"\nwall per fit incl. Laplace: median {np.median(walls):.0f} s\n")
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
