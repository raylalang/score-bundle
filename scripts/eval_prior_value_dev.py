#!/usr/bin/env python
"""The prior-value experiment: does score-informed structure add value to
direct waveform inference? (DEV, exploratory, no claims.)

Three cross-note priors over the per-note intonation constants c, at
matched marginal variance and identical everything else (acoustic
likelihood, data, optimizer, steps, boundaries fixed at annotations):

  indep     c ~ N(0, sigma^2 I)                  (no structure)
  temporal  c ~ N(0, sigma^2 K_chain)            (note-order chain)
  graph     c ~ N(0, sigma^2 K_score)            (score graph: time AND
                                                  pitch proximity)

sigma^2 is the track's own quasi-truth c variance (population scale, no
tuning); K = (I + L)^{-1} diagonal-normalized for both structured arms.

Two conditions per chain:
  masked  the middle eligible note's audio is REMOVED from the
          likelihood (maximal acoustic ambiguity: the waveform-level
          analogue of the thesis's held-out imputation task);
  clean   full audio (the evidence-dominated regime).

Committed hypotheses, stated before the run:
  H1 (ambiguity): under masking, the structured priors recover the
      hidden note's intonation better than the independent prior, which
      can only return the prior mean; the score graph's edge over the
      temporal chain, if any, comes from pitch-proximity coupling and is
      expected mainly where the chain contains a recurring pitch
      (stratified).
  H2 (evidence dominates): with clean audio the three arms tie; the
      prior's value is specific to ambiguity.
  Falsification: H1 fails if the graph arm is not better than the
      independent arm on the masked cells (paired). A rigorous negative
      is a reportable answer to the thesis's central question.

Scoring: hidden-note |c_hat - c_qt| (c_qt = the quasi-truth NLLS centre
on the corpus ground-truth curve), z and coverage at 90% from a joint
Laplace over (c, u) at the MAP, deadpan |c_qt| as the no-information
reference. Sharded run k/n + report; cells in results/phase3_cells/.

THE TUNING-SCALAR FOLLOW-UP (2026-10-15; env PRIORVAL_TUNING_SD=<cents>,
unset = the measured first run, bit-unchanged).  The first run's reading
flagged that much of the structured win may be common-mode: a track's
shared tuning offset, transferred from the observed notes to the hidden
one by either structured prior, while the independent prior can only
return zero.  The follow-up gives EVERY arm a free per-chain tuning
scalar t ~ N(0, sd_t^2), c = t*1 + c~, with c~ under the arm's own prior
as before.  Marginalizing t is exact and needs no new machinery: the
prior covariance becomes K_arm + sd_t^2 * 1 1^T (rank-one, identical
addition in all three arms, so marginal variances stay matched).  With
sd_t = 50 cents the scalar is effectively free: the observed notes pin
it (prior shrinkage of the chain mean <= var_c / (var_c + n_obs sd_t^2),
printed by the smoke verb), which makes the independent arm a
demeaning baseline while keeping one Laplace, one optimizer, one
scoring.  The note-level scale var_c is deliberately left as in the
first run (one change at a time: any movement is the scalar's).

Committed hypotheses, stated before the run:
  H3 (attribution): the independent arm's masked error drops (it now
      receives the common mode).  If the graph-vs-independent paired
      gap on masked cells survives (bootstrap CI excludes zero), the
      first-span headline is NOTE-LEVEL expression transfer; if it
      collapses (CI covers zero), the headline was tuning bookkeeping
      and is reported verbatim as such.  The recurring-pitch stratum
      for graph-vs-temporal stays the pre-stated place for a score-
      specific edge.
  H4 (calibration): masked coverage@90 rises toward nominal for all
      arms (the miscentring diagnosis); if it stays far below, the
      diagnosis was wrong and the Laplace band itself is suspect.
  H5 (control): the clean condition still ties three ways.
  Cross-run: the same 47 cells in the same order as the first run, so
      per-arm tuning-minus-first contrasts are exactly paired when the
      first run's cells are present (they are gitignored artifacts).

    PRIORVAL_TUNING_SD=50 OMP_NUM_THREADS=2 PYTHONPATH=src:scripts \
        python scripts/eval_prior_value_dev.py smoke
    bash scripts/run_prior_value_eval.sh 8 50     # sharded + report

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_prior_value_dev.py smoke
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_prior_value_dev.py run 0/8
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_prior_value_dev.py report
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
ARMS = ("indep", "temporal", "graph")
TUNING_SD = (float(os.environ["PRIORVAL_TUNING_SD"])
             if os.environ.get("PRIORVAL_TUNING_SD") else None)
# separate artifacts per variant: the first run's cells and results file
# are never clobbered by the follow-up (and vice versa)
CELLS_PREFIX = "priorval" if TUNING_SD is None else "priorval_tuning"
BASE_PREFIX = "priorval"
OUT_MD = ("results/prior_value_dev.md" if TUNING_SD is None
          else "results/prior_value_tuning_dev.md")
N_STEPS = 300
S_SCALE = 1.0
MASK_GUARD_S = 0.05   # mask 5x the window softness beyond the note's
                      # extent: without it the note's sigmoid leak onto
                      # unmasked transition samples acts as loud, WRONG
                      # evidence (measured in the smoke: indep arm pulled
                      # 4 sigma from its prior mean by boundary junk)
_Z90 = 1.6448536269514722


def arm_cov(arm: str, midis, onsets, durs, var_c: float) -> np.ndarray:
    """(K, K) prior covariance over c, diagonal-matched to var_c."""
    k_n = len(midis)
    if arm == "indep":
        return _with_tuning(var_c * np.eye(k_n))
    if arm == "temporal":
        A = np.zeros((k_n, k_n))
        for i in range(k_n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    else:
        from score_bundle.graph import build_adjacency
        from score_bundle.score import Score

        score = Score.from_arrays(np.asarray(midis, float),
                                  np.asarray(onsets, float),
                                  np.asarray(durs, float),
                                  np.zeros(k_n, dtype=int))
        A = build_adjacency(score)
    L = np.diag(A.sum(1)) - A
    K = np.linalg.inv(np.eye(k_n) + S_SCALE * L)
    d = np.sqrt(np.diag(K))
    K = K / d[:, None] / d[None, :]
    return _with_tuning(var_c * K)


def _with_tuning(K: np.ndarray) -> np.ndarray:
    """The free per-chain tuning scalar, marginalized: + sd_t^2 1 1^T
    (no-op in the first-run configuration)."""
    if TUNING_SD is None:
        return K
    return K + TUNING_SD ** 2 * np.ones_like(K)


def chain_c_laplace(x, t, midis, fit, hypers, c_prior_cov, sample_mask,
                    n_harm=8, n_chunk=2, amp_var=10.0, softness=0.010):
    """sd of each c_k from the joint (c, u) Laplace at the MAP, with the
    c prior and the sample mask included, boundaries fixed."""
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
    b_lo = torch.as_tensor(fit.b_lo, dtype=torch.float64)
    b_hi = torch.as_tensor(fit.b_hi, dtype=torch.float64)
    msk = (torch.as_tensor(np.asarray(sample_mask, bool))
           if sample_mask is not None else None)
    chol_c = torch.linalg.cholesky(torch.as_tensor(
        np.asarray(c_prior_cov, float), dtype=torch.float64))
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

    def nlj(theta):
        pos, cols, cs = 0, [], []
        prior_term = 0.0
        for k in range(k_n):
            c_k = theta[pos]
            u_k = theta[pos + 1:pos + sizes[k]]
            pos += sizes[k]
            cs.append(c_k)
            ia, ib = slices[k]
            ts = t_t[ia:ib]
            cents = c_k + interp_knots(u_k, knots_t[k], ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, n_harm, n_chunk)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            cols.append(Phi_k * w.unsqueeze(1))
            prior_term = prior_term + 0.5 * (u_k @ torch.cholesky_solve(
                u_k.unsqueeze(1), chols[k]).squeeze(1))
        c_vec = torch.stack(cs)
        prior_term = prior_term + 0.5 * (c_vec @ torch.cholesky_solve(
            c_vec.unsqueeze(1), chol_c).squeeze(1))
        Phi = torch.cat(cols, dim=1)
        if msk is not None:
            ll = collapsed_loglik_torch(x_t[msk], Phi[msk],
                                        fit.noise_var, amp_var)
        else:
            ll = collapsed_loglik_torch(x_t, Phi, fit.noise_var, amp_var)
        return -ll + prior_term

    theta0 = torch.as_tensor(
        np.concatenate([np.concatenate([[fit.c[k]], fit.us[k]])
                        for k in range(k_n)]), dtype=torch.float64)
    H = torch.autograd.functional.hessian(nlj, theta0)
    H = 0.5 * (H + H.T)
    try:
        cov = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        ev, evec = torch.linalg.eigh(H)
        ev = ev.clamp(min=1e-8 * float(ev.max().clamp(min=1.0)))
        cov = (evec / ev) @ evec.T
    cov = cov.numpy()
    sds, pos = [], 0
    for k in range(k_n):
        sds.append(float(np.sqrt(max(cov[pos, pos], 0.0))))
        pos += sizes[k]
    return np.asarray(sds)


def iter_cells():
    import soundfile as sf
    from scipy.signal import resample_poly

    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.urmp import read_notes_annotation
    from eval_phase3_waveform_dev import selected

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        chains = select_chains(notes)
        if not chains:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        gt_c = d["est_gt"][:, 0]
        var_c = float(np.var(gt_c[np.isfinite(gt_c)
                                  & (np.abs(gt_c) <= 150.0)]))
        for idx in chains:
            interior = [j for j in range(1, len(idx) - 1)
                        if np.isfinite(gt_c[idx[j]])
                        and abs(gt_c[idx[j]]) <= 150.0]
            if not interior:
                continue
            j_mask = interior[len(interior) // 2]
            x, t, t0, b_annot = chain_segment(audio, notes, idx)
            cuts = np.concatenate([[0.0], b_annot, [t[-1]]])
            mask = np.ones(x.size, dtype=bool)
            a = int(np.searchsorted(t, cuts[j_mask] - MASK_GUARD_S))
            b = int(np.searchsorted(t, cuts[j_mask + 1] + MASK_GUARD_S))
            mask[a:b] = False
            midis = [float(d["midi"][i]) for i in idx]
            onsets = [float(notes["onset"][i]) for i in idx]
            durs = [float(notes["duration"][i]) for i in idx]
            recur = any(abs(midis[j] - midis[j_mask]) < 0.5
                        for j in range(len(idx)) if j != j_mask)
            yield gidx, dict(key=key, idx=idx, instr=d["instrument"],
                             x=x, t=t, b_annot=b_annot, midis=midis,
                             onsets=onsets, durs=durs, var_c=var_c,
                             j=j_mask, gt_c=float(gt_c[idx[j_mask]]),
                             mask=mask, recur=recur)
            gidx += 1


def run_cell(ch, hypers, n_steps=N_STEPS):
    from score_bundle.phase3.curve import chain_reference, fit_chain

    out = {}
    for cond in ("masked", "clean"):
        mask = ch["mask"] if cond == "masked" else None
        c0, nv = chain_reference(ch["x"], ch["t"], ch["midis"],
                                 ch["b_annot"], sample_mask=mask)
        for arm in ARMS:
            K_c = arm_cov(arm, ch["midis"], ch["onsets"], ch["durs"],
                          ch["var_c"])
            t0_ = time.time()
            fit = fit_chain(ch["x"], ch["t"], ch["midis"], ch["b_annot"],
                            hypers=hypers, c_init=c0, noise_var=nv,
                            c_prior_cov=K_c, sample_mask=mask,
                            fix_boundaries=True, n_steps=n_steps)
            sds = chain_c_laplace(ch["x"], ch["t"], ch["midis"], fit,
                                  hypers, K_c, mask)
            j = ch["j"]
            out[f"{cond}_{arm}"] = {
                "c": float(fit.c[j]), "sd": float(sds[j]),
                "wall": time.time() - t0_}
    return out


def stage_smoke():
    hypers = load_hypers()
    for gidx, ch in iter_cells():
        t0_ = time.time()
        res = run_cell(ch, hypers)
        wall = time.time() - t0_
        n = sum(1 for _ in iter_cells())
        print(f"cell 0: {ch['key']} notes {ch['idx']} mask #{ch['j']} "
              f"(gt_c {ch['gt_c']:+.1f}, recur={ch['recur']})")
        if TUNING_SD is not None:
            n_obs = len(ch["idx"]) - 1
            shrink = ch["var_c"] / (ch["var_c"] + n_obs * TUNING_SD ** 2)
            print(f"  tuning scalar sd {TUNING_SD:.0f} cents: var_c "
                  f"{ch['var_c']:.0f}, n_obs {n_obs}, prior shrinkage of "
                  f"the chain mean <= {shrink:.3f}")
        for k, v in res.items():
            print(f"  {k}: c {v['c']:+.2f} +/- {v['sd']:.2f} "
                  f"[{v['wall']:.0f}s]")
        print(f"PROJECTION: {n} cells x {wall:.0f}s = "
              f"{n * wall / 3600:.1f} h CPU / 8 shards = "
              f"{n * wall / 8 / 3600:.1f} h wall", flush=True)
        return
    print("no cells")


def stage_run(shard: str) -> None:
    k_sh, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    hypers = load_hypers()
    rows = []
    for gidx, ch in iter_cells():
        if gidx % nsh != k_sh:
            continue
        res = run_cell(ch, hypers)
        rec = {k: ch[k] for k in ("key", "idx", "instr", "j", "gt_c",
                                  "recur", "var_c")}
        rec.update(res)
        rows.append(rec)
        print(f"{ch['key']} notes {ch['idx']} mask#{ch['j']} "
              f"gt {ch['gt_c']:+.1f}: masked "
              + " ".join(f"{a}:{res['masked_' + a]['c']:+.1f}"
                         for a in ARMS)
              + f" [{sum(v['wall'] for v in res.values()):.0f}s]",
              flush=True)
    out = f"{CELLS_DIR}/{CELLS_PREFIX}.shard{k_sh}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} cells)")


def stage_report() -> None:
    rows = []
    for f in sorted(glob.glob(f"{CELLS_DIR}/{CELLS_PREFIX}.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    if not rows:
        print("no shards")
        return
    rng = np.random.default_rng(31)
    base = []
    if TUNING_SD is not None:
        for f in sorted(glob.glob(f"{CELLS_DIR}/{BASE_PREFIX}.shard*.pkl")):
            base.extend(pickle.load(open(f, "rb")))
        if base and [(r["key"], tuple(r["idx"])) for r in base] != \
                [(r["key"], tuple(r["idx"])) for r in rows]:
            print("WARNING: first-run cells do not pair with this run; "
                  "cross-run contrasts skipped")
            base = []

    def err(r, cond, arm):
        return abs(r[f"{cond}_{arm}"]["c"] - r["gt_c"])

    def paired(cond, a, b, sub=None):
        d = np.array([err(r, cond, a) - err(r, cond, b)
                      for r in (sub or rows)])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        return (f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{star} "
                f"(better on {np.mean(d < 0):.0%}, n={d.size})")

    title = ("# The prior-value experiment: score structure in direct "
             "waveform inference (DEV, exploratory, no claims)\n"
             if TUNING_SD is None else
             "# The prior-value experiment, tuning-scalar follow-up: "
             f"free per-chain tuning scalar (sd {TUNING_SD:.0f} cents) "
             "in all arms (DEV, exploratory, no claims)\n")
    lines = [
        title,
        "\nThree cross-note priors over per-note intonation constants at "
        "matched marginal\nvariance, identical likelihood/data/optimizer, "
        "boundaries fixed at annotations.\nConditions: the middle "
        "eligible note's audio MASKED from the likelihood (maximal\n"
        "ambiguity) vs clean. Committed hypotheses and falsification "
        "rule in the script\ndocstring "
        "(`scripts/eval_prior_value_dev.py`). Scored against the "
        "quasi-truth centre.\n",
        f"\n{len(rows)} cells (chains), tracks: "
        + ", ".join(sorted({f"{r['key']}({r['instr']})" for r in rows}))
        + f"; {sum(r['recur'] for r in rows)} cells have a recurring "
        "pitch.\n",
        "\n| condition | arm | median abs err (cents) | cov@90 | "
        "median sd |\n|---|---|---|---|---|\n"]
    for cond in ("masked", "clean"):
        for arm in ARMS:
            e = np.array([err(r, cond, arm) for r in rows])
            z = np.array([(r[f"{cond}_{arm}"]["c"] - r["gt_c"])
                          / max(r[f"{cond}_{arm}"]["sd"], 1e-9)
                          for r in rows])
            sd = np.array([r[f"{cond}_{arm}"]["sd"] for r in rows])
            lines.append(f"| {cond} | {arm} | {np.median(e):.2f} | "
                         f"{np.mean(np.abs(z) <= _Z90):.2f} | "
                         f"{np.median(sd):.2f} |\n")
    dp = np.array([abs(r["gt_c"]) for r in rows])
    lines.append(f"| masked | (deadpan reference) | {np.median(dp):.2f} "
                 "| --- | --- |\n")
    lines.append("\n## Paired contrasts, masked condition (negative "
                 "favours the first arm)\n\n")
    for a, b in (("graph", "indep"), ("temporal", "indep"),
                 ("graph", "temporal")):
        lines.append(f"- {a} vs {b}: {paired('masked', a, b)}\n")
    sub = [r for r in rows if r["recur"]]
    if len(sub) >= 5:
        lines.append(f"- graph vs temporal, recurring-pitch cells only: "
                     f"{paired('masked', 'graph', 'temporal', sub)}\n")
    lines.append("\n## Paired contrasts, clean condition (H2: expect "
                 "ties)\n\n")
    for a, b in (("graph", "indep"), ("temporal", "indep")):
        lines.append(f"- {a} vs {b}: {paired('clean', a, b)}\n")
    if base:
        lines.append("\n## Cross-run, masked condition: this run minus "
                     "the first run, per arm (negative = the scalar "
                     "helped; exactly paired cells)\n\n")
        for arm in ARMS:
            d = np.array([err(r, "masked", arm) - err(b, "masked", arm)
                          for r, b in zip(rows, base)])
            bs = np.array([rng.choice(d, d.size).mean()
                           for _ in range(2000)])
            lo, hi = np.quantile(bs, [.025, .975])
            star = "*" if (hi < 0 or lo > 0) else " "
            cov0 = np.mean([abs((b[f"masked_{arm}"]["c"] - b["gt_c"])
                                / max(b[f"masked_{arm}"]["sd"], 1e-9))
                            <= _Z90 for b in base])
            lines.append(f"- {arm}: {d.mean():+.2f} [{lo:+.2f}, "
                         f"{hi:+.2f}]{star} (better on {np.mean(d < 0):.0%}"
                         f", n={d.size}); first-run cov@90 {cov0:.2f}\n")
        lines.append("\nFirst-run graph vs indep, masked, same cells: "
                     + paired("masked", "graph", "indep", base) + "\n")
    walls = [v["wall"] for r in rows for k, v in r.items()
             if isinstance(v, dict) and "wall" in v]
    lines.append(f"\nwall per fit incl. Laplace: median "
                 f"{np.median(walls):.0f} s\n")
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
