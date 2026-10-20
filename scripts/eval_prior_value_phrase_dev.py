#!/usr/bin/env python
"""The prior-value experiment at PHRASE scale: does the score graph's
pitch-proximity coupling carry intonation information when a pitch
recurs at a distance?  (DEV, exploratory, no claims.)

Follow-up to the chain-scale negative (results/prior_value_v2_tuning_dev.md):
on 3-5-note chains, with the common-mode tuning scalar in every arm, the
independent, temporal and score-graph priors tie.  There, a recurring
pitch was always adjacent or nearly so, where the temporal chain reaches
it too.  This study selects longer legato runs (6-10 consecutive notes,
gaps <= 40 ms as in the chain machinery, no rests) in which the hidden
(middle) note's pitch recurs at index distance >= 2 and NOT at distance
1, so that the score graph has a direct pitch edge to the recurrence
and the temporal chain can reach it only through intervening notes.
Cells come from ALL development tracks (78), capped at 4 per track and
subsampled to 48 with rng(0), stratified by nothing (the selection rule
is the stratum).  Protocol v2 (hidden note dropped from the design) and
the free per-chain tuning scalar (sd 50 cents) in every arm, inherited
unchanged from scripts/eval_prior_value_dev.py --- this script changes
the cell selection and nothing else.

Committed hypotheses, stated before the run:
  H-C1 (score-specific structure): on masked cells the score-graph arm
       beats the temporal arm, paired, because of the distant-recurrence
       edge.  Falsification: the graph-vs-temporal CI covers zero =>
       pitch-proximity coupling adds nothing measurable at phrase scale
       either, and the thesis's methodological hypothesis has no
       support for intonation at any span tested; reported verbatim.
  H-C2 (structure vs none, given the common mode): graph (and temporal)
       vs the scalar-augmented independent arm, paired; expected small
       (the chain-scale result was a tie).
  H-C3 (control): clean rows tie.
  Also recorded: the recurrence distance in notes and seconds, and the
  score-graph edge weight between the hidden note and its nearest
  recurrence, for a dose-response reading.

    PYTHONPATH=src:scripts OMP_NUM_THREADS=2 python scripts/eval_prior_value_phrase_dev.py smoke
    bash scripts/run_prior_value_phrase_eval.sh 12   # sharded + report
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time

os.environ["PRIORVAL_DROP_HIDDEN"] = "1"
os.environ.setdefault("PRIORVAL_TUNING_SD", "50")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

import eval_prior_value_dev as pv                      # noqa: E402
from eval_phase3_chain_dev import (SR, chain_segment, NOTE_MIN_S,      # noqa: E402
                                   NOTE_MAX_S, GAP_LO, GAP_HI)
from eval_phase3_curve_dev import load_hypers         # noqa: E402

CELLS_DIR = "results/phase3_cells"
PREFIX = "priorval_phrase"
OUT_MD = "results/prior_value_phrase_dev.md"
ARMS = pv.ARMS
PHRASE_MIN, PHRASE_MAX = 6, 10
CAP_PER_TRACK = 4
N_CELLS = 48
_Z90 = pv._Z90
assert pv.DROP_HIDDEN and pv.TUNING_SD == 50.0


def legato_runs(notes):
    on = np.asarray(notes["onset"], float)
    du = np.asarray(notes["duration"], float)
    n = on.size
    elig = (du >= NOTE_MIN_S) & (du <= NOTE_MAX_S)
    out, i = [], 0
    while i < n:
        if not elig[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and elig[j + 1]
               and GAP_LO <= on[j + 1] - (on[j] + du[j]) <= GAP_HI):
            j += 1
        out.append(list(range(i, j + 1)))
        i = j + 1
    return out


def select_phrases(notes, midi, gt_c):
    """Non-overlapping 6-10-note windows whose middle note's pitch recurs
    at index distance >= 2 and not at distance 1."""
    picks = []
    for run in legato_runs(notes):
        L, s = len(run), 0
        while s + PHRASE_MIN <= L:
            w = min(PHRASE_MAX, L - s)
            idx = run[s:s + w]
            j = w // 2
            h = idx[j]
            ok = False
            if np.isfinite(gt_c[h]) and abs(gt_c[h]) <= 150.0:
                same = [q for q in range(w) if q != j and midi[idx[q]] == midi[h]]
                ok = bool(same) and all(abs(q - j) >= 2 for q in same)
            if ok:
                picks.append((idx, j))
                s += w
            else:
                s += 1
    return picks


def iter_cells():
    import soundfile as sf
    from scipy.signal import resample_poly
    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.graph import build_adjacency
    from score_bundle.score import Score

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    cands = []
    for p, tr in dev_unique_tracks():
        key = (p.index, tr.number)
        if key not in data or not os.path.exists(tr.audio):
            continue
        d = data[key]
        notes = read_notes_annotation(tr.notes)
        picks = select_phrases(notes, np.asarray(d["midi"]), d["est_gt"][:, 0])
        for idx, j in picks[:CAP_PER_TRACK]:
            cands.append((key, idx, j))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cands))[:N_CELLS]
    chosen = sorted(order.tolist())
    cache = {}
    for gidx, ci in enumerate(chosen):
        key, idx, j = cands[ci]
        d = data[key]
        tr = next(t for p, t in dev_unique_tracks()
                  if (p.index, t.number) == key)
        if key not in cache:
            notes = read_notes_annotation(tr.notes)
            audio48, sr48 = sf.read(tr.audio)
            cache[key] = (notes, resample_poly(np.asarray(audio48, float),
                                               SR, int(sr48)))
        notes, audio = cache[key]
        gt_c = d["est_gt"][:, 0]
        var_c = float(np.var(gt_c[np.isfinite(gt_c) & (np.abs(gt_c) <= 150.0)]))
        x, t, t0, b_annot = chain_segment(audio, notes, idx)
        cuts = np.concatenate([[0.0], b_annot, [t[-1]]])
        mask = np.ones(x.size, dtype=bool)
        a = int(np.searchsorted(t, cuts[j] - pv.MASK_GUARD_S))
        b = int(np.searchsorted(t, cuts[j + 1] + pv.MASK_GUARD_S))
        mask[a:b] = False
        midis = [float(d["midi"][i]) for i in idx]
        onsets = [float(notes["onset"][i]) for i in idx]
        durs = [float(notes["duration"][i]) for i in idx]
        same = [q for q in range(len(idx)) if q != j and midis[q] == midis[j]]
        dist = min(abs(q - j) for q in same)
        q_near = min(same, key=lambda q: abs(q - j))
        sc = Score.from_arrays(np.asarray(midis), np.asarray(onsets),
                               np.asarray(durs), np.zeros(len(idx), dtype=int))
        A = build_adjacency(sc)
        yield gidx, dict(key=key, idx=idx, instr=d["instrument"], x=x, t=t,
                         b_annot=b_annot, midis=midis, onsets=onsets, durs=durs,
                         var_c=var_c, j=j, gt_c=float(gt_c[idx[j]]), mask=mask,
                         recur=True, recur_dist=int(dist),
                         recur_dt=float(abs(onsets[q_near] - onsets[j])),
                         edge_w=float(A[j, q_near]), span=float(t[-1]))


def stage_smoke():
    hypers = load_hypers()
    cells = list(iter_cells())
    print(f"{len(cells)} cells; notes/cell median "
          f"{np.median([len(c['idx']) for _, c in cells]):.0f}; span median "
          f"{np.median([c['span'] for _, c in cells]):.1f}s; recurrence distance "
          f"median {np.median([c['recur_dist'] for _, c in cells]):.0f} notes / "
          f"{np.median([c['recur_dt'] for _, c in cells]):.1f}s; edge weight median "
          f"{np.median([c['edge_w'] for _, c in cells]):.2f}; instruments "
          f"{sorted({c['instr'] for _, c in cells})}")
    gidx, ch = cells[0]
    t0_ = time.time()
    res = pv.run_cell(ch, hypers)
    wall = time.time() - t0_
    print(f"cell 0: {ch['key']} {len(ch['idx'])} notes, hidden #{ch['j']} "
          f"(gt_c {ch['gt_c']:+.1f}, recur dist {ch['recur_dist']}, edge "
          f"{ch['edge_w']:.2f})")
    for k, v in res.items():
        print(f"  {k}: c {v['c']:+.2f} +/- {v['sd']:.2f} [{v['wall']:.0f}s]")
    n = len(cells)
    print(f"PROJECTION: {n} cells x {wall:.0f}s = {n * wall / 3600:.1f} h CPU "
          f"/ 12 shards = {n * wall / 12 / 3600:.1f} h wall", flush=True)


def stage_run(shard):
    k_sh, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(CELLS_DIR, exist_ok=True)
    hypers = load_hypers()
    rows = []
    for gidx, ch in iter_cells():
        if gidx % nsh != k_sh:
            continue
        res = pv.run_cell(ch, hypers)
        rec = {k: ch[k] for k in ("key", "idx", "instr", "j", "gt_c", "recur",
                                  "var_c", "recur_dist", "recur_dt", "edge_w",
                                  "span")}
        rec.update(res)
        rows.append(rec)
        print(f"{ch['key']} {len(ch['idx'])} notes hidden#{ch['j']} gt "
              f"{ch['gt_c']:+.1f} dist {ch['recur_dist']}: masked "
              + " ".join(f"{a}:{res['masked_' + a]['c']:+.1f}" for a in ARMS)
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

    def err(r, cond, arm):
        return abs(r[f"{cond}_{arm}"]["c"] - r["gt_c"])

    def paired(cond, a, b, sub=None):
        d = np.array([err(r, cond, a) - err(r, cond, b) for r in (sub or rows)])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975])
        star = "*" if (hi < 0 or lo > 0) else " "
        return (f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{star} "
                f"(better on {np.mean(d < 0):.0%}, n={d.size})")

    lines = [
        "# The prior-value experiment at phrase scale: distant pitch recurrence "
        "(protocol v2 + tuning scalar; DEV, exploratory, no claims)\n",
        "\n6-10-note legato runs from all development tracks in which the hidden "
        "(middle) note's pitch\nrecurs at index distance >= 2 and not at distance 1; "
        "otherwise the machinery of\n`scripts/eval_prior_value_dev.py` unchanged "
        "(hidden note dropped from the design, free per-chain\ntuning scalar in every "
        "arm). Committed hypotheses and falsification rule in the script\ndocstring "
        "(`scripts/eval_prior_value_phrase_dev.py`). Scored against the quasi-truth "
        "centre.\n",
        f"\n{len(rows)} cells, notes/cell median {np.median([len(r['idx']) for r in rows]):.0f}, "
        f"span median {np.median([r['span'] for r in rows]):.1f} s, recurrence distance "
        f"median {np.median([r['recur_dist'] for r in rows]):.0f} notes / "
        f"{np.median([r['recur_dt'] for r in rows]):.1f} s, hidden-to-recurrence edge "
        f"weight median {np.median([r['edge_w'] for r in rows]):.2f}; instruments: "
        + ", ".join(sorted({r['instr'] for r in rows})) + "\n",
        "\n| condition | arm | median abs err (cents) | cov@90 | median sd |\n"
        "|---|---|---|---|---|\n"]
    for cond in ("masked", "clean"):
        for arm in ARMS:
            e = np.array([err(r, cond, arm) for r in rows])
            z = np.array([(r[f"{cond}_{arm}"]["c"] - r["gt_c"])
                          / max(r[f"{cond}_{arm}"]["sd"], 1e-9) for r in rows])
            sd = np.array([r[f"{cond}_{arm}"]["sd"] for r in rows])
            lines.append(f"| {cond} | {arm} | {np.median(e):.2f} | "
                         f"{np.mean(np.abs(z) <= _Z90):.2f} | {np.median(sd):.2f} |\n")
    dp = np.array([abs(r["gt_c"]) for r in rows])
    lines.append(f"| masked | (deadpan reference) | {np.median(dp):.2f} | --- | --- |\n")
    lines.append("\n## Paired contrasts, masked condition (negative favours the "
                 "first arm)\n\n")
    for a, b in (("graph", "temporal"), ("graph", "indep"), ("temporal", "indep")):
        lines.append(f"- {a} vs {b}: {paired('masked', a, b)}\n")
    hi_w = [r for r in rows if r["edge_w"] >= np.median([q["edge_w"] for q in rows])]
    if len(hi_w) >= 5:
        lines.append(f"- graph vs temporal, upper half of edge weight only: "
                     f"{paired('masked', 'graph', 'temporal', hi_w)}\n")
    lines.append("\n## Paired contrasts, clean condition (H-C3: expect ties)\n\n")
    for a, b in (("graph", "indep"), ("temporal", "indep")):
        lines.append(f"- {a} vs {b}: {paired('clean', a, b)}\n")
    d = np.array([err(r, "masked", "graph") - err(r, "masked", "temporal") for r in rows])
    w = np.array([r["edge_w"] for r in rows])
    lines.append(f"\nDose-response: Pearson r between (graph - temporal) masked error "
                 f"and the hidden-to-recurrence edge weight = {np.corrcoef(d, w)[0, 1]:+.2f}\n")
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
