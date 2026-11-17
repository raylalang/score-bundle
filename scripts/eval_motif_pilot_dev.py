#!/usr/bin/env python
"""Stage one of the motif study, first pilot (DEVELOPMENT ONLY, exploratory, no claim).

Question (docs/slides/proposal_endgoal.tex, frames 6-8): do annotated motif
occurrences carry expressive information beyond temporal smoothing?  The
score graph gains one edge family built from the BPS-motif annotations
(Chou et al.): a note i in occurrence O_k of type q gets the relative
position u_i = (b_i - b_k^start) / (b_k^end - b_k^start), and notes of two
different occurrences of the same type are joined by

    (W_motif)_ij = exp(-(u_i - u_j)^2 / (2 ell_u^2)),  ell_u = 0.1 (fixed).

The graph becomes W = W_time (+ W_pitch) + w_m W_motif with w_m learned by
the per-piece evidence (log w_m is one more graph parameter, the c_graph
pattern of eval_graphgp.py); the rest of the model (ICM coupling, score
features, floored noise, exact evidence) is the headline model without the
music-model embeddings (score features only, so no contamination question
arises and no GPU is needed).

Arms (paired on identical masks and data):
    time          build_adjacency(ell_p -> inf): the time-only Gaussian graph,
                  the attribution-winning form on both corpora
    time_pitch    build_adjacency(): the thesis's score graph
    time_motif    time + w_m W_motif
    time_pitch_motif  time_pitch + w_m W_motif

Data: the ASAP Beethoven sonata first movements with an annotated
counterpart, first 400 matched notes of one aligned performance per
movement (the thesis's excerpt convention).  Folders 3-1 and 24-1 are
EXCLUDED: they sit in the piano confirmation set, which stays untouched.
Notes are matched to the annotation by (crotchet onset, MIDI pitch) after a
global offset search; a movement is used only if >= 85% of its excerpt
notes match.  Membership: a note belongs to occurrence k if its onset lies
in [start_k, end_k) and its per-note annotated type equals the occurrence
type (the csv_notes column; a note can hold only one type there).

Masking schemes, two seeds each:
    random   the thesis protocol, 40% of notes hidden at random
    occ      whole occurrences hidden: for every type with >= 2 occurrences,
             each occurrence is hidden with probability 0.4, at least one
             occurrence of the type stays observed; non-motif notes observed

Scoring on the hidden notes, per channel: Gaussian log score with the
predictive variance floor of the published path (diag Sigma_y + noise) and
squared error.  Two note sets: ALL hidden notes, and MOTIF hidden notes
(inside an occurrence whose type has at least one other occurrence with an
observed note).  Paired per piece (seeds averaged), bootstrap B = 2000.

Hypotheses, stated before the run (2026-11-04):
  H1  (primary) on MOTIF hidden notes under the occ scheme, time_motif has a
      lower timing log score than time: the paired 95% interval excludes 0.
  H1b the same on timing RMSE.
  Falsification: if the H1 interval covers zero or lies above it, the
  annotated motif edges add nothing measurable beyond two-beat temporal
  smoothing at excerpt scale on this corpus; the pilot is reported as such.
  Report-only: the learned w_m per piece; the pitch arm; the random scheme;
  the other two channels.

    PYTHONPATH=src:scripts python scripts/eval_motif_pilot_dev.py --stage align
    PYTHONPATH=src:scripts python scripts/eval_motif_pilot_dev.py --stage run --shard 0/12
    PYTHONPATH=src:scripts python scripts/eval_motif_pilot_dev.py --stage report

PILOT TWO (2026-11-17, DEVELOPMENT ONLY): the per-channel motif weight.  Pilot
one learned ONE weight w_m inside the adjacency, shared by the three channels
through B (x) K_G, and the velocity bands narrowed too far (+0.069 nats, coverage
0.878 -> 0.823).  The `_pc` arms let the family enter as its own diagonal
coregionalization block through the feature-kernel path of MultiOutputGraphGP:

    K_m = U_m diag(g(nu_m; s_m)) U_m^T,  g = 1/(1 + s_m nu), s_m = 1 fixed,
    from the Laplacian of W_motif on the motif notes (zero elsewhere),
    X_m = U_m diag(sqrt g)  passed as one more feature block, so the evidence
    learns one scale c_m[channel] per channel (n_graph = 0, no eigh per step).

Arms: time, time_motif (pilot one's shared weight), time_motif_pc,
time_pitch_motif_pc; both masking schemes; FOUR seeds (0-3; seeds 0-1 are the
pilot-one masks, bit-identical).  Output results/motif_pilot2/.
  H2  (primary) on MOTIF hidden notes under the occ scheme, time_motif_pc keeps
      the timing log-score gain of pilot one against time (paired 95% interval
      below 0) AND removes the velocity penalty: the velocity dNLL interval
      covers or lies below 0 and velocity coverage is >= 0.87.
  Falsification: the timing gain lost => the family's value came from the
      shared coupling, not from the edges; the velocity penalty kept => the
      penalty is not the shared weight; reported as such either way.
  Report-only: c_m per channel per piece; the pitch arm; the random scheme;
      the pilot-one arm on four seeds (seed stability).

STAGE TWO (2026-11-17/18, DEVELOPMENT ONLY): predicted occurrences.  The edge
family is built from the ISMIR 2026 model's held-out (cross-validated)
predictions instead of the annotations (`--occ-source pred --pred-root DIR
--pred-q Q`: DIR/sweep_qQ/fold*/motifs/NN-1.json, occurrence = a list of
(crotchet onset, pitch) notes, type = pattern index).  A predicted occurrence
is mapped to the excerpt by the same (onset, pitch) key after the piece's
annotation offset; its span is [min onset, max onset] of the matched notes.
MASKS AND THE SCORING SET STAY ON THE ANNOTATED OCCURRENCES (identical cells to
pilots one and two), so the only change is the edge source.  Arms:
time_motif_pred, time_motif_pc_pred (and the pitch variants).
  H3  on MOTIF hidden notes under the occ scheme, time_motif_pc_pred has a
      timing log-score gain against time whose point estimate is at least half
      of the annotated arm's on the same cells.
  Falsification: the interval covers 0 => the predictions do not carry the
      annotated edges' value at excerpt scale; reported as such.
  Report-only: note-level precision/recall of predicted vs annotated motif
      membership per piece.

    PYTHONPATH=src:scripts python scripts/eval_motif_pilot_dev.py --stage align \
        --out results/motif_pilot2
    PYTHONPATH=src:scripts python scripts/eval_motif_pilot_dev.py --stage run \
        --out results/motif_pilot2 --seeds 4 \
        --arms time,time_motif,time_motif_pc,time_pitch_motif_pc --shard 0/14
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
import warnings

import numpy as np

ASAP = "/home/ray/Research/data/asap-dataset"
BPS = "/home/ray/Research/ISMIR/graph-motif-discovery/datasets/Beethoven_motif-main"
OUT = "results/motif_pilot"          # pilot one; --out overrides
ARMS = ["time", "time_pitch", "time_motif", "time_pitch_motif"]   # --arms overrides
SCHEMES = ["random", "occ"]           # --schemes overrides
SEEDS = 2                             # --seeds overrides
MAX_NOTES = 400
ELL_U = 0.1
S_M = 1.0                             # fixed shape of the per-channel motif block
PRED = {"root": None, "q": None}      # stage two: --pred-root / --pred-q
EXCLUDE = {"3-1", "24-1"}   # piano confirmation set (positions 30-49 of the shuffle)
CHANNELS = ["tau", "log r", "v"]


# ------------------------------------------------------------------ data
def asap_first_movements():
    """(folder, bps_name) pairs: every annotated first movement with an ASAP score,
    preferring the *_no_repeat folder when both exist (the annotation has no
    written-out repeats)."""
    out = []
    for n in range(1, 33):
        base = f"{n}-1"
        if base in EXCLUDE:
            continue
        cands = [f"{base}_no_repeat", base]
        for c in cands:
            if os.path.isdir(f"{ASAP}/Beethoven/Piano_Sonatas/{c}"):
                out.append((c, f"{n:02d}-1"))
                break
    return out


def bps_annotation(name):
    rows = list(csv.DictReader(open(f"{BPS}/csv_notes/{name}.csv")))
    onset = np.array([float(r["onset"]) for r in rows])
    pitch = np.array([int(r["midi_number"]) for r in rows])
    ntype = [r["type"].strip() for r in rows]
    occ = []
    for r in csv.DictReader(open(f"{BPS}/csv_label/{name}.csv")):
        occ.append({"start": float(r["start"]), "end": float(r["end"]),
                    "type": r["type"].strip(), "track": int(r["track"])})
    return onset, pitch, ntype, occ


def load_excerpt(folder, max_notes=MAX_NOTES):
    """First aligned performance of the folder -> (score_m, y, crotchet onsets)."""
    from score_bundle import features
    from score_bundle.score import Score
    import pretty_midi
    warnings.filterwarnings("ignore")
    ann = features.load_asap_annotations(ASAP)
    perfs = sorted(p for p in ann if p.startswith(f"Beethoven/Piano_Sonatas/{folder}/")
                   and ann[p].get("score_and_performance_aligned"))
    if not perfs:
        return None
    perf = perfs[0]
    score, obs = features.load_asap(perf, ASAP, annotations=ann)
    # crotchet onsets of the score notes from the score MIDI ticks (unit-free)
    pm = pretty_midi.PrettyMIDI(f"{ASAP}/{os.path.dirname(perf)}/midi_score.mid")
    notes = features._midi_notes_sorted(pm)
    q_all = np.array([pm.time_to_tick(n.start) / pm.resolution for n in notes])
    score_m, y = features.asap_performance_variables(score, obs)
    q = q_all[np.asarray(obs["mask"], dtype=bool)]
    if len(score_m) > max_notes:
        score_m = Score(score_m.notes[:max_notes]); y = y[:max_notes]; q = q[:max_notes]
    return perf, score_m, y, q


def match_annotation(q, pitch, b_onset, b_pitch):
    """Index into the annotation for each excerpt note (or -1), after a global
    offset search on the first 30 notes of each side."""
    best = (-1, 0.0)
    for off in np.unique(np.round(b_onset[:30][:, None] - q[:30][None, :], 3)):
        key = set(zip(np.round((b_onset - off) * 16).astype(int), b_pitch))
        m = sum((int(round(x * 16)), p) in key for x, p in zip(q, pitch))
        if m > best[0]:
            best = (m, float(off))
    off = best[1]
    key = {}
    for k, (x, p) in enumerate(zip(np.round((b_onset - off) * 16).astype(int), b_pitch)):
        key.setdefault((x, p), k)
    idx = np.array([key.get((int(round(x * 16)), p), -1) for x, p in zip(q, pitch)])
    return idx, off


def occurrence_membership(q, idx, ntype, occ, off):
    """List of occurrences restricted to the excerpt: dicts with 'type', 'notes'
    (excerpt indices), 'u' (relative positions)."""
    out = []
    for k, o in enumerate(occ):
        s, e = o["start"] - off, o["end"] - off
        if e <= s:
            continue
        members = [i for i in range(len(q))
                   if s <= q[i] < e and idx[i] >= 0 and ntype[idx[i]] == o["type"]]
        if len(members) < 2:
            continue
        u = (q[members] - s) / (e - s)
        out.append({"k": k, "type": o["type"], "notes": np.array(members), "u": u,
                    "start": s, "end": e})
    return out


def motif_adjacency(n, occs, ell_u=ELL_U):
    W = np.zeros((n, n))
    for a in range(len(occs)):
        for b in range(a + 1, len(occs)):
            if occs[a]["type"] != occs[b]["type"]:
                continue
            ia, ib = occs[a]["notes"], occs[b]["notes"]
            du = occs[a]["u"][:, None] - occs[b]["u"][None, :]
            blk = np.exp(-(du ** 2) / (2 * ell_u ** 2))
            W[np.ix_(ia, ib)] = np.maximum(W[np.ix_(ia, ib)], blk)
            W[np.ix_(ib, ia)] = np.maximum(W[np.ix_(ib, ia)], blk.T)
    np.fill_diagonal(W, 0.0)
    return W


def predicted_occurrences(q, pitch, off, name):
    """Stage two: the ISMIR model's held-out predictions for annotation `name`,
    mapped to the excerpt by (crotchet onset - off, pitch).  Returns a list in
    the same form as occurrence_membership (type = pattern index)."""
    import json
    hits = glob.glob(f"{PRED['root']}/sweep_q{PRED['q']}/fold*/motifs/{name}.json")
    if len(hits) != 1:
        raise RuntimeError(f"{name}: {len(hits)} prediction files under {PRED}")
    pats = json.load(open(hits[0]))["patterns"]
    key = {}
    for i, (x, pp) in enumerate(zip(np.round(q * 16).astype(int), pitch)):
        key.setdefault((x, int(pp)), i)
    out = []
    for t, pat in enumerate(pats):
        for k, occ in enumerate(pat):
            members = sorted({key[(int(round((on - off) * 16)), int(pp))]
                              for on, pp in occ
                              if (int(round((on - off) * 16)), int(pp)) in key})
            if len(members) < 2:
                continue
            s, e = q[members].min(), q[members].max()
            if e <= s:
                continue
            out.append({"k": (t, k), "type": f"pred{t}", "notes": np.array(members),
                        "u": (q[members] - s) / (e - s), "start": s, "end": e})
    return out


def motif_feature_block(n, occs, s_m=S_M, ell_u=ELL_U):
    """Per-channel design (pilot two): X_m with X_m X_m^T = K_m, the shape-
    normalized graph kernel of the motif subgraph (zero rows for non-motif
    notes), so that the feature-kernel path learns one scale per channel."""
    from score_bundle.graph import laplacian
    W = motif_adjacency(n, occs, ell_u)
    on = np.where(W.sum(1) > 0)[0]
    X = np.zeros((n, len(on)))
    if len(on) == 0:
        return X
    nu, U = np.linalg.eigh(laplacian(W[np.ix_(on, on)]))
    g = 1.0 / (1.0 + s_m * np.clip(nu, 0.0, None))
    X[on] = U * np.sqrt(g)
    return X


def parse_arm(arm):
    """'time_pitch_motif_pc_pred' -> (base, motif, per_channel, predicted)."""
    a = arm
    pred = a.endswith("_pred"); a = a[:-5] if pred else a
    pc = a.endswith("_pc"); a = a[:-3] if pc else a
    motif = a.endswith("_motif"); a = a[:-6] if motif else a
    assert a in ("time", "time_pitch"), arm
    assert motif or not (pc or pred), arm
    return a, motif, pc, pred


def build_pieces():
    rows, pieces = [], []
    for folder, name in asap_first_movements():
        got = load_excerpt(folder)
        if got is None:
            rows.append((folder, name, "no aligned performance")); continue
        perf, score_m, y, q = got
        b_on, b_pi, ntype, occ = bps_annotation(name)
        idx, off = match_annotation(q, score_m.pitch, b_on, b_pi)
        frac = float((idx >= 0).mean())
        occs = occurrence_membership(q, idx, ntype, occ, off)
        types = {}
        for o in occs:
            types.setdefault(o["type"], []).append(o)
        n_rep = sum(len(v) for v in types.values() if len(v) >= 2)
        motif_notes = np.zeros(len(q), dtype=bool)
        for t, v in types.items():
            if len(v) >= 2:
                for o in v:
                    motif_notes[o["notes"]] = True
        rows.append((folder, name, f"{len(q)} notes, matched {frac:.2f}, offset {off:+.2f}, "
                     f"{len(occs)} occurrences ({n_rep} of repeated types, "
                     f"{len([t for t,v in types.items() if len(v)>=2])} types), "
                     f"{int(motif_notes.sum())} motif notes"))
        if frac < 0.85 or n_rep < 2:
            rows[-1] = rows[-1][:2] + (rows[-1][2] + "  -> EXCLUDED",)
            continue
        piece = {"folder": folder, "bps": name, "performance": perf,
                 "pitch": score_m.pitch, "onset": score_m.onset,
                 "duration": score_m.duration, "voice": score_m.voice,
                 "y": np.asarray(y), "q": q, "occs": occs, "offset": off,
                 "motif_notes": motif_notes, "match_frac": frac}
        if PRED["root"]:
            piece["occs_pred"] = predicted_occurrences(q, score_m.pitch, off, name)
            pm = np.zeros(len(q), dtype=bool)
            for o in piece["occs_pred"]:
                pm[o["notes"]] = True
            tp = int((pm & motif_notes).sum())
            piece["pred_prec"] = tp / max(int(pm.sum()), 1)
            piece["pred_rec"] = tp / max(int(motif_notes.sum()), 1)
            rows[-1] = rows[-1][:2] + (rows[-1][2] + f"; predicted: {len(piece['occs_pred'])} "
                       f"occurrences, {int(pm.sum())} notes, precision {piece['pred_prec']:.2f} "
                       f"recall {piece['pred_rec']:.2f}",)
        pieces.append(piece)
    return pieces, rows


# ------------------------------------------------------------------ masks
def make_mask(p, scheme, seed):
    from score_bundle.imputation_eval import random_mask
    n = len(p["y"])
    rng = np.random.default_rng(1000 * seed + 17)
    if scheme == "random":
        return random_mask(n, rng, observed_frac=0.6)
    mask = np.ones(n, dtype=bool)
    types = {}
    for o in p["occs"]:
        types.setdefault(o["type"], []).append(o)
    for t, v in sorted(types.items()):
        if len(v) < 2:
            continue
        hide = rng.random(len(v)) < 0.4
        if hide.all():
            hide[rng.integers(len(v))] = False
        if not hide.any():
            hide[rng.integers(len(v))] = True
        for o, h in zip(v, hide):
            if h:
                mask[o["notes"]] = False
    if mask.all():
        mask[rng.integers(n)] = False
    return mask


def motif_eval_set(p, mask):
    """Hidden motif notes whose type has another occurrence with an observed note."""
    types = {}
    for o in p["occs"]:
        types.setdefault(o["type"], []).append(o)
    sel = np.zeros(len(mask), dtype=bool)
    for t, v in types.items():
        if len(v) < 2:
            continue
        for o in v:
            others_obs = any(mask[o2["notes"]].any() for o2 in v if o2 is not o)
            if others_obs:
                sel[o["notes"]] = True
    return sel & ~mask


# ------------------------------------------------------------------ model
def arm_setup(p, arm):
    from eval_graphgp import zscore_cols
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import piece_score
    from score_bundle.graph import build_adjacency, laplacian

    score = piece_score(p)
    X = zscore_cols(rich_score_features(score, rff_dim=0))
    feats = [np.concatenate([X, np.ones((len(X), 1))], axis=1)]
    base, motif, pc, pred = parse_arm(arm)
    ell_p = 1e6 if base == "time" else 4.0
    W0 = build_adjacency(score, ell_p=ell_p)
    occs = p["occs_pred"] if pred else p["occs"]
    if motif and not pc:
        Wm = motif_adjacency(len(score), occs)

        def graph_eig(gp_params):  # [log w_m]
            return np.linalg.eigh(laplacian(W0 + np.exp(gp_params[0]) * Wm))
        return feats, graph_eig, 1, np.log([1.0])
    if motif and pc:
        feats = feats + [motif_feature_block(len(score), occs)]
    eig = np.linalg.eigh(laplacian(W0))

    def graph_eig(gp_params):
        return eig
    return feats, graph_eig, 0, np.zeros(0)


def run_cell(p, arm, mask, maxiter=200):
    from eval_graphgp import fit_and_predict
    feats, graph_eig, n_graph, g0 = arm_setup(p, arm)
    Y = np.asarray(p["y"], dtype=float)
    cell, info = fit_and_predict(Y, mask, feats, graph_eig, n_graph, g0, "additive",
                                 maxiter=maxiter)
    return cell, info


# ------------------------------------------------------------------ stages
def stage_align(args):
    pieces, rows = build_pieces()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/pieces.pkl", "wb") as fh:
        pickle.dump(pieces, fh)
    lines = ["| ASAP folder | annotation | excerpt |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    open(f"{OUT}/align.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"{len(pieces)} pieces kept -> {OUT}/pieces.pkl")


def stage_run(args):
    with open(f"{OUT}/pieces.pkl", "rb") as fh:
        pieces = pickle.load(fh)
    k, n = map(int, args.shard.split("/"))
    jobs = [(pi, sc, s, arm) for pi in range(len(pieces)) for sc in SCHEMES
            for s in range(SEEDS) for arm in ARMS]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    if args.limit:
        jobs = jobs[:args.limit]
    out = {}
    import time
    for pi, sc, s, arm in jobs:
        p = pieces[pi]
        mask = make_mask(p, sc, s)
        t0 = time.time()
        cell, info = run_cell(p, arm, mask, maxiter=args.maxiter)
        held = ~mask
        out[(pi, sc, s, arm)] = {"cell": cell, "held_idx": np.where(held)[0],
                                 "motif_sel": motif_eval_set(p, mask)[held],
                                 "info": info}
        print(f"[{k}/{n}] {p['folder']} {sc} seed{s} {arm}: {time.time()-t0:.0f}s "
              f"hidden {int(held.sum())} motif-hidden {int(motif_eval_set(p, mask).sum())} "
              f"{info.get('graph_params','')}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/cells.shard{k}.pkl", "wb") as fh:
        pickle.dump(out, fh)


def bootstrap_ci(vals, B=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stage_report(args):
    from score_bundle.metrics import gaussian_nll
    with open(f"{OUT}/pieces.pkl", "rb") as fh:
        pieces = pickle.load(fh)
    cells = {}
    for f in glob.glob(f"{OUT}/cells.shard*.pkl"):
        with open(f, "rb") as fh:
            cells.update(pickle.load(fh))
    lines = [f"# Motif pilot (dev): {len(pieces)} Beethoven first movements, "
             f"{len(cells)} cells\n"]
    effects = {}
    for sc in SCHEMES:
        for subset in ["motif", "all"]:
            lines.append(f"\n## scheme = {sc}, hidden notes = {subset}\n")
            lines.append("| arm | channel | NLL | RMSE | cov@0.90 | dNLL vs time [95% CI] | dRMSE vs time [95% CI] | n pieces |")
            lines.append("|---|---|---|---|---|---|---|---|")
            per = {}  # (arm, ch) -> per-piece (nll, mse) averaged over seeds
            for arm in ARMS:
                for c in range(3):
                    vals = []
                    for pi in range(len(pieces)):
                        nl, ms, cv, ns = [], [], [], []
                        for s in range(SEEDS):
                            r = cells.get((pi, sc, s, arm))
                            if r is None:
                                continue
                            yt, pr, sd, ch = r["cell"]
                            sel = ch == c
                            if subset == "motif":
                                sel &= np.tile(r["motif_sel"], 3)
                            if sel.sum() < 3:
                                continue
                            z = (yt[sel] - pr[sel]) / sd[sel]
                            nl.append(float(np.mean(0.5 * z ** 2 + np.log(sd[sel]) + 0.5 * np.log(2 * np.pi))))
                            ms.append(float(np.mean((yt[sel] - pr[sel]) ** 2)))
                            cv.append(float(np.mean(np.abs(z) < 1.6449)))
                            ns.append(int(sel.sum()))
                        if nl:
                            vals.append((pi, np.mean(nl), np.mean(ms), np.mean(cv), np.sum(ns)))
                    per[(arm, c)] = {v[0]: v[1:] for v in vals}
            for arm in ARMS:
                for c in range(3):
                    d = per[(arm, c)]; d0 = per[("time", c)]
                    common = sorted(set(d) & set(d0))
                    if not common:
                        continue
                    nll = np.mean([d[i][0] for i in common])
                    rmse = np.sqrt(np.mean([d[i][1] for i in common]))
                    cov = np.mean([d[i][2] for i in common])
                    if arm == "time":
                        dn = dr = "—"
                    else:
                        m, lo, hi = bootstrap_ci([d[i][0] - d0[i][0] for i in common])
                        star = "*" if (hi < 0 or lo > 0) else ""
                        dn = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"
                        m, lo, hi = bootstrap_ci([np.sqrt(d[i][1]) - np.sqrt(d0[i][1]) for i in common])
                        star = "*" if (hi < 0 or lo > 0) else ""
                        dr = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"
                        mn, lon, hin = bootstrap_ci([d[i][0] - d0[i][0] for i in common])
                        effects[f"{sc}|{subset}|{arm}|{CHANNELS[c]}"] = {
                            "dnll": [mn, lon, hin], "drmse": [m, lo, hi],
                            "n": len(common), "nll": float(nll), "rmse": float(rmse),
                            "cov": float(cov)}
                    lines.append(f"| {arm} | {CHANNELS[c]} | {nll:.3f} | {rmse:.4f} | {cov:.3f} | {dn} | {dr} | {len(common)} |")
    # learned motif weights (shared-weight arms) and per-channel scales (pc arms)
    import json
    extra = {}
    shared = [a for a in ARMS if parse_arm(a)[1] and not parse_arm(a)[2]]
    pcarms = [a for a in ARMS if parse_arm(a)[2]]
    for arm in shared:
        lines.append(f"\n## learned w_m per piece ({arm} arm, geometric mean over seeds)\n")
        lines.append("| piece | " + " | ".join(f"w_m ({sc})" for sc in SCHEMES) + " | occurrences | motif notes |")
        lines.append("|---|" + "---|" * len(SCHEMES) + "---|---|")
        wm = {}
        for pi, p in enumerate(pieces):
            ws = {}
            for sc in SCHEMES:
                w = [cells[(pi, sc, s, arm)]["info"]["graph_params"][0]
                     for s in range(SEEDS) if (pi, sc, s, arm) in cells]
                ws[sc] = float(np.exp(np.mean(np.log(w)))) if w else np.nan
            wm[p["folder"]] = ws.get("occ", np.nan)
            lines.append(f"| {p['folder']} | " + " | ".join("—" if np.isnan(ws[sc]) else f"{ws[sc]:.2f}" for sc in SCHEMES)
                         + f" | {len(p['occs'])} | {int(p['motif_notes'].sum())} |")
        extra[f"w_m_occ|{arm}"] = wm
    k, ntri = 3, 6
    for arm in pcarms:
        lines.append(f"\n## learned per-channel motif scale c_m per piece ({arm} arm, occ scheme, geometric mean over seeds)\n")
        lines.append("| piece | c_m tau | c_m log r | c_m v | c_score tau | c_score log r | c_score v |")
        lines.append("|---|---|---|---|---|---|---|")
        cm = {}
        for pi, p in enumerate(pieces):
            xs = [np.asarray(cells[(pi, "occ", s, arm)]["info"]["x_hat"])
                  for s in range(SEEDS) if (pi, "occ", s, arm) in cells]
            if not xs:
                continue
            x = np.mean(xs, axis=0)           # mean of log-scales = geometric mean
            c0 = np.exp(x[1 + ntri: 1 + ntri + k]); c1 = np.exp(x[1 + ntri + k: 1 + ntri + 2 * k])
            cm[p["folder"]] = c1.tolist()
            lines.append(f"| {p['folder']} | " + " | ".join(f"{v:.3g}" for v in c1) + " | "
                         + " | ".join(f"{v:.3g}" for v in c0) + " |")
        extra[f"c_m_occ|{arm}"] = cm
    if pieces and "pred_prec" in pieces[0]:
        lines.append("\n## predicted vs annotated motif membership (note level)\n")
        lines.append("| piece | predicted occurrences | precision | recall |")
        lines.append("|---|---|---|---|")
        for p in pieces:
            lines.append(f"| {p['folder']} | {len(p['occs_pred'])} | {p['pred_prec']:.2f} | {p['pred_rec']:.2f} |")
        extra["pred_membership"] = {p["folder"]: [p["pred_prec"], p["pred_rec"]] for p in pieces}
    txt = "\n".join(lines) + "\n"
    open(f"{OUT}/report.md", "w").write(txt)
    out = {"effects": effects, "arms": ARMS, "seeds": SEEDS, "schemes": SCHEMES}
    out.update(extra)
    if "w_m_occ|time_motif" in extra:
        out["w_m_occ"] = extra["w_m_occ|time_motif"]   # pilot-one key, kept for the figure script
    json.dump(out, open(f"{OUT}/effects.json", "w"), indent=1)
    print(txt)


def main():
    global OUT, ARMS, SCHEMES, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["align", "run", "report"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--occ-source", choices=["annot", "pred"], default="annot")
    ap.add_argument("--pred-root", default="/home/ray/Research/ISMIR/graph-motif-discovery/runs/camera_ready_sweep")
    ap.add_argument("--pred-q", default=None, help="sweep quantile dir suffix, e.g. 80")
    args = ap.parse_args()
    OUT, SEEDS = args.out, args.seeds
    ARMS = [a for a in args.arms.split(",") if a]
    SCHEMES = [a for a in args.schemes.split(",") if a]
    for a in ARMS:
        parse_arm(a)
    if args.occ_source == "pred" or any(parse_arm(a)[3] for a in ARMS):
        assert args.pred_q, "--pred-q is required for predicted occurrences"
        PRED["root"], PRED["q"] = args.pred_root, args.pred_q
    {"align": stage_align, "run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
