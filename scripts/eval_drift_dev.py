#!/usr/bin/env python
"""Within-note time-variation study (DEV ONLY, read-only).

The committee question behind this study: the constant-parameter vibrato
model eq:cents-curve/eq:vibrato and the chunk-mean loudness of eq:loudness
compress each note to time-averages -- how much real time variation does
that discard, and does the discarded structure belong in the graph GP?

  A. Is intonation drift music or tracker artifact?  Refit every note that
     is identifiable in BOTH curves with an added linear drift term; compare
     tracker vs ground-truth slopes (significance, sign agreement, rank
     correlation; raw Pearson is tail-dominated and reported with a
     winsorized companion).
  B. Loudness within-note trend: OLS slope over the four chunk log-RMS
     values (the eq:loudness construction), signed, per family, and sized
     against the across-note spread of ell.
  C. Structure: lag-1 autocorrelation of the slopes along the note
     sequence, slope-vs-slope and slope-vs-duration correlations.  This is
     the channel-worthiness test: only across-note structure can benefit
     from the graph prior.
  D. Vibrato extent growth: first- vs second-half extent on long notes.

Reads the dev caches + URMP annotations + audio; fits nothing global; the
confirmation split is never touched.  Results: results/phase2_drift_dev.md.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_drift_dev.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))


def drift_slope(tt, x, f):
    """Slope (and SE, and extent) of x ~ c + s*(t-mean) + a sin + b cos."""
    tc = tt - tt.mean()
    th = 2 * np.pi * f * tt
    A = np.stack([np.ones(tt.size), tc, np.sin(th), np.cos(th)], 1)
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ beta
    dof = max(tt.size - 4, 1)
    try:
        cov = (r @ r / dof) * np.linalg.inv(A.T @ A + 1e-10 * np.eye(4))
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = np.inf
    return float(beta[1]), se, float(np.hypot(beta[2], beta[3]))


def main() -> None:
    import soundfile as sf

    from score_bundle.phase2.intonation import cents_from_f0
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)
    from eval_phase2_real import dev_unique_tracks

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    rows = []
    for ki, (key, d) in enumerate(sorted(data.items())):
        f0c = f0s[key]
        tr = tracks[key]
        notes = read_notes_annotation(tr.notes)
        t_gt, f0_gt = read_f0_annotation(tr.f0s)
        ok = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
        ok &= f0c["prob"] >= np.quantile(f0c["prob"][ok], 0.2)
        ok_gt = np.isfinite(f0_gt) & (f0_gt > 0)
        audio, sr = sf.read(tr.audio)
        audio = np.asarray(audio, dtype=float)
        for i in range(d["onset"].size):
            on, du = notes["onset"][i], notes["duration"][i]
            rec = {"key": key, "i": i, "dur": du, "instr": d["instrument"]}
            if d["ident"][i] and np.isfinite(d["est"][i, 2]):
                sel = ok & (f0c["t"] >= on) & (f0c["t"] < on + du)
                if sel.sum() >= 12:
                    tt = f0c["t"][sel] - on
                    x = cents_from_f0(f0c["f0"][sel], 440.0,
                                      float(d["midi"][i] - 69))
                    f = np.exp(d["est"][i, 2])
                    s, se, _ = drift_slope(tt, x, f)
                    rec.update(sl_tr=s, se_tr=se)
                    if du > 0.8:
                        mid = tt.mean()
                        h1, h2 = tt < mid, tt >= mid
                        if h1.sum() >= 8 and h2.sum() >= 8:
                            _, _, g1 = drift_slope(tt[h1], x[h1], f)
                            _, _, g2 = drift_slope(tt[h2], x[h2], f)
                            rec.update(g1=g1, g2=g2)
            if d["ident_gt"][i] and np.isfinite(d["est_gt"][i, 2]):
                sel = ok_gt & (t_gt >= on) & (t_gt < on + du)
                if sel.sum() >= 12:
                    ttg = t_gt[sel] - on
                    xg = cents_from_f0(f0_gt[sel], 440.0,
                                       float(d["midi"][i] - 69))
                    s, se, _ = drift_slope(ttg, xg,
                                           np.exp(d["est_gt"][i, 2]))
                    rec.update(sl_gt=s, se_gt=se)
            a, b = int(on * sr), int((on + du) * sr)
            seg = audio[max(a, 0):min(b, audio.size)]
            if seg.size >= 8:                    # eq:loudness chunking
                chunks = np.array_split(seg, 4)
                lr = np.array([np.log(np.sqrt(np.mean(c ** 2)) + 1e-8)
                               for c in chunks])
                edges = np.cumsum([0] + [c.size for c in chunks]) / sr
                tc = (edges[:-1] + edges[1:]) / 2
                tcm = tc - tc.mean()
                sl = float(tcm @ lr / (tcm @ tcm))
                r = lr - lr.mean() - sl * tcm
                se = float(np.sqrt((r @ r / 2) / (tcm @ tcm)))
                rec.update(sl_ell=sl, se_ell=se)
            rows.append(rec)
        print(f"track {ki + 1}/{len(data)} {key} done", flush=True)

    def col(name):
        return np.array([r.get(name, np.nan) for r in rows])

    sl_tr, se_tr = col("sl_tr"), col("se_tr")
    sl_gt, se_gt = col("sl_gt"), col("se_gt")
    sl_ell, se_ell = col("sl_ell"), col("se_ell")
    dur = col("dur")
    instr = np.array([r["instr"] for r in rows])
    fams = [("strings", ("vn", "va", "vc", "db")),
            ("winds", ("fl", "cl", "ob", "sax", "bn")),
            ("brass", ("tpt", "hn", "tbn", "tba"))]

    both = np.isfinite(sl_tr) & np.isfinite(sl_gt)
    strong = both & (np.abs(sl_tr) > 2 * se_tr) & (np.abs(sl_gt) > 2 * se_gt)
    print("\n=== A. intonation drift: tracker vs ground truth ===")
    print(f"notes identifiable in both curves: {both.sum()}")
    print(f"slope significant (|s|>2SE): tracker "
          f"{np.mean(np.abs(sl_tr[both]) > 2 * se_tr[both]):.1%}, GT "
          f"{np.mean(np.abs(sl_gt[both]) > 2 * se_gt[both]):.1%}")
    print(f"median |slope|: tracker {np.median(np.abs(sl_tr[both])):.1f}, "
          f"GT {np.median(np.abs(sl_gt[both])):.1f} cents/s "
          f"(median SE ~{np.median(se_tr[both]):.1f})")
    gt_span = np.abs(sl_gt[both]) * dur[both]
    print(f"GT total drift over note: median {np.median(gt_span):.1f}, "
          f"q90 {np.quantile(gt_span, .9):.1f} cents")
    a, b = sl_tr[strong], sl_gt[strong]

    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    wa = np.clip(a, *np.quantile(a, [.02, .98]))
    wb = np.clip(b, *np.quantile(b, [.02, .98]))
    print(f"both-significant notes: {strong.sum()} "
          f"({strong.sum() / both.sum():.0%}): sign agreement "
          f"{np.mean(np.sign(a) == np.sign(b)):.0%}, Spearman "
          f"{np.corrcoef(rank(a), rank(b))[0, 1]:.2f}, winsorized-2% "
          f"Pearson {np.corrcoef(wa, wb)[0, 1]:.2f} (raw Pearson "
          f"{np.corrcoef(a, b)[0, 1]:.2f} is tail-dominated)")
    print("per-family GT drift (n, median |slope| cents/s, "
          "median total cents):")
    for fam, members in fams:
        m = both & np.isin(instr, members)
        if m.sum():
            print(f"  {fam:8s} {m.sum():5d}  "
                  f"{np.median(np.abs(sl_gt[m])):5.1f}  "
                  f"{np.median(np.abs(sl_gt[m]) * dur[m]):5.1f}")

    print("\n=== B. loudness within-note trend ===")
    fin = np.isfinite(sl_ell)
    sig = fin & (np.abs(sl_ell) > 2 * se_ell)
    span = np.abs(sl_ell[fin]) * dur[fin]
    ell_sd = np.mean([np.nanstd(d["ell"]) for d in data.values()])
    print(f"notes {fin.sum()}, slope significant {sig.mean():.1%} of finite")
    print(f"|within-note change|: median {np.median(span):.2f} log-RMS = "
          f"{np.median(span) / ell_sd:.0%} of the across-note sd of ell "
          f"({ell_sd:.2f})")
    print(f"significant slopes negative (decay): "
          f"{np.mean(sl_ell[sig] < 0):.0%}; per family:")
    for fam, members in fams:
        m = sig & np.isin(instr, members)
        if m.sum():
            print(f"  {fam:8s} n={m.sum():5d}  median slope "
                  f"{np.median(sl_ell[m]):+.2f}  neg "
                  f"{np.mean(sl_ell[m] < 0):.0%}")

    print("\n=== C. across-note structure (channel-worthiness) ===")

    def lag1(seqs):
        cs = []
        for v in seqs:
            v = np.asarray(v)
            x0, x1 = v[:-1], v[1:]
            p = np.isfinite(x0) & np.isfinite(x1)
            if p.sum() >= 10:
                cs.append(np.corrcoef(x0[p], x1[p])[0, 1])
        return float(np.mean(cs)), len(cs)

    def seqs_of(name):
        by = {}
        for r in rows:
            by.setdefault(r["key"], {})[r["i"]] = r.get(name, np.nan)
        return [[d.get(i, np.nan) for i in range(max(d) + 1)]
                for d in by.values()]

    c1, nt = lag1(seqs_of("sl_tr"))
    c1e, nte = lag1(seqs_of("sl_ell"))
    print(f"lag-1 autocorr along the note sequence: drift {c1:+.2f} "
          f"({nt} tracks), loudness slope {c1e:+.2f} ({nte} tracks)")
    bb = np.isfinite(sl_tr) & np.isfinite(sl_ell)
    print(f"corr(drift slope, loudness slope) = "
          f"{np.corrcoef(sl_tr[bb], sl_ell[bb])[0, 1]:+.3f} (n={bb.sum()})")
    bd = np.isfinite(sl_tr)
    print(f"corr(|drift slope|, duration) = "
          f"{np.corrcoef(np.abs(sl_tr[bd]), dur[bd])[0, 1]:+.3f}")

    print("\n=== D. vibrato extent growth (long notes, half-split) ===")
    g1, g2 = col("g1"), col("g2")
    gg = np.isfinite(g1) & np.isfinite(g2)
    print(f"notes {gg.sum()}: median(gamma2-gamma1) = "
          f"{np.median(g2[gg] - g1[gg]):+.2f} cents, second half larger on "
          f"{np.mean(g2[gg] > g1[gg]):.0%} "
          f"(medians {np.median(g1[gg]):.1f} vs {np.median(g2[gg]):.1f})")


if __name__ == "__main__":
    main()
