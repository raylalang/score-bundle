#!/usr/bin/env python
"""delta_vib channel-candidate study (URMP development side by default;
in PHASE2_SPLIT=confirmation mode the gt/pyin stages feed the guarded
one-shot pipeline with _conf-suffixed caches — see eval_phase2_real.py).

Answers the last open Phase-2 channel question (docs/phase2_prereg_design.md):
is the vibrato onset delay delta_vib identifiable often enough, and measured
reliably enough, to be worth a channel?  Uses the GATED estimator
(`fit_vibrato_note_gated`, the model draft eq:vibrato actually specifies);
the evaluated bundle's ungated fit cannot measure an onset delay at all
(its delta is a phase modulo one period).

DECISION CRITERION — STATED BEFORE THE NUMBERS WERE LOOKED AT (2026-08-13).
delta_vib joins the Phase-2 channel set only if ALL of:
  C-A  coverage: delta identifiable on >= 25% of vibrato-identifiable notes,
       on BOTH the GT-derived and the pyin curves;
  C-B  agreement: on jointly delta-identifiable notes, median
       |delta_pyin - delta_GT| <= 40 ms (about a quarter vibrato period);
  C-C  non-degradation: the gated fit's intonation error vs the GT-derived
       quasi-truth is not worse than the ungated fit's by more than 10%
       (median absolute error, pyin curves);
  C-D  signal: the across-note spread of identifiable deltas within a track
       exceeds their median reported standard error (else the channel is
       estimator noise).
Anything less: delta_vib stays OUT of the evaluated bundle and out of the
registered claims; the measurement is reported as a development finding.

Stages:
    gt      gated fits on URMP's ground-truth F0 curves (fast, no audio)
    pyin    gated + ungated fits on tracked curves (caches pyin output to
            F0_CACHE for reuse; slow, librosa)
    report  verdict against C-A..C-D (a dev-side decision record; the
            criterion was frozen for the 2026-08-13 study)

    OMP_NUM_THREADS=1 PYTHONPATH=src:scripts python scripts/eval_delta_vib.py gt
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_real import CONF_MODE, RANGES, selected_tracks  # noqa: E402

_SFX = "_conf" if CONF_MODE else ""
F0_CACHE = f".cache/urmp_f0{_SFX or '_dev'}.pkl"
GT_OUT = f".cache/delta_vib_gt{_SFX}.pkl"
PYIN_OUT = f".cache/delta_vib_pyin{_SFX}.pkl"
OUT_MD = f"results/delta_vib{_SFX or '_dev'}.md"
CONF_Q = 0.2          # the estimator chain's measured confidence filter
MIN_FRAMES = 4


def _note_frames(t, cents_ok, onset, duration):
    """Indices of usable frames per note (voiced/finite already applied)."""
    for i in range(onset.size):
        sel = cents_ok & (t >= onset[i]) & (t < onset[i] + duration[i])
        yield i, np.where(sel)[0]


def _fit_track(t, f0, ok, onset, duration, midi, gated=True):
    from score_bundle.phase2.intonation import (cents_from_f0,
                                                fit_vibrato_note,
                                                fit_vibrato_note_gated)
    from score_bundle.phase2.targets import F_REF

    fit = fit_vibrato_note_gated if gated else fit_vibrato_note
    rows = []
    for i, idx in _note_frames(t, ok, onset, duration):
        if idx.size < MIN_FRAMES:
            rows.append(None)
            continue
        cents = cents_from_f0(f0[idx], F_REF, float(midi[i] - 69))
        out = fit(t[idx] - onset[i], cents)
        rows.append(out)
    return rows


def stage_gt() -> None:
    from score_bundle.phase2.targets import hz_to_semitone
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)

    data = {}
    for p, tr in selected_tracks():
        notes = read_notes_annotation(tr.notes)
        t, f0 = read_f0_annotation(tr.f0s)
        ok = np.isfinite(f0) & (f0 > 0)
        midi = hz_to_semitone(notes["pitch_hz"])
        rows = _fit_track(t, f0, ok, notes["onset"], notes["duration"], midi)
        data[(p.index, tr.number)] = {"instrument": tr.instrument,
                                      "gated_gt": rows}
        nid = sum(1 for r in rows if r and r["delta_identifiable"])
        nvib = sum(1 for r in rows if r and r["vibrato_identifiable"])
        print(f"  {p.index:02d} tr{tr.number} {tr.instrument:<4} "
              f"delta-id {nid}/{nvib} vib-id", flush=True)
    with open(GT_OUT, "wb") as fh:
        pickle.dump(data, fh)
    print(f"wrote {GT_OUT}")


def stage_pyin() -> None:
    import soundfile as sf

    from score_bundle.phase2.intonation import extract_f0
    from score_bundle.phase2.targets import hz_to_semitone
    from score_bundle.phase2.urmp import read_notes_annotation

    f0cache = {}
    if os.path.exists(F0_CACHE):
        with open(F0_CACHE, "rb") as fh:
            f0cache = pickle.load(fh)
    data = {}
    for p, tr in selected_tracks():
        key = (p.index, tr.number)
        notes = read_notes_annotation(tr.notes)
        if key not in f0cache:
            audio, sr = sf.read(tr.audio)
            fmin, fmax = RANGES[tr.instrument]
            f0cache[key] = extract_f0(np.asarray(audio, dtype=float), sr,
                                      fmin=float(fmin), fmax=float(fmax))
            with open(F0_CACHE, "wb") as fh:   # checkpoint per track
                pickle.dump(f0cache, fh)
        py = f0cache[key]
        ok = py["voiced"] & np.isfinite(py["f0"]) & (py["f0"] > 0)
        if ok.any():
            floor = np.quantile(py["prob"][ok], CONF_Q)
            ok = ok & (py["prob"] >= floor)
        midi = hz_to_semitone(notes["pitch_hz"])
        args = (py["t"], py["f0"], ok, notes["onset"], notes["duration"], midi)
        data[key] = {"instrument": tr.instrument,
                     "gated_pyin": _fit_track(*args, gated=True),
                     "ungated_pyin": _fit_track(*args, gated=False)}
        nid = sum(1 for r in data[key]["gated_pyin"]
                  if r and r["delta_identifiable"])
        print(f"  {key} delta-id {nid}", flush=True)
    with open(PYIN_OUT, "wb") as fh:
        pickle.dump(data, fh)
    print(f"wrote {PYIN_OUT}")


def stage_report() -> None:
    with open(GT_OUT, "rb") as fh:
        gt = pickle.load(fh)
    with open(PYIN_OUT, "rb") as fh:
        py = pickle.load(fh)

    def rate(rows_key, data):
        num = den = 0
        for d in data.values():
            for r in d[rows_key]:
                if r and r["vibrato_identifiable"]:
                    den += 1
                    num += bool(r["delta_identifiable"])
        return num / max(den, 1), num, den

    ra, na, da = rate("gated_gt", gt)
    rb, nb, db = rate("gated_pyin", py)

    # C-B agreement on jointly identifiable notes
    diffs = []
    for key in gt:
        if key not in py:
            continue
        for rg, rp in zip(gt[key]["gated_gt"], py[key]["gated_pyin"]):
            if rg and rp and rg["delta_identifiable"] \
                    and rp["delta_identifiable"]:
                diffs.append(abs(rp["delta"] - rg["delta"]))
    diffs = np.array(diffs)

    # C-C non-degradation of intonation vs GT quasi-truth
    eg, eu = [], []
    for key in py:
        if key not in gt:
            continue
        for rg, rp_g, rp_u in zip(gt[key]["gated_gt"], py[key]["gated_pyin"],
                                  py[key]["ungated_pyin"]):
            if rg is None or rp_g is None or rp_u is None:
                continue
            eg.append(abs(rp_g["c"] - rg["c"]))
            eu.append(abs(rp_u["c"] - rg["c"]))
    eg, eu = np.median(eg), np.median(eu)

    # C-D signal: per-track delta spread vs reported SE
    spreads, ses = [], []
    for d in py.values():
        vals = [r["delta"] for r in d["gated_pyin"]
                if r and r["delta_identifiable"]]
        se = [np.sqrt(r["var_delta"]) for r in d["gated_pyin"]
              if r and r["delta_identifiable"] and np.isfinite(r["var_delta"])]
        if len(vals) >= 5:
            spreads.append(np.std(vals))
            ses.append(np.median(se))
    spreads, ses = np.array(spreads), np.array(ses)

    ca = ra >= 0.25 and rb >= 0.25
    cb = diffs.size > 0 and float(np.median(diffs)) <= 0.040
    cc = eg <= 1.10 * eu
    cd = spreads.size > 0 and float(np.median(spreads / np.maximum(ses, 1e-9))) > 1.0

    lines = [
        "# delta_vib channel-candidate study (URMP development side)", "",
        "Gated estimator = draft eq:vibrato exactly; criterion C-A..C-D "
        "stated in scripts/eval_delta_vib.py BEFORE the numbers were "
        "computed.", "",
        f"- C-A coverage: delta identifiable on {ra:.1%} of "
        f"vibrato-identifiable notes on GT curves ({na}/{da}), "
        f"{rb:.1%} on pyin curves ({nb}/{db}) — "
        f"{'PASS' if ca else 'FAIL'} (>= 25% both)",
        f"- C-B agreement: median |delta_pyin - delta_GT| = "
        f"{np.median(diffs) * 1000 if diffs.size else float('nan'):.0f} ms "
        f"on {diffs.size} jointly identifiable notes — "
        f"{'PASS' if cb else 'FAIL'} (<= 40 ms)",
        f"- C-C non-degradation: gated intonation median |err| {eg:.2f} c "
        f"vs ungated {eu:.2f} c vs GT quasi-truth — "
        f"{'PASS' if cc else 'FAIL'} (<= 1.10x)",
        f"- C-D signal: median per-track spread(delta)/median-SE = "
        f"{np.median(spreads / np.maximum(ses, 1e-9)) if spreads.size else float('nan'):.2f} — "
        f"{'PASS' if cd else 'FAIL'} (> 1)",
        "",
        f"**VERDICT: delta_vib {'IN' if (ca and cb and cc and cd) else 'OUT'}"
        f"** (all four must pass).",
    ]
    os.makedirs("results", exist_ok=True)
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    {"gt": stage_gt, "pyin": stage_pyin, "report": stage_report}[sys.argv[1]]()
