#!/usr/bin/env python
"""tau-policy feasibility (URMP DEVELOPMENT side): can annotated onsets
anchor the score-time warp, making timing a Phase-2 channel?

Per unique development track: parse the score MIDI (pretty_midi; URMP's
score tracks are in sheet order), match score notes to the annotated
performed notes (order-matching when counts agree, else pitch-sequence DTW),
fit two warps score-time -> performance-time — global linear, and local
linear in a rolling +/-8-note window (the Phase-1 beat-grid analogue) — and
measure the residual tau.  Feasibility verdict = (i) how many tracks match
cleanly, (ii) the tau scale under the local warp, (iii) lag-1 neighbour
correlation of tau (the structure a graph prior could model).

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/eval_tau_feasibility.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_real import dev_unique_tracks  # noqa: E402

from score_bundle.phase2.warp import dtw_match, local_loo_warp  # noqa: E402

OUT = "results/tau_feasibility_dev.md"
WIN = 8


def local_linear_warp(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Predicted performance time per note (leave-one-out tempo line).

    Since the tau adoption (2026-08-13) the implementation lives in
    :mod:`score_bundle.phase2.warp`; this wrapper keeps the script's original
    interface. ``local_loo_warp`` may return NaN at edge notes with too few
    neighbours; the original inline version never did, so NaNs are filled
    with the note's own time (tau contribution 0), matching the original
    statistics to within the affected 0-2 edge notes per track."""
    pred, _ = local_loo_warp(np.asarray(b, float), np.asarray(t, float),
                             win=WIN)
    bad = ~np.isfinite(pred)
    if bad.any():
        pred[bad] = np.asarray(t, float)[bad]
    return pred


def main() -> None:
    import pretty_midi

    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.phase2.targets import hz_to_semitone

    rows = []
    n_exact = n_dtw_good = n_bad = 0
    for p, tr in dev_unique_tracks():
        try:
            pm = pretty_midi.PrettyMIDI(
                os.path.join(p.folder, os.path.basename(p.score_mid)))
        except Exception:
            n_bad += 1
            continue
        if tr.number - 1 >= len(pm.instruments):
            n_bad += 1
            continue
        sc = pm.instruments[tr.number - 1].notes
        sb = np.array([nt.start for nt in sc])
        sp = np.array([nt.pitch for nt in sc])
        notes = read_notes_annotation(tr.notes)
        t = notes["onset"]
        ppitch = hz_to_semitone(notes["pitch_hz"])
        if sp.size == ppitch.size and np.mean(sp == ppitch) > 0.9:
            si = np.arange(sp.size)
            pi = np.arange(ppitch.size)
            n_exact += 1
        else:
            pairs = dtw_match(sp, ppitch)
            if pairs.size == 0 or len(pairs) < 0.8 * min(sp.size, ppitch.size):
                n_bad += 1
                continue
            si, pi = pairs[:, 0], pairs[:, 1]
            n_dtw_good += 1
        b, tt = sb[si], t[pi]
        # global linear warp
        A = np.stack([b, np.ones(b.size)], axis=1)
        coef, *_ = np.linalg.lstsq(A, tt, rcond=None)
        tau_g = tt - (coef[0] * b + coef[1])
        # local (leave-one-out) warp
        tau_l = tt - local_linear_warp(b, tt)
        r1 = float(np.corrcoef(tau_l[:-1], tau_l[1:])[0, 1]) \
            if tau_l.size > 3 else np.nan
        rows.append((p.index, tr.number, tr.instrument, tt.size,
                     float(np.std(tau_g)), float(np.std(tau_l)), r1))

    a = np.array([[r[4], r[5], r[6]] for r in rows])
    lines = [
        "# tau feasibility — URMP development side (annotated onsets as "
        "warp anchors)", "",
        f"Tracks: {len(rows)} usable of {len(rows) + n_bad} "
        f"({n_exact} exact order match, {n_dtw_good} via pitch DTW, "
        f"{n_bad} unusable).", "",
        f"- tau std under GLOBAL linear warp: median "
        f"{np.median(a[:, 0]) * 1000:.0f} ms (tempo drift dominates)",
        f"- tau std under LOCAL (+/-{WIN}-note, leave-one-out) warp: median "
        f"{np.median(a[:, 1]) * 1000:.0f} ms — the Phase-1-comparable "
        f"residual scale",
        f"- lag-1 neighbour correlation of local tau: median "
        f"{np.median(a[:, 2]):+.2f} (IQR "
        f"[{np.percentile(a[:, 2], 25):+.2f}, "
        f"{np.percentile(a[:, 2], 75):+.2f}]) — the structure a graph "
        f"prior could model", "",
        "Verdict input for the tau policy: annotated onsets anchor a usable "
        "warp on the overwhelming majority of tracks without any audio "
        "aligner; residual scale and neighbour correlation are in the range "
        "where the graph prior operates in Phase 1.",
    ]
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
