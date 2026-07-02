"""Vienna 4x22 Piano Corpus loader (performer-identification downstream task).

The Vienna 4x22 corpus (Goebl 1999; ``CPJKU/vienna4x22``) is 22 skilled pianists each
performing the **same four** excerpts on a Bösendorfer SE reproducing grand, with
note-level score↔performance alignments in the ``.match`` format:

    Chopin_op10_no3, Chopin_op38, Mozart_K331_1st-mov, Schubert_D783_no15  x  p01..p22

Two properties make it the right corpus for performer classification, which ASAP/MAESTRO
cannot support:

* **Real performer labels** — the ``pNN`` id is consistent across all four pieces, so a
  classifier can be trained on some pieces and tested on a held-out piece (leave-one-
  piece-out), forced to generalize a performer's *style*, not memorize a rendition.
* **No score confound** — every performer plays the identical notes, so the only signal is
  the expression ``y = [tau, log r, v]``. That is a far cleaner test of whether our
  inferred expressive variables (and the graph prior applied as a feature cleaner) carry
  performer-discriminative information than the composer/era probe on ASAP (where
  expression is confounded with the score itself).

``.match`` parsing needs the optional ``partitura`` package (``pip install partitura``,
>=1.2.0), imported lazily so the numpy-only Phase-1 core still imports without it. The
variable-extraction maths (:func:`performance_variables`) is pure numpy and tested
directly on synthetic arrays.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .features import normalize_velocity
from .score import Score

_MATCH_RE = re.compile(r"^(?P<piece>.+)_p(?P<perf>\d+)\.match$")

PIECES = ("Chopin_op10_no3", "Chopin_op38", "Mozart_K331_1st-mov", "Schubert_D783_no15")


@dataclass
class ViennaRecord:
    """One Vienna 4x22 performance (one pianist playing one excerpt)."""

    piece: str          # e.g. "Chopin_op38"
    performer: str      # e.g. "p07" (consistent across pieces)
    match_path: str     # absolute path to the .match file


def load_vienna_meta(root: str, match_subdir: str = "match") -> List[ViennaRecord]:
    """Scan ``root/match`` for ``{piece}_p{NN}.match`` files → :class:`ViennaRecord`s.

    ``root`` is a local checkout of the corpus (``git clone`` of ``CPJKU/vienna4x22`` or
    ``OFAI/vienna4x22_rematched``). Returns all performances found, sorted by (piece,
    performer). Raises ``FileNotFoundError`` if the match directory is missing.
    """
    match_dir = os.path.join(root, match_subdir)
    if not os.path.isdir(match_dir):
        raise FileNotFoundError(
            f"no match directory at {match_dir!r}; clone the corpus first "
            "(see docs/vienna_4x22_scoping.md)."
        )
    records: List[ViennaRecord] = []
    for name in sorted(os.listdir(match_dir)):
        m = _MATCH_RE.match(name)
        if not m:
            continue
        records.append(
            ViennaRecord(
                piece=m.group("piece"),
                performer=f"p{m.group('perf')}",
                match_path=os.path.join(match_dir, name),
            )
        )
    return records


def _local_tempo(
    beats: np.ndarray, perf_onset: np.ndarray, window: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-note tempo-implied onset and local seconds-per-beat via a moving linear fit.

    For each note, fit a line ``onset ≈ spb·beat + c`` to the notes within ``±window``
    beats and read the implied onset and local slope (spb) at that note.  A steady
    performance lies on a line, so every residual is zero; a single displaced note barely
    moves the local fit, so it shows a residual.  Chord-safe (no division by beat gaps).
    Falls back to a global line where a window is degenerate.
    """
    n = beats.size
    a_glob, b_glob = np.polyfit(beats, perf_onset, 1) if np.ptp(beats) > 1e-9 else (0.0, perf_onset.mean())
    implied = np.empty(n)
    spb = np.empty(n)
    for i in range(n):
        sel = np.abs(beats - beats[i]) <= window
        if sel.sum() >= 2 and np.ptp(beats[sel]) > 1e-9:
            a, b = np.polyfit(beats[sel], perf_onset[sel], 1)
        else:
            a, b = a_glob, b_glob
        implied[i] = a * beats[i] + b
        spb[i] = a
    return implied, spb


def performance_variables(
    beats: np.ndarray,
    perf_onset: np.ndarray,
    perf_duration: np.ndarray,
    dur_beats: np.ndarray,
    velocity: np.ndarray,
    tempo_window: float = 2.0,
) -> np.ndarray:
    """Phase-1 targets ``y = [tau, log r, v]`` from matched score/performance arrays.

    The tempo is a **local linear fit** of performed onset against score beat over a
    ``±tempo_window``-beat neighbourhood (:func:`_local_tempo`) — smooth enough that a
    global ritardando is absorbed as tempo, fine enough that a note played early/late
    *relative to its local pulse* shows a residual, and robust to chords (no division by
    beat gaps).  Then

        tau_i   = performed onset - implied onset                    (onset residual, s)
        log r_i = log(performed duration / (local spb · notated beats))  (articulation)
        v_i     = centred, unit-scaled MIDI velocity

    This matches the intent of :func:`features.asap_performance_variables` (there the warp
    came from an external beat grid; Vienna .match files have no separate grid, so the
    tempo is estimated from the matched notes).  All inputs are length-N arrays aligned to
    the matched score notes.  Pure numpy.
    """
    beats = np.asarray(beats, dtype=float)
    perf_onset = np.asarray(perf_onset, dtype=float)
    perf_duration = np.asarray(perf_duration, dtype=float)
    dur_beats = np.asarray(dur_beats, dtype=float)
    if beats.size < 2:
        raise ValueError("need >=2 matched notes for the tempo estimate")

    implied, spb = _local_tempo(beats, perf_onset, tempo_window)
    nominal = np.clip(spb, 1e-4, None) * np.clip(dur_beats, 1e-6, None)  # warp-implied dur (s)
    nominal = np.clip(nominal, 1e-4, None)

    tau = perf_onset - implied
    logr = np.log(np.clip(perf_duration, 1e-4, None) / nominal)
    v = normalize_velocity(velocity)
    return np.stack([tau, logr, v], axis=1)


def load_vienna_performance(match_path: str) -> Tuple[Score, np.ndarray]:
    """Load one ``.match`` file into a score support + Phase-1 targets ``(score, y)``.

    Needs ``partitura`` (imported lazily). Reads the matched score/performance note arrays
    (``partitura.load_match(..., create_score=True)``), keeps the notes present in *both*
    (``label == 'match'``), and returns the score support ``S`` (pitch, beat onset, beat
    duration) plus ``y = [tau, log r, v]`` of shape ``(n_matched, 3)`` — exactly the inputs
    the graph prior and downstream tasks consume.
    """
    try:
        import partitura as pt
    except Exception as exc:  # pragma: no cover - exercised only without partitura
        raise ImportError(
            "load_vienna_performance needs partitura (`pip install partitura`, >=1.2.0)."
        ) from exc

    performance, alignment, score = pt.load_match(match_path, create_score=True)
    snote = score.note_array()
    pnote = performance.note_array()
    idx = pt.musicanalysis.performance_codec.get_matched_notes(
        spart_note_array=snote, ppart_note_array=pnote, alignment=alignment
    )
    s = snote[idx[:, 0]]
    p = pnote[idx[:, 1]]

    beats = np.asarray(s["onset_beat"], dtype=float)
    dur_beats = np.clip(np.asarray(s["duration_beat"], dtype=float), 1e-3, None)
    pitch = np.asarray(s["pitch"], dtype=int)
    perf_onset = np.asarray(p["onset_sec"], dtype=float)
    perf_dur = np.clip(np.asarray(p["duration_sec"], dtype=float), 1e-4, None)
    vel = np.asarray(p["velocity"], dtype=float)

    order = np.argsort(beats, kind="mergesort")
    beats, dur_beats, pitch = beats[order], dur_beats[order], pitch[order]
    perf_onset, perf_dur, vel = perf_onset[order], perf_dur[order], vel[order]

    y = performance_variables(beats, perf_onset, perf_dur, dur_beats, vel)
    score_support = Score.from_arrays(pitch, beats, dur_beats)
    return score_support, y
