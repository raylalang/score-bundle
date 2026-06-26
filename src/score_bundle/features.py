"""Phase-1 target extraction from aligned score-performance data.

These functions turn an alignment (score beats <-> performed seconds) into the
Phase-1 performance variables.  The tempo curve is estimated *here* (feature
extraction), so that, conditional on it, the onset residual is well defined and
the note-level field is jointly Gaussian (concept note, section 8.2).

Dataset loaders (ASAP / MAESTRO / ATEPP) are intentionally left as stubs: drop in
your local parsing and return arrays in these units.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .score import Score


def estimate_tempo_curve(beats: np.ndarray, perf_onset: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    """Local seconds-per-beat at each note from an alignment.

    Estimated by finite differences of performed time vs score beats, then
    (optionally) smoothed.  Returns one value per note (length N).
    """
    beats = np.asarray(beats, dtype=float)
    perf_onset = np.asarray(perf_onset, dtype=float)
    order = np.argsort(beats)
    b, t = beats[order], perf_onset[order]
    spb = np.gradient(t, b)  # d(seconds)/d(beat)
    spb = np.clip(spb, 1e-3, None)
    if smooth > 0:
        spb = _moving_average(spb, smooth)
    out = np.empty_like(spb)
    out[order] = spb
    return out


def tempo_implied_onset(beats: np.ndarray, sec_per_beat: np.ndarray) -> np.ndarray:
    """Cumulative time warp T(b) = integral of seconds-per-beat over score time."""
    beats = np.asarray(beats, dtype=float)
    order = np.argsort(beats)
    b = beats[order]
    spb = np.asarray(sec_per_beat, dtype=float)[order]
    db = np.diff(b, prepend=b[0])
    t = np.cumsum(spb * db)
    t = t - t[0]
    out = np.empty_like(t)
    out[order] = t
    return out


def onset_residual(beats, perf_onset, sec_per_beat) -> np.ndarray:
    """tau_i = performed onset - tempo-implied onset (seconds)."""
    implied = tempo_implied_onset(beats, sec_per_beat)
    implied = implied - implied.mean() + np.asarray(perf_onset, float).mean()
    return np.asarray(perf_onset, dtype=float) - implied


def articulation_ratio(score: Score, perf_duration, sec_per_beat) -> np.ndarray:
    """log r_i = log(performed duration / nominal duration)."""
    nominal = score.duration * np.asarray(sec_per_beat, dtype=float)
    nominal = np.clip(nominal, 1e-6, None)
    return np.log(np.clip(np.asarray(perf_duration, float), 1e-6, None) / nominal)


def normalize_velocity(midi_velocity, lo: float = 0.0, hi: float = 127.0) -> np.ndarray:
    """Map MIDI velocity to a centered, roughly unit-scale dynamics variable."""
    v = (np.asarray(midi_velocity, dtype=float) - lo) / (hi - lo)
    return (v - v.mean())


def _moving_average(x: np.ndarray, window: float) -> np.ndarray:
    k = max(1, int(window))
    if k == 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


# --- ASAP loader (aligned score <-> performance) -------------------------
#
# ASAP is the only corpus with aligned symbolic score and performance, so it carries the
# Phase-1 targets y = [tau, log r, v].  ``asap_annotations.json`` gives, per performance,
# two index-aligned beat grids in seconds: ``midi_score_beats`` (in the quantized score
# MIDI's clock) and ``performance_beats`` (in the performance's clock).  That pair is the
# time warp linking the two.  We:
#   1. read the score MIDI (the support S: pitch, beat onset, beat duration),
#   2. read the performance MIDI (onset/duration in seconds, velocity),
#   3. warp each score note to its predicted performance time and match it to the
#      nearest same-pitch performance note (greedy, onset order),
# yielding per-score-note aligned arrays the rest of this module turns into y.
#
# ``pretty_midi`` is an optional dependency; it is imported lazily so the numpy-only
# Phase-1 core still imports without it.

ASAP_ANNOTATIONS = "asap_annotations.json"
ASAP_METADATA = "metadata.csv"


@dataclass
class AsapRecord:
    """One ASAP performance row (from ``metadata.csv``)."""

    performance: str            # rel path of the performance MIDI (== annotations key)
    folder: str                 # piece folder (rel)
    composer: str
    title: str
    midi_score: str             # rel path of the quantized score MIDI
    maestro_midi: Optional[str] # MAESTRO counterpart rel path, or None (contamination xref)


def load_asap_meta(root: str, metadata_name: str = ASAP_METADATA) -> List[AsapRecord]:
    """Parse ASAP's ``metadata.csv`` into :class:`AsapRecord`s.

    The ``maestro_midi_performance`` column (when present) cross-references the exact
    MAESTRO file a performance was drawn from — the hook for the train/eval contamination
    guard (a Phase-1 eval performance whose MAESTRO twin was in Phase-0 pretraining is not
    a clean recovery test).
    """
    records: List[AsapRecord] = []
    with open(os.path.join(root, metadata_name), newline="") as fh:
        for row in csv.DictReader(fh):
            maestro = (row.get("maestro_midi_performance") or "").strip()
            maestro_rel = maestro.replace("{maestro}/", "") if maestro else None
            records.append(
                AsapRecord(
                    performance=row["midi_performance"],
                    folder=row["folder"],
                    composer=row["composer"],
                    title=row["title"],
                    midi_score=row["midi_score"],
                    maestro_midi=maestro_rel,
                )
            )
    return records


def load_asap_annotations(root: str, annotations_name: str = ASAP_ANNOTATIONS) -> dict:
    """Load the ASAP annotations JSON (performance key -> beat grids / metadata)."""
    with open(os.path.join(root, annotations_name)) as fh:
        return json.load(fh)


def _midi_notes_sorted(pm) -> list:
    notes = [n for ins in pm.instruments if not ins.is_drum for n in ins.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _match_score_to_performance(
    score_notes: list, perf_notes: list, s2p, tol_sec: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy same-pitch nearest-time matching of score notes to performance notes.

    For each score note (onset order) we predict its performance onset via the warp ``s2p``
    and claim the closest unused performance note of identical pitch within ``tol_sec``.
    Returns (perf_onset, perf_duration, velocity, matched) aligned to ``score_notes``;
    unmatched entries are NaN / False.
    """
    from collections import defaultdict

    by_pitch: Dict[int, list] = defaultdict(list)
    for j, n in enumerate(perf_notes):
        by_pitch[n.pitch].append(j)
    used = [False] * len(perf_notes)

    n = len(score_notes)
    perf_onset = np.full(n, np.nan)
    perf_dur = np.full(n, np.nan)
    vel = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)

    for i, sn in enumerate(score_notes):
        pred = float(s2p(sn.start))
        best_j, best_d = -1, np.inf
        for j in by_pitch.get(sn.pitch, ()):
            if used[j]:
                continue
            d = abs(perf_notes[j].start - pred)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= tol_sec:
            used[best_j] = True
            pn = perf_notes[best_j]
            perf_onset[i] = pn.start
            perf_dur[i] = max(pn.end - pn.start, 1e-4)
            vel[i] = pn.velocity
            matched[i] = True
    return perf_onset, perf_dur, vel, matched


def load_asap(
    performance: str,
    root: str,
    annotations: Optional[dict] = None,
    match_tol_beats: float = 2.0,
) -> Tuple[Score, dict]:
    """Load one aligned ASAP performance into a score support + observation dict.

    Parameters
    ----------
    performance: relative path of the performance MIDI (the annotations key), e.g.
        ``"Bach/Fugue/bwv_846/Shi05M.mid"``.
    root: the ASAP dataset directory.
    annotations: optional pre-loaded annotations dict (avoids re-reading the JSON in loops).
    match_tol_beats: max score-to-performance onset mismatch (in beats) to accept a match.

    Returns ``(score, obs)`` where ``score`` is the support S (all score notes) and ``obs``
    has arrays aligned to it: ``beats`` (score-beat onset), ``perf_onset`` /
    ``perf_duration`` (performance seconds, NaN if unmatched), ``velocity`` (MIDI, NaN if
    unmatched), and ``mask`` (bool, True where matched).  Feed ``obs`` to
    :func:`asap_performance_variables` for y = [tau, log r, v].
    """
    try:
        import pretty_midi
    except Exception as exc:  # pragma: no cover - exercised only without pretty_midi
        raise ImportError(
            "load_asap needs pretty_midi (install the train extra: `pip install -e '.[train]'`)."
        ) from exc

    ann = annotations if annotations is not None else load_asap_annotations(root)
    if performance not in ann:
        raise KeyError(f"{performance!r} not in ASAP annotations")
    entry = ann[performance]
    if not entry.get("score_and_performance_aligned", False):
        raise ValueError(f"{performance!r} is not score/performance aligned")

    sb = np.asarray(entry["midi_score_beats"], dtype=float)   # score-clock seconds
    pb = np.asarray(entry["performance_beats"], dtype=float)  # performance-clock seconds

    score_path = os.path.join(root, os.path.dirname(performance), "midi_score.mid")
    perf_path = os.path.join(root, performance)
    score_pm = pretty_midi.PrettyMIDI(score_path)
    perf_pm = pretty_midi.PrettyMIDI(perf_path)
    score_notes = _midi_notes_sorted(score_pm)
    perf_notes = _midi_notes_sorted(perf_pm)
    if not score_notes:
        raise ValueError(f"no score notes in {score_path}")

    # score-seconds -> score-beats via the score MIDI's own beat grid (handles tempo maps)
    grid = np.asarray(score_pm.get_beats(), dtype=float)
    beat_idx = np.arange(len(grid), dtype=float)
    starts = np.array([n.start for n in score_notes])
    ends = np.array([n.end for n in score_notes])
    b_onset = np.interp(starts, grid, beat_idx)
    b_dur = np.clip(np.interp(ends, grid, beat_idx) - b_onset, 1e-3, None)

    # warp score-seconds -> performance-seconds (the alignment), and a seconds tolerance
    s2p = lambda t: np.interp(t, sb, pb)  # noqa: E731 - tiny local warp
    perf_beat_period = float(np.median(np.diff(pb))) if len(pb) > 1 else 0.5
    tol_sec = match_tol_beats * perf_beat_period

    perf_onset, perf_dur, vel, matched = _match_score_to_performance(
        score_notes, perf_notes, s2p, tol_sec
    )

    # tempo warp from the beat annotations: score-beat number -> performance seconds.
    # Strictly increasing in beat, so it is robust to chords (notes sharing a beat) — unlike
    # per-note finite differences (estimate_tempo_curve), which divide by zero on duplicates.
    beat_grid = np.interp(sb, grid, beat_idx)

    score = Score.from_arrays(
        [n.pitch for n in score_notes], list(b_onset), list(b_dur)
    )
    obs = {
        "beats": b_onset,
        "perf_onset": perf_onset,
        "perf_duration": perf_dur,
        "velocity": vel,
        "mask": matched,
        "beat_grid": beat_grid,   # score-beat number of each annotated beat
        "perf_grid": pb,          # performance seconds of each annotated beat
    }
    return score, obs


def asap_performance_variables(
    score: Score, obs: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn an ASAP ``(score, obs)`` into Phase-1 targets y = [tau, log r, v].

    The tempo-implied onset and nominal (notated) duration come from the ASAP beat-grid
    warp (``beat_grid`` -> ``perf_grid``), so:

        tau_i   = performed onset - tempo-implied onset            (onset residual, seconds)
        log r_i = log(performed duration / warp-implied duration)  (articulation)
        v_i     = centered MIDI velocity                            (dynamics)

    consistent with :func:`onset_residual` / :func:`articulation_ratio` but computed from the
    strictly-increasing global warp (chord-safe).  Restricts to matched notes; returns
    ``(score_matched, y)`` with ``y`` of shape ``(n_matched, 3)``.
    """
    mask = np.asarray(obs["mask"], dtype=bool)
    if mask.sum() < 2:
        raise ValueError("need >=2 matched notes for the tempo warp")

    beats = np.asarray(obs["beats"], dtype=float)[mask]
    perf_onset = np.asarray(obs["perf_onset"], dtype=float)[mask]
    perf_dur = np.asarray(obs["perf_duration"], dtype=float)[mask]
    vel = np.asarray(obs["velocity"], dtype=float)[mask]
    bg = np.asarray(obs["beat_grid"], dtype=float)
    pg = np.asarray(obs["perf_grid"], dtype=float)

    matched_notes = [n for n, m in zip(score.notes, mask) if m]
    score_m = Score(matched_notes)
    d_beats = score_m.duration

    implied = np.interp(beats, bg, pg)                       # tempo-implied onset (s)
    nominal = np.interp(beats + d_beats, bg, pg) - implied   # warp-implied duration (s)
    nominal = np.clip(nominal, 1e-4, None)

    tau = perf_onset - implied
    logr = np.log(np.clip(perf_dur, 1e-4, None) / nominal)
    v = normalize_velocity(vel)
    y = np.stack([tau, logr, v], axis=1)
    return score_m, y


def asap_clean_performances(
    records: Sequence[AsapRecord], maestro_train_rel: Sequence[str]
) -> List[AsapRecord]:
    """Filter ASAP records to those whose MAESTRO twin is NOT in the Phase-0 train set.

    ``maestro_train_rel`` is the set of MAESTRO relative MIDI paths used for LM pretraining.
    A performance with no MAESTRO counterpart is kept (it cannot be contaminated via the
    MAESTRO overlap).  This is the train/eval contamination guard for Phase-1 evaluation.
    """
    train = set(maestro_train_rel)
    return [r for r in records if (r.maestro_midi is None) or (r.maestro_midi not in train)]
