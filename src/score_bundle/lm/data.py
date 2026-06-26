"""Corpus generation and next-token batching for Phase-0 pretraining.

Framework-agnostic (NumPy): the same token streams feed either the NumPy or the
PyTorch model.  ``random_corpus`` provides a tiny synthetic corpus so the examples
and tests run without any external MIDI data.  For real pretraining use the MAESTRO
loaders below (``load_maestro_meta`` / ``maestro_note_events`` /
``iter_maestro_note_streams``), which turn MAESTRO MIDI into ``NoteEvent`` streams
compatible with :class:`MidiTokenizer`.

``pretty_midi`` is an optional dependency (the ``train`` extra); the MAESTRO loaders
import it lazily so this module still imports under the numpy-only core.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .tokenizer import MidiTokenizer, NoteEvent


def random_sequence(rng: np.random.Generator, n_notes: int = 40) -> List[NoteEvent]:
    """A plausible-ish monophonic piano line: pitch random walk, increasing onsets."""
    pitch = 60 + np.cumsum(rng.integers(-3, 4, size=n_notes))
    pitch = np.clip(pitch, 36, 90)
    gaps = rng.choice([0.25, 0.5, 0.5, 1.0], size=n_notes)
    onset = np.concatenate([[0.0], np.cumsum(gaps)[:-1]])
    dur = gaps * rng.uniform(0.6, 1.0, size=n_notes)
    vel = rng.integers(40, 100, size=n_notes)
    return [NoteEvent(int(pitch[i]), float(onset[i]), float(dur[i]), int(vel[i])) for i in range(n_notes)]


def random_corpus(rng: np.random.Generator, n_seqs: int = 64, n_notes: int = 40) -> List[List[NoteEvent]]:
    return [random_sequence(rng, n_notes) for _ in range(n_seqs)]


# --- MAESTRO ingestion ---------------------------------------------------
#
# MAESTRO ships a metadata table (``maestro-v3.0.0.json``) and per-piece MIDI under
# ``<year>/<file>.midi``.  We read note events with ``pretty_midi`` and express onsets
# and durations in **beats** (TIME_SHIFT/DURATION are quantized to beats/grid by the
# tokenizer), recovering the beat grid from the MIDI's own tempo map via
# ``PrettyMIDI.get_beats()``.  The dataset's own ``split`` column gives a leakage-safe
# train/val/test partition (MAESTRO guarantees a composition appears in only one split).

MAESTRO_JSON = "maestro-v3.0.0.json"


@dataclass
class MaestroRecord:
    """One MAESTRO performance (a row of the metadata table)."""

    midi_path: str       # absolute path to the .midi file
    split: str           # 'train' | 'validation' | 'test'
    composer: str
    title: str
    year: int
    duration: float      # seconds


def load_maestro_meta(
    root: str, split: Optional[str] = None, json_name: str = MAESTRO_JSON
) -> List[MaestroRecord]:
    """Parse MAESTRO's metadata table into :class:`MaestroRecord`s.

    ``root`` is the MAESTRO release directory (the one containing
    ``maestro-v3.0.0.json`` and the per-year MIDI folders).  ``split`` optionally
    filters to one of ``'train' | 'validation' | 'test'``.
    """
    meta_path = os.path.join(root, json_name)
    with open(meta_path) as fh:
        cols = json.load(fh)  # dict-of-columns, each keyed by stringified row index
    keys = list(cols["midi_filename"].keys())
    records: List[MaestroRecord] = []
    for k in keys:
        rec = MaestroRecord(
            midi_path=os.path.join(root, cols["midi_filename"][k]),
            split=cols["split"][k],
            composer=cols["canonical_composer"][k],
            title=cols["canonical_title"][k],
            year=int(cols["year"][k]),
            duration=float(cols["duration"][k]),
        )
        if split is None or rec.split == split:
            records.append(rec)
    return records


def maestro_split(records: Sequence[MaestroRecord], strict_dedup: bool = False) -> dict:
    """Group records by MAESTRO's official train/validation/test split.

    The official split is composition-safe by construction (the dataset authors keep a
    composition within a single split).  A handful of generic/program ``canonical_title``
    strings nonetheless collide across splits; with ``strict_dedup=True`` we additionally
    drop any *eval* piece whose title also appears in ``train``, giving a conservative,
    leakage-free held-out set (the contamination guard CLAUDE.md asks for).  The dropped
    titles are reported via :func:`title_overlaps`.
    """
    groups: dict = {"train": [], "validation": [], "test": []}
    for rec in records:
        groups.setdefault(rec.split, []).append(rec)
    if strict_dedup:
        train_titles = {r.title for r in groups.get("train", [])}
        for name in ("validation", "test"):
            groups[name] = [r for r in groups.get(name, []) if r.title not in train_titles]
    return groups


def title_overlaps(records: Sequence[MaestroRecord]) -> dict:
    """Titles that appear in more than one split, mapped to the splits they span.

    A diagnostic for contamination: an empty result means the title-level partition is
    clean; otherwise these are the pieces ``strict_dedup`` removes from eval.
    """
    spans: dict = {}
    for rec in records:
        spans.setdefault(rec.title, set()).add(rec.split)
    return {t: sorted(s) for t, s in spans.items() if len(s) > 1}


def _seconds_to_beats(times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Map performance seconds to beat positions using the MIDI beat grid.

    ``beats[i]`` is the time (s) of beat ``i``; the beat number is the index.  Linear
    within the grid, linearly extrapolated past either end using the local beat period
    (so notes before the first / after the last detected beat keep a sane spacing).
    """
    times = np.asarray(times, dtype=float)
    idx = np.arange(len(beats), dtype=float)
    if len(beats) < 2:
        # degenerate tempo map: fall back to a default 120 BPM (0.5 s/beat)
        t0 = beats[0] if len(beats) else 0.0
        return (times - t0) / 0.5
    p_lo = beats[1] - beats[0]
    p_hi = beats[-1] - beats[-2]
    # sentinel anchors far outside the grid preserve the end slopes under np.interp
    xp = np.concatenate(([beats[0] - 1e6 * p_lo], beats, [beats[-1] + 1e6 * p_hi]))
    fp = np.concatenate(([idx[0] - 1e6], idx, [idx[-1] + 1e6]))
    return np.interp(times, xp, fp)


def maestro_note_events(
    midi_path: str, time_unit: str = "beats", max_notes: Optional[int] = None
) -> List[NoteEvent]:
    """Read a MAESTRO MIDI into onset-ordered :class:`NoteEvent`s.

    ``time_unit='beats'`` (default) expresses onset/duration in beats via the MIDI's
    tempo map — the unit the tokenizer quantizes.  ``time_unit='seconds'`` keeps raw
    performance seconds.  All instruments are merged (MAESTRO is solo piano, one track,
    but we merge defensively).  Velocities stay MIDI 0..127.
    """
    if time_unit not in ("beats", "seconds"):
        raise ValueError("time_unit must be 'beats' or 'seconds'")
    try:
        import pretty_midi
    except Exception as exc:  # pragma: no cover - exercised only without pretty_midi
        raise ImportError(
            "maestro_note_events needs pretty_midi (install the train extra: "
            "`pip install -e '.[train]'`)."
        ) from exc

    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [n for ins in pm.instruments if not ins.is_drum for n in ins.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))
    if max_notes is not None:
        notes = notes[:max_notes]
    if not notes:
        return []

    if time_unit == "seconds":
        onsets = np.array([n.start for n in notes])
        ends = np.array([n.end for n in notes])
    else:
        beats = np.asarray(pm.get_beats(), dtype=float)
        starts = np.array([n.start for n in notes])
        ends_s = np.array([n.end for n in notes])
        onsets = _seconds_to_beats(starts, beats)
        ends = _seconds_to_beats(ends_s, beats)
    durations = np.clip(ends - onsets, 1e-4, None)
    return [
        NoteEvent(int(n.pitch), float(onsets[i]), float(durations[i]), int(n.velocity))
        for i, n in enumerate(notes)
    ]


def iter_maestro_note_streams(
    root: str,
    split: str = "train",
    limit: Optional[int] = None,
    time_unit: str = "beats",
    max_notes: Optional[int] = None,
) -> Iterator[List[NoteEvent]]:
    """Yield one :class:`NoteEvent` stream per MAESTRO piece in ``split`` (in order).

    ``limit`` caps the number of pieces (handy for quick runs); ``max_notes`` caps notes
    per piece.  Pieces that fail to parse are skipped with a warning rather than aborting
    a long pretraining ingest.
    """
    records = load_maestro_meta(root, split=split)
    if limit is not None:
        records = records[:limit]
    for rec in records:
        try:
            notes = maestro_note_events(rec.midi_path, time_unit=time_unit, max_notes=max_notes)
        except Exception as exc:  # pragma: no cover - corrupt-file guard
            print(f"[maestro] skipping {rec.midi_path}: {exc}")
            continue
        if notes:
            yield notes


def encode_corpus(tokenizer: MidiTokenizer, corpus: List[List[NoteEvent]]) -> List[List[int]]:
    return [tokenizer.encode(seq) for seq in corpus]


def pack_tokens(token_seqs: List[List[int]]) -> np.ndarray:
    """Concatenate sequences into a single 1-D stream for windowed LM training."""
    return np.concatenate([np.asarray(s, dtype=np.int64) for s in token_seqs])


def lm_batches(
    stream: np.ndarray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
    n_batches: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (x, y) int arrays of shape (batch, block); y is x shifted by one."""
    hi = len(stream) - block_size - 1
    if hi <= 0:
        raise ValueError("stream shorter than block_size + 1")
    for _ in range(n_batches):
        ix = rng.integers(0, hi, size=batch_size)
        x = np.stack([stream[i : i + block_size] for i in ix])
        y = np.stack([stream[i + 1 : i + 1 + block_size] for i in ix])
        yield x, y
