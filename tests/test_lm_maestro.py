"""MAESTRO ingestion tests (Phase 0).

These exercise the real-MIDI loaders in ``score_bundle.lm.data``.  They need both
``pretty_midi`` and a local MAESTRO release; when either is missing the tests skip
(so the numpy-only suite still runs).  Point them at a release with the env var
``MAESTRO_ROOT`` (the directory holding ``maestro-v3.0.0.json``), or rely on the
known candidate paths below.
"""
import os

import numpy as np
import pytest

from score_bundle.lm import data
from score_bundle.lm.tokenizer import MidiTokenizer

pretty_midi = pytest.importorskip("pretty_midi")

_CANDIDATES = [
    os.environ.get("MAESTRO_ROOT", ""),
    "/home/ray/Research/data/maestro-v3.0.0",
    "/home/ray/data/maestro-v3.0.0",
]


def _maestro_root():
    for c in _CANDIDATES:
        if c and os.path.exists(os.path.join(c, data.MAESTRO_JSON)):
            return c
    return None


ROOT = _maestro_root()
needs_maestro = pytest.mark.skipif(ROOT is None, reason="no local MAESTRO release found")


@needs_maestro
def test_meta_and_split_are_leakage_safe():
    recs = data.load_maestro_meta(ROOT)
    assert len(recs) > 0
    groups = data.maestro_split(recs)
    assert set(groups) >= {"train", "validation", "test"}
    # every record points at a real file
    assert all(os.path.exists(r.midi_path) for r in recs[:5])
    # split filter agrees with the grouping
    train = data.load_maestro_meta(ROOT, split="train")
    assert len(train) == len(groups["train"])

    # strict dedup removes exactly the title-colliding eval pieces, leaving a clean set
    strict = data.maestro_split(recs, strict_dedup=True)
    train_titles = {r.title for r in strict["train"]}
    for name in ("validation", "test"):
        assert len(strict[name]) <= len(groups[name])
        assert all(r.title not in train_titles for r in strict[name])
    # after dedup, no title spans train<->val or train<->test
    spans = data.title_overlaps([*strict["train"], *strict["validation"], *strict["test"]])
    assert all("train" not in v for v in spans.values())


@needs_maestro
def test_note_events_are_well_formed():
    rec = data.load_maestro_meta(ROOT, split="train")[0]
    notes = data.maestro_note_events(rec.midi_path, max_notes=300)
    assert len(notes) > 0
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets)  # onset-ordered
    assert all(n.duration > 0 for n in notes)
    assert all(21 <= n.pitch <= 108 for n in notes)
    assert all(0 <= n.velocity <= 127 for n in notes)


@needs_maestro
def test_real_file_round_trips_within_quantization_tolerance():
    """A real MAESTRO piece survives encode->decode within one grid step.

    Quantization is per-note rounding of absolute onset and duration to the beat grid,
    so the round-trip error is bounded by 1/grid with no accumulation — as long as no
    inter-onset gap or duration is clipped.  We use a generous tokenizer so the first
    few hundred notes of a real performance are not clipped, then assert the bound.
    """
    rec = data.load_maestro_meta(ROOT, split="train")[0]
    notes = data.maestro_note_events(rec.midi_path, max_notes=300)

    grid = 24
    big = grid * 64  # 64 beats: comfortably past any normal gap/duration
    tok = MidiTokenizer(grid=grid, max_shift_steps=big, max_dur_steps=big)

    decoded = tok.decode(tok.encode(notes))
    assert len(decoded) == len(notes)

    tol = 1.0 / grid + 1e-9
    for a, b in zip(notes, decoded):
        assert a.pitch == b.pitch
        assert abs(a.onset - b.onset) <= tol
        assert abs(a.duration - b.duration) <= tol
        # velocity is bucketed (n_vel_bins) — recovered to within one bin width
        assert abs(a.velocity - b.velocity) <= 128 / tok.n_vel_bins
