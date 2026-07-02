"""Tests for the Vienna 4x22 loader + performer-classification wiring.

Exercise the pure-numpy variable maths, filename/meta parsing, and the grouped
(leave-one-piece-out) classifier on synthetic fixtures — no partitura or real corpus
needed (the ``.match`` parsing path is import-guarded and covered only when the corpus is
present).
"""
import os

import numpy as np
import pytest

from score_bundle.downstream import grouped_nearest_centroid
from score_bundle.vienna import (
    ViennaRecord,
    load_vienna_meta,
    performance_variables,
)


def test_meta_parses_piece_and_performer(tmp_path):
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    for piece in ("Chopin_op38", "Mozart_K331_1st-mov"):
        for perf in (1, 7, 22):
            (match_dir / f"{piece}_p{perf:02d}.match").write_text("")
    (match_dir / "README.txt").write_text("ignore me")  # non-match file ignored
    recs = load_vienna_meta(str(tmp_path))
    assert len(recs) == 6
    assert all(isinstance(r, ViennaRecord) for r in recs)
    assert {r.piece for r in recs} == {"Chopin_op38", "Mozart_K331_1st-mov"}
    assert {r.performer for r in recs} == {"p01", "p07", "p22"}
    # performer ids are consistent across pieces (the property the eval relies on)
    by_piece = {}
    for r in recs:
        by_piece.setdefault(r.piece, set()).add(r.performer)
    assert by_piece["Chopin_op38"] == by_piece["Mozart_K331_1st-mov"]


def test_meta_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vienna_meta(str(tmp_path))


def _dense_beats(n_beats=6, per_beat=3):
    """Score beats with ``per_beat`` notes on each integer beat (sub-beat positions)."""
    sub = np.arange(per_beat) / per_beat
    return np.repeat(np.arange(float(n_beats)), per_beat) + np.tile(sub, n_beats)


def test_performance_variables_recovers_tempo_and_units():
    # steady 0.5 s/beat, no expression: tau ~ 0, log r ~ 0 (interior), v centred.
    beats = _dense_beats()
    spb = 0.5
    perf_onset = beats * spb
    dur_beats = np.full(beats.size, 1.0 / 3)     # each note a third of a beat
    perf_duration = dur_beats * spb
    velocity = np.full(beats.size, 64.0)
    y = performance_variables(beats, perf_onset, perf_duration, dur_beats, velocity)
    assert y.shape == (beats.size, 3)
    np.testing.assert_allclose(y[:, 0], 0.0, atol=1e-9)          # tau ~ 0
    np.testing.assert_allclose(y[3:-3, 1], 0.0, atol=1e-6)       # log r ~ 0 (interior)
    np.testing.assert_allclose(y[:, 2], 0.0, atol=1e-9)          # v centred


def test_performance_variables_detects_rubato_and_articulation():
    beats = _dense_beats()
    perf_onset = beats * 0.5
    dur_beats = np.full(beats.size, 1.0 / 3)
    perf_duration = dur_beats * 0.5
    velocity = np.linspace(40, 100, beats.size)

    y0 = performance_variables(beats, perf_onset, perf_duration, dur_beats, velocity)
    # delay one interior off-beat note; its beat's median pulse (3 notes) barely moves,
    # so the note shows a positive onset residual.
    i = 7
    perf_onset2 = perf_onset.copy(); perf_onset2[i] += 0.05
    y1 = performance_variables(beats, perf_onset2, perf_duration, dur_beats, velocity)
    assert y1[i, 0] > y0[i, 0] + 0.02             # late note -> larger tau

    # hold one note much longer -> positive log-articulation
    perf_dur2 = perf_duration.copy(); perf_dur2[i] *= 3.0
    y2 = performance_variables(beats, perf_onset, perf_dur2, dur_beats, velocity)
    assert y2[i, 1] > y0[i, 1] + 0.3
    assert y0[-1, 2] > y0[0, 2]                    # louder notes -> larger v


def test_performance_variables_chord_safe():
    # two notes share beat 2 (a chord) -> median warp must not divide by zero
    beats = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    perf_onset = beats * 0.5
    dur_beats = np.ones(6)
    perf_duration = dur_beats * 0.5
    velocity = np.full(6, 70.0)
    y = performance_variables(beats, perf_onset, perf_duration, dur_beats, velocity)
    assert np.all(np.isfinite(y))


def test_grouped_nearest_centroid_leave_piece_out():
    # 3 performers x 3 pieces; each performer has a distinct style offset that is
    # consistent across pieces, plus a per-piece shift shared by everyone (the confound
    # that leave-one-piece-out must survive).
    rng = np.random.default_rng(0)
    style = {"A": np.array([0.0, 0.0]), "B": np.array([3.0, 0.0]), "C": np.array([0.0, 3.0])}
    piece_shift = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 5.0]), 2: np.array([-4.0, 2.0])}
    X, perf, piece = [], [], []
    for pf in style:
        for pc in piece_shift:
            X.append(style[pf] + piece_shift[pc] + rng.normal(scale=0.05, size=2))
            perf.append(pf); piece.append(pc)
    acc, n = grouped_nearest_centroid(np.array(X), perf, piece)
    assert n == 9
    assert acc > 0.8   # performer style recoverable despite per-piece shifts
