"""ASAP aligned score<->performance loader tests (Phase 1).

These exercise ``score_bundle.features.load_asap`` and friends on the real ASAP corpus.
They need ``pretty_midi`` and a local ASAP clone; otherwise they skip (the numpy-only
core suite still runs).  Point them at a clone with ``ASAP_ROOT`` or rely on the candidate
paths below.
"""
import os

import numpy as np
import pytest

from score_bundle import features
from score_bundle.score import Score

pretty_midi = pytest.importorskip("pretty_midi")

_CANDIDATES = [
    os.environ.get("ASAP_ROOT", ""),
    "/home/ray/Research/data/asap-dataset",
    "/home/ray/data/asap-dataset",
]


def _asap_root():
    for c in _CANDIDATES:
        if c and os.path.exists(os.path.join(c, features.ASAP_ANNOTATIONS)):
            return c
    return None


ROOT = _asap_root()
needs_asap = pytest.mark.skipif(ROOT is None, reason="no local ASAP clone found")


@needs_asap
def test_metadata_has_maestro_xref():
    meta = features.load_asap_meta(ROOT)
    assert len(meta) > 0
    # the MAESTRO cross-reference (contamination key) is present for many rows
    assert any(r.maestro_midi is not None for r in meta)
    # records point at real score/performance files (spot-check the first few)
    for r in meta[:3]:
        assert os.path.exists(os.path.join(ROOT, r.performance))
        assert os.path.exists(os.path.join(ROOT, r.midi_score))


def _first_aligned(ann):
    return next(k for k, v in ann.items() if v.get("score_and_performance_aligned"))


@needs_asap
def test_load_one_aligned_performance():
    ann = features.load_asap_annotations(ROOT)
    key = _first_aligned(ann)
    score, obs = features.load_asap(key, ROOT, annotations=ann)

    assert isinstance(score, Score) and len(score) > 0
    mask = obs["mask"]
    assert mask.shape == (len(score),)
    assert mask.mean() > 0.8  # most score notes find a performance match
    # obs arrays are aligned to the score support
    for field in ("beats", "perf_onset", "perf_duration", "velocity"):
        assert obs[field].shape == (len(score),)
    # matched entries are finite; the warp grids are strictly increasing in beat
    assert np.isfinite(obs["perf_onset"][mask]).all()
    assert np.all(np.diff(obs["beat_grid"]) > 0)


@needs_asap
def test_performance_variables_are_finite_and_centered():
    ann = features.load_asap_annotations(ROOT)
    key = _first_aligned(ann)
    score, obs = features.load_asap(key, ROOT, annotations=ann)
    score_m, y = features.asap_performance_variables(score, obs)

    assert len(score_m) == int(obs["mask"].sum())
    assert y.shape == (len(score_m), 3)
    assert np.isfinite(y).all()
    # velocity channel is mean-centered by construction
    assert abs(y[:, 2].mean()) < 1e-9
    # timing/articulation are real spreads, not degenerate
    assert y[:, 0].std() > 0 and y[:, 1].std() > 0


@needs_asap
def test_contamination_filter_drops_overlap():
    meta = features.load_asap_meta(ROOT)
    # treat every referenced MAESTRO twin as if it were in LM-train -> all xref'd perfs drop
    all_twins = [r.maestro_midi for r in meta if r.maestro_midi]
    clean = features.asap_clean_performances(meta, all_twins)
    n_xref = sum(r.maestro_midi is not None for r in meta)
    assert len(clean) == len(meta) - n_xref
    # with no train set, nothing is contaminated
    assert len(features.asap_clean_performances(meta, [])) == len(meta)
