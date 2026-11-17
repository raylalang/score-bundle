"""Pins for the motif-study pilot machinery (scripts/eval_motif_pilot_dev.py):
the per-channel block factorizes the motif graph kernel exactly, arm names
parse, and predicted occurrences map to the excerpt by (onset, pitch)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
mp = pytest.importorskip("eval_motif_pilot_dev")


def _toy_occs():
    q = np.arange(12, dtype=float)
    occs = [{"k": 0, "type": "a", "notes": np.array([0, 1, 2]), "u": np.array([0.0, 0.5, 1.0]),
             "start": 0.0, "end": 2.0},
            {"k": 1, "type": "a", "notes": np.array([6, 7, 8]), "u": np.array([0.0, 0.5, 1.0]),
             "start": 6.0, "end": 8.0}]
    return q, occs


def test_motif_feature_block_factorizes_the_graph_kernel():
    from score_bundle.graph import laplacian
    q, occs = _toy_occs()
    n = len(q)
    X = mp.motif_feature_block(n, occs, s_m=1.0)
    W = mp.motif_adjacency(n, occs)
    on = np.where(W.sum(1) > 0)[0]
    nu, U = np.linalg.eigh(laplacian(W[np.ix_(on, on)]))
    K = (U * (1.0 / (1.0 + np.clip(nu, 0, None)))) @ U.T
    assert X.shape == (n, len(on))
    assert np.allclose(X[on] @ X[on].T, K, atol=1e-10)
    off = np.setdiff1d(np.arange(n), on)
    assert np.all(X[off] == 0)
    assert np.allclose(np.diag(X[on] @ X[on].T)[0], K[0, 0])
    # no motif notes at all -> an empty block, no crash
    assert mp.motif_feature_block(n, [], s_m=1.0).shape == (n, 0)


def test_parse_arm():
    assert mp.parse_arm("time") == ("time", False, False, False)
    assert mp.parse_arm("time_pitch_motif") == ("time_pitch", True, False, False)
    assert mp.parse_arm("time_motif_pc") == ("time", True, True, False)
    assert mp.parse_arm("time_pitch_motif_pc_pred") == ("time_pitch", True, True, True)
    with pytest.raises(AssertionError):
        mp.parse_arm("time_pc")


def test_predicted_occurrences_map_by_onset_and_pitch(tmp_path):
    root = tmp_path / "sweep_q80" / "fold0" / "motifs"
    root.mkdir(parents=True)
    # annotation frame = excerpt frame + 2 crotchets; one 3-note pattern twice,
    # one note of the second occurrence absent from the excerpt
    pats = [[[[2.0, 60], [3.0, 62], [4.0, 64]], [[8.0, 60], [9.0, 62], [10.5, 99]]],
            [[[5.0, 70]]]]
    (root / "01-1.json").write_text(json.dumps({"piece": "01-1", "patterns": pats}))
    mp.PRED["root"], mp.PRED["q"] = str(tmp_path), "80"
    q = np.arange(10, dtype=float)
    pitch = np.array([60, 62, 64, 65, 67, 69, 60, 62, 64, 65])
    occs = mp.predicted_occurrences(q, pitch, off=2.0, name="01-1")
    assert [list(o["notes"]) for o in occs] == [[0, 1, 2], [6, 7]]
    assert all(o["type"] == "pred0" for o in occs)
    assert np.allclose(occs[0]["u"], [0.0, 0.5, 1.0])
    assert np.allclose(occs[1]["u"], [0.0, 1.0])
