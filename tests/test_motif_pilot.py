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


def test_heldout_performances_refused_without_consent(monkeypatch):
    ann = {f"Beethoven/Piano_Sonatas/4-1/{n}.mid": {"score_and_performance_aligned": True}
           for n in ("A01", "B02", "C03", "D04", "E05", "F06")}
    ann["Beethoven/Piano_Sonatas/4-1/Z99.mid"] = {"score_and_performance_aligned": False}
    monkeypatch.delenv("MOTIF_HELDOUT_I_AM_SURE", raising=False)
    monkeypatch.setitem(mp.PERF, "source", "first")
    assert mp.performances_for("4-1", ann) == ["Beethoven/Piano_Sonatas/4-1/A01.mid"]
    monkeypatch.setitem(mp.PERF, "source", "heldout")
    with pytest.raises(SystemExit, match="REFUSED"):
        mp.performances_for("4-1", ann)
    monkeypatch.setenv("MOTIF_HELDOUT_I_AM_SURE", "yes")
    monkeypatch.setitem(mp.PERF, "cap", 4)
    got = mp.performances_for("4-1", ann)
    assert got == [f"Beethoven/Piano_Sonatas/4-1/{n}.mid" for n in ("B02", "C03", "D04", "E05")]
    assert "A01" not in " ".join(got)          # the development performance never re-used
    monkeypatch.setitem(mp.PERF, "cap", 1)
    assert mp.performances_for("4-1", ann) == ["Beethoven/Piano_Sonatas/4-1/B02.mid"]
    monkeypatch.setitem(mp.PERF, "source", "first")


def test_report_groups_performances_by_piece(tmp_path, monkeypatch):
    """Two performances of one piece and one of another: the per-piece table
    averages the performances before the bootstrap (n pieces = 2, not 3)."""
    import pickle
    rng = np.random.default_rng(0)
    n = 30
    pieces = []
    for folder, group in (("4-1 / A", "4-1"), ("4-1 / B", "4-1"), ("7-1 / C", "7-1")):
        pieces.append({"folder": folder, "group": group, "occs": [], "motif_notes": np.ones(n, bool)})
    cells = {}
    for pi in range(3):
        for s in range(2):
            for arm in ("time", "time_motif_pc"):
                held = n // 2
                yt = rng.normal(size=3 * held); pr = yt + rng.normal(scale=0.5, size=3 * held)
                sd = np.full(3 * held, 0.6); ch = np.repeat(np.arange(3), held)
                cells[(pi, "occ", s, arm)] = {"cell": (yt, pr, sd, ch), "held_idx": np.arange(held),
                                              "motif_sel": np.ones(held, bool),
                                              "info": {"x_hat": np.zeros(16).tolist()}}
    pickle.dump(pieces, open(tmp_path / "pieces.pkl", "wb"))
    pickle.dump(cells, open(tmp_path / "cells.shard0.pkl", "wb"))
    monkeypatch.setattr(mp, "OUT", str(tmp_path))
    monkeypatch.setattr(mp, "ARMS", ["time", "time_motif_pc"])
    monkeypatch.setattr(mp, "SCHEMES", ["occ"])
    monkeypatch.setattr(mp, "SEEDS", 2)
    mp.stage_report(None)
    import json
    eff = json.load(open(tmp_path / "effects.json"))["effects"]
    assert eff["occ|motif|time_motif_pc|tau"]["n"] == 2


def test_heldout_runner_refuses_without_env():
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    env = {k: v for k, v in os.environ.items() if k != "MOTIF_HELDOUT_I_AM_SURE"}
    out = subprocess.run(["bash", os.path.join(repo, "scripts", "run_motif_heldout.sh")],
                         capture_output=True, text=True, cwd=repo, env=env)
    assert out.returncode != 0 and "REFUSED" in out.stdout
