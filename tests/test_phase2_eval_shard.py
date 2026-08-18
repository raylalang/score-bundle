"""Shard-equivalence proof for the Phase-2 evaluation.

The sharded path (stage_run over k/n fragments + stage_report merging them in
canonical cell order) must produce BYTE-IDENTICAL output to the single-process
path, at the same BLAS thread count — this is what lets the sharded driver
(and eventually the guarded confirmation runner) stand in for the registered
`stage_eval` logic.  Runs on a synthetic mini-cache; no URMP data involved.
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

epr = importlib.import_module("eval_phase2_real")


def _mini_cache(rng: np.random.Generator, n_tracks: int = 4,
                n_notes: int = 40) -> dict:
    instruments = ["vn", "fl", "tpt", "vc"]
    data = {}
    for t in range(n_tracks):
        n = n_notes
        est = np.column_stack([
            rng.normal(0.0, 15.0, n),                    # c (cents), usable
            np.log(np.abs(rng.normal(25.0, 5.0, n))),    # log gamma
            np.log(np.abs(rng.normal(6.0, 0.5, n))),     # log f
        ])
        ident = rng.random(n) < 0.6
        est[~ident, 1] = np.nan
        est[~ident, 2] = np.nan
        var = np.abs(rng.normal(1.0, 0.2, (n, 3)))
        var[~ident, 1:] = np.nan
        ell = rng.normal(0.0, 0.4, n)
        tau = rng.normal(0.0, 0.08, n)
        tau[:1] = np.nan                                  # edge note missing
        dvib = np.where(rng.random(n) < 0.4,
                        np.abs(rng.normal(0.15, 0.05, n)), np.nan)
        data[(t + 1, 1)] = {
            "piece": t + 1, "name": f"synth{t}", "instrument": instruments[t],
            "onset": np.arange(n) * 0.4,
            "duration": np.full(n, 0.35),
            "midi": rng.integers(55, 84, n),
            "est": est, "var": var, "ident": ident,
            "n_frames": np.full(n, 20),
            "est_gt": est + rng.normal(0.0, 1.0, (n, 3)),
            "var_gt": var.copy(), "ident_gt": ident.copy(),
            "ell": ell, "var_ell": np.abs(rng.normal(0.01, 0.003, n)),
            "tau": tau, "var_tau": np.abs(rng.normal(0.002, 5e-4, n)),
            "dvib": dvib, "var_dvib": np.where(np.isfinite(dvib),
                                               np.abs(rng.normal(1e-3, 3e-4, n)),
                                               np.nan),
            "dvib_gt": dvib + rng.normal(0.0, 0.01, n),
        }
    return data


@pytest.fixture()
def synth_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(".cache")
    cache = ".cache/synthetic_phase2.pkl"
    with open(cache, "wb") as fh:
        pickle.dump(_mini_cache(np.random.default_rng(7), n_tracks=3),
                    fh)
    monkeypatch.setattr(epr, "CACHE", cache)
    # cap the optimizer identically for BOTH paths: equivalence is about the
    # shard/merge/render plumbing, not fit quality, and full maxiter makes
    # the test minutes-heavy (numerical gradients over 34 hyperparameters)
    from score_bundle.gp import MultiOutputGraphGP
    orig_fit = MultiOutputGraphGP.fit

    def capped_fit(self, Y, mask, x0=None, noise_floor=None, maxiter=300,
                   noise_fixed=None, b_diagonal=False):
        return orig_fit(self, Y, mask, x0=x0, noise_floor=noise_floor,
                        maxiter=15, noise_fixed=noise_fixed,
                        b_diagonal=b_diagonal)

    monkeypatch.setattr(MultiOutputGraphGP, "fit", capped_fit)
    return tmp_path


def _read_outputs():
    with open(epr.TONAL_OUT_MD) as fh:
        md = fh.read()
    with open(epr.TONAL_CELLS, "rb") as fh:
        cells = pickle.load(fh)
    return md, cells


def test_sharded_path_is_byte_identical_to_single(synth_env):
    import shutil

    # single-process reference (tonal=True exercises the 5-system superset)
    epr.stage_eval(tonal=True)
    md_ref, cells_ref = _read_outputs()
    assert "| c (cents) |" in md_ref            # tables actually rendered

    # sharded 3 ways: fragments differ, the merged report must not
    shutil.rmtree(epr._cells_dir(True))
    for k in range(3):
        epr.stage_run(f"{k}/3", tonal=True)
    epr.stage_report(tonal=True)
    md_shard, cells_shard = _read_outputs()

    assert md_shard == md_ref
    assert cells_shard["per_track"] == cells_ref["per_track"]
    assert cells_shard["rows"].keys() == cells_ref["rows"].keys()
    for sname in cells_ref["rows"]:
        for tgt in ("est", "gt"):
            for c in cells_ref["rows"][sname][tgt]:
                a = cells_ref["rows"][sname][tgt][c]
                b = cells_shard["rows"][sname][tgt][c]
                assert a["n"] == b["n"]
                assert a["se"] == b["se"]        # exact float equality
                assert a["nll"] == b["nll"]


def test_report_refuses_incomplete_fragments(synth_env):
    epr.stage_run("0/3", tonal=True)             # shards 1/3, 2/3 missing
    with pytest.raises(SystemExit):
        epr.stage_report(tonal=True)
