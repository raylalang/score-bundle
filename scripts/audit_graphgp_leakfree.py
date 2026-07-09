"""End-to-end zero-leak audit of the GP-first run path (eval_graphgp.stage_run).

Trims the real cache to 1 piece x 1 seed and runs the ACTUAL stage_run twice for
b_featlm (the current best config): once clean, once with held-out y corrupted to
1e6. Predictions and stds must be bitwise identical; the ground-truth column must
differ (proving the corruption landed). The embedding-side invariance (held-out
velocities never reach the mask-aware embeddings) was already proven bitwise on the
shared precompute path (scripts/audit_kernel_leakfree.py) and is pinned by
tests/test_lm_leakage.py + tests/test_graphgp.py.
"""
import copy
import os
import pickle
import sys
import types

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import eval_graphgp as eg

SCRATCH = os.environ.get("AUDIT_TMP", "/tmp/score_bundle_leak_audit_gp")
os.makedirs(SCRATCH, exist_ok=True)

with open(".cache/asap_arrays_named.pkl", "rb") as fh:
    blob = pickle.load(fh)
with open(".cache/kernel_sweep_inputs.pkl", "rb") as fh:
    real_inputs = pickle.load(fh)
with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
    emb_all = pickle.load(fh)

mask00 = real_inputs["masks"][(0, 0)]
held = ~mask00
print(f"piece 0: {len(mask00)} notes, {held.sum()} held out")

# trimmed inputs + emb dump (1 piece, 1 seed)
tin = os.path.join(SCRATCH, "inputs.pkl")
with open(tin, "wb") as fh:
    pickle.dump({"masks": {(0, 0): mask00},
                 "meta": {**real_inputs["meta"], "n_eval_pieces": 1, "seeds": 1}}, fh)
temb = os.path.join(SCRATCH, "emb.pkl")
with open(temb, "wb") as fh:
    pickle.dump({"emb_ma": {(0, 0): emb_all["emb_ma"][(0, 0)]}, "meta": {}}, fh)


def args(cache, outdir):
    return types.SimpleNamespace(
        arrays_cache=cache, inputs=tin, emb_dump=temb, out_dir=outdir,
        configs="b_featlm", shard="0/1", maxiter=200, corpus_pieces=20,
        baseline="", eval_start=0, guard=False,
        fixed_mean_inputs=".cache/kernel_sweep_inputs_featlm.pkl")


cells = {}
for tag, corrupt in (("clean", False), ("dirty_y", True)):
    ev0 = copy.deepcopy(blob["eval"][0])
    if corrupt:
        ev0["y"] = ev0["y"].copy()
        ev0["y"][held, :] = 1e6
    cpath = os.path.join(SCRATCH, f"cache_{tag}.pkl")
    with open(cpath, "wb") as fh:
        pickle.dump({"head": blob["head"], "eval": [ev0], "meta": blob["meta"]}, fh)
    outdir = os.path.join(SCRATCH, f"res_{tag}")
    eg.stage_run(args(cpath, outdir))
    with open(os.path.join(outdir, "b_featlm.pkl"), "rb") as fh:
        cells[tag] = pickle.load(fh)["cells"][("GP", 0, 0)]

yt_c, pr_c, sd_c, _ = cells["clean"]
yt_d, pr_d, sd_d, _ = cells["dirty_y"]
assert not np.array_equal(yt_c, yt_d), "corruption did not reach the y column"
ok = np.array_equal(pr_c, pr_d) and np.array_equal(sd_c, sd_d)
print(f"[GP run path] predictions+stds bitwise invariant to held-out y: {ok}")
assert ok, "LEAK: held-out y reached GP predictions"
print("GP AUDIT PASSED (bitwise).")
