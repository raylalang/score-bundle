"""End-to-end zero-leak audit of the kernel sweep's REAL code paths.

Runs scripts/eval_kernels.py stage_precompute and stage_run twice each on a
1-piece x 1-seed trimmed copy of the real cache: once clean, once with held-out
notes' performance data corrupted (velocities -> extreme values for precompute;
y targets -> huge values for run). Every output that feeds a prediction must be
bitwise identical; the y check also verifies the ground-truth column DID change
(so the corruption really happened).
"""
import copy
import os
import pickle
import sys
import types

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import eval_kernels as ek  # the actual script under audit

SCRATCH = os.environ.get("AUDIT_TMP", "/tmp/score_bundle_leak_audit")
os.makedirs(SCRATCH, exist_ok=True)
CACHE = ".cache/asap_arrays_named.pkl"
INPUTS = ".cache/kernel_sweep_inputs.pkl"

with open(CACHE, "rb") as fh:
    blob = pickle.load(fh)
with open(INPUTS, "rb") as fh:
    real_inputs = pickle.load(fh)
mask00 = real_inputs["masks"][(0, 0)]
held = ~mask00
print(f"piece 0: {len(mask00)} notes, {held.sum()} held out under seed-0 mask")


def args(**kw):
    a = types.SimpleNamespace(
        arrays_cache=None, inputs=None, out_dir=None,
        checkpoint="checkpoints/maestro_scaled/best.pt", n_eval_pieces=1, seeds=1,
        observed_frac=0.6, l2=10.0, placeholder_vel=64, noise_floor_frac=0.05,
        mean="lm", kernels="harmonic_vl", boot=2000, device=None,
        eval_start=0, mask_seed_base=1000, dump_embeddings=None, baseline="additive")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def write_cache(path, eval0):
    trimmed = {"head": blob["head"], "eval": [eval0], "meta": blob["meta"]}
    with open(path, "wb") as fh:
        pickle.dump(trimmed, fh)


# --- A: precompute path — corrupt HELD-OUT velocities in the LM input source ----
clean0 = copy.deepcopy(blob["eval"][0])
dirty0 = copy.deepcopy(clean0)
dirty0["velocity"] = dirty0["velocity"].copy()
dirty0["velocity"][held] = np.where(np.arange(held.sum()) % 2, 1.0, 127.0)

mus = {}
for tag, ev0 in (("clean", clean0), ("dirty_vel", dirty0)):
    cpath = os.path.join(SCRATCH, f"cache_{tag}.pkl")
    ipath = os.path.join(SCRATCH, f"inputs_{tag}.pkl")
    write_cache(cpath, ev0)
    ek.stage_precompute(args(arrays_cache=cpath, inputs=ipath))
    with open(ipath, "rb") as fh:
        out = pickle.load(fh)
    mus[tag] = out["mu_lm"][(0, 0)]
    assert np.array_equal(out["masks"][(0, 0)], mask00), "mask sequence drifted"

same_mu = np.array_equal(mus["clean"], mus["dirty_vel"])
print(f"[A] strict mu_LM invariant to held-out velocities: {same_mu}")
assert same_mu, "LEAK: held-out velocity reached mu_LM"
# cross-process GPU runs may pick different cuBLAS algorithms -> float-level drift;
# the leak contract (clean == dirty, same process) is the bitwise one above.
drift = float(np.max(np.abs(mus["clean"] - real_inputs["mu_lm"][(0, 0)])))
print(f"[A] reproduction vs sweep cache: max |diff| = {drift:.2e} (float noise only)")
assert drift < 1e-5, "trimmed precompute does not reproduce the sweep's cached mu"

# --- B: run path — corrupt HELD-OUT y targets ------------------------------------
dirty_y0 = copy.deepcopy(clean0)
dirty_y0["y"] = dirty_y0["y"].copy()
dirty_y0["y"][held, :] = 1e6

cells = {}
for tag, ev0 in (("clean", clean0), ("dirty_y", dirty_y0)):
    cpath = os.path.join(SCRATCH, f"cache_{tag}.pkl")
    rdir = os.path.join(SCRATCH, f"results_{tag}")
    write_cache(cpath, ev0)
    ek.stage_run(args(arrays_cache=cpath, inputs=os.path.join(SCRATCH, "inputs_clean.pkl"),
                      out_dir=rdir))
    with open(os.path.join(rdir, "harmonic_vl.pkl"), "rb") as fh:
        cells[tag] = pickle.load(fh)["cells"]

for mean_name in ("LM", "zero"):
    yt_c, pr_c, sd_c, ch_c = cells["clean"][(mean_name, 0, 0)]
    yt_d, pr_d, sd_d, ch_d = cells["dirty_y"][(mean_name, 0, 0)]
    assert not np.array_equal(yt_c, yt_d), "corruption did not reach the y column"
    pred_ok = np.array_equal(pr_c, pr_d) and np.array_equal(sd_c, sd_d)
    print(f"[B] mu={mean_name}: predictions+stds invariant to held-out y: {pred_ok} "
          f"(ground-truth column changed as expected)")
    assert pred_ok, f"LEAK: held-out y reached predictions (mu={mean_name})"

print("\nAUDIT PASSED: zero leak on both real code paths (bitwise).")
