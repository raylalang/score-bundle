"""Diagnostic: is the LM+graph NLL blowup on the 30-piece set driven by a few pieces,
and does the calibration-split EB fit (calib) tame it?  Reuses the robust arrays cache.

Per piece (1 mask seed), for the LM mean, report held-out RMSE / NLL under:
  off            (mean only)
  graph-marglik  (EB by in-sample marginal likelihood)
  graph-calib    (EB by held-out calibration-split NLL)
Sorted by marglik NLL so the offenders surface at the bottom.
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, "src")
from score_bundle import imputation_eval as ie
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import evaluate as _eval

CACHE = ".cache/robust_arrays.pkl"
N_EVAL = 30

with open(CACHE, "rb") as fh:
    blob = pickle.load(fh)
head_arr, eval_arr = blob["head"], blob["eval"][:N_EVAL]
H = np.concatenate([p["emb"] for p in head_arr]); Yh = np.concatenate([p["y"] for p in head_arr])
W = lmfeat.fit_prior_mean_head(H, Yh, l2=10.0)

variants = [(False, False, "marglik"), ("graph", True, "marglik"), ("calib", True, "calib")]
rng = np.random.default_rng(1000)  # same base as robust seed 0

rows = []
from score_bundle.score import Score
def piece_score(p):
    return Score.from_arrays(p["pitch"], p["onset"], p["duration"], p["voice"])

for pi, p in enumerate(eval_arr):
    score = piece_score(p)
    y = p["y"]
    mask = ie.random_mask(len(y), rng, observed_frac=0.6)
    mu_lm = lmfeat.apply_prior_mean(p["emb"], W)
    cells = ie.impute_methods(score, y, {"LM": mu_lm}, mask, fit_hyper=True,
                              graph_variants=variants, rng=rng)
    m = {}
    for lab in ["off" if False else False, "graph", "calib"]:
        c = cells[("LM", lab)]
        e = _eval(c.y, c.pred, c.std, level=0.9)
        m[lab] = (e["rmse"], e["nll"])
    rows.append((pi, f"piece{pi}", len(y), m))

rows.sort(key=lambda r: r[3]["graph"][1])  # by marglik NLL
print(f"{'piece':28s} {'N':>4s} | {'off RMSE/NLL':>16s} | {'marglik RMSE/NLL':>18s} | {'calib RMSE/NLL':>18s}")
for pi, name, n, m in rows:
    o, g, c = m[False], m["graph"], m["calib"]
    print(f"{str(name)[:28]:28s} {n:4d} | {o[0]:6.3f}/{o[1]:8.3f} | {g[0]:6.3f}/{g[1]:9.3f} | {c[0]:6.3f}/{c[1]:9.3f}")

def summ(lab):
    nlls = np.array([m[lab][1] for *_, m in [(r[0], r[1], r[2], r[3]) for r in rows]])
    rmses = np.array([m[lab][0] for *_, m in [(r[0], r[1], r[2], r[3]) for r in rows]])
    return rmses, nlls

print("\nsummary (per-piece, n=%d):" % len(rows))
for lab, name in [(False, "off"), ("graph", "marglik"), ("calib", "calib")]:
    rm, nl = summ(lab)
    print(f"  {name:8s} RMSE mean {rm.mean():.3f} median {np.median(rm):.3f} | "
          f"NLL mean {nl.mean():8.3f} median {np.median(nl):7.3f} max {nl.max():8.3f} "
          f"| #pieces NLL>1: {(nl>1).sum()}")
