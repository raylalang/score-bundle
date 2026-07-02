"""Confirm the 45ep Stage-2 graph-on tau blowup is isolated collapsed cells.

Reproduces the eval's LM-s2x3 pipeline exactly (LOO embeddings, head_rng=7 head,
l2=10, identical masks) and reports per-(seed, piece) tau graph-on RMSE.
"""
import time

import numpy as np
import torch

from score_bundle import imputation_eval as ie
from score_bundle.downstream import load_piece_arrays, piece_score
from score_bundle.lm import features as lmfeat
from score_bundle.lm import masked as mk
from score_bundle.lm.model_torch import build_model
from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
from score_bundle.metrics import evaluate

device = "cuda:1"
ckpt = torch.load("checkpoints/maestro_masked_3x/best.pt", map_location=device,
                  weights_only=False)
model = build_model(ckpt["cfg"]).to(device)
model.load_state_dict(ckpt["model"]); model.eval()
tok = MidiTokenizer()

head, ev, meta = load_piece_arrays(".cache/asap_arrays_named.pkl")
ev = ev[:30]

def notes_of(p):
    return [NoteEvent(int(pi), float(oi), float(di), int(np.clip(v, 1, 127)))
            for pi, oi, di, v in zip(p["pitch"], p["onset"], p["duration"], p["velocity"])]

Yh = np.concatenate([p["y"] for p in head])
head_rng = np.random.default_rng(7)
H = np.concatenate([
    mk.masked_note_embeddings_loo(model, tok, notes_of(p),
                                  ie.random_mask(len(p["y"]), head_rng, 0.6))
    for p in head
])
W = lmfeat.fit_prior_mean_head(H, Yh, l2=10.0)
print("head fit", flush=True)

cells = []
pool = [[], [], []]
for s in range(4):
    seed_rng = np.random.default_rng(1000 + s)
    for pi, p in enumerate(ev):
        t0 = time.time()
        score, y = piece_score(p), p["y"]
        mask = ie.random_mask(len(y), seed_rng, observed_frac=0.6)
        emb = mk.masked_note_embeddings_loo(model, tok, notes_of(p), mask)
        mu = lmfeat.apply_prior_mean(emb, W)
        out = ie.impute_methods(score, y, {"s2x3": mu}, mask, fit_hyper=True,
                                rng=seed_rng, noise_floor_frac=0.05)
        cell = out[("s2x3", True)]
        t = cell.channel == 0
        yy, pp_, ss_ = cell.y[t], cell.pred[t], cell.std[t]
        pool[0].append(yy); pool[1].append(pp_); pool[2].append(ss_)
        r = float(np.sqrt(np.mean((yy - pp_) ** 2)))
        cells.append((s, pi, r))
        print(f"  s{s} p{pi:2d} tauRMSE {r:7.3f}  {time.time()-t0:5.1f}s", flush=True)
    print(f"seed {s + 1}/4 done", flush=True)

m = evaluate(*[np.concatenate(v) for v in pool], level=0.9)
bad = [c for c in cells if c[2] > 0.5]
print(f"\npooled tau graph-on: RMSE {m['rmse']:.4f} NLL {m['nll']:.4f} "
      f"cov {m['coverage@0.90']:.3f} | cells >0.5: {len(bad)}/120")
for s, pi, r in sorted(bad, key=lambda c: -c[2]):
    print(f"    seed {s} piece {pi:2d}  tau RMSE {r:.3f}")
