#!/usr/bin/env python
"""Attribution across masking levels (DEV): covariance shares + graph x LM
redundancy at obs 0.50/0.70/0.80/0.90, seed 0, config b_featlm.

Completes the decomposition study: is the anchor-rate attribution (features
carry the mean, embeddings on velocity, graph modest; graph x LM near zero)
stable as observation density changes?

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_rate_shares.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_posterior_components import COMPS, cov_shares, fit_components  # noqa: E402

TAGS = ["obs0.50", "obs0.70", "obs0.80", "obs0.90"]
CHANNELS = ["tau", "log r", "v"]
OUT = "results/rate_shares_dev.md"


def main() -> None:
    from score_bundle.downstream import load_piece_arrays

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    lines = ["# Attribution across masking levels (DEV, seed 0, b_featlm)", ""]
    for tag in TAGS:
        with open(f".cache/masksweep_inputs_{tag}.pkl", "rb") as fh:
            masks = pickle.load(fh)["masks"]
        with open(f".cache/masksweep_emb_{tag}.pkl", "rb") as fh:
            dump = pickle.load(fh)
            embs = dump.get("emb_ma", dump)
        shares, corrs = [], []
        for pi in range(30):
            mask = masks[(pi, 0)]
            _, comps = fit_components(ev[pi], mask, embs[(pi, 0)])
            shares.append(cov_shares(comps, mask))
            print(f"  {tag} piece {pi} done", flush=True)
        A = np.array(shares)                      # (30, comp, channel)
        hidden = 1.0 - float(tag[3:7])
        lines.append(f"## {hidden:.0%} hidden")
        lines.append("")
        lines.append("| component | " + " | ".join(CHANNELS) + " |")
        lines.append("|---|---|---|---|")
        for j, (_, label, _) in enumerate(COMPS):
            cells = []
            for c in range(3):
                v = A[:, j, c]
                v = v[np.isfinite(v)]
                cells.append(f"{np.mean(v):+.2f} ± {np.std(v):.2f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        print("\n".join(lines[-6:]), flush=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
