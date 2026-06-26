#!/usr/bin/env python
"""Parse a train_lm.py log into history.json and plot the LM training curve.

    python scripts/plot_training_curve.py --log <train.log> \
        --out-json checkpoints/maestro_scaled/history.json \
        --out-fig figures/lm_training_curve.png --title "MAESTRO LM (d=256, L=4)"

Pulls the per-epoch ``epoch N: train_loss .. | val_loss .. | val_ppl ..`` lines, so it works
on any run's stdout log (no dependency on the trainer persisting history).
"""
from __future__ import annotations

import argparse
import json
import os
import re

EPOCH_RE = re.compile(
    r"epoch\s+(\d+):\s*train_loss\s+([0-9.]+)\s*\|\s*val_loss\s+([0-9.]+)\s*\|\s*val_ppl\s+([0-9.]+)"
)


def parse_log(path: str) -> dict:
    epochs, tr, vl, ppl = [], [], [], []
    with open(path) as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                vl.append(float(m.group(3)))
                ppl.append(float(m.group(4)))
    return {"epoch": epochs, "train_loss": tr, "val_loss": vl, "val_ppl": ppl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--title", default="LM training curve")
    args = ap.parse_args()

    hist = parse_log(args.log)
    if not hist["epoch"]:
        raise SystemExit(f"no epoch lines found in {args.log}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(hist, fh, indent=2)
        print(f"wrote {args.out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], "o-", label="train", color="#1f77b4")
    ax1.plot(hist["epoch"], hist["val_loss"], "s-", label="val", color="#d62728")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy (nats/token)")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(hist["epoch"], hist["val_ppl"], "s-", color="#d62728")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation perplexity")
    lo, hi = hist["val_ppl"][-1], hist["val_ppl"][0]
    ax2.set_title(f"Val perplexity  {hi:.1f} → {lo:.1f}")
    ax2.grid(alpha=0.3)
    ax2.annotate(f"{lo:.2f}", (hist["epoch"][-1], lo),
                 textcoords="offset points", xytext=(-8, 8))

    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"wrote {args.out_fig}  ({len(hist['epoch'])} epochs)")


if __name__ == "__main__":
    main()
