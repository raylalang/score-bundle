#!/usr/bin/env python
"""Mask-seed robustness for the Phase-2 dev results (seeds 2, 3; DEV ONLY).

Answers the committee question "are two mask seeds enough?" with a measured
line: the same six-channel evaluation on two FRESH mask seeds, compared to
the registered-protocol dev run (seeds 0, 1). Purely a development
robustness study — the registered protocol (seeds (0, 1)) is untouched, and
`scripts/eval_phase2_real.py` is not modified: this driver overrides the
module's seed tuple and output paths from outside, so the registered code
path stays byte-identical.

    # shard k/n (run 8 in parallel at OMP_NUM_THREADS=2), then report:
    python scripts/run_phase2_seeds23.py run 0/8
    python scripts/run_phase2_seeds23.py report
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_phase2_real as epr  # noqa: E402

assert not epr.CONF_MODE, "dev-only robustness study"
epr.SEEDS = (2, 3)
epr.OUT_MD = "results/phase2_seeds23_dev.md"
epr.CELLS_PKL = "results/phase2_seeds23_cells.pkl"
_orig_cells_dir = epr._cells_dir
epr._cells_dir = lambda tonal: os.path.join("results", "phase2_cells",
                                            "seeds23")

if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "run":
        epr.stage_run(sys.argv[2] if len(sys.argv) > 2 else "0/1",
                      tonal=False)
    elif verb == "report":
        epr.stage_report(tonal=False)
    else:
        raise SystemExit("usage: run [k/n] | report")
