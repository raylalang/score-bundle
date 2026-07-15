# Post-hoc replication set — results (2026-07-16)

> **Status: independent out-of-selection replication; NOT confirmation-grade.**
> 30 fresh contamination-filtered ASAP pieces (positions 50–79 of the same
> seed-0 corpus shuffle, extracted to `.cache/asap_arrays_named80.pkl`;
> positions 0–49 verified byte-identical to the published cache, so the
> development and confirmation sets are untouched). These pieces were never
> seen by any selection decision, but the run was not preregistered — it
> replicates the development ladder, it does not re-adjudicate the headline.
> Masks/embeddings from a disjoint seed base (7000), leak-free per mask,
> guard on. Runner: `scripts/run_replication_set.sh`; report:
> `scripts/report_replication.py`, `logs/replication_report.log`.

## Results (30 fresh pieces × 4 seeds, pooled)

| config | RMSE | NLL | cov@.9 | dev value (reference) |
|---|---|---|---|---|
| features only | 0.3922 | −0.323 | 0.929 | 0.3683 / −0.370 |
| **proposed model** | **0.3814** | **−0.356** | 0.929 | 0.3601 / −0.404 |
| proposed, no graph | 0.3997 | −0.277 | 0.935 | 0.3755 / −0.337 |

Paired per-piece contrasts (bootstrap 95% CIs over the 30 fresh pieces):

| contrast | ΔRMSE | ΔNLL | pieces negative |
|---|---|---|---|
| graph value | **−0.0187 [−0.0260, −0.0120]\*** | **−0.0797 [−0.0993, −0.0613]\*** | 27/30, 29/30 |
| embedding value | **−0.0102 [−0.0159, −0.0051]\*** | **−0.0334 [−0.0464, −0.0221]\*** | 25/30, 28/30 |

## Reading

1. **The ordering replicates** (proposed < features-only < no-graph), and both
   ingredient contributions are significant on both axes on pieces nobody ever
   looked at — the strongest kind of dev-grade evidence available without
   spending a fresh preregistration.
2. **The graph's NLL contribution on fresh pieces (−0.080) matches its
   confirmed value (−0.074)** almost exactly — three independent piece sets
   (dev, confirmation, replication) now agree on the graph's calibration role.
3. **Absolute levels are worse than dev** (0.381 vs 0.360 RMSE), consistent
   with the documented selection-inflation pattern (the confirmation showed the
   same: every system's numbers flatter on the reused set). The *effects*
   replicate; the absolute dev levels do not, and were never claimed to.
4. Coverage 0.929–0.935: the mild conservatism transfers unchanged.

## Reproduce

```bash
bash scripts/run_replication_set.sh          # extraction gate + precompute + 12 shards
PYTHONPATH=src:scripts python scripts/report_replication.py
```
