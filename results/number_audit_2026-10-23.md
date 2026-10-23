# Number audit of the thesis against its records (2026-10-23, writing freeze)

Every numeric statement in `docs/thesis/draft.tex` (abstract macros, every
table, every inline contrast in the piano, strings-and-winds and waveform
chapters, Appendix app:trackscale, dataset counts) was checked against its
record (`results/*.md`, `docs/*_results.md`, `logs/*.log`,
`evidence/phase2_confirmation/`). Read-only audit by a subagent, fixes by
hand.

## Mismatches found and fixed

1. Downstream, completion from an excerpt: the thesis and the prose of
   `docs/downstream_tasks_results.md` said NLL 0.21 → 0.01; the record
   (`logs/completion.log`, observed fraction 0.5) says 0.2076 → 0.0962,
   i.e. 0.21 → 0.10. The results doc's own table was right. Fixed in both.
2. Piano attribution, posterior decomposition: the thesis implied every
   decomposition figure used 30 pieces × 4 anchor masks. Per the record's
   provenance note (`docs/posterior_decomposition_results.md`), only the
   covariance shares are 30 × 4; the component correlations, the ARD
   switch-off share, the per-piece dominance and the disagreement fractions
   are the first anchor mask only. The thesis now says so. Values unchanged.
3. Waveform, spectral-mixture deviation prior: the thesis called the paired
   mean +0.11 cents "significantly worse"; the record's hardening note
   (`results/phase3_smprior_dev.md`, 2026-09-16) says the note-level star
   does not survive piece-level clustering (Wilcoxon p = 0.63, not
   significant after BH) and the verdict is "no gain, not measured harm".
   The thesis now says that.

## Unsourced numbers, recomputed from the caches

- Strings-and-winds data section: "per-track vibrato-identifiable fraction
  from 2% to 97%, median 38%" and "one track falls below the 30-usable-note
  minimum". Recomputed from `.cache/urmp_targets_dev.pkl` (78 tracks):
  min 0.020, max 0.971, median 0.384; one track with 29 usable notes
  (MIN_NOTES = 30 in `scripts/eval_phase2_real.py`). Confirmed.
- Tonal figure caption, "61% of pairs improving": computed by
  `scripts/make_tonal_intonation.py` from `results/phase2_tonal_cells.pkl`
  at figure time. cells pickle absent (gitignored, not on this checkout); the 61% stays as the figure script computed it

## Figure-caption numbers computed inside figure scripts (no separate record)

fig:boundary-deploy (39.7 → 19.5 ms, 73% improve), fig:span-calibration
(27 of 40 beyond ±8 sd), fig:vibrato-compare, fig:posterior-example
(49/50, 49/50, 46/50), fig:drift-study, the sec:phase2-delta figure. These
are produced by the figure scripts from the same cells the records
summarise; they are not contradicted by any record.

## Checked and consistent

All abstract and development macros; tab:headline-conf, tab:perchannel
(three variants), tab:headline-exec, tab:kernels, tab:downstream-both,
tab:phase2-dev, tab:phase2-conf, tab:drift-study, tab:phase3-study,
tab:phase3-priorvalue, tab:trackscale; every inline contrast, interval,
star and n in the three results chapters and the appendices; every dataset
count (30/20 ASAP; 44/13/78 URMP; 376 notes / 7 tracks; 47 chains / 154
boundaries; 40 joints; 48 phrases; 153 cells; 120 cells; 30 × 4).
