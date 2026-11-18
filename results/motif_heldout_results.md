# Motif pilot (dev): 58 Beethoven first movements, 696 cells


## scheme = occ, hidden notes = motif

| arm | channel | NLL | RMSE | cov@0.90 | dNLL vs time [95% CI] | dRMSE vs time [95% CI] | n pieces |
|---|---|---|---|---|---|---|---|
| time | tau | -1.502 | 0.0643 | 0.968 | — | — | 18 |
| time | log r | 0.902 | 0.6219 | 0.891 | — | — | 18 |
| time | v | -1.017 | 0.0877 | 0.878 | — | — | 18 |
| time_motif | tau | -1.520 | 0.0610 | 0.966 | -0.018 [-0.191, +0.225] | -0.0037 [-0.0076, -0.0007]* | 18 |
| time_motif | log r | 0.876 | 0.5977 | 0.878 | -0.026 [-0.098, +0.034] | -0.0286 [-0.0494, -0.0109]* | 18 |
| time_motif | v | -1.033 | 0.0847 | 0.847 | -0.016 [-0.042, +0.009] | -0.0032 [-0.0059, -0.0007]* | 18 |
| time_motif_pc | tau | -1.496 | 0.0635 | 0.976 | +0.006 [-0.049, +0.051] | -0.0005 [-0.0040, +0.0026] | 18 |
| time_motif_pc | log r | 0.850 | 0.6005 | 0.919 | -0.051 [-0.104, -0.012]* | -0.0232 [-0.0411, -0.0064]* | 18 |
| time_motif_pc | v | -1.102 | 0.0810 | 0.934 | -0.085 [-0.177, -0.022]* | -0.0060 [-0.0121, -0.0019]* | 18 |

## scheme = occ, hidden notes = all

| arm | channel | NLL | RMSE | cov@0.90 | dNLL vs time [95% CI] | dRMSE vs time [95% CI] | n pieces |
|---|---|---|---|---|---|---|---|
| time | tau | -1.503 | 0.0642 | 0.968 | — | — | 18 |
| time | log r | 0.901 | 0.6216 | 0.891 | — | — | 18 |
| time | v | -1.017 | 0.0877 | 0.878 | — | — | 18 |
| time_motif | tau | -1.520 | 0.0610 | 0.966 | -0.017 [-0.190, +0.225] | -0.0036 [-0.0073, -0.0007]* | 18 |
| time_motif | log r | 0.875 | 0.5974 | 0.878 | -0.026 [-0.098, +0.034] | -0.0284 [-0.0493, -0.0108]* | 18 |
| time_motif | v | -1.033 | 0.0847 | 0.847 | -0.016 [-0.042, +0.009] | -0.0032 [-0.0058, -0.0007]* | 18 |
| time_motif_pc | tau | -1.497 | 0.0634 | 0.976 | +0.007 [-0.049, +0.051] | -0.0005 [-0.0040, +0.0026] | 18 |
| time_motif_pc | log r | 0.850 | 0.6002 | 0.919 | -0.051 [-0.104, -0.012]* | -0.0231 [-0.0411, -0.0063]* | 18 |
| time_motif_pc | v | -1.102 | 0.0810 | 0.934 | -0.085 [-0.177, -0.022]* | -0.0060 [-0.0121, -0.0019]* | 18 |

## learned w_m per piece (time_motif arm, geometric mean over seeds)

| piece | w_m (occ) | occurrences | motif notes |
|---|---|---|---|
| 2-1 / OUSSET01.mid | 0.00 | 19 | 198 |
| 2-1 / Yang02.mid | 0.03 | 18 | 194 |
| 4-1 / Benabdallah01.mid | 2.80 | 26 | 228 |
| 4-1 / CaiC01.mid | 0.01 | 26 | 228 |
| 4-1 / Chon03M.mid | 5.72 | 26 | 228 |
| 4-1 / LeeE02M.mid | 3.10 | 26 | 227 |
| 5-1 / SunD02M.mid | 0.19 | 24 | 176 |
| 5-1 / Wong01.mid | 0.24 | 24 | 176 |
| 7-1 / KOLESO01.mid | 0.00 | 8 | 167 |
| 7-1 / Larionova04M.mid | 0.00 | 8 | 168 |
| 7-1 / LeeS02M.mid | 0.00 | 8 | 167 |
| 7-1 / Lee_E01.mid | 0.00 | 8 | 167 |
| 8-1 / Na06M.mid | 0.13 | 11 | 62 |
| 11-1 / DeTurck01.mid | 57.71 | 16 | 174 |
| 11-1 / LiC03M.mid | 13.21 | 16 | 172 |
| 11-1 / LiYZ03M.mid | 0.00 | 16 | 171 |
| 11-1 / MaximovI02M.mid | 15.72 | 16 | 172 |
| 12-1 / Kleisen05M.mid | 0.03 | 11 | 122 |
| 12-1 / Zhao02.mid | 0.58 | 10 | 116 |
| 15-1 / Sekino01.mid | 0.04 | 18 | 153 |
| 15-1 / ZhaoA02.mid | 35.63 | 18 | 156 |
| 16-1 / Khmara01.mid | 0.15 | 18 | 107 |
| 16-1 / Khmara05M.mid | 0.37 | 18 | 107 |
| 16-1 / LeeSH02M.mid | 0.14 | 18 | 107 |
| 16-1 / Shybayeva02.mid | 0.00 | 18 | 107 |
| 17-1_no_repeat / GuoE02M.mid | 4.10 | 23 | 159 |
| 17-1_no_repeat / HONG01.mid | 2.78 | 23 | 164 |
| 17-1_no_repeat / KabuliL02M.mid | 374.45 | 23 | 164 |
| 17-1_no_repeat / Lee02.mid | 1.47 | 22 | 161 |
| 18-1 / ChenGuang03M.mid | 0.13 | 26 | 143 |
| 18-1 / KOLESO06M.mid | 0.17 | 26 | 142 |
| 18-1 / LeungR02M.mid | 0.39 | 26 | 143 |
| 18-1 / Levitsky01.mid | 7.05 | 26 | 144 |
| 21-1_no_repeat / GonzalezJ06M.mid | 0.31 | 12 | 95 |
| 21-1_no_repeat / KimSY02M.mid | 0.26 | 11 | 91 |
| 21-1_no_repeat / Lariviere01.mid | 0.59 | 11 | 91 |
| 21-1_no_repeat / LeeN02M.mid | 0.35 | 11 | 91 |
| 22-1 / KOLESO04M.mid | 7.99 | 24 | 154 |
| 22-1 / WongDoe01.mid | 5.31 | 24 | 156 |
| 26-1_no_repeat / Kim02M.mid | 0.00 | 18 | 76 |
| 26-1_no_repeat / Tysman01.mid | 0.30 | 18 | 75 |
| 26-1_no_repeat / Ye02.mid | 2.21 | 18 | 78 |
| 27-1 / Fung01.mid | 0.09 | 19 | 145 |
| 27-1 / GiacomelliN04M.mid | 0.00 | 19 | 145 |
| 27-1 / Ko01.mid | 0.09 | 19 | 145 |
| 27-1 / Ko05M.mid | 0.14 | 19 | 145 |
| 28-1 / KimHJ02M.mid | 1.38 | 21 | 117 |
| 28-1 / KyykhynenT06M.mid | 0.47 | 20 | 116 |
| 28-1 / WangY04M.mid | 0.00 | 20 | 115 |
| 28-1 / Zhdanov01.mid | 0.00 | 21 | 116 |
| 30-1 / Hirata01.mid | 1.30 | 12 | 39 |
| 30-1 / Izzard01.mid | 0.19 | 12 | 39 |
| 30-1 / Katyukova01.mid | 0.76 | 12 | 39 |
| 30-1 / MaslovV02M.mid | 1.83 | 13 | 39 |
| 31-1 / AbdelmoulaJS03M.mid | 0.00 | 5 | 25 |
| 31-1 / Ahfat02M.mid | 0.00 | 5 | 25 |
| 31-1 / Ahn01.mid | 0.00 | 5 | 25 |
| 31-1 / BrendleA03M.mid | 0.00 | 5 | 25 |

## learned per-channel motif scale c_m per piece (time_motif_pc arm, occ scheme, geometric mean over seeds)

| piece | c_m tau | c_m log r | c_m v | c_score tau | c_score log r | c_score v |
|---|---|---|---|---|---|---|
| 2-1 / OUSSET01.mid | 3.94e-05 | 0.322 | 0.0124 | 4.01e-05 | 0.0261 | 0.000159 |
| 2-1 / Yang02.mid | 8.06e-05 | 7e-12 | 0.000368 | 2.36e-05 | 0.0229 | 0.000197 |
| 4-1 / Benabdallah01.mid | 8.95e-08 | 0.261 | 0.00404 | 8.89e-06 | 0.00127 | 0.000322 |
| 4-1 / CaiC01.mid | 2.7e-08 | 0.0536 | 0.000805 | 5.35e-06 | 0.00198 | 0.000514 |
| 4-1 / Chon03M.mid | 2.19e-08 | 0.275 | 0.00477 | 9.47e-06 | 0.000602 | 0.000337 |
| 4-1 / LeeE02M.mid | 4.52e-08 | 0.172 | 1.64e-08 | 3.71e-06 | 0.00475 | 0.000221 |
| 5-1 / SunD02M.mid | 1.37e-06 | 0.412 | 1.92e-06 | 1.17e-06 | 0.00633 | 0.00015 |
| 5-1 / Wong01.mid | 8.69e-07 | 0.129 | 4.88e-06 | 4.54e-06 | 0.0026 | 0.000202 |
| 7-1 / KOLESO01.mid | 0.000877 | 7.33e-06 | 0.000566 | 1.76e-05 | 0.0139 | 0.000428 |
| 7-1 / Larionova04M.mid | 3.97e-08 | 0.0338 | 5.27e-05 | 3.88e-06 | 0.0142 | 0.000392 |
| 7-1 / LeeS02M.mid | 1.41e-05 | 2.4e-05 | 4.25e-07 | 3.63e-05 | 0.0192 | 0.000192 |
| 7-1 / Lee_E01.mid | 2.56e-05 | 0.0103 | 2.66e-06 | 2.76e-05 | 0.0153 | 0.000313 |
| 8-1 / Na06M.mid | 1.72e-07 | 0.00305 | 0.0585 | 0.00012 | 0.00255 | 0.00226 |
| 11-1 / DeTurck01.mid | 3.71e-07 | 0.00227 | 7.23e-07 | 4.19e-05 | 0.0206 | 9.3e-05 |
| 11-1 / LiC03M.mid | 9.56e-09 | 0.000137 | 0.00773 | 4.04e-06 | 0.0153 | 0.000128 |
| 11-1 / LiYZ03M.mid | 1.61e-08 | 3.45e-06 | 0.000772 | 3.05e-06 | 0.0168 | 0.000229 |
| 11-1 / MaximovI02M.mid | 9.69e-13 | 1.34e-05 | 0.00358 | 1.17e-05 | 0.0118 | 6.24e-05 |
| 12-1 / Kleisen05M.mid | 0.0167 | 0.000232 | 0.000473 | 0.000102 | 0.0173 | 6.74e-05 |
| 12-1 / Zhao02.mid | 0.00166 | 6.1e-05 | 0.00263 | 0.000153 | 0.0363 | 0.0002 |
| 15-1 / Sekino01.mid | 0.000828 | 0.188 | 0.00132 | 1.59e-06 | 0.00404 | 0.000165 |
| 15-1 / ZhaoA02.mid | 5.99e-08 | 0.0128 | 5.04e-07 | 7.7e-05 | 0.0219 | 0.000254 |
| 16-1 / Khmara01.mid | 0.000651 | 0.301 | 3.84e-07 | 4.08e-06 | 0.00202 | 0.000362 |
| 16-1 / Khmara05M.mid | 4.36e-08 | 0.32 | 5.2e-07 | 2.8e-06 | 0.00403 | 0.000343 |
| 16-1 / LeeSH02M.mid | 7.65e-09 | 0.29 | 1.62e-08 | 2.26e-05 | 0.00151 | 0.000224 |
| 16-1 / Shybayeva02.mid | 0.000412 | 0.347 | 7.53e-08 | 1.01e-05 | 0.0178 | 0.000627 |
| 17-1_no_repeat / GuoE02M.mid | 0.0176 | 0.646 | 0.0428 | 1.42e-05 | 0.0167 | 7.09e-05 |
| 17-1_no_repeat / HONG01.mid | 0.0299 | 0.589 | 0.0301 | 5.56e-05 | 0.00715 | 8.87e-05 |
| 17-1_no_repeat / KabuliL02M.mid | 0.0286 | 0.0829 | 0.0347 | 1.78e-05 | 0.025 | 0.00013 |
| 17-1_no_repeat / Lee02.mid | 0.0142 | 0.845 | 0.034 | 1.96e-05 | 0.00672 | 0.000305 |
| 18-1 / ChenGuang03M.mid | 1.07e-08 | 0.438 | 1.64e-05 | 3.25e-05 | 0.00557 | 0.000174 |
| 18-1 / KOLESO06M.mid | 2.89e-11 | 0.000195 | 2.01e-11 | 2.17e-16 | 0.0055 | 0.000158 |
| 18-1 / LeungR02M.mid | 6.1e-08 | 0.13 | 8.77e-07 | 4.18e-05 | 0.0055 | 0.000217 |
| 18-1 / Levitsky01.mid | 9.56e-08 | 0.618 | 2.08e-07 | 3.94e-05 | 0.00534 | 8.8e-05 |
| 21-1_no_repeat / GonzalezJ06M.mid | 9.31e-06 | 0.399 | 0.0195 | 0.000112 | 0.0028 | 0.000172 |
| 21-1_no_repeat / KimSY02M.mid | 7.08e-06 | 0.066 | 0.00714 | 3.58e-06 | 0.00629 | 0.000187 |
| 21-1_no_repeat / Lariviere01.mid | 4.59e-07 | 0.0418 | 1.34e-05 | 2.23e-06 | 0.00454 | 0.000463 |
| 21-1_no_repeat / LeeN02M.mid | 4.17e-06 | 0.391 | 0.0103 | 6.28e-08 | 0.00748 | 7.36e-05 |
| 22-1 / KOLESO04M.mid | 6.03e-13 | 0.00633 | 0.032 | 1.62e-05 | 1.7e-06 | 0.000429 |
| 22-1 / WongDoe01.mid | 1.18e-17 | 0.629 | 0.00103 | 2.9e-06 | 0.00181 | 0.000902 |
| 26-1_no_repeat / Kim02M.mid | 2.33e-07 | 0.26 | 2.15e-06 | 0.000102 | 0.00178 | 0.00037 |
| 26-1_no_repeat / Tysman01.mid | 1.59e-07 | 0.29 | 2.01e-05 | 6.96e-05 | 0.00503 | 0.000175 |
| 26-1_no_repeat / Ye02.mid | 2.85e-07 | 0.0202 | 5.31e-05 | 9.18e-06 | 0.00276 | 0.000175 |
| 27-1 / Fung01.mid | 0.00048 | 0.000507 | 3.09e-06 | 2.29e-06 | 0.00924 | 0.000107 |
| 27-1 / GiacomelliN04M.mid | 8.18e-06 | 1.56e-16 | 2.12e-08 | 1.01e-06 | 0.00677 | 0.00016 |
| 27-1 / Ko01.mid | 7.17e-10 | 0.000116 | 1.6e-07 | 1.3e-06 | 0.00264 | 0.00015 |
| 27-1 / Ko05M.mid | 1.59e-13 | 1.17e-05 | 1.79e-07 | 7.17e-07 | 0.00106 | 0.000129 |
| 28-1 / KimHJ02M.mid | 9.59e-10 | 5.25e-17 | 0.0169 | 0.000577 | 0.00302 | 0.000766 |
| 28-1 / KyykhynenT06M.mid | 1.61e-07 | 2.79e-05 | 0.00934 | 0.00029 | 0.0114 | 0.00053 |
| 28-1 / WangY04M.mid | 1.66e-06 | 0.0157 | 0.0164 | 0.000114 | 0.00611 | 0.000619 |
| 28-1 / Zhdanov01.mid | 8.22e-21 | 1.03e-09 | 0.0176 | 0.000756 | 0.00376 | 0.000591 |
| 30-1 / Hirata01.mid | 1.32e-07 | 6e-07 | 0.0101 | 0.000256 | 0.00176 | 0.000531 |
| 30-1 / Izzard01.mid | 7.96e-06 | 3.31e-05 | 0.00244 | 0.000171 | 0.00333 | 0.000496 |
| 30-1 / Katyukova01.mid | 7.7e-07 | 0.0194 | 0.0187 | 0.000174 | 0.00374 | 0.000759 |
| 30-1 / MaslovV02M.mid | 1.68e-05 | 0.0543 | 0.00767 | 0.000142 | 0.00204 | 0.000492 |
| 31-1 / AbdelmoulaJS03M.mid | 2.03e-07 | 2.46 | 0.049 | 1.13e-05 | 0.015 | 5.57e-05 |
| 31-1 / Ahfat02M.mid | 8.78e-09 | 0.788 | 0.0593 | 0.00027 | 0.00685 | 0.000106 |
| 31-1 / Ahn01.mid | 3.4e-06 | 2.69 | 0.0425 | 3.65e-05 | 0.0186 | 0.000157 |
| 31-1 / BrendleA03M.mid | 8.22e-07 | 0.592 | 0.0615 | 0.000186 | 0.0105 | 6.06e-05 |

## VERDICT against the registered claims (scored 2026-11-18, run 13:43-15:31 +09:00, 1 h 48 min, 14 shards x OMP=2, code state 1fa6d77 = tag motif-registration-2026-11-18; decision rule verbatim from docs/motif_prereg_design.md @ that tag)

The report's title line says "58 Beethoven first movements": the generator counts performances; the pool is 18 movements, 58 performances, averaged within the movement before pairing (n pieces = 18 in every row).

- **M1 (articulation RMSE on hidden motif notes, occurrence-hidden scheme, shared arm `time_motif` minus `time`, PRIMARY) — PASS.** Paired per-piece ΔRMSE −0.0286 [−0.0494, −0.0109]\*, percentile bootstrap over the 18 pieces, B = 2000, rng 0. Negative mean, interval excludes zero. 15 of 18 pieces below zero. Development basis −0.0345 [−0.0600, −0.0125]\* (pilot two, 21 pieces, four seeds) reproduced at 0.83 of its size on performances no run had read; the per-piece effects correlate at 0.88 with the development ones on the same 18 movements.
- **M2 (timing log score, same notes, SECONDARY, non-gating) — FAIL.** Paired ΔNLL −0.018 [−0.191, +0.225], interval covers zero (development −0.071 [−0.230, +0.097], also not starred; pre-stated power 0.33, a miss expected). Reported verbatim. Timing RMSE (report-only) −0.0037 s [−0.0076, −0.0007]\*.
- **M3 (report-only), shared arm:** velocity log score −0.016 [−0.042, +0.009] (the pilot-one penalty, +0.069\* on two seeds and +0.033 on four, is gone), velocity RMSE −0.0032 [−0.0059, −0.0007]\*, velocity coverage 0.847 at nominal 0.90 (time-only arm 0.878); articulation coverage 0.878 (time-only 0.891); timing coverage 0.966.
- **M3 (report-only), per-channel arm `time_motif_pc`:** articulation log score −0.051 [−0.104, −0.012]\* and RMSE −0.0232 [−0.0411, −0.0064]\*; velocity log score −0.085 [−0.177, −0.022]\* and RMSE −0.0060 [−0.0121, −0.0019]\*, coverage 0.934; timing +0.006 [−0.049, +0.051], RMSE −0.0005, nothing. The development picture (articulation in both forms, velocity repaired by the per-channel form, timing switched off) reproduces.
- **Learned weights:** shared w_m above 0.1 on 37 of 58 performances (median 0.21); per-channel c_m above 0.1 on articulation on 26 of 58 performances and on no performance on timing or velocity, above 0.01 on 5 (timing), 37 (articulation), 18 (velocity).
- Cells: 696 of 696 fits (58 performances x 4 seeds x 3 arms), no skips, no tracebacks; the occurrence-hidden scheme only, as the frozen commands state. The "hidden notes = all" table equals the motif table because under this scheme every hidden note is a motif note.
- **Headline, by the registered rule:** annotated motif structure carries expressive information beyond temporal smoothing, confirmed once on held-out performances of the development pieces (disclosed): articulation accuracy on hidden returns. The test is across performers of the same movements, not across repertoire, and is labelled so wherever it appears.
- Evidence: evidence/motif_heldout/ (run log with the performance list, driver log, 14 shard logs, align.md, report.md, effects.json, pieces.pkl md5 71c06d636712953f057da3ab86fe93cb, 14 cell shards). The pool is spent; never rerun.
