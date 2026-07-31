# Posterior component correlations at held-out notes (DEV)

30 dev pieces, anchor-mask seed 0, config b_featlm. Mean per-note posterior correlation between component pairs (negative = explaining-away / redundancy).

| pair | tau | log r | v |
|---|---|---|---|
| graph x LM features | -0.004 ± 0.008 | -0.025 ± 0.041 | -0.059 ± 0.037 |
| score features x LM features | -0.067 ± 0.085 | -0.079 ± 0.109 | -0.183 ± 0.072 |
| graph x score features | -0.041 ± 0.081 | -0.185 ± 0.105 | -0.179 ± 0.097 |
