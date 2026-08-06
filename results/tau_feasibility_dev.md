# tau feasibility — URMP development side (annotated onsets as warp anchors)

Tracks: 76 usable of 78 (38 exact order match, 38 via pitch DTW, 2 unusable).

- tau std under GLOBAL linear warp: median 369 ms (tempo drift dominates)
- tau std under LOCAL (+/-8-note, leave-one-out) warp: median 79 ms — the Phase-1-comparable residual scale
- lag-1 neighbour correlation of local tau: median +0.59 (IQR [+0.43, +0.64]) — the structure a graph prior could model

Verdict input for the tau policy: annotated onsets anchor a usable warp on the overwhelming majority of tracks without any audio aligner; residual scale and neighbour correlation are in the range where the graph prior operates in Phase 1.
