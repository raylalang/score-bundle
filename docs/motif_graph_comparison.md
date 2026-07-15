# Note-graph comparison — score-bundle vs. the motif-discovery graph (2026-07-14)

> **Purpose (supervisor comment, 2026-07-13):** examine how the motif-discovery
> work (`~/Research/graph-motif-discovery`, LPCC on BPS-Motif) constructs its
> note graph, and adopt anything our score graph is missing. Status: comparison
> done; one relation adopted for a dev A/B (**sustain-overlap**), running as
> `c_overlap` (results section to be appended).

## The two constructions

| | motif-discovery (`motif_discovery/graph/graph_builder.py`) | score-bundle (`graph.build_adjacency*`) |
|---|---|---|
| edge semantics | three *typed* relations, kept separate | one weighted kernel + optional typed bonuses |
| simultaneity | same onset, k=4 nearest in pitch | chord bonus on all same-onset pairs (`chord_weight`) |
| temporal proximity | k=8 within Δt = 1 beat | Gaussian beat-distance kernel `exp(−Δb²/2ℓ_b²)` |
| pitch proximity | (via simultaneity ranking only) | Gaussian pitch-distance kernel `exp(−Δp²/2ℓ_p²)` |
| **sustain overlap** | **k=8 pairs where the earlier note still sounds** | **absent until now** |
| voice leading | — | stepwise different-onset bonus (`vl_weight`) |
| sparsification | kNN per relation | optional kNN (`knn=`), dense by default |
| edge weights | unweighted (GNN learns) | weights are the model (Laplacian → GP prior) |
| consumer | pair-scorer / GNN over embeddings | graph Matérn GP prior |

The deep difference is the consumer: LPCC hands an unweighted typed graph to a
learned scorer, so relation *types* carry the information; the GP prior consumes
one weighted Laplacian, so *weights* carry it. Our harmonic variant
(`build_adjacency_harmonic`) already is the "typed relations, per-type learned
weight" unification — chord and voice-leading families with weights fit by
evidence — so the constructions are philosophically compatible; each typed
relation in one framework maps to a weighted edge family in the other.

## What was adopted

**Sustain-overlap** is the one relation the motif graph has that ours could not
express: the beat-distance kernel never sees durations, so a long note sounding
through a neighbour's onset (pedal points, suspensions, held basses — pairs a
pianist physically co-articulates) is treated like any other time-distance pair.
Added as a third edge family `overlap_weight` in `build_adjacency_harmonic`
(same-onset pairs excluded — those are the chord family; symmetric; ablates to
the base graph at weight 0), and as eval config `c_overlap` (learned
`ℓ_b, ℓ_p, overlap_w` by evidence, mirroring `c_graph`'s learned `ℓ_b, ℓ_p` so
the paired contrast isolates the relation).

Not adopted, with reasons: kNN-typed unweighted edges (the GP needs weights;
kNN sparsification already exists as an option and pieces are small enough for
dense), and the k-nearest-in-pitch simultaneity ranking (the chord family plus
the pitch kernel covers it in weighted form).

## Prior expectation (recorded before the A/B finished)

The harmonic study found chord+voice-leading edges significant on the plain
graph and measured-redundant once the LM embeddings enter as features. The
honest expectation is the same fate for overlap edges: a real effect vs.
`c_graph` in the feature-only regime would already justify the borrow;
survival next to the embeddings would be an upgrade to the thesis graph.
Either way the result is reported.

## Dev A/B protocol

`scripts/run_overlap_ab.sh`: `c_graph` vs `c_overlap`, guard-on, published
40%-hidden anchor masks, 30 dev pieces × 4 seeds, paired per-piece bootstrap
CIs. Confirmation set untouched.

## Results

*(to be appended when `results/graphgp_overlap/` completes)*
