# The score bundle, taken seriously — a geometric direction

**Purpose.** Give the fibre-bundle framing a concrete, testable form, so it can be a real
research direction (and a clear one to discuss with the supervisor) rather than a metaphor —
*including* a built-in test for when it is only a metaphor. The load-bearing idea is small and
implementable: our score-graph prior is the **flat** special case of a **graph connection
Laplacian**, and "making the bundle non-trivial" means giving that connection curvature.

---

## 0. TL;DR

- A *performance* is a **section** of a bundle: each score event (base) carries a vector of
  continuous performance variables (fibre); a section picks one vector per note.
- What we have today is a **trivial** bundle (a product) with a Gaussian prior over sections —
  i.e. the connection is flat. That is a cylinder, not a Möbius strip.
- Three things could make it genuinely non-trivial *and* musical: **gauge freedom** (tempo,
  reference pitch, gain are only defined up to a choice), a **connection** (tempo already acts
  as parallel transport of the timing reference; micro-timing is the covariant deviation), and
  **holonomy/curvature** (does the expressive reference drift around a repeat?).
- The concrete model is a **graph connection Laplacian**: couple neighbouring notes by
  `‖x_i − A_ij x_j‖` with edge transforms `A_ij`. It **reduces to the current GMRF when every
  `A_ij = I`**, stays closed-form Gaussian, and so inherits all our inference machinery.
- **Litmus test** (keeps it honest): the geometry is doing work iff there is a measurable
  nonzero curvature/holonomy, or a gauge symmetry that demonstrably improves recovery/
  calibration. If neither, it is (beautiful) decoration — and saying so is itself a clean result.

---

## 1. What we already have is a (trivial) bundle

- **Base** `B`: the score, as a graph of note events.
- **Fibre** `F_i ≅ ℝ^k`: the performance variables on note `i` (Phase 1: `[τ, log r, v]`).
- **Section** `s: B → E`: one vector per note = one performance.
- **Our prior**: a Gaussian field over sections, `p(y) ∝ exp(−½ yᵀ Q y)` with
  `Q = λI + η L` and `L` the scalar graph Laplacian applied per channel.

In bundle terms this is the *trivial* bundle `B × ℝ^k` with a flat connection: the prior
penalizes `‖y_i − y_j‖` between neighbours, i.e. it compares fibres by the **identity** map.
The interesting content of "fibre bundle" — connection, curvature, twist — is exactly what a
product throws away.

## 2. Where non-triviality could be real and musical

### 2.1 Gauge freedom
Several variables are only meaningful *relative*, never absolute — the definition of a gauge:
- **Time:** absolute onset is meaningless; only deviation from a local reference (the tempo)
  matters. Choosing a tempo curve = choosing a gauge.
- **Pitch:** the reference (A=440 vs 442) is a free choice; only cents-relative-to-it matter.
- **Dynamics:** absolute loudness / recording gain is unrecoverable; only relative dynamics.

So there is a structure group `G` of fibrewise transformations (global time shift/scale, pitch
shift, gain) under which the *audible* content is invariant. **Design principle:** the model
should be invariant/equivariant to `G`. This is the most likely-to-help consequence — a clean
inductive bias (transposition-, tempo-, gain-equivariance).

### 2.2 The connection
A *connection* is a rule for comparing fibres at different base points (parallel transport).
**Tempo is already a connection on the time fibre**: it predicts the next onset, and
micro-timing `τ` is the *covariant* deviation from that prediction — not an absolute number.
We do this for tempo; the move is to do it for the whole prior.

### 2.3 The graph prior is the flat case of a connection Laplacian
Attach a fibre vector `x_i ∈ ℝ^d` to each note and a transform `A_ij ∈ G ⊆ GL(d)` to each edge
(with `A_ji = A_ij^{-1}`). Define the smoothness energy

```
E(x) = ½ Σ_{i~j} w_ij ‖x_i − A_ij x_j‖²  =  xᵀ 𝓛 x,
```

where `𝓛` is the **connection Laplacian** (a block matrix: diagonal blocks `d_i·I_d`,
off-diagonal blocks `−w_ij A_ij`). Use it as the prior precision `Q = λI + η𝓛`.

- **Reduces to today's model:** if every `A_ij = I_d`, then `𝓛 = L ⊗ I_d` and we recover the
  current per-channel GMRF exactly. Our model is the *flat connection*.
- **Stays tractable:** `Q` is still a sparse SPD block matrix, so the posterior is the same
  closed-form Gaussian and all Phase-1 inference/calibration carries over unchanged.
- **This is a standard object** (graph connection Laplacian / vector diffusion maps, Singer &
  Wu 2012; used for angular synchronization, cryo-EM, etc.) — so we are not inventing
  mathematics, only applying a real bundle construction to the score.

Choices of `A_ij` (from trivial to ambitious):
1. **Identity** — current model (baseline).
2. **Hand-designed by musical relation** — e.g. between a note and its transposed/octave
   partner in another voice, transport intonation/timing accordingly; a `U(1)` phase rotation
   to transport vibrato phase between notes.
3. **Learned (a "neural connection")** — `A_ij = exp(f_θ(h_i, h_j))` with `f_θ` outputting a
   skew-symmetric matrix (so `A_ij ∈ SO(d)`), `h` the Phase-0 LM embeddings. The connection is
   then learned from data while remaining a genuine group-valued connection.

### 2.4 Holonomy / curvature — a style descriptor
Transport the reference around a loop `γ` (a repeat, an ABA return) and you get the
**holonomy** `H(γ) = A_{i_1 i_2} A_{i_2 i_3} ⋯ A_{i_k i_1}`. Flat ⟺ `H(γ) = I` for all loops.
Musically, nonzero holonomy on the timing fibre is *systematic rubato drift* — a performer who
rushes and never quite gives the time back. **This is a novel, computable descriptor of
performance** that falls straight out of the geometry; it need not improve any prediction
metric to be interesting (it may distinguish performers/styles).

### 2.5 Circle fibres and real topology (most speculative)
Where a fibre is a **circle** rather than a line — pitch-class (mod octave), vibrato phase,
intonation mod octave — the bundle can be genuinely twisted (no global trivialization) with
integer invariants (winding / Chern-like numbers). This connects to existing geometric music
theory (Tymoczko's orbifolds of chords). Highest risk, best "wow", likely a side-chapter.

## 3. Recommended direction

- **Load-bearing (do this):** (i) enforce gauge invariance (transposition / global tempo /
  gain); (ii) replace the scalar graph prior with a **connection Laplacian** and test whether a
  non-trivial connection (hand-designed, then learned) beats the flat one on recovery and
  calibration. This is a small, closed-form change to the existing prior.
- **Exploratory (the "obsession-satisfying" part):** holonomy/curvature of the learned timing
  connection around repeats and phrase-returns as a stylistic descriptor; circle-fibre topology
  for intonation/vibrato. These give the framing teeth without betting the thesis on them.

## 4. Experiments / falsifiable claims

1. **Gauge invariance.** Check whether predictions are invariant under transposition / global
   tempo scaling / gain; if the LM features are not, build the invariance in and test if
   generalization improves.
2. **Connection vs flat.** Implement `‖x_i − A_ij x_j‖` coupling; compare flat (`A=I`),
   hand-designed `A_ij`, and learned `A_ij` on the same held-out ASAP imputation/calibration
   benchmark. Claim: a non-trivial connection lowers error and/or improves calibration.
3. **Holonomy descriptor.** Add repeat/return edges to the score graph; compute holonomy of the
   timing connection; test whether it correlates with measured rubato drift or separates
   performers.

## 5. Litmus test (is the geometry doing work, or decoration?)

The bundle earns its place iff **at least one** holds:
- a **measurable curvature/holonomy ≠ 0** that corresponds to something musical, or
- a **gauge symmetry or non-trivial connection that improves** recovery/calibration over the
  flat model.

If neither, the framing is narrative only — and reporting *that* (with the connection-Laplacian
experiment as evidence) is a legitimate, honest outcome.

## 6. Risks / honest caveats

- Much fibre-bundle language is **isomorphic re-description** of a Gaussian field on a product
  space. The discipline above (insist on curvature, or a symmetry that helps) is what prevents
  mysticism.
- Learned connections add parameters and identifiability concerns; keep `G` small (e.g.
  `SO(d)`) and regularize toward flat (`A_ij ≈ I`).
- The topological (circle-fibre) story is elegant but may not touch the metrics; treat as
  analysis, not as the core claim.

## 7. Implementation note

This is a contained addition to the existing repo: a `connection_laplacian(graph, transforms)`
builder that assembles the block precision `Q = λI + η𝓛`, fed straight into the current
`GraphGaussianField` (which already does closed-form inference for arbitrary SPD `Q`). The flat
case must reproduce today's numbers exactly — that is the first unit test.

## 8. Pointers

- Singer & Wu, *Vector diffusion maps and the connection Laplacian* (CPAM, 2012) — the discrete
  graph-bundle construction used here.
- Tymoczko, *The Geometry of Musical Chords* (Science, 2006) — orbifold/continuous-pitch geometry.
- Gauge-equivariant networks (e.g. Cohen et al.) — for the learned-connection / equivariance angle.
- Lindgren, Rue & Lindström (2011); Borovitskiy et al. (2021) — the (scalar) graph-SPDE priors we
  are generalizing (see the concept note, §16).
