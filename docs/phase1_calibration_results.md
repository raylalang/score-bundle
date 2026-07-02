# Phase-1 calibration results — held-out ASAP imputation

First end-to-end evidence for the core thesis claim: a **score-graph residual prior on top
of a learned LM mean** improves both recovery *and* calibration over hand-built baselines on
real, held-out, contamination-filtered performances.

> **⚠ Correction (2026-07-02): read the [corrected results](#corrected-results-2026-07-02)
> first.** The original tables below have two problems that were found and fixed:
> (1) a **velocity-target leak** into the LM mean, which inflated the `v`-channel LM
> numbers, and (2) the known **EB noise-variance collapse**, now fixed by a noise floor.
> The corrected headline still supports the thesis claim — and with tighter significance —
> but with smaller (honest) margins for the LM mean. Details in the new section.

> **✔ Update (2026-07-03, branch `phase0-leakfree-restart`): the leak has a principled
> fix that is honest *and* recovers accuracy.** The score-only band-aid below removes the
> leak by corrupting the velocity input; the real fix is a **leak-free read-out** — read the
> per-note embedding at the pre-velocity (DURATION) token, which is causally blind to the
> note's own velocity, with no retraining. See
> [Leak-free read-out](#leak-free-read-out-2026-07-03). The `v`-channel LM RMSE goes
> 0.118 (band-aid) → **0.090** (leak-free), still far above the 0.057 leaky floor (i.e.
> genuinely honest), because the band-aid also destroyed *neighbour* velocity context that
> the leak-free read-out keeps.

## Corrected results (2026-07-02)

### Finding A — the LM mean leaked the velocity target

The per-note embedding `h_i` is read off the hidden state at each note's **VELOCITY
token**; a causal transformer's hidden state at position *t* includes the input token at
*t* itself, so `h_i` contains the note's own (bin-quantized) performed velocity and the
ridge head could partially *decode* the `v` target rather than predict it. Masking was
applied in y-space only, never in the LM's input, so this affected held-out notes too.
The tell: LM `v` RMSE 0.039 ≈ the 32-bin quantization width (≈0.031).

Fix: **score-only embeddings** — every note is tokenized with a constant placeholder
velocity (64), so `mu_LM` is a purely score-conditioned prior mean
(`scripts/extract_asap_arrays.py` caches both variants). On the same 50 eval pieces the
LM-mean `v` RMSE moves 0.057 → 0.112 (score-only), while `tau` (0.143 vs 0.144) and
`log r` (0.724 vs 0.723) are unchanged — exactly the leak signature. All corrected
numbers below use score-only embeddings. (Conditioning on *observed* notes' velocities
would be legitimate; mask-aware embeddings are future work — score-only is conservative.)

## Leak-free read-out (2026-07-03)

The score-only band-aid removes the leak by *corrupting the input* (constant velocity 64),
which also destroys the **neighbour** velocity context the model legitimately uses. The
principled fix keeps the real performed velocities in the input but **reads the per-note
embedding one token earlier**, at the pre-velocity (DURATION) token: in the fixed 4-token
group `[TIME_SHIFT, PITCH, DURATION, VELOCITY]`, that state is *causally blind* to the
note's own velocity, yet under next-token pretraining it is exactly the state trained to
predict it — so the fix needs **no retraining**. `lm/features.py:note_score_positions`,
selected via `note_embeddings(..., readout="pre_velocity")`; cached as `emb_leakfree`
(`scripts/extract_asap_arrays.py`, schema v2); proven invariant to the note's own velocity
by `tests/test_lm_leakage.py`.

**Per-note read-out A/B (40 head + 30 eval pieces, small LM, `v` channel):**

```
                     v RMSE (LM mean)        v RMSE (LM + graph)   v cov@.9   v NLL (LM+graph)
emb (leaky)             0.057  ← dishonest       —                    —          —
emb_scoreonly (band-aid) 0.118                   0.083               0.911      -1.070
emb_leakfree (fix)       0.090                   0.079               0.899      -1.127
```

The leak-free read-out is **honest** (0.090 sits far above the 0.057 leaky floor — it is
predicting, not decoding; the structural test guarantees no leak) **and recovers ~20% of
the accuracy the band-aid threw away** (0.118 → 0.090 for the mean; 0.083 → 0.079 with the
graph), with better `v` calibration (NLL −1.07 → −1.13). `tau`/`log r` are unchanged (they
never leaked).

**Pooled headline, leak-free (30 × 4, `--embeddings emb_leakfree --noise-floor-frac 0.05`,
`logs/robust_leakfree.log`):** `LM + graph` RMSE **0.3929** [0.354, 0.416], NLL **−0.332**
[−0.42, −0.24], cov 0.917 — the best cell, essentially matching the score-only pooled numbers
(the `v` gain is diluted by `tau`/`log r` in the pooled RMSE) but now honest by construction
and better on the channel that leaked. The graph advantage stays significant on both axes:

```
paired per-piece diff (negative = LM+graph better)      RMSE                  NLL
LM+graph vs LM mean-only     -0.0520 [-0.0684,-0.0362]*   -0.1490 [-0.1838,-0.1166]*
LM+graph vs ridge mean-only  -0.0281 [-0.0454,-0.0134]*   -0.1122 [-0.1575,-0.0665]*
LM+graph vs zero+graph       -0.0112 [-0.0172,-0.0051]*   -0.0233 [-0.0642,+0.0185]
```

`emb_leakfree` is now the default `--embeddings` in the eval scripts. (Stage 2 — a masked
score-conditioned pretraining objective that aligns the LM's training with this read-out and
adds bidirectional score context — is the next step on the restart branch.)

### Mask-aware check (Stage 1.5): does `emb_leakfree` still cheat via held-out neighbours?

Residual concern: the pre-velocity read-out is blind to the note's *own* velocity, but the
input stream still contains the **held-out neighbours'** true velocities, and their
information could reach a held-out note's mean directly (LM context) or indirectly (graph
coupling between held-out notes). The strict protocol recomputes embeddings **per mask**
with held-out notes' velocities replaced by the placeholder (64) — only *observed* notes'
velocities enter the input. `scripts/eval_asap_maskaware.py` runs the exact robust-eval
protocol (30 pieces × 4 seeds, identical masks, `noise_floor_frac=0.05`) three ways:
`LM-lf` = leak-free as published; `LM-ma` = mask-aware embeddings under the published head;
`LM-ma-mh` = mask-aware with a matched head (fit on mask-aware head-split embeddings).
Full log: `logs/maskaware.log`.

```
pooled (identical masks)   graph      RMSE      NLL   cov@.9  cal-err
LM-lf (published)          on       0.3928  -0.3357    0.918    0.066
LM-ma (strict)             on       0.3930  -0.3223    0.921    0.067
LM-ma-mh (strict+head)     on       0.3924  -0.3264    0.921    0.066

paired per-piece diff                              RMSE                       NLL
mask-aware vs leak-free   (graph on)   +0.0003 [-0.0003,+0.0008]   +0.0135 [+0.0075,+0.0195]*
mask-aware+mh vs leak-free (graph on)  -0.0003 [-0.0010,+0.0004]   +0.0093 [+0.0033,+0.0149]*
mask-aware vs leak-free   (mean only)  +0.0006 [-0.0006,+0.0018]   +0.0190 [+0.0054,+0.0341]*
graph on vs off           (mask-aware) -0.0523 [-0.0688,-0.0372]*  -0.1581 [-0.1937,-0.1259]*
```

**Verdict: the leak-free read-out is confirmed (essentially) clean.** Pooled RMSE is
statistically indistinguishable (+0.0003, CI spans 0); NLL shows a **tiny but significant**
gap (+0.013 pooled, ≈8% of the graph's own NLL margin of −0.158), i.e. a trace of
neighbour-velocity information does survive. Per-channel it is confined to `v`, as the leak
mechanics predict (mean-only `v` RMSE 0.0899 → 0.0989; graph-on 0.0779 → 0.0810; `tau` /
`log r` unchanged to 3 decimals). The matched head does not close the NLL gap, so it is
context information, not head mismatch. **Every qualitative conclusion survives the strict
protocol** — the graph's paired advantage under mask-aware embeddings is RMSE −0.0523* /
NLL −0.1581*, indistinguishable from the published leak-free contrasts.

**Honest headline going forward (strict, mask-aware): `LM + graph` pooled RMSE 0.3930,
NLL −0.322, cov 0.921** — quote these instead of the leak-free numbers when strictness
matters; they cost ~nothing. Note the deployment nuance: at *rendering* time (completion
from an excerpt) the mask-aware protocol is also the only feasible one, since unheard notes
have no velocities to feed; the published leak-free numbers correspond to the *transcription*
setting where the full performance is available as input and only the readout is masked —
both are legitimate, they answer different questions.

### The score-feature rival: does the LM mean earn its place? (2026-07-03)

The published baselines never included a *strong cheap* rival to `μ_LM`: the per-piece
ridge baseline is fit per piece on ~60% observed notes, while the LM head is fit
**cross-piece** on the 40 head pieces. A hand-built score-feature representation fit under
the *identical* head protocol answers the honest question: is a pretrained LM needed at
all, or would 25 score-derived features do? `rich_score_features()`
(`src/score_bundle/baselines.py`) builds per-note features (pitch level/z-score/local
deviation, contour, intervals, log-duration and local duration context, metrical phase
sin/cos at periods 1/2/4, IOIs, local note density, chord size/rank, piece position, edge
proximity, voice count), optionally lifted with 256 random Fourier features;
`scripts/eval_asap_feature_baseline.py` fits ridge heads on the head split (5-fold grouped
CV for each representation's l2), then runs the exact robust protocol (30 pieces × 4
seeds, identical masks, `noise_floor_frac=0.05`). Full log: `logs/feature_baseline.log`.

Head-split CV was a dead heat — LM 0.4344, feat-lin 0.4353, feat-rff 0.4346 — and CV
picked `l2=100` for the grid heads (this matters below).

```
pooled (identical masks)   graph      RMSE       NLL   cov@.9
zero                       on       0.4041   -0.3083    0.923
feat (rff)                 on       0.7473   11.3660    0.918   <- tau-contaminated, see below
LM                         on       0.9739    4.5583    0.916   <- tau-contaminated, see below
feat+LM (concat)           on       0.3872   -0.3478    0.917
feat (rff)                 off      0.4504   -0.1466    0.901
LM                         off      0.4473   -0.1953    0.894
feat+LM                    off      0.4394   -0.2077    0.895

paired per-piece diff                             RMSE                        NLL
LM vs feat        (mean only)   -0.0037 [-0.0100,+0.0023]    -0.0489 [-0.0782,-0.0185]*
feat+LM vs feat   (graph on)    -0.1153 [-0.3330,-0.0052]*  -11.9157 [-35.7150,-0.0053]*
feat+LM vs LM     (graph on)    -0.1588 [-0.4709,-0.0014]*   -5.0226 [-15.0747,+0.0163]
```

**Verdict — the honest reading.**

1. **On average error the LM mean does *not* beat hand-built score features.** Mean-only
   pooled RMSE is statistically indistinguishable (−0.0037, CI spans 0). The claim "a
   pretrained LM is required for a good prior mean" is not supported and must not be made.
2. **The LM's real, significant edge is calibration and dynamics.** Mean-only NLL is
   −0.049* in the LM's favour, and per-channel the gap is concentrated in `v` (mean-only
   RMSE 0.0890 vs 0.1098; graph-on 0.0771 vs 0.0802). `tau`/`log r` are feature-reachable;
   learned dynamics are where pretraining pays.
3. **They stack.** `feat+LM` graph-on is the best cell we have measured on this protocol —
   pooled 0.3872 / −0.348, beating the published `LM+graph` 0.3928 / −0.336 and
   significantly beating either input alone on RMSE. The two representations carry
   complementary information. (Candidate headline upgrade, but adopt only after a rerun
   under the published `l2=10` head and the strict mask-aware protocol.)
4. **The `feat`-on and `LM`-on pooled/tau rows above are contaminated by an EB failure
   mode, not a real regression** — see the caveat below. Uncontaminated comparisons:
   all mean-only rows, the `log r` / `v` per-channel rows, and `feat+LM` (whose fits did
   not collapse: `tau` graph-on 0.1570).

**New caveat — the EB `tau` fragility is head-l2-sensitive (single-cell catastrophe).**
The graph-on `tau` cells for `feat` (1.11 / 34.3) and `LM` (1.55 / 13.9) blew up in this
run while every published `l2=10` run had the same cell stable (~0.158).
`scripts/diag_tau_head_l2.py` isolates the cause on identical masks: with the published
`l2=10` head, pooled `tau` graph-on is 0.1585 / −0.814 with 4/120 mildly-elevated cells
(all the same hard piece, RMSE 0.57–0.63, all seeds); with the CV-selected `l2=100` head
the *same* 4 cells persist unchanged **plus one single catastrophic EB fit collapse
(seed 2, piece 28, `tau` RMSE 17.0)** that alone drives the pooled cell to 1.58 / +14.5.
So the mechanism is the documented bad-mean-on-`tau` EB fragility; the head's ridge
strength changes *exposure* to the catastrophic regime (a slightly different prior mean on
one piece tips one fit past what `noise_floor_frac=0.05` can catch, and the graph
propagates it). Consequences: (a) pooled graph-on numbers are not robust to a single
collapsed fit — report medians or per-cell screens alongside; (b) a guard is worth
implementing (clip the fitted prior mean to a multiple of the observed residual scale, or
fall back to zero-mean when the EB objective lands in the collapsed regime) — open
follow-up, protocol-changing, so not retrofitted into published numbers.

### Stage 2: masked, score-conditioned pretraining vs the Stage-1 read-out (2026-07-03)

Stage 2 replaces the causal next-token objective with the task we actually pose at
inference: a **bidirectional** transformer (`GPTConfig(causal=False)`, one extra `[MASK]`
embedding) trained to predict each hidden note's velocity bin from all notes' score
tokens plus the *observed* notes' velocities, with a per-window observed fraction
`ρ ~ U(0.1, 0.9)` and CE at masked positions only (`src/score_bundle/lm/masked.py`,
`scripts/train_lm_masked.py`). Two budgets, both `d=512/L=6` (= `maestro_scaled`
geometry): **15×500 steps** (budget-matched to Stage 1; best val masked-CE@0.6 **2.532**)
and **45×500** (3×; **2.484**); chance = log 32 = 3.466. A/B:
`scripts/eval_asap_stage2.py`, logs `logs/eval_stage2.log` / `logs/eval_stage2_3x.log`.

**Read-out lesson (a leak we caught by its signature).** The first A/B run read observed
notes at their real velocity token; bidirectionally that state *contains the note's own
velocity*, so the ridge head part-decoded its target on observed rows and the EB noise
fit (which uses observed residuals) collapsed — v coverage 0.55, and the direct-velocity
variant (which returned the observation itself at observed notes → residuals exactly
zero → `noise_floor_frac × 0 = 0`) hit NLL ~10⁹ (`logs/eval_stage2_naive_readout.log`).
Fix: **leave-one-out read-out** (`masked_note_embeddings_loo`) — every note, observed or
hidden, is read at a `[MASK]`ed own-velocity position, conditioning on the *other*
observed notes; `direct_velocity_mean` now returns the model expectation for all notes.
Structural test pins the observed-note invariance. Moral: *any* readout fit on observed
rows needs the Stage-1 guarantee (own target never in the readout), and the EB fit is
the canary.

**Budget-matched result (corrected, 30 pieces × 4 seeds, identical masks):**

```
pooled                     graph      RMSE       NLL   cov@.9
LM-s1 (leak-free, pub.)    on       0.3928   -0.3371    0.919
LM-s2 (LOO, strict)        on       0.4028   -0.3245    0.923
LM-s2d (direct v)          on       0.4028   -0.3282    0.923

paired per-piece diff                        RMSE                       NLL
LM-s2 vs LM-s1   (graph on)    +0.0108 [+0.0048,+0.0171]*   +0.0126 [-0.0045,+0.0281]
LM-s2 vs LM-s1   (mean only)   +0.0260 [+0.0144,+0.0379]*   +0.0385 [+0.0083,+0.0684]*
LM-s2d vs LM-s2  (graph on)    -0.0001 [-0.0001,-0.0000]*   -0.0037 [-0.0058,-0.0013]*
graph on vs off  (LM-s2)       -0.0673 [-0.0877,-0.0491]*   -0.1797 [-0.2140,-0.1485]*
```

**Verdict — the aligned objective does not beat the Stage-1 read-out at matched
budget.** Pooled RMSE is slightly but significantly worse (+0.011*), NLL a wash — and
note the comparison is strict-vs-non-strict: LM-s2 is mask-aware *by construction*,
so its fair Stage-1 reference is the strict headline (0.3930 / −0.322), against which
Stage-2's NLL −0.3245 is a tie. Per-channel: Stage-2 is *better on τ* (0.1569 vs
0.1585 graph-on, mean-only −0.754 vs −0.693 NLL), worse on `log r` (0.675 vs 0.657) and
`v` RMSE (0.0788 vs 0.0776) though with better v coverage (0.916 vs 0.903). The direct
velocity prediction edges out the ridge head (NLL −0.0037*) — the amortized model is at
least as good as a head on its own embeddings, with no head to fit. The 3× budget run
improves every channel's *mean-only* cell over the 15-epoch run (pooled 0.4653 vs
0.4715; v 0.0990 vs 0.1004; τ 0.1618) but stays behind Stage-1 pooled (+0.0202*
mean-only), and its graph-on rows are contaminated by the EB τ collapse below.

Interpretation, honestly: the training/read-out mismatch we set out to fix was real,
but fixing it bought *cleanliness*, not accuracy — the causal pre-velocity state was
already a strong summary of expressive context, and bidirectional conditioning on ~60%
observed velocities adds little the graph does not already recover (the graph's own
margin on Stage-2 means, −0.067*/−0.180*, matches its margin on Stage-1 means).
`emb_leakfree` (Stage 1) **stays the published default**; Stage 2 stands as (a) the only
read-out that satisfies the strict protocol by construction, (b) a direct amortized
conditional `p(v_hidden | score, v_obs)` needing no ridge head, and (c) the negative
result: objective alignment ≠ better prior mean at matched compute.

**The EB τ collapse recurs — under the published `l2=10` head this time.** The 45-epoch
model's graph-on τ cell exploded (0.4185 / +0.398 pooled) while its mean-only τ was fine
(0.1618, *better* than Stage-1). A per-cell screen (`logs/diag_tau_s2x3.log`) confirms
the mechanism: **one catastrophic cell (seed 3, piece 28, τ RMSE 4.51) accounts for the
entire pooled excess** — the other 119 cells pool to ~0.157 with only the chronic piece 7
mildly elevated (0.57–0.62, all seeds). And it is the **same piece 28** that collapsed
under the l2=100 head (there: seed 2, RMSE 17.0). So the collapse is not specific to the
head's ridge strength or the embedding family: piece 28's EB τ fit sits on a knife edge,
and which mask seed tips it depends on small changes in the prior mean. The EB guard
(previous subsection) is now supported by evidence at two heads and two embedding
families, always localized to the same piece.

### Finding B — the EB noise floor fixes the collapse

`fit_laplacian_field(..., noise_floor=...)` clamps `noise_var` inside the EB objective
(evals use `noise_floor_frac=0.05`, i.e. 5% of the observed residual variance). This is
the fit-side counterpart of the predictive-variance floor of finding #1 below.

### Corrected headline (30 eval pieces × 4 mask seeds, score-only embeddings)

`scripts/eval_asap_robust.py --embeddings emb_scoreonly [--noise-floor-frac 0.05]`,
logs in `logs/robust_scoreonly{,_floor}.log`:

```
                             no floor                        noise_floor_frac = 0.05
mean   graph      RMSE [95% CI]        NLL [95% CI]       RMSE [95% CI]        NLL [95% CI]      cov@.9
zero   off      0.5664 [0.508,0.598]  -0.007 [-0.12,0.12]   0.5664 [0.508,0.598]  -0.007 [-0.12,0.12]  0.869
zero   graph    0.4038 [0.364,0.424]  -0.307 [-0.41,-0.20]  0.4041 [0.364,0.424]  -0.308 [-0.41,-0.21]  0.923
ridge  off      0.4233 [0.377,0.448]  -0.219 [-0.32,-0.12]  0.4233 [0.377,0.448]  -0.219 [-0.32,-0.12]  0.917
ridge  graph    2.6450 [0.370,2.017]  130.41 [-0.38,388.3]  0.9566 [0.360,0.847]   4.614 [-0.40,14.89]  0.918
LM     off      0.4505 [0.407,0.474]  -0.105 [-0.20,-0.00]  0.4505 [0.407,0.474]  -0.105 [-0.20,-0.00]  0.894
LM     graph    0.4065 [0.360,0.433]   0.076 [-0.40, 0.91]  0.3939 [0.356,0.417]  -0.314 [-0.41,-0.22]  0.922
```

With the floor, **`LM + graph` is the best cell on both axes and the paired bootstrap is
now significant on both**:

```
paired per-piece diff (negative = LM+graph better)      RMSE                NLL
LM+graph vs LM mean-only     -0.0551 [-0.0736,-0.0377]*   -0.2083 [-0.2483,-0.1714]*
LM+graph vs ridge mean-only  -0.0267 [-0.0443,-0.0113]*   -0.0943 [-0.1378,-0.0544]*
LM+graph vs zero+graph       -0.0098 [-0.0164,-0.0034]*   -0.0055 [-0.0437,+0.0271]
```

Honest reading: the graph residual is doing most of the calibration work (zero+graph is
nearly as good on NLL); the (score-only) LM mean adds a small but significant RMSE gain.
The large LM-vs-zero dynamics gap in the original table was mostly the leak. The
`ridge + graph` τ blow-up (finding #2 below) is tamed but not cured by the floor —
it is a bad-mean artefact, not a fit artefact.

### Corrected per-channel breakdown (score-only + floor; `logs/channelwise_floor.log`)

```
[tau]                                    [log r]                                  [v]
mean   graph  RMSE    NLL   cov         mean   graph  RMSE    NLL   cov         mean   graph  RMSE    NLL   cov
zero   off   0.163 -0.796 0.946         zero   off   0.960  1.495 0.784         zero   off   0.119 -0.719 0.877
zero   on    0.156 -0.800 0.945         zero   on    0.678  0.975 0.908         zero   on    0.080 -1.099 0.917
ridge  off   0.157 -0.785 0.948         ridge  off   0.709  1.036 0.911         ridge  off   0.101 -0.906 0.892
ridge  on    1.874 21.693 0.927         ridge  on    0.652  0.932 0.909         ridge  on    0.081 -1.116 0.909
LM     off   0.167 -0.709 0.927         LM     off   0.753  1.123 0.890         LM     off   0.118 -0.729 0.864
LM     on    0.158 -0.819 0.944         LM     on    0.659  0.946 0.911         LM     on    0.083 -1.070 0.911
```

The channel story survives, with honest magnitudes: the graph fixes articulation
coverage (0.78 → 0.91) and roughly halves velocity RMSE for *every* mean; on the
corrected `v` channel the LM mean no longer dominates (LM+graph 0.083 ≈ zero+graph
0.080 ≈ ridge+graph 0.081 — the old 0.039 was the leak). τ is near the warp noise floor,
as before.

### Per-channel variance rescaling (upgrade 2)

A conformal-style per-(cell, channel) std scale, fit on the **head** split
(`metrics.std_rescale_factor`, `scripts/eval_asap_channelwise.py --var-rescale`), applied
to the eval predictions. For `LM + graph`, pooled:

```
                 coverage@0.9   cal-err   NLL
before rescale       0.922       0.067   -0.314
after  rescale       0.899       0.048   -0.234
```

Coverage lands on the 0.90 target and the PIT calibration error improves, at a modest
NLL cost (the τ scales ≈ 0.75 shrink intervals that the KS/coverage metrics reward but
the log-score does not). Report it as an optional post-hoc step: use it when nominal
coverage is the contract, skip it when log-score is.

### The LM-size ablation does not survive the leak fix

Re-running the identical corrected eval with the big model's score-only embeddings
(`logs/robust_scoreonly_floor_big.log`):

```
                 LM+graph pooled (corrected)
LM (val ppl)     RMSE                  NLL                  cov@.9
small (10.85)    0.3939 [0.356,0.417]  -0.314 [-0.41,-0.22]  0.922
big   (9.66)     0.3945 [0.357,0.417]  -0.313 [-0.40,-0.23]  0.921
```

Identical within noise. The gain reported in the original ablation below ("concentrates
in velocity/dynamics") was the leak: a lower-perplexity model reconstructs its own
(quantized) velocity input token more faithfully, which looked like a better prior mean.
**With an honest score-conditioned mean, scaling the LM 1.3× did not improve the
downstream prior at this scale** — a real finding for the thesis (the structured residual,
not LM capacity, is what carries the result), and a caution for the "scale with
transcribed corpora" plan: measure downstream, not perplexity.

## Setup

- **Prior mean source `μ`** ∈ {`zero`, `ridge` (hand-built score-feature ridge), `LM`
  (`μ_LM` from the MAESTRO-pretrained MusicGPT per-note embeddings)}.
- **Graph residual** ∈ {`off`, `on`}: `off` predicts the mean with a homoscedastic residual
  std (isolates the mean); `on` runs the closed-form GMRF posterior centered on that mean
  (`y − μ ~ N(0, Q_G⁻¹)`), with per-piece empirical-Bayes `(λ, η, noise_var)`.
- **Targets** `y = [τ, log r, v]` (onset residual, articulation, centred velocity) from the
  ASAP beat-grid warp.
- **Data hygiene.** 20 held-out ASAP pieces, **piece-disjoint** from the 40-piece head-fit
  split; **contamination-filtered** — any ASAP performance whose MAESTRO twin was in Phase-0
  pretraining is dropped (1036 → 653 aligned performances), so the LM never saw the eval
  performance. LM: `checkpoints/maestro_scaled/best.pt`, val perplexity 10.85.
- Reproduce: `python scripts/eval_asap_calibration.py --asap-root ../data/asap-dataset
  --maestro-root ../data/maestro-v3.0.0 --checkpoint checkpoints/maestro_scaled/best.pt`.

## Pooled result (over τ / log r / v)

```
mean source    graph      RMSE      NLL   cov@.9  cal-err
zero           off      0.5507   0.0007    0.869    0.187
zero           on       0.3782  -0.3551    0.924    0.089
ridge          off      0.3898  -0.2468    0.919    0.084
ridge          on       0.5187   0.9280    0.907    0.079
LM             off      0.4101  -0.3320    0.881    0.060
LM             on       0.3731  -0.6042    0.926    0.080   ← best RMSE & best NLL
```

Lower RMSE / NLL / cal-err is better; coverage closer to 0.90 is better.

**`LM + graph` is the best cell on both recovery (RMSE 0.373) and calibration (NLL −0.604,
coverage 0.926).** The graph residual improves calibration over no-graph for the zero and LM
means (coverage 0.87/0.88 → 0.92/0.93, NLL down). This is the headline: the learned LM prior
mean plus a structured graph residual beats every baseline on both axes.

## Per-channel breakdown

```
[tau]                                    [log r]                                  [v]
mean        graph  RMSE    NLL   cov     mean        graph  RMSE    NLL   cov     mean        graph  RMSE    NLL   cov
zero        off   0.185 -0.872 0.947     zero        off   0.927  1.517 0.790     zero        off   0.126 -0.642 0.868
zero        on    0.175 -0.941 0.948     zero        on    0.625  0.922 0.916     zero        on    0.086 -1.047 0.909
ridge       off   0.175 -0.900 0.950     ridge       off   0.643  0.986 0.925     ridge       off   0.108 -0.827 0.882
ridge       on    0.656  2.951 0.899     ridge       on    0.608  0.889 0.918     ridge       on    0.086 -1.056 0.902
LM          off   0.186 -0.729 0.934     LM          off   0.683  1.045 0.911     LM          off   0.060 -1.312 0.799
LM          on    0.177 -0.840 0.946     LM          on    0.620  0.913 0.917     LM          on    0.039 -1.886 0.915
```

- **Dynamics (`v`) — the clean win.** `LM + graph` reaches RMSE 0.039 / NLL −1.89 /
  coverage 0.915. The LM embedding predicts velocity far better than zero or ridge
  (LM-off 0.060 vs zero-off 0.126), and the graph adds calibrated structure on top.
- **Articulation (`log r`) — the graph fixes calibration.** A zero mean badly undercovers
  (0.79); the graph residual lifts every mean to ~0.92 coverage. Point error converges to
  RMSE ~0.61 across means.
- **Timing (`τ`) — residuals are tiny** (RMSE ~0.18) and already near the noise floor; the
  graph barely moves it and it slightly over-covers (~0.95). Not where the graph earns its keep.

## Two findings worth carrying into the writeup

1. **Predictive-variance floor is necessary.** The GMRF posterior was wildly overconfident
   (NLL ~1e22, undercoverage) until the held-out predictive std included the observation
   noise: a held-out *observation* `y = f + ε` has variance `diag(Σ_y) + noise_var`, and the
   latent `diag(Σ_y)` collapses toward 0 at well-pinned nodes. Fixed in
   `imputation_eval._predict_channel`; pinned by
   `test_graph_predictive_std_includes_observation_noise`.
2. **The graph helps a good mean and can hurt a bad one.** `ridge + graph` *hurts* pooled
   RMSE (0.39 → 0.52), entirely via the `τ` channel (ridge-on τ: RMSE 0.656, NLL 2.95): the
   ridge baseline extrapolates timing poorly and the graph propagates that bad mean. This is
   a baseline-quality artefact, not a defect of the graph prior — worth a footnote.

## LM-size ablation (does a better LM mean → a better prior?)

A second MusicGPT was pretrained on MAESTRO at ~1.3× the size (d=512, L=8, 25.6M params,
val perplexity **9.66** vs the small model's **10.85**; curves in
`figures/lm_training_curve{,_big}.png`) and run through the **identical** held-out eval
(`logs/eval_big.log`). The downstream `LM + graph` cell improves modestly but consistently:

```
                LM+graph  pooled            [v] (dynamics)
LM (val ppl)    RMSE      NLL    cov     RMSE     NLL
small (10.85)   0.3731   -0.604  0.926   0.039   -1.89
big   (9.66)    0.3700   -0.647  0.930   0.032   -2.04
```

The gain concentrates exactly where the learned mean carries the most signal — **velocity/
dynamics** (RMSE −18%, NLL −0.15) — while timing/articulation are unchanged. So "better LM →
better `μ_LM`" holds, but its leverage is channel-specific. (The `ridge + graph` τ blowup is
*worse* in this run, NLL 16.7 on τ — same baseline-quality artefact as finding #2, unrelated
to LM size.)

## Robustness (bootstrap CIs + a numerical caveat)

`scripts/eval_asap_robust.py` re-runs the comparison over **30 eval pieces × 4 mask seeds**
with **percentile-bootstrap 95% CIs over pieces** and a paired bootstrap, and writes
`figures/reliability_diagram.png` + `figures/pit_histogram.png`. It surfaced an important
**numerical fragility**, not in the method but in the empirical-Bayes fit:

- **Per-piece, the headline holds cleanly.** A single-seed per-piece diagnostic
  (`scripts/diag_robust_blowup.py`, `logs/diag_blowup.log`) shows `LM + graph` beats
  `LM`-mean-only on **every** piece — pooled RMSE 0.388 vs 0.442, NLL −0.544 vs −0.235,
  **median NLL −0.583, worst-piece NLL 0.46** (no blowup).
- **But the pooled *mean* NLL is not robust.** Across the full 4-seed run, a *minority of
  held-out mask realizations* drive the in-sample marginal-likelihood EB fit to degenerate
  hyperparameters (tiny `noise_var` → overconfident posterior), spiking that cell's NLL as
  high as ~20. Pooled-mean NLL then reads +6.9 (CI `[-0.65, +20.4]` — the heavy right tail is
  the tell), even though the **median** cell is healthy. RMSE, being bounded by the data
  scale, does **not** blow up (mean 0.388).

**Fix (next step, not yet applied):** floor `noise_var` inside `fit_laplacian_field`'s
objective (the same principle as the predictive-variance floor in finding #1), and/or report
median/trimmed NLL. The calibration-split variant (`fit_laplacian_field_calib`) is *not* a
reliable cure here (its single-seed mean NLL −0.413 is slightly worse than marglik's −0.544).

## Open follow-ups

- **Robustify the EB fit:** `noise_var` floor (or a weak prior) inside
  `fit_laplacian_field`, then re-run `eval_asap_robust.py` for a clean CI table; report
  median/trimmed NLL alongside the mean.
- Calibration-split hyperparameters, and a per-channel temperature/variance-scaling baseline,
  to push coverage to exactly 0.90.
- The `τ` channel is near the noise floor; consider whether the warp resolution limits it.
- Aria frozen-feature upper-bound baseline (Step 4) is stubbed — aria is not installed here.
