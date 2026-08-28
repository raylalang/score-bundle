# Meeting prep (2026-08, private) — post-confirmation edition

**How to use this document.** To study: read top to bottom, then say
Part 1 aloud twice. In the room: the pocket card is your glance-sheet,
Part 1 is the spine you speak, Part 2 is what you reach for when asked.
Part 3 is background — study it, never present it.
**Pages are the PRINTED ones (footer / ToC); your PDF viewer's counter
runs 2 ahead.** Before anything else: **sync Overleaf with `main`**.

---

## Pocket card

**Pages:** phase table 5 · §3.9 17–25 · data-point Fig 3.2 = 20 ·
tonal Fig 3.3 = 22 · truth/estimate/GP Fig 3.4 = 23 · results 23–25
(headline + ledger 24 · dev Table 3.3 AND confirmation Table 3.4 both
on 25) · drift paragraph 20 · **Phase-3 study §3.10 = 26–28, its
Table 3.5 = 27, its Fig 3.6 = 30** · channels Fig 3.5 = 29 · §5.3 = 35
· Phase-2 architecture Fig 8.1 = 50.

**Eight numbers** (CONFIRMATION, as-given, paired vs no-graph, one-shot):
intonation −0.88\* (dev basis −0.89\* — reproduced) · vibrato
calibration −3.0\*/−0.56\* (the seed-sensitive extent star HELD) ·
coverage 0.88–0.91 (all six) · timing calibration −0.03 ns = **C4
failed** · adverse cell: timing recovery +0.003\* (≈3 ms) · drift:
10 cents/note, 97% GT sign agreement, graph-white (lag-1 +0.03 vs τ's
+0.59) · tonal (dev, exploratory) −0.21\*/−0.05\* · waveform (dev): 2.3 cents
from truth with NO tracker, beats the estimator on winds, self-check
coverage 0.90 exactly · Phase-1 confirmed 0.376 vs 0.393\*.

**The meeting = Beats 3, 4, 5 ONLY** (verdict → drift → Phase 3), then
two asks: (1) how prominent should the failed timing claim be — honesty
exhibit or one line?; (2) next registration: waveform-integration
channel or tonal metric (tonal needs his corpus sign-off; power check
says calibration-primary).

---

# Part 1 — The story you tell (THREE beats, exclusively)

- The meeting = Beats 3, 4, 5. Nothing else.
- If he wants orientation first: ONE sentence — "score + performance in,
  per-note expressive variables with honest error bars out, one GP on the
  score graph; Phase 1 confirmed it on piano" (phase table p. 5) — then
  straight to Beat 3.
- **Opening line (memorize):** "Three results since we spoke: I spent the
  confirmation and it passed; I measured your comment about the sine
  model; and Phase 3 is open, with its first results. In that order?"

---

### BEAT 3 — The confirmation verdict
*(Table 3.4 p. 25 · headline + ledger p. 24)*

**Headline:** "Three of the four registered claims confirmed; the fourth
failed and is reported verbatim."

**Setup — say briefly:**
- claims + protocol frozen on the 17th: git tag, one shot, every number
  reported whatever it says
- pool: 13 pieces, untouched since the split was frozen in early August
- 40 unique tracks (after recording dedup) · 2 seeds · 30% of notes hidden
- run: ~1.5 h, sharded; estimator code byte-identical to the tag

**The verdict — claim by claim:**
- C1 intonation recovery: **−0.877 [−1.236, −0.538]\*** → PASS
  - dev basis was −0.891\* → reproduced almost exactly
- C2 vibrato calibration, BOTH channels required:
  - extent: **−2.990 [−6.575, −0.435]\*** → PASS
  - rate: **−0.564 [−0.739, −0.392]\*** → PASS
  - say: "the extent half was the one I flagged as seed-sensitive — the
    claim I said could honestly fail. It held."
- C3 coverage: **0.88–0.91 on all six channels** (band was [0.85, 0.95])
  → PASS
- decision rule (fixed in advance): C1 ∧ C2 ∧ C3 ⇒ **headline CONFIRMED**
  - the headline: "the unchanged graph prior extends to real audio with
    calibrated uncertainty, at confirmation level"

**The honest part — volunteer, don't wait:**
- C4 timing calibration (secondary, non-gating): **FAILED**
  - −0.030 [−0.061, +0.006] — right sign, CI includes zero
  - dev basis was −0.292\*
- starred cell AGAINST the graph: timing recovery **+0.003 [+0.000,
  +0.007]\*** (≈3 ms on a 65 ms RMSE)
- timing summary sentence: "coverage on target (0.91), but no confirmed
  graph win on timing"
- dev's adverse loudness cell (+0.042\*) did NOT replicate: +0.015 ns
  → adverse cells move in both directions too

**If he asks why trust the rest:**
- "The rule was written before the data and was allowed to say no.
  It said no once."

**If he pushes on why timing failed:**
- the pool warped BETTER than dev (40/40 tracks, 15 exact, 0 failures;
  dev had 2 failures) → the alignment-error structure the dev star sat
  on isn't there
- τ's noise row is diagonal: carries scale, not correlation along score
  time (stated limitation; Future Work names the fix: correlated row)
- heavy-tailed calibration stars were known to move — extent was the
  flagged one (held), τ's wasn't (didn't)

---

### BEAT 4 — The drift study: his comment, answered
*(¶ "What the per-note compression discards", p. 20)*

**Headline:** "You were right — I measured it — and the missing structure
is exactly the part a graph across notes cannot help with."

**Setup:**
- his comment: the sine model may be too simple; loudness and intonation
  shift over time within a note
- response: refit every identifiable dev note with an added linear drift
  term — on the tracker curve AND the ground-truth curve, independently
- n = 5,145 notes measurable in both curves

**The drift is real music, not tracker noise:**
- significant slope in each curve independently: 65.5% (tracker) /
  67.1% (ground truth) of notes
- direction agreement between the two curves: **97%**
- rank correlation: **0.91**

**How big:**
- median drift across a note: **10.5 cents** (top tenth: >30)
- our c cell's reported precision: **0.9 cents**
- → c is a precise average of a moving quantity

**Loudness moves even more — but it's mostly envelope:**
- median within-note change ≈ **1.4×** the across-note spread of ℓ
- 65% of significant slopes are FALLING (brass 81%) → decay envelope,
  not expression
- strings nearly balanced (57/43) → where real swells live

**The decisive measurement — the slopes are graph-white:**
- lag-1 autocorrelation along the note sequence: **+0.03** (intonation),
  **+0.06** (loudness)
- compare **+0.59** for timing — the channel where the graph earns its
  keep

**Why this defends the model (all three):**
- mismatch is priced in: cell variances are residual-based → drifting
  notes report more uncertainty → why calibration held at confirmation
- quasi-truth uses the same ruler → graph contrasts compare like with
  like
- the discarded structure is within-note and graph-white across notes →
  a drift channel is one the graph could not help → per-note resolution
  is the right level for THIS model
- where it belongs: Phase 3's frame-level likelihood

**Punchline (memorize):** "Your comment, measured, turned into the
argument for the next phase — the last beat."

**If he pushes:**
- changed the estimator? → no; frozen under the tag; the confirmation
  ran it as registered; the drift study is the reason NOT to add a
  channel
- extent growth within notes? → mild: 2nd half larger on 53%, median
  +0.27 cents; mostly absorbed by the delay gate

---

### BEAT 5 — Phase 3 is open, with results
*(§3.10.1 pp. 26–28 · Table 3.5 p. 27 · Fig 3.6 p. 30)*

**Headline:** "The audio itself as the observation — no tracker — gets
within ~2 cents of ground truth; I can prove the machinery is calibrated;
and feeding it back into the bundle already improves inference on every
test pair."

**Setup:**
- premise: stop deriving targets through a tracker; the waveform IS the
  observation
- likelihood: harmonic basis, amplitudes marginalized in closed form,
  pitch curve = the same Phase-2 channel variables
- study: **376 notes · 7 tracks · all 3 families** · grid over the
  intonation centre, centred on the SCORE → exact 1-D inference
- four position models compared

**Result 1 — accuracy (Fig 3.6 panel A):**
- the ladder, median cents from ground-truth centre:
  - constant c: 2.81
  - + AR(1) noise: 2.73
  - + drift term (Beat 4's!): 2.40
  - + deviation prior: **2.29**
- reference: full pyin+NLLS estimator chain = **2.01**
- winds: waveform **2.67 vs estimator 3.27** → "the waveform BEATS the
  tracker chain where trackers struggle"
- the drift term helps here too: paired −0.18 cents, better on 57%

**Result 2 — the calibration diagnosis (Fig 3.6 panel B):**
- every variant: coverage **1–3%** at nominal 90% vs ground truth —
  INVARIANT under all three fixes
- accuracy climbs every time; coverage never moves
- the resolving experiment: model-true synthetic notes → coverage
  **exactly 0.90** (median |z| 0.53) → the machinery IS calibrated
- ⇒ the gap is an **estimand gap**: harmonic-model c vs NLLS-on-GT c =
  two different functionals of the same performance, ~2 cents apart
- no within-model fix can close a between-question gap → why all three
  failed
- (AR(1) even made it worse: misfit is in-band at the harmonics, out of
  a stationary floor's reach)

**Result 3 — the integration (the closer):**
- design: the waveform posterior enters the bundle like every estimator
  — a measurement with an honest noise row = posterior variance + a
  discrepancy floor at the ~2-cent scale
- measured, as a 7th channel (coupling learned per piece):
  - floor self-calibrates at **3.1 cents** from visible notes only —
    the estimand-gap scale, found automatically
  - held-out intonation improves on **14 of 14** track-seed pairs
  - **−1.21 cents\*** · NLL **−0.24\*** · coverage held at 0.87
- the control: remove the floor → calibration vs truth degrades
  (+0.22\*), nothing improves → the floor is provably what keeps it
  honest

**Punchline (memorize):** "Phase 3 doesn't replace Phase 2's noise
discipline — it feeds it."

**Honest labels — say once:**
- all of Beat 5: development-level, exploratory, one corpus, intonation
  only, no claims registered
- any claim gets its own registration — same discipline as always

**If he pushes:**
- why grid inference? → 1-D, exact to a 0.03-cent step; no Laplace
  claimed; the joint loop over all notes = the named open work
- vibrato scaffold is from the estimator? → yes, on identifiable notes,
  recorded as the one estimator ingredient; the c inference never sees
  the tracker

---

### CLOSE — the two asks (both arise from the beats)

- from Beat 3: "The failed timing claim — how prominent in the thesis
  narrative? I lean visible: it's the honesty exhibit that makes the
  three passes credible."
- from Beat 5: "Next registration — the waveform-integration channel, or
  the tonal metric first?"
  - tonal: needs his corpus sign-off (pool is spent; draft proposes
    Bach10 + reused pool, disclosed; power check ⇒ calibration-primary)
  - integration: feasibility evidence is in; same discipline applies

---

# Part 2 — When he asks

**Terms, one line each (blank-out insurance):**
*as-given* = the estimator's own variances used directly as observation
noise · *cell / cell mask* = one (note, channel) entry / which entries
are observed · *MCAR vs MNAR* = missing by coin flip vs missing because
of the value (eval mask is the first, estimator cells the second) ·
*gated fit* = the eq:vibrato-exact estimator, flat until the delay
(the ungated fit's "delay" is only a phase) · *quasi-truth* = same
estimator run on ground-truth pitch; independent of the tracker, not the
estimator · *development/confirmation* = validation/test, renamed for
the one-shot discipline · *one-shot* = the pool is spent whatever the
outcome; no reruns, no added seeds.

**The confirmation**
- *What does "confirmed" mean exactly?* The claims and the decision rule
  were committed (git tag) before any confirmation piece was touched;
  the pool was then evaluated once, under the frozen protocol, and the
  pre-stated rule — C1, C2, C3 all pass — was met. It is confirmation in
  the preregistration sense: the result was predicted, not found.
- *Why did timing fail?* Three honest hypotheses, stated in the thesis's
  order: the development star (−0.29\*) sat partly on alignment-error
  structure the pool doesn't share (the pool warped *better*: all 40
  tracks, 15 exact — dev had failures and wider residuals); the τ noise
  row is diagonal, carrying scale but not correlation along score time
  — the stated limitation; and dev calibration stars on heavy-tailed
  channels were already known to move (the extent star was the flagged
  one — it held; τ's didn't). Coverage on τ stayed at 0.91, so the
  posterior is honest there — it just isn't better than no-graph.
- *A starred cell against you?* Yes, timing recovery +0.003\* — about
  three milliseconds against a 65 ms RMSE. Reported next to C4's
  failure. Notably the development set's adverse cell (loudness NLL
  +0.04\*) did NOT replicate at confirmation (+0.015 ns) — adverse
  cells move too, in both directions.
- *Could you have re-run with more seeds?* No — the registered rule is
  one shot, no added seeds. That is what makes the three passes worth
  something.
- *Brass?* At confirmation, brass is the *strongest* intonation family
  (−1.74\*), where on dev it was the ns one — family-level splits are
  noisy at n=13 instruments; the bundle-level claims are the stable
  ones.

**Data and method**
- *What is the system, in one breath?* Score plus noisy per-note
  measurements in; one graph-structured Gaussian posterior over every
  (note, channel) cell out, with honest error bars. The graph encodes
  one belief — expressive behavior varies smoothly across neighboring
  notes — and the per-piece evidence decides how much graph, features,
  coupling, and noise. Evaluation: hide 30% of notes, predict them,
  score recovery AND calibration.
- *The figures show MISSING cells — mistake?* Deliberate, and the
  point: short notes physically cannot contain 1.5 vibrato cycles, so
  those cells are structurally unmeasurable — the missing entries are
  the problem statement, and the model's job is to put calibrated
  posteriors in them (Fig 3.4's case B shows truth agreeing when it
  does).
- *Which instruments?* URMP's thirteen, three families: strings
  (vn/va/vc/db), woodwind (fl/ob/cl/bn/sax), brass (tpt/hn/tbn/tba);
  violin dominates the dev side; no piano, no voice.
- *Targets are estimator outputs — what does recovery mean?* Exactly
  that, and the thesis says so wherever numbers appear: agreement with
  the estimator, weaker in kind than Phase 1. The quasi-truth
  cross-check gives the same ordering — and at confirmation as-given
  again beat the learned scale on quasi-truth calibration (0.80/0.85 vs
  0.72/0.73), closing that design question at confirmation level too.
- *Missing cells aren't missing at random.* Correct; the draft names
  the mechanism (Rubin). Eval mask MCAR by construction; estimator
  cells informative (missing when vibrato is short or weak). Held-out
  scores are interpretable for the vibrato-identifiable sub-population;
  filled cells are prior extrapolations.
- *Isn't the sine model too simple? (his own point)* Measured, ¶ p. 20:
  drift real (10 cents/note median, 97% direction agreement with GT),
  loudness moves even more (137% of the channel's across-note spread,
  mostly decay envelope — 81% of brass slopes fall). Priced into the
  residual-based cell variances — which is why calibration held at
  confirmation. And graph-white across notes, so it belongs to Phase
  3's frame-level likelihood, not to a new GP channel.
- *Is loudness comparable to velocity?* Not assumed — ℓ is what the
  microphone heard (log RMS, ≈8.7 dB per unit, per-track centred), v is
  what the finger did. Same role in the bundle, different physical
  quantity.
- *Why as-given noise, not learned?* Measured three times now: the
  pilot preferred it, the learned scale collapses with octave-failure
  cells present, and at confirmation as-given was again the
  better-calibrated variant vs quasi-truth.
- *Where does alignment error go?* Into τ's noise row — the tempo
  line's predictive variance. Diagonal noise carries scale, not
  correlation along score time; that stated limitation is the leading
  explanation for C4's failure.

**Process and what's next**
- *Can you resynthesize audio from the Phase-2 output?* The expressive
  skeleton, yes — pitch curve, per-note levels, timing around a
  supplied tempo line; a parametric synthesizer could render a
  caricature today. The audio itself, no: envelopes, timbre, noise are
  the waveform layer — Phase 3's likelihood, the one block the
  architecture marks as changing. In Phase 2 the model never sees audio
  (a fixed estimator reduces it first); in Phase 3 the waveform is the
  observation.
- *How does the waveform study work, in one breath?* Cut one note's
  audio at the annotated boundaries; the pitch curve is the Phase-2
  channel model; harmonic amplitudes (8 harmonics × 4 chunks) are
  marginalized exactly; slide the intonation centre over a grid and read
  the likelihood — exact 1-D inference, no tracker anywhere.
- *Why are the waveform posteriors overconfident?* Provably not a bug:
  on model-true synthetics the same machinery covers at exactly 90%.
  The overconfidence against ground truth is an estimand gap — the
  harmonic model's c and the NLLS estimator's c are different
  functionals of the same performance, ~2 cents apart — and we showed
  no within-model fix closes it (better mean model: accuracy improves,
  coverage doesn't; colored noise: worse; deviation prior: best
  accuracy, coverage unmoved). That's Fig 3.6's panel B.
- *So what's the Phase-3 plan?* The near-term design is already
  measured (dev, exploratory, `results/phase3_integration_dev.md`): the
  waveform posterior as a 7th bundle channel with its calibrated
  discrepancy floor improves held-out intonation on 14 of 14 track-seed
  pairs (−1.2 cents\*, NLL −0.24\*), and the no-floor control is
  significantly worse-calibrated against truth — the floor works.
  Frontier: a bridge model for the estimand gap, then the joint prior
  over all notes' position variables. Any claim needs its own
  registration.
- *Gaussian tails?* Known Phase-1 limitation. Student-t prototype
  exists; gated on its own future confirmation set.
- *Will you adopt the tonal metric?* Only through its own preregistered
  confirmation — and that needs a decision on the confirmation data
  (ask 2): URMP's pool is spent; reuse-with-disclosure or Bach10.
- *Why did it hurt piano but help intonation?* Expression travels
  register proximity; temperament travels the circle of fifths.
  Intonation is the first channel whose target IS pitch.

---

# Part 3 — Background depth (study only)

**The study ledger — what was actually done:**
1. *Tracker calibration.* pyin vs URMP ground truth before trusting it:
   2–5 cents median per instrument; confidence predicts errors →
   variances as-given + lowest quintile dropped.
   `results/tracker_calibration_dev.md`.
2. *Frozen split.* Composition-level, data-blind, unit-pinned — forced:
   arrangements share byte-identical recordings. `phase2/splits.py`.
3. *Evaluation grown 4→6 channels.* 77/78 unique dev tracks, 30%
   hidden, three systems, both axes + quasi-truth cross-check.
   `results/phase2_real_results.md`.
4. *τ adopted.* Feasibility first (76/78 tracks, 79 ms residual, lag-1
   +0.59), then the LOO tempo line with aligner σ in the noise row.
   `phase2/warp.py`.
5. *δ_vib decided.* Gated estimator built to match eq:vibrato exactly;
   criterion committed before numbers; IN, no claim.
   `results/delta_vib_dev.md`.
6. *Registration.* C1–C4 + decision rule frozen 2026-08-17 (tag), one
   dated erratum; guarded runner. `docs/phase2_prereg_design.md`.
7. *Fresh-seed robustness.* Recovery to two decimals; extent NLL star
   seed-sensitive → C2 risk named in advance.
   `results/phase2_seeds23_dev.md`.
8. *Circle of fifths.* Hypothesis pre-committed; helps intonation both
   axes, re-imposes the timing penalty. `results/phase2_tonal_dev.md`.
9. *Drift study (his comment).* Drift real in both curves (97% sign
   agreement, Spearman 0.91 — quote the robust statistics, the raw
   Pearson is tail-dominated); graph-white; no new channel; §3.9 ¶.
   `results/phase2_drift_dev.md`.
10. *CONFIRMATION SPENT (2026-08-27).* 40 unique tracks, 1h21m, all 40
    warped (15 exact/25 DTW/0 failed). C1 −0.877\*, C2 −2.990\*/−0.564\*,
    C3 0.88–0.91 → headline CONFIRMED; C4 −0.030 ns FAILED + adverse τ
    recovery +0.003\*. Evidence archived the moment it existed.
    `results/phase2_confirmation_results.md` (verdict section at the
    bottom).
11. *Phase-1 addendum* (§5.3 p. 35): posterior decomposes exactly by
    component — features carry the mean, the graph carries calibration;
    coupling earns its keep on velocity only.
12. *Full audit + math pass.* Every number re-verified against its log;
    three math errors fixed; terminology checked against the field.
13. *Phase 3 opened + scale study (2026-08-27..28).* Woodbury collapsed
    likelihood (unit-pinned); 376-note study, 4 position models:
    accuracy ladder 2.81→2.29 cents (estimator 2.01, waveform wins on
    winds), coverage invariant ~0.02 vs quasi-truth, self-check = 0.90
    exactly ⇒ estimand gap; discrepancy-floor integration designed.
    `results/phase3_waveform_dev.md`, thesis §3.10.1.

**How to study (~100 min, three beats only):** Overleaf sync → read the
confirmation paragraph + Tables 3.3/3.4 (pp. 24–25) → say Beat 3 aloud
twice, the claim-by-claim sequence a third time → read the drift ¶
(p. 20) → say Beat 4 aloud → read §3.10.1 (pp. 26–28) with Fig 3.6
(p. 30) → say Beat 5 aloud, the three punchline lines once more → the
two asks → from Part 2, say aloud: "why did timing fail", "what does
confirmed mean", "why overconfident", "what's the Phase-3 plan".

**If a question stumps you:** "that's measured — let me follow up with
the exact number." Everything here has a file behind it.
