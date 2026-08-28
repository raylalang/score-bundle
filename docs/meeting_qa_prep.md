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

The meeting discusses Beats 3, 4, 5 and nothing else. If he wants
orientation first, give ONE sentence — "score plus performance in, per-note
expressive variables with honest error bars out, one GP on the score graph;
Phase 1 confirmed it on piano" (phase table p. 5 if needed) — then go
straight to Beat 3.

**Open with the map (memorize this line):** "Three results since we spoke:
I spent the confirmation and it passed; I measured your comment about the
sine model; and Phase 3 is open, with its first results. In that order?"

---

### BEAT 3 — The confirmation verdict (Table 3.4, p. 25; headline + ledger p. 24)

**Headline to memorize:** *"Three of the four registered claims confirmed;
the fourth failed and is reported verbatim."*

**Setup (context you speak briefly):**
- On the 17th I froze four claims and the full protocol — git tag, one
  shot, every number reported whatever it says.
- The pool: 13 pieces held back since the split was frozen in early
  August, never touched by any decision. 40 unique tracks after
  deduplicating shared recordings, two mask seeds, 30% of notes hidden.
- The run itself: about an hour and a half, sharded; protocol verbatim
  (I diffed the estimator code against the tag first — byte-identical).

**The verdict, claim by claim (the spoken sequence):**
- C1, intonation recovery: **−0.877 [−1.236, −0.538], starred.** The
  development basis was −0.891 — it reproduced almost exactly.
- C2, vibrato calibration, BOTH channels required: extent **−2.990
  [−6.575, −0.435]***, rate **−0.564 [−0.739, −0.392]***. And say it:
  "the extent half was the one I had flagged as seed-sensitive — the
  claim I told you could honestly fail. It held."
- C3, coverage: **0.88–0.91 on all six channels**, inside the registered
  [0.85, 0.95] band.
- Decision rule, fixed in advance: C1 ∧ C2 ∧ C3 ⇒ **the Phase-2 headline
  is confirmed: the unchanged graph prior extends to real audio with
  calibrated uncertainty, at confirmation level.**

**The honest part (volunteer it, don't wait):**
- C4, timing calibration (secondary, does not gate the others):
  **FAILED.** −0.030 [−0.061, +0.006] — right sign, interval includes
  zero; the development basis was −0.292*.
- Plus a small starred cell *against* the graph: timing recovery +0.003
  [+0.000, +0.007]* — about 3 ms on a 65 ms RMSE.
- So the honest timing summary: coverage on target (0.91), but no
  confirmed graph win on the adopted timing channel.
- Also worth volunteering: the development set's one adverse cell
  (loudness NLL +0.042*) did NOT replicate at confirmation (+0.015, ns)
  — adverse cells move in both directions too.

**The line if he asks why trust the rest:**
- "The failure is precisely why the confirmations mean something — the
  rule was written before the data and it was allowed to say no. It said
  no once."

**If he pushes on why timing failed (full answer in Part 2):**
- Three hypotheses, in the thesis's order: the pool warped *better* than
  dev (all 40 tracks, 15 exact matches, zero failures — dev had 2
  failures), so the alignment-error structure the dev star partly sat on
  isn't there; the τ noise row is diagonal — carries scale, not
  correlation along score time (stated limitation, and the follow-up is
  named in Future Work: a correlated noise row); heavy-tailed
  calibration stars were already known to move (extent was the flagged
  one — it held; τ's didn't).

---

### BEAT 4 — The drift study: his comment, answered (¶ "What the per-note compression discards", p. 20)

**Headline to memorize:** *"You were right — I measured it — and the
missing structure is exactly the part a graph across notes cannot help
with."*

**Setup:**
- His comment: the sine vibrato model may be too simple — loudness and
  even intonation shift over time within a note.
- I didn't argue; I refit every identifiable development note with an
  added linear drift term — on our tracker's curve AND on the
  ground-truth curve, independently. 5,145 notes measurable in both.

**What the measurement says (spoken sequence):**
- The drift is real music, not tracker noise: about **two thirds of
  notes carry a significant slope in each curve independently** (65.5%
  tracker, 67.1% ground truth), the two curves agree on its direction
  **97%** of the time, rank correlation **0.91**.
- Size: **median 10.5 cents of drift across a note** (a tenth of it
  drifts more than 30). Our intonation cell's reported precision is 0.9
  cents — so c is a *precise average of a moving quantity*, an order of
  magnitude coarser as a "constant".
- Loudness moves even more: the median within-note change is about
  **1.4× the across-note spread** of the loudness channel — but 65% of
  the significant slopes are *falling* (81% in brass): that part is
  decay envelope, not expression. Strings are nearly balanced — that's
  where real swells live.
- The decisive second measurement: those within-note slopes carry
  **essentially zero correlation from note to note** — lag-1
  autocorrelation +0.03 (intonation) and +0.06 (loudness), against
  **+0.59 for timing**, the channel where the graph earns its keep.

**Why this defends the model rather than damaging it (say all three):**
- The mismatch is priced in: the cell variances are residual-based, so a
  drifting note reports itself as more uncertain — which is why
  calibration held at confirmation.
- The quasi-truth passes through the same parameterization — the
  graph-vs-no-graph contrasts compare like with like.
- The structure the sine model discards is *within* notes and
  graph-white *across* notes: a drift channel is one the graph prior
  could not help. So per-note resolution is the right level for THIS
  model — and the discarded part is precisely what Phase 3's
  frame-level likelihood exists to carry.

**The punchline (memorize):** *"Your comment, measured, turned into the
argument for the next phase — which is the last beat."*

**If he pushes:**
- "Did you change the estimator?" — No; it's frozen under the
  registration and the confirmation ran on it as registered. The drift
  study is the reason NOT to add a channel (graph-white), recorded in
  §3.9 with the numbers.
- "Vibrato extent growth?" — measured too, mild: second half larger on
  53% of long notes, median +0.27 cents; mostly absorbed by the delay
  gate.

---

### BEAT 5 — Phase 3 is open, with results (§3.10.1 pp. 26–28; Table 3.5 p. 27; Fig 3.6 p. 30)

**Headline to memorize:** *"The audio itself as the observation — no
tracker — gets within about 2 cents of ground truth; I can prove the
machinery is calibrated; and feeding it back into the Phase-2 bundle
already improves inference on every single test pair."*

**Setup:**
- Phase 3's premise: stop deriving per-note targets from a tracker; let
  the waveform be the observation. The likelihood: harmonic synthesis
  basis, amplitudes marginalized exactly (closed form), pitch curve
  parameterized by the same Phase-2 channel variables.
- Study: **376 notes, 7 tracks, all three families**; inference is a
  grid over the intonation centre, centred on the *score* — exact 1-D
  inference, nothing approximate; four position models compared.

**Result 1 — accuracy (Fig 3.6 panel A, spoken sequence):**
- Constant-c model: median 2.81 cents from the ground-truth-derived
  centre. Add AR(1) noise: 2.73. Add the drift term from Beat 4: 2.40.
  Add a marginalized deviation prior: **2.29**.
- Reference: the full pyin-plus-NLLS estimator chain sits at 2.01. So
  the waveform, with no tracker anywhere, is at near-parity — "and on
  the winds it is BETTER: 2.67 versus 3.27 — winds are where pitch
  trackers struggle, and the waveform doesn't care."
- Note the arc: the drift term from your comment improves the waveform
  model too (paired −0.18 cents, better on 57% of notes).

**Result 2 — the calibration diagnosis (Fig 3.6 panel B; the deep part):**
- Every variant's posterior is overconfident against ground truth —
  coverage 1–3% at nominal 90% — and it is INVARIANT: better mean model,
  colored noise floor, deviation prior — accuracy climbs every time,
  coverage never moves.
- The resolving experiment: on synthetic notes where the model is true
  by construction, the same machinery covers at **exactly 0.90** (median
  |z| 0.53). The inference is perfectly calibrated.
- Therefore the overconfidence is an **estimand gap**: the constant the
  harmonic model estimates and the constant the NLLS estimator defines
  on the ground-truth curve are *different functionals of the same
  performance*, about 2 cents apart. No within-model fix can close a
  between-question gap — which is why all three didn't. (AR(1) even
  made it worse — the misfit lives in-band, at the harmonics, where a
  stationary noise floor can't reach.)

**Result 3 — the integration (the closer):**
- The design conclusion: treat the waveform posterior the way Phase 2
  treats every estimator — a measurement with an honest noise row: its
  posterior variance PLUS a discrepancy floor at the measured ~2-cent
  scale.
- Measured: as a **7th bundle channel** (coupling learned per piece by
  the evidence; floor calibrated from visible notes only — it lands at
  3.1 cents, the estimand-gap scale, found automatically), held-out
  intonation improves on **14 of 14 track-seed pairs**: −1.21 cents
  starred, NLL −0.24 starred, coverage held at 0.87.
- And the control: remove the floor, and calibration against truth
  degrades significantly (+0.22*) while nothing improves — the floor is
  provably what keeps it honest.
- **The punchline (memorize):** *"Phase 3 doesn't replace Phase 2's
  noise discipline — it feeds it. Same as-given principle we just
  confirmed, one more measurement channel."*

**Honest labels (say once):** all of Beat 5 is development-level and
exploratory — one corpus, intonation only, no claims registered; any
claim gets its own registration, same discipline as always.

**If he pushes:**
- "Why grid inference?" — 1-D, exact up to a 0.03-cent step; no Laplace
  machinery claimed; the joint loop over all notes is the open work and
  is named as such.
- "Isn't the vibrato scaffold from the estimator?" — Yes, on
  identifiable notes, and it's recorded as the one estimator-supplied
  ingredient; the inference over c itself never sees the tracker.

---

### CLOSE — the two asks (both arise from the beats)

1. From Beat 3: "The failed timing claim — how prominent do you want it
   in the thesis narrative? I lean toward keeping it visible as the
   honesty exhibit; it's what makes the three passes credible."
2. From Beat 5: "The next registration: the waveform-integration channel,
   or the tonal metric first? For the tonal one the power check says
   calibration must be the primary claim and the corpus needs your
   sign-off (the pool is spent; the draft design proposes Bach10 plus
   the reused pool, disclosed). For the integration, the feasibility
   evidence is in and the same discipline applies."

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
