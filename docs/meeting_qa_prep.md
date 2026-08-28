# Meeting prep (2026-08, private) — post-confirmation edition

**How to use this document.** To study: read top to bottom, then say
Part 1 aloud twice. In the room: the pocket card is your glance-sheet,
Part 1 is the spine you speak, Part 2 is what you reach for when asked.
Part 3 is background — study it, never present it.
**Pages are the PRINTED ones (footer / ToC); your PDF viewer's counter
runs 2 ahead.** Before anything else: **sync Overleaf with `main`**.

---

## Pocket card

**Pages:** phase table 5 · §3.9 17–26 · data-point Fig 3.2 = 20 ·
**drift ¶ + Table 3.3 + Fig 3.3 = 20–21** · tonal Fig 3.4 = 23 ·
truth/estimate/GP Fig 3.5 = 24 · results 24–26 (headline + ledger 25 ·
dev Table 3.4 = 25 · confirmation Table 3.5 = 26) · **Phase-3 study
§3.10.1 = 27–28, its Table 3.6 = 28, its Fig 3.7 = 30** · channels
Fig 3.6 = 29 · §5.3 = 35 · Phase-2 architecture Fig 8.1 = 50.

**Eight numbers** (CONFIRMATION, as-given, paired vs no-graph, one-shot):
intonation −0.88\* (dev basis −0.89\* — reproduced) · vibrato
calibration −3.0\*/−0.56\* (the seed-sensitive extent star HELD) ·
coverage 0.88–0.91 (all six) · timing calibration −0.03 ns = **C4
failed** · adverse cell: timing recovery +0.003\* (≈3 ms) · drift:
10 cents/note, 97% GT sign agreement, graph-white (lag-1 +0.03 vs τ's
+0.59) · tonal (dev, exploratory) −0.21\*/−0.05\* · waveform (dev): 2.3 cents
from truth with NO tracker, beats the estimator on winds, self-check
coverage 0.90 exactly · Phase-1 confirmed 0.376 vs 0.393\*.

**The meeting = Beat 4 → Beat 5** (drift → Phase 3). The confirmation
is TO THE SIDE: one breath + Table 3.5 (p. 26) only if it comes up.
Asks: (1) next registration: waveform-integration channel or tonal
metric (tonal needs his corpus sign-off; power check says
calibration-primary); (2) only if the confirmation is discussed: how
prominent should the failed timing claim be?

---

# Part 1 — The story you tell (Beat 4 → Beat 5; confirmation to the side)

- The meeting = Beat 4, then Beat 5. Start directly on Beat 4 — it
  answers HIS comment, so it's the natural opening.
- If he wants orientation first: ONE sentence — "score + performance in,
  per-note expressive variables with honest error bars out, one GP on
  the score graph; Phase 1 confirmed it on piano" (phase table p. 5) —
  then into Beat 4.
- **Opening line (memorize):** "Last time you said the sine model might
  be too simple — that loudness and even intonation shift over time.
  I didn't argue; I measured it. And the measurement ended up opening
  Phase 3."
- The confirmation: mention in passing where it supports a point
  ("...which is why calibration held at confirmation"), and give the
  one-breath version below ONLY if he asks. Don't open with it.

---

### TO THE SIDE — the confirmation (deploy only if it comes up)
*(confirmation Table 3.5 p. 26 · headline + ledger p. 25 · full detail: Part 2
"The confirmation" block)*

**One breath:** "I registered four claims before touching the held-back
pool, then spent it — once, protocol frozen under a git tag. Three
confirmed: intonation recovery reproduced the dev number almost exactly
(−0.88\*), vibrato calibration passed on both channels — including the
extent star I'd flagged as the one that could fail — and coverage sat at
0.88–0.91 on all six. The fourth, timing calibration, failed — CI
includes zero — and it's in the table verbatim, next to a small starred
cell against us."

**If he wants one more level:**

- the pool: 13 pieces / 40 unique tracks, untouched since the
  data-blind split of Aug 6
- decision rule fixed in advance: C1 ∧ C2 ∧ C3 ⇒ headline confirmed —
  and it was
- trust line: "the rule was allowed to say no; it said no once"
- why timing failed, three hypotheses → Part 2

---

### BEAT 4 — The drift study: his comment, answered
*(¶ "What the per-note compression discards" + Table 3.3 + Fig 3.3,
pp. 20–21 — the figure IS the beat: A = the centre moving, B = two
witnesses, C vs D = graph-white vs the structure the graph uses)*

**Headline:** "You were right — I measured it — and the missing structure
is exactly the part a graph across notes cannot help with."

**What "drift" is — the concrete picture (Fig 3.3 A — have it open):**

- our sine model assumes a note has ONE pitch centre c, held flat for
  the whole note, with vibrato wiggling around it
- drift = the centre itself sliding while the note is held
- example: a note that starts 5 cents flat and ends 5 cents sharp has
  10 cents of drift — and the model reports c ≈ 0, a precise number for
  a pitch the player never actually held
- his comment was exactly this: "loudness and even intonation can shift
  over time" — the model pretends they don't, within a note

**What I actually did (the experiment, step by step):**

- take every note with identifiable vibrato that exists in both curves:
  n = 5,145
- refit each note's sine model with ONE extra ingredient: a straight
  tilt under the vibrato (the drift term) — nothing else changes
- do it on TWO pitch curves independently: our tracker's output AND the
  human-corrected ground-truth curve
- why two: if drift showed up only in the tracker's curve it could be
  measurement noise; if two independent measurements of the same note
  show the same tilt, it's the player

**Finding 1 — the drift is real music:**

- the tilt is statistically significant (slope > 2× its own SE) on
  about two thirds of notes — in EACH curve separately (65.5% tracker,
  67.1% ground truth)
- the two curves agree on the tilt's DIRECTION on **97%** of notes;
  rank correlation **0.91**
- two independent witnesses, same story → the player really slides

**Finding 2 — it's big relative to our precision:**

- median total drift across a note: **10.5 cents** (top tenth: >30)
- the c cell's reported precision: **0.9 cents**
- so: c is a precisely-measured AVERAGE of a moving quantity — precise
  about the average, silent about the movement

**Finding 3 — loudness (his other example) moves even more, but look
at the direction:**

- same check via the four chunk levels: median within-note change ≈
  **1.4×** the note-to-note spread of the loudness channel — so yes,
  bigger than what we model
- but 65% of the significant slopes FALL — and 81% in brass: a note
  naturally decays after its attack; that is the instrument's envelope,
  not the player's phrasing
- strings are ~50/50 rising/falling — that's where genuine swells live

**Finding 4 — the decisive one (Fig 3.3 C vs D): could OUR model even
use this?**

- what the graph prior does, in one line: it lets neighbouring notes
  share information — so it can only help with quantities that are
  CORRELATED between neighbours
- the test: does note i's drift predict note i+1's drift? (lag-1
  autocorrelation along the piece)
- answer: **no** — +0.03 for intonation drift, +0.06 for loudness
  slope; each note's drift is its own private event
- the contrast that makes it vivid: timing sits at **+0.59** —
  neighbours strongly share timing, and that is exactly the channel
  where the graph earns its keep
- conclusion: a "drift channel" would be data the graph cannot smooth,
  denoise, or fill in — dead weight in THIS model

**Why the confirmed results survive his criticism (all three):**

- the drift is already PRICED IN: a drifting note fits the sine model
  worse → its residual is bigger → its cells carry bigger error bars →
  the GP trusts it less → that is WHY coverage stayed honest at
  confirmation
- the contrasts are fair: the "truth" we score against was made with
  the same flat-centre ruler, so graph-vs-no-graph compares like with
  like
- the resolution argument: structure that lives WITHIN notes and is
  uncorrelated ACROSS notes belongs to a frame-level model — which is
  literally Phase 3's likelihood (next beat)

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
*(§3.10.1 pp. 27–28 · Table 3.6 p. 28 · Fig 3.7 p. 30)*

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

**Result 1 — accuracy (Fig 3.7 panel A):**

- the ladder, median cents from ground-truth centre:
    - constant c: 2.81
    - + AR(1) noise: 2.73
    - + drift term (Beat 4's!): 2.40
    - + deviation prior: **2.29**
- reference: full pyin+NLLS estimator chain = **2.01**
- winds: waveform **2.67 vs estimator 3.27** → "the waveform BEATS the
  tracker chain where trackers struggle"
- the drift term helps here too: paired −0.18 cents, better on 57%

**Result 2 — the calibration diagnosis (Fig 3.7 panel B):**

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

### HIS TWO KERNEL POINTERS (discuss in the meeting; papers found)

**What he most likely meant:**

- "periodic kernel Gaussian process" → the classic periodic
  (exp-sine-squared) kernel, MacKay 1998 / Rasmussen–Williams §4.3;
  in practice the QUASI-periodic form (periodic × Matérn decay), which
  lets amplitude and phase drift
- "generalized spectral mixture" → the spectral-mixture (SM) kernel of
  Wilson & Adams (ICML 2013) — kernel = learned Gaussian mixture over
  frequencies — and its non-stationary extension literally NAMED the
  generalized spectral mixture (GSM) kernel: Remes, Heinonen & Kaski,
  "Non-Stationary Spectral Kernels" (NeurIPS 2017) — frequencies and
  lengthscales become input-dependent functions
- closest to OUR domain: Alvarado & Stowell — GPs for music audio with
  quasi-periodic component × amplitude envelope (arXiv 1606.01039) and
  Matérn spectral mixture harmonic priors for pitch detection (arXiv
  1705.07104)

**How they fit our problem (the mapping to say):**

- both pointers name the SAME move: replace the parametric sine vibrato
  model with a GP PRIOR on the within-note pitch curve
    - kernel = quasi-periodic (vibrato) + smooth trend (the drift term)
    - our constant-c + fixed sine = the degenerate limit of that prior
- the GSM kernel is precisely the "sine too simple" fix: input-dependent
  frequency/amplitude = vibrato rate and extent that drift within the
  note — the structure the drift study measured
- and we already built the slot it plugs into: the Phase-3 deviation
  prior IS a GP on curve deviations with a crude bump kernel; his
  kernels are the principled family for exactly that slot — Gaussian,
  so the collapsed waveform likelihood machinery carries over
- honest boundary to state: within-note curves are where this lives;
  ACROSS notes the drift study says the structure is white, so the
  graph prior's role is untouched
- secondary connections if he goes there: our graph kernel g(ν) is
  already spectral (on the Laplacian spectrum) — a spectral MIXTURE
  there would generalize the additive/Matérn family; and multi-output
  SM kernels (Parra & Tobar, NeurIPS 2017; multi-task GSM, Chen et al.)
  generalize our ICM coupling B

**The line to close it:** "So your two pointers are the right family for
the two slots we just measured as open — the estimator's sine model and
the waveform likelihood's curve prior. I'd fold them into the Phase-3
design rather than re-open the frozen Phase-2 estimator."

---

### CLOSE — the asks

- from Beat 5 (the main ask): "Next registration — the
  waveform-integration channel, or the tonal metric first?"
    - tonal: needs his corpus sign-off (pool is spent; draft proposes
    Bach10 + reused pool, disclosed; power check ⇒ calibration-primary)
    - integration: feasibility evidence is in; same discipline applies
- only if the confirmation was discussed: "The failed timing claim —
  how prominent in the thesis narrative? I lean visible: it's the
  honesty exhibit that makes the three passes credible."

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
  posteriors in them (Fig 3.5's case B shows truth agreeing when it
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
- *Isn't the sine model too simple? (his own point)* Measured — ¶ +
  Table 3.3 + Fig 3.3, pp. 20–21:
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
  accuracy, coverage unmoved). That's Fig 3.7's panel B.
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

**How to study (~90 min, Beat 4 → Beat 5):** Overleaf sync → read the
drift ¶ + Table 3.3 + Fig 3.3 (pp. 20–21) → say Beat 4 aloud twice
(the opening line + Finding 4 a third time, with Fig 3.3 C/D in view) →
read §3.10.1 (pp. 27–28) with Fig 3.7 (p. 30) → say
Beat 5 aloud, punchlines once more → say the confirmation one-breath
aloud once (that's all it needs) → the asks → from Part 2, say aloud:
"why overconfident", "what's the Phase-3 plan", "what does confirmed
mean", "why did timing fail".

**If a question stumps you:** "that's measured — let me follow up with
the exact number." Everything here has a file behind it.
