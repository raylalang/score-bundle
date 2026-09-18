# Meeting prep (2026-09, private) — the measured-answers week

**How to use this document.** To study: the TALK MAP below with the deck
open is the formal skeleton — one row per deck segment, claim + number +
visual. Part 1 is the same content in spoken register; use it only to
find words, not to learn the structure. In the room: the pocket card is
your glance-sheet, the deck is the spine, Part 2 is what you reach for
when asked.
The RESERVE section (previous meetings' beats: drift study, Phase 3,
confirmation) is still valid background — reach for it only if asked.
**Pages are the PRINTED ones (footer / ToC); your PDF viewer's counter
runs 2 ahead.** Before anything else: **sync Overleaf with `main`**
(draft is now 71 pp; Future Work carries the new measured answers).

**Naming care (two different "AR(1)"s):** in the Phase-3 ladder, "AR(1)"
is a within-note NOISE model on the waveform residual (it barely helped,
2.73 cents). The NEW result is an AR(1) CORRELATION in the timing
channel's noise row ACROSS notes — different object, different level;
say "correlated timing noise across notes" in the room, never bare
"AR(1)".

---

## Pocket card

**Pages (draft updated 2026-09-16, 71 pp):** phase table 6 · Phase-2
chapter 30–39 (dev Table 7.1 = 36 · confirmation Table 7.2 = 37) ·
attribution §5.2 = 21 · Future Work (the measured answers live here)
42–46 · App E drift E.3 = 60, tonal E.4 = 61 · App F Phase-3 = 62–64.
Ledger doc: `results/exploration_week_2026-09.md`.

**The week in eight numbers** (all DEVELOPMENT, exploratory, published
tables unchanged):
timing-tail blow-up cell 105.2 → −1.7 under the t predictive (floors:
no effect) · articulation recovery −0.018\* from robust timing noise
(the coupling spillover) · correlated timing noise: NLL −0.111\*, RMSE
−20%\* (≈99→80 ms), correlation detected on 97–100% of cells, three
independent runs · completion worst cell 480,266 → 0.7 under
coverage-test + disagreement guard · SM: wins the curve (6.3 vs 9.6 c frame-level) but ×3 less
stable at producing the sine-defined scalars — swap closed · SM curve prior: no gain (median tie; the
adverse mean does not survive piece clustering) ·
learned filter: switched off on 90% of cells.

**Hardening (say it if probed on statistics):** every main star survives
a week-wide Benjamini–Hochberg pass over all 14 contrasts
(`logs/week_bh_recheck.log`); the swap verdict hardens (even intonation is
adverse per piece); the Slot-B "harm" does NOT survive piece clustering —
say "no gain", never "harm". **Power (the registration is drafted,
`docs/corrnoise_prereg_DRAFT.md`):** the timing-calibration claim stars
with probability ≥0.99 from 6 pieces on any pool option; recovery needs
~13–15 pieces and is secondary by design — so even Bach10-sized pools
work for the primary claim.

**FRAMING: the kernel direction, fully measured; the thesis's own needs,
answered.** Two weeks since the last meeting: open with the 60-second
orientation (Part 1 head; deck frames 2–13 are the thesis refresher +
recent-work overview). The deck is THE living deck now: thesis-shaped,
general-audience register — no claim codes (say "timing calibration",
not "C4") and no repo paths on slides. Then the verdicts + two asks.
Asks: (1) the corpus/pool decision — the correlated-timing-noise result
is registration-grade and answers the one failed confirmation claim, and
the tonal-metric registration waits on the same decision; (2) blessing
of the opt-in adoption defaults (nothing reported changes without its
own registration).

---

# TALK MAP — the meeting in six rows (study THIS, deck open)

| # | say (one sentence) | the number | the visual / math | deck |
|---|---|---|---|---|
| 0 Orientation | One model: a multi-output GP on the score graph; Phase 1 confirmed, Phase 2 confirmed with one failed secondary claim (timing calibration), Phase 3 scoped. | conf. 0.376 vs 0.393\* | model frames 4–7, Phase-2 method math 9–10, phase status 8/11/12 | 2–13 |
| 1 Kernels, slot one | The SM-GP wins the curve but is less stable at producing the sine-defined scalars our channels are; swap closed by the pre-committed rule. | 6.3 vs 9.6 cents; extent +1.686\*, rate +0.164\* | SM section: model 14, read-outs 15, test 16, win 17, loss 18, verdict 19 | 14–19 |
| 2 Kernels, slot two | At the waveform deviation prior: no gain at matched rank; a learned spectral filter switches itself off. | 2.25 vs 2.29 cents tie; off on 90% | — | 19, 28 |
| 3 Timing tail | A variance-matched Student-t predictive kills the blow-up (floors do nothing: tail-shape, not scale); the robust fit also repairs articulation via the coupling. | 105.2 → −1.7; artic. −0.018\* | t-predictive backup frame | 20, 25–26 |
| 4 C4 repair — MAIN | One correlated-noise parameter in the timing row, chosen by the evidence per piece, improves calibration AND cuts timing error 20%. | NLL −0.111\*, RMSE ≈99→80 ms, ρ>0 on 97–100% of cells, three runs | Σ equation (frame 21), mechanism figure (frame 22) | 21–23 |
| 5 Asks | Pool decision; opt-in blessing; kernels three paths (default: register now, curve-level channels eventually). | calibration power ≥0.99 from 6 pieces | power table (frame 23) | 23–24 |

### Reading the two SM figures (the formal version of verdict 1)

- **Explainer (frame 17).** A: one lively note, both models of its curve
  (the GP tracks it, the rigid sine cannot). B: the GP posterior split
  into centre + drift + vibrato components; the read-outs come from these
  REALIZED components, not from the kernel's process parameters (the
  estimand rule). C: the three numbers with intervals, two estimators.
- **Worst case (frame 18).** A and B are the SAME note measured twice
  (tracked vs ground-truth curve). The SM fits the curve better in BOTH
  panels — that is expected and is NOT the test. The estimator's product
  is not the curve; it is three numbers per note fed to the bundle, and
  those sit in the legends: sine 2.5 Hz on both curves (partly
  manufactured — that is its grid floor, where 13–18% of barely-vibrato
  notes sit), SM 2.0 Hz on one and 9.5 Hz on the other. The two headline
  lines above the panels state exactly this. Better curve in, different
  numbers out — the whole two-sided verdict in one picture.
- **If asked "but the blue fit looks better":** "Exactly — in both
  panels, and that is the measured half of the verdict. The other half is
  in the legends: the numbers the pipeline actually consumes flip between
  two measurements of the same note, and that instability survives
  restriction to strong-vibrato notes, so it is intrinsic to the
  translation into scalars, not an artifact of weak notes or of the
  rule."

---

# Part 1 — The story you tell

### ORIENTATION — 60 seconds, say FIRST (two weeks have passed)

"Quick orientation since it has been two weeks. The thesis is one model:
a multi output Gaussian process on the score graph. Score and performance
go in, per note expressive variables come out, each with a calibrated
error bar. Phase 1, piano, is confirmed by its preregistered one shot.
Phase 2 carried the same prior unchanged to strings and winds and is
confirmed by its own one shot, with one secondary claim failed and
reported verbatim: timing calibration. Phase 3, the audio likelihood, is
scoped with a first study measured. Last time we met was the reading
week on the kernel direction from our discussion. Since then, two
weeks: the estimator test we agreed on, then a measuring week. And the failed timing claim is exactly
where today's main result attaches."

*(Deck frames 2–12 carry the whole thesis refresher — problem, the four
model frames, the two Phase-2 method frames with the estimator math, one
status frame per phase — and frame 13 the recent-work overview; let them
do the work. If he waves it off, skip straight to the opening line.)*

- **Opening line (memorize):** "I spent the two weeks measuring instead
  of building new things. The kernel direction from our last discussion
  is now fully measured —
  both places they could enter our pipeline — and every limitation the
  thesis had documented about itself got converted into a measured
  answer."

### THE LADDER — four verdicts, in speaking order

**1. The kernel direction, slot one (the estimator).** The spectral-mixture GP
is the better model of the curve — measured frame-by-frame against ground
truth, it wins clearly (6.3 vs 9.6 cents median). What it cannot do is
impersonate the current estimator: our channels ARE the sine fit's own
parameters, and translating a richer posterior into that scalar
vocabulary is less stable than never leaving it (extent and rate
decisively, by the pre-committed rule). So the swap is off — the
confirmed bundle keeps its estimator — and the real finding is that the
model may have outgrown the interface.

**2. The kernel direction, slot two (the waveform curve prior).** The
curve-level win pointed at the Phase-3 deviation prior — measured there
too:
no gain at matched rank (2.25 vs 2.29 cents median, tie; the paired mean
significantly against, because a vibrato band pinned at a wrong scaffold
rate is worse than neutral bumps). And a LEARNED filter on the graph
spectrum: the evidence switches it off on 90% of pieces. So the kernels'
value to us is conceptual — curve-level scoring, a coherence bound, and
a read-out rule (estimate realized quantities, not process parameters) —
not numerical. The question is closed with measurements, not opinions.

**3. The timing tail (our known Phase-1 wart).** A variance-matched
Student-t predictive kills it completely — the documented blow-up cell
goes from NLL 105 to −1.7 with coverage intact — and the floors we had
suggested as the cheap alternative do nothing (it is a tail-shape
problem, not a scale problem). At inference level the robust fit also
improves ARTICULATION (−0.018\*): a timing outlier corrupts coupled
channels through the coregionalization; only robustifying the fit undoes
that.

**4. The failed confirmation claim (C4, timing calibration).** Its named
follow-up is measured and works: one AR(1) correlation parameter in the
timing noise row across notes — chosen by the evidence itself on
97–100% of pieces — improves calibration AND cuts timing error ~20%.
Hardened: joint refit and fresh mask seeds reproduce it digit-for-digit.
This is the registration candidate.

*(Not spoken, unless he asks: a third documented limitation — the
completion blow-up — is also closed, by the proposed coverage test plus a
disagreement guard; it is deploy-mode engineering with no claim attached.
Backup deck frame + `completion_fallback_dev.md` carry it.)*

### VERDICT 1, EXPANDED — the SM work, spoken (3 minutes; this is what
you told him you would do, so walk it before the verdict)

"What I built is exactly what I described in my note. The current
estimator fits one rigid shape to each note's cents curve: a constant
centre plus a phase-locked sinusoid, by least squares. The replacement
treats the curve as a draw from a Gaussian process whose kernel is the
spectral mixture: two damped-cosine components, one at the vibrato rate
with a finite coherence, one at zero frequency for the drift we measured.
The point of that kernel is that its parameters ARE the quantities we
want: the component frequency is the rate, its power carries the extent,
its coherence is a new phase-stability knob. Each note is fit by the same
exact marginal likelihood the rest of the model uses, over the note's
tracked frames — a grid over the frequency because the evidence is
multimodal, then a local optimization; uncertainties from the curvature
of the evidence, which is the GP's analogue of the delta method."

"Implementing it taught us three things before any comparison ran. The
process-level parameters are NOT what the channels mean. The ensemble
power understates this note's realized amplitude, so extent must be read
from the posterior vibrato component. The process constant and the slow
drift split a note-level offset arbitrarily, so the centre must be read
as the realized average. And the unconstrained evidence prefers an
incoherent band on real curves, so the vibrato component needs a
coherence bound to even BE a vibrato. Each read-out was redesigned
accordingly — that estimand rule is a durable result of this work."

"The test was committed before the run: both estimators on the tracked
curve AND on the ground-truth curve of the same note, each scored against
its own ground-truth output; five thousand development notes; parameter
accuracy as the pass/fail rule. Two results came out, and both matter.
At curve level the GP wins clearly — it is the better model of what the
player did. At parameter level it loses on stability — and here is the
honest subtlety: our channels are literally the sine model's parameters,
so the test asks both models to answer in the sine fit's vocabulary. The
GP answers by translating a posterior over curves into three scalars, and
that translation wobbles — we verified this is intrinsic, not a
population artifact, by re-testing on strongly-vibrated notes only. So
the committed rule correctly protects the confirmed bundle: no estimator
swap. But the scientific reading is not that the model failed — it is
that the model outgrew the scalar interface. The principled paths, if we
ever want them, are marginalizing the rate instead of point-picking it,
or moving the channels themselves to curve level, which is Phase-3
territory."

*(If he wants the math in front of him: hand over or screen-share
`docs/sm_estimator_note.pdf` — two pages, problem/model/fit/outputs/
limits/test, with the outcome banner. The SM section is now the deck's
spine: frames 14–16 carry the model, the read-out rules, and the test
design, 17–18 the two pictures, 19 the verdict.)*

### THE ASKS

- "The correlated-timing-noise result is registration-grade — three
  independent runs, both axes. To make it a claim I need a fresh pool:
  can we decide the corpus question? (The tonal-metric registration
  waits on the same decision.)"
- "Everything is adopted opt-in only — published paths are bit-unchanged,
  pinned by tests. Are you comfortable with these as the recommended
  defaults for new runs?"
- "The kernels: the spectral mixture won at curve level and lost at
  impersonating
  the sine fit's parameters. I see three paths and want your read:
  (a) marginalize the rate instead of point-picking it and retest the
  estimator; (b) accept that the scalar channels are the limitation and
  aim the kernels at curve-level channels inside Phase 3; (c) close it
  and spend the next cycle on the timing registration. My default is (c)
  now and (b) eventually, but this is genuinely your call."

### One-breath honesty line (if asked "so the papers helped?")

"The spectral mixture turned out to be the better model of the curve — we
measured that. They could not beat the incumbent at producing the
incumbent's own parameters, which is what the current pipeline consumes;
that swap is closed by a pre-committed rule. What they changed is how we
score curves, bound coherence, and read out realized quantities — and
they opened the question of whether the scalar interface itself is the
eventual limitation."


# RESERVE — measured results (deploy ONLY if Ray chooses to open them)
*(everything below is intact from the results-week prep: drift study →
Phase 3 → confirmation to the side)*

## (was Part 1) Beat 4 → Beat 5; confirmation to the side

- The meeting = Beat 4, then Beat 5. Start directly on Beat 4 — it
  answers HIS comment, so it's the natural opening.
- If he wants orientation first: ONE sentence — "score + performance in,
  per-note expressive variables with honest error bars out, one GP on
  the score graph; Phase 1 confirmed it on piano" (phase table p. 6) —
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
*(confirmation Table 7.2 p. 37 · dev ledger Table 7.1 p. 36 · full detail: Part 2
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
*(¶ "What the per-note compression discards" + Table E.1 + Fig E.1,
App E.3 p. 59 — the figure IS the beat: A = the centre moving, B = two
witnesses, C vs D = graph-white vs the structure the graph uses)*

**Headline:** "You were right — I measured it — and the missing structure
is exactly the part a graph across notes cannot help with."

**What "drift" is — the concrete picture (Fig E.1 A — have it open):**

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

**Finding 4 — the decisive one (Fig E.1 C vs D): could OUR model even
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
*(App F.1 pp. 61–63 · Table F.1 p. 62 · Fig F.1 p. 62)*

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

**Result 1 — accuracy (Fig F.1 panel A):**

- the ladder, median cents from ground-truth centre:
    - constant c: 2.81
    - + AR(1) noise: 2.73
    - + drift term (Beat 4's!): 2.40
    - + deviation prior: **2.29**
- reference: full pyin+NLLS estimator chain = **2.01**
- winds: waveform **2.67 vs estimator 3.27** → "the waveform BEATS the
  tracker chain where trackers struggle"
- the drift term helps here too: paired −0.18 cents, better on 57%

**Result 2 — the calibration diagnosis (Fig F.1 panel B):**

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

**The line to close it:** "So the kernel families are the right tools for
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

# Part 2 — When he asks (THIS meeting)

Each answer is 2 to 4 spoken sentences with its number and source.

**"But the GP fits the curve better — your own panel A shows it. How is
that a loss?"**
Two different questions. The GP wins at describing the curve — we
measured that too, frame by frame against the ground truth, and it wins
clearly. It loses at reproducing its three NUMBERS across two independent
measurements of the same note, because flexibility lets the decomposition
settle differently on each curve, while the rigid fit asks a narrower
question and gets the same answer twice. Our pipeline consumes the three
numbers, so reproducibility was the committed pass/fail axis. The
curve-level win is exactly what sent us to the waveform prior — where it
did not convert either. And we checked the population objection: even
restricted to notes with unambiguous vibrato, the reproducibility loss
stars on all three read-outs — the instability is intrinsic, a heavy
tail of decomposition flips, not an artifact of the test.
*(sm_estimator_dev.md M1 + metric caveat + M4 + stratified addendum)*

**"So my kernel suggestions did not help. Was the week wasted?"**
No — it did help, measurably: the spectral mixture is the better model of the
curve itself (6.3 vs 9.6 cents frame-level against ground truth). What
they could not do is impersonate the sine fit's own parameters, which is
what the current pipeline consumes, and each of those tests cost one
development study. And what the papers forced on us did
pay: curve-level scoring, the coherence bound, and the
realized-quantities read-out rule are now working parts of the pipeline.
The week's numerical wins came from our own documented limitations,
which the papers' concepts helped us fix. *(ledger; kernel review §0)*

**"These are development results. Why should I trust them?"**
Three reasons, in increasing strength: every main effect survives a
Benjamini Hochberg pass over all fourteen contrasts of the week; the
headline result is reproduced three independent ways (original masks,
fresh masks, joint refit); and we claim nothing from them --- turning the
timing result into a claim is exactly what the drafted registration is
for. *(logs/week_bh_recheck.log; corrnoise_tau_dev.md)*

**"You changed the pipeline after confirmation. Is that not scope creep?"**
Nothing published moved: the new capabilities are off by default, and
tests pin bit-equality of the default paths --- the same pattern as the
existing safeguard, which the thesis already documents. Any change to a
reported protocol needs its own registration; that rule is written into
each results doc. *(tests/test_adoption_optin.py; CLAUDE.md)*

**"Coverage went DOWN with your robust fit. How is that good?"**
It went from 0.948 to 0.919 at a nominal 0.90 --- from over-coverage
toward the target. The Gaussian fit was inflating the timing noise floor
to absorb outliers, which padded everyone's intervals; the robust fit
stops that, so intervals are honest rather than merely wide.
*(t_noise_em_dev.md)*

**"Explain the correlated timing noise in one breath."**
Timing targets are defined against an alignment warp, and warp error
drifts smoothly along the piece, so neighbouring notes share it. One
correlation parameter writes that down; the evidence then detects it on
97 to 100 percent of pieces, and prediction improves because a
neighbour's observed error now carries information about a held-out
note's error. *(corrnoise_tau_dev.md)*

**"New corpus or re-use the spent pool?"**
The power table says the calibration claim is safe either way (0.99 or
better from six pieces). A new corpus gives a pristine one-shot; re-use
is honest but only replication-grade, since those pieces are no longer
untouched by any decision. My preference is register on a new corpus and
report the spent pool as a labeled replication --- and the tonal
registration can share whatever pool you choose. *(corrnoise prereg
draft, corpus options)*

**"What happened to the generalized spectral mixture?"**
Parked with its gate stated: it was queued behind the plain spectral
mixture passing, which did not happen. It remains the design record for
the one thing plain SM cannot express, the vibrato onset delay.
*(kernel review §4)*

**"The articulation improvement from robust timing noise --- will you use it?"**
It is recorded and implemented, not deployed: adoption is opt in, and
promoting it into reported numbers would need a registration. Its value
today is diagnostic --- it proves observed outliers in one channel
corrupt the coupled channels, which is a property of the
coregionalization worth knowing. *(t_noise_em_dev.md)*

**"Could the timing fix have rescued C4 retroactively?"**
No, and we did not try: the confirmation pool is spent, and its one-shot
discipline covers evaluations, not only fits --- we never rescored it.
C4 stays failed as reported; the fix earns its own registration or
nothing. *(corrnoise_tau_dev.md, scope note)*

---

# Part 2b — Background Q&A from the previous meetings (study only)

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
  posteriors in them (Fig 4.1's case B shows truth agreeing when it
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
  Table E.1 + Fig E.1, App E.3 p. 59:
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
  accuracy, coverage unmoved). That's Fig F.1's panel B.
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
   Pearson is tail-dominated); graph-white; no new channel; Chapter-7 ¶.
   `results/phase2_drift_dev.md`.
10. *CONFIRMATION SPENT (2026-08-27).* 40 unique tracks, 1h21m, all 40
    warped (15 exact/25 DTW/0 failed). C1 −0.877\*, C2 −2.990\*/−0.564\*,
    C3 0.88–0.91 → headline CONFIRMED; C4 −0.030 ns FAILED + adverse τ
    recovery +0.003\*. Evidence archived the moment it existed.
    `results/phase2_confirmation_results.md` (verdict section at the
    bottom).
11. *Phase-1 addendum* (§5.2 p. 21): posterior decomposes exactly by
    component — features carry the mean, the graph carries calibration;
    coupling earns its keep on velocity only.
12. *Full audit + math pass.* Every number re-verified against its log;
    three math errors fixed; terminology checked against the field.
13. *Phase 3 opened + scale study (2026-08-27..28).* Woodbury collapsed
    likelihood (unit-pinned); 376-note study, 4 position models:
    accuracy ladder 2.81→2.29 cents (estimator 2.01, waveform wins on
    winds), coverage invariant ~0.02 vs quasi-truth, self-check = 0.90
    exactly ⇒ estimand gap; discrepancy-floor integration designed.
    `results/phase3_waveform_dev.md`, thesis App F.1.

**How to study (~90 min, Beat 4 → Beat 5):** Overleaf sync → read the
drift ¶ + Table E.1 + Fig E.1 (App E.3 p. 59) → say Beat 4 aloud twice
(the opening line + Finding 4 a third time, with Fig E.1 C/D in view) →
read App F.1 (pp. 61–63) with Fig F.1 (p. 62) → say
Beat 5 aloud, punchlines once more → say the confirmation one-breath
aloud once (that's all it needs) → the asks → from Part 2, say aloud:
"why overconfident", "what's the Phase-3 plan", "what does confirmed
mean", "why did timing fail".

**If a question stumps you:** "that's measured — let me follow up with
the exact number." Everything here has a file behind it.
