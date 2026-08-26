# Meeting prep (2026-08, private)

**How to use this document.** To study: read top to bottom, then say
Part 1 aloud twice. In the room: the pocket card is your glance-sheet,
Part 1 is the spine you speak, Part 2 is what you reach for when asked.
Part 3 is background — study it, never present it.
**Pages are the PRINTED ones (footer / ToC); your PDF viewer's counter
runs 2 ahead.** Before anything else: **sync Overleaf with `main`**.

---

## Pocket card

**Pages:** phase table 5 · §3.9 17–25 · data-point Fig 3.2 = 20 ·
tonal Fig 3.3 = 22 · truth/estimate/GP Fig 3.4 = 23 · results 23–24
(headline + ledger on 24, results Table 3.3 on 25) · channels
Fig 3.5 = 26 · §5.3 = 31 · Phase-2 architecture Fig 8.1 = 46.

**Eight numbers** (as-given variant, paired vs no-graph, development):
intonation −0.89\* · extent/rate recovery −0.26\*/−0.30\* · vibrato
calibration −3.8\*/−0.43\* · timing calibration −0.29\* · coverage
0.88–0.91 (all six) · adverse cell: loudness NLL +0.04\* against · tonal
−0.21\*/−0.05\* · Phase-1 confirmed 0.376 vs 0.393\*.

**Three asks:** (1) review the registered claims — when do we spend the
pool? (note: C2's extent half is seed-sensitive); (2) is the modest claim
posture right?; (3) tonal confirmation vs Phase-3 scoping next.

---

# Part 1 — The story you tell (the spine, ~10 minutes of talk)

### Beat 0 — Open
"I picked things back up and focused on Phase 2." If he needs
orientation first, give Beat 1 in full; otherwise one sentence of it and
move on.

### Beat 1 — The big picture (phase table, p. 5)
Thirty seconds: "The project: given the score and a performance, infer
how it was played — per-note timing, dynamics, intonation, vibrato — with
honest error bars, from one Gaussian-process model built on the score's
own structure. Phase 1 proved that on piano with a preregistered test:
0.376 vs 0.393 RMSE against the strongest two-stage pipeline, coverage
0.925. Since then I carried the same model, unchanged, to real audio."
If he wants more, walk the table: Phase 0 = the music model whose
embeddings feed the GP; Phase 1 = confirmed, attribution measured
(features recover, the graph calibrates); Phase 2 = this stretch;
Phase 3 = scoped.

### Beat 2 — What Phase 2 is (§3.9, p. 17–19)
"Nothing in the prior changes — what changes is the channel set, how the
targets are obtained, and the noise, which stops being negligible. The
per-note vector is now six evaluated channels: intonation, vibrato extent
and rate, loudness, timing, and the vibrato onset delay. Every value is
estimated from the recording, each with its own uncertainty — that is the
point: the observation noise becomes a modelling term."
Two channel decisions were settled by measurement, not taste: timing
comes from the corpus's annotated onsets through a leave-one-out tempo
line (no audio aligner; the aligner error goes into the noise row), and
the onset delay passed a criterion committed before its numbers existed
(identifiable on 95%/97% of vibrato-identifiable notes, 18 ms agreement
with ground truth).

### Beat 3 — What one data point looks like (Fig 3.2, p. 20)
"This is one observation. Top: a one-and-a-half-second violin note — the
tracker's confidence-kept pitch frames, the fitted model riding them, and
you can see the note starts straight: vibrato begins only at the dashed
line, 144 milliseconds in — that is the delay channel. On the right, the
note's six-cell record, every cell with its own uncertainty — note the
honest ±37 ms on timing; that is the warp's noise row. Bottom: a
third-of-a-second note, too short to identify vibrato — three cells
missing. That is the cell mask. And the bottom row: how the audio becomes
the loudness cell — the note's waveform quartered, log-RMS per chunk,
mean minus the track mean, and the chunk spread is the error bar; you can
see the crescendo inside the note widening it."
(Banded dots question: pyin's ~10-cent frequency grid; the fit averages
across bins.)

### Beat 4 — What happened (results, p. 23–25; Fig 3.4 p. 23; channels Fig 3.5, p. 27)
"Paired against the no-graph ablation, under the noise variant we
declared as default: recovery improves significantly on intonation and
both vibrato channels — minus 0.89 cents, minus 0.26, minus 0.30 — and
calibration improves significantly on both vibrato channels and on
timing — minus 3.8, minus 0.43, minus 0.29 in NLL. Coverage sits at
0.88 to 0.91 at nominal 90% on every channel of the six. It's all in
Table 3.3, next page (25)."
Then the honesty, unprompted: "Three recovery contrasts are not
significant — loudness, timing, delay — and one cell is significantly
*against* the graph: loudness calibration, plus 0.04. It replicates on
fresh seeds. I'd rather you hear that from me than find it."
The validation shot, Fig 3.4 (p. 23): "here are all three layers on two
notes — the true pitch curve, the equation's estimate, and the GP's
prediction. On a held-out note the truth falls inside the GP's bands for
centre, extent, and rate — and the delay prediction misses, which is
exactly why the delay carries no claim. On an estimator-missing note the
GP's fill puts truth within about one standard deviation on all three
vibrato parameters." Fig 3.5 (p. 27) if wanted: the four channels across
one track, including the wide-interval edge note in timing and the delay
panel where sub-zero stretches appear only in GP extrapolations, never in
observed values.

### Beat 5 — Where it stands (the discipline)
"The claim set is preregistered — committed on the seventeenth, before
any confirmation piece was touched: intonation recovery; vibrato
calibration on both channels; coverage inside 0.85 to 0.95 on all six;
timing calibration as a secondary claim. The delay channel deliberately
carries no claim — its graph contrasts are neutral, and adoption and
claims are separate gates. One shot, every number reported; the 13-piece
pool is untouched and the run is staged to take an afternoon. One thing I
know already and want on the table: on fresh mask seeds the recovery
results reproduce to two decimals, but the extent-channel calibration
star does not — so that half of the vibrato-calibration claim is the one
the one-shot could honestly fail."

### Beat 6 — The bonus finding (tonal panel, Fig 3.3, p. 22)
"One exploratory result I like: the circle-of-fifths metric that *hurt*
piano expression *helps* intonation — minus 0.21 cents with the interval
clear of zero, and it re-imposes the known penalty on timing, exactly
what the hypothesis predicted. First sign in this thesis that a
music-theoretic geometry, not just music-theoretic edges, earns its
place. Exploratory by design — adopting it would take its own
preregistered confirmation."

### Beat 7 — Close (the asks)
1. "The claims are registered — do you want to review them before the
   pool is spent, and when should we run it?"
2. "Is the modest posture right for the committee — no delay claim,
   calibration first?"
3. "After the confirmation: tonal-metric confirmation, or Phase-3
   scoping?"

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
estimator · *development/confirmation* = validation/test, renamed for the
one-shot discipline.

**Data and method**
- *What is the system, in one breath?* Score plus noisy per-note
  measurements in; one graph-structured Gaussian posterior over every
  (note, channel) cell out, with honest error bars. The graph encodes one
  belief — expressive behavior varies smoothly across neighboring notes —
  and the per-piece evidence decides how much graph, features, coupling,
  and noise. Evaluation: hide 30% of notes, predict them, score recovery
  AND calibration.
- *The figures show MISSING cells — mistake?* Deliberate, and the point:
  short notes physically cannot contain 1.5 vibrato cycles, so those
  cells are structurally unmeasurable — the missing entries are the
  problem statement, and the model's job is to put calibrated posteriors
  in them (Fig 3.4's case B shows truth agreeing when it does).
- *Which instruments?* URMP's thirteen, three families: strings
  (vn/va/vc/db), woodwind (fl/ob/cl/bn/sax), brass (tpt/hn/tbn/tba);
  violin dominates the dev side (27 of 101 tracks); no piano, no voice.
- *Targets are estimator outputs — what does recovery mean?* Exactly
  that, and the thesis says so wherever numbers appear: agreement with
  the estimator, weaker in kind than Phase 1. The quasi-truth cross-check
  gives the same ordering on every channel — it isolates tracker error.
  Fig 3.4's bottom row makes it visible: the estimator sits 8 cents from
  truth on a sparse note and the GP recovers the estimator's value.
- *Missing cells aren't missing at random.* Correct; the draft names the
  mechanism (Rubin). Eval mask MCAR by construction; estimator cells
  informative (missing when vibrato is short or weak). Held-out scores
  are interpretable for the vibrato-identifiable sub-population; filled
  cells are prior extrapolations — the delay panel's sub-zero stretches
  are the visible signature.
- *Is loudness comparable to velocity?* Not assumed — ℓ is what the
  microphone heard (log RMS, ≈8.7 dB per unit, per-track centred), v is
  what the finger did (a keystroke value). Same role in the bundle,
  different physical quantity; the thesis lists cross-phase
  comparability as a claim to defend, not a convention.
- *Where is the loudness equation from?* Our construction from standard
  ingredients (RMS energy as the textbook loudness proxy, log as the dB
  convention); the four-chunk design is ours so the channel arrives with
  a standard error — Fig 3.2's bottom row shows the whole path.
- *Why as-given noise, not learned?* Measured twice: the pilot preferred
  it, and on real data the learned scale collapses when octave-failure
  cells are present; after the failure rule, as-given is still better
  calibrated vs quasi-truth (0.82–0.86 vs 0.72–0.86).
- *Where does alignment error go?* Into τ's noise row — the tempo line's
  predictive variance. Diagonal noise carries scale, not correlation
  along score time; that stated limitation is why the τ claim is
  calibration-only.

**Results and honesty**
- *Any cell against you?* One: loudness calibration, +0.04 significantly
  against, replicating on fresh seeds. Loudness carries no claim.
- *Two seeds enough?* Fresh seeds 2, 3: every direction reproduces,
  recovery to two decimals, adverse cell replicates. What moves: the
  heavy-tailed calibration stars (extent −3.8\* → −0.22 ns). Known
  before the pool is spent.
- *Why no claim on the delay if you adopted it?* Adoption and claims are
  separate gates: measurable (pre-stated criterion), so it is in the
  bundle; graph-neutral on dev, so claiming it would be claim-shopping.
- *External validity?* 13 instruments, three families; vibrato win
  inside every family, intonation inside strings and woodwind (brass ns
  alone). Beyond URMP the alignment problem returns — stated future work.

**Process and what's next**
- *Can you resynthesize the audio from the Phase-2 output?* The expressive
  skeleton, yes — the six channels reconstruct a pitch curve (Fig 3.4's
  drawn GP curves are exactly that), per-note levels, and timing around a
  supplied tempo line; a parametric synthesizer could render it today at
  caricature level. The audio itself, no: envelopes, timbre, and noise
  are the waveform layer, which is precisely Phase 3's likelihood — the
  one block the architecture marks as changing with the phase. Mirror on
  the analysis side: in Phase 2 the model never sees audio (a fixed
  estimator reduces it to per-note numbers first — hence the
  recovery-scope caveat and the identifiability missingness); in Phase 3
  the waveform itself is the observation, which dissolves both at the
  price of exact inference.
- *Gaussian tails?* Known Phase-1 limitation (one τ-outlier fit cost the
  confirmation NLL tie). Student-t prototype exists; gated on its own
  future confirmation set.
- *What if a registered claim fails?* Reported verbatim next to the
  claim; the pool is spent either way; Phase 2 was scoped so a negative
  costs the thesis nothing.
- *Will you adopt the tonal metric?* Only through its own preregistered
  confirmation; it is exploratory by the registered design.
- *Why did it hurt piano but help intonation?* Expression travels
  register proximity; temperament travels the circle of fifths.
  Intonation is the first channel whose target IS pitch — which is why
  this was pre-committed as the cheap decisive test.

---

# Part 3 — Background depth (study only)

**The study ledger — what was actually done this stretch:**
1. *Tracker calibration.* pyin vs URMP ground truth before trusting it:
   2–5 cents median per instrument; confidence predicts errors (gross
   9.6%→0.4% across quintiles) → variances as-given + lowest quintile
   dropped. `results/tracker_calibration_dev.md`.
2. *Frozen split.* Composition-level, data-blind, unit-pinned — forced,
   not just prudent: arrangements share byte-identical recordings.
   13-piece pool untouched. `phase2/splits.py`.
3. *Evaluation grown 4→6 channels.* 77/78 unique dev tracks, 30% hidden,
   three systems, both axes + coverage + quasi-truth cross-check.
   `results/phase2_real_results.md`.
4. *τ adopted.* Feasibility first (76/78 tracks, 79 ms residual, lag-1
   +0.59), then the LOO tempo line with aligner σ in the noise row; the
   two no-warp tracks keep their other channels, τ cells missing.
   `phase2/warp.py`.
5. *δ_vib decided.* The audit found the ungated fit's delay is a phase →
   gated estimator built to match eq:vibrato exactly; criterion committed
   before numbers; 95%/97% coverage, 18 ms truth-agreement → IN, no
   claim. `results/delta_vib_dev.md`.
6. *Registration.* C1–C4 + decision rule frozen 2026-08-17 (tag), one
   dated non-claim-altering erratum; guarded afternoon runner staged.
   `docs/phase2_prereg_design.md`.
7. *Fresh-seed robustness.* Recovery to two decimals; extent NLL star
   seed-sensitive → C2 risk named. `results/phase2_seeds23_dev.md`.
8. *Circle of fifths.* Hypothesis pre-committed; helps intonation both
   axes, re-imposes the timing penalty. `results/phase2_tonal_dev.md`.
9. *Phase-1 addendum* (post-lab-talk, §5.3 p. 32): posterior decomposes
   exactly by component — features carry the mean, the graph carries
   calibration, graph × embeddings near-orthogonal; coupling earns its
   keep on velocity only; harmonic-edge question closed, no adoption.
10. *Full audit + math pass.* Every number re-verified against its log;
    three math errors fixed; missingness mechanism named; terminology
    checked against the field with four canonical citations added.

**How to study (~90 min):** Overleaf sync → read §3.9 (p. 17–25) slowly →
say Part 1 aloud twice → read the results pages 23–25 against the pocket
card → skim the ledger → say the loudness, delay, and MNAR answers from
Part 2 aloud → the asks.

**If a question stumps you:** "that's measured — let me follow up with
the exact number." Everything here has a file behind it.
