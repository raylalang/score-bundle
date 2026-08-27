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
on 25) · drift paragraph 20 · channels Fig 3.5 = 27 · §5.3 = 32 ·
Phase-2 architecture Fig 8.1 = 46.

**Eight numbers** (CONFIRMATION, as-given, paired vs no-graph, one-shot):
intonation −0.88\* (dev basis −0.89\* — reproduced) · vibrato
calibration −3.0\*/−0.56\* (the seed-sensitive extent star HELD) ·
coverage 0.88–0.91 (all six) · timing calibration −0.03 ns = **C4
failed** · adverse cell: timing recovery +0.003\* (≈3 ms) · drift:
10 cents/note, 97% GT sign agreement, graph-white (lag-1 +0.03 vs τ's
+0.59) · tonal (dev, exploratory) −0.21\*/−0.05\* · Phase-1 confirmed
0.376 vs 0.393\*.

**Three asks:** (1) how prominent should the failed timing claim be in
the narrative — honesty exhibit or one line?; (2) the next registration
is the tonal metric — power check says: calibration as primary claim
(recovery is a coin flip below ~20 pieces), Bach10 + reused-URMP
combined, disclosed; (3) Phase 3 OPENED — first
waveform inference on a real note: 0.7–1.6 cents from GT, no tracker;
overconfident widths = the drift mismatch, now the phase's measured
starting constraint.

---

# Part 1 — The story you tell (the spine, ~10 minutes of talk)

### Beat 0 — Open
"Two things since we spoke. First, your point about the sine model being
too simple — I didn't argue with it, I measured it, and you were right
in a specific way I can show you. Second: the registered claims are no
longer pending — I ran the one-shot confirmation. Three of the four
claims confirmed; the fourth failed and is reported verbatim."
Then let him pick which thread first. Default order below: verdict,
then drift, then asks.

### Beat 1 — The big picture, if he wants orientation (phase table, p. 5)
Thirty seconds: "The project: given the score and a performance, infer
how it was played — per-note timing, dynamics, intonation, vibrato — with
honest error bars, from one Gaussian-process model built on the score's
own structure. Phase 1 proved that on piano with a preregistered test:
0.376 vs 0.393 RMSE against the strongest two-stage pipeline, coverage
0.925. Phase 2 carried the same model, unchanged, to real audio — and is
now confirmed the same way Phase 1 was."

### Beat 2 — What Phase 2 is, if needed (§3.9, p. 17–19)
"Nothing in the prior changes — what changes is the channel set, how the
targets are obtained, and the noise, which stops being negligible. The
per-note vector is six evaluated channels: intonation, vibrato extent
and rate, loudness, timing, and the vibrato onset delay. Every value is
estimated from the recording, each with its own uncertainty — that is
the point: the observation noise becomes a modelling term."
(Fig 3.2 p. 20 for what one data point looks like; the walkthrough is in
the previous edition's Beat 3 muscle memory: long note with the 144 ms
delay, short note with missing vibrato cells, waveform → loudness cell.)

### Beat 3 — The confirmation verdict (THE beat; Table 3.4, p. 25)
"On the seventeenth I registered four claims and froze the protocol —
git tag, one shot, every number reported whatever it says. I then spent
the pool: 13 pieces held back since the split was frozen in early
August, 40 unique tracks, the registered protocol verbatim, about an
hour and a half sharded.
Claim one, intonation recovery: minus 0.877, interval clear of zero —
the development basis was minus 0.891, so it reproduced almost exactly.
Claim two, vibrato calibration on BOTH channels: minus 3.0 on extent,
minus 0.56 on rate, both starred — and the extent half was the one I had
flagged to you as seed-sensitive, the claim I said could honestly fail.
It held. Claim three, coverage: 0.88 to 0.91 on all six channels, inside
the registered band. By the decision rule those three passing means the
Phase-2 headline is confirmed: the unchanged graph prior extends to real
audio with calibrated uncertainty, at confirmation level.
Claim four — timing calibration, the secondary claim — failed. Minus
0.03 with the interval touching zero, against a development basis of
minus 0.29. And there's a small starred cell *against* the graph on
timing recovery: plus three milliseconds on a 65-millisecond RMSE. Both
are in the table verbatim. So the honest summary of timing on this
corpus: coverage on target, but no confirmed graph win — the aligner
error we always said lives in that channel is presumably where the
development signal went."
If asked why trust the rest: "the failure is precisely why the
confirmations mean something — the rule was written before the data,
and it was allowed to say no. It said no once."

### Beat 4 — The drift study (his comment, answered; ¶ p. 20)
"You asked whether the sine model is too simple — loudness and
intonation shifting over time. I refit every identifiable note in the
development set with an added drift term, in both our tracker's curve
and the ground-truth curve independently. You were right: the drift is
real music, not tracker noise — two-thirds of notes have a significant
slope, the two curves agree on its direction 97% of the time, and the
median note drifts about 10 cents — an order of magnitude above the
intonation cell's reported precision. Loudness moves even more, though
most of that is decay envelope, not expression.
But the decisive measurement is the second one: those within-note slopes
carry essentially zero correlation from note to note — lag-1 of +0.03,
where timing sits at +0.59. The structure the sine model discards is
real, but it is exactly the structure a graph across notes cannot help
with. So the per-note resolution is the right level for *this* model —
and the discarded part is precisely what Phase 3's frame-level
likelihood is designed to carry. Your comment, measured, turns into the
argument for the next phase."

### Beat 5 — The bonus finding, if time (tonal, Fig 3.3, p. 22)
"One exploratory result: the circle-of-fifths metric that *hurt* piano
expression *helps* intonation — minus 0.21 cents, interval clear of
zero, and it re-imposes the known penalty on timing, exactly what the
hypothesis predicted. First sign that a music-theoretic geometry earns
its place. Adopting it needs its own preregistered confirmation — which
raises a question I want your view on."

### Beat 6 — Close (the asks)
1. "How prominently do you want the failed timing claim in the thesis
   narrative — I lean toward keeping it visible as the honesty exhibit."
2. "The tonal metric deserves a confirmation, and I ran the power
   check: at Bach10's size the recovery claim would star only half the
   time even though the effect is real — so the honest design is
   calibration as the primary claim, recovery secondary, on Bach10 plus
   the reused URMP pool with the reuse disclosed. Draft design is
   written; I'd like your sign-off on the corpus choice before I freeze
   anything."
3. "Phase 3 is opened: I ran the first waveform-likelihood inference on
   a real note — no tracker, the audio as the observation — and it
   localizes intonation within about a cent of ground truth. It also
   showed exactly the failure mode the drift study predicted:
   overconfidence under the constant-parameter curve. So the phase
   starts from a measured design constraint, not a sketch. Priority
   next: the joint model over z, or the tonal confirmation first?"

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
11. *Phase-1 addendum* (§5.3 p. 32): posterior decomposes exactly by
    component — features carry the mean, the graph carries calibration;
    coupling earns its keep on velocity only.
12. *Full audit + math pass.* Every number re-verified against its log;
    three math errors fixed; terminology checked against the field.

**How to study (~90 min):** Overleaf sync → read §3.9 (p. 17–25) slowly,
ending on the confirmation paragraph + Table 3.4 (p. 25) → say Part 1
aloud twice, Beat 3 a third time → read the drift ¶ (p. 20) → say the
"why did timing fail" and "what does confirmed mean" answers aloud →
the asks.

**If a question stumps you:** "that's measured — let me follow up with
the exact number." Everything here has a file behind it.
