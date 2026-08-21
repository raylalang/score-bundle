# Meeting prep — study plan, study ledger, Q&A (2026-08, private)

For Ray only; everything grounded, every claim has a file behind it.
**Page numbers are the PRINTED ones (page footer / ToC).** Your PDF
viewer's own page counter runs 2 ahead (title + abstract are unnumbered):
printed 17 = viewer page 19.

## Study plan (~90 minutes, in order)

1. **First (5 min): sync Overleaf with current `main`.** Pushed repo is not
   a synced Overleaf; what the professor opens must be this state.
2. **§3.9, printed p. 17–23 (25 min).** Read once, slowly. The arc: nothing
   in the prior changes — channel set, targets, and noise do. On the way
   through: the data-point figure (Fig. 3.2, p. 20) and the tonal panel
   (Fig. 3.3, p. 21).
3. **The three figure walkthroughs below (10 min).** Say each aloud once.
4. **The results paragraphs, p. 22–23 (10 min).** The headline block and
   honesty ledger are on p. 23. Memorize the know-cold table.
5. **The study ledger below (20 min).** This is "what have you actually
   done" at one level deeper than the numbers.
6. **The Q&A (15 min).** Say the loudness, δ_vib, and MNAR answers aloud.
7. **The asks (5 min).** The planning-flavored close.
   Optional Phase-1 depth: §5.3, printed p. 30.

## The big picture (if he needs re-orientation, lead with this)

**30 seconds.** "The project: given the score and a performance, infer how
it was played — per-note timing, dynamics, intonation, vibrato — with
honest error bars, from one Gaussian-process model built on the score's
own structure. Phase 1 proved that on piano with a preregistered test.
Since then I carried the same model, unchanged, to real audio recordings,
and preregistered the claims for its confirmation."

**2 minutes** (point at the phase table, printed p. 5):
- Goal: one generative model of expressive performance — forward it
  synthesizes, inverted it transcribes; the contribution is structure plus
  calibration, not raw accuracy.
- Phase 0: a from-scratch music language model; its per-note embeddings
  feed the GP as a feature kernel.
- Phase 1 (piano, done): the multi-output graph GP beat the strongest
  two-stage pipeline on held-out imputation and passed a preregistered
  one-shot confirmation (0.376 vs 0.393 RMSE, coverage 0.925). Attribution
  measured: features recover, the graph calibrates.
- This stretch (Phase 2): the same prior, unchanged, on real recordings —
  six channels now, including timing from annotated onsets and the vibrato
  onset delay. The graph helps where it should, coverage is on target, the
  misses are reported. Claims preregistered; 13-piece pool untouched.
- Bonus finding: the circle-of-fifths geometry that hurt piano expression
  helps intonation.
- Next: spend the pool on his go; then tonal confirmation or Phase 3.

## Numbers to know cold (as-given variant, paired vs no-graph, dev)

| number | what it is, and why it matters |
|---|---|
| −0.89 cents\* | intonation recovery gain. The primary claim (C1); survives fresh seeds at −0.90. |
| −0.26\* / −0.30\* | vibrato extent / rate recovery. Fresh seeds: −0.25 / −0.30 — rock stable. |
| −3.8\* / −0.43\* | vibrato calibration (dNLL). The −3.8 is the seed-sensitive one (−0.22 ns on fresh seeds): C2's risky half. |
| −0.29\* | timing calibration (dNLL). C4; the graph's value on τ is calibration, matching the Phase-1 story. |
| 0.88–0.91 | coverage at nominal 90%, all six channels. C3's window is [0.85, 0.95]. |
| +0.04\* against | loudness dNLL: the bundle's ONE significant adverse cell. Replicates on fresh seeds (+0.048). Say it before anyone finds it. |
| −0.21\* / −0.05\* | tonal − plain on intonation, both axes. Exploratory; 61% of track-seeds improve. |
| tag 2026-08-17 | registration; pool of 13 pieces UNSPENT; the run is staged, one afternoon. |

## Study ledger — what was actually done this stretch (what / why / result / where)

1. **Tracker calibration.** Measured pyin against URMP's ground-truth
   pitch before trusting it. Median error 2–5 cents per instrument (an
   order of magnitude under vibrato extents); its confidence predicts its
   errors (gross errors 9.6% → 0.4% across quintiles). Fixed the chain:
   variances as-given, lowest confidence quintile dropped.
   → `results/tracker_calibration_dev.md`.
2. **Frozen split.** Dev/confirmation at the *composition* level,
   constructed data-blind, unit-pinned. Not just prudent: arrangements of
   one composition share byte-identical track recordings, so any finer
   split leaks audio. The 13-piece confirmation pool untouched since.
   → `phase2/splits.py`, the prereg doc.
3. **Real-audio evaluation, grown 4 → 6 channels.** Full pipeline on 77 of
   78 unique dev tracks, 30% of notes hidden, three systems (as-given =
   declared default, learned-scale, no-graph), both axes plus coverage,
   with a GT quasi-truth cross-check giving the same ordering. Results =
   the know-cold table; the ns cells and the adverse loudness cell are
   reported, not averaged away.
   → `results/phase2_real_results.md` (registration-grade run).
4. **Timing channel adopted (τ).** Possible without an audio aligner
   because URMP annotates note onsets. Feasibility measured first (76/78
   tracks warp cleanly; residual 79 ms; lag-1 correlation +0.59 —
   Phase-1-scale structure); then adopted as the leave-one-out tempo line
   of eq:localwarp, with the aligner error entering the noise row as the
   line's predictive variance. The two no-warp tracks stay in the eval;
   their τ cells are simply missing.
   → `results/tau_feasibility_dev.md`, `phase2/warp.py`.
5. **Vibrato delay decided (δ_vib IN, no claim).** The audit finding that
   forced the study: the bundle's ungated fit returns a *phase*, not a
   delay — so a gated estimator matching eq:vibrato exactly was built,
   and the in/out criterion committed BEFORE any number existed. Verdict:
   identifiable on 95%/97% of vibrato-identifiable notes (GT/pyin), 18 ms
   median agreement with truth, no intonation degradation, spread 8× its
   own SE. In the bundle; carries no registered claim because its graph
   contrasts are neutral — adoption and claims are separate gates.
   → `results/delta_vib_dev.md`, `scripts/eval_delta_vib.py`.
6. **Registration (2026-08-17).** Claims frozen in a committed revision
   before any confirmation piece is touched: C1 intonation recovery, C2
   vibrato calibration (both channels), C3 coverage in [0.85, 0.95] on
   all six, C4 timing calibration (secondary, non-gating); δ_vib
   deliberately claim-free. One shot, every number reported. A guarded
   runner is staged (double consent required) so the spend takes one
   afternoon. One dated, non-claim-altering erratum exists (a dropped
   significance star on the adverse loudness cell — restored).
   → `docs/phase2_prereg_design.md`, tag `phase2-registration-2026-08-17`.
7. **Fresh-seed robustness (seeds 2, 3).** Every direction reproduces;
   recovery deltas match to two decimals; the adverse cell replicates.
   The extent-channel NLL star does NOT survive (−3.8\* → −0.22 ns): C2's
   extent half is the riskiest registered claim — known before the pool
   is spent, which is the point of the discipline.
   → `results/phase2_seeds23_dev.md`.
8. **Circle-of-fifths geometry (exploratory).** Hypothesis committed
   before the run: the tonal metric that HURT piano expression should
   HELP intonation if temperament structure is real. It does — both axes
   (−0.21\*/−0.05\*) — and re-imposes the known penalty on timing, exactly
   as predicted. The first geometry-level (not edge-level) positive of
   the thesis. Adoption would need its own preregistered confirmation.
   → `results/phase2_tonal_dev.md`, Fig. 3.3.
9. **Phase-1 addendum (in the draft, post-lab-talk).** The posterior
   decomposes exactly by prior component: features carry the mean, the
   graph carries calibration; graph × embedding components are
   near-orthogonal (complements, not rivals); the cross-channel coupling
   earns its keep on velocity only; the harmonic-edge question closed
   with no adoption (a density gradient, not a uniform win).
   → §5.3 printed p. 30, `docs/posterior_decomposition_results.md`.
10. **Full audit + committee-level math pass.** Every quoted number
    re-verified against its evidence log; reproduction tolerances
    measured; three genuine math errors found and fixed (B's diagonal is
    not the prior variance; the per-channel guard has no overconfidence
    screen; exact nesting holds on the shared-shape slice); the
    missingness mechanism named (MCAR eval mask vs MNAR estimator cells);
    terminology checked against the GP/MIR/prereg literatures, four
    canonical citations added.

## Figure walkthroughs (say each aloud once)

**Fig. 3.2, printed p. 20 — one data point.** "This is what one
observation looks like. Top: a one-and-a-half-second violin note — the
tracker's confidence-kept pitch frames, the fitted model riding them, and
you can see the note starts straight: the vibrato begins only at the
dashed line, 144 milliseconds in — that is the delay channel. On the
right, the note's six-cell record, every cell with its own uncertainty —
note the honest ±37 ms on timing, that is the warp's noise row. Bottom: a
third-of-a-second note — too short to identify vibrato, so three cells
are missing. That is the cell mask." (If asked about the banded dots:
pyin's ~10-cent frequency grid; the fit averages across bins.)

**Fig. 3.3, printed p. 21 — the tonal contrast.** "Each dot is one
track-seed pair: intonation error with the circle-of-fifths metric minus
with the plain graph. The mean is minus 0.21 cents with the interval
clear of zero; 61% of pairs improve. Small, real, and exploratory by
design — adopting it would take its own preregistered confirmation."

**Fig. 3.4, printed p. 25 — four channels across a track.** "Same violin
track, 30% of notes hidden. Intonation with the 90% band and the hidden
notes as open circles; vibrato extent, where squares are
estimator-missing cells the GP fills; timing with the aligner's own error
bars — including one honestly wide edge note; and the delay channel,
where sub-zero stretches only ever appear in GP extrapolations, never in
observed values."

## Anticipated questions (one-line answers)

**Q: The targets are estimator outputs — what does "recovery" even mean?**
A: Exactly that, and the thesis says so wherever numbers appear: recovery
= agreement with the estimator, weaker in kind than Phase 1. The
quasi-truth cross-check (same estimator on URMP's ground-truth pitch)
gives the same system ordering on every channel — it isolates tracker
error, not estimator bias.

**Q: Why estimator variances as-given instead of learned?**
A: Measured twice: the synthetic pilot preferred as-given, and real data
re-confirmed it — with octave-failure cells present the learned scale
collapses, and after the failure rule as-given is still better calibrated
vs quasi-truth (0.82–0.86 vs 0.72–0.86 coverage).

**Q: Any cell against you?**
A: One, named in the draft: loudness — recovery ns and NLL +0.04
significantly against the graph under the default variant. It replicates
on fresh seeds; loudness carries no claim. Timing and delay recovery are
also ns (their value is calibration / bundle membership).

**Q: Two mask seeds — enough?**
A: Measured on fresh seeds 2, 3: every direction reproduces, recovery
deltas to two decimals, the adverse cell replicates. What moves: the
heavy-tailed calibration stars (extent −3.8\* → −0.22 ns). So C2's extent
half is the risky claim — known before the pool is spent.

**Q: Why does δ_vib get no claim if you adopted it?**
A: Adoption and claims are separate gates. The delay is measurable
(criterion fixed before the numbers), so it belongs in the bundle; its
graph contrasts are neutral on dev, so claiming it would be
claim-shopping.

**Q: The estimator's missing cells aren't missing at random, are they?**
A: Correct — the draft names the mechanism (Rubin): the evaluation mask
is MCAR by construction; the estimator's missingness is informative
(missing when vibrato is short or weak). Held-out scores stay
interpretable for the vibrato-identifiable sub-population; the filled
cells are prior extrapolations, possibly biased toward audible vibrato —
the delay panel's sub-zero stretches are the visible signature.

**Q: Where does alignment error go in τ?**
A: Into the noise row — the LOO tempo line's predictive variance.
Diagonal noise carries the error's scale, not its correlation along score
time; that stated limitation is why the τ claim is calibration-only.

**Q: Gaussian tails?**
A: The known Phase-1 limitation (one τ-outlier fit cost the confirmation
NLL tie). A Student-t prototype exists; future work gated on its own
confirmation set.

**Q: When do you spend the confirmation pool?**
A: My ask today. Registered, one shot, staged to run in an afternoon,
untouched.

**Q: What if a registered claim fails?**
A: Reported verbatim next to the claim; the pool is spent either way;
Phase 1 stands alone and Phase 2 was scoped so a negative costs the
thesis nothing.

**Q: Will you adopt the tonal metric?**
A: It is exploratory by the registered design; adoption is a
channel-dependent metric choice needing its own preregistered
confirmation. What it shows already: geometry helps exactly where the
target is pitch, hurts where Phase 1 said it would.

**Q: Why did the tonal metric hurt piano expression but help intonation?**
A: Expression travels register proximity; temperament and a shared tuning
reference travel the circle of fifths. Intonation is the first channel
whose target IS pitch — which is why the thesis pre-committed this as the
cheap decisive test.

**Q: External validity?**
A: URMP: 13 instruments, three families; vibrato win inside every family,
intonation inside strings and woodwind (brass ns alone). Beyond URMP the
alignment problem returns — stated future work.

## Blank-out insurance (six definitions, one line each)

- **as-given**: the estimator's own reported variances used directly as
  the observation noise (vs a learned per-channel rescaling).
- **cell / cell mask**: one (note, channel) entry / the pattern of which
  entries are observed.
- **MCAR vs MNAR**: missing by a coin flip vs missing *because of* the
  value (short/weak vibrato) — the eval mask is the first, estimator
  cells the second.
- **gated fit**: the eq:vibrato-exact estimator — flat until the onset
  delay, sinusoid after; the ungated fit's "delay" is only a phase.
- **quasi-truth**: the same per-note estimator run on the corpus's
  ground-truth pitch curves; independent of the tracker, not of the
  estimator.
- **development vs confirmation**: validation vs test, renamed to make
  the one-shot discipline explicit.

## The three asks
1. The claim set is preregistered — want to review it before the pool is
   spent, and when do we run it? (Raise: the fresh-seed check shows C2's
   extent half is the seed-sensitive claim — the one-shot could fail it
   honestly.)
2. Is the modest claim posture (no δ_vib claim, calibration-first) right
   for the committee?
3. Next priority: tonal-metric confirmation or Phase-3 scoping?

## In the room
- Have open: printed p. 25 (Fig. 3.4) and p. 22–23 (results; headline on
  23). Data question → p. 20 (Fig. 3.2); tonal question → p. 21
  (Fig. 3.3).
- Opening line: "I picked things back up and focused on Phase 2." If he
  needs orientation: the 30-second version, then the phase table on p. 5.
- If a question stumps you: "that's measured — let me follow up with the
  exact number." Everything above has a file behind it.
