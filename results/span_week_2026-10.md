# The span week, 2026-10-15 to 2026-10-21: the score prior inside waveform inference

One-page ledger, in plain language, of everything done this week. Every
number here is copied from the results file named beside it. Development
work throughout; nothing registered was touched; no claims.

## The question

The thesis's method says: a prior built from the score should help
recover a note's expression when the audio is ambiguous. Until this
week, nothing tested that at the waveform. The test is a held-out
comparison: take a short run of notes, remove one note (or one joint
between notes) from the audio, and ask three priors to guess it from
its neighbours — one that treats notes as independent, one that links
each note only to the next in time, and the score graph, which links
notes that are close in time *and* in pitch. Same audio, same
optimizer, same amount of prior spread. Score the guess against the
annotation.

## Monday 2026-10-15

**Housekeeping found by the handoff.** Main was not actually pushed;
two cache files the experiment reads existed only in the old worktree
(the fitted curve-prior hyperparameters and a newer dev-targets file).
Both installed, the old one backed up. The worktree's raw per-cell
results and logs were copied into the main checkout.

**The follow-up the handoff asked for** (`results/prior_value_tuning_dev.md`):
give every prior a free "tuning scalar", a number shared by all notes
of a run that absorbs the player's overall tuning offset. The numbers
made no sense: the independent prior got worse (28 cents median error),
and every prior scored worse than guessing zero (14 cents).

**The diagnosis.** Hiding a note's audio samples did not hide the note:
its model terms still reached the neighbours' audio through the soft
edges of its window. The leftover weight was tiny, but because the
model lets amplitudes be anything, that leftover acted as loud, wrong
evidence — on one chain it moved the fit by 38 units against a prior
worth 4 to 35. Every hidden-note estimate in the 10-14 run, in every
arm, came from that leak: zero correlation with the truth, spread of
±85 cents. **The 10-14 result ("structure wins under ambiguity") was
withdrawn**; an erratum sits on `results/prior_value_dev.md`, the table
left as measured.

**The fix, protocol v2.** The hidden note is removed from the model
entirely (`fit_chain(drop_notes=)`), not just its samples. A unit test
pins that with no structure the hidden note's estimate is exactly the
prior's mean. The unchanged code path was checked to regenerate the
10-14 table bit for bit.

**Two reruns on the same 47 chains (4 development tracks).**
- Without the scalar (`results/prior_value_v2_dev.md`): structure
  halves the error, 14.4 → 6.8 (time-only) and 7.6 (score graph)
  cents; the two structured priors tie.
- With the scalar in every arm (`results/prior_value_v2_tuning_dev.md`):
  all three priors tie — score graph vs independent −0.02 cents
  [−0.33, +0.32]. **On short chains, the only thing neighbours tell you
  about a hidden note's intonation is the shared tuning offset.**
  Bands are honest (coverage 0.89–0.96); with full audio everything
  ties.

Documents brought to this state: thesis Chapter 8 section (rewritten),
abstract, contributions, research-question text, Future Work; brief;
roadmap (with the 2027-02-02 submission and ~2027-01-07 presentation
dates); CLAUDE.md. The boundary-prior extension was written as a
proposal, not run (`docs/boundary_prior_proposal.md`).

## Saturday 2026-10-18

Ray chose to run both follow-ups: B (boundaries) and C (phrase scale),
"we can always roll back". Code and hypotheses committed before either
run (46ba871).

## Monday 2026-10-20

**B, boundaries** (`results/boundary_prior_dev.md`, 47 chains, 40 with
distinct pitches across the hidden joint). Same comparison over the
timing of note boundaries: start every boundary from the score's tempo
line, remove the middle joint from the model (hard cuts, samples
masked), prior over the boundary shifts, tuning-style offset scalar in
every arm.
- The three priors tie on the hidden joint (score graph vs independent
  −2.45 ms [−8.38, +3.32]; medians 28–31 ms).
- Every prior beats the tempo line by 10–14 ms — the shared offset
  again.
- With full audio the three priors land on identical boundaries
  (19.6 ms median; the chain study's deployment arm measured 19.5):
  the prior costs nothing.
- Two calibration facts: hidden-joint bands far too wide (coverage
  1.00; the prior scale is the track's full timing spread), observed-
  joint bands far too narrow (0.7 ms, coverage 0.05): the model finds
  the acoustic transition, the annotation marks the musical onset.

**C, phrase scale** (`results/prior_value_phrase_dev.md`, 48 runs of
6–10 notes drawn from all 78 development tracks, the hidden note's pitch
recurring at least two notes away). The one place the score graph has
an edge the time-only prior lacks.
- The pre-stated main contrast passes: score graph beats the time-only
  prior by 1.63 cents [−2.93, −0.46].
- But the score graph ties the independent-with-scalar prior (−0.07),
  and the time-only prior is *worse* than independent (+1.56). The
  graph's win is undoing the damage the time-only prior does at phrase
  scale, not new information. The gain does not grow with the strength
  of the recurrence link (r = −0.03).

**The three-span verdict.** Against the thesis's own baselines
(independent, time-only), the score prior is never worse and beats the
time-only prior at phrase scale. Against the right baseline — share the
common offset with every prior — it adds nothing measurable, for
intonation at either span or for timing at chain scale. The weak form
of the hypothesis survives; the strong form has failed every test so
far. The honest positive: a note or joint with no audio inherits its
context's offset with a trustworthy band.

Documents: thesis Chapter 8 (second and third spans, closing
paragraph), Discussion and Conclusion sentences; brief sections A,
D.4, F.1, G; roadmap; CLAUDE.md; README; DDSP memo; slides README.
Full test suite passed (verified after the commit).

## Tuesday 2026-10-21

Eight figures from the stored cells (`scripts/make_span_viz.py`,
`docs/thesis/figures/span_*.png`): the design as a system, one chain
end to end, the leakage diagnosis, the scalar attribution, the paired
contrasts over three spans, calibration strips, the boundary gains, the
phrase dose-response. All in the thesis section (81 pp); five as deck
backup frames (pages 47–51 after the 2026-10-21 split of the setup
slides; main line unchanged).

## What is open, for Ray

1. Stop here and present the three-span verdict as the waveform
   chapter's measured answer, or go on to the within-note curve level
   or the other channels (proposal first either way).
2. Whether anything here is worth registering (only the weak form and
   the common-offset gain are positive).
3. The registration pool for the correlated-timing-noise and tonal
   follow-ups.
4. Sync Overleaf from main; retire the old worktree (fully merged).

## Lessons that now apply to every prior-value design

- A hidden element must be removed from the model, not just its
  samples.
- Every arm gets the common-mode scalar, or the comparison measures
  tuning bookkeeping.
- Before reading any contrast, check that the independent arm equals
  its analytic reference (the prior mean, or the tempo line).

## Commits

c6cc2a0 (v2 measured) · 969e03d, 61bdbeb (scalar, erratum, v2 code) ·
46ba871 (B and C code) · e641c73 (B) · acd3076 (C, verdict) · 10d3e8a
(docs current) · 3afaaae (figures).
