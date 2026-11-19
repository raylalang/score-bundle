# Codex audit prompt (2026-11-19): the material written on 2026-11-18

Paste everything below the rule into Codex, opened at the root of this
repository on `main` at or after commit `2e470be`. Codex reads files and
writes a findings list. It does not edit the thesis, the deck, any
registration document, any results record or anything under `evidence/`.

---

You are auditing a master's thesis repository one day after its last
result lines were written in. Everything you need is on disk. Read
before you judge, and quote file and line for every finding.

## What happened on 2026-11-18 and why it needs an audit

Two pre-specified, run-once evaluations finished that day and their
results were written into the thesis, the deck and the speaker notes in
one afternoon, in a single pass, by one author. Nothing has been read
back since. Your job is to find what a sharp examiner would find before
the examiner does, and to find any place where the writing drifted from
the measured record.

The two evaluations:

1. Waveform, phrase scale: `results/phrase_prior_confirmation_results.md`
   (the block "VERDICT against the registered claims" is the scored
   outcome), registration `docs/phrase_prior_prereg_design.md` at tag
   `phrase-prior-registration-2026-11-18`, evidence
   `evidence/phrase_prior_confirmation/`. W1 passed, W2 tie, W3 pass.
2. Motif, held-out performances: `results/motif_heldout_results.md`
   (same block name), registration `docs/motif_prereg_design.md` at tag
   `motif-registration-2026-11-18`, evidence `evidence/motif_heldout/`.
   M1 passed, M2 missed, M3 report-only.

The development records behind them:
`results/prior_value_phrase_dev.md`, `results/motif_pilot_dev.md`,
`results/motif_pilot2_dev.md`, `results/motif_stage2_dev.md`.

The records and the registration documents at their tags are the ground
truth. Where the thesis or the deck disagrees with a record, the record
is right and the finding goes against the thesis or the deck. Do not
propose changing a record.

## Files under audit

Thesis, one file: `docs/thesis/draft.tex` (89 pages when built). The
parts written or changed on 2026-11-18:

- The motif chapter, `\chapter{Motif Structure in the Score Graph}`
  (label `ch:motif`, about lines 3090 to 3476): sections
  `sec:motif-family`, `sec:motif-conditional`, `sec:motif-design`,
  `sec:motif-dev` with `tab:motif-dev` and the subsection
  `sec:motif-stage2`, `sec:motif-heldout` with `tab:motif-heldout`,
  `sec:motif-meaning`; equations `eq:motif-u`, `eq:motif-adj`,
  `eq:motif-shared`, `eq:motif-pc`, `eq:motif-cond`, `eq:motif-mix`;
  figure `fig:motif-graph`.
- The waveform chapter's held-out subsection `sec:phase3-heldout` with
  `tab:phase3-heldout` (about lines 3007 to 3089), and the sentences in
  that chapter that used to say no held-out evaluation had been run.
- The framing sentences added elsewhere: the abstract, the research
  question paragraph and the contributions list and the stage table in
  the Introduction, the Discussion's limitations, the Future Work
  section `sec:future-motif`, the Conclusion.
- The protocol section `sec:hygiene` in the data chapter, which must
  disclose every re-use of the strings-and-winds held-out test set
  (it has now been used three times: once as the primary test, twice
  for follow-up hypotheses, once at the waveform level) and the nature
  of the motif held-out pool (new performances of the development
  movements, not new repertoire).

Deck: `docs/slides/deck.tex` (45 pages). Slides 30 and 31 (the three
spans, with the held-out row), 32 (time window), 33 to 39 (the motif
section: divider, family, conditional, development table, predicted
occurrences, pool and claims, result), 40 (Summary), 41 (Plan).
Speaker notes: `docs/slides/speaker_notes.md`, the entries for those
slides. The notes must say what the slide says, in the slide's numbers.

Style authority for the thesis: `docs/thesis/style_guide.md`. The
checker `python scripts/check_thesis_style.py --git HEAD` passes on the
current file. Your findings on style are about what the checker cannot
see: register, claim strength, protocol narrative leaking into results
prose, a word from the banned list in a new sentence.

Code that defines what the equations mean: `scripts/eval_motif_pilot_dev.py`
(the edge family, the two arms, the masking schemes, the scoring sets,
the bootstrap), `scripts/run_motif_heldout.sh`, `scripts/power_motif.py`,
`scripts/eval_prior_value_phrase_dev.py` and
`scripts/run_phrase_prior_confirmation.sh`. `src/score_bundle/gp.py` is
the model. The notation table is in `CLAUDE.md` and in the thesis
appendix `app:notation`.

## Checks, in order of importance

### A. Every number against its record

For each number in `tab:motif-dev`, `tab:motif-heldout`,
`tab:phase3-heldout` and in the prose of the sections listed above,
find the line in the record that it comes from. Report any number that
is not in a record, that differs from the record by more than its
printed rounding, or that is attributed to the wrong arm, channel,
scheme, seed count or piece count. Pay attention to:

- Piece and performance counts. The motif development studies use 21
  movements; the held-out test uses 18 movements and 58 performances
  (the generator of the frozen report titles its tables by performance
  count, so "58" is a performance count and never a movement count).
  Two movements, 3-1 and 24-1, were excluded from the motif study
  because they are piano held-out test pieces. The waveform held-out
  test uses 48 phrases from 20 of the 40 tracks; the development study
  used 48 phrases from 78 development tracks.
- Coverage numbers for the time-only arm, which differ between
  development and held-out (development 0.970 / 0.891 / 0.878 against
  held-out 0.968 / 0.891 / 0.878 for timing / articulation / velocity).
  Check which set each table prints.
- Seed counts: pilot one has two seeds, pilot two and the held-out run
  four. A sentence that gives a pilot-one number with a pilot-two seed
  count, or the reverse, is a finding.
- The M2 result. It is a miss by the fixed rule (−0.018 [−0.191,
  +0.225]), and the development basis was also not starred (−0.071).
  Any sentence that makes the timing effect sound supported, or that
  hides the miss, is a finding. So is any sentence that makes the miss
  sound like evidence of absence: the pre-stated power was 0.33.
- The per-piece correlation 0.88 between development and held-out
  effects, and "15 of 18 movements below zero". Confirm both appear in
  the record with those values.
- In the waveform section: −0.82 [−1.59, −0.05], the development basis
  −1.63 [−2.93, −0.46], the pre-stated expectation of about −0.9, the
  W2 tie −0.13 [−0.44, +0.17], the family breakdown strings −1.55 /
  winds −0.54 / brass −0.44 with none of the three starred, the
  coverages 0.94 to 0.98.

### B. Claim strength against the fixed decision rules

Read each registration document at its tag (`git show
motif-registration-2026-11-18:docs/motif_prereg_design.md` and the same
for the phrase one). Then read every sentence in the thesis, the deck
and the speaker notes that states what the held-out result means.
Report any sentence that claims more than the rule licenses. The
licensed headlines are:

- Motif: annotated motif structure carries expressive information
  beyond temporal smoothing, supported once on held-out performances of
  the development movements, for articulation accuracy on hidden
  returns. Across performers of the same movements, not across
  repertoire. Timing not supported.
- Waveform: at phrase scale the score graph beats a naive temporal
  prior on the hidden note (the weak form), on held-out tracks, with a
  smaller effect than on development. Nothing beyond the common tuning
  offset (the strong form fails again).

Specific places to read closely: the abstract's new sentence, the
contributions list, the stage table row, the first paragraph of
`sec:motif-meaning`, the Conclusion, deck slides 39 and 40, and the
speaker note for slide 39. Report any place that says "confirmed" or
"holds" without the "across performers of the same movements"
qualification nearby, and any place where the two held-out results are
summed into a stronger joint statement than either supports.

Also report the opposite error: a hedge so heavy that a passed primary
reads as a failure, or a development number presented where the
held-out number should be.

### C. The mathematics against the code

- `eq:motif-u` and `eq:motif-adj`: the relative position inside an
  occurrence, the Gaussian in that position with length 0.1, the
  many-to-many pairing between occurrences of one type. Check them
  against the edge construction in `scripts/eval_motif_pilot_dev.py`.
  Report any mismatch in what is normalized, what is symmetric, which
  pairs are joined, and whether notes shared by two occurrences are
  handled the way the text says.
- `eq:motif-shared`: the shared-weight arm, W = W_time (+ W_pitch) +
  w_m W_motif, log w_m learned by evidence. Confirm the held-out arm
  `time_motif` has no pitch term and the text says so.
- `eq:motif-pc`: the per-channel arm as one more feature block built
  from the motif Laplacian, per-channel scale learned by evidence. Check
  the factorization against the code and the unit test
  `tests/test_motif_pilot.py`. Check that the fixed shape parameter is
  described as the text describes it.
- `eq:motif-cond` and `eq:motif-mix`: the conditional mean and
  variance of a hidden note. Check the signs, the role of the degree
  term and the noise scale, and that the symbols are the ones in
  `app:notation` and `CLAUDE.md` (no symbol reused for two things; in
  particular `σ` is the posterior standard deviation only, `S` is the
  score support, `B` the coregionalization matrix).
- `sec:phase3-heldout`: the three priors at matched marginal variance,
  the common-mode scalar in every arm, the hidden note dropped from the
  design (not only its samples). Confirm the section states all three
  and that the frozen runner does what the section says.

### D. Protocol and disclosure

- `sec:hygiene` is the one place the protocol is described. Confirm it
  now names every use of the strings-and-winds held-out test set and
  the motif pool's relation to the development pieces, and that the
  results sections do not re-narrate the protocol beyond one sentence.
- The thesis must not say "registered", "preregistered", "frozen",
  "tagged", "claim", "pool", "dev", "confirmation set", "report-only",
  "cell", or any calendar date in prose. The checker catches most of
  these in old text. Read the new sections for any that slipped in.
- Report any sentence that lets the reader think the motif held-out
  performances are new repertoire, or that the waveform held-out tracks
  had not been used before.

### E. Cross-document consistency

The same fact must read the same in `draft.tex`, `deck.tex`,
`speaker_notes.md`, `docs/ROADMAP.md`, `docs/meeting_qa_prep.md` and
`CLAUDE.md`. Report any number, count, arm name, slide number or verdict
wording that differs between any two of them. Slide references inside
the deck and the notes (the Plan slide refers to other slides by
number) must point at the right slides in the 45-page build.

### F. Structure and reading order

Read the motif chapter as an examiner who has read the rest of the
thesis. Report:

- any term used before it is defined in this chapter (occurrence, type,
  return, hidden return, the two masking schemes, the scoring set);
- any result stated in the development section and then re-stated in
  the held-out section as if new;
- any place where the chapter's own narrative (why one test, why
  articulation as primary, why timing was expected to miss) is missing
  or arrives after the result;
- whether `fig:motif-graph` is referred to in the text and whether the
  caption matches what the figure shows (it comes from the proposal
  deck and draws the two edge families);
- whether `tab:motif-dev` and `tab:motif-heldout` have the same column
  layout and the captions make the difference between them unmissable
  (development, 21 movements, four seeds, versus held-out, 18 movements,
  58 performances).

### G. The questions an examiner would ask

Write, as the last section of your findings, the five questions you
would ask at the defence about the material under audit, each with the
sentence in the thesis that answers it or the note that nothing does.
Candidates: why a held-out test across performers and not across
pieces; what the articulation effect size means in units a musician
recognises; why the timing effect appears on development with two seeds
and vanishes with four; what the per-channel form's velocity result
means given it was not the primary; whether 0.83 of the development
effect on performances the development run never read is the same
finding or a smaller one; what the strong form's repeated failure at
phrase scale says about the score graph's value in the waveform
setting.

## What you may not do

- Do not edit `draft.tex`, `deck.tex`, `speaker_notes.md`, anything
  under `docs/*prereg_design.md`, `results/` or `evidence/`.
- Do not run `scripts/run_motif_heldout.sh`,
  `scripts/run_phrase_prior_confirmation.sh`,
  `scripts/run_phase2_confirmation.sh` or
  `scripts/run_phase2_reuse_confirmation.sh`. Each is a one-shot that
  has been run; the held-out sets are spent. Reading them is fine.
- Do not start any computation that fits a model. `pytest` is allowed.
- Do not propose new experiments. The result lines are closed; the
  remaining work is writing.

## Output

A single Markdown file `docs/codex_audit_findings_2026-11-19.md` with
one numbered finding per item, ordered by severity (a wrong number or an
over-claim first, a style slip last). Each finding has:

1. File and line (`docs/thesis/draft.tex:3412`).
2. The sentence or table cell, quoted.
3. The record line it conflicts with, quoted, with its file and line,
   or the rule it breaks.
4. A proposed replacement sentence, written in the style guide's voice
   (no em-dashes, no semicolons, no asterisks, no dates, no
   bookkeeping vocabulary, present tense for what the results show).

End with the section G questions. If a check turns up nothing, say so in
one line under its heading so the reader knows it was done.
