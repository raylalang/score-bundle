# Codex prompt (2026-11-25): whole-thesis audit and a trim plan

Paste everything below the rule into Codex, opened at the root of this
repository on `main`. Codex reads and writes one findings file. It does
not edit the thesis, the deck, any record, any registration document or
anything under `evidence/`.

---

You are reading a master's thesis as its examiner and as its editor. The
draft is `docs/thesis/draft.tex`, one file, 92 pages when built
(`tectonic draft.tex` in `docs/thesis/`, PDF `docs/thesis/draft.pdf`).
The author finds it far too long. Your job has two parts: audit the whole
draft, then propose what to cut so that every remaining page serves the
thesis statement. Read before you judge, and quote file and line for
every finding.

## The thesis statement

Use this as the test for every section. If a section does not support,
qualify or bound this statement, it is a candidate for cutting.

> Expressive performance can be recovered note by note, with calibrated
> uncertainty, by Bayesian score-conditioned inference under one
> multi-output Gaussian process on the score graph. This is supported on
> held-out piano data and held-out strings-and-winds data. The waveform
> and motif studies establish how far the score structure carries: a
> short time window does the work on every corpus where it was tested,
> the returns of a motif add information on articulation, and at the
> waveform nothing beyond the context's common offset has been found.

If you think the statement itself is wrong or mis-scoped after reading
the draft, say so first, with the sentence you would put in its place.

## Where the length is now

Chapter start pages in the current build:

| chapter | starts | pages |
|---|---|---|
| 1 Introduction | 4 | 5 |
| 2 Background and Related Work | 9 | 2 |
| 3 Model and Formulation | 11 | 8 |
| 4 Data and Experimental Protocol | 19 | 4 |
| 5 Results: Piano from Aligned MIDI | 23 | 8 |
| 6 Downstream Tasks | 31 | 2 |
| 7 Results: Strings and Winds from Tracked Audio | 33 | 12 |
| 8 The Waveform as the Observation | 45 | 13 |
| 9 Motif Structure in the Score Graph | 58 | 7 |
| 10 Discussion | 65 | 2 |
| 11 Future Work | 67 | 4 |
| 12 Conclusion | 71 | 2 |
| A Notation | 73 | 2 |
| B Development lineage | 75 | 2 |
| C Piano development studies | 77 | 4 |
| D Strings-and-winds development studies | 81 | 5 |
| E Calibration profile | 86 | 2 |
| Bibliography | 88 | 4 |

The author has not fixed a page target. Give the plan in tiers so the
author can choose: a first tier that brings the draft to about 75 pages
with no loss of argument, and a second tier that brings it to about 60
to 65 pages by also moving or cutting supporting material.

## Part 1: audit of the whole draft

Two earlier audits covered the material written on 2026-11-18 (the motif
chapter, the waveform held-out subsection and their framing; see
`docs/codex_audit_findings_2026-11-19.md` and its second-pass section).
Do not repeat those findings unless the current text still has the
problem. Concentrate on everything else: chapters 1 to 7, the first three
sections of chapter 8, the Discussion, Future Work, the Conclusion and
the appendices.

The records are the ground truth. Where the thesis disagrees with a
record, the record is right. The main ones:

- piano: `docs/graphgp_first_design.md`, `logs/confirmation_verdict.log`,
  `docs/phase1_calibration_results.md`, `docs/posterior_decomposition_results.md`,
  `docs/masking_sweep_results.md`, `docs/deep_baseline_results.md`,
  `docs/kernel_comparison_results.md`, `docs/kernel_multirate_results.md`,
  `docs/theory_features_results.md`, `docs/replication_set_results.md`,
  `docs/downstream_tasks_results.md`, `results/piano_timegauss_dev.md`
- strings and winds: `docs/phase2_prereg_design.md` (at tag
  `phase2-registration-2026-08-17`), `results/phase2_confirmation_results.md`,
  `results/phase2_real_results.md`, `results/corrnoise_confirmation_results.md`,
  `results/phase2_tonal_confirmation_results.md`, `results/phase2_drift_dev.md`,
  `results/phase2_tonal_dev.md`, `results/phase2_seeds23_dev.md`,
  `results/delta_vib_dev.md`, `results/tracker_calibration_dev.md`,
  `results/phase2_chain_prior_dev.md`, `results/phase2_timegauss_prior_dev.md`,
  `docs/phase2_synthetic_pilot.md`
- waveform: `results/phase3_*_dev.md`, `results/prior_value_*_dev.md`,
  `results/boundary_prior*_dev.md`, `results/phrase_prior_confirmation_results.md`
- exploration week: `results/exploration_week_2026-09.md`

Checks, in order of importance:

1. **Numbers against records.** Every number in a table or in prose that
   is not in a record, differs from it beyond rounding, or is attributed
   to the wrong set, arm, channel, seed count or piece count.
2. **Claim strength.** Any sentence that claims more than the evidence
   licenses: a development result read as held-out, a tie read as a win,
   a point estimate whose interval covers zero read as an effect, an
   attribution stated more broadly than the comparison that supports it
   (the earlier audits found this pattern repeatedly, for example "two
   beats" where the strings-and-winds graph uses seconds, and "no timing
   effect" where only the timing log score showed none).
3. **The same fact told twice or more.** List every result that appears
   in more than one place (abstract, introduction, results chapter,
   Discussion, Future Work, Conclusion, appendix) with the locations.
   This list feeds Part 2 directly.
4. **Internal consistency.** Numbers, counts and names that differ
   between two places in the draft. Forward references to sections that
   no longer contain what is promised.
5. **Style.** `docs/thesis/style_guide.md` is the authority and
   `python scripts/check_thesis_style.py --git HEAD` passes on the
   current file. Report only what the checker cannot see: register,
   bookkeeping narrative in results prose, self-grading, the protocol
   retold outside `sec:hygiene`.

## Part 2: the trim plan

For every section and subsection of the draft (use the `\section` and
`\subsection` list; there are 111), give one row:

| label or title | pages now | verdict | why, against the thesis statement | what survives if cut |
|---|---|---|---|---|

Verdicts: **keep**, **compress** (say to how many lines or pages),
**move to appendix**, **cut** (the content leaves the thesis; git history
keeps it). "What survives" is the sentence or table row that stays in
the main line when a section is compressed, moved or cut, so that no
result loses its statement and no reader loses a definition they need
later.

Then give the two tiers, each as a list of the rows it applies, with the
page saving of each and the running total.

Rules the plan must respect:

- Every held-out result stays in the main line with its interval: the
  piano test (Table `tab:headline-conf`), the strings-and-winds test
  (`tab:phase2-conf`), the two follow-up hypotheses (`sec:phase2-reuse`),
  the waveform phrase test (`tab:phase3-heldout`) and the motif test
  (`tab:motif-heldout`). Results that did not support their hypothesis
  stay as visibly as those that did.
- Every disclosure stays: the protocol section `sec:hygiene`, the three
  uses of the strings-and-winds test set, the nature of the motif
  held-out material, the leakage correction, the estimand gap.
- The attribution (the time window does the work, the pitch term adds
  nothing on intonation) stays, because the motif chapter is motivated
  by it.
- No number may change in what remains. Compressing means fewer words
  around the same numbers, or fewer numbers, never different numbers.
- Development studies that only scope a result (an alternative tried and
  rejected, a sensitivity check, a pilot) are the natural candidates for
  compression to one sentence or for removal. Say which ones an examiner
  would still expect to see at least named.
- Background, Future Work and the appendices should be judged by whether
  the main line needs them. Future Work that repeats results, and
  appendices that repeat main-line text, are candidates.
- Figures: list each `\figph` and figure environment with keep, move or
  cut, and say which figures carry an argument the text cannot.
- Prefer cutting repetition first, then scoping studies, then supporting
  analysis. Do not propose cutting the posterior decomposition, the
  downstream boundaries or the waveform chapter wholesale without saying
  what the thesis statement loses.

## What you may not do

- Do not edit `docs/thesis/draft.tex`, the deck, the speaker notes, any
  `docs/*prereg_design.md`, anything under `results/` or `evidence/`.
- Do not run any one-shot runner (`scripts/run_motif_heldout.sh`,
  `scripts/run_phrase_prior_confirmation.sh`,
  `scripts/run_phase2_confirmation.sh`,
  `scripts/run_phase2_reuse_confirmation.sh`) and do not fit any model.
  Building the PDF and running `pytest` are allowed.
- Do not propose new experiments.

## Output

One Markdown file, `docs/codex_trim_plan_2026-11-25.md`, in this order:

1. A one-paragraph verdict: is the thesis statement right, and where does
   the length come from.
2. Part 1 findings, numbered, ordered by severity, each with file and
   line, the quoted text, the record line it conflicts with (file and
   line) or the rule it breaks, and a proposed replacement sentence in
   the style guide's voice (no em-dashes, no semicolons, no asterisks,
   no dates, no bookkeeping vocabulary).
3. The repetition list from check 3.
4. The section table of Part 2.
5. The figure table.
6. Tier 1 and Tier 2, each with its rows, savings and running total.
7. Five questions an examiner would still ask after Tier 2, each with
   the sentence that answers it in the trimmed draft or the note that
   the trim removed the answer.

If a check turns up nothing, say so in one line under its heading.
