# Thesis style guide (2026-10-28, Ray: "almost a whole makeover")

The thesis is an academic document read by a committee. It is not a
project report, a lab notebook, or a status update. Every sentence should
read as if written once, at the end, by someone describing a finished
piece of research to a reader who was not there while it was done.

This guide governs `docs/thesis/draft.tex` and `docs/thesis/chapters/*.tex`.
It changes the writing, never the science: see "Hard constraints" first.

## Hard constraints (the checker enforces these)

- Every number in the text stays exactly as it is. No rounding, no
  re-deriving, no adding, no dropping. The only numbers that may disappear
  are calendar dates (see Vocabulary).
- Every `\label`, `\ref`, `\eqref`, `\cite`, `\figph`, equation
  environment, and `tabular` body stays byte-identical. Captions may be
  reworded. Column headers inside tables may not.
- The order of chapters, sections and figures stays. Section titles may be
  rewritten (see Titles). No section is added or removed.
- No claim becomes stronger or weaker. A result that was reported as not
  supported stays not supported, in plain words. Hedges stay hedges.
- The standing wording rules hold: no "Phase N" (use the piano,
  strings-and-winds and waveform settings), no "quasi" (the targets derived
  from annotated curves are "the reference"), no em-dashes, no semicolons,
  no asterisks for significance.
- `\textbf` is used only for the defining instance of a term and never for
  emphasis. `\emph` is used for genuine contrast only. Targets: at most
  five `\textbf` and twelve `\emph` per chapter file.
- No `\paragraph{}` run-in headers remain. See Structure.

## Voice and tense

- Impersonal by default: "the model", "the evaluation", "this thesis". The
  first person plural is allowed for design decisions ("we choose a Matérn
  kernel because ...") and should be rare.
- Present tense for the model, its properties, and what the results show.
  Past tense for what was done ("the held-out set was evaluated once").
- No second person, no questions in the body, no exclamation.
- The writer does not grade the work. Delete "honest", "honestly",
  "rigorous", "careful", "deliberately", "verbatim", "faithfully",
  "as such", "measured" used as a badge. State what was done and let the
  reader judge.

## Register and vocabulary

The draft grew out of a preregistered workflow, and the workflow's
bookkeeping leaked into the prose. The protocol is described ONCE, in the
data-and-protocol chapter (the section on selection discipline), in plain
methodological language. After that, results chapters report predictions,
estimates and interpretations, with no procedural narrative.

| Do not write | Write instead |
|---|---|
| confirmation set, confirmation pool, the pool | held-out test set (or "the test set" once defined) |
| dev, dev set, dev-only | development set |
| claim (as a noun for a hypothesis) | hypothesis, prediction |
| registered, preregistered, frozen, tagged | pre-specified, fixed in advance |
| one-shot, run once, spent, never rerun | evaluated once (say it once, in the protocol chapter) |
| verdict, PASS, FAIL, passed, failed, win, lose, tie | supported / not supported; better / worse / no difference |
| repaired | revised, resolved |
| gate, gating, non-gating, report-only | (describe the role in words: "the pre-specified primary measure", "a secondary measure") |
| as-given, as given | the estimator's own variances (define once, then "the given variances") |
| cell, cells (as a unit of the grid) | piece-seed combination, or simply "evaluation" |
| guard, guarded, opt-in, config, variant | describe what the mechanism does |
| starred, star, `*` | "the interval excludes zero" or "significant at the stated level" |
| any file, script, function, tag or path name in prose | describe the computation; identifiers go to a footnote at most, once per chapter |
| any calendar date | delete; chronology belongs to the lineage appendix, and even there as "an earlier version" |
| erratum, withdrawn run, protocol error | describe the corrected design in the methods and, where the error is scientifically informative, state the lesson in one sentence without narrating the episode |
| "the committee", "the professor", "Ray", "the meeting" | delete |
| headline (as a noun) | main result, principal result |
| delta (for a difference) | difference |

Technical terms already chosen stay: the proposed model, the two-stage
pipeline, the music model, the score graph, coregionalization, the
reference centre, the estimand gap, the common offset.

Significance is stated in words, once per result: "the paired difference
is −0.014 with a 95 percent interval of [−0.025, −0.003]". Do not add a
star, a "significant" badge, or a parenthetical "(sig.)".

## Structure

- No run-in `\paragraph{}` headers. Each becomes either (a) the topic
  sentence of the paragraph it introduced, or (b) a `\subsection{}` with a
  topical title, when the section genuinely has parts. Prefer (a). Labels
  attached to a removed `\paragraph` move to the enclosing section or the
  new subsection.
- Every paragraph opens with a topic sentence and develops it in four to
  eight sentences. No one-sentence paragraphs. Average sentence under
  twenty-five words.
- Dashes become commas, colons, or new sentences. Parentheses are rare and
  never nested. No colon-labels such as "Design:" or "Findings:".
- Bulleted lists only for genuinely parallel items the reader will refer
  back to (contributions, hypotheses, the notation table). Argument is
  prose.
- Titles name the topic, not the outcome. "Posterior decomposition", not
  "where the graph earns its keep". "Error spotting", not "clear win".
  Chapter titles stay as they are.
- Captions describe what is shown. They do not argue, grade, or repeat
  the result's interpretation.
- Footnotes carry what a reader may want but the argument does not need:
  identifiers, reproduction details.

## How a result is reported

1. What was predicted and why (one or two sentences, in the words of the
   hypothesis, not the bookkeeping).
2. What was found: the estimate, its interval, the direction, in words
   and numbers.
3. What it means for the question the chapter asks.

Example, before:

> \paragraph{The timing claim, repaired.} The one registered claim that
> failed (C4) named its remedy ... \textbf{N1 passed} ($-0.121$ ...).

After:

> The timing hypothesis of the earlier evaluation was not supported, and
> the diagnosis pointed to the noise model: alignment error is correlated
> along score time, which a diagonal noise term cannot represent. The
> revised noise model places an autoregressive block in the timing
> channel's noise covariance, ... On the held-out test set the paired
> difference in timing log score is −0.121 with a 95 percent interval of
> [−0.156, −0.089], so the revised model is better calibrated on timing.

## What the rewrite is not

It is not a shortening. Page count may move either way by a few pages.
It is not a chance to add content, results, or figures. It is not a
restructuring beyond the conversions named above.
