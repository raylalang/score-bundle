# Codex prompt (2026-11-25): what does not belong in a finished thesis

Paste everything below the rule into Codex, opened at the root of this
repository on `main`. Codex reads and writes one file. It does not edit
the thesis or anything else.

---

You are the external examiner and the editor of a master's thesis,
`docs/thesis/draft.tex` (one file, 75 pages when built with
`tectonic draft.tex` in `docs/thesis/`). Two earlier audits fixed its
numbers and its claim strength, and a first trim removed repeated results.
This round asks a different question.

## The question

The author's words: the thesis should be clean, and it is not simply a
compilation of the work. Find every passage that is there because of how
the work happened rather than because the argument needs it.

The author's own example. §10.3 says "Leakage was found and corrected", and
§4.4 and the contributions list tell the story of a read-out that once saw
each note's own target and was fixed before any held-out evaluation. That
is a development hiccup, not a result. What the reader needs is the design
rule it taught (the read-out is taken before the note's own target, which
§3.5 states) and the audit that shows the evaluation is leak-free. The
story of the bug is fat.

Apply that standard to the whole draft. For each passage ask:

1. Does it state the method, a result, or a limit that a reader needs to
   understand, trust or bound a claim? Then it stays.
2. Does it tell how the work got there: a bug found and fixed, a
   preliminary version, a pilot that came first, a design that was tried
   and rejected, the order in which things were done, an earlier system
   the current one replaced? Then it goes, or is rewritten as the design
   rule it produced.
3. Does it narrate the evaluation protocol again ("fixed in advance", "by
   the decision rule fixed in advance", "archived with the run logs",
   "reproduced digit for digit")? The protocol belongs in one place,
   §4.5. A result needs its outcome and at most a pointer.
4. Does it report a study of an alternative that was not adopted? Keep it
   only if an examiner would expect it to be named, and then in one
   sentence. Say which ones an examiner would expect.
5. Is it implementation detail (optimiser names, grid spacings, sampling
   rates, seconds per fit, unit tests, repository file names, code
   numbering)? It goes unless a claim depends on it.
6. Is it the author commenting on the work (defending a choice, grading a
   result, announcing that something "deserves" attention, saying a
   limitation is "reported rather than patched")? It goes.
7. Does the framing itself carry project history (chapters introduced as
   "stage 1 of the progression", a status column saying "Implemented", a
   baseline described as "the earlier development system")? Propose the
   framing a reader who never saw the project would need.

Also report anything you notice that is wrong, where the text contradicts
another part of the thesis or the records.

## What must stay

Do not propose removing any of these, however procedural they sound:

- the development and held-out distinction, stated once in §4.5;
- every use and re-use of a held-out test set (the strings-and-winds test
  pieces were used three times, the motif test uses further performances
  of development movements);
- every held-out result with its interval, including the ones that went
  against the model;
- the estimand gap at the waveform;
- the leak-free read-out as a property of the method, and the
  velocity-centring limitation;
- any number in what remains (shortening means fewer words around the same
  numbers, or fewer numbers, never different numbers).

The records under `results/` and `docs/*_results.md` are the ground truth
if you need to check what a passage refers to.

## What you may not do

- Do not edit `docs/thesis/draft.tex` or any other file except your output.
- Do not run model fits or any `scripts/run_*` script. Building the PDF is
  allowed.
- Do not propose new experiments.

## Output

One Markdown file, `docs/codex_fat_audit_2026-11-25.md`:

1. One paragraph: what kind of fat dominates, and roughly how many pages
   it occupies.
2. A table per category above (process history, protocol restated,
   alternatives not adopted, implementation detail, self-commentary,
   project framing), one row per passage: section and line, the quoted
   text, and the action (cut, or the replacement sentence written in the
   thesis's voice: no em-dashes, no semicolons, no asterisks, no dates).
3. Anything wrong or contradictory found along the way, with file and line
   and the record that shows it.
4. A short list of passages that look procedural but must stay, with the
   reason, so the revision does not over-cut.
5. An estimate of the page saving and of the resulting length.
