# Score-Bundle Models — meeting digest (2026-07-03)

*Every number here is from a re-run on held-out data.*

## The project in one paragraph

Given the printed score and a recorded performance (piano MIDI), the system infers
**how each written note was played** — its timing, articulation, and loudness —
each with an honest error bar. Two ingredients carry the result, with distinct
roles: a **score graph** (notes connected by adjacency in score-time, voice, and
chord) makes the error bars trustworthy and cuts prediction error substantially
below any per-note predictor; a small **music network** we train ourselves on
piano performances adds loudness knowledge and sharper confidence.

## Headline result

Held-out ASAP, 30 pieces × 4 mask seeds, 40% of notes hidden. Lower RMSE/NLL is
better; coverage should sit near the nominal 90%.

| System | Error (RMSE) | Confidence quality (NLL) | 90%-interval coverage |
|---|---|---|---|
| Predict zero | 0.566 | −0.007 | 87% |
| Graph alone | 0.404 | −0.308 | 92% |
| **Network + graph** | **0.393** | **−0.322** | **92%** |
| Features + network + graph | 0.388 | −0.333 | 92% |

**The full model is best on both axes at once — lowest error and best-calibrated
confidence — and the graph's contribution is statistically significant on both,
piece by piece.**

## What we found

1. **The graph earns its place.** It cuts error well below per-note prediction and
   brings the nominal-90% intervals to ~92% coverage. Its advantage over the same
   prediction *without* the graph is significant on both accuracy and confidence.
2. **The network's specific contribution is loudness and calibration.** Against a
   strong hand-built baseline (25 score features fit under the same protocol) the
   network **ties on average error**; its real, significant edge is loudness
   (~19% lower error on that channel) and confidence quality. Best of all, the two
   **stack** — features + network is the best system in the table.
3. **A task-matched training variant doesn't help.** We also trained a
   bidirectional, mask-aware version aligned exactly with the evaluation task; at
   matched compute it does **not** beat the simpler network (3× compute doesn't
   change that). The straightforward setup already captures the signal.

## Where the structure helps — six downstream demonstrations

A single boundary emerges across all six: **the graph helps whenever notes are
judged one at a time (recovery, confidence, spotting errors), and does not help
when a whole performance is collapsed into a summary.**

- **Error spotting** — *clear win*: it flags corrupted notes better than any
  structure-free baseline, and its edge *grows* as the errors get subtler.
- **Cleaning noisy transcriptions** — *win* at a known noise level (better error
  and equally honest confidence).
- **Completing a performance from an excerpt** — *partial*: the network guess
  carries the far-future; the graph helps where hidden notes have nearby heard
  neighbours.
- **Knowing when to trust a prediction** — *qualified win*: setting aside the
  least-confident predictions genuinely lowers error on the rest.
- **Guessing musical era from playing style** — *negative*: a note-level tool does
  not help piece-level style summaries.
- **Identifying the pianist** (Vienna 4x22, 22 pianists × 4 shared excerpts) —
  *mixed*: the extracted expression **does** carry a performer's identity (3–4×
  chance), validating what we recover; smoothing it with the graph does not help
  this summary.

## How the evaluation is kept trustworthy

- **Unseen test pieces.** 30 held-out pieces the network never saw. Because ASAP
  reuses many MAESTRO recordings, every test performance whose training twin was
  seen is removed (≈1000 → 650), so the numbers reflect generalization.
- **No peeking at the target.** The network's per-note read-out is taken *before*
  a note's own loudness enters it, so it cannot copy the answer; a strict variant
  additionally blanks hidden notes' loudness from the input — at essentially no
  cost to the numbers.
- **A self-checking fit.** The per-piece confidence fit validates itself on
  held-back observed notes and falls back to a conservative fit if it misbehaves;
  on healthy fits it changes nothing. Every pooled table also reports the median,
  so no single bad cell can hide in the mean.

## Decision for the meeting

Adopt **features + network + graph** as the headline system (the better numbers),
or keep **network + graph** as the headline with features reported as an available
add-on (the cleaner single-model story)? Both are supported; the contribution —
*structure + calibration, with each ingredient's marginal value isolated* — is the
same either way.

## Figures

![The graph's contribution per prior mean](figures/digest_headline.png)

![Where the network earns its place (per channel)](figures/digest_channels.png)
