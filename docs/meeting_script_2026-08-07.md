# Meeting script, 2026-08-07 (~30 min + questions)

Page numbers refer to the current draft.pdf (52 pages). Each block gives
POINT (where your finger goes) and SAY (speakable lines, adapt freely).

---

## Opening (1 min, no pointer)

SAY: "Three updates since last time. First, the thesis model is now
interpretable from the inside. We can decompose its posterior exactly by
information source, and that is new math in the draft. Second, that closed
the harmonic-edge question without spending any confirmation data. Third,
Phase 2 moved from the synthetic pilot to real recordings, and the graph
prior wins on every channel there. I will end with three decisions I need
from you."

---

## Segment 1. The posterior decomposition (10 min)

**POINT: p. 12, Eq. (3.18)**, the model in one equation.

SAY: "Start from the model in one equation. The covariance is a sum of
three terms. The coupled graph GP, the feature kernels, and the noise."

**POINT: p. 12, Eq. (3.20)**, finger on the sum over zeta.

SAY: "The new observation is that the latent field itself is a sum of
independent Gaussian components. One for the graph, and one per feature
set. So zeta here runs over three components in Phase 1. The graph, the
score features, and the music model. Careful, that is not the three terms
of (3.18). The middle term of (3.18) contains two components, and the
noise term is not a component at all."

**POINT: p. 13, Eq. (3.21).**

SAY: "Because the components are independent, each one has its own exact
posterior mean, this is just Gaussian conditioning. And the key point is
that they share the same weight vector, so they sum exactly to the full
posterior mean. Nothing is approximated."

**POINT: p. 13, Eq. (3.22).**

SAY: "The posterior covariance between two different components is this
cross term, and it is negative wherever two sources can explain the same
variation. So the model itself tells us where its information sources are
redundant and where they are complementary."

**POINT: p. 13, Eq. (3.23).**

SAY: "Both identities are exact, the component means sum to the posterior
mean and the pairwise covariances sum to the posterior covariance, and both
are pinned by unit tests."

**Flip to p. 27, Section 5.3, Eq. (5.1), then Eq. (5.2) on p. 28.**

SAY: "Two diagnostics follow. The covariance share, which allocates the
posterior mean per channel and sums to one exactly. And the per-note
component correlation from the cross term. Now the measured picture."

SAY (numbers, from the paragraph between the two equations): "The score
features carry most of the mean on every channel. The music model has its
largest share on loudness, 0.34, exactly the channel where it carries
signal the hand-built features lack. The graph's share of the mean is
modest. That is the earlier ablation story, features recover, the graph
calibrates, but now shown inside a single fitted model. And the correlation
between the graph and the music model is essentially zero, minus 0.004 to
minus 0.06 across channels. They are complements, not rivals."

**POINT: p. 28, Figure 5.2** (the snapshot).

SAY: "Here is one development piece with the mean split into its component
curves. On velocity the orange music-model curve tracks the note-to-note
alternation, the green score features carry the phrase shape, and the two
big articulation events are carried almost entirely by the blue graph
component, propagated from observed neighbours."

**POINT: p. 29, Figure 5.3** (the contrast pair).

SAY: "And which component dominates is a property of the piece. Same
channel, two pieces. Top, a Liszt piece where the loudness rests on the
graph and the music model is switched off entirely. Bottom, a Debussy piece
where it rests on the music model. The per-piece evidence assigns the
division of labour, and at the extremes it inverts completely."

---

## Segment 2. The harmonic-edge question, closed (3 min)

**POINT: p. 32, Section 5.10**, then the closing paragraph of that section
(the sentence starting "Second, the same comparison under the final model
at every level reveals a clean density gradient", around p. 34, just after
Table 5.5).

SAY: "We also closed the harmonic-edge question. The earlier finding was
that chord and voice-leading edges become redundant once the music model is
in the kernel. That turned out to be true only at our operating point. Ran
the comparison at every masking level. When observation is dense, the extra
edges win recovery, significantly. At our 40 percent operating point they
tie. At 50 percent hidden they add nothing and we saw one fit collapse from
degenerate learned length scales. The reading is mechanistic. Edge weights
are per-piece hyperparameters, and like every per-piece quantity they need
observed coverage to be determined. So the thesis keeps the plain graph, no
confirmation set was spent, and the harmonic family is recorded as a
dense-observation refinement."

---

## Segment 3. Phase 2 on real audio (12 min)

**POINT: p. 16, Section 3.9 heading.**

SAY: "The big update. Phase 2 moved from the synthetic pilot to real
recordings, and I want to show the discipline first, then the result."

**POINT: p. 19, the paragraph "Data, and what recovery can mean here".**

SAY: "The corpus is URMP. 44 chamber pieces, separately recorded monophonic
tracks, 13 instruments, with ground-truth pitch at exactly our hop size.
Three things happened before any model touched the data. One, the tracker
was calibrated against that ground truth. Median error is 2 to 5 cents per
instrument, an order of magnitude below vibrato extents, and the tracker's
confidence genuinely predicts its errors, gross octave slips fall from
about 10 percent in the least confident frames to under half a percent in
the most confident. That fixes the estimator chain by measurement, we use
the supplied variances and discard the least confident fifth of frames.
Two, the development and confirmation split is frozen at the composition
level, built deterministically before looking at any data. And that
granularity turned out to be mandatory, not just prudent, because
arrangements of the same composition share identical track recordings.
Three, we say plainly what recovery means here. The targets are estimator
outputs, so the claim is agreement with the estimator, weaker in kind than
Phase 1, with the ground-truth curves as an independent cross-check."

**POINT: p. 17, Eq. (3.34)** if asked how frames become targets, otherwise
skip. The identifiability thresholds are in the itemized list right below
it.

**POINT: p. 20, the paragraph "First results on real audio".**

SAY: "The full pipeline then ran end to end on 77 development tracks with
30 percent of notes hidden. The result. The graph prior improves held-out
recovery on every channel of the bundle, with calibrated intervals. Paired
against the no-graph ablation, intonation improves by 0.92 cents,
vibrato extent and rate by 0.15 and 0.17 in log units, loudness by 0.010,
every one of those significant, and coverage sits at 0.88 to 0.91 against a
nominal 90 percent. The ordering is unchanged when scored against the
ground-truth-derived targets. And the win holds separately within strings,
woodwinds, and brass, so no instrument family is carrying the average."

SAY (the honesty point, same paragraph): "One target-quality rule was
added, and it is validated rather than assumed. About 2 percent of
intonation targets are octave slips. The ground-truth cross-check rejects
every single one of them, 0 out of 358, so they are marked missing, the
same status as unidentifiable vibrato."

**POINT: p. 21, Figure 3.2.**

SAY: "Here is one violin track. Top panel, intonation. The dots are the
estimator's own values with their own uncertainties, the band is the model.
Bottom panel is the capability that no per-note method has. The squares are
notes where the estimator cannot identify a vibrato extent at all, and the
model supplies a posterior there anyway, from the neighbours through the
graph and from the coupled channels."

**POINT: p. 18, Eq. (3.36).**

SAY: "Last Phase-2 point. The biggest threat to validity was alignment
error entering timing. On this corpus it has a measured resolution. The
annotated onsets anchor the warp with no audio aligner on 76 of 78 tracks.
Each note is judged against a tempo line fitted from its neighbours,
excluding itself, that is equation 3.36. The residual timing has a median
spread of 79 milliseconds with a neighbour correlation of 0.59. That is
piano-scale timing with exactly the structure the graph models. So timing
is viable as a Phase-2 channel here."

SAY (once, plainly): "Everything in this segment is development evidence.
The 13-piece confirmation pool is frozen and untouched."

---

## Segment 4. Decisions (4 min)

**POINT: p. 5, Table 1.1** (the phase table).

SAY: "So the state of the programme is this. Phase 1 confirmed and now
interpretable. Phase 2 running on real data at development level. I need
three decisions to move to a confirmable Phase-2 claim."

**POINT: Future Work chapter, the Phase-2 section, the list of four
questions and the closing sentence naming the blockers.**

SAY: "One, do we adopt timing into the evaluated bundle, the feasibility
says yes. Two, is the vibrato onset delay in or out as a channel. Three,
once those are fixed I freeze the claim set and register the confirmation,
and it runs exactly once."

---

## If asked (pointers for likely questions)

- "Why is noise not a component?" Point to p. 13, the paragraph after
  Eq. (3.23). A hidden note's noise is never observed and is independent of
  everything observed, so its conditional mean is zero. It reappears as the
  exact residual at observed notes and as the additive floor in the
  predictive variance.
- "Why zeta?" The letters a and b are taken, amplitudes and beat onset.
  The component index got its own letter.
- "Is the decomposition causal?" No, and the draft says so at the end of
  the Section 3.6 paragraph. It is the fitted model's own attribution under
  its learned scales.
- "How do you avoid fooling yourselves on Phase 2?" Point to Section 4.4,
  p. 23, the selection-hygiene discipline, and to the frozen split in the
  Phase-2 data paragraph. Same rules as Phase 1, and the Phase-1
  confirmation caught one development advantage failing to replicate, so
  the discipline has teeth.
- "What is the headline Phase-1 number again?" Table 5.1, p. 26. RMSE 0.376
  against 0.393 for the strongest two-stage system, paired difference
  significant, coverage 92.5 percent, preregistered and run once.
- "Could the harmonic edges come back?" Yes, as a dense-observation
  refinement, through a second preregistered confirmation set, never
  through development numbers. End of Section 5.10.

## What not to open unless asked

Downstream chapter, deep baselines, masking sweep, theory features,
Phase 3. All unchanged since the lab talk.
