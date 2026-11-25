# Fat audit of the thesis draft (Claude, 2026-11-25)

Draft audited: `docs/thesis/draft.tex` at `de75f72`, 75 pages after the
Tier 1 trim. Line numbers refer to that revision. This is the first of two
independent reads; Codex produces the second
(`docs/codex_fat_prompt_2026-11-25.md`) without seeing this one, and the
revision uses both.

## The test

The thesis is a finished argument, not a compilation of the work. A
sentence earns its place only if a reader needs it to understand, trust or
bound a claim of the thesis. The test for each passage:

1. **Does it state the method, a result, or a limit?** Keep it.
2. **Does it tell how the work got there** (a bug found and fixed, a
   preliminary version, a pilot that preceded, a rejected design, the order
   things were tried)? Cut it, or rewrite it as the design rule it taught.
3. **Does it narrate the protocol again** ("fixed in advance", "by the
   decision rule fixed in advance", "archived with the run logs")? The
   protocol is stated once in §4.5. Elsewhere one pointer at most.
4. **Does it report a scoping study** (an alternative tried and not
   adopted)? Keep only if an examiner would ask about it, and then in one
   sentence.
5. **Is it implementation detail** (grid spacings, optimiser, unit tests,
   seconds per fit, repository file names)? Cut.

What must stay for trust: the development and held-out distinction, every
re-use of a test set, every held-out result including the ones that went
against the model, the estimand gap, the leak-free read-out rule as a
design property, the velocity-centring limitation.

## 1. Process history (cut, or rewrite as the design it taught)

| where | text | action |
|---|---|---|
| §1.3 L237 | "The two-stage pipeline is the earlier development system" | Reframe the two-stage pipeline as the **standard construction** (plug-in trend plus residual kriging), the baseline of the piano test. Its role in the thesis is comparator, not ancestor. Same reframing in App. B L3223–3227 ("reached through an earlier version ... the earlier version survives"). |
| §1.4 L262–266 | contribution 4, "a leak in the music-model read-out that was found, corrected and is now excluded" | Contribution becomes the protocol: development and held-out split with disclosed re-use, contamination filtering, leak-free read-out by construction with a bitwise audit. |
| §4.4 L910–923 | "An earlier read-out let the music model see each note's own velocity ... re-measured after the correction ... retracted $0.057$ ... manufactured an apparent scaling gain" | Keep: the read-out rule is in §3.5 already; the bitwise audit; the centring limitation. Cut the history and the retracted number. The model-size result (1.3× larger model, identical downstream) can move to §5.6 as "perplexity does not predict downstream value". |
| §5.5 L1329–1334 | "Two failure modes of the fit were found and resolved ... one (piece, seed) fit in 120, produced timing errors of 4.5 and 17" | This is the two-stage era's debugging. The noise floor and the safeguard are method (§3.7). Cut §5.5's opening paragraph; keep §5.5.1 (the held-out tail of the proposed model) under a plain heading. |
| §5.6 L1355–1357 | "its read-out admitted the leakage pattern that the causal read-out excludes" | Cut the clause. |
| §6 intro L1370–1374 | "Every task was run under the two-stage pipeline during development and again under the proposed model" | Cut. Report the tasks under the proposed model against their baselines. |
| §6.1 L1405–1412 | two-stage AUROC "0.94 → 0.98 at 3σ, 0.85 → 0.94 at 2σ" and "the advantage is larger under the proposed model than under the two-stage pipeline" | Cut the two-stage rows; keep the proposed model's AUROC against the structure-free baselines. Same for §6.4 "so the property transfers from the two-stage pipeline to the proposed model". |
| §7.1.4 L1657–1666 | "The synthetic pilot reverses the a-priori guess ... Real data gave the same answer ... with estimator octave failures present, the learned scale collapses" | State the choice and its evidence in two sentences: given variances are used because they are better calibrated than a learned scale on synthetic and on development data. |
| §7.3.1 L1736–1742 | synthetic pilot subsection | Fold into §7.1.3/§7.1.4 as one sentence of validation, delete the subsection. |
| §8.1.2 L2096–2097 | "The two-note pilot that preceded the study showed the same at 0.7–1.6 cents" | Cut. |
| §8.1.3 L2123–2134 | "A first integration study ... Waveform-native calibration, a bridge model ... is the frontier beyond that. This empirically established constraint, rather than the synthesis sketch, is the starting point" | Keep the fusion result with its visibility condition; cut the last two sentences (future work and self-narration). |
| §8.2.1 L2145–2156 | "The second round of studies implements ... in an automatic-differentiation framework, with equality to the reference implementation verified in unit tests ... gained +717 nats over 24 notes" | Rewrite as method: curve MAP by gradient, Laplace bands, evidence-fitted prior. Cut chronology, unit tests, the nats figure. |
| §8.2.2 L2192–2199 | "A perturb-and-recover pilot on two-note segments ... 7 of 8 pairs ... no phase pathology" | Keep the finding (acoustic versus musical onset, median 25 ms); cut the pilot framing. |
| §8.2.3 L2204–2215 | "The follow-up study models the transition between notes as a region ... The data reject the region hypothesis" | A rejected design. Keep one sentence: boundaries are sharp edges, free transition widths collapse to the minimum. Cut the rest. |
| §8.3.1 L2296–2312 | "If only its samples were masked ... a well of about 38 nats ... In a preliminary version of the protocol that masked only the samples, every arm's masked estimate was accordingly a leakage-driven optimum ... The lesson applies to every prior-value design" | This is the waveform-level leak history. Keep the design rule in one sentence (the hidden element is removed from the acoustic design, not only from the samples, because its window columns would otherwise carry evidence). Cut the preliminary version, the 38 nats and "the lesson". |
| §9.4.1–9.4.2 L2732–2790 | "The first study ... The second study repeated the first ... four seeds, the first two of which reproduce the first study exactly" | Present the development evidence as one four-seed study with two forms. The two-seed timing star becomes one sentence: a timing effect visible on two seeds did not hold on four, which is why articulation became the primary. Merge the two subsections. |
| §9.4.2 L2800 | "With the pitch term added, the per-channel form's articulation and velocity gains both have intervals that include zero" | A third arm nobody needs. Cut. |
| §10.3 L3031 | "Leakage was found and corrected" | Rewrite: "The evaluation path is audited for target leakage and filtered for cross-corpus contamination." |

## 2. Protocol narrated again outside §4.5

§4.5 states the protocol once. Results sections need the outcome and, at
most, a pointer.

| where | text | action |
|---|---|---|
| §1.2 L216–217 | "Its hypotheses were fixed before any test number was seen" | Keep (first statement), it is the claim of the protocol. |
| §5.1 L1077–1080 | "The decision rule fixed in advance adopted the proposed model if ..." | Keep: this explains a mixed outcome and is the one place it matters. |
| §7.4 L1815–1820 | "By the decision rule fixed in advance, with the three main hypotheses supported, the principal result ... Every number of the evaluation, all three systems and the per-entry evidence are archived with the run logs." | Cut the rule restatement and the archive sentence. |
| §7.4 caption L1824 | "(the held-out test set evaluated once, given-variance noise model, protocol fixed in advance)" | Cut "protocol fixed in advance". |
| §7.5 L1876–1883 | the whole opening paragraph on what was known when, archived targets, masks and seeds | Compress to two sentences and the pointer to §4.5. |
| §7.5.1 L1908–1913 | "The risk stated in advance, that this test set's timing channel is cleaner ... By the decision rule fixed in advance the hypothesis is supported." | Cut both. Keep the comparator sentence. |
| §7.5.1 L1895–1897 | "the correlated-noise option of Chapter 11 is numerically identical to this computation" | Cut (implementation). |
| §7.5.2 L1937–1947 | "By the decision rule fixed in advance ... The three plain arms reproduced Table 7.2 digit for digit ... Two descriptive results are recorded beside the hypotheses ... is suggestive, was not a pre-specified hypothesis, and is not pursued further. Of the two follow-up hypotheses, then, one was supported and one was not, and the held-out test set has now been used twice." | Keep the outcome and the secondary result. Cut the rest. |
| §8.3.1 L2314–2324 | "Hypotheses and a falsification rule were fixed in advance of each evaluation. Structure is predicted to be better ..." | These are development studies. State what is compared and what would count as structure adding information; cut the protocol language. |
| §8.3.2 L2380–2381 | "the pre-stated falsification condition is met" | Rewrite: "so the structured gain is the transfer of the shared offset". |
| §8.3.5 L3046–3051 (prior label) | "fixed in advance, following §4.5 ... used here for the third time, as §4.5 discloses" | Keep one pointer; the pre-stated expectation of −0.9 and the power stay (they bound the reading). |
| §9.3 L2720–2728 | "The hypotheses were fixed before each run. The first study predicted ... The second study predicted ... The third study predicted ..." | With the dev studies merged (§1 above), state the one development question and, for the predicted-occurrence study, the prediction and its outcome together in §9.4.3. |
| §9.5 L2866–2876 | power-estimate paragraph including the RMSE-definition caveat | Keep power 0.95 and 0.33 in one sentence; the definition caveat can go to a footnote or be cut (it does not change the reading). |

## 3. Self-commentary and defensive asides

| where | text | action |
|---|---|---|
| §5.5.1 L1345 | "the limitation is reported rather than patched" | Cut. |
| §6.5 L1460–1462 | "Both boundaries are statements this thesis makes, and each is bounded by evidence on both sides." | Cut. |
| §7.1.3 L1618 | "The missingness mechanism deserves naming" | Rewrite: "The missingness has two mechanisms." |
| §7.1.6 L1719 | "This is development evidence." | Cut; the paragraph already says development tracks. |
| §7.1.6 L1687–1700 | "There are two defensible responses. One folds ... The other withholds ... This is the single largest threat to validity in this setting, and it has no analogue on piano." | Cut the menu of alternatives; state what was done (annotated onsets anchor the warp, its error scale enters the noise row, its correlation is the subject of §7.5.1). |
| §7.4 L1864–1869 | "Relative to the piano setting ... so the extension shows that the same prior extends, on a held-out test set, to channels it was never designed for" | Cut (repeats the chapter opening and the result). |
| §8 opening L1984–1990 | "One asymmetry ... is part of designing the waveform setting in full rather than a detail" | Compress to the fact (loudness and articulation are not under the graph prior in this setting). |
| §8.3.3 L2416–2423 | "Setting the prior width at chain scale instead ... No single population-scale width calibrates all three arms, and a per-chain width fitted by evidence is the principled remedy, which was not pursued here." | Scoping; cut to "the masked bands are too wide at the population scale". |
| §9.2 L2683–2685 | "The second outcome is the family's own falsification route and needs no separate test." | Cut. |
| §9.4.3 L2842–2846 | "Of the two movements with precision below 0.2, one receives a large learned weight ... among the rest the size of the effect does not order by F1" | Cut. |

## 4. Implementation detail

| where | text | action |
|---|---|---|
| §3.7 L857 | optimiser (L-BFGS-B, log-Cholesky) | Keep log-Cholesky (it is how positive semidefiniteness holds); the optimiser name can go. |
| §3.8.1 L824 | "the equality is verified numerically" | Keep: it supports the nesting claim. |
| §5.2.2 L1165–1166 | "both are verified by unit tests" | Cut "by unit tests"; the identities follow by linearity. |
| §8.1.1 L2032–2066 | 16 kHz, 2 s cap, $p = 64$ columns, $O(mp^2)$ Woodbury "verified in unit tests", grid "1-cent ... 0.03 cents (0.06 ...)", Hann bumps with 5-cent prior sd | Compress to one paragraph: per-note segments cut at annotated boundaries, harmonic basis with chunked amplitudes marginalised exactly, grid inference over $c$, four position models. |
| §8.3.2 L2391–2392 | "kernel shape $s=1$, and about 150 s per fit including the Laplace approximation" | Cut. |
| footnotes L1110–1111, L1371–1372, L3234–3238, L3370–3371, L3504–3507 | repository file names (`docs/...md`, `results/...md`) | Cut all. A thesis does not cite its own working files. If a reproducibility statement is wanted, one sentence in §4.5 that code and records are available. |
| §1.5 footnote L290–292 | "In the code, the results files and the names of archived artifacts the four stages are numbered 0–3" | Cut. |
| App. A L3214–3217 | "An entry of the strings-and-winds mask is a (note, channel) pair, whereas the evaluation unit of the piano tables is a (piece, seed) combination and is always written out as a fit." | Cut. |

## 5. Scoping studies that do not serve the statement

Candidates for one sentence or removal. An examiner will want the first
three named; the rest can go.

| study | where | proposal |
|---|---|---|
| calibrated deep baselines | §5.6, App. C.3 (1 page) | Keep, short: it is the obvious rival. Cut the leave-one-out follow-up paragraph. |
| hand-built features versus music model | §5.6, App. C.2 | Keep one sentence in §5.6, delete C.2 (its numbers are in §5.2.1). |
| masking-level sweep | §4.6, App. C.1 + figure | Keep §4.6's three lines; cut C.1 and its figure. |
| kernel comparison in the two-stage regime | §3.2.2, §5.6, App. C.5 | One sentence in §5.6 (spectral profile inert, replacement metric worse, chord edges redundant under the proposed model). Delete C.5; move its label target. |
| theory columns and embedding probes | §11.5, App. C.6 | Cut both. The tonality probe is interesting but answers no question of the thesis. |
| task-matched pretraining | §5.6 | Cut. |
| tonal metric | §3.2.2 eq. (fifths, tonal), §7.1.7, §7.5.2 | Keep: it is a held-out hypothesis. The definitions in §3.2.2 can move to §7.1.7 where they are used. |
| chord graph | §3.2.2 eq. (chord), §5.6 | Cut the equation and the definition; one sentence in §5.6. |
| synthetic pilot (strings) | §7.3.1 | Fold into one sentence (see §1). |
| scalar-study position-model variants (AR(1), drift, deviation prior) | §8.1.1–8.1.2, Table 8.1 | Keep the table (it is the evidence for the estimand gap); trim the prose about each variant. |
| transition-region hypothesis | §8.2.3 | See §1. |
| chain-scale boundary width | §8.3.3 | See §3. |
| per-channel form with pitch term | §9.4.2 | Cut. |
| random masking scheme | §9.3, §9.4.1 | Keep the definition (it explains why the occurrence scheme is needed); cut "Under the random scheme the family is mostly invisible" lines. |
| era classification, performer identification | §6 intro | Either report them in one table row each (they are the evidence for the structure boundary) or drop the boundary's second half. Do not "record them elsewhere". |

## 6. Structural fat

1. **Chronology framing.** "stage 1 of the progression" (L1050), "stage 2 of
   the observation ladder" (L1468), "stage 3 of the thesis's progression"
   (L1954), "the destination of the programme" (L274), and the "Status"
   column of Table 1.1 are project-management language. Proposal: name the
   three settings by their observation (aligned MIDI, tracked audio,
   waveform) and turn Table 1.1 into "setting, observation, what is
   inferred, evidence", without "Implemented" or "the endpoint".
2. **The two-stage pipeline as ancestor.** See §1. Appendix B then becomes
   the definition of the baseline and can shrink into §3.8.1.
3. **"Across performers, not repertoire"** appears in §4.5, §9.5, §9.6,
   §10.2, the abstract and the stage table. Keep it in §4.5, §9.5 and §10.2.
4. **The attribution** (short time window, pitch term adds nothing on
   intonation) is stated in the abstract, §1.2, §1.4, §8.3.4, §9 opening,
   §10.2, Conclusion and App. D.3. Keep it in App. D.3 (evidence), §9
   opening (motivation), §10.2 (limit) and the abstract.
5. **"Development set" labels** are needed on numbers, not repeated as
   sentences ("All rows are exploratory results on the development set",
   "Each study is exploratory evidence from the development set on four to
   nine instruments").

## 7. Correctness items found during the read

1. **Figure 8.4 caption (L2229–2236)** still says "score-nominal starting
   points, and no annotation access in the fit", contradicting the corrected
   text (chains cut at annotated notes, boundary refinement given a local
   alignment).
2. **§4.7 L1040–1044**: "The pre-specified criterion for a structured prior
   requires improvement on both recovery and calibration against the
   independent, temporal-only, ridge and music-model-mean-only baselines."
   The piano held-out test compared the proposed model with the two-stage
   pipeline and its own ablations, not with those four baselines, and the
   adoption rule allowed NLL to be not significantly worse. The sentence
   describes the development criterion of the two-stage era. Rewrite it to
   describe what was actually tested.
3. **Title** says "with Calibrated Uncertainty". After the audits, the
   supported statement is coverage close to nominal for stated targets on
   piano and tracked audio, and under-coverage of the reference at the
   waveform. "with Predictive Uncertainty" or "with Uncertainty" would match.
   This is the author's call.
4. **§2.3 L363**: "every hyperparameter is learned by the exact
   evidence" survived the earlier fix (the graph geometry is fixed).

## 8. Expected effect

Rough page estimates on the 75-page build: process history 2 pages,
protocol restatements 1 page, self-commentary 0.75, implementation detail
1.25, scoping studies and Appendix C 3, structural repetition 1. About 9
pages, to roughly 66, before any of the Tier 2 cuts in
`docs/codex_trim_plan_2026-11-25.md`, some of which overlap with §5 here.
No held-out result, interval or disclosure is touched by any proposal
above.
