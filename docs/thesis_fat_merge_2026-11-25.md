# Fat audit, round 3: merged decisions (2026-11-25)

Inputs: `docs/thesis_fat_audit_claude_2026-11-25.md` (Claude) and
`docs/codex_fat_audit_2026-11-25.md` (Codex, independent). The two reads
agree on almost every passage: the leak story, the two-stage pipeline framed
as an ancestor, the protocol retold in results, the preliminary waveform
masking story, the serial motif studies, repository footnotes,
implementation detail and self-commentary. This file records where they
differed and how the revision resolves it, and the correctness items each
found.

## Where the reads differed

| question | Claude | Codex | decision |
|---|---|---|---|
| Table 1.1 (stages and status) | reframe as setting / observation / evidence | cut, replace with prose | **Cut.** Codex also found it mislabels loudness and counts four stages in five rows. |
| Chord-graph equation (3.2.1) | cut with the scoping studies | keep, it defines the strongest piano comparator | **Keep** (Codex is right), with $i \neq j$ added to the onset indicator. |
| Pre-stated expectation (−0.9) and power (0.3–0.5) in §8.3.5 | keep | cut | **Keep, one sentence.** They bound the reading of a small held-out effect. |
| Masking sweep, App. C.1 and figure | cut C.1 and figure, keep §4.6 | keep as sensitivity with the leave-one-out caveat | **§4.6 only**, carrying the caveat. C.1 and its figure go. |
| Theory columns, App. C.5 | cut | one-sentence control | **One sentence in §5.6**, section cut. |
| Deep baselines, App. C.3 | keep, short | keep the comparison, cut the follow-up and commentary | Same: first paragraph kept. |
| Where the piano adoption rule lives | §5.1 | §4.5 | **§5.1**, where it explains the mixed outcome. §1.2 and §4.7 no longer state a different criterion. |
| Student-t remedy | §11.1 | §11.1 only | §11.1, with a pointer from the tail section. |
| Downstream summary tasks | report in one line or drop | report in one line | One line each in the chapter opening. |

## Correctness items (both reads, verified against the records)

1. Piano: §1.2 and §4.7 stated a criterion (improvement on both axes
   against four baselines) that the held-out test never used. Fixed; the
   actual adoption rule stays in §5.1. (Claude, Codex)
2. The 4.5 and 17 timing errors in §5.5 were two-stage fits with collapsed
   scalar graph parameters, not the proposed model's coupling. The
   paragraph is cut. (Codex; `docs/phase1_calibration_results.md`)
3. Pooled piano RMSE concatenates seconds, a log ratio and normalised
   velocity, so articulation dominates it. §4.7 and §5.3 now say so; deck
   slide 13 and its note follow. (Codex; `features.py`, `eval_graphgp.py`)
4. Architecture captions claimed only the likelihood changes between
   settings, and drew the waveform system as if implemented. Fixed. (Codex)
5. Chord equation lacked $i \neq j$; "every hyperparameter" in §2.3. Fixed.
   (Codex, Claude)
6. Table 1.1 errors. Table cut. (Codex)
7. Figure 4.1 called the annotated-curve estimate "truth"; §7.3 mixed
   predictive coverage against estimator targets with latent coverage
   against references. Both labelled. (Codex)
8. Held-out URMP table printed stars instead of intervals; §4.5 said
   significance is never marked. Table now prints both endpoints; the
   correlated-noise coverage contrast, the tonal loudness contrast and the
   waveform family intervals are added. (Codex)
9. Scalar waveform table said "no pitch tracker in the loop" while the
   vibrato scaffold comes from the tracked estimator. Fixed. (Codex)
10. Figure 8.4 caption still said "no annotation access in the fit"; the
    width comparison's sample (128 non-unison boundaries, two starts) now
    stated. (Claude, Codex)
11. Phrase selection also excludes distance-one recurrences and caps four
    phrases per track; the clean control's actual criterion stated. (Codex)
12. "No element-level information" read from an interval that includes zero
    becomes "none detected". (Codex)
13. The motif scope reason ("no annotated movement outside the development
    set has an aligned performance") ignored the two movements reserved for
    the piano test. Fixed. (Codex)
14. §10.1 said features have least to say on articulation; the measured
    shares put the combined feature share lowest on velocity. Clause cut.
    (Codex)
15. "Every interpolation-style task", the notation table's "σ has no other
    use", and the evaluation-unit sentence. Fixed or cut. (Codex)
16. The Conclusion ran the untested whole-performance loop and the tested
    strong form together. Separated. (Codex)
