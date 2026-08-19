# Meeting prep — study plan + anticipated questions (2026-08, private prep sheet)

For Ray only. Every answer is grounded; pointers name the evidence. The
frame stays modest: development-labeled results, registered claims, pool
unspent. Page numbers refer to the current 54-page `docs/thesis/draft.pdf`.

## Study plan (~75 minutes, in order)

1. **First (5 min): sync Overleaf with current `main`** — what the professor
   opens must be this state, not July's.
2. **§3.9, p. 19–24 (20 min).** Read once, slowly. The arc: nothing in the
   prior changes; the channel set, the targets, and the noise do. The two
   channel decisions are measured (delay in with 95/97% coverage and 18 ms
   truth-agreement; τ via annotated onsets); the τ threat and its diagonal
   -noise limitation are stated, not hidden.
3. **The figure, p. 25 (10 min).** Practice narrating the four panels aloud
   once — intonation band + hidden circles; extent with GP-filled squares;
   τ with aligner error bars (point at the wide edge note yourself, before
   anyone asks); delay with GP-filled squares (sub-zero excursions are GP
   extrapolations; every observed delay is nonnegative).
4. **The results paragraph, p. 23–24 (10 min).** Memorize the know-cold
   table below, including the adverse cell.
5. **The tonal passage, p. 22 (5 min).** One sentence to say: "the metric
   that hurt expression helps exactly where the target is pitch — and it
   re-imposes the timing penalty, which is what the hypothesis predicted."
6. **The Q&A below (20 min).** Read twice; say the loudness and δ_vib
   answers out loud once.
7. **The asks (5 min).** They are the planning-flavored close.
   (Optional depth if he digs into Phase 1: §5.3, p. 30 — shares .69/.60/.40,
   graph×embeddings ≈ 0, complements not rivals.)

## Numbers to know cold (as-given variant, paired vs no-graph, dev)

| what | value |
|---|---|
| intonation recovery | −0.89 cents\* |
| vibrato extent / rate recovery | −0.26\* / −0.30\* |
| vibrato calibration (dNLL) | −3.8\* / −0.43\* |
| timing calibration (dNLL) | −0.29\* |
| coverage @ 90%, all six channels | 0.88–0.91 |
| the adverse cell | loudness dNLL +0.04\* against |
| tonal − plain, intonation | −0.21\* RMSE, −0.05\* NLL |
| registration | tag 2026-08-17; pool of 13 UNSPENT |

**Q: The targets are estimator outputs — what does "recovery" even mean?**
A: Exactly that, and the thesis says so wherever numbers appear: recovery =
agreement with the estimator, weaker in kind than Phase 1. The quasi-truth
cross-check (same NLLS on URMP's ground-truth pitch) gives the same system
ordering on every channel. (§3.9 "what recovery can mean"; results tables.)

**Q: Why estimator variances as-given instead of learned?**
A: Measured twice: the synthetic pilot preferred as-given, and real data
re-confirmed it the hard way — with octave-failure cells present the learned
scale collapses, and after the failure rule as-given is still the
better-calibrated variant vs quasi-truth (0.82–0.86 vs 0.72–0.86 coverage).

**Q: Any cell against you?**
A: Yes, one, and it is named in the draft: loudness under the default
variant — recovery ns and NLL +0.04 *significantly* against the graph.
Loudness carries no claim. Timing and delay recovery are also ns (their
value is calibration / bundle membership respectively).

**Q: Two mask seeds — enough?**
A: Measured (fresh seeds 2, 3, same protocol): every direction reproduces
and the recovery deltas match to two decimals (−0.898\*/−0.254\*/−0.299\*
vs −0.891\*/−0.256\*/−0.300\*); the adverse loudness cell replicates
(+0.048\*). The stars on heavy-tailed contrasts move: extent dNLL −3.8\* →
−0.22 ns, τ recovery ns → starred. Honest reading: point estimates are
stable, the extent-channel NLL star is seed-sensitive — so C2's extent
half is the riskiest registered claim, known before the pool is spent.
(`results/phase2_seeds23_dev.md`.)

**Q: Why does δ_vib get no claim if you adopted it?**
A: Adoption and claims are separate gates. The delay is measurable
(95%/97% coverage, 18 ms agreement with truth — criterion committed before
the numbers), so it belongs in the bundle; its graph contrasts are neutral
on dev, so registering a claim on it would be claim-shopping.

**Q: Where does alignment error go in τ?**
A: Into the noise row — the LOO tempo line's predictive variance. A diagonal
noise carries the error's scale but not its correlation along score time;
that stated limitation is why the registered τ claim is calibration-only.

**Q: Gaussian tails?**
A: The known Phase-1 limitation (one τ-outlier cell cost the confirmation
NLL tie). A Student-t prototype exists and is future work gated on its own
confirmation set — not applied post hoc.

**Q: When do you spend the confirmation pool?**
A: That is one of my asks today. It is registered (frozen claims, one shot,
every number reported), staged to run in an afternoon, and untouched.

**Q: What if a registered claim fails?**
A: It is reported verbatim next to the claim; the pool is spent either way;
Phase 1 stands on its own and Phase 2 was scoped so a negative costs the
thesis nothing.

**Q: The circle-of-fifths result — will you adopt the tonal metric?**
A: It is exploratory by the registered design. Adoption would be a
channel-dependent metric choice needing its own preregistered confirmation.
What it already shows: the geometry helps exactly where the target is pitch
(intonation, both axes) and re-imposes the known penalty on timing.

**Q: Why did the tonal metric hurt piano expression but help intonation?**
A: Expression travels register proximity; temperament and a shared tuning
reference travel the circle of fifths. The intonation channel is the first
whose target *is* pitch — that is why the thesis called this the cheap
decisive test before running it.

**Q: External validity?**
A: URMP: 13 instruments, three families; the vibrato win holds within every
family, intonation within strings and woodwind (brass ns on its own).
Beyond URMP the alignment problem returns — stated future work.

## The three asks
1. The Phase-2 claim set is preregistered — want to review it before the
   pool is spent, and when should we run it? (Worth raising: the fresh-seed
   check shows C2's extent half is the seed-sensitive claim — the one-shot
   could fail it honestly.)
2. Is the modest claim posture (no δ_vib claim, calibration-first) right
   for the committee?
3. Next priority: tonal-metric confirmation or Phase-3 scoping?

## In the room
- Have open: p. 25 (the figure) and p. 23 (the results paragraph).
- Opening line: "I picked things back up and focused on Phase 2."
- If a question stumps you: "that's measured — let me follow up with the
  exact number" is a fine answer; everything above has a file behind it.
