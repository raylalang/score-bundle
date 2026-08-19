# Meeting prep — anticipated questions, one-line answers (2026-08, private prep sheet)

One page, for Ray only. Every answer is grounded; pointers name the evidence.
The frame stays modest: development-labeled results, registered claims,
pool unspent.

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
A: The paired unit is (track, seed): n = 148–153 pairs with bootstrap CIs.
A fresh-seed dev check (seeds 2, 3) is running; expectation is ordering
stability, and the result will be quotable either way.

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
   pool is spent, and when should we run it?
2. Is the modest claim posture (no δ_vib claim, calibration-first) right
   for the committee?
3. Next priority: tonal-metric confirmation or Phase-3 scoping?

## Before the meeting
- Sync Overleaf with current `main` (pushed repo ≠ synced Overleaf).
- Pages to have open: §3.9, sec:phase2-real results paragraph (know the
  loudness cell), fig:phase2-real (now four panels), the tonal passage.
