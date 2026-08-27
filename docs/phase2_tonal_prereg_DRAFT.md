# DRAFT — tonal-metric confirmation design (NOT REGISTERED)

> **Status: PROPOSAL ONLY (2026-08-27).** Nothing here is frozen. This
> document exists so the corpus decision can be discussed concretely at
> the next meeting; registration happens only after that decision, as a
> separate commit + tag, before any confirmation data is touched.

## What would be claimed

The development result to confirm (`results/phase2_tonal_dev.md`,
exploratory, hypothesis pre-committed 2026-08-17): replacing the pitch
metric of the score graph with circle-of-fifths (tonal) distance helps
the intonation channel on both axes and re-imposes the known replacement
penalty on timing.

Candidate claims (tonal vs plain graph, as-given, paired per (track,
seed)):

- **T1 (intonation recovery):** paired dRMSE on `c` negative and starred.
  Development basis: −0.213 [−0.339, −0.085]*.
- **T2 (intonation calibration):** paired dNLL on `c` negative and
  starred. Development basis: −0.050*.
- **T3 (honest boundary, pre-stated):** the timing penalty is expected
  (dev +0.027*); it is *predicted*, reported, and does not gate T1–T2.
  Adoption, if T1–T2 pass, is channel-scoped: tonal metric for the
  intonation channel's graph only, plain metric elsewhere — matching the
  per-channel ARD philosophy (the evidence already switches kernels per
  piece; this switches a metric per channel).

## The corpus decision (the open question — professor's call)

The 13-piece URMP confirmation pool was spent on 2026-08-27 (Phase-2
claims C1–C4). Three options for the tonal confirmation, honestly
characterized:

1. **Reuse the spent URMP pool, disclosed.** The tonal-vs-plain contrast
   was never evaluated on it, so the specific claim is untested there;
   but the data are no longer untouched (the plain-graph fits and
   targets have been seen), so this is *replication-grade*, not
   pristine-confirmation-grade. Cheapest; weakest discipline. If chosen,
   the thesis must label it "confirmation on reused pool (disclosed)".
2. **Fresh corpus: Bach10** (10 chorale pieces, vn/cl/sax/bn stems with
   frame-level GT F0 + note annotations — the standard URMP companion).
   Pristine one-shot discipline; requires porting the loader + tracker
   calibration first (the 2–5 cents/instrument check must be re-run
   there before trusting the estimator). Moderate cost (~2–3 days of
   development-side groundwork before registration), strongest claim.
   Risks: different recording conditions; woodwind-heavy (only one
   string instrument), so T1 would be dominated by non-string
   intonation; smaller n (10 pieces → wider CIs; dev effect −0.21 with
   CI half-width ~0.13 at n≈150 pairs suggests n≈40–60 pairs may be
   marginal — a power check on dev subsamples is required before
   freezing).
3. **Both:** register Bach10 as primary, the reused URMP pool as a
   disclosed secondary replication. Most complete; most work.

Recommendation to bring to the meeting: **option 3 if the power check
passes on Bach10's n; otherwise option 1 with prominent disclosure.**

## Protocol constants (would be frozen at registration)

Identical to the Phase-2 registration except: systems = tonal graph GP
(as-given) vs plain graph GP (as-given), paired; the no-graph ablation
reported for context; 30% hidden; seeds (0, 1); B = 2000, rng 31, 95%
CI; OMP_NUM_THREADS=4; sharded runner (`run_phase2_eval.sh` tonal mode /
`eval-tonal` verbs already exist and are shard-equivalence-tested).
One shot, every number reported.

## Pre-registration checklist (before the tag)

- [ ] Corpus decision made with the professor.
- [ ] If Bach10: loader + MD5 manifest; tracker calibration table
      (dev-side); composition-level split N/A (10 distinct pieces) but
      contamination check vs URMP/MAESTRO repertoire.
- [ ] Power check: subsample dev to the confirmation's expected n and
      verify T1's dev effect would still star (else the registration is
      set up to fail for sample-size reasons, not truth reasons).
- [ ] Freeze claims T1–T3 verbatim + decision rule + this protocol in
      the registration commit; tag `phase2-tonal-registration-<date>`.
