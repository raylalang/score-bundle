# Handover — for the review session & the Notion update

*Written 2026-07-10 by the Claude Code session that did the work described below.
Audience: Raynaldi + Claude Cowork. Purpose: (1) orient your joint review, (2) give
Cowork everything needed to write the Notion page. Delete this file once consumed.*

## What happened, in one arc

The thesis model changed this week, and the change was confirmed on untouched data.

1. **Two-stage era (through 2026-07-09 morning).** The pipeline was: LM/feature
   plug-in mean → per-channel score-graph GMRF on the residual. Its development
   optimum: feat+LM mean + chord/voice-leading graph, RMSE 0.379 (dev), adopted
   after a kernel comparison and a bitwise zero-leak audit.
2. **Reformulation (2026-07-09).** A review of the graph-GP literature
   (Borovitskiy 2021; Venkitaraman 2018; e-GGP 2021 — PDFs in `related_works/`)
   identified three orthodoxy gaps: independent channels, fixed graph parameters,
   plug-in instead of marginalized mean. Closing them gives **one multi-output
   graph GP** (`src/score_bundle/gp.py`): coregionalized channels ⊗ spectral graph
   kernel + score features and mask-aware LM embeddings as linear kernels (=
   marginalized Bayesian linear mean) + per-channel floored noise, all fit by exact
   per-piece marginal likelihood. The old pipeline is **provably a nested special
   case** (unit tests pin the equality).
3. **Preregistered one-shot confirmation (2026-07-09).** Protocol, systems, claims,
   and decision rule were committed BEFORE the data existed (see the preregistration
   block in `docs/graphgp_first_design.md` — do not edit it); 20 fresh ASAP pieces
   (cache indices 31–50), run exactly once. Outcome: **RMSE 0.376 vs 0.393**
   (paired −0.0137, significant), graph's calibration contribution −0.074
   (significant), coverage 92.5%; the pooled-NLL comparison tied — one piece with a
   few extreme timing outliers (diagnosed precisely; median piece is
   better-calibrated under the GP). Decision rule ⇒ **GP-first is the thesis
   model.**
4. **Hardening + restructure (2026-07-09→10).** One-code-state rerun of every
   config; an attribution disentangler (the win comes from per-piece Bayesian
   feature weighting, NOT from coupling per se); all six downstream tasks
   re-validated under the GP; 12-seed dev robustness (everything holds); a
   three-reviewer adversarial sweep with all findings fixed; the thesis draft
   restructured around the GP-first model with the two-stage work as its labeled
   development/ablation record; the draft **compiles clean** (tectonic is installed
   in the `score-bundle` env; `docs/draft.pdf`, 47 pp).

## Current state

- **Branch `thesis-gpfirst-restructure`** = the shippable state. Merged into
  local `main`: NOT yet (that + push is yours after review). **Nothing has been
  pushed** since the GP-first restructure.
- `main` (pushed) still shows the two-stage story + a digest addendum; merging
  will carry the branch's deletions (old meeting digests, status snapshot) — they
  remain in git history only, per Raynaldi's decision.
- Full pytest green; both leak audits pass bitwise; confirmation artifacts
  untouched since first written (verify: mtimes under `results/graphgp_conf*`).

## Where every number lives (cite logs, never memory)

| Claim | Source |
|---|---|
| Confirmation table (headline) | `logs/confirmation_verdict.log` |
| Dev ladder + per-channel (v2, one code state) | `logs/graphgp_v2_report.log` |
| Attribution / disentangler, paired contrasts | `logs/graphgp_v2_report_vsbfl.log` |
| 12-seed dev robustness | `logs/dev12_report.log` |
| Downstream re-validation (all six tasks) | `logs/downstream_gpfirst_report.log` + `logs/overnight_performer.log` |
| Guard A/B + τ-tail diagnosis | `logs/guarded_ab_verdict.log` |
| Two-stage development record | `docs/phase1_calibration_results.md`, `docs/kernel_comparison_results.md` (banners explain their status) |
| Student-t prototype (dev-only) | `logs/robust_tau_devcheck.log` |

Primary doc: `docs/graphgp_first_design.md`. Thesis: `docs/draft.tex` (+ built PDF).

## For the Notion page (suggested structure)

Add a dated entry (2026-07-10) to the "Score-Bundle Models" note, §1 status list
(same toggle style as previous entries). Content, in the note's plain-language
register:

- **The model changed, and the change was tested the hard way.** One Gaussian
  process on the score graph now does everything the old two-part system did — the
  old system turned out to be a special case of it. Before switching, the claims
  were written down in advance and tested exactly once on 20 pieces no decision had
  ever touched: the new model recovers hidden notes better (error 0.376 vs 0.393,
  significant piece-by-piece), its confidence intervals stay honest (92.5% at the
  nominal 90%), and the score graph's contribution to honest confidence was
  confirmed independently.
- **What we learned about why:** the accuracy comes from letting each piece weigh
  its own evidence about how expression follows the score; the honesty of the
  error bars comes from the graph. The music model earns its place as a feature
  inside the kernel.
- **All six demonstrations re-checked under the new model** (error-spotting
  stronger; denoising/selective transfer; era and performer-ID negatives stand);
  one new boundary found and documented: adapting to a piece needs enough of the
  piece — rendering from a short excerpt stays with the old cross-piece predictor.
- **Honesty items (do not omit):** the pooled confidence-quality comparison TIED on
  fresh pieces (one outlier-heavy piece; documented with remedies as future work);
  the development numbers flatter every model (that's why the confirmation set
  exists); nothing about the old published numbers changed — they are the
  development record.

## Rules that govern this content (please keep them)

1. **The confirmation set is spent.** Nothing may be run/tuned/evaluated on cache
   pieces 31–50 again; a future *second* confirmation set is the only honest venue
   for new confirmed claims (e.g. the Student-t prototype).
2. **Dev numbers are always labeled development.** They adjudicated model
   selection; presenting them as unbiased is the one misrepresentation this whole
   effort exists to prevent.
3. **The preregistration block and confirmation-results block in
   `docs/graphgp_first_design.md` are frozen text.**
4. Recompile before sharing the PDF: `cd docs && tectonic draft.tex`.

## Open items (owner: Raynaldi)

- Review → merge `thesis-gpfirst-restructure` → main → push.
- Notion entry (Cowork, per above).
- Supervisor conversation about the reformulation (surface-level is fine — the
  digest-style summary above is the script).
- Thesis prose that must be in your voice: title/name page finalization, related-work
  narrative depth, acknowledgments.
- Parked research: Student-t τ likelihood (prototype exists, dev-checked),
  extrapolation-safe completion, Phase 2/3, connection Laplacian.
