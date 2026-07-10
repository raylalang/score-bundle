# Handoff to Claude Code — thesis restructure, vocabulary, figures

*Written by the Cowork session, 2026-07-10, on top of `dff3185`. Everything below is
either already done (and left **uncommitted** in the working tree) or requested.
Delete this file once consumed.*

---

## 0. State you are inheriting

`git status` shows an uncommitted working tree from Cowork:

```
 M .gitignore
 M docs/kernel_comparison_experiment.md
 M scripts/make_digest_figures.py
RM docs/draft.tex            -> docs/thesis/draft.tex
RM docs/figures/*.png        -> docs/thesis/figures/*.png     (7 files, tracked as renames)
```

Please **review and keep** these; do not revert. Details in §1–§3.

**Hard constraints for everything below.**

1. **The confirmation set is spent.** Nothing may be re-run, tuned, or evaluated on
   confirmation pieces 31–50. No task here requires it.
2. **The preregistration block and the confirmation-results block in
   `docs/graphgp_first_design.md` are frozen text.** Do not reword them, including
   their use of the phrase "old adopted headline." The renaming in §5 applies to
   thesis prose *outside* those blocks.
3. **This is a prose and structure change. No number may move.** After the work,
   every RMSE/NLL/coverage/CI in `draft.tex` must be byte-identical to before. If a
   number changes, stop and report — it means something was mis-transcribed.

---

## 1. Layout: `docs/thesis/` is now self-contained (done)

The professor reads the PDF via Overleaf, not the repo. `docs/thesis/` is now the
complete, standalone LaTeX project:

```
docs/thesis/draft.tex
docs/thesis/figures/*.png   (7)
```

Changes made:

- `\graphicspath` simplified to `{{figures/}}`; the dead `docs/figures/` fallback
  branch removed from `\figph`.
- `scripts/make_digest_figures.py`: `OUT = "docs/thesis/figures"`.
- `.gitignore`: negation updated to `!docs/thesis/figures/`.
- `docs/kernel_comparison_experiment.md`: path references updated.

Verified: compiles standalone from inside `docs/thesis/` with plain `pdflatex`
(two passes) — 49 pp, 0 errors, 0 undefined references, 0 missing figures.
Please keep it self-contained: no `\input` outside the folder, no `.bib`.

**Note on `\figph`.** Its missing-figure fallback prints the filename via
`\texttt{\detokenize{#1}}`. This is deliberate — filenames contain `_`, and without
`\detokenize` every missing figure throws `Missing $ inserted`. Do not "simplify" it.

---

## 2. Vocabulary: internal terms removed from prose (done — please maintain)

A sweep removed project-internal vocabulary from `draft.tex` **and** from the figure
labels in `make_digest_figures.py`. Canonical terms from now on:

| Do not write | Write |
|---|---|
| GP-first, single-GP model | **the proposed model** |
| a_diag, a_icm, b_feat, b_featlm, c_harm, d_hybrid, GP-feat | descriptive row labels only |
| guard | **safeguard** (the `sec:guard-gp` label is unchanged) |
| LM | **music model** |
| dev, dev-set | **development set / development record** |
| ladder | **ablation sequence** |
| marglik | **marginal likelihood** |
| disentangler | **attribution control** |
| Stage-2 | **task-matched pretraining** |
| leak-free / target-blind | *(drop the coinage; the property is already stated)* |
| rmse@50% | RMSE on the confident half |
| one code state | a single code version |
| `\texttt{logs/…}`, `\texttt{docs/…}` in prose | "the accompanying evidence archive" |

Two things that are **not** jargon and should stay: `two-stage` (a real term of art —
two-step estimation, plug-in trend + residual kriging) and `coregionalization`.

`ICM` is expanded on use. `$\mu_{\mathrm{LM}}$` keeps its subscript as *notation*
(the notation appendix glosses it as "music-model mean") — do not rewrite it to
`\mathrm{music model}`; that was a bug I introduced and fixed.

Also fixed: the executive summary's downstream table was still carrying two-stage-era
verdicts (era "0.37 vs 0.489", completion "extrapolation neutral") that contradicted
Chapter 10's re-validation. It now matches `tab:downstream-both`.

---

## 3. The main request: restructure Chapter 7 (Model and Formulation)

### The problem

Chapter 7 currently builds a **complete working model** across §§7.3–7.6 — per-channel
graph GP, plug-in mean, per-channel posterior, per-channel empirical Bayes — and then
in §7.7 replaces it with a different one. Because a reader would otherwise take
§7.5's posterior as *the* posterior of the thesis model (it isn't; it's the decoupled,
precision-side one), I patched those four sections with a `(development form)` marker.

That marker is a bad fix. "Development form" describes *when we built it*, not what it
is — exactly the kind of internal vocabulary §2 removes. And it can't simply be
deleted, because the **ordering** is what makes it necessary.

### The fix

Restructure the chapter from *"simple model → general model"* to
*"components → assembly → constraints."* Nothing is ever superseded, so nothing needs
a scope marker, and the term evaporates.

**Target structure:**

1. **Setting** — score, per-note variables `y_i = [τ, log r, v]`, the warp. *(unchanged)*
2. **The score graph** — `W_ij`, Laplacian `L_G`. *(unchanged)*
3. **The graph kernel** — spectral construction `K_G(s) = U g(ν; s) Uᵀ`, shape-normalized;
   the additive / Matérn / diffusion family; cite Borovitskiy, Lindgren. Present it
   **covariance-side from the start**.
4. **Coupling the channels** — the coregionalization matrix `B`, carrying all scale.
5. **Side information as kernels** — score features and the music model's mask-aware
   per-note embeddings entering as linear kernels `diag(c_f) ⊗ X_f X_fᵀ`; state that a
   linear kernel *is* the marginalized Bayesian linear mean, and contrast it in one
   paragraph with a plug-in mean (forward-reference §8).
6. **Observation model and posterior** — noise, the masking convention (held-out notes
   contribute no likelihood term), exact conjugate posterior for the **full** model,
   and the floored predictive variance. One derivation, no per-channel detour.
7. **Hyperparameters** — `s`, `B`, `c_f`, `ς²` learned **jointly** by one exact
   per-piece marginal likelihood. The noise floor and the safeguard live here.
8. **Constraints and special cases** *(new)* — define the switches: `B` diagonal
   (channels independent), drop the feature kernels (zero mean), freeze the graph
   shape, plug the mean in rather than marginalize it. Show that turning them all on
   simultaneously yields the classical **two-stage plug-in baseline**, and *derive the
   per-channel precision posterior here* — `Σ_y = (Q_G + P)⁻¹`, `m = Σ_y(Q_G μ + P ỹ)`
   — as the consequence of those constraints. Keep the existing statement that the
   equality is pinned by a unit test. Keep the special-cases table (`tab:system-posteriors`)
   but rename its rows by **constraint**, not by history.
9. **Evaluation: recovery and calibration** — *(unchanged)*
10. **Phase 2 / Phase 3** — *(unchanged)*

**Delete** every `(development form)` marker (§§7.3–7.6 titles) and the interim
scoping sentences I added to them; they exist only to paper over the old ordering.

**Preserve verbatim:** the `\label{}`s (`sec:graphgp`, `sec:posterior`, `sec:eb`,
`sec:gpfirst`, `sec:guard-gp`, `eq:gpfirst`, `eq:additive`, `eq:matern`, …). Many are
cross-referenced from Parts I and II; if a label must move, update every `\ref`.

---

## 4. Why the two-stage baseline stays — and how to frame it

It is **not** kept for narrative continuity. Frame it as intentional, on three logical
grounds, and say so in *Constraints and special cases*:

1. **It is the null.** Every claim in the thesis has the form "relaxing constraint X
   helps." Testing that requires the corner of the constraint space where all the
   constraints hold at once. That corner would exist even if the full GP had been
   written on day one; it is the origin of the ablation grid, not a memento.
2. **It is what the literature does.** Plug-in trend plus residual GP is standard
   practice. Beating it makes the headline a claim *about the field*, not about our
   own past: *"the coupled, marginalized model beats the standard construction, on
   pieces untouched by any modelling decision, under a claim registered in advance."*
   That is strictly stronger than "the new model beats the old one," and it is
   history-independent.
3. **It wins one regime.** For rendering from a short excerpt, the cross-piece plug-in
   head is the better estimator, because pooling generalizes where per-piece adaptation
   cannot. That is a live recommendation about a constraint (see the extrapolation
   boundary), not a historical note.

---

## 5. Renaming: retire "the old headline"

Outside the frozen blocks (constraint 2 above), replace every historical name with a
descriptive one:

- "the old headline", "the previous headline", "the then-adopted headline",
  "the two-stage era" → **"the two-stage plug-in baseline"** (or, where the strongest
  configuration is specifically meant, **"the strongest constrained configuration"**).
- In the ablation and headline tables, name rows by their **constraints** — e.g.
  "diagonal `B`, plug-in mean, fixed graph" — rather than by when we adopted them.
- The confirmation comparison should read: *the proposed model versus its strongest
  constrained special case.*

Where the prose must refer to what was preregistered, it is fine and honest to say
"the system preregistered as the comparison target (the strongest constrained
configuration)". Do not restate the frozen block's wording as if it had been different.

---

## 6. Figures

`scripts/make_digest_figures.py` label strings are **already corrected** ("proposed
model (full)", "proposed model, no graph", "network (music model)", "Reliability —
proposed model"). I regenerated five of seven figures from `evidence/`:

- ✅ `digest_headline`, `digest_channels`, `digest_kernels`, `digest_collapse`,
  `gpfirst_confirmation`
- ❌ `gpfirst_reliability`, `gpfirst_pit` — **you must regenerate these.**

`fig_gp_calibration` needs the *development* shards
`results/graphgp_v2/b_featlm.shard*.pkl`, which are **not in `evidence/`** (only
`graphgp_v2_report.log` is). Those two PNGs therefore still render the old title.
Just rerun `fig_gp_calibration`; no label edits needed.

**Please also:**

- **Rename the figure files** `gpfirst_confirmation.png`, `gpfirst_reliability.png`,
  `gpfirst_pit.png` → `proposed_*.png`, updating `make_digest_figures.py` and the
  `\figph{...}` / `\includegraphics{...}` calls in `draft.tex`. The filenames still
  carry the retired name.
- After the Chapter 7 restructure, check that any figure **axis label or legend**
  referring to a system uses the constraint-based names from §5.

---

## 7. An evidence gap worth closing

`evidence/` cannot currently reproduce two thesis figures, because the development
shards (`results/graphgp_v2/`) were not archived. Either archive them, or state the
limitation explicitly in `evidence/README.md` (development artifacts are regenerable
from code + data, unlike the one-shot confirmation cells — so this is a convenience
gap, not an integrity one; but a reader should not have to discover it by failing).

---

## 8. Verification checklist before you commit

- [ ] `cd docs/thesis && pdflatex -halt-on-error draft.tex` twice → 0 errors,
      0 undefined references/citations, 0 "missing figure" boxes.
- [ ] Also builds under `tectonic`.
- [ ] Compiles in a **fresh empty directory** containing only `draft.tex` + `figures/`.
- [ ] `grep -icE 'GP-first|single-GP|development form|restricted form|old headline|previous headline|\bdev\b|marglik|ladder|disentangler' docs/thesis/draft.tex` → 0.
- [ ] Every number in `draft.tex` unchanged versus the pre-restructure version
      (`git diff` should show prose and structure only; diff the rendered numbers if unsure).
- [ ] Frozen blocks in `docs/graphgp_first_design.md` untouched (`git diff` clean there).
- [ ] `pytest` green; both leak audits pass bitwise.
- [ ] No file under `results/graphgp_conf*` or the confirmation inputs modified
      (check mtimes).

---

## 9. Not yours

Cowork keeps: the Notion page (already rewritten, GP-first, lean; a frozen archive
snapshot of the two-stage era lives on a separate page), and the thesis-facing prose
once you hand the restructured chapter back. Raynaldi keeps: the thesis statement,
title page, related-work narrative, acknowledgments.
