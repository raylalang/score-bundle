# Figure brief — the three pipeline/architecture diagrams

*Written by the Cowork session, 2026-07-10, for whoever works on these next (Claude Code
or otherwise). Everything a cold start needs: what the figures must argue, the design
language, the constraints, and the parts only the code environment can finish.
Delete once consumed.*

---

## 1. What these figures are for

Not decoration. Each figure must carry an argument that the prose also makes, so that a
reader who only looks at the pictures still gets the thesis's structural claim:

> **Across all phases, only the likelihood changes. The graph-GP prior is reused unchanged.**

That is the load-bearing idea. Every design decision below exists to serve it. A figure
that merely labels boxes has failed even if it is pretty.

Corollaries the figures must also encode:

- Phase 1's **visible/hidden split is an evaluation device, not part of the model** —
  held-out notes simply carry no likelihood term.
- Phase 3 has **no visible/hidden split at all**: given only the score and the audio,
  every note's expression is latent and inferred jointly. Conjugacy is lost there.
- The model is **two branches that fuse**: graph *structure* and *evidence* (side
  information). This mirrors the attribution result — the graph makes the confidence
  honest; per-piece Bayesian feature weighting wins the accuracy.

---

## 2. Two candidate styles (Ray decides; do not delete either)

| Style | Files | Character |
|---|---|---|
| **A — restrained schematic** | `pipeline_phase{1,2,3}.{tex,pdf,svg}` | Thin rules, near-monochrome, one accent. Dashed enclosure = shared prior. Reads like a methods figure. |
| **B — architecture-diagram** | `arch_phase{1,2,3}.{tex,pdf}` | Isometric volumes, Okabe–Ito colour, two colour-coded branches, circled operators, glyph outputs. Reads like an ML-paper architecture figure. |

Style B follows a reference the user supplied (`hybrid_architecture_sample.svg`, a
CNN+Transformer hybrid diagram). **The choice between A and B is taste and belongs to
Ray.** Do not consolidate them unilaterally.

---

## 3. Design language of Style B (derived from the reference)

- **Palette: Okabe–Ito, colourblind-safe.** `#0072B2` blue, `#E69F00` amber, `#009E73`
  green, `#CC79A7` pink, `#D55E00` vermillion, greys `#333/#777/#9A9A9A/#555`.
- **Isometric cuboids = data volumes.** Three faces (front rect, top parallelogram,
  right parallelogram), tinted 45 / 22 / 65 % of the branch colour.
- **Rounded rectangles = operations.** `rx≈2pt`, 12 % fill of the branch colour.
- **Offset stacked rects = repetition** (the music model's `×L` blocks).
- **Small italic grey annotations *under* every volume** giving its shape (`N×N`,
  `N×26`, `3N×3N`). This is what makes the figure teach rather than label.
- **Curved Bézier arrows** for forks and fusion; straight for in-branch flow.
- **Circled operators** for `⊗` (Kronecker lift) and `⊕` (sum of terms).
- **Glyph outputs, not text boxes.** Phase 1/2 output = three point estimates with
  error bars (this *is* the contribution: calibrated per-note uncertainty). Phase 3
  output = error bars plus a drawn resynthesis waveform; Phase 3 input = a drawn
  damped-oscillation waveform.

### Semantic colour assignment — do not shuffle

| Colour | Meaning |
|---|---|
| blue | **graph structure** branch: score graph → `L_𝒢` → spectral shaping `g(ν;s)` → `K_G` |
| amber | **evidence** branch: score features → `X_feat`; music model (×L) → `H`; both → linear kernels → `X_f X_fᵀ` |
| green | **fusion**: `⊕` summing Kronecker terms + noise → joint kernel `K` |
| pink | **inference head** |
| **vermillion** | **the likelihood — the only element that changes across phases** |

The vermillion block is the single most important thing in the figure. It must be the
only element whose content differs between Phase 1, 2 and 3, and it must be visually
distinct enough that a reader flipping between the three sees *exactly one thing move*.

Dashed stroke = **not yet implemented** (all of Phase 2 and 3's new parts).

---

## 4. Per-phase content

**Phase 1 (implemented).** Observation = performance MIDI, per-note targets `ỹ` (N×3).
Likelihood: `ỹ = y + e` on *observed* notes; masked notes contribute no term. Head:
exact conjugate GP posterior. Output: per-note `τᵢ, log rᵢ, vᵢ ± √((Σ_y)ᵢᵢ + ς²)`.

**Phase 2 (not implemented).** Observation = monophonic audio → `f₀` extraction →
*derived targets* (`cᵢ` cents, vibrato). Likelihood: same form, but `Σ_e` is now
**substantive** — targets are estimated, not exact. Coregionalization grows to `k×k`,
kernel to `kN×kN`. Head is **unchanged** — say so on the figure.

**Phase 3 (not implemented).** Observation = waveform `x ∈ ℝᴹ`, *no* performance MIDI.
Audio likelihood: `x = Φ(z)a + ε`, amplitudes marginalised exactly,
`p(x|z) = 𝒩(Φμ_a, ΦΣ_aΦᵀ + K_n)`. The prior becomes the structured component of `p(z)`.
Head: **non-conjugate** — Laplace / VI over `z`, `a` recovered in closed form after.

---

## 5. Hard constraints

1. **Vocabulary.** The thesis has been de-jargoned. Never write, in a figure or its
   caption: `GP-first`, `single-GP`, `two-stage era`, `old headline`, `dev`, `LM`,
   `guard`, `ladder`, `marglik`, or any config codename (`b_featlm`, `a_diag`, …).
   Say **the proposed model**, **the two-stage plug-in baseline**, **development set**,
   **music model**, **safeguard**, **ablation sequence**.
2. **Invent no numbers.** I wrote `N×d` and `k×k` rather than guess the embedding width
   or the Phase-2 channel count. See §7.
3. **Colourblind-safe palette only.** No red/green load-bearing distinctions.
4. **Dashed ⇒ not implemented.** Never dash something that exists.
5. **The figures must not outrun the text.** Nothing may appear in a figure that the
   thesis does not define.

---

## 6. Mandatory workflow: render, look, fix

**These figures compile cleanly while being visually broken.** Every defect below passed
`pdflatex` with zero errors and zero overfull boxes, and was caught *only* by rendering
to PNG and looking at it:

- the `score features` box did not feed its own volume — the arrow jumped straight to
  the linear kernels, leaving `X_feat` floating unconnected;
- the `θ = (s, B, {c_f}, ς²)` annotation sat on top of the joint-kernel label;
- the "sum of Kronecker terms" note sat on the noise arrow;
- in Phase 2, the audio input box landed on top of the `diag(c_f)` volume;
- in Phase 3, the `p(z)` edge label collided with the prior enclosure;
- the joint-kernel equation broke mid-expression across two lines;
- `couples the components of z` hyphenated as `com-ponents`.

So the loop is not optional:

```bash
pdflatex -interaction=nonstopmode arch_phase1.tex
pdftoppm -png -r 105 -singlefile arch_phase1.pdf /tmp/preview
# then actually open/read /tmp/preview.png and look at it
```

Iterate until nothing overlaps, no word hyphenates inside a box, every volume has an
inbound arrow, and every annotation clears its neighbours. Expect 3–4 passes.

---

## 7. What only the code environment can finish

I could not do these from the Cowork sandbox. They are the highest-value handoff items.

1. **Real dimensions.** Replace `N×d` with the actual per-note embedding width from the
   pretrained checkpoint, and `k×k` / `kN×kN` with the actual Phase-2 channel count once
   the vibrato parameterisation is fixed. Read them from the checkpoint config and the
   channel registry rather than assuming.
2. **The SVGs are unverified.** `pipeline_phase{1,2,3}.svg` were hand-authored but
   **never rendered** — this sandbox has no SVG rasteriser and `pip` is firewalled.
   Install `cairosvg` or `rsvg-convert`, render them, look at them, fix what you find.
   This is a real hole in what has been handed over. Style B has **no SVGs at all** yet.
3. **`pytest` and the leak audits** have not been run by Cowork at any point in this
   work — `scipy`/`pytest` are not installable here. Every claim about them is inherited.

---

## 8. Files and build

```
docs/thesis/figures/
  pipeline_phase{1,2,3}.tex   # Style A, standalone TikZ
  pipeline_phase{1,2,3}.pdf   # compiled
  pipeline_phase{1,2,3}.svg   # hand-authored, UNVERIFIED
  arch_phase{1,2,3}.tex       # Style B, standalone TikZ
  arch_phase{1,2,3}.pdf       # compiled
```

Build: `cd docs/thesis/figures && pdflatex -interaction=nonstopmode <file>.tex`.
The thesis resolves them via `\graphicspath{{figures/}}`; `\includegraphics{arch_phase1}`
works. Note `.gitignore` carries `!docs/thesis/figures/*.pdf` — figure PDFs are tracked
deliberately, because `\includegraphics` needs them on a fresh clone.

Neither style is referenced from `draft.tex` yet. Suggested placement once chosen:
Phase 1 in Chapter 7 beside the joint-kernel equation; Phases 2 and 3 in Future Work.

---

## 9. If you want to compare agents rather than speculate

Give the next session only `hybrid_architecture_sample.svg` plus §§1, 3, 4, 5 of this
brief — *not* the existing TikZ — and have it produce Phase 1 from scratch. Then diff the
two figures. That isolates the design brief from the model, and tells you which is
actually doing the work. Withhold §6 if you want to test whether it discovers the
render–inspect loop on its own; include it if you just want the figure.
