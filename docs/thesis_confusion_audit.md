# Fresh-eyes confusion audit of draft.tex (2026-09-03)

Basis for the planned structural simplification (see the session plan).
Audited at main = 0ed215b (62 pp). Rule for the restructure: **no number,
claim, or honesty disclosure is deleted — only relocated or compressed.**

## The single most important structural fact

Chapter 3 ("Model and Formulation") contains two complete empirical
results chapters — Phase-2 development + confirmation (§3.9, pp. 17–26)
and the Phase-3 study (§3.10, pp. 26–30), ~14 of 62 pages — while the
chapter titled "Results" is Phase-1 only and the chapter titled "Data"
never mentions URMP.

## Top 10 confusion sources (fix class: a = prose edit, b = appendix, c = restructure)

1. **Results hidden inside the Model chapter; Phase 2/3 each told three
   times** (abstract/intro pre-telling, §3.9–3.10 full telling, Future
   Work §8.2–8.3 retelling — §8.2 even opens "no longer future work").
   Fix (c): Phase 2 becomes its own chapter (model-delta → data → dev →
   confirmation); Phase 3 → short Future-Work section + appendix study.
2. **Two models run in parallel throughout** ("two-stage" ×43; mixed
   rows in Tables 5.2/5.4/6.1; §5.10 justifies the current kernel under
   the retired system). Fix (c)+(b): keep the one-sentence nesting
   statement and the confirmation comparison row in the main line;
   lineage tables/constraint enumeration → "Development lineage" appendix.
3. **Eight inline development studies** interrupt the argument (drift
   study 2.5 pp mid-§3.9; tonal exploratory; synthetic pilot; mask sweep;
   §5.7–5.10 incl. 3-pp kernel study; linear probe). Fix (b): each keeps
   a 1–2-sentence verdict + pointer in the main line.
4. **Current vs ablation vs exploratory illegible** — e.g. the kernel
   adoption paragraph buries its decision in line 29 of one paragraph;
   the tonal metric verdict is retold four times with alternating signs.
   Fix (a)+(b): decision-first prose.
5. **Posterior decomposition derived ~1,300 lines before its only use**
   (§3.6 → used in §5.3). Fix (c): move the derivation to §5.3's head.
6. **Notation overload** — App A needs a 17-line "deliberate
   disambiguations" paragraph for live collisions (s, P, B, δ, ℓ, c, Φ,
   K). Fix (a): rename (δ^vib→d_i, tracker conf P_j→π_j, harmonic count
   K→H, "harmonic graph"→"chord graph"); trim Table 3.1 to Phase-1/2 rows.
7. **Long-range cross-references both directions** (sec:phase2 ×14,
   sec:attribution ×10, ...; two double-labelled sections). Mostly
   dissolved by fixes 1 and 5; collapse double labels.
8. **§3.9 ordered as lab notebook**: URMP (the corpus) is introduced
   AFTER four passages reporting numbers computed on it. Fix (c):
   model-delta / data / development / confirmation order.
9. **Four claims each repeated 7–8×** (attribution slogan ×8; one-shot
   discipline ×8; nesting ×7; Phase-2 status ×7). Fix (a): one canonical
   statement (Results), one echo (Conclusion), one clause (Abstract).
   ~3–4 pages recoverable.
10. **Abstract and contributions are the first and worst obstacles**:
    one 41-line abstract paragraph, ~15 numbers, a seed-sensitivity
    caveat meaningless before §3.9; contribution 5 is typeset as a
    placeholder "(Planned/future)"; the title concept "Score-Bundle" is
    unexplained until p. 52 while "bundle" is used in a different sense
    from p. 17. Fix (a)+(c): 3-paragraph abstract; demote contribution 5;
    one intro paragraph for the title concept.

## Minimal main line (~30 pp target; current 62)

- Ch 1 Intro (~3 pp): contributions 1–4; two-stage defined in two
  sentences; title concept explained here.
- Ch 2 Background (~2 pp): move spectral-kernel paragraph to Future Work.
- Ch 3 Model (~6 pp): out — decomposition derivation (→ Ch 5),
  constraint-switch detail (→ appendix; keep "Exact nesting").
- Ch 4 Data (~4 pp): + URMP (moved from §3.9); out — mask sweep.
- Ch 5 Phase-1 result (~5 pp): headline, attribution (+ relocated
  derivation), per-channel, two measured failure modes. Out: §5.2
  selection history, §5.5 leak episode, §5.7–5.10.
- Ch 6 Downstream (~2 pp): single-verdict table (two-stage column →
  appendix).
- Ch 7 Phase-2 result (~5 pp, NEW chapter): model delta, data pointer,
  dev results, confirmation, gate. Out: musical glossary (→ footnote/
  appendix), drift study, tonal test, synthetic pilot (→ appendices).
- Ch 8–10 Discussion/Future/Conclusion (~4 pp): kill §8.2/8.3 retellings.
- Appendices: A notation; B development lineage (two-stage); C
  methodology record (leak episode, safeguard, audits, centring
  disclosure); D kernel & feature studies; E Phase-2 development studies
  (drift, pilot, tonal, noise variants); F Phase-3 (math + 376-note
  study).

## Page accounting

Main line ~28–31 pp; ~23 pp relocated to appendices (Phase-3 4.5, kernel
study 3, drift 2.5, music-theory 2, constraint detail 1.5, decomposition
1.5, leak 1.5, §5.7–5.9 1, mask sweep 1, pilot+tonal 1.5, existing
App B 2); ~3–4 pp recovered from repetition. Everything else unchanged.
