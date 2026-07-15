# Slides

`deck.tex` — beamer 16:9, compiles with `tectonic deck.tex` (0 errors/overfull).

**Content-first build**: all text, equations, and tables are final and match the
verified-numbers table in `docs/presentation_brief.md` (DEV/CONF/REPL/SYNTH
badges on every number-bearing frame). Graphics are `\figslot{file}{height}{note}`
placeholders — dashed boxes naming the intended file and what it should show.

Workflow for the illustration pass:
1. Flip `\def\showfigures{0}` → `1`: every slot whose file already exists in
   `../thesis/figures/` (most of them) renders the real graphic; only genuinely
   missing art (e.g. `deck_concept.png` on slide 2) stays a box.
2. Draw/adjust the remaining art, drop files into `docs/thesis/figures/`,
   recompile. Heights are set per slot and easy to tune in place.

16 main frames + 5 backup frames (per-channel, kernel table, Gaussian tail +
Student-t, embedding probes, Phase-2 pilot).
