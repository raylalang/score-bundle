# Slides

`deck.tex` — beamer 16:9, compiles with `tectonic deck.tex` (0 errors/overfull).

`deck.tex` is the one living thesis deck (consolidated 2026-10-14 from
`deck_week.tex` and `deck_zemi_v1.tex`, which are frozen copies): 55
pages (main line to page 46, backup divider 47, backup pages 48 to 55),
content-named chapters (piano / strings-and-winds / waveform),
no "Phase N" wording, datasets with pieces and notes on every setup
frame, every delta naming its comparator. The waveform chapter presents
methodology and first development studies, no claims. The week of
2026-10-15 to 23 is on the main line as two slides (2026-10-23): page
43, the score prior inside the waveform inference at three spans
(design maths plus the paired table); page 44, what in the graph does
the work (the edge weight with and without the pitch term, the
track-scale contrasts on both corpora). The backup frames carry the
detail: pages 50 to 55 (design, one chain, paired contrasts, the
withdrawn first run, calibration, the track-scale table; figures
`span_*.png` from `scripts/make_span_viz.py`).
Figures live in
`../thesis/figures/`; house style Okabe–Ito (blue = this model,
vermilion = reference, green = pYIN), dpi 200. Audio demo clips for the
opening frame: `expressive.wav`, `flat.wav`. `deck_kernels.tex` is the
September kernel-week deck, historical.

**Speaker's notes: `speaker_notes.md`.** Standing rule (Ray, 2026-10-21):
every change to `deck.tex` is followed by a change to `speaker_notes.md`
(same commit). The notes define every symbol on each slide and give an
example; frame numbers are PDF page numbers.
