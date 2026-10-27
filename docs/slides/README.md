# Slides

`deck.tex` — beamer 16:9, compiles with `tectonic deck.tex` (0 errors/overfull).

`deck.tex` is the one living thesis deck (consolidated 2026-10-14 from
`deck_week.tex` and `deck_zemi_v1.tex`, which are frozen copies): 37
pages since the 2026-10-28 cut (main line to page 33, backup divider
34, backup pages 35 to 37; it was 56 pages, commit 9fe8778, before Ray
asked for findings only),
content-named chapters (piano / strings-and-winds / waveform),
no "Phase N" wording, datasets with pieces and notes on every setup
frame, every delta naming its comparator. The waveform chapter presents
methodology and first development studies, no claims. The week of
2026-10-15 to 23 is on the main line as two slides: page 30, the
score prior inside the waveform inference at three spans (design maths
plus the paired table); page 31, what in the graph does the work (the
edge weight with and without the pitch term, the track-scale contrasts
on both corpora). The two registered one-shots on the re-used
strings-and-winds pool (2026-10-27: correlated timing noise passed, the
tonal metric missed its calibration primary) are page 20, right after
the strings-and-winds results, with the Plan on page 33 stating the
outcomes. Backup: page 35 the three-note curve figure, 36 the paired
three-span contrasts, 37 the track-scale table (figures `span_*.png`
from `scripts/make_span_viz.py`).
Figures live in
`../thesis/figures/`; house style Okabe–Ito (blue = this model,
vermilion = reference, green = pYIN), dpi 200. Audio demo clips for the
opening frame: `expressive.wav`, `flat.wav`. `deck_kernels.tex` is the
September kernel-week deck, historical.

**Speaker's notes: `speaker_notes.md`.** Standing rule (Ray, 2026-10-21):
every change to `deck.tex` is followed by a change to `speaker_notes.md`
(same commit). The notes define every symbol on each slide and give an
example; frame numbers are PDF page numbers.
