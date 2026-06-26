"""Aria frozen-feature baseline (Phase-1 upper bound) — import-guarded stub.

Design decision (CLAUDE.md, ``docs/music_lm_design.md`` §6.1): the **aria model** is a
frozen-feature **upper-bound baseline**, never our backbone.  This module mirrors the
:func:`score_bundle.lm.features.note_embeddings_long` interface so aria embeddings drop
straight into the same ``h_i -> μ_LM`` head + graph-residual comparison as our own LM — the
only change at the call site is the embedding extractor.

It is a stub because aria is **not installed in this environment** (no ``aria`` package, no
checkpoint).  Nothing here imports aria at module load, so the Phase-1 core still imports
without it; the import happens only inside :func:`aria_note_embeddings`, which raises a clear
hint if aria is missing.  When wiring aria for real, also **guard contamination**: aria is
trained on large transcribed corpora that overlap MAESTRO/ASAP, so its "upper bound" may be
optimistic on any eval piece it effectively memorised — note this in the writeup.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_INSTALL_HINT = (
    "aria is not available in this environment. The aria frozen-feature baseline is optional "
    "(an upper-bound comparison, not the backbone). To enable it, install the aria model + a "
    "checkpoint, then implement the marked extraction below. See docs/music_lm_design.md §6.1 "
    "and remember the contamination caveat (aria's pretraining overlaps MAESTRO/ASAP)."
)


def aria_available() -> bool:
    """True iff the aria model package can be imported (it cannot, here)."""
    try:
        import aria  # noqa: F401  (presence check only)
    except Exception:
        return False
    return True


def aria_note_embeddings(notes, checkpoint: Optional[str] = None) -> np.ndarray:
    """Per-note frozen aria embeddings, shape ``(len(notes), d)`` — same contract as
    :func:`score_bundle.lm.features.note_embeddings_long`.

    Raises :class:`NotImplementedError` with an install hint when aria is unavailable, so a
    caller can ``try``/``except`` it and fall back to skipping the baseline.
    """
    if not aria_available():
        raise NotImplementedError(_INSTALL_HINT)
    # --- when aria IS installed, implement here -----------------------------
    # 1. load the frozen checkpoint (``checkpoint`` or a default),
    # 2. tokenize ``notes`` in aria's scheme,
    # 3. run a forward pass and read the per-note hidden states (aligned to score nodes),
    # 4. return them as (len(notes), d).  Then this plugs into fit_prior_mean_head exactly
    #    like our own note_embeddings_long.
    raise NotImplementedError(_INSTALL_HINT)  # pragma: no cover - aria not installed here
