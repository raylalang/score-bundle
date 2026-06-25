"""Phase 2 (extension): intonation and vibrato for continuously pitched instruments.

These variables exist only on voice / strings / winds, NOT on piano.  Labels are
derived from f0 extraction, so they are noisier than Phase-1 piano targets.  The
score-graph prior and inference machinery from Phase 1 are reused; only new channels
(``c``, ``vibrato``, ``f0``) are added.
"""
from . import intonation

__all__ = ["intonation"]
