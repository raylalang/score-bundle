"""Performance-variable definitions and the phased channel registry.

Phase 1 (piano):   y_i = [tau_i, log r_i, v_i]
Phase 2 (mono):    the evaluated URMP bundle [c, log gamma, log f, ell, tau,
                   delta_vib] (draft eq:phase2-channels lists k=7 with log r;
                   articulation has no measured offset chain on URMP)
Phase 3 (waveform):+ harmonic amplitudes a_i (timbre)

Each channel is a scalar (or vector) field over the score nodes with its own graph
length scales.  Phase 1 channels are modeled independently with the shared graph;
the joint prior over the stacked field y in R^{kN} is block-diagonal across channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    phase: int
    realistic_for: str
    description: str


# Registry mirrors the "Performance variables" table of the concept note.
CHANNELS: Dict[str, Channel] = {
    "tau": Channel("tau", "s", 1, "all", "onset residual vs tempo-implied onset"),
    "log_r": Channel("log_r", "dimensionless", 1, "all (esp. piano)", "articulation/duration ratio (log)"),
    "v": Channel("v", "norm. MIDI velocity", 1, "keyboard reliable", "per-note dynamics"),
    "pedal": Channel("pedal", "on/off|cont.", 1, "piano only", "sustain pedal (optional)"),
    "c": Channel("c", "cents", 2, "voice/strings/winds — not piano", "intonation deviation (vibrato-free centre)"),
    "log_gamma": Channel("log_gamma", "log cents", 2, "voice/strings/winds", "vibrato extent (log)"),
    "log_f": Channel("log_f", "log Hz", 2, "voice/strings/winds", "vibrato rate (log)"),
    "ell": Channel("ell", "log RMS, per-track centred", 2, "monophonic", "loudness"),
    "delta_vib": Channel("delta_vib", "s", 2, "voice/strings/winds", "vibrato onset delay (gated fit)"),
    "a": Channel("a", "—", 3, "narrow/monophonic only", "harmonic amplitudes / timbre"),
}

PHASE1_CHANNELS: List[str] = ["tau", "log_r", "v"]
# The evaluated URMP bundle (registered 2026-08-17); tau is the Phase-1
# timing channel re-anchored by the annotated-onset warp (phase2/warp.py).
PHASE2_CHANNELS: List[str] = ["c", "log_gamma", "log_f", "ell", "tau",
                              "delta_vib"]
PHASE3_CHANNELS: List[str] = ["a"]


def channels_for_phase(phase: int) -> List[str]:
    """Return channel names available up to and including ``phase``."""
    return [name for name, ch in CHANNELS.items() if ch.phase <= phase]
