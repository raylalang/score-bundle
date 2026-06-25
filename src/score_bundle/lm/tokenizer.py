"""MIDI-like, note-structured tokenizer (Phase 0).

Each note becomes a fixed 4-token group, in onset order:

    [ TIME_SHIFT(delta) , PITCH(p) , DURATION(d) , VELOCITY(v) ]

plus PAD / BOS / EOS sentinels.  The fixed per-note stride makes detokenization
trivial and lets us read a per-note embedding off the VELOCITY token (aligned with
the score graph's nodes).  Closest in spirit to REMI (Huang & Yang 2020) and the
MIDI-like encoding of Oore et al. (2018), simplified for a constant stride.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3
FIELDS = ("time_shift", "pitch", "duration", "velocity")


@dataclass
class NoteEvent:
    pitch: int       # MIDI number
    onset: float     # beats (score or performance time)
    duration: float  # beats
    velocity: int    # 0..127


class MidiTokenizer:
    def __init__(
        self,
        grid: int = 24,            # subdivisions per beat
        max_shift_steps: int = 96,  # max inter-onset gap (in grid steps)
        max_dur_steps: int = 96,
        pitch_min: int = 21,
        pitch_max: int = 108,
        n_vel_bins: int = 32,
    ):
        self.grid = grid
        self.max_shift_steps = max_shift_steps
        self.max_dur_steps = max_dur_steps
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.n_pitch = pitch_max - pitch_min + 1
        self.n_vel_bins = n_vel_bins

        # contiguous id ranges
        self.ts_base = N_SPECIAL
        self.pitch_base = self.ts_base + (max_shift_steps + 1)
        self.dur_base = self.pitch_base + self.n_pitch
        self.vel_base = self.dur_base + max_dur_steps
        self.vocab_size = self.vel_base + n_vel_bins

    # --- encode -----------------------------------------------------------
    def encode(self, notes: Sequence[NoteEvent], add_bos_eos: bool = True) -> List[int]:
        toks: List[int] = [BOS] if add_bos_eos else []
        running = 0  # quantized onset position in grid steps
        for n in sorted(notes, key=lambda e: (e.onset, e.pitch)):
            onset_steps = round(n.onset * self.grid)
            shift = int(min(max(onset_steps - running, 0), self.max_shift_steps))
            running += shift
            dur_steps = int(min(max(round(n.duration * self.grid), 1), self.max_dur_steps))
            pidx = int(min(max(n.pitch, self.pitch_min), self.pitch_max)) - self.pitch_min
            vbin = min(self.n_vel_bins - 1, int(n.velocity) * self.n_vel_bins // 128)
            toks += [
                self.ts_base + shift,
                self.pitch_base + pidx,
                self.dur_base + (dur_steps - 1),
                self.vel_base + vbin,
            ]
        if add_bos_eos:
            toks.append(EOS)
        return toks

    # --- decode -----------------------------------------------------------
    def token_type(self, tok: int) -> Tuple[str, int]:
        """Return (field_name, value) for a content token, or ('special', tok)."""
        if tok < N_SPECIAL:
            return ("special", tok)
        if tok < self.pitch_base:
            return ("time_shift", tok - self.ts_base)
        if tok < self.dur_base:
            return ("pitch", tok - self.pitch_base + self.pitch_min)
        if tok < self.vel_base:
            return ("duration", tok - self.dur_base + 1)
        return ("velocity", tok - self.vel_base)

    def decode(self, tokens: Sequence[int]) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        running = 0
        field_idx = 0
        buf = {}
        for tok in tokens:
            name, val = self.token_type(tok)
            if name == "special":
                continue
            expected = FIELDS[field_idx]
            if name != expected:
                # resync: drop partial group and restart on a time_shift
                field_idx = 0
                buf = {}
                if name != "time_shift":
                    continue
            buf[name] = val
            field_idx += 1
            if field_idx == 4:
                running += buf["time_shift"]
                vbin = buf["velocity"]
                vel = int((vbin + 0.5) * 128 / self.n_vel_bins)
                notes.append(
                    NoteEvent(
                        pitch=buf["pitch"],
                        onset=running / self.grid,
                        duration=buf["duration"] / self.grid,
                        velocity=vel,
                    )
                )
                field_idx = 0
                buf = {}
        return notes
