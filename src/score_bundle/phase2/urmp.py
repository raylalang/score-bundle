"""URMP dataset loader (Phase 2 corpus).

URMP (Li et al., IEEE TMM): 44 simple multi-instrument classical pieces,
separately recorded tracks.  Per piece folder ``N_piece_inst1_inst2[...]``:

    AuMix_N_piece_insts.wav        mixed audio (48 kHz, 24-bit, mono)
    AuSep_n_inst_N_piece.wav       per-track audio
    F0s_n_inst_N_piece.txt         frame-level ground-truth pitch
                                   (46 ms windows, 10 ms hop; col 0 = frame
                                   centre time s, col 1 = f0 Hz, 0 = silent)
    Notes_n_inst_N_piece.txt       note-level truth (onset s, pitch Hz,
                                   duration s)
    Sco_N_piece_insts.mid          MIDI score

Only the annotation/score inventory is parsed here (numpy-only); audio
loading is left to the caller (``phase2.intonation.extract_f0`` or the
provided F0s files — the latter double as a calibration reference for the
tracker's confidence, since both live on the same 10 ms hop grid).

Datasets live outside the repo; pass the root explicitly
(e.g. ``../data/urmp/Dataset``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

_FOLDER_RE = re.compile(r"^(\d{2})_([A-Za-z0-9]+)((?:_[a-z]+)+)$")

INSTRUMENTS = {"vn": "violin", "va": "viola", "vc": "cello", "db": "double bass",
               "fl": "flute", "ob": "oboe", "cl": "clarinet", "sax": "saxophone",
               "bn": "bassoon", "tpt": "trumpet", "hn": "horn",
               "tbn": "trombone", "tba": "tuba"}


@dataclass
class UrmpTrack:
    number: int                 # 1-based track index (score order)
    instrument: str             # abbreviation, e.g. "vn"
    audio: str                  # AuSep path
    f0s: str                    # F0s annotation path
    notes: str                  # Notes annotation path


@dataclass
class UrmpPiece:
    index: int                  # piece number N (1..44)
    name: str                   # distinct abbreviation, e.g. "Jupiter"
    folder: str                 # absolute folder path
    instruments: List[str] = field(default_factory=list)
    score_mid: str = ""
    mix_audio: str = ""
    tracks: List[UrmpTrack] = field(default_factory=list)


def load_urmp_meta(root: str) -> List[UrmpPiece]:
    """Scan a URMP root for piece folders and build the file inventory.

    Missing per-track files are tolerated (paths set to "") so a partial
    download still loads; callers should check the fields they need.
    """
    pieces: List[UrmpPiece] = []
    for entry in sorted(os.listdir(root)):
        m = _FOLDER_RE.match(entry)
        if not m:
            continue
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        idx, name = int(m.group(1)), m.group(2)
        insts = m.group(3).strip("_").split("_")
        piece = UrmpPiece(index=idx, name=name, folder=folder,
                          instruments=insts)
        files = set(os.listdir(folder))
        tag = f"{m.group(1)}_{name}"
        suffix = f"{tag}{m.group(3)}"
        if f"Sco_{suffix}.mid" in files:
            piece.score_mid = os.path.join(folder, f"Sco_{suffix}.mid")
        if f"AuMix_{suffix}.wav" in files:
            piece.mix_audio = os.path.join(folder, f"AuMix_{suffix}.wav")
        for n, inst in enumerate(insts, start=1):
            def p(prefix: str, ext: str) -> str:
                fname = f"{prefix}_{n}_{inst}_{tag}.{ext}"
                return os.path.join(folder, fname) if fname in files else ""
            piece.tracks.append(UrmpTrack(
                number=n, instrument=inst,
                audio=p("AuSep", "wav"), f0s=p("F0s", "txt"),
                notes=p("Notes", "txt")))
        pieces.append(piece)
    return pieces


def read_f0_annotation(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-level ground-truth pitch: (times s, f0 Hz); f0 == 0 on silence."""
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_2d(arr)
    return arr[:, 0], arr[:, 1]


def read_notes_annotation(path: str) -> Dict[str, np.ndarray]:
    """Note-level ground truth: onset (s), pitch (Hz), duration (s)."""
    arr = np.atleast_2d(np.loadtxt(path, dtype=float))
    return {"onset": arr[:, 0], "pitch_hz": arr[:, 1], "duration": arr[:, 2]}
