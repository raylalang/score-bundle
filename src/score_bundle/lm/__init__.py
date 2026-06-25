"""Phase 0 — from-scratch symbolic music language model (PyTorch).

A decoder-only Transformer over note-structured MIDI tokens, built from the ground up
(hand-written causal attention). It is a generative backbone and a representation learner
whose per-note embeddings feed the Phase-1 graph prior (see ``docs/music_lm_design.md``).

- ``tokenizer``    : note-structured MIDI tokenizer (NoteEvent <-> tokens) — NumPy
- ``data``         : synthetic corpus + next-token batching — NumPy
- ``model_torch``  : the from-scratch PyTorch model (forward / embed / generate + train)
- ``features``     : per-note embeddings -> learned prior mean (Phase-1 bridge)

``tokenizer``/``data`` are framework-agnostic; the model is PyTorch (optional dependency,
import-guarded so this package still imports without torch).
"""
from __future__ import annotations

from . import data, features, model_torch, tokenizer
from .tokenizer import MidiTokenizer, NoteEvent
from .model_torch import GPTConfig, build_model, train_lm

__all__ = [
    "tokenizer",
    "data",
    "model_torch",
    "features",
    "MidiTokenizer",
    "NoteEvent",
    "GPTConfig",
    "build_model",
    "train_lm",
]
