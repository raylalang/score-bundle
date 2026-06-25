"""Phase 0 — from-scratch symbolic music language model.

A decoder-only Transformer over note-structured MIDI tokens, built from the ground up.
It serves as a generative backbone and a representation learner whose per-note
embeddings feed the Phase-1 graph prior (see ``docs/music_lm_design.md``).

- ``tokenizer``    : note-structured MIDI tokenizer (NoteEvent <-> tokens)
- ``data``         : synthetic corpus + next-token batching
- ``model_numpy``  : from-scratch forward pass + sampling (no deps, testable)
- ``model_torch``  : trainable PyTorch twin (optional dependency)
- ``features``     : per-note embeddings -> learned prior mean (Phase-1 bridge)
"""
from __future__ import annotations

from . import data, features, model_numpy, tokenizer
from .tokenizer import MidiTokenizer, NoteEvent
from .model_numpy import GPTConfig, init_params, forward, generate

__all__ = [
    "tokenizer",
    "data",
    "model_numpy",
    "features",
    "MidiTokenizer",
    "NoteEvent",
    "GPTConfig",
    "init_params",
    "forward",
    "generate",
]
