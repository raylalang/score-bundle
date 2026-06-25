"""Corpus generation and next-token batching for Phase-0 pretraining.

Framework-agnostic (NumPy): the same token streams feed either the NumPy or the
PyTorch model.  ``random_corpus`` provides a tiny synthetic corpus so the examples
and tests run without any external MIDI data; replace it with real loaders
(MAESTRO/ATEPP/Lakh) for actual pretraining.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from .tokenizer import MidiTokenizer, NoteEvent


def random_sequence(rng: np.random.Generator, n_notes: int = 40) -> List[NoteEvent]:
    """A plausible-ish monophonic piano line: pitch random walk, increasing onsets."""
    pitch = 60 + np.cumsum(rng.integers(-3, 4, size=n_notes))
    pitch = np.clip(pitch, 36, 90)
    gaps = rng.choice([0.25, 0.5, 0.5, 1.0], size=n_notes)
    onset = np.concatenate([[0.0], np.cumsum(gaps)[:-1]])
    dur = gaps * rng.uniform(0.6, 1.0, size=n_notes)
    vel = rng.integers(40, 100, size=n_notes)
    return [NoteEvent(int(pitch[i]), float(onset[i]), float(dur[i]), int(vel[i])) for i in range(n_notes)]


def random_corpus(rng: np.random.Generator, n_seqs: int = 64, n_notes: int = 40) -> List[List[NoteEvent]]:
    return [random_sequence(rng, n_notes) for _ in range(n_seqs)]


def encode_corpus(tokenizer: MidiTokenizer, corpus: List[List[NoteEvent]]) -> List[List[int]]:
    return [tokenizer.encode(seq) for seq in corpus]


def pack_tokens(token_seqs: List[List[int]]) -> np.ndarray:
    """Concatenate sequences into a single 1-D stream for windowed LM training."""
    return np.concatenate([np.asarray(s, dtype=np.int64) for s in token_seqs])


def lm_batches(
    stream: np.ndarray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
    n_batches: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (x, y) int arrays of shape (batch, block); y is x shifted by one."""
    hi = len(stream) - block_size - 1
    if hi <= 0:
        raise ValueError("stream shorter than block_size + 1")
    for _ in range(n_batches):
        ix = rng.integers(0, hi, size=batch_size)
        x = np.stack([stream[i : i + block_size] for i in ix])
        y = np.stack([stream[i + 1 : i + 1 + block_size] for i in ix])
        yield x, y
