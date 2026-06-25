"""score-bundle: Bayesian score-informed performance transcription.

A score graph supplies a structured Gaussian prior over per-note expressive
performance variables; the posterior returns per-note estimates with uncertainty.
Phase 1 (this package's working core) uses targets read off aligned data; Phases 2
and 3 (intonation/vibrato and a differentiable-synthesizer likelihood) are provided
as interfaces/stubs.

See ``docs/architecture.svg`` for the high-level picture.
"""
from __future__ import annotations

from . import baselines, features, graph, lm, metrics, prior, synthetic, variables
from .graph import build_adjacency, chain_adjacency, laplacian
from .model import GraphGaussianField, fit_laplacian_field
from .prior import laplacian_precision, matern_precision
from .score import Note, Score

__version__ = "0.1.0"

__all__ = [
    "Note",
    "Score",
    "build_adjacency",
    "chain_adjacency",
    "laplacian",
    "laplacian_precision",
    "matern_precision",
    "GraphGaussianField",
    "fit_laplacian_field",
    "graph",
    "prior",
    "metrics",
    "baselines",
    "features",
    "synthetic",
    "variables",
    "lm",
]
