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
from .gp import MultiOutputGraphGP
from .graph import build_adjacency, chain_adjacency, laplacian
from .model import (GraphGaussianField, SpectralGaussianField, fit_laplacian_field,
                    fit_spectral_field, fit_spectral_field_guarded)
from .prior import (SPECTRAL_KERNELS, laplacian_precision, matern_precision,
                    spectral_covariance)
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
    "MultiOutputGraphGP",
    "GraphGaussianField",
    "SpectralGaussianField",
    "fit_laplacian_field",
    "fit_spectral_field",
    "fit_spectral_field_guarded",
    "SPECTRAL_KERNELS",
    "spectral_covariance",
    "graph",
    "prior",
    "metrics",
    "baselines",
    "features",
    "synthetic",
    "variables",
    "lm",
]
