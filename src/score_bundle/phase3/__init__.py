"""Phase 3 (extension): differentiable-synthesizer audio likelihood.

The waveform is x = Phi(z) a + eps, where z holds the nonlinear position variables
(tempo, timing, intonation, vibrato) and a the linear-Gaussian harmonic amplitudes.
Given z, the amplitudes are marginalized in closed form; inference over the nonlinear
z (Laplace / VI with score-based initialization) is the remaining work and is left as
a documented stub.
"""
from . import synth, waveform_model

__all__ = ["synth", "waveform_model"]
