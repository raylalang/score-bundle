"""Phase 3 (extension): differentiable-synthesizer audio likelihood.

The waveform is x = Phi(z) a + eps, where z holds the nonlinear position variables
(tempo, timing, intonation, vibrato) and a the linear-Gaussian harmonic amplitudes.
Given z, the amplitudes are marginalized in closed form.  Inference over the
nonlinear z is now implemented for the within-note cents curve in :mod:`.curve`
(gradient MAP + full joint Laplace, PyTorch, import-guarded); joint inference
over note boundaries remains the open step (`waveform_model.infer_positions`).
"""
from . import curve, synth, waveform_model

__all__ = ["curve", "synth", "waveform_model"]
