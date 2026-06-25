import numpy as np

from score_bundle.phase3 import synth, waveform_model


def test_amplitude_posterior_recovers_known_amps():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1.0, 2000)
    f0 = np.full_like(t, 5.0)            # 5 Hz fundamental (toy)
    n_harm = 3
    Phi = synth.harmonic_design_matrix(f0, t, n_harm)
    a_true = rng.standard_normal(2 * n_harm)
    x = Phi @ a_true + 0.01 * rng.standard_normal(t.size)

    Sigma_a = np.eye(2 * n_harm) * 10.0
    m_a, S_a = waveform_model.amplitude_posterior(x, Phi, Sigma_a, noise_var=1e-3)
    assert np.sqrt(np.mean((m_a - a_true) ** 2)) < 0.05
    assert np.all(np.diag(S_a) > 0)


def test_collapsed_loglik_finite():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 0.5, 800)
    f0 = np.full_like(t, 8.0)
    Phi = synth.harmonic_design_matrix(f0, t, 2)
    x = rng.standard_normal(t.size)
    ll = waveform_model.collapsed_loglik(x, Phi, np.eye(4), noise_var=1.0)
    assert np.isfinite(ll)


def test_position_inference_is_stub():
    try:
        waveform_model.infer_positions()
        assert False
    except NotImplementedError:
        pass
