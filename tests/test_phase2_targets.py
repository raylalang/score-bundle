"""note_targets: frame assignment, ET reference, and estimator hand-off."""
import numpy as np

from score_bundle.phase2.targets import hz_to_semitone, note_targets


def _curve(sr=100.0, dur=3.2):
    t = np.arange(0.0, dur, 1.0 / sr)
    f0 = np.full(t.size, np.nan)
    voiced = np.zeros(t.size, dtype=bool)
    return t, f0, voiced


def test_recovers_known_vibrato_per_note():
    t, f0, voiced = _curve()
    onset = np.array([0.1, 1.6, 3.0])
    dur = np.array([1.4, 1.3, 0.05])          # third note too short
    midi = np.array([69, 72, 60])             # A4, C5, C4
    truth = [(12.0, 30.0, 5.0), (-8.0, 22.0, 6.5), (0.0, 0.0, 0.0)]
    for (c, g, f), o, d, m in zip(truth, onset, dur, midi):
        sel = (t >= o) & (t < o + d)
        cents = c + g * np.sin(2 * np.pi * f * (t[sel] - o))
        f_note = 440.0 * 2.0 ** ((m - 69) / 12.0)
        f0[sel] = f_note * 2.0 ** (cents / 1200.0)
        voiced[sel] = True
    out = note_targets(t, f0, voiced, None, onset, dur,
                       440.0 * 2.0 ** ((midi - 69) / 12.0))
    assert list(out["midi"]) == [69, 72, 60]
    assert out["ident"][0] and out["ident"][1] and not out["ident"][2]
    assert abs(out["est"][0, 0] - 12.0) < 1.5
    assert abs(np.exp(out["est"][0, 1]) - 30.0) < 3.0
    assert abs(np.exp(out["est"][1, 2]) - 6.5) < 0.2
    # short note: c still estimable, vibrato channels missing
    assert abs(out["est"][2, 0]) < 2.0 and np.isnan(out["est"][2, 1])
    assert np.all(np.isfinite(out["var"][:2, :]))


def test_confidence_filter_drops_low_quantile():
    t, f0, voiced = _curve(dur=1.0)
    onset, dur, midi = np.array([0.0]), np.array([1.0]), np.array([69])
    f0[:] = 440.0
    voiced[:] = True
    prob = np.linspace(0.0, 1.0, t.size)
    # poison the lowest-confidence fifth with octave errors
    low = prob < 0.2
    f0[low] = 220.0
    out = note_targets(t, f0, voiced, prob, onset, dur, np.array([440.0]),
                       conf_quantile=0.2)
    assert abs(out["est"][0, 0]) < 2.0         # octave frames filtered out


def test_hz_to_semitone_reference():
    assert hz_to_semitone(np.array([440.0]))[0] == 69
    assert hz_to_semitone(np.array([261.63]))[0] == 60
