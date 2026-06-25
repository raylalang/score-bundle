import numpy as np

from score_bundle.synthetic import make_synthetic, random_mask, random_score


def test_make_synthetic_shapes():
    rng = np.random.default_rng(0)
    data = make_synthetic(rng, n=40)
    assert len(data.score) == 40
    assert data.y_true.shape == (40,)
    assert data.y_obs.shape == (40,)
    assert data.L.shape == (40, 40)


def test_random_mask_keeps_both_classes():
    rng = np.random.default_rng(1)
    for frac in (0.0, 0.5, 1.0):
        mask = random_mask(30, rng, observed_frac=frac)
        assert mask.any() and (~mask).any()


def test_random_score_monotone_onsets():
    rng = np.random.default_rng(2)
    score = random_score(25, rng)
    assert np.all(np.diff(score.onset) >= 0)
    assert (score.duration > 0).all()
