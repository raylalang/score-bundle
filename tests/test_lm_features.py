import numpy as np

from score_bundle import Score, build_adjacency, laplacian, laplacian_precision, GraphGaussianField
from score_bundle.lm import features, model_numpy as mnp
from score_bundle.lm.tokenizer import MidiTokenizer
from score_bundle.lm.data import random_sequence


def test_note_embeddings_align_with_notes():
    rng = np.random.default_rng(0)
    tok = MidiTokenizer()
    notes = random_sequence(rng, n_notes=15)
    tokens = tok.encode(notes)
    cfg = mnp.GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=128)
    params = mnp.init_params(cfg, rng)

    assert len(features.note_velocity_positions(tok, tokens)) == len(notes)
    emb = features.note_embeddings(params, cfg, tok, tokens)
    assert emb.shape == (len(notes), cfg.d_model)


def test_prior_mean_feeds_graph_prior():
    rng = np.random.default_rng(1)
    tok = MidiTokenizer()
    notes = random_sequence(rng, n_notes=20)
    tokens = tok.encode(notes)
    cfg = mnp.GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=128)
    params = mnp.init_params(cfg, rng)

    emb = features.note_embeddings(params, cfg, tok, tokens)
    y = (np.array([n.velocity for n in notes], float) - 70.0) / 30.0
    mu, W = features.fit_prior_mean(emb, y, l2=1.0)
    assert mu.shape == (len(notes),)
    assert np.isfinite(mu).all()

    score = Score.from_arrays(
        [n.pitch for n in notes], [n.onset for n in notes], [n.duration for n in notes]
    )
    Q = laplacian_precision(laplacian(build_adjacency(score)), lam=0.5, eta=2.0)
    m, std = GraphGaussianField(Q, mean=mu).posterior(y, noise_var=0.1)
    assert m.shape == (len(notes),) and np.isfinite(m).all()
    assert (std > 0).all()
