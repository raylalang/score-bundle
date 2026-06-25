import numpy as np

from score_bundle.lm import model_numpy as mnp
from score_bundle.lm.tokenizer import MidiTokenizer
from score_bundle.lm.data import random_sequence


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    tok = MidiTokenizer()
    cfg = mnp.GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=128)
    params = mnp.init_params(cfg, rng)
    tokens = np.asarray(tok.encode(random_sequence(rng, n_notes=12)))
    return rng, tok, cfg, params, tokens


def test_forward_shapes():
    _, tok, cfg, params, tokens = _setup()
    logits, hidden = mnp.forward(params, tokens, cfg)
    T = len(tokens)
    assert logits.shape == (T, tok.vocab_size)
    assert hidden.shape == (T, cfg.d_model)
    assert np.isfinite(logits).all()


def test_attention_is_causal():
    _, tok, cfg, params, tokens = _setup()
    _, h1 = mnp.forward(params, tokens, cfg)
    tokens2 = tokens.copy()
    # change the LAST token; earlier hidden states must be unchanged (causality)
    tokens2[-1] = (tokens2[-1] + 5) % tok.vocab_size
    _, h2 = mnp.forward(params, tokens2, cfg)
    assert np.allclose(h1[:-1], h2[:-1], atol=1e-10)
    assert not np.allclose(h1[-1], h2[-1])


def test_generate_valid_tokens():
    rng, tok, cfg, params, tokens = _setup()
    out = mnp.generate(params, cfg, tokens[:10], max_new_tokens=8, rng=rng, top_k=8)
    assert len(out) == 18
    assert ((out >= 0) & (out < tok.vocab_size)).all()
