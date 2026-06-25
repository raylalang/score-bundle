"""Tests for the from-scratch PyTorch music LM.

These require torch; when it is absent each test returns early (a no-op) so the
NumPy-only suite still runs. On a machine with torch installed they assert real behaviour.
"""
import numpy as np

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from score_bundle.lm.tokenizer import MidiTokenizer
from score_bundle.lm.data import random_sequence
from score_bundle.lm import features


def _model(tok, seed=0):
    from score_bundle.lm.model_torch import GPTConfig, build_model

    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=128, dropout=0.0)
    return build_model(cfg).eval(), cfg


def test_forward_shapes():
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    arr = np.asarray(tok.encode(random_sequence(np.random.default_rng(0), 12)))
    idx = torch.as_tensor(arr, dtype=torch.long)[None]
    logits = model(idx)
    assert logits.shape == (1, idx.shape[1], tok.vocab_size)


def test_attention_is_causal():
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    arr = np.asarray(tok.encode(random_sequence(np.random.default_rng(0), 12)))
    h1 = model.embed(torch.as_tensor(arr, dtype=torch.long)[None])[0]
    arr2 = arr.copy()
    arr2[-1] = (arr2[-1] + 5) % tok.vocab_size  # change only the last token
    h2 = model.embed(torch.as_tensor(arr2, dtype=torch.long)[None])[0]
    assert torch.allclose(h1[:-1], h2[:-1], atol=1e-5)
    assert not torch.allclose(h1[-1], h2[-1])


def test_generate_valid_tokens():
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    prompt = torch.as_tensor(tok.encode(random_sequence(np.random.default_rng(0), 5)), dtype=torch.long)[None]
    out = model.generate(prompt, max_new_tokens=8, top_k=8)[0]
    assert out.numel() == prompt.shape[1] + 8
    assert int(out.min()) >= 0 and int(out.max()) < tok.vocab_size


def test_embeddings_feed_graph_prior():
    if not HAS_TORCH:
        return
    from score_bundle import Score, build_adjacency, laplacian, laplacian_precision, GraphGaussianField

    tok = MidiTokenizer()
    model, cfg = _model(tok)
    notes = random_sequence(np.random.default_rng(1), n_notes=20)
    tokens = tok.encode(notes)

    assert len(features.note_velocity_positions(tok, tokens)) == len(notes)
    emb = features.note_embeddings(model, tok, tokens)
    assert emb.shape == (len(notes), cfg.d_model)

    y = (np.array([n.velocity for n in notes], float) - 70.0) / 30.0
    mu, _ = features.fit_prior_mean(emb, y, l2=1.0)
    assert mu.shape == (len(notes),) and np.isfinite(mu).all()

    score = Score.from_arrays(
        [n.pitch for n in notes], [n.onset for n in notes], [n.duration for n in notes]
    )
    Q = laplacian_precision(laplacian(build_adjacency(score)), lam=0.5, eta=2.0)
    m, std = GraphGaussianField(Q, mean=mu).posterior(y, noise_var=0.1)
    assert m.shape == (len(notes),) and np.isfinite(m).all() and (std > 0).all()
