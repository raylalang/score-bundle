"""Smoke tests for the Phase-0 pretraining loop (torch).

Tiny CPU run on the synthetic corpus: the schedule, eval, checkpointing, sampling, and a
real loss decrease all exercised without needing MAESTRO.  Skips when torch is absent so
the numpy-only suite still runs.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from score_bundle.lm.tokenizer import MidiTokenizer
from score_bundle.lm import data
from score_bundle.lm import train as lmtrain
from score_bundle.lm.model_torch import GPTConfig, build_model


def _tiny_streams(tok, rng):
    corpus = data.random_corpus(rng, n_seqs=40, n_notes=40)
    stream = data.pack_tokens(data.encode_corpus(tok, corpus))
    return stream, corpus


def test_cosine_lr_warmup_and_decay():
    cfg = lmtrain.TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10)
    total = 100
    assert lmtrain.cosine_lr(0, cfg, total) < cfg.lr          # warming up
    assert lmtrain.cosine_lr(9, cfg, total) == pytest.approx(cfg.lr, rel=1e-6)  # peak
    assert lmtrain.cosine_lr(total, cfg, total) == pytest.approx(cfg.min_lr, rel=1e-6)  # decayed


def test_train_loop_decreases_loss(tmp_path):
    rng = np.random.default_rng(0)
    tok = MidiTokenizer()
    stream, corpus = _tiny_streams(tok, rng)

    torch.manual_seed(0)
    gpt = GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_layer=2, n_head=4, block_size=128, dropout=0.0)
    model = build_model(gpt)

    cfg = lmtrain.TrainConfig(
        block_size=128, batch_size=16, epochs=2, steps_per_epoch=40,
        warmup_steps=10, eval_batches=10, out_dir=str(tmp_path), cache_dir=None, seed=0,
    )
    hist = lmtrain.train(model, stream, stream, cfg, tok=tok, sample_prompt=corpus[0][:8], verbose=False)

    assert len(hist.train_loss) == 2
    assert hist.train_loss[-1] < hist.train_loss[0]   # learning happened
    assert len(hist.val_ppl) == 2 and all(np.isfinite(hist.val_ppl))
    assert hist.best_path is not None and (tmp_path / "best.pt").exists()


def test_sample_continuation_decodes_notes():
    rng = np.random.default_rng(1)
    tok = MidiTokenizer()
    torch.manual_seed(0)
    gpt = GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=128, dropout=0.0)
    model = build_model(gpt)
    prompt = data.random_sequence(rng, 8)
    notes = lmtrain.sample_continuation(model, tok, prompt, max_new_tokens=16, top_k=8)
    assert isinstance(notes, list)
    assert all(21 <= n.pitch <= 108 for n in notes)
