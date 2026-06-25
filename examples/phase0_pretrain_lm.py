"""Phase 0 — pretrain (or, without torch, sanity-run) the from-scratch music LM.

With PyTorch installed this trains the tiny MusicGPT for a few steps on a synthetic
corpus and prints the loss going down.  Without torch it falls back to the NumPy
model: build it, run a forward pass, and sample a short continuation — enough to
exercise the tokenizer + architecture end to end.

Run:  python examples/phase0_pretrain_lm.py
"""
import numpy as np

from score_bundle.lm import data, model_numpy as mnp
from score_bundle.lm.tokenizer import MidiTokenizer


def main() -> None:
    rng = np.random.default_rng(0)
    tok = MidiTokenizer()
    corpus = data.random_corpus(rng, n_seqs=64, n_notes=40)
    stream = data.pack_tokens(data.encode_corpus(tok, corpus))
    print(f"vocab={tok.vocab_size}  corpus tokens={len(stream)}")

    block_size = 128
    try:
        import torch  # noqa: F401
        from score_bundle.lm.model_torch import GPTConfig, build_model, train_lm

        cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_layer=3, n_head=4, block_size=block_size)
        model = build_model(cfg)
        batches = data.lm_batches(stream, block_size, batch_size=16, rng=rng, n_batches=200)
        losses = train_lm(model, batches, log_every=50)
        print(f"[torch] start loss {losses[0]:.3f} -> end loss {losses[-1]:.3f}")
    except ImportError:
        cfg = mnp.GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_layer=2, n_head=4, block_size=block_size)
        params = mnp.init_params(cfg, rng)
        prompt = np.asarray(tok.encode(corpus[0])[:20])
        out = mnp.generate(params, cfg, prompt, max_new_tokens=40, rng=rng, top_k=8)
        notes = tok.decode(list(out))
        print(f"[numpy] no torch installed — ran forward + sampling; decoded {len(notes)} notes "
              f"from {len(out)} tokens (untrained model, plumbing demo).")


if __name__ == "__main__":
    main()
