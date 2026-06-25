"""Phase 0 — pretrain the from-scratch (PyTorch) music LM on a synthetic corpus.

Trains the tiny MusicGPT for a few hundred steps and prints loss + a sampled
continuation. Replace ``data.random_corpus`` with a real MAESTRO/Aria-MIDI loader for
actual pretraining.

Run:  python examples/phase0_pretrain_lm.py   (needs torch: pip install -e '.[train]')
"""
import numpy as np

from score_bundle.lm import data
from score_bundle.lm.tokenizer import MidiTokenizer


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("This example needs PyTorch. Install with:  pip install -e '.[train]'")
        return

    from score_bundle.lm.model_torch import GPTConfig, build_model, train_lm

    rng = np.random.default_rng(0)
    tok = MidiTokenizer()
    corpus = data.random_corpus(rng, n_seqs=64, n_notes=40)
    stream = data.pack_tokens(data.encode_corpus(tok, corpus))
    print(f"vocab={tok.vocab_size}  corpus tokens={len(stream)}")

    block = 128
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_layer=3, n_head=4, block_size=block)
    model = build_model(cfg)
    batches = data.lm_batches(stream, block, batch_size=16, rng=rng, n_batches=200)
    losses = train_lm(model, batches, log_every=50)
    print(f"start loss {losses[0]:.3f} -> end loss {losses[-1]:.3f}")

    import torch

    prompt = torch.as_tensor(tok.encode(corpus[0])[:20], dtype=torch.long)[None]
    out = model.generate(prompt, max_new_tokens=40, top_k=8)[0].tolist()
    print(f"sampled {len(tok.decode(out))} notes from {len(out)} tokens")


if __name__ == "__main__":
    main()
