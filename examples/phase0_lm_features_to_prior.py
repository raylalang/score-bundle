"""Phase 0 -> Phase 1 bridge: LM embeddings as a learned prior mean.

Extract per-note embeddings from the (here untrained) NumPy LM, fit a ridge head
h_i -> mu_i, and plug mu into the Phase-1 graph prior so the GMRF models the residual
y - mu.  With an untrained model this is a plumbing demo (mu is uninformative); after
real pretraining, mu carries the expressive signal and the graph prior adds calibrated
structured uncertainty on top.

Run:  python examples/phase0_lm_features_to_prior.py
"""
import numpy as np

from score_bundle import Score, build_adjacency, laplacian, laplacian_precision, GraphGaussianField
from score_bundle import metrics
from score_bundle.lm import data, features, model_numpy as mnp
from score_bundle.lm.tokenizer import MidiTokenizer


def main() -> None:
    rng = np.random.default_rng(1)
    tok = MidiTokenizer()
    notes = data.random_sequence(rng, n_notes=50)
    tokens = tok.encode(notes)

    cfg = mnp.GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_layer=2, n_head=4, block_size=256)
    params = mnp.init_params(cfg, rng)
    emb = features.note_embeddings(params, cfg, tok, tokens)   # (N, d)

    # a stand-in expressive target: normalized velocity per note
    y = (np.array([n.velocity for n in notes], dtype=float) - 70.0) / 30.0
    mu, _ = features.fit_prior_mean(emb, y, l2=1.0)

    score = Score.from_arrays([n.pitch for n in notes], [n.onset for n in notes], [n.duration for n in notes])
    Q = laplacian_precision(laplacian(build_adjacency(score)), lam=0.5, eta=2.0)

    m_lm, _ = GraphGaussianField(Q, mean=mu).posterior(y, noise_var=0.1)
    m_zero, _ = GraphGaussianField(Q, mean=0.0).posterior(y, noise_var=0.1)

    print(f"embeddings: {emb.shape}, notes: {len(notes)}")
    print(f"RMSE(LM-mean prior)   = {metrics.rmse(y, m_lm):.4f}")
    print(f"RMSE(zero-mean prior) = {metrics.rmse(y, m_zero):.4f}")
    print("(untrained LM: numbers are illustrative; the pipeline is what matters)")


if __name__ == "__main__":
    main()
