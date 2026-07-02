"""Stage-2 masked objective: structural guarantees + a learning smoke test.

Pins the properties the Stage-2 design rests on:
  1. masking touches only velocity tokens, targets only masked positions;
  2. ``causal=False`` really is bidirectional (and ``causal=True`` really is not);
  3. the Stage-2 read-out is leak-free *by construction* — a hidden note's embedding
     is exactly invariant to its own performed velocity;
  4. a tiny model trained on the masked objective actually learns a deterministic
     pitch->velocity rule (the objective is learnable end-to-end).

Torch-guarded (no-op without torch), mirroring tests/test_lm_torch.py.
"""
import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
from score_bundle.lm.data import pack_tokens, random_sequence
from score_bundle.lm import masked as mk

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")


def _model(tok, causal=False, seed=0, d=32, layers=2):
    from score_bundle.lm.model_torch import GPTConfig, build_model

    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=mk.masked_vocab_size(tok), d_model=d, n_layer=layers,
                    n_head=4, block_size=256, dropout=0.0, causal=causal)
    return build_model(cfg)


def test_mask_velocities_targets_and_scope():
    tok = MidiTokenizer()
    rng = np.random.default_rng(0)
    notes = random_sequence(rng, n_notes=50)
    x = np.asarray(tok.encode(notes, add_bos_eos=False))
    xm, tg = mk.mask_velocities(x, tok, rng, observed_frac=0.5)
    is_vel = mk.velocity_token_mask(tok, x)
    hidden = xm == mk.mask_token_id(tok)
    assert hidden.any() and (~hidden).any()
    assert (is_vel | ~hidden).all()                      # only velocity tokens masked
    assert (tg[hidden] == x[hidden]).all()               # targets = original ids there
    assert (tg[~hidden] == mk.IGNORE_INDEX).all()        # loss nowhere else
    assert (xm[~hidden] == x[~hidden]).all()             # nothing else altered
    # rate is respected on average
    rates = []
    for s in range(20):
        r = np.random.default_rng(s)
        _, t = mk.mask_velocities(x, tok, r, observed_frac=0.7)
        rates.append((t != mk.IGNORE_INDEX).sum() / is_vel.sum())
    assert 0.15 < float(np.mean(rates)) < 0.45           # ~30% hidden


def test_mask_velocities_always_supervises_something():
    tok = MidiTokenizer()
    rng = np.random.default_rng(1)
    notes = random_sequence(rng, n_notes=8)
    x = np.asarray(tok.encode(notes, add_bos_eos=False))
    for s in range(30):
        _, tg = mk.mask_velocities(x, tok, np.random.default_rng(s), observed_frac=1.0)
        assert (tg != mk.IGNORE_INDEX).any()


@needs_torch
def test_bidirectional_flag_controls_information_flow():
    tok = MidiTokenizer()
    rng = np.random.default_rng(2)
    notes = random_sequence(rng, n_notes=20)
    x = np.asarray(tok.encode(notes, add_bos_eos=False))
    x2 = x.copy()
    x2[-1] = tok.vel_base + 3  # change the LAST token only
    for causal, should_change in [(False, True), (True, False)]:
        model = _model(tok, causal=causal)
        model.eval()
        with torch.no_grad():
            h1 = model.embed(torch.as_tensor(x)[None])[0, 0]
            h2 = model.embed(torch.as_tensor(x2)[None])[0, 0]
        changed = bool((h1 - h2).abs().max() > 1e-6)
        assert changed == should_change, f"causal={causal}"


@needs_torch
def test_stage2_readout_is_leakfree_by_construction():
    """A hidden note's embedding must be EXACTLY invariant to its own velocity."""
    tok = MidiTokenizer()
    rng = np.random.default_rng(3)
    notes = random_sequence(rng, n_notes=24)
    model = _model(tok, causal=False)
    observed = np.ones(len(notes), dtype=bool)
    observed[[5, 11, 17]] = False
    H1 = mk.masked_note_embeddings_long(model, tok, notes, observed)
    bumped = [NoteEvent(n.pitch, n.onset, n.duration,
                        30 if i in (5, 11, 17) else n.velocity)
              for i, n in enumerate(notes)]
    H2 = mk.masked_note_embeddings_long(model, tok, bumped, observed)
    np.testing.assert_allclose(H1, H2, atol=1e-6)
    # ...while an OBSERVED note's velocity does reach the model
    bumped_obs = [NoteEvent(n.pitch, n.onset, n.duration,
                            30 if i == 0 else n.velocity)
                  for i, n in enumerate(notes)]
    H3 = mk.masked_note_embeddings_long(model, tok, bumped_obs, observed)
    assert np.abs(H1 - H3).max() > 1e-6


@needs_torch
def test_masked_training_learns_pitch_velocity_rule():
    """End-to-end: velocity deterministically tied to pitch must become predictable."""
    from score_bundle.lm.train import TrainConfig

    tok = MidiTokenizer()
    rng = np.random.default_rng(4)

    def piece():
        notes = random_sequence(rng, n_notes=40)
        return [NoteEvent(n.pitch, n.onset, n.duration,
                          int(np.clip((n.pitch - 30), 1, 127)))  # vel = f(pitch)
                for n in notes]

    stream = pack_tokens([tok.encode(piece(), add_bos_eos=False) for _ in range(40)])
    model = _model(tok, causal=False, d=64, layers=2)
    cfg = TrainConfig(block_size=128, batch_size=16, epochs=6, steps_per_epoch=150,
                      lr=3e-3, warmup_steps=30, eval_batches=8, out_dir="")
    hist = mk.train_masked(model, stream, stream, cfg, tok, verbose=False)
    # 32 velocity bins -> chance CE = log 32 ~ 3.47; the rule is deterministic
    assert hist.val_loss[-1] < 1.0
    assert hist.val_loss[-1] < hist.val_loss[0]


@needs_torch
def test_direct_velocity_mean_units():
    tok = MidiTokenizer()
    n = 6
    vel = np.array([40, 60, 80, 100, 50, 70], dtype=float)
    observed = np.array([True, True, True, False, False, True])
    # logits sharply peaked at each note's true bin -> prediction ~ bin center
    logits = np.full((n, tok.n_vel_bins), -30.0)
    bins = (vel.astype(int) * tok.n_vel_bins) // 128
    logits[np.arange(n), bins] = 30.0
    v = mk.direct_velocity_mean(tok, logits, vel, observed)
    offset = np.mean(vel[observed] / 127.0)
    np.testing.assert_allclose(v[observed], vel[observed] / 127.0 - offset, atol=1e-9)
    centers = (bins + 0.5) * 128.0 / tok.n_vel_bins
    np.testing.assert_allclose(v[~observed], centers[~observed] / 127.0 - offset, atol=0.02)
