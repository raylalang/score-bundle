"""The headline Phase-0 regression: the predictive per-note embedding must NOT leak the
note's own performed velocity.

Root cause of the 2026-07-02 correction: reading the per-note embedding at the VELOCITY
token (the historical ``readout="velocity"``) exposes the note's own velocity to a
"score-conditioned" prior mean. The leak-free read-out (``readout="pre_velocity"``, the
DURATION token) is causally blind to that velocity. These tests prove the invariance
structurally on a tiny model, with no training — the property is exact by causality.

Torch-guarded (no-op without torch), mirroring tests/test_lm_torch.py.
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
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4,
                    block_size=128, dropout=0.0)
    return build_model(cfg).eval(), cfg


def _flip_velocity(tok, tokens, note_idx):
    """Return a copy of ``tokens`` with note ``note_idx``'s VELOCITY token set to a
    different velocity bin (still a valid velocity token, so the stride is preserved)."""
    vel_pos = features.note_velocity_positions(tok, tokens)
    p = vel_pos[note_idx]
    field, vbin = tok.token_type(tokens[p])
    assert field == "velocity"
    new = tok.vel_base + (vbin + tok.n_vel_bins // 2) % tok.n_vel_bins
    out = list(tokens)
    out[p] = new
    return out, p


def test_score_positions_are_pre_velocity():
    tok = MidiTokenizer()
    notes = random_sequence(np.random.default_rng(0), n_notes=10)
    tokens = tok.encode(notes, add_bos_eos=False)
    vel = features.note_velocity_positions(tok, tokens)
    score = features.note_score_positions(tok, tokens)
    assert len(vel) == len(score) == len(notes)
    assert score == [p - 1 for p in vel]
    # each score position really is a DURATION token
    assert all(tok.token_type(tokens[p])[0] == "duration" for p in score)


def test_pre_velocity_readout_does_not_leak_velocity():
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    notes = random_sequence(np.random.default_rng(1), n_notes=16)
    tokens = tok.encode(notes, add_bos_eos=False)

    emb_v = features.note_embeddings(model, tok, tokens, readout="velocity")
    emb_s = features.note_embeddings(model, tok, tokens, readout="pre_velocity")
    assert emb_v.shape == emb_s.shape == (len(notes), cfg.d_model)

    j = 8
    tokens2, p = _flip_velocity(tok, tokens, j)
    emb_v2 = features.note_embeddings(model, tok, tokens2, readout="velocity")
    emb_s2 = features.note_embeddings(model, tok, tokens2, readout="pre_velocity")

    # leak-free read-out is INVARIANT to the note's own velocity (causally blind)
    np.testing.assert_allclose(emb_s2[j], emb_s[j], atol=1e-5)
    # the historical velocity read-out DOES change (it saw the flipped velocity) -> the leak
    assert not np.allclose(emb_v2[j], emb_v[j], atol=1e-5)
    # notes strictly before j are unaffected in BOTH read-outs (causal)
    np.testing.assert_allclose(emb_s2[:j], emb_s[:j], atol=1e-5)
    np.testing.assert_allclose(emb_v2[:j], emb_v[:j], atol=1e-5)


def test_pre_velocity_readout_still_uses_context():
    """Leak-free != context-free: changing an EARLIER note's score (pitch) token must move a
    later note's pre-velocity embedding (it still conditions on the score context)."""
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    notes = random_sequence(np.random.default_rng(2), n_notes=16)
    tokens = tok.encode(notes, add_bos_eos=False)
    emb_s = features.note_embeddings(model, tok, tokens, readout="pre_velocity")

    # flip the PITCH token of note 3 (an earlier note), keep it a valid pitch token
    vel = features.note_velocity_positions(tok, tokens)
    pitch_pos = vel[3] - 2  # group is [TIME_SHIFT, PITCH, DURATION, VELOCITY]
    field, val = tok.token_type(tokens[pitch_pos])
    assert field == "pitch"
    tokens2 = list(tokens)
    tokens2[pitch_pos] = tok.pitch_base + ((val - tok.pitch_min + 3) % tok.n_pitch)
    emb_s2 = features.note_embeddings(model, tok, tokens2, readout="pre_velocity")

    assert not np.allclose(emb_s2[8], emb_s[8], atol=1e-5)   # later note sees the change


def test_readout_kwarg_defaults_and_validates():
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    notes = random_sequence(np.random.default_rng(3), n_notes=12)
    tokens = tok.encode(notes, add_bos_eos=False)
    # default readout is the historical velocity position (backward compatible)
    default = features.note_embeddings(model, tok, tokens)
    velo = features.note_embeddings(model, tok, tokens, readout="velocity")
    np.testing.assert_array_equal(default, velo)
    # aliases resolve to the same leak-free positions
    a = features.note_embeddings(model, tok, tokens, readout="pre_velocity")
    b = features.note_embeddings(model, tok, tokens, readout="score")
    np.testing.assert_array_equal(a, b)
    # unknown readout raises
    try:
        features.note_embeddings(model, tok, tokens, readout="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_note_embeddings_long_readout_matches_single_window():
    """On a piece that fits one window, note_embeddings_long(readout=...) must equal
    note_embeddings(readout=...) — the windowed path uses the same positions."""
    if not HAS_TORCH:
        return
    tok = MidiTokenizer()
    model, cfg = _model(tok)
    notes = random_sequence(np.random.default_rng(4), n_notes=16)
    tokens = tok.encode(notes, add_bos_eos=False)
    for readout in ("velocity", "pre_velocity"):
        single = features.note_embeddings(model, tok, tokens, readout=readout)
        long = features.note_embeddings_long(model, tok, notes, readout=readout)
        np.testing.assert_allclose(long, single, atol=1e-5)
