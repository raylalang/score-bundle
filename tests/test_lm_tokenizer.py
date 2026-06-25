import numpy as np

from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent, BOS, EOS


def test_roundtrip_recovers_notes():
    tok = MidiTokenizer(grid=24, n_vel_bins=32)
    notes = [
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.5, 0.5, 64),
        NoteEvent(67, 1.0, 1.0, 100),
        NoteEvent(72, 2.0, 0.25, 40),
    ]
    toks = tok.encode(notes)
    dec = tok.decode(toks)
    assert len(dec) == len(notes)
    for a, b in zip(notes, dec):
        assert a.pitch == b.pitch
        assert abs(a.onset - b.onset) <= 1.0 / tok.grid + 1e-9
        assert abs(a.duration - b.duration) <= 1.0 / tok.grid + 1e-9
        assert abs(a.velocity - b.velocity) <= 128 / tok.n_vel_bins


def test_token_ids_in_vocab_and_grouped():
    tok = MidiTokenizer()
    notes = [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(62, 0.5, 0.5, 90)]
    toks = tok.encode(notes)
    assert toks[0] == BOS and toks[-1] == EOS
    assert all(0 <= t < tok.vocab_size for t in toks)
    # 2 notes * 4 fields + BOS + EOS
    assert len(toks) == 2 * 4 + 2


def test_token_type_classification():
    tok = MidiTokenizer()
    assert tok.token_type(BOS) == ("special", BOS)
    assert tok.token_type(tok.pitch_base)[0] == "pitch"
    assert tok.token_type(tok.pitch_base)[1] == tok.pitch_min
    assert tok.token_type(tok.vel_base)[0] == "velocity"
