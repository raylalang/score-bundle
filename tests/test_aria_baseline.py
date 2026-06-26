"""The aria frozen-feature baseline must be import-guarded (no hard dependency).

aria is the optional upper-bound baseline, not the backbone; the Phase-1 core must import
and these tests must pass whether or not aria is installed.
"""
import numpy as np
import pytest

from score_bundle.lm import aria_baseline as ab


def test_module_imports_without_aria():
    # importing the module must not require aria; aria_available is a clean bool either way
    assert isinstance(ab.aria_available(), bool)


def test_extractor_raises_clear_hint_when_unavailable():
    notes = []  # contents irrelevant: the availability check fires first
    if ab.aria_available():
        pytest.skip("aria is installed; the stub path is not exercised")
    with pytest.raises(NotImplementedError) as exc:
        ab.aria_note_embeddings(notes)
    assert "aria is not available" in str(exc.value)


def test_interface_matches_note_embeddings_contract():
    # documents the contract: extractor takes (notes, checkpoint=None). We only check the
    # signature is callable with that shape; behaviour is covered above.
    import inspect

    sig = inspect.signature(ab.aria_note_embeddings)
    assert list(sig.parameters) == ["notes", "checkpoint"]
