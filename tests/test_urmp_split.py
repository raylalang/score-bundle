"""The frozen URMP composition split: reproducibility and constraints."""
import numpy as np

from score_bundle.phase2.splits import (CONF_COMPOSITIONS, CONF_PIECES,
                                        DEV_PIECES, FAMILIES, PINNED_DEV,
                                        URMP_PIECES, construct_split,
                                        urmp_split)


def test_constructor_reproduces_frozen_literals():
    keys, pieces = construct_split(0)
    assert keys == CONF_COMPOSITIONS
    assert pieces == CONF_PIECES


def test_partition_complete_disjoint_and_composition_pure():
    assert sorted(DEV_PIECES + CONF_PIECES) == sorted(URMP_PIECES)
    assert not set(DEV_PIECES) & set(CONF_PIECES)
    conf_keys = {URMP_PIECES[i][0] for i in CONF_PIECES}
    dev_keys = {URMP_PIECES[i][0] for i in DEV_PIECES}
    assert not conf_keys & dev_keys          # no composition straddles


def test_constraints_hold():
    # (a) pinned development
    for key in PINNED_DEV:
        assert all(URMP_PIECES[i][0] != key for i in CONF_PIECES)
    # (b)+(c) coverage on both sides
    for side in (DEV_PIECES, CONF_PIECES):
        assert {URMP_PIECES[i][1] for i in side} == {2, 3, 4, 5}
        fams = {FAMILIES[x] for i in side for x in URMP_PIECES[i][2]}
        assert fams == {"strings", "wood", "brass"}
    # (d) budget
    assert 13 <= len(CONF_PIECES) <= 17


def test_urmp_split_partitions_meta():
    class P:  # minimal stand-in for UrmpPiece
        def __init__(self, i):
            self.index = i
    pieces = [P(i) for i in sorted(URMP_PIECES)]
    dev, conf = urmp_split(pieces)
    assert [p.index for p in dev] == list(DEV_PIECES)
    assert [p.index for p in conf] == list(CONF_PIECES)
