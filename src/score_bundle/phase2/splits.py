"""URMP development/confirmation split (Phase 2) — frozen, composition-level.

The prereg design (docs/phase2_prereg_design.md) fixes the contamination
rule: URMP reuses compositions across arrangements, so the split is by
COMPOSITION — every arrangement of one composition lands on the same side.

The split below was constructed deterministically and data-blind on
2026-08-06 (no audio or annotation values consulted; inputs are only the
published Table 1 metadata of URMP_doc.pdf) by ``construct_split`` with
``np.random.default_rng(0)``, subject to:

  (a) Jupiter pinned to DEVELOPMENT — its violin track was used in the
      2026-08-06 loader/tracker smoke test, so it can never be
      confirmation-grade;
  (b) both sides contain every ensemble type (duet/trio/quartet/quintet);
  (c) both sides contain all three instrument families
      (strings / woodwind / brass);
  (d) confirmation holds 13–17 of the 44 pieces (the composition count
      follows from the group sizes; seed 0 yields 7 compositions).

The literal piece lists are FROZEN here and mirrored in the prereg design
doc; a unit test re-runs the constructor and pins it to these literals, so
neither can drift silently.  numpy + stdlib only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# piece index -> (composition key, ensemble size, instruments) — URMP_doc Table 1
URMP_PIECES: Dict[int, Tuple[str, int, Tuple[str, ...]]] = {
    1: ("Jupiter", 2, ("vn", "vc")),
    2: ("Sonata", 2, ("vn", "vn")),
    3: ("Dance", 2, ("fl", "cl")),
    4: ("Allegro", 2, ("fl", "fl")),
    5: ("Entertainer", 2, ("tpt", "tpt")),
    6: ("Entertainer", 2, ("sax", "sax")),
    7: ("GString", 2, ("tpt", "tbn")),
    8: ("Spring", 2, ("fl", "vn")),
    9: ("Jesus", 2, ("tpt", "vn")),
    10: ("March", 2, ("tpt", "sax")),
    11: ("Maria", 2, ("ob", "vc")),
    12: ("Spring", 3, ("vn", "vn", "vc")),
    13: ("Hark", 3, ("vn", "vn", "va")),
    14: ("Waltz", 3, ("fl", "fl", "cl")),
    15: ("Surprise", 3, ("tpt", "tpt", "tbn")),
    16: ("Surprise", 3, ("tpt", "tpt", "sax")),
    17: ("Nocturne", 3, ("vn", "fl", "cl")),
    18: ("Nocturne", 3, ("vn", "fl", "tpt")),
    19: ("Pavane", 3, ("cl", "vn", "vc")),
    20: ("Pavane", 3, ("tpt", "vn", "vc")),
    21: ("Rejouissance", 3, ("cl", "tbn", "tba")),
    22: ("Rejouissance", 3, ("sax", "tbn", "tba")),
    23: ("Rejouissance", 3, ("cl", "sax", "tba")),
    24: ("Pirates", 4, ("vn", "vn", "va", "vc")),
    25: ("Pirates", 4, ("vn", "vn", "va", "sax")),
    26: ("King", 4, ("vn", "vn", "va", "vc")),
    27: ("King", 4, ("vn", "vn", "va", "sax")),
    28: ("Fugue", 4, ("fl", "ob", "cl", "bn")),
    29: ("Fugue", 4, ("fl", "fl", "ob", "cl")),
    30: ("Fugue", 4, ("fl", "fl", "ob", "sax")),
    31: ("Slavonic", 4, ("tpt", "tpt", "hn", "tbn")),
    32: ("Fugue", 4, ("vn", "vn", "va", "vc")),
    33: ("Elise", 4, ("tpt", "tpt", "hn", "tbn")),
    34: ("Fugue", 4, ("tpt", "tpt", "hn", "tbn")),
    35: ("Rondeau", 4, ("vn", "vn", "va", "db")),
    36: ("Rondeau", 4, ("vn", "vn", "va", "vc")),
    37: ("Rondeau", 4, ("fl", "vn", "va", "cl")),
    38: ("Jerusalem", 5, ("vn", "vn", "va", "vc", "db")),
    39: ("Jerusalem", 5, ("vn", "vn", "va", "sax", "db")),
    40: ("Miserere", 5, ("fl", "fl", "ob", "cl", "bn")),
    41: ("Miserere", 5, ("fl", "fl", "ob", "sax", "bn")),
    42: ("Arioso", 5, ("tpt", "tpt", "hn", "tbn", "tba")),
    43: ("Chorale", 5, ("tpt", "tpt", "hn", "tbn", "tba")),
    44: ("K515", 5, ("vn", "vn", "va", "va", "db")),
}

FAMILIES = {"vn": "strings", "va": "strings", "vc": "strings", "db": "strings",
            "fl": "wood", "ob": "wood", "cl": "wood", "bn": "wood",
            "sax": "wood", "tpt": "brass", "hn": "brass", "tbn": "brass",
            "tba": "brass"}

PINNED_DEV = ("Jupiter",)          # constraint (a); see module docstring

# FROZEN split (output of construct_split(0), 2026-08-06) ------------------
CONF_COMPOSITIONS = ("Chorale", "Elise", "Fugue", "Jesus", "Pirates",
                     "Surprise", "Waltz")
CONF_PIECES = (9, 14, 15, 16, 24, 25, 28, 29, 30, 32, 33, 34, 43)
DEV_PIECES = tuple(i for i in sorted(URMP_PIECES) if i not in CONF_PIECES)


def _coverage(pieces: Sequence[int]) -> Tuple[set, set]:
    sizes = {URMP_PIECES[i][1] for i in pieces}
    fams = {FAMILIES[x] for i in pieces for x in URMP_PIECES[i][2]}
    return sizes, fams


def construct_split(seed: int = 0) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Reconstruct the frozen split (documented procedure; see docstring).

    Greedy pass: shuffle the composition keys with ``default_rng(seed)``,
    assign groups to the confirmation pool until it holds >= 13 pieces,
    skipping pinned-development compositions and any group that would push
    it past 17 pieces.  Repair pass (deterministic): for each ensemble size
    missing from confirmation, move over the alphabetically first smallest
    development group of that size (never pinned); then, while confirmation
    exceeds 17 pieces, move back the alphabetically first smallest
    confirmation group whose removal keeps size/family coverage intact.
    """
    groups: Dict[str, List[int]] = {}
    for i, (key, _, _) in URMP_PIECES.items():
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = [keys[j] for j in rng.permutation(len(keys))]

    conf: List[str] = []
    n = 0
    for k in order:
        if k in PINNED_DEV or n >= 13:
            continue
        if n + len(groups[k]) <= 17:
            conf.append(k)
            n += len(groups[k])

    def conf_pieces(sel: Sequence[str]) -> List[int]:
        return sorted(i for k in sel for i in groups[k])

    # repair: ensemble-size coverage on the confirmation side
    for size in (2, 3, 4, 5):
        if size not in {URMP_PIECES[i][1] for i in conf_pieces(conf)}:
            cands = sorted(k for k in keys
                           if k not in conf and k not in PINNED_DEV
                           and any(URMP_PIECES[i][1] == size for i in groups[k]))
            cands.sort(key=lambda k: (len(groups[k]), k))
            conf.append(cands[0])
    # repair: shrink back if over budget, preserving coverage
    while len(conf_pieces(conf)) > 17:
        for k in sorted(conf, key=lambda k: (len(groups[k]), k)):
            rest = [c for c in conf if c != k]
            sizes, fams = _coverage(conf_pieces(rest))
            if sizes == {2, 3, 4, 5} and fams == {"strings", "wood", "brass"}:
                conf.remove(k)
                break
        else:  # pragma: no cover - construction-time safety
            raise RuntimeError("cannot repair split within budget")
    return tuple(sorted(conf)), tuple(conf_pieces(conf))


def urmp_split(pieces):
    """Partition ``load_urmp_meta`` output into (development, confirmation)."""
    dev = [p for p in pieces if p.index in set(DEV_PIECES)]
    conf = [p for p in pieces if p.index in set(CONF_PIECES)]
    return dev, conf
