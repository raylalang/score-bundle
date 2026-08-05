"""URMP loader: inventory parsing against a synthetic fixture of the
documented layout (URMP_doc.pdf naming convention). numpy-only."""
import os

import numpy as np

from score_bundle.phase2.urmp import (load_urmp_meta, read_f0_annotation,
                                      read_notes_annotation)


def _make_fixture(root):
    folder = os.path.join(root, "01_Jupiter_vn_vc")
    os.makedirs(folder)
    names = ["AuMix_01_Jupiter_vn_vc.wav", "Sco_01_Jupiter_vn_vc.mid",
             "AuSep_1_vn_01_Jupiter.wav", "AuSep_2_vc_01_Jupiter.wav",
             "Notes_1_vn_01_Jupiter.txt", "Notes_2_vc_01_Jupiter.txt",
             "F0s_1_vn_01_Jupiter.txt", "F0s_2_vc_01_Jupiter.txt",
             "Vid_01_Jupiter_vn_vc.mp4"]
    for nme in names:
        with open(os.path.join(folder, nme), "w") as fh:
            if nme.startswith("F0s"):
                fh.write("0.023 440.0\n0.033 0\n0.043 441.5\n")
            elif nme.startswith("Notes"):
                fh.write("0.10 440.0 0.50\n0.70 493.9 0.25\n")
    # a non-piece folder that must be ignored
    os.makedirs(os.path.join(root, "Supplementary"))
    return folder


def test_inventory_matches_documented_layout(tmp_path):
    root = str(tmp_path)
    _make_fixture(root)
    pieces = load_urmp_meta(root)
    assert len(pieces) == 1
    p = pieces[0]
    assert p.index == 1 and p.name == "Jupiter"
    assert p.instruments == ["vn", "vc"]
    assert p.score_mid.endswith("Sco_01_Jupiter_vn_vc.mid")
    assert p.mix_audio.endswith("AuMix_01_Jupiter_vn_vc.wav")
    assert [t.instrument for t in p.tracks] == ["vn", "vc"]
    assert p.tracks[0].audio.endswith("AuSep_1_vn_01_Jupiter.wav")
    assert p.tracks[1].f0s.endswith("F0s_2_vc_01_Jupiter.txt")


def test_annotation_readers(tmp_path):
    folder = _make_fixture(str(tmp_path))
    t, f0 = read_f0_annotation(os.path.join(folder, "F0s_1_vn_01_Jupiter.txt"))
    np.testing.assert_allclose(t, [0.023, 0.033, 0.043])
    assert f0[1] == 0.0 and f0[0] == 440.0
    notes = read_notes_annotation(os.path.join(folder,
                                               "Notes_1_vn_01_Jupiter.txt"))
    np.testing.assert_allclose(notes["onset"], [0.10, 0.70])
    np.testing.assert_allclose(notes["duration"], [0.50, 0.25])


def test_missing_files_tolerated(tmp_path):
    folder = os.path.join(str(tmp_path), "02_Sonata_vn_vn")
    os.makedirs(folder)
    open(os.path.join(folder, "Sco_02_Sonata_vn_vn.mid"), "w").close()
    pieces = load_urmp_meta(str(tmp_path))
    assert len(pieces) == 1
    p = pieces[0]
    assert p.instruments == ["vn", "vn"]
    assert p.tracks[0].audio == "" and p.tracks[1].f0s == ""
    assert p.score_mid != ""
