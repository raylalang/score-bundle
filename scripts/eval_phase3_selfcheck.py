"""Synthetic self-consistency: is the waveform posterior calibrated when
the model IS true?  Take real notes' fitted structure (curve at gt_c,
LS amplitudes, fitted noise), synthesize x* = Phi a + noise, infer c with
the same flat pipeline, record z = (mean - c_true)/sd.  If z ~ N(0,1),
the machinery is internally calibrated and the field overconfidence is
model/estimand mismatch, not a bug."""
import os
import pickle, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
import soundfile as sf
from scipy.signal import resample_poly
from score_bundle.phase2.urmp import read_notes_annotation
import eval_phase3_waveform_dev as ev

rng = np.random.default_rng(7)
zs = []
t_start = time.time()
for key, d, tr in ev.selected()[:4]:            # 4 tracks x 10 notes
    notes = read_notes_annotation(tr.notes)
    idx = ev.eligible_notes(d, notes)[:10]
    audio48, sr48 = sf.read(tr.audio)
    audio = resample_poly(np.asarray(audio48, dtype=float), ev.SR, int(sr48))
    for i in idx:
        on, du = notes["onset"][i], min(notes["duration"][i], ev.MAX_SEG_S)
        x = audio[int((on + 0.02) * ev.SR):int((on + du - 0.02) * ev.SR)]
        if x.size < ev.SR // 8:
            continue
        t = np.arange(x.size) / ev.SR
        midi = float(d["midi"][i])
        c_true = float(d["est_gt"][i, 0])
        f0 = ev.f0_curve(t, midi, c_true)
        Phi = ev.chunked_design(f0, t)
        a_hat, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        r = x - Phi @ a_hat
        nv = float(r @ r / max(x.size - Phi.shape[1], 1))
        x_syn = Phi @ a_hat + rng.normal(0.0, np.sqrt(nv), x.size)
        mean, sd, _, _ = ev.infer_c(x_syn, t, midi, False, {})
        zs.append((mean - c_true) / sd)
zs = np.array(zs)
os.makedirs("results/phase3_cells", exist_ok=True)
pickle.dump(zs, open("results/phase3_cells/selfcheck_z.pkl", "wb"))
print(f"n={zs.size}  median |z| = {np.median(np.abs(zs)):.2f}  "
      f"cov@90 = {np.mean(np.abs(zs) <= 1.6449):.2f}  "
      f"mean z = {zs.mean():+.2f}  [{time.time()-t_start:.0f}s]")
