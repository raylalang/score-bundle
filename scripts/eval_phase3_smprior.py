#!/usr/bin/env python
"""Slot B measured: the SM kernel as the Phase-3 pitch-curve deviation prior (DEV).

The kill test's surviving direction: at curve level the SM prior described
the cents curve better than the sinusoid, and the Phase-3 deviation prior
is already a crude GP (eight Hann bumps, diagonal prior).  This study
swaps that basis for the exact low-rank representation of a two-component
spectral-mixture GP prior on the within-note deviation:

  k(tau) = sd^2/2 * [ exp(-2 pi^2 v1 tau^2) cos(2 pi f tau)   (vibrato band,
                       f from the note's scaffold, coherence v1 = (f/8)^2)
                    + exp(-2 pi^2 v2 tau^2) ]                  (drift band),

eigendecomposed on a coarse within-note grid, top-8 eigenvectors
interpolated to the samples and scaled by sqrt(eigenvalue) (so the
coefficient prior is the identity and the GP prior is carried by the
basis), each column de-meaned so a constant shift stays identified as c
-- everything else (Jacobian finite differences, exact amplitude
marginalization, coarse+fine grids) identical to the published
`infer_c_devprior`.  The bump variant is recomputed on the same segments,
so the comparison is paired note for note.  Scored against the
quasi-truth centre like the published study.  Development, exploratory,
no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_smprior.py run K/N
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_phase3_smprior.py report
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_waveform_dev import (AMP_VAR, MAX_SEG_S, SR, _Z90,  # noqa: E402
                                      chunked_design, eligible_notes,
                                      f0_curve, infer_c_devprior,
                                      posterior_stats, selected)

OUT_DIR = "results/phase3_smprior"


def sm_dev_basis(t, scaffold, n_comp=8, sd=5.0, grid_n=100):
    """Top-``n_comp`` scaled eigenvectors of the SM deviation kernel at the
    samples ``t`` (columns de-meaned; coefficient prior = identity)."""
    span = float(t[-1] - t[0]) if t.size > 1 else 1.0
    f = float(np.clip(scaffold.get("f", 6.0), 2.5, 9.0))
    v1 = (f / 8.0) ** 2
    v2 = max(0.1 / span ** 2, 0.05)
    tg = np.linspace(t[0], t[-1], grid_n)
    tau = tg[:, None] - tg[None, :]
    K = 0.5 * sd ** 2 * (np.exp(-2 * np.pi ** 2 * v1 * tau ** 2)
                         * np.cos(2 * np.pi * f * tau)
                         + np.exp(-2 * np.pi ** 2 * v2 * tau ** 2))
    w, V = np.linalg.eigh(K)
    order = np.argsort(w)[::-1][:n_comp]
    cols = []
    for j in order:
        lam = max(float(w[j]), 0.0)
        # eigenvector is orthonormal on the grid; density-normalize so the
        # interpolated function keeps the kernel's pointwise scale
        phi = np.interp(t, tg, V[:, j] * np.sqrt(grid_n / max(span, 1e-9)))
        col = np.sqrt(lam * span / grid_n) * phi
        col = col - col.mean()
        cols.append(col)
    return np.stack(cols, axis=1)


def infer_c_smprior(x, t, midi, scaffold):
    """`infer_c_devprior` with the SM eigenbasis in place of the Hann bumps."""
    from score_bundle.phase3.waveform_model import collapsed_loglik_lowrank

    def ll_white(c, nv, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        Sigma_a = np.eye(Phi.shape[1]) * AMP_VAR
        return collapsed_loglik_lowrank(x, Phi, Sigma_a, noise_var=nv)

    def noise_at(c, **kw):
        Phi = chunked_design(f0_curve(t, midi, c, **kw), t)
        beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        r = x - Phi @ beta
        return float(r @ r / max(x.size - Phi.shape[1], 1))

    coarse = np.arange(-50.0, 50.0 + 1e-9, 1.0)
    nv = noise_at(0.0, **scaffold)
    best = (-np.inf, 0.0, 0.0)
    for s in np.linspace(-40.0, 40.0, 9):
        kw = dict(scaffold, slope=float(s))
        lls = np.array([ll_white(c, nv, **kw) for c in coarse])
        j = int(np.argmax(lls))
        if lls[j] > best[0]:
            best = (float(lls[j]), float(coarse[j]), float(s))
    _, c0, s_hat = best
    kw = dict(scaffold, slope=s_hat)

    B = sm_dev_basis(t, scaffold)            # (m, 8), prior = identity

    def ll_dev(c, nv):
        f0 = f0_curve(t, midi, c, **kw)
        Phi = chunked_design(f0, t)
        a_hat, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        mean_wave = Phi @ a_hat
        eps = 1.0
        J = np.empty((t.size, B.shape[1]))
        for jb in range(B.shape[1]):
            f0p = f0 * 2.0 ** (eps * B[:, jb] / 1200.0)
            J[:, jb] = (chunked_design(f0p, t) @ a_hat - mean_wave) / eps
        Phi_aug = np.concatenate([Phi, J], axis=1)
        Sig = np.diag(np.concatenate([np.full(Phi.shape[1], AMP_VAR),
                                      np.ones(B.shape[1])]))
        return collapsed_loglik_lowrank(x, Phi_aug, Sig, noise_var=nv)

    nv = noise_at(c0, **kw)
    fine = c0 + np.arange(-3.0, 3.0 + 1e-9, 0.06)
    lls = np.array([ll_dev(c, nv) for c in fine])
    mean, sd = posterior_stats(fine, lls)
    return mean, sd, s_hat, 0.0


def run(shard: str) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    from score_bundle.phase2.urmp import read_notes_annotation

    k, nsh = (int(v) for v in shard.split("/"))
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    gidx = 0
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        idx = eligible_notes(d, notes)
        mine = [i for i in idx if (gidx + idx.index(i)) % nsh == k]
        gidx += len(idx)
        if not mine:
            continue
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, dtype=float), SR, int(sr48))
        for i in mine:
            t0 = time.time()
            on, du = notes["onset"][i], min(notes["duration"][i], MAX_SEG_S)
            a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
            x = audio[max(a, 0):min(b, audio.size)]
            if x.size < SR // 8:
                continue
            t = np.arange(x.size) / SR
            midi = float(d["midi"][i])
            scaffold = {}
            if d["ident"][i] and np.isfinite(d["est"][i, 1]):
                scaffold = dict(gamma=float(np.exp(d["est"][i, 1])),
                                f=float(np.exp(d["est"][i, 2])),
                                delta=float(d["dvib"][i])
                                if np.isfinite(d["dvib"][i]) else 0.0)
            rec = {"key": key, "i": i, "instr": d["instrument"],
                   "dur": float(du), "gt_c": float(d["est_gt"][i, 0])}
            rec["bump"] = infer_c_devprior(x, t, midi, scaffold)
            rec["sm"] = infer_c_smprior(x, t, midi, scaffold)
            rows.append(rec)
            print(f"{key} note {i} ({d['instr' 'ument']}): bump "
                  f"{rec['bump'][0]:+.2f}+/-{rec['bump'][1]:.2f}, sm "
                  f"{rec['sm'][0]:+.2f}+/-{rec['sm'][1]:.2f} gt "
                  f"{rec['gt_c']:+.2f} [{time.time() - t0:.0f}s]",
                  flush=True)
    out = f"{OUT_DIR}/cells.shard{k}of{nsh}.pkl"
    pickle.dump(rows, open(out, "wb"))
    print(f"wrote {out} ({len(rows)} notes)")


def report() -> None:
    rows = []
    for f in sorted(glob.glob(f"{OUT_DIR}/cells.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    print(f"{len(rows)} notes")
    fams = [("strings", ("vn", "va", "vc", "db")),
            ("winds", ("fl", "cl", "ob", "sax", "bn")),
            ("brass", ("tpt", "hn", "tbn", "tba"))]
    for name in ("bump", "sm"):
        err = np.array([abs(r[name][0] - r["gt_c"]) for r in rows])
        z = np.array([(r[name][0] - r["gt_c"]) / max(r[name][1], 1e-9)
                      for r in rows])
        print(f"{name:5s} median |err| {np.median(err):5.2f}  "
              f"q90 {np.quantile(err, .9):5.2f}  "
              f"cov@90 {np.mean(np.abs(z) <= _Z90):.3f}")
    d = np.array([abs(r["sm"][0] - r["gt_c"]) - abs(r["bump"][0] - r["gt_c"])
                  for r in rows])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, d.size, size=(2000, d.size))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"paired |err| sm - bump: dmean {d.mean():+.3f} "
          f"[{lo:+.3f},{hi:+.3f}]{star} (median {np.median(d):+.3f})")
    for fam, members in fams:
        mfam = [r["instr"] in members for r in rows]
        if any(mfam):
            df = d[np.array(mfam)]
            print(f"  {fam:8s} n={df.size:3d} dmean {df.mean():+.3f} "
                  f"median {np.median(df):+.3f}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    if sys.argv[1] == "report":
        report()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "0/1")


if __name__ == "__main__":
    main()
