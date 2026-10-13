"""Differentiable curve-from-waveform inference (Phase 3, PyTorch).

The gradients-through-synthesis step of the DDSP direction
(docs/ddsp_review.md), with the model UNCHANGED: the chunked harmonic
design of the Phase-3 development study, the amplitude-collapsed
Gaussian waveform likelihood (:mod:`.waveform_model`, Woodbury form),
and the within-note GP curve prior (the two-component kernel of
:mod:`..phase2.sm_estimator`) on a knot grid:

    cents(t) = c + interp(t; knots, u),   u ~ N(0, K_knots)
    x | c, u ~ N(0, amp_var * Phi(cents) Phi^T + noise_var * I)

MAP over (c, u) by Adam from a coarse-grid c initializer (flat curve);
the noise variance is profiled at the initializer and held fixed, as in
the Nelder-Mead prototype.  NB the likelihood depends on (c, u) only
through c + interp(u), so (c + delta, u - delta) is a likelihood-flat
direction curved only by the prior on u: the joint Laplace covariance is
legitimately large along it, and :meth:`CurveFit.curve` bands are exactly
invariant to it (the quantity to consume; never read ``cov[0, 0]`` alone
as "the uncertainty of c").  Per-note uncertainty and log evidence come
from a FULL joint Laplace approximation at the MAP — this supersedes the
prototype's diagonal Laplace (scripts/proto_phase3_curve.py, kept as the
dated record).  The prior on c is flat (improper), so the evidence is
defined up to one shared constant: values compare across kernel
hyperparameters, not across models with a different c prior.

Equality with the numpy reference path (synth.harmonic_design_matrix,
waveform_model.collapsed_loglik_lowrank, sm_estimator.sm_kernel) is
pinned by tests/test_phase3_curve.py.

Requires PyTorch.  The import is guarded (like the Phase-0 LM) so the
numpy-only package still imports without torch; calling anything here
without it raises ImportError.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

try:  # optional dependency (the numpy core must import without torch)
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch
    _HAS_TORCH = False

_LOG2PI = math.log(2.0 * math.pi)


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "score_bundle.phase3.curve requires PyTorch "
            "(pip install -e '.[train]'); the numpy reference path is "
            "synth.py + waveform_model.py."
        )


@dataclass(frozen=True)
class CurvePrior:
    """Within-note curve prior: two-component kernel + knot grid.

    Defaults are the documented prototype SCALES (not fits): a vibrato
    component of ~sqrt(w1) cents around mu1 Hz with bandwidth v1, and a
    drift component of ~sqrt(w2) cents decorrelating over ~1/sqrt(v2) s.
    ``knots_per_s`` is a resolution parameter (Nyquist for vibrato —
    results/phase3_curve_proto_dev.md), not a tuning knob.
    """

    w1: float = 30.0
    mu1: float = 5.5
    v1: float = 0.4
    w2: float = 60.0
    v2: float = 0.3
    knots_per_s: float = 16.0
    j_min: int = 12
    j_max: int = 32

    def n_knots(self, span_s: float) -> int:
        return int(np.clip(round(span_s * self.knots_per_s),
                           self.j_min, self.j_max))


def sm_kernel_torch(tau, w1, mu1, v1, w2, v2):
    """The two-component kernel k(tau), differentiable in everything.

    Mirrors :func:`score_bundle.phase2.sm_estimator.sm_kernel`; the
    hyperparameters may be python floats or torch scalars (the latter is
    what the evidence-based hyperparameter fit differentiates through).
    """
    _require_torch()
    two_pi2 = 2.0 * math.pi ** 2
    return (w1 * torch.exp(-two_pi2 * v1 * tau ** 2)
            * torch.cos(2.0 * math.pi * mu1 * tau)
            + w2 * torch.exp(-two_pi2 * v2 * tau ** 2))


def chunked_design_torch(cents, t, midi: float, n_harm: int = 8,
                         n_chunk: int = 4):
    """Chunked harmonic design Phi(cents), differentiable in cents.

    Torch mirror of the development study's design (eval_phase3_waveform_dev
    .chunked_design over synth.harmonic_design_matrix): cumulative-phase
    oscillator columns [cos, sin] per harmonic, repeated per time chunk so
    the amplitude envelope is piecewise constant.  Shape
    (len(t), 2 * n_harm * n_chunk).
    """
    _require_torch()
    f0 = 440.0 * torch.pow(
        cents.new_tensor(2.0), (midi - 69.0 + cents / 100.0) / 12.0)
    dt = torch.diff(t, prepend=t[:1])
    integral = torch.cumsum(f0 * dt, dim=0)
    cols = []
    for k in range(1, n_harm + 1):
        phi = 2.0 * math.pi * k * integral
        cols.append(torch.cos(phi))
        cols.append(torch.sin(phi))
    base = torch.stack(cols, dim=1)
    m = t.shape[0]
    edges = np.linspace(0, m, n_chunk + 1).astype(int)
    blocks = []
    for q in range(n_chunk):
        msk = torch.zeros(m, 1, dtype=base.dtype, device=base.device)
        msk[edges[q]:edges[q + 1]] = 1.0
        blocks.append(base * msk)
    return torch.cat(blocks, dim=1)


def collapsed_loglik_torch(x, Phi, noise_var: float, amp_var: float):
    """log N(x; 0, amp_var * Phi Phi^T + noise_var I), Woodbury form.

    Torch mirror of waveform_model.collapsed_loglik_lowrank with the
    isotropic amplitude prior Sigma_a = amp_var I (the only form the
    studies use); differentiable in Phi.  O(m p^2), never forms (m, m).
    """
    _require_torch()
    m, p = Phi.shape
    G = Phi.T @ Phi + (noise_var / amp_var) * torch.eye(
        p, dtype=Phi.dtype, device=Phi.device)
    L = torch.linalg.cholesky(G)
    Px = Phi.T @ x
    w = torch.cholesky_solve(Px.unsqueeze(1), L).squeeze(1)
    quad = (x @ x - Px @ w) / noise_var
    logdet = ((m - p) * math.log(noise_var)
              + 2.0 * torch.log(torch.diagonal(L)).sum()
              + p * math.log(amp_var))
    return -0.5 * (m * _LOG2PI + logdet + quad)


def fit_noise_torch(x, Phi) -> float:
    """Residual noise variance at the least-squares amplitudes (profile)."""
    _require_torch()
    beta = torch.linalg.lstsq(Phi, x.unsqueeze(1)).solution.squeeze(1)
    r = x - Phi @ beta
    return float(r @ r) / max(x.shape[0] - Phi.shape[1], 1)


def knot_gram(knots, prior: CurvePrior, w1=None, mu1=None, v1=None,
              w2=None, v2=None):
    """Prior gram K on the knot grid (jittered) with its Cholesky factor.

    Hyperparameter arguments override the prior's values and may be torch
    scalars with requires_grad (the evidence-fit path); otherwise the
    prior's floats are used.  Returns (K, L) with K = L L^T.
    """
    _require_torch()
    w1 = prior.w1 if w1 is None else w1
    mu1 = prior.mu1 if mu1 is None else mu1
    v1 = prior.v1 if v1 is None else v1
    w2 = prior.w2 if w2 is None else w2
    v2 = prior.v2 if v2 is None else v2
    tau = knots[:, None] - knots[None, :]
    K = sm_kernel_torch(tau, w1, mu1, v1, w2, v2)
    K = K + (1e-6 * (w1 + w2)) * torch.eye(
        knots.shape[0], dtype=knots.dtype, device=knots.device)
    return K, torch.linalg.cholesky(K)


def interp_knots(u, knots, t):
    """Linear interpolation of knot values u onto t, differentiable in u."""
    _require_torch()
    j = knots.shape[0]
    idx = torch.clamp(torch.searchsorted(knots, t) - 1, 0, j - 2)
    t0, t1 = knots[idx], knots[idx + 1]
    w = (t - t0) / (t1 - t0)
    return (1 - w) * u[idx] + w * u[idx + 1]


def make_neg_log_joint(x, t, midi: float, knots, chol_K, noise_var: float,
                       amp_var: float, n_harm: int,
                       n_chunk: int) -> Callable:
    """Negative log joint -log p(x, u | c) over packed theta = [c, u].

    Includes the prior normalizer (0.5 log det K + 0.5 J log 2pi) so the
    Laplace evidence is comparable across kernel hyperparameters; the flat
    prior on c contributes no term (improper, shared constant).
    """
    _require_torch()
    j = knots.shape[0]
    logdet_k = 2.0 * torch.log(torch.diagonal(chol_K)).sum()

    def nlj(theta):
        c, u = theta[0], theta[1:]
        cents = c + interp_knots(u, knots, t)
        Phi = chunked_design_torch(cents, t, midi, n_harm, n_chunk)
        alpha = torch.cholesky_solve(u.unsqueeze(1), chol_K).squeeze(1)
        prior = 0.5 * (u @ alpha) + 0.5 * logdet_k + 0.5 * j * _LOG2PI
        return -collapsed_loglik_torch(x, Phi, noise_var, amp_var) + prior

    return nlj


@dataclass
class CurveFit:
    """MAP curve with full joint Laplace uncertainty and evidence."""

    c: float                      # constant centre (cents)
    u: np.ndarray                 # knot values (J,)
    knots: np.ndarray             # knot times (J,)
    cov: np.ndarray               # Laplace covariance over (c, u), (J+1, J+1)
    noise_var: float              # profiled waveform noise variance
    log_evidence: float           # Laplace log p(x) (up to the flat-c const)
    neg_log_joint: float          # nlj at the MAP
    laplace_pd: bool              # Hessian was PD (eigen-clipped if False)
    midi: float = 0.0

    def _weights(self, tq: np.ndarray) -> np.ndarray:
        """Rows of interp weights over (c, u) — cents(tq) = A @ [c, u]."""
        tq = np.asarray(tq, dtype=float)
        j = self.knots.size
        idx = np.clip(np.searchsorted(self.knots, tq) - 1, 0, j - 2)
        w = (tq - self.knots[idx]) / (self.knots[idx + 1] - self.knots[idx])
        A = np.zeros((tq.size, j + 1))
        A[:, 0] = 1.0
        rows = np.arange(tq.size)
        A[rows, 1 + idx] += 1.0 - w
        A[rows, 1 + idx + 1] += w
        return A

    def curve(self, tq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd of cents(t) at query times tq.

        cents(t) is linear in (c, u), so the band is exact given the
        Laplace covariance: var = a^T Cov a with a the interp weights.
        """
        A = self._weights(tq)
        mean = A @ np.concatenate([[self.c], self.u])
        var = np.einsum("ij,jk,ik->i", A, self.cov, A)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def average(self, n_grid: int = 200) -> Tuple[float, float]:
        """Posterior mean and sd of the realized average deviation.

        The uniform-grid time average of cents(t) over the note — the
        Phase-2 intonation channel's estimand — is a linear functional
        of (c, u), so its variance is exact under the Laplace covariance
        (and, like the bands, invariant to the likelihood-flat (c, u)
        direction).
        """
        a = self._weights(
            np.linspace(self.knots[0], self.knots[-1], n_grid)).mean(axis=0)
        avg = float(a @ np.concatenate([[self.c], self.u]))
        var = float(a @ self.cov @ a)
        return avg, float(np.sqrt(max(var, 0.0)))


def _laplace(nlj: Callable, theta) -> Tuple[np.ndarray, float, bool]:
    """Full Laplace at theta: (covariance, log det H, PD flag).

    A non-PD Hessian (imperfect MAP) is eigen-clipped at a small positive
    floor rather than failed — flagged so callers can report it.
    """
    H = torch.autograd.functional.hessian(nlj, theta)
    H = 0.5 * (H + H.T)
    try:
        L = torch.linalg.cholesky(H)
        logdet_h = float(2.0 * torch.log(torch.diagonal(L)).sum())
        cov = torch.cholesky_inverse(L)
        return cov.cpu().numpy(), logdet_h, True
    except Exception:
        evals, evecs = torch.linalg.eigh(H)
        floor = 1e-8 * float(evals.max().clamp(min=1.0))
        evals = evals.clamp(min=floor)
        logdet_h = float(torch.log(evals).sum())
        cov = (evecs / evals) @ evecs.T
        return cov.cpu().numpy(), logdet_h, False


@dataclass
class ChainFit:
    """Joint MAP over a chain of notes with transition-region boundaries.

    Boundary k sits between notes k and k+1 and is a REGION, not a point:
    note k+1 fades in at ``b_lo[k]``, note k fades out at
    ``b_hi[k] = b_lo[k] + width[k]`` (two independent sigmoid edges of
    fixed softness; the overlap is where both notes sound).  No Laplace
    here — batch fits report MAP quantities only.
    """

    c: np.ndarray                 # per-note centres (K,)
    us: list                      # per-note knot values
    knots: list                   # per-note knot times (segment-local)
    b_lo: np.ndarray              # fade-in of note k+1 (K-1,)
    b_hi: np.ndarray              # fade-out of note k (K-1,)
    widths: np.ndarray            # b_hi - b_lo (K-1,)
    window_mass: np.ndarray       # per-note mean window over its init extent
    noise_var: float              # after the mid-run refit
    noise_var_init: float         # at the reference initializer
    neg_log_joint: float

    def curve(self, k: int, tq: np.ndarray) -> np.ndarray:
        """MAP cents curve of note k at segment-local times tq."""
        return self.c[k] + np.interp(np.asarray(tq, float),
                                     self.knots[k], self.us[k])


def design_support_torch(cents, t_slice, midi: float, ia: int, m: int,
                         n_harm: int = 8, n_chunk: int = 2):
    """Harmonic design on a sample slice [ia, ia+len(t_slice)), zero-padded
    to m rows.

    The chunked design of :func:`chunked_design_torch` built only over the
    note's support (phase origin at the slice start — the quadrature pair
    absorbs the offset), so chain designs don't waste columns on regions a
    note's window has already zeroed.  Differentiable in cents.
    """
    _require_torch()
    block = chunked_design_torch(cents, t_slice, midi, n_harm, n_chunk)
    ms, p = block.shape
    pads = []
    if ia > 0:
        pads.append(torch.zeros(ia, p, dtype=block.dtype,
                                device=block.device))
    pads.append(block)
    if ia + ms < m:
        pads.append(torch.zeros(m - ia - ms, p, dtype=block.dtype,
                                device=block.device))
    return torch.cat(pads, dim=0)


def chain_reference(x: np.ndarray, t: np.ndarray, midis, b_ref, *,
                    prior: CurvePrior = CurvePrior(), n_harm: int = 8,
                    n_chunk: int = 2, amp_var: float = 10.0,
                    init_guard: float = 0.03, guard_frac: float = 0.0,
                    c_half_range: float = 50.0,
                    device: str = "cpu") -> Tuple[np.ndarray, float]:
    """Per-note coarse c's and the shared segment noise variance, computed
    at REFERENCE boundaries b_ref (delta-independent — every perturbation
    arm of one chain must reuse this, or the arms optimize different
    objectives; the 10-09 pilot had that confound).

    Each note is hard-sliced at b_ref with an ``init_guard`` margin (plus
    ``guard_frac`` of the slice per side, for low-trust references like
    score-nominal boundaries); the single-note coarse 1-cent grid under a
    flat curve picks c; noise_var is then profiled once on the full
    segment with the stitched flat-curve design at those c's.
    """
    _require_torch()
    dev = torch.device(device)
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64,
                          device=dev)
    t_np = np.asarray(t, float)
    m = t_np.size
    sr = (m - 1) / max(t_np[-1] - t_np[0], 1e-9)
    k_n = len(midis)
    cuts = np.concatenate([[t_np[0]], np.asarray(b_ref, float),
                           [t_np[-1]]])
    c0s = np.zeros(k_n)
    grid = np.arange(-c_half_range, c_half_range, 1.0)
    with torch.no_grad():
        for k in range(k_n):
            lo, hi = cuts[k], cuts[k + 1]
            g = init_guard + guard_frac * (hi - lo)
            ia = int(np.searchsorted(t_np, lo + g))
            ib = int(np.searchsorted(t_np, hi - g))
            if ib - ia < int(0.08 * sr):          # degenerate slice
                ia = int(np.searchsorted(t_np, lo))
                ib = int(np.searchsorted(t_np, hi))
            xs = x_t[ia:ib]
            ts = torch.as_tensor(t_np[ia:ib], dtype=torch.float64,
                                 device=dev)
            Phi0 = chunked_design_torch(torch.zeros_like(ts), ts,
                                        float(midis[k]), n_harm, n_chunk)
            nv = fit_noise_torch(xs, Phi0)
            lls = []
            for c in grid:
                Phi = chunked_design_torch(torch.full_like(ts, float(c)),
                                           ts, float(midis[k]), n_harm,
                                           n_chunk)
                lls.append(float(collapsed_loglik_torch(xs, Phi, nv,
                                                        amp_var)))
            c0s[k] = float(grid[int(np.argmax(lls))])
        # shared noise at the stitched flat-curve design (hard cuts)
        cols = []
        for k in range(k_n):
            ia = int(np.searchsorted(t_np, cuts[k]))
            ib = int(np.searchsorted(t_np, cuts[k + 1]))
            ts = torch.as_tensor(t_np[ia:ib], dtype=torch.float64,
                                 device=dev)
            cents = torch.full_like(ts, c0s[k])
            cols.append(design_support_torch(cents, ts, float(midis[k]),
                                             ia, m, n_harm, n_chunk))
        noise_var = fit_noise_torch(x_t, torch.cat(cols, dim=1))
    return c0s, noise_var


def fit_chain(x: np.ndarray, t: np.ndarray, midis, b_init, *,
              prior: CurvePrior = CurvePrior(),
              hypers: Optional[dict] = None,
              c_init: Optional[np.ndarray] = None,
              noise_var: Optional[float] = None,
              free_width: bool = True, width_init: float = 0.018,
              fixed_width: float = 0.010,
              width_prior_median: float = 0.018,
              width_prior_sigma: float = 0.6,
              softness: float = 0.010, min_width: float = 0.001,
              n_steps: int = 500, lr_curve: float = 0.4,
              lr_db_ms: float = 2.0, lr_gap: float = 0.1,
              n_harm: int = 8, n_chunk: int = 2, amp_var: float = 10.0,
              support_margin: float = 0.2, order_margin: float = 0.05,
              order_weight: float = 1e7,
              device: str = "cpu") -> ChainFit:
    """Joint gradient MAP over per-note curves + transition-region
    boundaries for one chain of notes.

    Boundaries are optimized as offsets db (ms) from ``b_init``; with
    ``free_width`` the gap carries a lognormal width prior (soft bounds —
    a sigmoid-bounded width parks at the bound with zero gradient),
    otherwise the width is frozen at ``fixed_width`` (the control arm).
    An ordering hinge keeps note k's fade-out from overrunning boundary
    k+1's fade-in.  ``c_init``/``noise_var`` should come from
    :func:`chain_reference` at the chain's REFERENCE boundaries so that
    perturbation arms share one objective; if omitted they are computed
    at ``b_init`` itself.  noise_var is refit once (no-grad) halfway.
    """
    _require_torch()
    dev = torch.device(device)
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64,
                          device=dev)
    t_np = np.asarray(t, float)
    t_t = torch.as_tensor(t_np, dtype=torch.float64, device=dev)
    m = t_np.size
    k_n = len(midis)
    b_init = np.asarray(b_init, float)
    assert b_init.size == k_n - 1

    if c_init is None or noise_var is None:
        c0s, nv0 = chain_reference(x, t, midis, b_init, prior=prior,
                                   n_harm=n_harm, n_chunk=n_chunk,
                                   amp_var=amp_var, device=device)
        c_init = c0s if c_init is None else c_init
        noise_var = nv0 if noise_var is None else noise_var
    nv_init = float(noise_var)

    cuts = np.concatenate([[t_np[0]], b_init, [t_np[-1]]])
    knots_list, chol_list, slices = [], [], []
    for k in range(k_n):
        lo, hi = cuts[k], cuts[k + 1]
        j = prior.n_knots(hi - lo)
        kn = np.linspace(lo, hi, j)
        kt = torch.as_tensor(kn, dtype=torch.float64, device=dev)
        _, chol = knot_gram(kt, prior, **(hypers or {}))
        knots_list.append((kn, kt))
        chol_list.append(chol)
        ia = int(np.searchsorted(t_np, max(t_np[0], lo - support_margin)))
        ib = int(np.searchsorted(t_np, min(t_np[-1], hi + support_margin)))
        slices.append((ia, ib))

    c = torch.tensor(np.asarray(c_init, float), dtype=torch.float64,
                     device=dev, requires_grad=True)
    us = [torch.zeros(kn.size, dtype=torch.float64, device=dev,
                      requires_grad=True) for kn, _ in knots_list]
    db_ms = torch.zeros(k_n - 1, dtype=torch.float64, device=dev,
                        requires_grad=True)
    g0 = float(np.log(np.expm1(max(width_init - min_width, 1e-4))))
    gap = torch.full((k_n - 1,), g0, dtype=torch.float64, device=dev,
                     requires_grad=free_width)
    b_init_t = torch.as_tensor(b_init, dtype=torch.float64, device=dev)

    def edges():
        b_lo = b_init_t + db_ms / 1000.0
        width = (min_width + torch.nn.functional.softplus(gap)
                 if free_width
                 else torch.full_like(b_lo, fixed_width))
        return b_lo, b_lo + width, width

    def assemble(b_lo, b_hi):
        cols = []
        for k in range(k_n):
            ia, ib = slices[k]
            ts = t_t[ia:ib]
            _, kt = knots_list[k]
            cents = c[k] + interp_knots(us[k], kt, ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, n_harm, n_chunk)
            w = torch.ones(m, dtype=torch.float64, device=dev)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            cols.append(Phi_k * w.unsqueeze(1))
        return torch.cat(cols, dim=1)

    def nlj():
        b_lo, b_hi, width = edges()
        Phi = assemble(b_lo, b_hi)
        loss = -collapsed_loglik_torch(x_t, Phi, noise_var, amp_var)
        for u, chol in zip(us, chol_list):
            loss = loss + 0.5 * (u @ torch.cholesky_solve(
                u.unsqueeze(1), chol).squeeze(1))
        if free_width:
            loss = loss + 0.5 * (((torch.log(width)
                                   - math.log(width_prior_median))
                                  / width_prior_sigma) ** 2).sum()
        for k in range(k_n - 2):
            viol = torch.relu(b_hi[k] - (b_lo[k + 1] - order_margin))
            loss = loss + order_weight * viol ** 2
        return loss

    groups = [{"params": [c] + us, "lr": lr_curve},
              {"params": [db_ms], "lr": lr_db_ms}]
    if free_width:
        groups.append({"params": [gap], "lr": lr_gap})
    opt = torch.optim.Adam(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    decay_from = int(0.75 * n_steps)
    for step in range(n_steps):
        if step >= decay_from:
            s = 0.5 * (1.0 + math.cos(
                math.pi * (step - decay_from) / max(n_steps - decay_from,
                                                    1)))
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
        opt.zero_grad()
        loss = nlj()
        loss.backward()
        opt.step()
        if step == n_steps // 2:          # one no-grad noise refit
            with torch.no_grad():
                b_lo, b_hi, _ = edges()
                Phi = assemble(b_lo, b_hi)
                noise_var = fit_noise_torch(x_t, Phi)

    with torch.no_grad():
        final = float(nlj())
        b_lo, b_hi, width = edges()
        mass = np.zeros(k_n)
        for k in range(k_n):
            w = torch.ones(m, dtype=torch.float64, device=dev)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            ia = int(np.searchsorted(t_np, cuts[k]))
            ib = max(int(np.searchsorted(t_np, cuts[k + 1])), ia + 1)
            mass[k] = float(w[ia:ib].mean())

    return ChainFit(
        c=c.detach().cpu().numpy(),
        us=[u.detach().cpu().numpy() for u in us],
        knots=[kn for kn, _ in knots_list],
        b_lo=b_lo.cpu().numpy(), b_hi=b_hi.cpu().numpy(),
        widths=width.cpu().numpy(), window_mass=mass,
        noise_var=float(noise_var), noise_var_init=nv_init,
        neg_log_joint=final)


def loglik_hessian(x: np.ndarray, t: np.ndarray, midi: float, fit: CurveFit,
                   *, n_harm: int = 8, n_chunk: int = 4,
                   amp_var: float = 10.0, device: str = "cpu") -> np.ndarray:
    """Hessian of the NEGATIVE log likelihood alone at the fit's MAP.

    No prior terms: this is the hyperparameter-independent curvature block,
    precomputed once per note so :func:`evidence_fixed_map` can move the
    kernel hyperparameters cheaply (the full Laplace Hessian is this plus
    the prior precision padded over u).
    """
    _require_torch()
    dev = torch.device(device)
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64,
                          device=dev)
    t_t = torch.as_tensor(np.asarray(t, float), dtype=torch.float64,
                          device=dev)
    knots = torch.as_tensor(fit.knots, dtype=torch.float64, device=dev)

    def nll(theta):
        cents = theta[0] + interp_knots(theta[1:], knots, t_t)
        Phi = chunked_design_torch(cents, t_t, midi, n_harm, n_chunk)
        return -collapsed_loglik_torch(x_t, Phi, fit.noise_var, amp_var)

    theta = torch.as_tensor(np.concatenate([[fit.c], fit.u]),
                            dtype=torch.float64, device=dev)
    H = torch.autograd.functional.hessian(nll, theta)
    return (0.5 * (H + H.T)).cpu().numpy()


def evidence_fixed_map(loglik_map: float, u, knots, H_lik, prior: CurvePrior,
                       hypers: Optional[dict] = None):
    """Laplace log evidence as a function of the kernel hyperparameters,
    with the MAP and the likelihood curvature FROZEN (coordinate-ascent /
    envelope step of the evidence-based hyperparameter fit).

    log Z(phi) = loglik(theta*) - u*^T K(phi)^{-1} u* / 2
                 - log det K(phi) / 2 - log det H(phi) / 2 + log(2 pi) / 2,
    H(phi) = H_lik + blockdiag(0, K(phi)^{-1}).

    ``u``, ``knots``, ``H_lik`` are torch tensors; ``hypers`` values may
    require grad.  At the fit's own hyperparameters this reproduces
    CurveFit.log_evidence exactly (test-pinned).  Returns a torch scalar.
    """
    _require_torch()
    j = knots.shape[0]
    _, chol_k = knot_gram(knots, prior, **(hypers or {}))
    logdet_k = 2.0 * torch.log(torch.diagonal(chol_k)).sum()
    eye = torch.eye(j, dtype=knots.dtype, device=knots.device)
    k_inv = torch.cholesky_solve(eye, chol_k)
    prec = torch.zeros(j + 1, j + 1, dtype=knots.dtype, device=knots.device)
    prec[1:, 1:] = k_inv
    sign, logdet_h = torch.linalg.slogdet(H_lik + prec)
    alpha = torch.cholesky_solve(u.unsqueeze(1), chol_k).squeeze(1)
    return (loglik_map - 0.5 * (u @ alpha) - 0.5 * logdet_k
            - 0.5 * logdet_h + 0.5 * _LOG2PI)


def fit_note(x: np.ndarray, t: np.ndarray, midi: float,
             prior: CurvePrior = CurvePrior(), *, n_harm: int = 8,
             n_chunk: int = 4, amp_var: float = 10.0, n_steps: int = 400,
             lr: float = 0.4, c_half_range: float = 50.0,
             device: str = "cpu",
             hypers: Optional[dict] = None) -> CurveFit:
    """Gradient MAP + full Laplace for one note's cents curve.

    x: audio segment (numpy), t: sample times (s), midi: nominal pitch.
    ``hypers`` optionally overrides kernel hyperparameters (floats or
    torch scalars — see :func:`knot_gram`).  Same initializer as the
    prototype: coarse 1-cent c grid under a flat curve, noise profiled
    at c = 0 then refit at the grid optimum and held fixed.
    """
    _require_torch()
    dev = torch.device(device)
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64,
                          device=dev)
    t_np = np.asarray(t, float)
    t_t = torch.as_tensor(t_np, dtype=torch.float64, device=dev)

    j = prior.n_knots(float(t_np[-1] - t_np[0]))
    knots_np = np.linspace(t_np[0], t_np[-1], j)
    knots = torch.as_tensor(knots_np, dtype=torch.float64, device=dev)
    _, chol_k = knot_gram(knots, prior, **(hypers or {}))

    # initializer: coarse c grid, flat curve, profiled noise
    with torch.no_grad():
        def flat_ll(c: float, nv: float) -> float:
            cents = torch.full_like(t_t, c)
            Phi = chunked_design_torch(cents, t_t, midi, n_harm, n_chunk)
            return float(collapsed_loglik_torch(x_t, Phi, nv, amp_var))

        Phi0 = chunked_design_torch(torch.zeros_like(t_t), t_t, midi,
                                    n_harm, n_chunk)
        nv0 = fit_noise_torch(x_t, Phi0)
        grid = np.arange(-c_half_range, c_half_range, 1.0)
        c0 = float(grid[int(np.argmax([flat_ll(c, nv0) for c in grid]))])
        Phi_c0 = chunked_design_torch(torch.full_like(t_t, c0), t_t, midi,
                                      n_harm, n_chunk)
        noise_var = fit_noise_torch(x_t, Phi_c0)

    nlj = make_neg_log_joint(x_t, t_t, midi, knots, chol_k, noise_var,
                             amp_var, n_harm, n_chunk)
    theta = torch.cat([
        torch.tensor([c0], dtype=torch.float64, device=dev),
        torch.zeros(j, dtype=torch.float64, device=dev),
    ]).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = nlj(theta)
        loss.backward()
        opt.step()

    theta = theta.detach()
    nlj_map = float(nlj(theta))
    cov, logdet_h, pd = _laplace(nlj, theta)
    log_ev = -nlj_map + 0.5 * (j + 1) * _LOG2PI - 0.5 * logdet_h
    return CurveFit(
        c=float(theta[0]), u=theta[1:].cpu().numpy(), knots=knots_np,
        cov=cov, noise_var=noise_var, log_evidence=log_ev,
        neg_log_joint=nlj_map, laplace_pd=pd, midi=midi)
