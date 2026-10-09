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

    def curve(self, tq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd of cents(t) at query times tq.

        cents(t) is linear in (c, u), so the band is exact given the
        Laplace covariance: var = a^T Cov a with a the interp weights.
        """
        tq = np.asarray(tq, dtype=float)
        j = self.knots.size
        idx = np.clip(np.searchsorted(self.knots, tq) - 1, 0, j - 2)
        w = (tq - self.knots[idx]) / (self.knots[idx + 1] - self.knots[idx])
        A = np.zeros((tq.size, j + 1))
        A[:, 0] = 1.0
        rows = np.arange(tq.size)
        A[rows, 1 + idx] += 1.0 - w
        A[rows, 1 + idx + 1] += w
        mean = A @ np.concatenate([[self.c], self.u])
        var = np.einsum("ij,jk,ik->i", A, self.cov, A)
        return mean, np.sqrt(np.maximum(var, 0.0))


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
