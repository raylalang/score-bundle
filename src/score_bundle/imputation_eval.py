"""Held-out imputation comparison: prior mean source x graph residual.

The Phase-0/1 question (design §6): does a score-graph *residual* prior on top of a learned
LM mean improve recovery **and** calibration over the LM mean alone (and over hand-built
baselines)?  This module runs the controlled comparison on aligned per-note targets:

    mean source  in {zero, ridge-feature, LM}     (mu)
    graph        in {off, on}                      (model y - mu with the GMRF or not)

For each (mean, graph) cell we mask a held-out fraction of notes and predict them, then
score recovery (RMSE) and calibration (coverage / PIT / NLL).  "Graph off" predicts the
mean itself with a homoscedastic residual std (isolating the mean); "graph on" runs the
closed-form GMRF posterior centered on that mean (the structured residual).

NumPy-only (Phase-1 core): the LM mean ``mu_LM`` is passed in as an array, so this module
never imports torch.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Held-out outcome for one (mean, graph) cell: parallel arrays over held-out notes,
# with ``channel`` tagging which y-component each entry came from (for per-channel reports).
CellResult = namedtuple("CellResult", ["y", "pred", "std", "channel"])

from .baselines import ridge_impute
from .graph import build_adjacency, laplacian
from .metrics import evaluate as _evaluate_metrics
from .model import GraphGaussianField, fit_laplacian_field, fit_laplacian_field_calib
from .prior import laplacian_precision
from .score import Score


def random_mask(n: int, rng: np.random.Generator, observed_frac: float = 0.6) -> np.ndarray:
    """Boolean observed-mask of length ``n`` with ~``observed_frac`` True (>=1 held out)."""
    mask = rng.random(n) < observed_frac
    if mask.all():
        mask[rng.integers(n)] = False
    if not mask.any():
        mask[rng.integers(n)] = True
    return mask


def _predict_channel(
    score: Score,
    L: np.ndarray,
    y: np.ndarray,
    mean_full: np.ndarray,
    mask: np.ndarray,
    use_graph: bool,
    lam: float,
    eta: float,
    noise_var: Optional[float],
    fit_hyper: bool,
    graph_hyper: str = "marglik",
    rng: Optional[np.random.Generator] = None,
    noise_floor_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Held-out (pred, std) for one channel under one mean and one graph setting.

    ``graph_hyper`` selects how the graph hyperparameters are chosen when ``fit_hyper``:
    ``"marglik"`` (in-sample marginal likelihood) or ``"calib"`` (held-out calibration-split
    NLL, :func:`fit_laplacian_field_calib`).  ``noise_floor_frac > 0`` floors the EB
    ``noise_var`` at that fraction of the observed residual variance (guards against the
    degenerate noise_var -> 0 fits that spike held-out NLL).
    """
    held = ~mask
    if use_graph:
        if fit_hyper:
            if graph_hyper == "calib":
                field, hp = fit_laplacian_field_calib(L, y, mask=mask, mean=mean_full, rng=rng)
            else:
                floor = noise_floor_frac * float(np.var((y - mean_full)[mask]))
                field, hp = fit_laplacian_field(L, y, mask=mask, mean=mean_full,
                                                noise_floor=floor)
            nv = hp["noise_var"]
        else:
            Q = laplacian_precision(L, lam=lam, eta=eta)
            field = GraphGaussianField(Q, mean=mean_full)
            nv = noise_var if noise_var is not None else 0.05
        m, std = field.posterior(y, nv, mask=mask)
        # Posterior-predictive std of a held-out *observation* y = f + e: the latent
        # field variance diag(Sigma_y) collapses toward 0 at well-pinned nodes, so the
        # interval must include the observation noise nv (else NLL/coverage blow up).
        pred_std = np.sqrt(std[held] ** 2 + nv)
        return m[held], pred_std
    # no graph: predict the mean, with a homoscedastic std from the observed residual
    resid = (y - mean_full)[mask]
    s = float(np.std(resid)) if resid.size > 1 else 1.0
    return mean_full[held], np.full(int(held.sum()), max(s, 1e-6))


def impute_methods(
    score: Score,
    y: np.ndarray,
    means: Dict[str, np.ndarray],
    mask: np.ndarray,
    fit_hyper: bool = True,
    lam: float = 0.5,
    eta: float = 2.0,
    noise_var: Optional[float] = None,
    graph_variants: Optional[List[Tuple[object, bool, str]]] = None,
    rng: Optional[np.random.Generator] = None,
    noise_floor_frac: float = 0.0,
) -> Dict[Tuple[str, object], CellResult]:
    """Run every (mean source, graph variant) cell on one piece.

    ``y`` is (N,) or (N, k); ``means`` maps a name -> a full-length mean array of matching
    shape (the LM mean must be out-of-sample, fit on a separate split).  ``graph_variants`` is
    a list of ``(label, use_graph, graph_hyper)`` tuples; the default
    ``[(False, False, "marglik"), (True, True, "marglik")]`` reproduces the classic 3x2 table
    with boolean keys.  Pass e.g. ``(" calib", True, "calib")`` to add a calibration-split
    column.  Returns a dict keyed by ``(mean_name, label)`` of :class:`CellResult`
    (held-out ``y``/``pred``/``std`` concatenated over channels, with a ``channel`` index).
    """
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, k = Y.shape
    L = laplacian(build_adjacency(score))
    held = ~mask
    if graph_variants is None:
        graph_variants = [(False, False, "marglik"), (True, True, "marglik")]

    out: Dict[Tuple[str, object], CellResult] = {}
    for name, mean_arr in means.items():
        M = np.asarray(mean_arr, dtype=float)
        if M.ndim == 1:
            M = M[:, None]
        for label, use_graph, ghyper in graph_variants:
            yt, pr, sd, ch = [], [], [], []
            for c in range(k):
                pred, std = _predict_channel(
                    score, L, Y[:, c], M[:, c], mask, use_graph, lam, eta, noise_var,
                    fit_hyper, graph_hyper=ghyper, rng=rng,
                    noise_floor_frac=noise_floor_frac,
                )
                yt.append(Y[held, c]); pr.append(pred); sd.append(std)
                ch.append(np.full(pred.shape[0], c, dtype=int))
            out[(name, label)] = CellResult(
                np.concatenate(yt), np.concatenate(pr),
                np.concatenate(sd), np.concatenate(ch),
            )
    return out


def ridge_mean(score: Score, y: np.ndarray, mask: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Hand-built ridge-feature prior mean over all notes (fit on observed; per channel)."""
    Y = np.asarray(y, dtype=float)
    single = Y.ndim == 1
    if single:
        Y = Y[:, None]
    cols = [ridge_impute(score, Y[:, c], mask, l2=l2)[0] for c in range(Y.shape[1])]
    M = np.stack(cols, axis=1)
    return M[:, 0] if single else M


class MetricAccumulator:
    """Pool held-out :class:`CellResult`s across pieces, then report metrics per cell."""

    def __init__(self) -> None:
        self._cells: Dict[Tuple[str, bool], List[List[np.ndarray]]] = {}

    def add(self, cell_results: Dict[Tuple[str, bool], CellResult]) -> None:
        for key, cell in cell_results.items():
            self._cells.setdefault(key, [[], [], [], []])
            self._cells[key][0].append(np.asarray(cell.y))
            self._cells[key][1].append(np.asarray(cell.pred))
            self._cells[key][2].append(np.asarray(cell.std))
            # tolerate legacy 3-field results (no channel) by tagging channel 0
            ch = getattr(cell, "channel", None)
            self._cells[key][3].append(
                np.asarray(ch) if ch is not None else np.zeros(len(cell.y), dtype=int)
            )

    def _pooled(self, key) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        yts, prs, sds, chs = self._cells[key]
        return (np.concatenate(yts), np.concatenate(prs),
                np.concatenate(sds), np.concatenate(chs))

    def report(self, level: float = 0.9) -> Dict[Tuple[str, bool], Dict[str, float]]:
        """Metrics per (mean, graph) cell, pooled over all channels and pieces."""
        rep = {}
        for key in self._cells:
            yt, pr, sd, _ = self._pooled(key)
            rep[key] = _evaluate_metrics(yt, pr, sd, level=level)
        return rep

    def report_by_channel(
        self, channel_names: Sequence[str], level: float = 0.9
    ) -> Dict[Tuple[str, bool, str], Dict[str, float]]:
        """Metrics per (mean, graph, channel) — exposes which y-component is miscalibrated."""
        rep = {}
        for key in self._cells:
            yt, pr, sd, ch = self._pooled(key)
            for ci, cname in enumerate(channel_names):
                sel = ch == ci
                if sel.any():
                    rep[(key[0], key[1], cname)] = _evaluate_metrics(
                        yt[sel], pr[sel], sd[sel], level=level
                    )
        return rep


def format_report(rep: Dict[Tuple[str, bool], Dict[str, float]], level: float = 0.9) -> str:
    """Pretty 3x2 table: mean source x graph, with RMSE / NLL / coverage / cal-error."""
    cov_key = "coverage@%.2f" % level
    lines = [f"{'mean source':14s} {'graph':6s} {'RMSE':>8s} {'NLL':>8s} {'cov@.9':>8s} {'cal-err':>8s}"]
    order = ["zero", "ridge", "LM"]
    names = {k[0] for k in rep}
    for name in order + sorted(names - set(order)):
        for use_graph in (False, True):
            key = (name, use_graph)
            if key not in rep:
                continue
            m = rep[key]
            lines.append(
                f"{name:14s} {('on' if use_graph else 'off'):6s} "
                f"{m['rmse']:8.4f} {m['nll']:8.4f} {m[cov_key]:8.3f} {m['calibration_error']:8.3f}"
            )
    return "\n".join(lines)


def format_report_by_channel(
    rep: Dict[Tuple[str, bool, str], Dict[str, float]],
    channel_names: Sequence[str],
    level: float = 0.9,
) -> str:
    """One sub-table per channel (from :meth:`MetricAccumulator.report_by_channel`)."""
    cov_key = "coverage@%.2f" % level
    order = ["zero", "ridge", "LM"]
    names = {k[0] for k in rep}
    blocks = []
    for cname in channel_names:
        lines = [
            f"[{cname}]",
            f"{'mean source':14s} {'graph':6s} {'RMSE':>8s} {'NLL':>8s} {'cov@.9':>8s} {'cal-err':>8s}",
        ]
        for name in order + sorted(names - set(order)):
            for use_graph in (False, True):
                key = (name, use_graph, cname)
                if key not in rep:
                    continue
                m = rep[key]
                lines.append(
                    f"{name:14s} {('on' if use_graph else 'off'):6s} "
                    f"{m['rmse']:8.4f} {m['nll']:8.4f} {m[cov_key]:8.3f} {m['calibration_error']:8.3f}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
