#!/usr/bin/env python
"""Two demonstration figures for the motif result, read from the evaluation
records only. No model is fitted here.

  motif_demo_example.png   one hidden statement of a motif in one held-out
                           performance: the held duration of each note, the
                           truth against the time-only prediction and the
                           motif-edge prediction (shared form), with the
                           other statements of the motif as observed context
  motif_demo_movements.png per movement: the articulation error of hidden
                           returns with and without motif edges on the
                           held-out performances, and the development effect
                           against the held-out effect

Inputs: evidence/motif_heldout/ (the pre-specified test, evaluated once) and
results/motif_pilot2/ (the development study, four seeds). The per-movement
effects are recomputed exactly as in scripts/eval_motif_pilot_dev.py
(stage_report) and asserted against the recorded summary numbers.

The example statement is chosen by a stated rule, not by eye: among all
fully hidden statements of at least five notes in the held-out cells
(every performance and seed), the one whose articulation RMSE change is
closest to the upper quartile of that distribution. So one statement in
four improves at least this much, and the caption says so.

    PYTHONPATH=src python scripts/make_motif_demo_figs.py
"""
from __future__ import annotations

import glob
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

INK, MUTED, GRID = "#1A1A1A", "#6B7280", "#D1D5DB"
BLUE, AMBER = "#0072B2", "#E69F00"
OUT = "docs/thesis/figures/"
HELD, DEV = "evidence/motif_heldout", "results/motif_pilot2"
MOTIF, BASE, ART = "time_motif", "time", 1     # shared form, time-only graph, articulation channel

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 200,
})


def load(d):
    pieces = pickle.load(open(f"{d}/pieces.pkl", "rb"))
    cells = {}
    for f in glob.glob(f"{d}/cells.shard*.pkl"):
        cells.update(pickle.load(open(f, "rb")))
    return pieces, cells


def per_movement(pieces, cells, seeds=4):
    """Movement -> (RMSE time-only, RMSE motif), hidden motif notes, occurrence
    scheme: MSE averaged over seeds, then over performances, then the root."""
    per = {}
    for arm in (BASE, MOTIF):
        g = {}
        for pi, p in enumerate(pieces):
            ms = []
            for s in range(seeds):
                r = cells.get((pi, "occ", s, arm))
                if r is None:
                    continue
                yt, pr, sd, ch = r["cell"]
                sel = (ch == ART) & np.tile(r["motif_sel"], 3)
                if sel.sum() < 3:
                    continue
                ms.append(np.mean((yt[sel] - pr[sel]) ** 2))
            if ms:
                g.setdefault(p.get("group", p["folder"]), []).append(np.mean(ms))
        per[arm] = {k: np.sqrt(np.mean(v)) for k, v in g.items()}
    common = sorted(set(per[BASE]) & set(per[MOTIF]))
    return {k: (per[BASE][k], per[MOTIF][k]) for k in common}


def pick_statement(pieces, cells):
    rows = []
    for (pi, sc, s, arm), r in cells.items():
        if sc != "occ" or arm != BASE or (pi, sc, s, MOTIF) not in cells:
            continue
        r1 = cells[(pi, sc, s, MOTIF)]
        hi = r["held_idx"]; n = len(hi)
        yt, pr0, _, _ = r["cell"]; _, pr1, _, _ = r1["cell"]
        pos = {j: k for k, j in enumerate(hi)}
        for oi, o in enumerate(pieces[pi]["occs"]):
            notes = np.asarray(o["notes"])
            if len(notes) < 5 or not all(j in pos for j in notes):
                continue
            k = np.array([pos[j] for j in notes])
            if not r["motif_sel"][k].all():
                continue
            t = yt[n + k]
            e0 = np.sqrt(np.mean((t - pr0[n + k]) ** 2))
            e1 = np.sqrt(np.mean((t - pr1[n + k]) ** 2))
            rows.append((e1 - e0, pi, s, oi))
    d = np.array([r[0] for r in rows])
    q25 = np.percentile(d, 25)
    best = min(rows, key=lambda r: abs(r[0] - q25))
    return best, d


def fig_example(pieces, cells, pick, dist):
    _, pi, s, oi = pick
    p = pieces[pi]
    r0, r1 = cells[(pi, "occ", s, BASE)], cells[(pi, "occ", s, MOTIF)]
    hi = r0["held_idx"]; n = len(hi)
    pos = {j: k for k, j in enumerate(hi)}
    o = p["occs"][oi]
    notes = np.asarray(o["notes"]); k = np.array([pos[j] for j in notes])
    yt, pr0, sd0, _ = r0["cell"]; _, pr1, sd1, _ = r1["cell"]
    truth, m0, s0, m1, s1 = yt[n + k], pr0[n + k], sd0[n + k], pr1[n + k], sd1[n + k]
    assert np.allclose(truth, p["y"][notes, ART])
    observed = np.ones(len(p["y"]), dtype=bool); observed[hi] = False

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    z = 1.6449
    for o2 in p["occs"]:
        if o2["type"] != o["type"] or o2 is o:
            continue
        nn = np.asarray(o2["notes"]); keep = observed[nn]
        if keep.sum() < 2:
            continue
        ax.plot(o2["u"][keep], np.exp(p["y"][nn[keep], ART]), color=GRID, lw=1.0,
                marker="o", ms=2.5, zorder=1)
    ax.plot([], [], color=GRID, lw=1.0, marker="o", ms=2.5,
            label="other statements of the motif, observed")
    u = o["u"]; du = 0.012
    ax.errorbar(u - du, np.exp(m0), yerr=[np.exp(m0) - np.exp(m0 - z * s0), np.exp(m0 + z * s0) - np.exp(m0)],
                fmt="s", color=AMBER, ms=5, lw=1.4, capsize=2,
                label=f"predicted, time window only (RMSE {np.sqrt(np.mean((truth - m0) ** 2)):.2f})", zorder=3)
    ax.errorbar(u + du, np.exp(m1), yerr=[np.exp(m1) - np.exp(m1 - z * s1), np.exp(m1 + z * s1) - np.exp(m1)],
                fmt="o", color=BLUE, ms=5, lw=1.4, capsize=2,
                label=f"predicted, with motif edges (RMSE {np.sqrt(np.mean((truth - m1) ** 2)):.2f})", zorder=3)
    ax.plot(u, np.exp(truth), color=INK, lw=1.6, marker="D", ms=5.5, zorder=4,
            label="what the pianist played (hidden from the model)")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls="--")
    ax.text(1.0, 1.0, " as written", color=MUTED, fontsize=8, va="center",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.25", "0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("position inside the statement (0 = first note, 1 = end of the statement)")
    ax.set_ylabel("held duration / written duration")
    ax.set_xlim(-0.05, 1.02)
    ax.set_title("Beethoven, Piano Sonata No. 17, first movement: one statement of a motif hidden, "
                 "a held-out performance", loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.3)
    frac = float(np.mean(dist < 0))
    fig.text(0.01, -0.12,
             f"Error bars: 90% predictive intervals. Chosen by rule: of {len(dist)} fully hidden statements (at least five notes,\n"
             f"all held-out performances and seeds), the one at the upper quartile of the improvement. One statement in four\n"
             f"improves at least this much, {100 * frac:.0f}% improve at all, the rest do not.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(OUT + "motif_demo_example.png", bbox_inches="tight")
    plt.close(fig)


def fig_movements(H, D):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42})
    names = sorted(H, key=lambda k: H[k][1] - H[k][0])
    y = np.arange(len(names))
    for yi, nm in zip(y, names):
        e0, e1 = H[nm]
        axa.plot([e0, e1], [yi, yi], color=GRID, lw=1.6, zorder=1)
        axa.scatter([e0], [yi], color=AMBER, marker="s", s=22, zorder=2)
        axa.scatter([e1], [yi], color=BLUE, s=24, zorder=3)
    axa.scatter([], [], color=AMBER, marker="s", s=22, label="time window only")
    axa.scatter([], [], color=BLUE, s=24, label="with motif edges")
    axa.set_yticks(y)
    axa.set_yticklabels([nm.split("_")[0].replace("-1", "") for nm in names], fontsize=8)
    axa.set_ylabel("sonata (first movement)")
    axa.set_xlabel("articulation error of hidden returns (RMSE)")
    nbetter = sum(H[k][1] < H[k][0] for k in H)
    axa.set_title(f"A   held-out performances: {nbetter} of {len(H)} movements improve", loc="left")
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)

    k = sorted(set(H) & set(D))
    a = np.array([H[x][1] - H[x][0] for x in k]); b = np.array([D[x][1] - D[x][0] for x in k])
    r = float(np.corrcoef(a, b)[0, 1])
    lim = 1.08 * max(np.abs(a).max(), np.abs(b).max())
    axb.axhline(0, color=MUTED, lw=0.7); axb.axvline(0, color=MUTED, lw=0.7)
    axb.plot([-lim, lim], [-lim, lim], color=GRID, lw=1.0, ls="--", zorder=0)
    axb.scatter(b, a, color=BLUE, s=26, zorder=3)
    axb.set_xlim(-lim, lim); axb.set_ylim(-lim, lim); axb.set_aspect("equal")
    axb.set_xlabel("development performance:\nchange in error with motif edges")
    axb.set_ylabel("held-out performances:\nchange in error with motif edges")
    both = int(((a < 0) & (b < 0)).sum())
    axb.text(-0.96 * lim, -0.92 * lim, f"better in both: {both}", fontsize=8.5, color=INK)
    axb.set_title(f"B   the same movements gain in both studies (r = {r:.2f})", loc="left")
    fig.text(0.01, -0.06,
             "Shared form against the time-only graph, hidden motif notes, whole occurrences hidden, four seeds. "
             "Held-out: 58 further performances of 18 movements, averaged within a movement.\n"
             "Mean change, held out: "
             f"{np.mean([H[x][1] - H[x][0] for x in H]):+.4f} [−0.0494, −0.0109]. ".replace("-", "−")
             + 
             "Development: one performance of 21 movements, 18 of them shown in B.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(OUT + "motif_demo_movements.png", bbox_inches="tight")
    plt.close(fig)
    return a, b, r, both


def main():
    hp, hc = load(HELD)
    dp, dc = load(DEV)
    H, D = per_movement(hp, hc), per_movement(dp, dc)
    eff = json.load(open(f"{HELD}/effects.json"))["effects"][f"occ|motif|{MOTIF}|log r"]
    dh = np.mean([H[x][1] - H[x][0] for x in H])
    assert abs(dh - eff["drmse"][0]) < 1e-9, (dh, eff["drmse"])
    a, b, r, both = fig_movements(H, D)
    assert len(H) == 18 and sum(H[k][1] < H[k][0] for k in H) == 15
    assert abs(r - 0.88) < 0.005 and both == 12
    pick, dist = pick_statement(hp, hc)
    fig_example(hp, hc, pick, dist)
    print(f"held-out mean {dh:+.4f}, r = {r:.3f}, both {both}; example {pick}, "
          f"{len(dist)} statements, {np.mean(dist < 0):.2f} improve")


if __name__ == "__main__":
    main()
