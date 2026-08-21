import pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from score_bundle.phase2.intonation import cents_from_f0, fit_vibrato_note_gated
from score_bundle.phase2.urmp import read_notes_annotation
from eval_phase2_real import dev_unique_tracks

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, ORANGE, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
TRACK = (1, 1)
CONF_Q = 0.2

d = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))[TRACK]
f0c = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))[TRACK]
tr = next(t for p, t in dev_unique_tracks() if (p.index, t.number) == TRACK)
notes = read_notes_annotation(tr.notes)

ok = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
floor = np.quantile(f0c["prob"][ok], CONF_Q)
ok = ok & (f0c["prob"] >= floor)

def frames(i):
    sel = ok & (f0c["t"] >= notes["onset"][i]) & (f0c["t"] < notes["onset"][i] + notes["duration"][i])
    return f0c["t"][sel], f0c["f0"][sel]

# pick: A = long identifiable note with clear delay; B = short unidentifiable
cand_a = [i for i in range(len(d["ident"])) if d["ident"][i]
          and np.isfinite(d["dvib"][i]) and d["dvib"][i] > 0.08
          and notes["duration"][i] > 0.8 and np.isfinite(d["tau"][i])
          and np.isfinite(d["est"][i, 1]) and np.exp(d["est"][i, 1]) > 12.0]
cand_b = [i for i in range(len(d["ident"])) if not d["ident"][i]
          and np.isfinite(d["est"][i, 0]) and notes["duration"][i] < 0.35
          and np.isfinite(d["tau"][i])]
A, B = cand_a[2], cand_b[2]

fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.4), dpi=200,
                         width_ratios=[3, 2])
for row, (i, tag) in enumerate([(A, "a long note: every cell observed"),
                                (B, "a short note: vibrato cells missing")]):
    t, f0 = frames(i)
    cents = cents_from_f0(f0, 440.0, float(d["midi"][i] - 69))
    tt = t - notes["onset"][i]
    ax = axes[row, 0]
    ax.plot(tt * 1e3, cents, ".", ms=4, color=MUTED,
            label="voiced, confidence-kept frames")
    fit = fit_vibrato_note_gated(tt, cents)
    if fit["vibrato_identifiable"]:
        g = np.linspace(0, tt.max(), 400)
        gate = g >= fit["delta"]
        curve = fit["c"] + np.where(
            gate, fit["gamma"] * np.sin(2 * np.pi * fit["f"] * (g - fit["delta"])), 0.0)
        ax.plot(g * 1e3, curve, color=BLUE, lw=1.6,
                label="gated NLLS fit (eq. 3.33)")
        ax.axvline(fit["delta"] * 1e3, color=PURPLE, lw=1.2, ls="--")
        ax.text(fit["delta"] * 1e3 + 8, ax.get_ylim()[0] + 2,
                r"$\delta^{\mathrm{vib}}$", color=PURPLE, fontsize=10)
    ax.axhline(fit["c"], color=VERM, lw=1.0, ls=":",
               label="vibrato-free centre $c_i$")
    ax.set_ylabel("cents vs written pitch", fontsize=9, color=INK)
    ax.set_title(f"note {i} ({'%.2f' % notes['duration'][i]} s) — {tag}",
                 fontsize=10, color=INK, loc="left", pad=30)
    if row == 1:
        ax.set_xlabel("time from note onset (ms)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left",
              bbox_to_anchor=(0.0, 1.0))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)

    # record card
    axc = axes[row, 1]
    axc.axis("off")
    g_lin = np.exp(d["est"][i, 1]) if np.isfinite(d["est"][i, 1]) else np.nan
    f_lin = np.exp(d["est"][i, 2]) if np.isfinite(d["est"][i, 2]) else np.nan
    g_sd = g_lin * np.sqrt(d["var"][i, 1]) if np.isfinite(g_lin) else None
    f_sd = f_lin * np.sqrt(d["var"][i, 2]) if np.isfinite(f_lin) else None
    vals = [
        (r"$\tau$ (ms)", d["tau"][i] * 1e3, np.sqrt(d["var_tau"][i]) * 1e3, ORANGE, True),
        (r"$\ell$ (log RMS, ctr.)", d["ell"][i], np.sqrt(d["var_ell"][i]), GREEN, True),
        (r"$c$ (cents)", d["est"][i, 0], np.sqrt(d["var"][i, 0]), VERM, True),
        (r"$\gamma$ (cents)", g_lin, g_sd, BLUE, False),
        (r"$f^{\mathrm{vib}}$ (Hz)", f_lin, f_sd, BLUE, False),
        (r"$\delta^{\mathrm{vib}}$ (ms)", d["dvib"][i] * 1e3 if np.isfinite(d["dvib"][i]) else np.nan,
         np.sqrt(d["var_dvib"][i]) * 1e3 if np.isfinite(d["var_dvib"][i]) else None, PURPLE, True),
    ]
    axc.text(0.02, 0.97, "the note's six-cell record", fontsize=10,
             color=INK, va="top", weight="bold")
    y = 0.82
    for name, v, s_, col, signed in vals:
        if np.isfinite(v):
            fmt = "{:+.2f}" if signed else "{:.2f}"
            txt = fmt.format(v) + (f"  ±{s_:.2f}" if s_ is not None and np.isfinite(s_) else "")
            axc.text(0.06, y, name, fontsize=9.5, color=INK)
            axc.text(0.55, y, txt, fontsize=9.5, color=col, family="monospace")
            axc.text(0.02, y, "●", fontsize=8, color=col)
        else:
            axc.text(0.06, y, name, fontsize=9.5, color=MUTED)
            axc.text(0.55, y, "MISSING cell", fontsize=9.5, color=VERM,
                     style="italic")
            axc.text(0.02, y, "○", fontsize=8, color=VERM)
        y -= 0.135
fig.tight_layout()
out = "docs/thesis/figures/phase2_datapoint_dev.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| notes", A, B)
