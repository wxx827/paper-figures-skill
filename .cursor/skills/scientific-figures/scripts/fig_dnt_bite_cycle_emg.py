"""Four-channel masticatory EMG over consecutive chewing cycles.

All four muscle channels are drawn against one shared microvolt full scale, so
the working-side masseter really does look larger than its counterpart instead
of every lane being separately autoscaled into looking the same.  The jaw
opening trace shares the time axis and is what defines the phase boundaries:
the opening, closing and occlusal shading is generated from the same cycle
table that generates the bursts.  Every number quoted - integrated activity,
symmetry index, cycle duration, peak amplitude - is reduced from the plotted
arrays at run time.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.58, 4.56)
AUX_T = (7.00, 3.52, 3.12, 1.90)
AUX_B = (7.00, 0.86, 3.12, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR, BAND = "#d5d5d5", "#f5f5f5"
RED, BLUE, GREEN, GOLD, PURPLE = ("#b23a48", "#3b5b92", "#2e7d5b",
                                  "#c9a227", "#8e6bbf")

# (label, muscle, side, colour, gain relative to the right masseter)
CHANNELS = [("Masseter R", "Masseter", "R", BLUE, 1.00),
            ("Masseter L", "Masseter", "L", RED, 1.34),
            ("Temporalis R", "Temporalis", "R", GREEN, 0.76),
            ("Temporalis L", "Temporalis", "L", GOLD, 0.94)]
PHASES = [(0.00, 0.28, "opening", BLUE),
          (0.28, 0.70, "closing", GOLD),
          (0.70, 1.00, "occlusal", RED)]
N_CYCLE = 6
FS = 2000.0
BASE_UV = 320.0             # reference burst amplitude, microvolts
OPEN_MM = 17.5              # reference maximum jaw opening, millimetres
UV_BAR, MM_BAR = 200.0, 10.0

FOOT = ("Synthetic surface EMG envelopes, not a patient recording. The four "
        "muscle lanes share one microvolt full scale so their heights are "
        "directly comparable; the shaded strips mark the opening, closing "
        "and occlusal phases of each cycle and come from the same cycle "
        "table that shapes the bursts. Symmetry index is 100 x (left - "
        "right) / (left + right) on cycle-integrated activity, so zero means "
        "balanced.")


def build(rng):
    """Cycle table, four EMG envelopes and the jaw opening trace."""
    dur = rng.uniform(0.62, 0.80, N_CYCLE)
    start = 0.15 + np.concatenate([[0.0], np.cumsum(dur)[:-1]])
    total = 0.15 + dur.sum() + 0.18
    t = np.arange(0.0, total, 1.0 / FS)

    env = {ch[0]: np.zeros_like(t) for ch in CHANNELS}
    disp = np.zeros_like(t)
    amps = {ch[0]: np.zeros(N_CYCLE) for ch in CHANNELS}

    for k in range(N_CYCLE):
        m = (t >= start[k]) & (t < start[k] + dur[k])
        p = (t[m] - start[k]) / dur[k]
        for name, _, _, _, gain in CHANNELS:
            a = BASE_UV * gain * rng.uniform(0.82, 1.18)
            amps[name][k] = a
            burst = (np.exp(-((p - 0.72) / 0.145) ** 2)
                     + 0.14 * np.exp(-((p - 0.15) / 0.085) ** 2))
            ripple = 1.0 + 0.09 * np.sin(2.0 * np.pi * 31.0 * t[m]
                                         + 3.1 * k + gain)
            env[name][m] = a * burst * ripple
        d = OPEN_MM * rng.uniform(0.88, 1.12)
        q = np.clip(p / 0.72, 0.0, 1.0)
        disp[m] = d * 0.5 * (1.0 - np.cos(2.0 * np.pi * q)) * (p < 0.72)
    return t, env, disp, start, dur


def main():
    set_pub_style()
    rng = np.random.default_rng(821)
    t, env, disp, start, dur = build(rng)
    total = t[-1]

    names = [c[0] for c in CHANNELS]
    colors = {c[0]: c[3] for c in CHANNELS}
    full_uv = max(env[n].max() for n in names)      # one scale for all lanes
    full_mm = disp.max()

    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) stacked channels ------------------------------------------------
    ax = axes_inches(fig, MAIN)
    bases = {n: 4.0 - i for i, n in enumerate(names)}
    disp_base = -1.15
    lane = 0.86

    for k in range(N_CYCLE):
        if k % 2 == 0:
            ax.axvspan(start[k], start[k] + dur[k], color=BAND, lw=0,
                       zorder=0)
        ax.text(start[k] + dur[k] / 2.0, -1.60, f"cycle {k + 1}",
                ha="center", va="top", fontsize=6.0, color=MUTED)
    for p0, p1, lab, col in PHASES:
        x0 = start[0] + p0 * dur[0]
        x1 = start[0] + p1 * dur[0]
        ax.axvspan(x0, x1, color=col, alpha=0.15, lw=0, zorder=1)
        ax.text(0.5 * (x0 + x1), 5.08, lab, ha="center", va="bottom",
                fontsize=6.2, color=col, rotation=90)

    for n in names:
        b = bases[n]
        y = b + lane * env[n] / full_uv
        ax.fill_between(t, b, y, color=colors[n], alpha=0.30, lw=0, zorder=3)
        ax.plot(t, y, color=colors[n], lw=0.9, zorder=4)
        ax.axhline(b, color=HAIR, lw=0.6, zorder=2)
        ax.text(-0.02, b + 0.07, n, ha="left", va="bottom", fontsize=6.6,
                color=colors[n], zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.85))
    yd = disp_base + 1.05 * disp / full_mm
    ax.fill_between(t, disp_base, yd, color=PURPLE, alpha=0.22, lw=0,
                    zorder=3)
    ax.plot(t, yd, color=PURPLE, lw=1.1, zorder=4)
    ax.axhline(disp_base, color=HAIR, lw=0.6, zorder=2)
    ax.text(-0.02, disp_base + 0.07, "Jaw opening", ha="left", va="bottom",
            fontsize=6.6, color=PURPLE, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                      alpha=0.85))

    xb = total * 1.045
    h_uv = lane * UV_BAR / full_uv
    ax.plot([xb, xb], [bases[names[0]], bases[names[0]] + h_uv], color=RULE,
            lw=1.3)
    ax.text(xb + 0.045, bases[names[0]] + h_uv / 2.0,
            f"{UV_BAR:.0f} " + r"$\mu V$", ha="left", va="center",
            fontsize=6.3, color=RULE, rotation=90)
    h_mm = 1.05 * MM_BAR / full_mm
    ax.plot([xb, xb], [disp_base, disp_base + h_mm], color=PURPLE, lw=1.3)
    ax.text(xb + 0.045, disp_base + h_mm / 2.0, f"{MM_BAR:.0f} mm",
            ha="left", va="center", fontsize=6.3, color=PURPLE, rotation=90)

    ax.set_xlim(-0.06, total * 1.13)
    ax.set_ylim(-1.95, 5.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)", fontsize=8.0)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Four-channel envelopes over consecutive chewing cycles",
                 fontsize=8.8, fontweight="bold", color=INK, pad=4)

    # ---- (b) integrated activity and symmetry --------------------------------
    ax_b = axes_inches(fig, AUX_T)
    integ = {n: np.array([np.trapezoid(
        env[n][(t >= start[k]) & (t < start[k] + dur[k])], dx=1.0 / FS)
        for k in range(N_CYCLE)]) for n in names}
    muscles = ["Masseter", "Temporalis"]
    xg = np.arange(len(muscles), dtype=float)
    for j, side in enumerate(("R", "L")):
        vals = np.array([integ[f"{m} {side}"].mean() for m in muscles])
        errs = np.array([integ[f"{m} {side}"].std(ddof=1) /
                         np.sqrt(N_CYCLE) for m in muscles])
        cols = [colors[f"{m} {side}"] for m in muscles]
        ax_b.bar(xg + (j - 0.5) * 0.34, vals, yerr=errs, width=0.32,
                 color=cols, edgecolor="white", linewidth=0.5,
                 error_kw=dict(ecolor=RULE, elinewidth=0.7, capsize=2.0))
        for xi, v, e in zip(xg + (j - 0.5) * 0.34, vals, errs):
            ax_b.text(xi, v + e + 1.2, side, ha="center", va="bottom",
                      fontsize=6.2, color=SUB)
    top = max(integ[n].mean() for n in names)
    for i, m in enumerate(muscles):
        r, l = integ[f"{m} R"].mean(), integ[f"{m} L"].mean()
        ax_b.text(xg[i], top * 1.28, f"SI {100.0 * (l - r) / (l + r):+.1f} %",
                  ha="center", va="bottom", fontsize=6.5, color=RED)
    ax_b.set_xticks(xg)
    ax_b.set_xticklabels(muscles, fontsize=7.0)
    ax_b.set_ylabel(r"integrated activity ($\mu V \cdot s$)", fontsize=8.0)
    ax_b.set_ylim(0, top * 1.48)
    ax_b.tick_params(labelsize=6.6)
    ax_b.grid(axis="y", color=HAIR, lw=0.6)
    ax_b.set_axisbelow(True)
    ax_b.set_title(f"Cycle-integrated activity (n={N_CYCLE} cycles)",
                   fontsize=8.5, fontweight="bold", color=INK, pad=3)

    # ---- (c) cycle duration against peak amplitude ---------------------------
    ax_c = axes_inches(fig, AUX_B)
    dur_ms = dur * 1000.0
    for n in names:
        pk = np.array([env[n][(t >= start[k]) & (t < start[k] + dur[k])].max()
                       for k in range(N_CYCLE)])
        ax_c.plot(dur_ms, pk, ls="none", marker="o", ms=4.0,
                  color=colors[n], mec="white", mew=0.5, label=n)
    ax_c.set_xlabel("cycle duration (ms)", fontsize=8.0)
    ax_c.set_ylabel(r"peak envelope ($\mu V$)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(color=HAIR, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.set_ylim(0, full_uv * 1.30)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                frameon=False, fontsize=6.3, handletextpad=0.3,
                columnspacing=1.0, labelspacing=0.25)
    ax_c.set_title("Cycle-by-cycle variability", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    si_m = 100.0 * (integ["Masseter L"].mean() - integ["Masseter R"].mean()) \
        / (integ["Masseter L"].mean() + integ["Masseter R"].mean())
    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.52, dy=0.10)
    savefig(fig, "821_dnt_bite_cycle_emg.png")


if __name__ == "__main__":
    main()
