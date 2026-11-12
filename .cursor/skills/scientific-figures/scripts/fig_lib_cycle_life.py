"""Cycle-life fade of an NMC811/graphite 21700 cell at three temperatures.

(a) Discharge capacity against cycle number.  Markers are measured
    checkpoints; lines are the same arrays.  End-of-life at 80 % of the
    first-cycle capacity is the dashed rule, and the cycle that crosses it
    is read off each curve.
(b) Coulombic efficiency of the same three series.
(c) Fade rate from a linear fit of cycles 200-500, in % per 100 cycles.

Illustrative synthetic ageing.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
INK = "#1a1a1a"
MUTED = "#777777"
SEED = 1217
Q0 = 4.85

CONDS = [
    dict(name="25 $^\\circ$C", k1=0.0040, k2=2.0e-5, ce0=0.9994, col=BLUE),
    dict(name="45 $^\\circ$C", k1=0.0072, k2=8.0e-5, ce0=0.9987, col=GREEN),
    dict(name="60 $^\\circ$C", k1=0.0120, k2=2.4e-4, ce0=0.9976, col=RED),
]


def capacity(n, k1, k2):
    return Q0 * (1.0 - k1 * np.sqrt(n) - k2 * n)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    n = np.arange(1, 801)
    series = []
    for cond in CONDS:
        q = capacity(n, cond["k1"], cond["k2"])
        q = q + rng.normal(0, 0.012, n.size)
        q[0] = Q0
        ce = cond["ce0"] - 4e-7 * n + rng.normal(0, 1.6e-4, n.size)
        ce = np.clip(ce, 0.992, 1.0)
        series.append((q, ce))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    eol = 0.80 * Q0
    ax.axhline(eol, color=MUTED, lw=0.8, ls=(0, (3.2, 1.8)), zorder=1)
    ax.text(12, eol + 0.04, "80 % EOL", fontsize=6.6, color=MUTED, va="bottom")
    eol_n = []
    for cond, (q, ce) in zip(CONDS, series):
        ax.plot(n, q, color=cond["col"], lw=1.5, zorder=4, label=cond["name"])
        chk = n[::25]
        ax.plot(chk, q[chk - 1], ls="none", marker="o", ms=3.4, mfc="white",
                mec=cond["col"], mew=0.7, zorder=5)
        hit = np.where(q <= eol)[0]
        eol_n.append(int(n[hit[0]]) if hit.size else None)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (A h)")
    ax.set_xlim(0, 800)
    ax.set_ylim(3.4, 5.05)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("1C/1C, 3.0-4.2 V", fontsize=9.0, fontweight="bold", pad=6)
    lines = []
    for cond, n_eol in zip(CONDS, eol_n):
        lines.append("{} EOL = {} cycles".format(
            cond["name"].replace(" $^\\circ$C", " C"),
            n_eol if n_eol else ">800"))
    ax.text(0.97, 0.96, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    for cond, (q, ce) in zip(CONDS, series):
        axb.plot(n, 100 * ce, color=cond["col"], lw=0.9, alpha=0.85)
    axb.set_xlabel("Cycle number", fontsize=8, labelpad=1.5)
    axb.set_ylabel("Coulombic efficiency (%)", fontsize=8, labelpad=1.5)
    axb.set_xlim(0, 800)
    axb.set_ylim(99.55, 100.02)
    axb.tick_params(labelsize=6.5)
    axb.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("Coulombic efficiency", fontsize=8.5, fontweight="bold",
                  pad=4)

    fade = []
    for q in (s[0] for s in series):
        sl = (n >= 200) & (n <= 500)
        slope = np.polyfit(n[sl], 100 * q[sl] / Q0, 1)[0]
        fade.append(-slope * 100.0)
    x = np.arange(len(CONDS))
    axc.bar(x, fade, color=[c["col"] for c in CONDS], width=0.62, zorder=3)
    axc.set_ylim(0.0, max(fade) * 1.22)
    axc.set_xticks(x)
    axc.set_xticklabels([c["name"] for c in CONDS], fontsize=7.0)
    axc.set_ylabel("Fade (% / 100 cycles)", fontsize=8, labelpad=1.5)
    axc.tick_params(labelsize=6.5)
    axc.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Linear fade, cycles 200-500", fontsize=8.5,
                  fontweight="bold", pad=4)
    for i, f in enumerate(fade):
        axc.text(i, f + 0.04, "{:.2f}".format(f), ha="center", va="bottom",
                 fontsize=6.6, color=INK)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1217_lib_cycle_life.png")


if __name__ == "__main__":
    main()
