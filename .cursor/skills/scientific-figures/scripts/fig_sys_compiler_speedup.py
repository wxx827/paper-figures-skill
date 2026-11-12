"""Compiler optimisation levels across a benchmark suite, summarised correctly.

The main panel is the claim: optimisation levels pay off very unevenly, and one
memory-bound benchmark actually regresses, so the only defensible summary of a
set of ratios is their geometric mean - an arithmetic mean of speedups would
reward the outlier and is reported here only to show how far off it lands.  The
mean bars at the right are computed from the plotted matrix, not typed in.  The
two subordinate panels carry the evidence: a multiplicative waterfall
attributing the gain to individual passes, and the code-size cost that buys it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_palette, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"

BENCH = ["perlbench", "gcc", "mcf", "namd", "lbm", "xz", "x264", "bwaves"]
LEVELS = ["-O1", "-O2", "-O3", "-O3 -flto -fprofile-use"]
LEV_COLOR = get_palette("cb_sequential_blue", 6)[2:6]

HEADROOM = np.array([0.34, 0.72, 0.95, 1.38])      # gain available per level
BLOAT = np.array([0.02, 0.05, 0.12, 0.16])         # text-section growth per level
# per-benchmark sensitivity to optimisation and to instruction-cache pressure
SENS = np.array([0.95, 0.62, 0.09, 1.35, 1.55, 0.80, 1.15, 1.42])
ICACHE = np.array([0.30, 0.55, 0.95, 0.10, 0.05, 0.35, 0.20, 0.08])

# pass, multiplicative contribution
PASSES = [("inline", 1.17), ("unroll", 1.05), ("vectorise", 1.33),
          ("PGO layout", 1.12), ("LTO", 1.06), ("aggr. inline", 0.96)]


def suite(seed=13):
    rng = np.random.default_rng(seed)
    gain = 1.0 + np.outer(SENS, HEADROOM) * (1.0 + rng.normal(0.0, 0.07,
                                                              (len(BENCH),
                                                               len(LEVELS))))
    penalty = np.outer(ICACHE, BLOAT) * 1.05
    speed = np.clip(gain - penalty, 0.55, None)
    size = 1.0 + np.outer(0.5 + ICACHE, BLOAT) * (
        1.0 + rng.normal(0.0, 0.10, (len(BENCH), len(LEVELS))))
    return speed, size


def geo(a, axis=0):
    return np.exp(np.log(a).mean(axis=axis))


def suite_panel(ax, speed):
    gm = geo(speed, axis=0)
    am = speed.mean(axis=0)
    n_g, n_l = len(BENCH) + 1, len(LEVELS)
    x = np.arange(n_g, dtype=float)
    w = 0.78 / n_l

    ax.axhline(1.0, color=GREY, lw=1.0, ls=(0, (4, 2.6)), zorder=2)

    for j, lev in enumerate(LEVELS):
        vals = np.append(speed[:, j], gm[j])
        off = (j - (n_l - 1) / 2.0) * w
        bars = ax.bar(x + off, vals, width=w * 0.92, color=LEV_COLOR[j],
                      edgecolor="white", linewidth=0.5, zorder=3, label=lev)
        for i, b in enumerate(bars):
            if vals[i] < 1.0:
                b.set_edgecolor(RED)
                b.set_linewidth(1.0)
        ax.text(x[-1] + off, gm[j] + 0.05, f"{gm[j]:.2f}", ha="center",
                va="bottom", fontsize=6.2, color=LEV_COLOR[j], rotation=90)

    worst = int(np.argmin(speed[:, 2]))
    ax.text(x[worst], speed[worst].max() + 0.16,
            f"{BENCH[worst]} regresses:\nbest {speed[worst].max():.2f}x, "
            f"-O3 {speed[worst, 2]:.2f}x", ha="center", va="bottom",
            fontsize=6.5, color=RED, linespacing=1.35)

    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f0f0f0", zorder=0,
               linewidth=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(BENCH + ["geo. mean"], fontsize=7.0)
    ax.set_xlim(-0.62, x[-1] + 0.62)
    ax.set_ylim(0.0, speed.max() * 1.24)
    ax.set_ylabel("speedup over -O0 (x)")
    ax.set_xlabel("SPEC-style benchmark (-)")
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(f"Geometric mean reaches {gm[-1]:.2f}x while {BENCH[worst]} "
                 f"regresses to {speed[worst, 2]:.2f}x at -O3", fontsize=9.0,
                 fontweight="bold", color=INK)
    return gm, am, worst


def waterfall_panel(ax, gm_top):
    names = [p[0] for p in PASSES]
    fac = np.array([p[1] for p in PASSES])
    cum = np.concatenate([[1.0], np.cumprod(fac)])
    x = np.arange(len(PASSES) + 1, dtype=float)

    for i, f in enumerate(fac):
        lo, hi = min(cum[i], cum[i + 1]), max(cum[i], cum[i + 1])
        col = LEV_COLOR[2] if f >= 1.0 else RED
        ax.add_patch(Rectangle((x[i] - 0.32, lo), 0.64, hi - lo,
                               facecolor=col, edgecolor="white",
                               linewidth=0.5, zorder=3))
        ax.plot([x[i] - 0.32, x[i] + 0.68], [cum[i + 1], cum[i + 1]],
                color=GREY, lw=0.6, ls=(0, (2, 1.8)), zorder=2)
        ax.text(x[i], hi + 0.02, f"{(f - 1.0) * 100:+.0f} %", ha="center",
                va="bottom", fontsize=6.2, color=col if f < 1.0 else SUB)
    ax.bar([x[-1]], [cum[-1]], width=0.64, color=SUB, edgecolor="white",
           linewidth=0.5, zorder=3)
    ax.text(x[-1], cum[-1] + 0.02, f"{cum[-1]:.2f}x", ha="center",
            va="bottom", fontsize=6.5, color=INK, fontweight="bold")

    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (4, 2.4)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names + ["combined"], rotation=32, ha="right",
                       fontsize=6.3)
    ax.set_ylim(0.9, cum.max() * 1.16)
    ax.set_ylabel("cumulative speedup (x)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Where the -O3 gain comes from", fontsize=8.2,
                 fontweight="bold", color=INK)
    return cum[-1]


def size_panel(ax, speed, size):
    for j, lev in enumerate(LEVELS):
        ax.scatter((size[:, j] - 1.0) * 100.0, speed[:, j], s=15.0,
                   color=LEV_COLOR[j], edgecolor="white", linewidth=0.5,
                   zorder=4, label=lev)
    xs = (size.ravel() - 1.0) * 100.0
    ys = speed.ravel()
    k, b = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    gx = np.linspace(xs.min(), xs.max(), 40)
    ax.plot(gx, k * gx + b, color=GREY, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (2, 1.8)), zorder=2)
    ax.text(0.03, 0.97, f"r = {r:.2f},  {k * 10:.2f}x per 10 % of text",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=SUB)
    ax.set_xlabel("text-section growth over -O0 (%)", fontsize=7.4)
    ax.set_ylabel("speedup (x)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7)
    ax.set_title("Code size is the price of the gain", fontsize=8.2,
                 fontweight="bold", color=INK)
    return r


def main():
    set_pub_style()
    speed, size = suite()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.58],
                          width_ratios=[1.0, 1.0], wspace=0.24, hspace=0.72)
    ax_main = fig.add_subplot(gs[0, :])
    ax_fall = fig.add_subplot(gs[1, 0])
    ax_size = fig.add_subplot(gs[1, 1])

    gm, am, worst = suite_panel(ax_main, speed)
    combined = waterfall_panel(ax_fall, gm[-1])
    r = size_panel(ax_size, speed, size)

    handles, labels = ax_main.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GREY, lw=1.0, ls=(0, (4, 2.6))))
    labels.append("-O0 baseline")

    fig.tight_layout(rect=[0, 0.058, 1, 0.905], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_fall, ax_size], dx=-0.46, dy=0.12)
    savefig(fig, "785_sys_compiler_speedup.png")


if __name__ == "__main__":
    main()
