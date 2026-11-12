"""Dry-matter partitioning in a greenhouse tomato crop, closed by construction.

A logistic total dry weight is grown day by day and each day's increment is
split between root, stem, leaf and fruit by partitioning coefficients that are
normalised to one before they are used.  Standing organ weights are therefore
the running sums of those increments, the percentage areas in the main panel
are those weights divided by the total, and they close to 100 % at every one
of the plotted days rather than approximately.  The harvest index in the lower
side panel is the fruit share of the same standing weights, so the figure has
exactly one source of truth.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
GOLD = "#c9a227"

SEED = 1034
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

NDAY = 121                     # days after transplanting
W_MAX = 520.0                  # asymptotic total dry weight, g per plant
W_RATE, W_MID = 0.062, 62.0
ANTHESIS = 34.0                # first truss anthesis, d
HI_MARK = 40.0                 # harvest-index level whose date is solved for

ORGANS = [("Root", "#8a6b4f"), ("Stem", "#9ccfae"), ("Leaf", "#2e7d5b"),
          ("Fruit", "#c0603f")]
HARVESTS = [15, 35, 55, 75, 95, 118]

STAGES = [("Establishment", 0, 24, "#f3f6f2"), ("Vegetative", 24, 48,
                                                "#e8f1e9"),
          ("Truss set", 48, 78, "#f6f1e4"), ("Harvest", 78, NDAY - 1,
                                             "#f7ece8")]


def partition(t):
    """Daily partitioning coefficients, normalised so the four sum to one."""
    root = 0.22 * np.exp(-t / 50.0) + 0.045
    stem = 0.28 * np.exp(-t / 70.0) + 0.080
    leaf = 0.55 * np.exp(-t / 45.0) + 0.100
    fruit = 0.95 / (1.0 + np.exp(-(t - (ANTHESIS + 12.0)) / 8.0))
    raw = np.vstack([root, stem, leaf, fruit])
    return raw / raw.sum(axis=0, keepdims=True)


def grow():
    """Standing organ dry weights, g per plant, from one logistic total."""
    t = np.arange(NDAY, dtype=float)
    total = W_MAX / (1.0 + np.exp(-W_RATE * (t - W_MID)))
    total = total - total[0] + 0.9              # transplant size, g
    inc = np.diff(total, prepend=0.0)           # inc[0] is the transplant
    coef = partition(t)
    w = np.cumsum(coef * inc, axis=1)
    return t, w, w.sum(axis=0), coef


def main_panel(ax, t, share, total, hi):
    for (name, a, b, col) in STAGES:
        ax.axvspan(a, b, color=col, zorder=1)
        ax.text(0.5 * (a + b), 101.6, name, ha="center", va="bottom",
                fontsize=6.2, color=SUB, zorder=4)

    ax.stackplot(t, share, colors=[c for (_n, c) in ORGANS], zorder=3,
                 edgecolor="white", linewidth=0.35)
    edges = np.cumsum(share, axis=0)
    for k in range(share.shape[0] - 1):
        ax.plot(t, edges[k], color="white", lw=0.9, zorder=5)

    dark = [pe.withStroke(linewidth=2.0, foreground="#25352c")]
    pale = [pe.withStroke(linewidth=2.2, foreground="white")]
    ax.axvline(ANTHESIS, color="white", lw=1.1, ls=(0, (4.0, 2.4)), zorder=6)
    ax.text(ANTHESIS - 1.6, 62.0, "first truss anthesis\nday {:.0f}"
            .format(ANTHESIS), ha="right", va="center", fontsize=6.4,
            color="white", linespacing=1.32,
            zorder=8).set_path_effects(dark)

    for k, (name, col) in enumerate(ORGANS):
        j = [96, 96, 96, 96][k]
        y = edges[k, j] - 0.5 * share[k, j]
        ax.text(t[j], y, "{:s}  {:.0f} %".format(name, share[k, j]),
                ha="center", va="center", fontsize=7.0, color="white",
                fontweight="bold", zorder=8)

    axr = ax.twinx()
    axr.plot(t, total, color=INK, lw=1.8, zorder=9)
    axr.plot(t, total * hi / 100.0, color=GOLD, lw=1.3,
             ls=(0, (4.5, 2.2)), zorder=9)
    axr.set_ylim(0.0, 620.0)
    axr.set_ylabel("standing dry weight (g plant$^{-1}$)", fontsize=7.6)
    axr.tick_params(axis="y", labelsize=7.0)
    axr.spines["top"].set_visible(False)
    axr.annotate("total dry weight {:.0f} g plant$^{{-1}}$ at day {:.0f}"
                 .format(total[-1], t[-1]), xy=(t[-14], total[-14]),
                 xytext=(58.0, 560.0), fontsize=6.5, color=INK, ha="left",
                 va="top", zorder=10,
                 arrowprops=dict(arrowstyle="-", color=SUB, lw=0.75,
                                 shrinkA=2, shrinkB=3)
                 ).set_path_effects(pale)
    axr.annotate("fruit dry weight {:.0f} g plant$^{{-1}}$"
                 .format(total[-1] * hi[-1] / 100.0), xy=(t[-14],
                 total[-14] * hi[-14] / 100.0), xytext=(60.0, 170.0),
                 fontsize=6.5, color="#7a5c12", ha="left", va="top",
                 zorder=10, arrowprops=dict(arrowstyle="-", color=GOLD,
                                            lw=0.75, shrinkA=2, shrinkB=3)
                 ).set_path_effects(pale)

    handles = [Patch(facecolor=c, edgecolor="white", lw=0.5, label=n)
               for (n, c) in ORGANS]
    handles += [Line2D([], [], color=INK, lw=1.8, label="total dry weight"),
                Line2D([], [], color=GOLD, lw=1.3, ls=(0, (4.5, 2.2)),
                       label="fruit dry weight")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.6, ncol=3, columnspacing=0.9, borderaxespad=0.35,
              labelspacing=0.24, bbox_to_anchor=(0.0, -0.012))

    ax.set_xlim(0.0, NDAY - 1)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("days after transplanting (d)")
    ax.set_ylabel("share of standing dry weight (%)")
    ax.set_title("Where each day's assimilate ends up in a 120 d tomato crop",
                 fontsize=9.0, fontweight="bold", color=INK, pad=13.0)


def harvest_panel(ax, t, w):
    x = np.arange(len(HARVESTS), dtype=float)
    bottom = np.zeros(len(HARVESTS))
    for k, (name, col) in enumerate(ORGANS):
        vals = w[k, HARVESTS]
        ax.bar(x, vals, bottom=bottom, width=0.62, color=col,
               edgecolor="white", linewidth=0.7, zorder=4, label=name)
        bottom = bottom + vals
    for xi, v in zip(x, bottom):
        ax.text(xi, v + 9.0, "{:.0f}".format(v), ha="center", va="bottom",
                fontsize=6, color=SUB, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(["{:d}".format(d) for d in HARVESTS], fontsize=6.4)
    ax.set_xlim(-0.62, len(HARVESTS) - 0.38)
    ax.set_ylim(0.0, bottom.max() * 1.24)
    ax.set_xlabel("destructive harvest, days after transplanting (d)",
                  fontsize=6.6)
    ax.set_ylabel("dry weight (g plant$^{-1}$)", fontsize=6.6)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.94, handlelength=1.1, ncol=2,
              columnspacing=0.8, borderaxespad=0.3, labelspacing=0.2)
    ax.set_title("Six destructive harvests, absolute weights", fontsize=8.0,
                 fontweight="bold", color=INK)


def index_panel(ax, t, hi, day_mark):
    ax.plot(t, hi, color="#c0603f", lw=1.8, zorder=6)
    ax.fill_between(t, 0.0, hi, color="#f0d9cf", alpha=0.75, zorder=3)
    ax.axhline(HI_MARK, color=SUB, lw=0.8, ls=(0, (3.6, 2.2)), zorder=4)
    if day_mark is not None:
        ax.plot([day_mark], [HI_MARK], marker="o", ms=4.6, mfc="white",
                mec=RED, mew=1.1, zorder=8)
        ax.annotate("HI passes {:.0f} % on day {:.0f}"
                    .format(HI_MARK, day_mark), xy=(day_mark, HI_MARK),
                    xytext=(day_mark + 4.0, 18.0), fontsize=6.1, color=RED,
                    ha="left", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                    shrinkA=2, shrinkB=4))
    ax.text(NDAY - 3.0, hi[-1] + 2.4, "final HI {:.1f} %".format(hi[-1]),
            ha="right", va="bottom", fontsize=6.4, color="#8c3f22",
            fontweight="bold", zorder=9)

    ax.set_xlim(0.0, NDAY - 1)
    ax.set_ylim(0.0, 72.0)
    ax.set_xlabel("days after transplanting (d)", fontsize=6.8)
    ax.set_ylabel("harvest index (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Fruit share of the standing crop", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    np.random.default_rng(SEED)

    t, w, total, coef = grow()
    share = 100.0 * w / total[None, :]
    hi = share[3]

    assert np.allclose(share.sum(axis=0), 100.0, atol=1e-9), \
        "organ shares must sum to exactly 100 % on every day"
    assert np.allclose(coef.sum(axis=0), 1.0, atol=1e-12), \
        "partitioning coefficients must sum to one"
    assert np.all(np.diff(total) >= 0.0), "standing weight cannot fall"
    assert hi[0] < 1.0 < hi[-1], "the crop must go from no fruit to fruit"

    above = np.nonzero(hi >= HI_MARK)[0]
    day_mark = float(t[above[0]]) if above.size else None

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_hi = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, share, total, hi)
    harvest_panel(ax_h, t, w)
    index_panel(ax_hi, t, hi, day_mark)

    leaf_peak = float(share[2].max())


    panel_labels_inches(fig, [ax_main, ax_h, ax_hi], dx=-0.62, dy=0.24)
    savefig(fig, "1034_hrt_dry_matter_partition.png")


if __name__ == "__main__":
    main()
