"""Dumbbell chart: before/after paired comparison with connecting lines.

Horizontal dumbbells for ten indicators, sorted by the size of the change and
coloured by its direction. End labels sit outside the pair so they never touch
the markers, and a right-hand column reports the signed change so the reader
does not have to subtract by eye.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes)

ENTITIES = ["Forest cover", "Water quality", "Soil organic carbon",
            "Habitat integrity", "Carbon stock", "Species richness",
            "Air quality", "Wetland extent", "Riparian buffer",
            "Erosion control"]
BEFORE_C, AFTER_C = PALETTES["blue_red"]
GAIN, LOSS = "#2e7d5b", "#b23a48"


def main():
    set_pub_style()
    rng = np.random.default_rng(351)
    before = rng.uniform(24, 78, len(ENTITIES))
    after = np.clip(before + rng.normal(4, 15, len(ENTITIES)), 5, 95)

    order = np.argsort(after - before)
    before, after = before[order], after[order]
    names = [ENTITIES[i] for i in order]
    delta = after - before

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(names))
    lo, hi = min(before.min(), after.min()), max(before.max(), after.max())
    pad = (hi - lo) * 0.16
    xlim = (lo - pad, hi + pad)

    for i, (b, a) in enumerate(zip(before, after)):
        col = GAIN if a >= b else LOSS
        ax.plot(xlim, [i, i], color="#ebebeb", lw=0.7, zorder=1)
        ax.plot([b, a], [i, i], color=col, lw=2.4, solid_capstyle="round",
                alpha=0.85, zorder=2)
        ax.scatter(b, i, color=BEFORE_C, s=58, zorder=4, edgecolors="white",
                   linewidths=0.9)
        ax.scatter(a, i, color=AFTER_C, s=58, zorder=4, edgecolors="white",
                   linewidths=0.9)
        left, right = (b, a) if b <= a else (a, b)
        lc, rc = (BEFORE_C, AFTER_C) if b <= a else (AFTER_C, BEFORE_C)
        off = (hi - lo) * 0.018
        ax.text(left - off, i, f"{left:.0f}", ha="right", va="center",
                fontsize=7.5, color=lc, zorder=5)
        ax.text(right + off, i, f"{right:.0f}", ha="left", va="center",
                fontsize=7.5, color=rc, zorder=5)

    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, d in enumerate(delta):
        ax.text(1.045, i, f"{d:+.1f}", transform=tr, ha="right", va="center",
                fontsize=7.8, fontweight="bold",
                color=GAIN if d >= 0 else LOSS, clip_on=False)
    ax.text(1.045, len(names) - 0.35, "$\\Delta$", transform=tr, ha="right",
            va="center", fontsize=8.5, fontweight="bold", color="#333333",
            clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Composite condition index (0-100)", fontsize=9)
    ax.set_title("Ecosystem indicators before and after the 2010 restoration "
                 "programme", fontsize=10.5, fontweight="bold", loc="left",
                 x=-0.16, pad=10)
    ax.grid(axis="x", lw=0.5, alpha=0.4)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BEFORE_C,
               markeredgecolor="white", markersize=7, label="2010 (before)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=AFTER_C,
               markeredgecolor="white", markersize=7, label="2020 (after)"),
        Line2D([], [], color=GAIN, lw=2.4, label="Improvement"),
        Line2D([], [], color=LOSS, lw=2.4, label="Decline"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=4, fontsize=8, frameon=False, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.03, 0.97, 1])
    savefig(fig, "39_dumbbell.png")


if __name__ == "__main__":
    main()
