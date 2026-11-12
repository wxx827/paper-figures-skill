"""2x3 box + jitter strip grid: dose-response comparison.

Slim boxplots with the raw points jittered on top, a white mean diamond, a
Welch bracket between control and the highest dose, and a Spearman trend test
across the four dose levels. Response shapes differ between panels - monotone,
saturating, hormetic and flat - so the six panels are not six copies of one
effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

GROUPS = ["Control", "Low", "Medium", "High"]
COLS = PALETTES["cb_sequential_blue"][2:6]
N = 70

# (title, unit, level, spread, group effects in units of spread)
METRICS = [
    ("Water yield",       "mm yr$^{-1}$",        310.0, 38.0, [0, .45, .95, 1.5]),
    ("Carbon storage",    "Mg C ha$^{-1}$",       61.0,  8.0, [0, .55, .85, .92]),
    ("Habitat quality",   "index (0-1)",           0.66, 0.10, [0, .35, .10, -.75]),
    ("Soil respiration",  "$\\mu$mol m$^{-2}$ s$^{-1}$", 3.4, 0.6, [0, .05, -.08, .03]),
    ("Net primary prod.", "g C m$^{-2}$ yr$^{-1}$", 610.0, 72.0, [0, .30, .62, 1.05]),
    ("Air quality index", "index",                48.0,  7.5, [0, -.35, -.72, -1.30]),
]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def panel(ax, rng, idx, spec):
    name, unit, level, spread, eff = spec
    data = [rng.normal(level + spread * e, spread * (0.85 + 0.06 * g), N)
            for g, e in enumerate(eff)]

    bp = ax.boxplot(data, widths=0.42, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222222", lw=1.2),
                    whiskerprops=dict(color="#666666", lw=0.8),
                    capprops=dict(color="#666666", lw=0.8), zorder=3)
    for box, col in zip(bp["boxes"], COLS):
        box.set(facecolor=col, alpha=0.35, edgecolor=col, lw=1.0)
    for i, (d, col) in enumerate(zip(data, COLS)):
        ax.scatter(i + 1 + rng.uniform(-0.13, 0.13, len(d)), d, s=8,
                   color=col, alpha=0.5, edgecolors="none", zorder=2)
        ax.scatter(i + 1, np.mean(d), marker="D", s=26, color="white",
                   edgecolors="#222222", linewidths=0.9, zorder=6)

    lo = min(float(np.min(d)) for d in data)
    hi = max(float(np.max(d)) for d in data)
    span = hi - lo
    p = stats.ttest_ind(data[0], data[-1], equal_var=False).pvalue
    yb, h = hi + span * 0.07, span * 0.035
    ax.plot([1, 1, len(GROUPS), len(GROUPS)], [yb, yb + h, yb + h, yb],
            color="#444444", lw=0.8, zorder=6)
    s = stars(p)
    ax.text((1 + len(GROUPS)) / 2, yb + h * 1.05, s, ha="center", va="bottom",
            fontsize=7 if s == "n.s." else 8.5,
            fontweight="normal" if s == "n.s." else "bold", color="#333333")

    dose = np.repeat(np.arange(len(GROUPS)), N)
    rho, ptr = stats.spearmanr(dose, np.concatenate(data))
    ax.text(0.035, 0.965,
            f"trend $\\rho$ = {rho:+.2f}\n"
            f"{'p < 0.001' if ptr < 1e-3 else f'p = {ptr:.3f}'}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            linespacing=1.3, zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))

    ax.set_ylim(lo - span * 0.07, hi + span * 0.26)
    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels(GROUPS, fontsize=7.5)
    ax.set_xlim(0.45, len(GROUPS) + 0.55)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.19, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(471)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), METRICS)):
        panel(ax, rng, k, spec)

    handles = [
        Line2D([], [], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#222222", markersize=5, label="Mean"),
        Line2D([], [], color="#222222", lw=1.2, label="Median"),
        Line2D([], [], color="#666666", lw=5, alpha=0.3, label="IQR"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COLS[2],
               markeredgecolor="none", markersize=4, label="Individual plot"),
    ]
    fig.tight_layout(rect=[0, 0.025, 1, 0.925], h_pad=2.6, w_pad=2.6)
    savefig(fig, "47_box_jitter_grid.png")


if __name__ == "__main__":
    main()
