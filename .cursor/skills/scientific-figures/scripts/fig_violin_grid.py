"""2x3 violin grid: violins + inner box + mean dot + significance brackets.

Each panel compares four protection zones for one ecosystem-service metric.
Violins are coloured by zone, overlaid with a slim boxplot (IQR + median), a
white mean marker, and two Welch-test brackets against the Core zone. Effect
directions differ between metrics, including one non-significant case.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

GROUPS = ["Core", "Buffer", "Transition", "Outside"]
COLORS = PALETTES["cb_muted"][:4]
N = 120

# (title, unit, level, spread, zone effects in units of spread)
METRICS = [
    ("Water yield",        "mm yr$^{-1}$",        320.0, 42.0, [0, .35, .78, 1.15]),
    ("Carbon storage",     "Mg C ha$^{-1}$",       68.0,  9.0, [0, -.12, -.45, -.95]),
    ("Habitat quality",    "index (0-1)",          0.74, 0.085, [0, -.28, -.85, -1.55]),
    ("Soil conservation",  "t ha$^{-1}$ yr$^{-1}$", 24.0, 5.2, [0, .08, .12, .04]),
    ("Net primary prod.",  "g C m$^{-2}$ yr$^{-1}$", 640.0, 78.0, [0, .30, .52, .22]),
    ("Comprehensive index", "z-score",             0.00, 0.75, [0, -.22, -.62, -1.25]),
]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#444444", lw=0.8,
            clip_on=False, zorder=6)
    weight = "normal" if text == "n.s." else "bold"
    size = 7 if text == "n.s." else 8.5
    ax.text((x1 + x2) / 2, y + h * 1.05, text, ha="center", va="bottom",
            fontsize=size, fontweight=weight, color="#333333", zorder=6)


def panel(ax, rng, idx, spec):
    name, unit, level, spread, eff = spec
    data = [rng.normal(level + spread * e, spread * (0.95 + 0.09 * g), N)
            for g, e in enumerate(eff)]

    parts = ax.violinplot(data, widths=0.82, showextrema=False)
    for pc, col in zip(parts["bodies"], COLORS):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
        pc.set_edgecolor(col)
        pc.set_linewidth(0.9)
    bp = ax.boxplot(data, widths=0.11, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222222", lw=1.1),
                    whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8))
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor="#555555", lw=0.7, alpha=0.95)
    for g, d in enumerate(data):
        ax.scatter(g + 1, np.mean(d), color="white", edgecolors="#222222",
                   s=20, zorder=7, linewidths=0.9)

    # use the full support so the violin tails are never clipped by the frame
    lo = min(float(np.min(d)) for d in data)
    hi = max(float(np.max(d)) for d in data)
    rng_y = hi - lo
    h = rng_y * 0.035
    for k, j in enumerate((2, 3)):
        p = stats.ttest_ind(data[0], data[j], equal_var=False).pvalue
        bracket(ax, 1, j + 1, hi + rng_y * (0.07 + 0.13 * k), h, stars(p))
    ax.set_ylim(lo - rng_y * 0.08, hi + rng_y * 0.34)

    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels(GROUPS, fontsize=7.5)
    ax.set_xlim(0.45, len(GROUPS) + 0.55)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.17, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(231)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), METRICS)):
        panel(ax, rng, k, spec)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#222222", markersize=5, label="Mean"),
        Line2D([], [], color="#222222", lw=1.2, label="Median"),
        Line2D([], [], color="#555555", lw=5, alpha=0.35, label="IQR"),
        Line2D([], [], color="#b3b3b3", lw=5, alpha=0.5,
               label="Kernel density"),
    ]
    fig.tight_layout(rect=[0, 0.025, 1, 0.925], h_pad=2.6, w_pad=2.6)
    savefig(fig, "30_violin_grid.png")


if __name__ == "__main__":
    main()
