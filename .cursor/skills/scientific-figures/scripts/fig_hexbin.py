"""Hexbin density scatter with marginal histograms and fits.

For point clouds too large to scatter.  The hexbin core carries both a linear
and a quadratic fit (the relationship is mildly curved), the marginals show
each variable on its own, and the colorbar sits in its own gridspec column so
it can never collide with the marginal panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
from pubstyle import set_pub_style, savefig, style_axes

CMAP = LinearSegmentedColormap.from_list(
    "teal", ["#f7fbfa", "#a9d6cc", "#2a8f86", "#0b3d2e"])
C_LIN, C_QUAD = "#b8503a", "#2c3e70"
N = 20000


def main():
    set_pub_style()
    rng = np.random.default_rng(91)
    x = rng.normal(0, 1, N)
    y = 0.7 * x + rng.normal(0, 0.7, N) + 0.15 * x ** 2
    x = x * 15 + 60
    y = y * 12 + 40

    fig = plt.figure(figsize=(7.8, 6.15))
    gs = GridSpec(2, 3, width_ratios=[4, 0.85, 0.16],
                  height_ratios=[0.85, 4], wspace=0.05, hspace=0.05,
                  left=0.10, right=0.90, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    hb = ax.hexbin(x, y, gridsize=42, cmap=CMAP, mincnt=1, linewidths=0.1)

    xr = np.linspace(x.min(), x.max(), 120)
    b1 = np.polyfit(x, y, 1)
    b2 = np.polyfit(x, y, 2)
    ax.plot(xr, np.polyval(b1, xr), color=C_LIN, lw=1.6, label="linear fit",
            zorder=5)
    ax.plot(xr, np.polyval(b2, xr), color=C_QUAD, lw=1.6, ls=(0, (4, 2)),
            label="quadratic fit", zorder=6)

    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    ax.text(0.035, 0.965,
            f"Pearson r = {r:.2f}\nSpearman $\\rho$ = {rho:.2f}\n"
            f"n = {N:,} plots", transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", linespacing=1.4, zorder=8,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.94))
    ax.legend(loc="lower right", fontsize=7.5, frameon=False,
              handlelength=1.8, borderaxespad=0.6)
    ax.set_xlabel("Canopy cover (%)")
    ax.set_ylabel("Above-ground biomass (Mg ha$^{-1}$)")
    ax.tick_params(labelsize=8)
    style_axes(ax)

    axt.hist(x, bins=70, color="#2a8f86", alpha=0.75, edgecolor="white",
             linewidth=0.2)
    axt.axis("off")
    axr.hist(y, bins=70, orientation="horizontal", color="#2a8f86",
             alpha=0.75, edgecolor="white", linewidth=0.2)
    axr.axis("off")

    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Plots per hexagon", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    savefig(fig, "18_hexbin.png")


if __name__ == "__main__":
    main()
