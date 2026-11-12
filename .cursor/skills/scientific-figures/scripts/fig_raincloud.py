"""Raincloud plots: half-violin + boxplot + jittered strip.

Each land-cover class gets a "cloud" (half violin), a slim box with the median
and IQR, and a "rain" of jittered raw observations.  The same colour is used
for a given land cover in both panels, group sizes are printed under the axis
and a Kruskal-Wallis test summarises the between-group contrast.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kruskal
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

LABELS = ["Forest", "Grassland", "Cropland", "Wetland", "Urban"]
# one consistent colour per land cover, reused in both panels
COLORS = ["#2e7d5b", "#7cb342", "#c9a227", "#3d6b8e", "#b0563f"]


def half_violin(ax, data, pos, color, width=0.34):
    kde = gaussian_kde(data)
    ys = np.linspace(data.min(), data.max(), 160)
    dens = kde(ys)
    dens = dens / dens.max() * width
    ax.fill_betweenx(ys, pos, pos + dens, color=color, alpha=0.55, lw=0,
                     zorder=3)
    ax.plot(pos + dens, ys, color=color, lw=0.9, zorder=4)


def raincloud(ax, groups, rng):
    for i, (d, col) in enumerate(zip(groups, COLORS)):
        half_violin(ax, d, i + 0.06, col)
        bp = ax.boxplot(d, positions=[i - 0.10], widths=0.11,
                        patch_artist=True, showfliers=False, zorder=5,
                        medianprops=dict(color="#1a1a1a", lw=1.1))
        for box in bp["boxes"]:
            box.set(facecolor="white", alpha=0.95, edgecolor=col, lw=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=col, lw=0.9)
        jit = rng.uniform(-0.34, -0.19, len(d))
        ax.scatter(i + jit, d, s=7, color=col, alpha=0.45, zorder=2,
                   linewidths=0.15, edgecolors="white")
        ax.plot([i - 0.10], [np.mean(d)], marker="D", ms=3.4, color="white",
                mec="#1a1a1a", mew=0.7, zorder=6)

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([f"{lab}\n(n = {len(g)})" for lab, g in
                        zip(LABELS, groups)], fontsize=7.5, linespacing=1.4)
    ax.set_xlim(-0.55, len(LABELS) - 0.35)
    ax.grid(axis="y", color="#ededed", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, length=2.5)
    style_axes(ax)

    h, p = kruskal(*groups)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.985, 0.975, f"Kruskal-Wallis H = {h:.1f}, {ptxt}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.92))


def main():
    set_pub_style()
    rng = np.random.default_rng(31)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.6))

    g1 = [rng.normal(m, s, n) for m, s, n in
          [(3.2, 0.6, 160), (2.6, 0.7, 150), (2.1, 0.5, 165),
           (3.6, 0.8, 120), (1.4, 0.4, 140)]]
    raincloud(axes[0], g1, rng)
    axes[0].set_ylabel("Carbon density (kg C m$^{-2}$)", fontsize=8.5)
    axes[0].set_title("Soil carbon density by land cover", fontsize=9,
                      fontweight="bold")

    g2 = [rng.gamma(k, sc, n) for k, sc, n in
          [(4.0, 1.0, 160), (3.0, 1.0, 150), (2.4, 1.0, 165),
           (5.0, 1.0, 120), (1.6, 1.0, 140)]]
    raincloud(axes[1], g2, rng)
    axes[1].set_ylabel("Water yield (mm yr$^{-1}$)", fontsize=8.5)
    axes[1].set_title("Water yield by land cover", fontsize=9,
                      fontweight="bold")

    for ax, lab in zip(axes, "ab"):
        add_panel_label(ax, f"({lab})", x=-0.10, y=1.01, size=10)
    fig.tight_layout(w_pad=3.0)
    savefig(fig, "12_raincloud.png")


if __name__ == "__main__":
    main()
