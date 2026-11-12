"""Ridgeline (joyplot): stacked overlapping density curves over time.

Overlapping filled kernel densities of plot-level NDVI, one ridge per survey
year, with a sequential green ramp.  Each ridge carries its median tick, and
the medians are joined by a thin dashed line so the shift of the whole
distribution is legible without reading the colour ramp.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from pubstyle import set_pub_style, savefig

CMAP = LinearSegmentedColormap.from_list(
    "grn", ["#dce8d5", "#7cb342", "#2e7d5b", "#0b3d2e"])
N_PER_YEAR = 500


def main():
    set_pub_style()
    rng = np.random.default_rng(41)
    years = list(range(1990, 2021, 3))
    n = len(years)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    overlap = 1.85
    xg = np.linspace(0.02, 0.98, 400)

    medians = []
    for i, yr in enumerate(years):
        loc = 0.30 + 0.22 * (i / (n - 1)) + rng.normal(0, 0.006)
        scale = 0.075 + 0.012 * rng.random()
        data = np.clip(rng.normal(loc, scale, N_PER_YEAR)
                       - rng.gamma(1.6, 0.012, N_PER_YEAR), 0.03, 0.95)
        dens = gaussian_kde(data)(xg)
        dens = dens / dens.max()
        base = (n - i) * 1.0
        col = CMAP(i / (n - 1))
        ax.fill_between(xg, base, base + dens * overlap, color=col,
                        alpha=0.9, lw=0, zorder=n - i)
        ax.plot(xg, base + dens * overlap, color="white", lw=0.7,
                zorder=n - i)
        med = np.median(data)
        medians.append((med, base))
        h = np.interp(med, xg, dens) * overlap
        ax.plot([med, med], [base, base + h * 0.32], color="white", lw=1.1,
                zorder=n - i + 0.5)
        ax.text(0.028, base + 0.16, str(yr), ha="right", va="bottom",
                fontsize=7.5, color="#444444")

    mx = [m for m, _ in medians]
    my = [b + 0.06 for _, b in medians]
    ax.plot(mx, my, color="#2b2b2b", lw=0.9, ls=(0, (3, 2)), zorder=n + 2)
    ax.scatter(mx, my, s=9, color="white", edgecolors="#2b2b2b",
               linewidths=0.7, zorder=n + 3)

    slope = np.polyfit(years, mx, 1)[0] * 10
    ax.text(0.975, 0.02,
            f"median NDVI +{slope:.3f} per decade\n"
            f"n = {N_PER_YEAR} plots per year",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            linespacing=1.4, zorder=n + 5,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.94))

    ax.set_xlim(0.0, 0.82)
    ax.set_ylim(0.4, n + overlap + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Growing-season NDVI")
    ax.set_title("Distribution shift of canopy greenness, 1990-2020",
                 fontsize=10.5, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    savefig(fig, "13_ridgeline.png")


if __name__ == "__main__":
    main()
