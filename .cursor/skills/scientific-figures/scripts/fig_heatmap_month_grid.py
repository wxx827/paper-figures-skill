"""2x3 month x year heatmap grid: seasonal temperature anomalies by region.

Each panel is a 12-month x 11-year matrix on a diverging scale centred on zero.
The synthetic field combines a regional warming trend, a basin-wide year effect
(an ENSO-like signal shared by all regions), a region-specific seasonal cycle
and month-to-month persistence, so the panels correlate with one another the
way real regional anomaly fields do. One shared colorbar for all six panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

# (region, warming trend in sigma per decade, seasonal amplitude)
REGIONS = [("Northeast", 0.62, 0.55), ("North China", 0.48, 0.42),
           ("East China", 0.35, 0.30), ("South China", 0.21, 0.22),
           ("Southwest", 0.29, 0.34), ("Northwest", 0.71, 0.61)]
MONTHS = list("JFMAMJJASOND")
YEARS = np.arange(2010, 2021)
CMAP = LinearSegmentedColormap.from_list("anom",
                                         PALETTES["div_rdbu"][::-1], N=256)


def panel(ax, rng, idx, spec, year_effect):
    name, trend, seas = spec
    n_y = len(YEARS)
    t = (YEARS - YEARS[0]) / 10.0
    data = (trend * t[:, None]
            + 0.75 * year_effect[:, None] * rng.normal(1.0, 0.25)
            + seas * np.sin(np.linspace(0, 2 * np.pi, 12) - 0.6)[None, :])
    # month-to-month persistence within each year
    e = rng.normal(0, 0.55, (n_y, 12))
    for m in range(1, 12):
        e[:, m] = 0.4 * e[:, m - 1] + np.sqrt(1 - 0.4 ** 2) * e[:, m]
    data = data - data.mean() + e

    im = ax.imshow(data, aspect="auto", cmap=CMAP, interpolation="nearest",
                   norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2))
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=6.5)
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(YEARS, fontsize=6.5)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_y, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor("#666666")

    ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
    ax.text(1.0, -0.20, f"trend {trend:+.2f} $\\sigma$ decade$^{{-1}}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color="#666666")
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.22, y=1.02, size=10)
    return im


def main():
    set_pub_style()
    rng = np.random.default_rng(381)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    year_effect = rng.normal(0, 0.55, len(YEARS))    # shared "ENSO" signal
    im = None
    for k, (ax, spec) in enumerate(zip(axes.ravel(), REGIONS)):
        im = panel(ax, rng, k, spec, year_effect)

    fig.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.145,
                        wspace=0.34, hspace=0.42)
    cax = fig.add_axes([0.925, 0.16, 0.014, 0.66])
    cb = fig.colorbar(im, cax=cax, extend="both",
                      ticks=[-2, -1, 0, 1, 2])
    cb.set_label("Temperature anomaly ($\\sigma$ relative to 1981-2010)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "42_heatmap_month_grid.png")


if __name__ == "__main__":
    main()
