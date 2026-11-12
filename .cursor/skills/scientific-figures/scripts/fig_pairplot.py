"""Correlation pair-grid per ecoregion (upper = r, lower = scatter, diag = kde).

Four pair grids, one per ecoregion, each tinted with the region's hue.  The
upper triangle shows the correlation coefficient on a diverging background so
that positive and negative associations are never confused; the diagonal
shows the marginal density and the lower triangle the scatter with its OLS
fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, PALETTES, savefig

VARS = ["NDVI", "GPP", "PRE", "TMP", "POP", "GDP", "IMP"]
# name, hue, (veg-block loading, water limitation, human-block loading,
#              greenness lost per unit impervious cover)
REGIONS = [
    ("Boreal forest", "#4a7fc0", (0.90, 0.20, 0.45, 0.10)),
    ("Temperate mosaic", "#7cb342", (0.70, 0.50, 0.70, 0.35)),
    ("Semi-arid steppe", "#e8a53a", (0.55, 0.85, 0.40, 0.20)),
    ("Subtropical urban", "#8e6bbf", (0.45, 0.35, 0.90, 0.75)),
]
DIV = LinearSegmentedColormap.from_list("div", PALETTES["div_rdbu"][::-1])
N = 90


def gen_data(rng, params):
    """Vegetation-productivity block plus a human-pressure block.

    The four ecoregions differ in how tightly productivity tracks the
    vegetation signal, how strongly water limits greenness, and how much of
    the landscape is driven by human pressure.
    """
    veg_load, water, hum_load, sealing = params
    e = lambda s=1.0: s * rng.normal(0, 1, N)
    veg = rng.normal(0, 1, N)
    hum = rng.normal(0, 1, N)
    pre = water * veg + np.sqrt(max(1 - water ** 2, 0.05)) * e()
    tmp = -0.45 * water * veg + e(0.85)
    ndvi = veg_load * veg - sealing * hum + e(0.30)
    gpp = (0.35 + 0.60 * veg_load) * veg + 0.25 * pre + e(0.30)
    pop = hum_load * hum + e(0.28)
    gdp = (0.75 * hum_load + 0.10) * hum + e(0.40)
    imp = hum_load * hum - 0.30 * veg_load * veg + e(0.35)
    return np.column_stack([ndvi, gpp, pre, tmp, pop, gdp, imp])


def pair_panel(fig, subspec, rng, title, hue, params, tag):
    data = gen_data(rng, params)
    n = len(VARS)
    gs = GridSpecFromSubplotSpec(n, n, subplot_spec=subspec, wspace=0.06,
                                 hspace=0.06)
    C = np.corrcoef(data, rowvar=False)
    first = None
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            first = first or ax
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d8d8d8")
                s.set_linewidth(0.4)
            if i == j:
                xs = np.linspace(data[:, i].min(), data[:, i].max(), 120)
                d = gaussian_kde(data[:, i])(xs)
                ax.fill_between(xs, 0, d, color=hue, alpha=0.5, linewidth=0)
                ax.plot(xs, d, color=hue, lw=0.7)
                ax.set_ylim(0, d.max() * 1.18)
            elif i < j:
                r = C[i, j]
                ax.set_facecolor(DIV((r + 1) / 2))
                txt = "white" if abs(r) > 0.62 else "#1a1a1a"
                ax.text(0.5, 0.5, f"{r:+.2f}", ha="center", va="center",
                        fontsize=6.8, color=txt, transform=ax.transAxes,
                        fontweight="bold" if abs(r) > 0.5 else "normal")
            else:
                ax.scatter(data[:, j], data[:, i], s=5.5, color=hue,
                           alpha=0.65, linewidths=0.15, edgecolors="white")
                b = np.polyfit(data[:, j], data[:, i], 1)
                xr = np.array([data[:, j].min(), data[:, j].max()])
                ax.plot(xr, np.polyval(b, xr), color="#333333", lw=0.7)
            if i == n - 1:
                ax.set_xlabel(VARS[j], fontsize=6.5, labelpad=1.5)
            if j == 0:
                ax.set_ylabel(VARS[i], fontsize=6.5, labelpad=1.5)
    first.text(0.0, 1.28, f"({tag}) {title}", transform=first.transAxes,
               ha="left", va="bottom", fontsize=9, fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(9)
    fig = plt.figure(figsize=(9.8, 10.2))
    outer = fig.add_gridspec(2, 2, wspace=0.16, hspace=0.20,
                             left=0.05, right=0.975, top=0.925, bottom=0.075)
    for k, (name, hue, params) in enumerate(REGIONS):
        pair_panel(fig, outer[k // 2, k % 2], rng, name, hue, params,
                   chr(97 + k))

    cax = fig.add_axes([0.36, 0.028, 0.28, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=DIV),
                      cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson r (upper-triangle cell shading)", fontsize=7.5,
                 labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    savefig(fig, "06_pairplot.png")


if __name__ == "__main__":
    main()
