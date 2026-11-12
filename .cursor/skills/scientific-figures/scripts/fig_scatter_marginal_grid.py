"""2x3 faceted scatter + regression with marginal density strips.

Each facet is one bivariate relationship on its own physical scale: points
coloured by a shared aridity covariate, an OLS fit with a proper 95% mean-
response band, and slim KDE strips on the top and right margins. Relationship
strength deliberately varies between facets, including one weak case, and the
statistics box reports r, the p-value and n.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

CMAP = plt.cm.YlGnBu
FIT_C = "#333333"
KDE_C = PALETTES["teal_rust"][0]

# (title, x label, y label, x range, slope sign, correlation strength)
PANELS = [
    ("Greenness vs warmth", "Summer temperature ($^\\circ$C)", "NDVI (-)",
     (14, 31), -1, 0.78),
    ("Water balance", "Precipitation (mm yr$^{-1}$)",
     "Evapotranspiration (mm yr$^{-1}$)", (280, 1450), +1, 0.88),
    ("Productivity vs leaf area", "LAI (m$^2$ m$^{-2}$)",
     "NPP (g C m$^{-2}$ yr$^{-1}$)", (0.4, 6.2), +1, 0.83),
    ("Runoff vs terrain", "Slope ($^\\circ$)", "Runoff coefficient (-)",
     (0.5, 26), +1, 0.34),
    ("Lapse rate", "Elevation (m a.s.l.)", "Land-surface temp. ($^\\circ$C)",
     (60, 2650), -1, 0.91),
    ("Snow and albedo", "Snow-cover fraction (-)", "Albedo (-)",
     (0.0, 0.92), +1, 0.72),
]
Y_RANGE = [(0.18, 0.82), (240, 980), (120, 1250), (0.05, 0.62),
           (6, 34), (0.12, 0.68)]


def one_panel(fig, subspec, rng, idx, spec, yrange):
    title, xlab, ylab, xr, sign, rho = spec
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec,
                                 width_ratios=[4.2, 0.75],
                                 height_ratios=[0.75, 4.2],
                                 wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    n = 220
    zx = rng.normal(0, 1, n)
    zy = sign * (rho * zx + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n))
    x = xr[0] + (xr[1] - xr[0]) * stats.norm.cdf(zx * 0.85)
    y = yrange[0] + (yrange[1] - yrange[0]) * stats.norm.cdf(zy * 0.85)
    cov = stats.norm.cdf(0.6 * zx + 0.8 * rng.normal(0, 1, n))

    sc = ax.scatter(x, y, c=cov, cmap=CMAP, s=17, alpha=0.85, linewidths=0.25,
                    edgecolors="white", vmin=0, vmax=1, zorder=3)

    res = stats.linregress(x, y)
    xg = np.linspace(x.min(), x.max(), 120)
    yhat = res.intercept + res.slope * xg
    se = np.sqrt(np.sum((y - (res.intercept + res.slope * x)) ** 2) / (n - 2))
    tv = stats.t.ppf(0.975, n - 2)
    ci = tv * se * np.sqrt(1 / n + (xg - x.mean()) ** 2 /
                           np.sum((x - x.mean()) ** 2))
    ax.fill_between(xg, yhat - ci, yhat + ci, color=FIT_C, alpha=0.16, lw=0,
                    zorder=4)
    ax.plot(xg, yhat, color=FIT_C, lw=1.5, zorder=5)

    r, p = stats.pearsonr(x, y)
    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.04 if r > 0 else 0.96, 0.96,
            f"r = {r:+.2f}\n{ptxt}\nn = {n}", transform=ax.transAxes,
            ha="left" if r > 0 else "right", va="top", fontsize=6.8,
            linespacing=1.3, zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))

    xs = np.linspace(x.min(), x.max(), 120)
    axt.fill_between(xs, gaussian_kde(x)(xs), color=KDE_C, alpha=0.35, lw=0.8,
                     edgecolor=KDE_C)
    ys = np.linspace(y.min(), y.max(), 120)
    axr.fill_betweenx(ys, gaussian_kde(y)(ys), color=KDE_C, alpha=0.35,
                      lw=0.8, edgecolor=KDE_C)
    for a in (axt, axr):
        a.axis("off")
        a.margins(0)
    axt.set_title(title, fontsize=8.8, fontweight="bold", pad=4)

    ax.set_xlabel(xlab, fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(lw=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(axt, f"({'abcdef'[idx]})", x=-0.20, y=0.6, size=10)
    return sc


def main():
    set_pub_style()
    rng = np.random.default_rng(391)
    fig = plt.figure(figsize=(11.4, 7.2))
    outer = fig.add_gridspec(2, 3, left=0.065, right=0.885, top=0.90,
                             bottom=0.10, wspace=0.42, hspace=0.48)
    sc = None
    for k, (spec, yr) in enumerate(zip(PANELS, Y_RANGE)):
        sc = one_panel(fig, outer[k // 3, k % 3], rng, k, spec, yr)

    cax = fig.add_axes([0.905, 0.28, 0.013, 0.44])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Aridity index (-)", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "43_scatter_marginal_grid.png")


if __name__ == "__main__":
    main()
