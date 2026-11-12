"""2x3 distribution grid: histogram + KDE overlay + mean/median reference lines.

Each panel shows one covariate on its real-world scale as a filled histogram
with a smooth KDE, dashed mean and dotted median lines, and a stats box
(n, mean, s.d., skewness). Axis limits are trimmed to the 0.2-99.8 percentile
range so heavy tails do not leave the panel mostly empty, and the callout box
moves to the emptier side of the distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, skew
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

FILL, LINE = "#a9d6cc", "#1a6f66"
MEAN_C = PALETTES["teal_rust"][1]

# (title, x-axis label, generator key)
VARS = [
    ("NDVI",               "NDVI (-)",                      "ndvi"),
    ("Mean air temperature", "Temperature ($^\\circ$C)",    "temp"),
    ("Annual precipitation", "Precipitation (mm yr$^{-1}$)", "precip"),
    ("Elevation",          "Elevation (m a.s.l.)",          "elev"),
    ("Population density", "Density (persons km$^{-2}$)",   "pop"),
    ("GDP per capita",     "GDP per capita (10$^3$ USD)",   "gdp"),
]


def sample(rng, key):
    if key == "ndvi":
        return np.clip(rng.normal(0.52, 0.14, 3000), 0.02, 0.98)
    if key == "temp":
        return rng.normal(12.4, 4.6, 3000)
    if key == "precip":
        return 180 + rng.gamma(3.1, 165, 3000)
    if key == "elev":
        return np.concatenate([rng.normal(340, 130, 1700),
                               rng.normal(1420, 260, 1300)])
    if key == "pop":
        return rng.lognormal(4.6, 1.05, 3000)
    return rng.lognormal(2.55, 0.62, 3000)


def panel(ax, rng, idx, spec):
    name, xlabel, key = spec
    d = sample(rng, key)
    lo, hi = np.percentile(d, [0.2, 99.8])
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad

    bins = np.linspace(lo, hi, 41)
    ax.hist(d, bins=bins, density=True, color=FILL, edgecolor="white",
            linewidth=0.35, alpha=0.95, zorder=2)
    xs = np.linspace(lo, hi, 300)
    ax.plot(xs, gaussian_kde(d)(xs), color=LINE, lw=1.8, zorder=4)

    mu, med, sg, sk = np.mean(d), np.median(d), np.std(d), skew(d)
    ax.axvline(mu, color=MEAN_C, ls="--", lw=1.1, zorder=5)
    ax.axvline(med, color="#333333", ls=":", lw=1.1, zorder=5)

    txt = (f"n = {d.size:,}\n$\\mu$ = {mu:.3g}\n$\\sigma$ = {sg:.3g}\n"
           f"skew = {sk:+.2f}")
    if sk > 0.25:          # long right tail -> the right side is emptier
        xa, ha = 0.97, "right"
    elif sk < -0.25:
        xa, ha = 0.03, "left"
    else:
        xa, ha = 0.03, "left"
    ax.text(xa, 0.96, txt, transform=ax.transAxes, ha=ha, va="top",
            fontsize=7, linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))

    ax.set_xlim(lo, hi)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                        useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.19, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(241)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), VARS)):
        panel(ax, rng, k, spec)

    handles = [Line2D([], [], color=FILL, lw=6, label="Histogram"),
               Line2D([], [], color=LINE, lw=1.8, label="Gaussian KDE"),
               Line2D([], [], color=MEAN_C, ls="--", lw=1.1, label="Mean"),
               Line2D([], [], color="#333333", ls=":", lw=1.1, label="Median")]
    fig.tight_layout(rect=[0, 0.025, 1, 0.945], h_pad=2.5, w_pad=2.6)
    savefig(fig, "31_hist_kde_grid.png")


if __name__ == "__main__":
    main()
