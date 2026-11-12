"""2x3 dual y-axis time series: primary line + secondary bars/line.

Each panel pairs a left-axis driver (line) with a right-axis response (bars or
dashed line). The two series are generated from a shared latent process with a
one-year lag and a panel-specific coupling strength, so the pairs are related
without being carbon copies of each other; the Pearson r of the plotted pair is
reported in the corner.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

# (title, left label, right label, mode, left range, right range, coupling)
PANELS = [
    ("Warming and greening", "Air temperature ($^\\circ$C)", "NDVI (-)",
     "line", (7.8, 10.6), (0.42, 0.58), 0.80),
    ("Rainfall and runoff", "Precipitation (mm yr$^{-1}$)",
     "Runoff coefficient (-)", "bar", (540, 980), (0.24, 0.46), 0.72),
    ("Economy and urban form", "GDP (10$^9$ USD)", "Urban area (km$^2$)",
     "line", (11.0, 38.0), (180, 430), 0.93),
    ("Fertilizer and N leaching", "Fertilizer (kg N ha$^{-1}$)",
     "N leaching (kg N ha$^{-1}$)", "bar", (145, 265), (12, 34), 0.66),
    ("Population and sealing", "Population (10$^4$ persons)",
     "Impervious cover (%)", "line", (98, 143), (18, 41), 0.88),
    ("Renewable capacity", "Solar capacity (GW)", "Wind capacity (GW)",
     "bar", (0.4, 9.2), (1.6, 7.4), 0.55),
]
LC, RC = PALETTES["cb_safe"][0], PALETTES["cb_safe"][5]


def ar1(rng, n, rho=0.65):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, 1)
    return (e - e.mean()) / (e.std() + 1e-9)


def rescale(z, lo, hi):
    z = (z - z.min()) / (np.ptp(z) + 1e-9)
    return lo + z * (hi - lo)


def panel(ax, rng, idx, spec):
    title, yl, yr, mode, rl, rr, rho = spec
    years = np.arange(2000, 2021)
    n = len(years)
    t = np.linspace(0, 1, n)
    latent = 1.9 * t + 0.55 * ar1(rng, n)
    y1 = rescale(latent + 0.25 * ar1(rng, n), *rl)
    lagged = np.r_[latent[0], latent[:-1]]
    y2 = rescale(rho * lagged + np.sqrt(1 - rho ** 2) * 1.1 * ar1(rng, n), *rr)

    ax.plot(years, y1, color=LC, lw=1.8, marker="o", ms=3.6, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5, label=yl)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel(yl, fontsize=8, color=LC)
    ax.tick_params(axis="y", labelcolor=LC, color=LC)
    ax.spines["left"].set_color(LC)
    ax.set_xlim(1999, 2021)
    ax.grid(axis="y", lw=0.5, alpha=0.35)

    ax2 = ax.twinx()
    if mode == "bar":
        ax2.bar(years, y2, width=0.62, color=RC, alpha=0.45, lw=0.5,
                edgecolor="white", zorder=1, label=yr)
        ax2.set_ylim(0, y2.max() * 1.18)
    else:
        ax2.plot(years, y2, color=RC, lw=1.5, ls="--", marker="s", ms=3.2,
                 markeredgecolor="white", markeredgewidth=0.4, zorder=3,
                 label=yr)
        pad = np.ptp(y2) * 0.18
        ax2.set_ylim(y2.min() - pad, y2.max() + pad * 1.6)
    ax2.set_ylabel(yr, fontsize=8, color=RC)
    ax2.tick_params(axis="y", labelcolor=RC, color=RC)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RC)
    ax2.spines["left"].set_color(LC)

    r = np.corrcoef(y1, y2)[0, 1]
    ax.text(0.035, 0.965, f"r = {r:+.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.20, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(321)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    for k, (ax, cfg) in enumerate(zip(axes.ravel(), PANELS)):
        panel(ax, rng, k, cfg)
    handles = [Line2D([], [], color=LC, lw=1.8, marker="o", ms=3.6,
                      label="Left axis: driver"),
               Line2D([], [], color=RC, lw=1.5, ls="--", marker="s", ms=3.2,
                      label="Right axis: response (line)"),
               Patch(facecolor=RC, alpha=0.45,
                     label="Right axis: response (bars)")]
    fig.tight_layout(rect=[0, 0.025, 1, 0.925], h_pad=2.8, w_pad=3.4)
    savefig(fig, "36_dualaxis_grid.png")


if __name__ == "__main__":
    main()
