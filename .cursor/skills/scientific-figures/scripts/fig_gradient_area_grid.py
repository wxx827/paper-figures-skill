"""2x3 gradient area time series: filled curves with a true vertical gradient.

The fill is an ``imshow`` ramp that runs from the baseline to the top of the
panel and is clipped to the polygon under the curve, so the colour fades
smoothly with height instead of being painted in vertical strips. Each panel
also carries its own units, an OLS trend with its significance, and a marked
peak year.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from scipy import stats
from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label)

# (metric, unit, level, amplitude, drift, light colour, dark colour)
METRICS = [
    ("NDVI", "index (-)", 0.46, 0.055, 0.115, "#e8f5e9", "#2e7d5b"),
    ("Land-surface temperature", "$^\\circ$C", 14.6, 0.85, 1.35,
     "#fdecef", "#b23a48"),
    ("Evapotranspiration", "mm yr$^{-1}$", 520.0, 42.0, -55.0,
     "#e8f0fa", "#3b5b92"),
    ("Net primary production", "g C m$^{-2}$ yr$^{-1}$", 640.0, 55.0, 90.0,
     "#fdf5e3", "#c9a227"),
    ("Leaf area index", "m$^2$ m$^{-2}$", 2.85, 0.32, 0.55,
     "#f3ecfa", "#8e6bbf"),
    ("Albedo", "fraction (-)", 0.192, 0.011, -0.016, "#eef0f1", "#5a5a5a"),
]


def series(rng, n, level, amp, drift):
    t = np.linspace(0, 1, n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 1)
    e /= (e.std() + 1e-9)
    return level + drift * t + amp * (0.75 * e + 0.45 * np.sin(2.6 * np.pi * t))


def panel(ax, rng, idx, spec):
    name, unit, level, amp, drift, c0, c1 = spec
    years = np.arange(2000, 2021)
    y = series(rng, len(years), level, amp, drift)
    top = y.max() * 1.16

    cmap = LinearSegmentedColormap.from_list("grad", [c0, c1])
    im = ax.imshow(np.linspace(0, 1, 512).reshape(-1, 1), cmap=cmap,
                   aspect="auto", origin="lower", zorder=1, alpha=0.95,
                   extent=[years[0], years[-1], 0, top])
    verts = np.column_stack([np.r_[years, years[::-1]],
                             np.r_[y, np.zeros_like(y)]])
    clip = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)

    ax.plot(years, y, color=c1, lw=1.9, zorder=4)
    peak = int(np.argmax(y))
    ax.scatter(years[peak], y[peak], color=c1, s=34, zorder=5,
               edgecolors="white", linewidths=1.0)
    dx = 7 if years[peak] < years[-6] else -7
    ax.annotate(f"peak {years[peak]}", (years[peak], y[peak]),
                xytext=(dx, 7), textcoords="offset points", fontsize=6.8,
                color=c1, ha="left" if dx > 0 else "right")

    res = stats.linregress(years, y)
    ax.text(0.97, 0.05,
            f"trend {res.slope * 10:+.3g} decade$^{{-1}}$\n"
            f"{'p < 0.001' if res.pvalue < 1e-3 else f'p = {res.pvalue:.3f}'}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            linespacing=1.3, zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.9))

    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, top)
    ax.set_xticks(np.arange(2000, 2021, 5))
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel(unit, fontsize=8)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=2)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.19, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(531)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), METRICS)):
        panel(ax, rng, k, spec)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], h_pad=2.6, w_pad=2.6)
    savefig(fig, "54_gradient_area_grid.png")


if __name__ == "__main__":
    main()
