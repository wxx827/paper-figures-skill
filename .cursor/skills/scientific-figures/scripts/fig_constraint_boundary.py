"""Constraint-boundary scatter grid (upper and lower envelope fits).

Each panel shows the raw cloud of plot-level carbon storage against one
driver, the binned 95th / 5th percentile boundary points and the quadratic
constraint curves fitted to them.  The R2 printed next to each curve is the
coefficient of determination of that curve against its own boundary points,
so it describes how well the envelope is captured.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

C_LOW, C_UP = "#2e8b74", "#b8503a"
C_CLOUD = "#c2c2c2"
NBINS = 18
QUANT = 0.95

DRIVERS = [
    ("Mean annual temperature", "$^{\\circ}$C", 16, 25, "flat"),
    ("Mean annual precipitation", "mm", 1300, 1700, "flat"),
    ("Slope", "degrees", 0, 50, "flat"),
    ("NDVI", "-", 0.2, 0.7, "rise"),
    ("Population density", "persons km$^{-2}$", 0, 5000, "hump"),
    ("GDP density", "10$^3$ USD km$^{-2}$", 0, 25000, "hump"),
]


def boundary_pts(x, y, nbins=NBINS, q=QUANT):
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    xc, up, lo = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() > 3:
            xc.append((a + b) / 2)
            up.append(np.quantile(y[m], q))
            lo.append(np.quantile(y[m], 1 - q))
    return np.array(xc), np.array(up), np.array(lo)


def fit_r2(xc, pts, deg=2):
    b = np.polyfit(xc, pts, deg)
    resid = pts - np.polyval(b, xc)
    ss_tot = np.sum((pts - pts.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return b, max(r2, 0.0)


def panel(ax, rng, name, unit, lo_x, hi_x, shape, idx):
    n = 2500
    x = rng.uniform(lo_x, hi_x, n)
    xn = (x - lo_x) / (hi_x - lo_x)
    if shape == "hump":
        ceil = 380 - 300 * (xn - 0.4) ** 2
    elif shape == "rise":
        ceil = 320 + 90 * xn
    else:
        ceil = 370 - 20 * xn
    # skewed fill of the feasible space: most sites sit well below the ceiling
    frac = rng.beta(2.3, 2.0, n)
    y = ceil * (0.12 + 0.88 * frac) + rng.normal(0, 9, n)

    ax.scatter(x, y, s=3.5, color=C_CLOUD, alpha=0.35, edgecolors="none",
               zorder=1, rasterized=True)

    xc, up, low = boundary_pts(x, y)
    xr = np.linspace(xc.min(), xc.max(), 120)
    for pts, col, va, yy in [(up, C_UP, "top", 0.965), (low, C_LOW, "bottom", 0.035)]:
        b, r2 = fit_r2(xc, pts)
        ax.scatter(xc, pts, s=15, color=col, zorder=5, edgecolors="white",
                   linewidths=0.4)
        ax.plot(xr, np.polyval(b, xr), color=col, lw=1.7, zorder=6)
        ax.text(0.975, yy, f"R$^2$ = {r2:.2f}", color=col,
                transform=ax.transAxes, ha="right", va=va, fontsize=6.5,
                fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                          lw=0.5, alpha=0.9))

    ax.set_xlabel(f"{name} ({unit})" if unit != "-" else name, fontsize=8)
    if idx % 3 == 0:
        ax.set_ylabel("Carbon storage (Mg C ha$^{-1}$)", fontsize=8)
    ax.set_ylim(20, 440)
    ax.tick_params(labelsize=7)
    style_axes(ax)
    add_panel_label(ax, f"({chr(97 + idx)})", x=-0.17, y=1.01, size=9.5)


def main():
    set_pub_style()
    rng = np.random.default_rng(15)
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.2))
    for i, (nm, unit, lo, hi, sh) in enumerate(DRIVERS):
        panel(axes.ravel()[i], rng, nm, unit, lo, hi, sh, i)

    handles = [
        Line2D([], [], color=C_UP, marker="o", ms=4.5, mec="white", mew=0.4,
               lw=1.7, label=f"Upper boundary ({int(QUANT * 100)}th percentile)"),
        Line2D([], [], color=C_LOW, marker="o", ms=4.5, mec="white", mew=0.4,
               lw=1.7, label=f"Lower boundary ({int((1 - QUANT) * 100)}th percentile)"),
        Line2D([], [], marker="o", ls="none", color=C_CLOUD, ms=4.5,
               label="Sample plots (n = 2500 per driver)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.2, w_pad=2.2)
    savefig(fig, "07_constraint_boundary.png")


if __name__ == "__main__":
    main()
