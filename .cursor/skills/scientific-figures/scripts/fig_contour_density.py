"""2D KDE contour density: filled contours + HDR lines + marginals.

Filled kernel-density contours for a bimodal leaf-trait cloud, with
highest-density-region lines at the 50 / 68 / 90 / 95% levels, group
centroids, a sparse scatter of raw observations and split marginal densities.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig, style_axes

CMAP = LinearSegmentedColormap.from_list(
    "purp", ["#ffffff", "#e6dcf2", "#b79fd6", "#8e6bbf", "#4b2e83"])
GROUPS = [("Deciduous", "#c9772f"), ("Evergreen", "#3d6b8e")]
HDR = [0.95, 0.90, 0.68, 0.50]
N = 2000


def hdr_levels(zz, dx, dy, fracs):
    flat = np.sort(zz.ravel())[::-1]
    cum = np.cumsum(flat) * dx * dy
    cum /= cum[-1]
    return [float(flat[np.searchsorted(cum, f)]) for f in fracs]


def main():
    set_pub_style()
    rng = np.random.default_rng(141)
    x1, y1 = rng.normal(18.5, 2.6, N), rng.normal(62, 11, N)
    x2 = rng.normal(25.5, 3.0, N)
    y2 = rng.normal(118, 19, N) + 1.4 * (x2 - 25.5)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    grp = np.concatenate([np.zeros(N, int), np.ones(N, int)])

    fig = plt.figure(figsize=(7.8, 6.15))
    gs = GridSpec(2, 3, width_ratios=[4, 0.85, 0.16],
                  height_ratios=[0.85, 4], wspace=0.05, hspace=0.05,
                  left=0.10, right=0.90, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    xx, yy = np.mgrid[x.min():x.max():140j, y.min():y.max():140j]
    kde = gaussian_kde(np.vstack([x, y]))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    dx = xx[1, 0] - xx[0, 0]
    dy = yy[0, 1] - yy[0, 0]

    cf = ax.contourf(xx, yy, zz, levels=14, cmap=CMAP, zorder=1)
    levels = hdr_levels(zz, dx, dy, HDR)
    cs = ax.contour(xx, yy, zz, levels=levels, colors="#3a2258",
                    linewidths=0.7, zorder=3)
    ax.clabel(cs, fmt={lv: f"{int(f * 100)}%" for lv, f in zip(levels, HDR)},
              fontsize=6.2, inline=True, inline_spacing=3)

    idx = rng.choice(len(x), 500, replace=False)
    for g, (name, col) in enumerate(GROUPS):
        m = idx[grp[idx] == g]
        ax.scatter(x[m], y[m], s=4.5, color=col, alpha=0.35,
                   edgecolors="none", zorder=2)
        gm = grp == g
        ax.plot(x[gm].mean(), y[gm].mean(), marker="X", ms=8, color=col,
                mec="white", mew=0.9, zorder=6)

    ax.set_xlabel("Leaf nitrogen concentration (mg g$^{-1}$)")
    ax.set_ylabel("Leaf mass per area (g m$^{-2}$)")
    ax.tick_params(labelsize=8)
    style_axes(ax)
    ax.legend(handles=[Line2D([], [], marker="X", ls="none", color=c, ms=6,
                              mec="white", mew=0.7, label=f"{n} centroid")
                       for n, c in GROUPS],
              loc="upper left", fontsize=7.5, frameon=False,
              borderaxespad=0.6)
    ax.text(0.985, 0.02, f"n = {2 * N:,} leaves",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.92))

    xs = np.linspace(x.min(), x.max(), 240)
    ys = np.linspace(y.min(), y.max(), 240)
    for g, (_name, col) in enumerate(GROUPS):
        m = grp == g
        axt.fill_between(xs, gaussian_kde(x[m])(xs), color=col, alpha=0.4,
                         linewidth=0)
        axt.plot(xs, gaussian_kde(x[m])(xs), color=col, lw=0.8)
        axr.fill_betweenx(ys, gaussian_kde(y[m])(ys), color=col, alpha=0.4,
                          linewidth=0)
        axr.plot(gaussian_kde(y[m])(ys), ys, color=col, lw=0.8)
    axt.axis("off")
    axr.axis("off")

    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("Probability density", fontsize=8)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.6)
    cb.formatter.set_powerlimits((-2, 2))
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    savefig(fig, "24_contour_density.png")


if __name__ == "__main__":
    main()
