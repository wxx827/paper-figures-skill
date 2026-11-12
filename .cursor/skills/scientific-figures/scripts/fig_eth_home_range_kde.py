"""Kernel utilisation distribution and the two isopleths that define a range.

The main panel is the claim: the animal's space use is strongly nested, so the
50 % core is a small fraction of the 95 % home range, and the minimum convex
polygon that a naive analysis would report inflates the estimate by including
ground the animal never visited.  Every area quoted is integrated from the
plotted density grid (or, for the polygon, from its convex hull), never typed
in.  The two subordinate panels are the sensitivity checks a reviewer asks
for: does the estimate saturate with sample size, and how much does it move
when the smoothing bandwidth moves.
"""
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, get_cmap, savefig, annotate_box
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    add_scale_bar, shared_limits)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GOLD = "#c9a227"

N_FIX = 340
UD_CMAP = get_cmap("cb_sequential_green")
KM2_HA = 100.0            # 1 km^2 = 100 ha


def relocations(seed=7, n=N_FIX):
    rng = np.random.default_rng(seed)
    parts = [(0.58, [1.22, 1.62], [[0.075, 0.022], [0.022, 0.052]]),
             (0.26, [2.42, 0.98], [[0.052, -0.014], [-0.014, 0.038]]),
             (0.16, [1.55, 1.45], [[0.52, 0.06], [0.06, 0.40]])]
    # Each component also carries a visit time, so the returned sequence is
    # the order the fixes were actually collected: the second patch is only
    # discovered part way through and the far excursions come later still.
    span = [(0.00, 1.0), (0.34, 1.0), (0.20, 1.0)]
    out, tim = [], []
    for (w, mu, cov), (t0, t1) in zip(parts, span):
        k = int(round(w * n))
        out.append(rng.multivariate_normal(mu, cov, k))
        tim.append(rng.uniform(t0, t1, k))
    xy = np.vstack(out)
    order = np.argsort(np.concatenate(tim))
    return xy[order][:n]


def h_ref(xy):
    """Reference bandwidth in km: the usual sigma * n^(-1/6) rule of thumb."""
    sd = 0.5 * (float(np.std(xy[:, 0])) + float(np.std(xy[:, 1])))
    return sd * xy.shape[0] ** (-1.0 / 6.0)


def ud_grid(xy, extent, nx=200, h=None):
    """Fixed-bandwidth isotropic Gaussian KDE on a regular grid.

    The bandwidth is held fixed rather than re-derived per subsample, so the
    accumulation curve in panel (b) measures how *area* grows with sample size
    instead of how the rule-of-thumb bandwidth shrinks with it.
    """
    x0, x1, y0, y1 = extent
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, nx)
    GX, GY = np.meshgrid(gx, gy)
    if h is None:
        h = h_ref(xy)
    d2 = ((GX.ravel()[:, None] - xy[None, :, 0]) ** 2
          + (GY.ravel()[:, None] - xy[None, :, 1]) ** 2)
    z = np.exp(-0.5 * d2 / h ** 2).sum(axis=1)
    z = z.reshape(GX.shape) / (2.0 * np.pi * h ** 2 * xy.shape[0])
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    return gx, gy, z, cell


def isopleth_level(z, cell, frac):
    """Density value enclosing ``frac`` of the total probability mass."""
    flat = np.sort(z.ravel())[::-1]
    cum = np.cumsum(flat) * cell
    cum = cum / cum[-1]
    return float(flat[min(int(np.searchsorted(cum, frac)), flat.size - 1)])


def iso_area(z, cell, frac):
    """Area in hectares enclosed by the ``frac`` isopleth of the grid."""
    lev = isopleth_level(z, cell, frac)
    return float((z >= lev).sum()) * cell * KM2_HA, lev


def frame_to_box(xs, ys, box_w, box_h, pad=0.10):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    mx, my = pad * (x1 - x0), pad * (y1 - y0)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    want, have = box_w / box_h, (x1 - x0) / (y1 - y0)
    if have < want:
        e = ((y1 - y0) * want - (x1 - x0)) / 2.0
        x0, x1 = x0 - e, x1 + e
    else:
        e = ((x1 - x0) / want - (y1 - y0)) / 2.0
        y0, y1 = y0 - e, y1 + e
    return x0, x1, y0, y1


def rasterize(artist):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def north_arrow(ax, x=0.955, y=0.845, size=0.075):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.text(x, y + size + 0.014, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.0, color=RULE)


def main_panel(fig, ax, xy, gx, gy, z, cell, extent):
    zp = 100.0 * z / z.max()
    a50, l50 = iso_area(z, cell, 0.50)
    a95, l95 = iso_area(z, cell, 0.95)
    p50, p95 = 100.0 * l50 / z.max(), 100.0 * l95 / z.max()

    cs = ax.contourf(gx, gy, zp, levels=np.linspace(0.0, 100.0, 13),
                     cmap=UD_CMAP, zorder=0)
    rasterize(cs)
    cs.set_edgecolor("face")

    hull = ConvexHull(xy)
    hv = np.vstack([xy[hull.vertices], xy[hull.vertices][:1]])
    a_mcp = float(hull.volume) * KM2_HA
    ax.plot(hv[:, 0], hv[:, 1], color=BLUE, lw=1.2, ls=(0, (5, 2.2, 1, 2.2)),
            zorder=4, label=f"MCP 100 %  ({a_mcp:.0f} ha)")

    ax.scatter(xy[:, 0], xy[:, 1], s=3.4, facecolor=INK, edgecolor="none",
               alpha=0.55, zorder=3, label=f"relocations (n={xy.shape[0]})")
    ax.contour(gx, gy, zp, levels=[p95], colors=[RULE], linewidths=1.4,
               linestyles=[(0, (4.5, 2.2))], zorder=5)
    ax.contour(gx, gy, zp, levels=[p50], colors=[RED], linewidths=1.9,
               linestyles="-", zorder=6)
    ax.plot([], [], color=RULE, lw=1.4, ls=(0, (4.5, 2.2)),
            label=f"95 % home range  ({a95:.0f} ha)")
    ax.plot([], [], color=RED, lw=1.9, label=f"50 % core  ({a50:.0f} ha)")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    ax.set_title("Core use is concentrated in two patches", fontsize=9.0,
                 fontweight="bold", color=INK)
    annotate_box(ax, f"core / home range = {100.0 * a50 / a95:.0f} %\n"
                     f"MCP / home range = {a_mcp / a95:.2f} x",
                 loc="upper left", fontsize=6.6, color=SUB)
    leg = ax.legend(fontsize=6.6, frameon=True, framealpha=0.92,
                    facecolor="white", edgecolor="#cccccc", borderpad=0.4,
                    loc="lower right", handlelength=1.8, borderaxespad=0.5)
    leg.set_zorder(8)
    add_scale_bar(ax, 0.5, "500 m", loc=(0.035, 0.055), height=0.016,
                  text_color=INK, halo="white", fontsize=6.4)
    north_arrow(ax)
    safe_colorbar(fig, cs, ax, label="utilisation density (% of peak)",
                  width=0.10, pad=0.07, labelsize=6.6,
                  ticks=[0, 20, 40, 60, 80, 100])
    return a50, a95, a_mcp


def accumulation(xy, extent, h, nx=96):
    ns = np.arange(20, xy.shape[0] + 1, 10)
    areas = []
    for n in ns:
        _, _, z, cell = ud_grid(xy[:n], extent, nx=nx, h=h)
        areas.append(iso_area(z, cell, 0.95)[0])
    return ns, np.array(areas)


def bandwidth_scan(xy, extent, h, nx=96):
    fac = np.linspace(0.55, 2.00, 12)
    a50, a95 = [], []
    for f in fac:
        _, _, z, cell = ud_grid(xy, extent, nx=nx, h=h * float(f))
        a50.append(iso_area(z, cell, 0.50)[0])
        a95.append(iso_area(z, cell, 0.95)[0])
    return fac, np.array(a50), np.array(a95)


def accum_panel(ax, ns, areas):
    asym = float(np.mean(areas[-4:]))
    ax.plot(ns, areas, color=BLUE, lw=1.5, marker="o", ms=2.6, mfc="white",
            mew=0.8, zorder=4)
    ax.axhline(asym, color=MUTED, lw=0.9, ls=(0, (3, 2.2)), zorder=2)
    ax.text(0.025, asym, f"asymptote {asym:.0f} ha", fontsize=6.3,
            color=MUTED, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())
    reach = ns[np.argmax(areas >= 0.95 * asym)]
    ax.axvline(reach, color=RED, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(0.025, 0.965, f"95 % of asymptote at n = {reach}", fontsize=6.3,
            color=RED, ha="left", va="top", transform=ax.transAxes)
    ax.set_xlabel("number of fixes n (-)", fontsize=7.4)
    ax.set_ylabel("area (ha)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Range area saturates with sample size", fontsize=8.2,
                 fontweight="bold", color=INK)


def band_panel(ax, fac, a50, a95):
    ax.plot(fac, a95, color=RULE, lw=1.5, ls=(0, (4.5, 2.2)), marker="s",
            ms=2.6, mfc="white", mew=0.8, label="95 % home range")
    ax.plot(fac, a50, color=RED, lw=1.7, marker="o", ms=2.6, mfc="white",
            mew=0.8, label="50 % core")
    ax.axvline(1.0, color=MUTED, lw=0.9, ls=(0, (2.4, 2.0)))
    ax.text(0.98, 0.035, "reference $h$", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=6.3, color=MUTED)
    span = 100.0 * (a95.max() - a95.min()) / a95[np.argmin(np.abs(fac - 1.0))]
    ax.set_xlabel("bandwidth factor $h/h_{ref}$ (-)", fontsize=7.4)
    ax.set_ylabel("area (ha)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.7, borderaxespad=0.25)
    ax.set_title(f"Bandwidth moves the 95 % area by {span:.0f} %",
                 fontsize=8.2, fontweight="bold", color=INK)
    return span


def main():
    set_pub_style()
    xy = relocations()
    extent = frame_to_box(xy[:, 0], xy[:, 1], 5.05, 4.62)
    h = h_ref(xy)
    gx, gy, z, cell = ud_grid(xy, extent, nx=200, h=h)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.80, 0.80, 5.05, 4.62))
    ax_acc = axes_inches(fig, (7.02, 3.40, 2.84, 2.02))
    ax_bw = axes_inches(fig, (7.02, 0.80, 2.84, 2.02))

    a50, a95, a_mcp = main_panel(fig, ax_main, xy, gx, gy, z, cell, extent)
    ns, areas = accumulation(xy, extent, h)
    fac, b50, b95 = bandwidth_scan(xy, extent, h)

    accum_panel(ax_acc, ns, areas)
    span = band_panel(ax_bw, fac, b50, b95)
    lo, hi = shared_limits([areas, b50, b95], pad=0.08, lo=0.0)
    for ax in (ax_acc, ax_bw):
        ax.set_ylim(lo, hi)

    panel_labels_inches(fig, [ax_main, ax_acc, ax_bw], dx=-0.46, dy=0.14)
    savefig(fig, "775_eth_home_range_kde.png")


if __name__ == "__main__":
    main()
