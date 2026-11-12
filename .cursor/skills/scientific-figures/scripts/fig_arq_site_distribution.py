"""Settlement survey: where the sites are, and what the pattern is worth.

The map is built so that its parts cannot contradict each other.  A smooth
elevation field is carved along two hand-drawn watercourses, so the rivers
really do sit in valleys and the contours really do bend around them, and
sites are then drawn from a preference surface that decays with distance to
water.  Marker area encodes site area, marker shape encodes site type and
marker colour encodes period, which keeps three variables legible without a
single one of them stealing the colour channel from the terrain.

Both side panels test the map rather than decorate it.  The rank-size plot
fits a line to the log-log hierarchy and compares it with a strict Zipf slope
of -1 anchored on the largest centre; the water-distance panel compares the
observed distances against the distribution 4000 random points in the same
survey block would produce, so "sites cluster on the rivers" is measured
rather than asserted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, cKDTree
from scipy.stats import ks_2samp

from pubstyle import set_pub_style, savefig, get_palette
from figkit import (add_scale_bar, axes_inches, halo_text,
                    panel_labels_inches, unclip_legend)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
WATER = "#6b93b0"
ACCENT = "#b23a48"

FW, FH = 10.4, 6.4
# Survey block, km.  The map box is 5.40 x 4.05 in for a 24 x 18 km window,
# so the scale is 4.44 km per inch in both directions and equal aspect is
# already satisfied by the box itself.
KX, KY = 24.0, 18.0

TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_house",
    ["#f5f2ea", "#e6e1d1", "#d3ccb6", "#bfb69b", "#a89e81", "#8f856a"])

PERIODS = ["Early Bronze", "Late Bronze", "Early Iron", "Late Iron"]
TYPES = [("open settlement", "o"), ("hillfort", "^"),
         ("cemetery", "s"), ("workshop", "D")]

MAIN_RIVER = np.array([[0.4, 12.8], [4.2, 11.4], [8.0, 11.9], [11.6, 9.8],
                       [15.0, 8.2], [18.4, 7.6], [21.0, 5.9], [23.6, 4.8]])
TRIBUTARY = np.array([[6.6, 1.2], [7.8, 4.0], [9.4, 6.6], [10.8, 8.6],
                      [11.6, 9.8]])


def spline(pts, n=260):
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    t /= t[-1]
    tt = np.linspace(0.0, 1.0, n)
    return np.column_stack([PchipInterpolator(t, pts[:, 0])(tt),
                            PchipInterpolator(t, pts[:, 1])(tt)])


def clip_rect(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a polygon to the map frame."""
    def half(pl, keep, inter):
        out = []
        for i in range(len(pl)):
            a, b = pl[i - 1], pl[i]
            ka, kb = keep(a), keep(b)
            if kb:
                if not ka:
                    out.append(inter(a, b))
                out.append(b)
            elif ka:
                out.append(inter(a, b))
        return out

    def cut(pl, axis, val, lower):
        def keep(p):
            return p[axis] >= val if lower else p[axis] <= val

        def inter(a, b):
            t = (val - a[axis]) / (b[axis] - a[axis])
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return half(pl, keep, inter)

    pl = [tuple(p) for p in poly]
    for axis, val, lower in ((0, x0, True), (0, x1, False),
                             (1, y0, True), (1, y1, False)):
        if not pl:
            return []
        pl = cut(pl, axis, val, lower)
    return pl


def main():
    set_pub_style()
    rng = np.random.default_rng(4471)

    gx = np.linspace(0.0, KX, 300)
    gy = np.linspace(0.0, KY, 226)
    GX, GY = np.meshgrid(gx, gy)

    river = spline(MAIN_RIVER)
    trib = spline(TRIBUTARY, 160)
    water = np.vstack([river, trib])
    tree = cKDTree(water)
    dist_grid = tree.query(np.column_stack([GX.ravel(),
                                            GY.ravel()]))[0].reshape(GX.shape)

    base = gaussian_filter(rng.normal(0.0, 1.0, GX.shape), 17.0)
    base = (base - base.min()) / (base.max() - base.min())
    Z = 90.0 + 210.0 * base + 3.4 * (KY - GY) + 1.1 * GX
    Z = Z - 95.0 * np.exp(-(dist_grid / 1.5) ** 2)      # carve the valleys

    # ---- draw sites from a water-preference surface ----------------------
    w = np.exp(-dist_grid / 0.95)
    gzx, gzy = np.gradient(Z, gy, gx)
    slope = np.hypot(gzx, gzy)
    w = w * np.exp(-(slope / 28.0) ** 2)
    w = w.ravel() / w.sum()
    xs, ys, tries = [], [], 0
    idx_pool = rng.choice(w.size, size=4000, p=w, replace=False)
    for idx in idx_pool:
        x = float(GX.ravel()[idx] + rng.uniform(-0.05, 0.05))
        y = float(GY.ravel()[idx] + rng.uniform(-0.05, 0.05))
        if not (0.7 < x < KX - 0.7 and 0.7 < y < KY - 0.7):
            continue
        if xs and min(np.hypot(np.array(xs) - x, np.array(ys) - y)) < 1.2:
            continue
        xs.append(x)
        ys.append(y)
        tries += 1
        if len(xs) == 26:
            break
    xs, ys = np.array(xs), np.array(ys)
    n_site = xs.size

    area = np.round(np.exp(rng.normal(1.05, 0.95, n_site)), 1)
    area = np.clip(area, 0.3, None)
    area[np.argsort(area)[-1]] = 41.6           # the primate centre
    period = rng.integers(0, 4, n_site)
    kind = rng.integers(0, 4, n_site)
    kind[np.argsort(area)[-3:]] = 0             # the big three are settlements
    d_site = tree.query(np.column_stack([xs, ys]))[0]

    order = np.argsort(area)[::-1]
    rank = np.arange(1, n_site + 1)
    sorted_area = area[order]
    lg_r, lg_a = np.log10(rank), np.log10(sorted_area)
    zslope, zint = np.polyfit(lg_r, lg_a, 1)
    zipf = sorted_area[0] / rank
    dev = float(np.mean(lg_a - np.log10(zipf)))

    rand_pts = np.column_stack([rng.uniform(0.7, KX - 0.7, 4000),
                                rng.uniform(0.7, KY - 0.7, 4000)])
    d_rand = tree.query(rand_pts)[0]
    ks = ks_2samp(d_site, d_rand)

    tier1 = order[:4]
    # Voronoi on four points alone leaves every cell unbounded; a ring of far
    # dummy generators closes them so they can be clipped to the frame.
    far = np.array([[-KX, -KY], [-KX, 2 * KY], [2 * KX, -KY], [2 * KX, 2 * KY],
                    [KX / 2, -3 * KY], [KX / 2, 3 * KY],
                    [-3 * KX, KY / 2], [3 * KX, KY / 2]])
    vor = Voronoi(np.vstack([np.column_stack([xs[tier1], ys[tier1]]), far]))

    pcol = get_palette("cb_sequential_purple", 6)[2:6]

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, (0.80, 1.60, 5.40, 4.05))
    ax_r = axes_inches(fig, (6.90, 3.60, 2.90, 2.05))
    ax_w = axes_inches(fig, (6.90, 1.60, 2.90, 1.40))

    # ---- (a) the survey block --------------------------------------------
    cf = ax_m.contourf(gx, gy, Z, levels=11, cmap=TERRAIN, alpha=0.95,
                       zorder=1)
    cf.set_edgecolor("face")
    cs = ax_m.contour(gx, gy, Z, levels=11, colors="#9d947c", linewidths=0.45,
                      zorder=2)
    ax_m.clabel(cs, cs.levels[::3], fmt="%.0f m", fontsize=6.0,
                colors="#7d745e", inline_spacing=2)
    ax_m.plot(river[:, 0], river[:, 1], color=WATER, lw=2.1, zorder=3,
              solid_capstyle="round")
    ax_m.plot(trib[:, 0], trib[:, 1], color=WATER, lw=1.3, zorder=3,
              solid_capstyle="round")

    for j in range(len(tier1)):
        vp = vor.regions[vor.point_region[j]]
        if not vp or -1 in vp:
            continue
        cp = clip_rect(vor.vertices[vp], 0.0, KX, 0.0, KY)
        if len(cp) > 2:
            ax_m.add_patch(Polygon(cp, closed=True, facecolor="none",
                                   edgecolor="#6a6a6a", lw=0.7,
                                   ls=(0, (4, 2.5)), zorder=4))
    for i in tier1:
        ax_m.add_patch(Circle((xs[i], ys[i]), 2.5, facecolor=ACCENT,
                              alpha=0.09, edgecolor=ACCENT, lw=0.8,
                              ls=(0, (2.0, 1.8)), zorder=4))

    for k, (tname, mk) in enumerate(TYPES):
        sel = kind == k
        if not np.any(sel):
            continue
        ax_m.scatter(xs[sel], ys[sel], s=9.0 + 4.6 * area[sel], marker=mk,
                     c=[pcol[p] for p in period[sel]], edgecolors="white",
                     linewidths=0.55, zorder=6)
    for j, i in enumerate(tier1):
        halo_text(ax_m, xs[i] + 0.45, ys[i] + 0.45,
                  f"T{j + 1}  {area[i]:.1f} ha", ha="left", va="bottom",
                  fontsize=6.4, color=INK, halo="white", halo_lw=2.2)

    ax_m.annotate("", xy=(6.9, 2.55), xytext=(6.9, 1.05),
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.1),
                  zorder=7)
    halo_text(ax_m, 6.9, 2.68, "N", ha="center", va="bottom", fontsize=7.5,
              color=INK, halo="white", halo_lw=2.2, fontweight="bold")
    add_scale_bar(ax_m, 5.0, "5 km", loc=(0.035, 0.048), height=0.016,
                  n_seg=2, color="#2b2b2b", text_color=INK, halo="white",
                  fontsize=6.4, zorder=8)

    ax_m.set_xlim(0.0, KX)
    ax_m.set_ylim(0.0, KY)
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("easting (km)", fontsize=8.0)
    ax_m.set_ylabel("northing (km)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0, length=2.6)
    ax_m.set_title(f"Survey block, {n_site} recorded sites",
                   fontsize=9.0, fontweight="bold", loc="left")

    leg_p = ax_m.legend(handles=[Line2D([0], [0], ls="none", marker="o",
                                        ms=4.4, color=pcol[i], mec="white",
                                        mew=0.5, label=PERIODS[i])
                                 for i in range(4)],
                        loc="upper left", fontsize=6.4, frameon=True,
                        framealpha=0.88, edgecolor=HAIR, facecolor="white",
                        handletextpad=0.35, borderpad=0.35,
                        labelspacing=0.32, title="period",
                        title_fontsize=6.6)
    unclip_legend(leg_p)

    # ---- (b) rank-size ----------------------------------------------------
    ax_r.plot(rank, zipf, color=RULE, lw=1.0, ls=(0, (4, 2.2)),
              label="Zipf, slope $-$1", zorder=3)
    ax_r.plot(rank, 10.0 ** (zslope * lg_r + zint), color=ACCENT, lw=1.3,
              label=f"fitted slope {zslope:.2f}", zorder=4)
    ax_r.scatter(rank, sorted_area, s=16.0, facecolor="white",
                 edgecolor=RULE, linewidths=0.8, zorder=5)
    ax_r.fill_between(rank, sorted_area, zipf, where=sorted_area < zipf,
                      color=ACCENT, alpha=0.10, lw=0, zorder=2)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("rank", fontsize=8.0)
    ax_r.set_ylabel("site area (ha)", fontsize=8.0)
    ax_r.tick_params(labelsize=7.0, length=2.6, which="both")
    ax_r.set_title("Rank-size: one centre, a thin middle",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_r.grid(lw=0.5, color=HAIR, alpha=0.7, which="both")
    ax_r.set_axisbelow(True)
    for s in ("top", "right"):
        ax_r.spines[s].set_visible(False)
    ax_r.legend(loc="lower left", fontsize=6.6, frameon=False,
                handlelength=1.7, labelspacing=0.3, borderpad=0.1)
    ax_r.annotate(f"mean log deviation {dev:+.2f} dex\n(primo-convex)",
                  xy=(rank[len(rank) // 2], zipf[len(rank) // 2]),
                  xytext=(0.42, 0.80), textcoords="axes fraction",
                  fontsize=6.4, color=SUB, ha="left", va="top",
                  linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                  shrinkA=0, shrinkB=3))

    # ---- (c) distance to water -------------------------------------------
    bins = np.linspace(0.0, max(d_site.max(), 6.0) * 1.05, 13)
    ax_w.hist(d_site, bins=bins, color=WATER, alpha=0.75, edgecolor="white",
              linewidth=0.5, label="observed sites", zorder=3)
    hn, _ = np.histogram(d_rand, bins=bins)
    ax_w.plot(0.5 * (bins[:-1] + bins[1:]), hn * n_site / d_rand.size,
              color=RULE, lw=1.2, marker="o", ms=2.6,
              label="random points, same block", zorder=4)
    ax_w.axvline(np.median(d_site), color=ACCENT, lw=1.0, ls=(0, (3, 2)),
                 zorder=5)
    ax_w.set_xlabel("distance to nearest watercourse (km)", fontsize=8.0)
    ax_w.set_ylabel("sites (n)", fontsize=8.0)
    ax_w.tick_params(labelsize=7.0, length=2.6)
    ax_w.set_title("Sites sit closer to water than chance",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_w.grid(axis="y", lw=0.5, color=HAIR, alpha=0.7)
    ax_w.set_axisbelow(True)
    for s in ("top", "right"):
        ax_w.spines[s].set_visible(False)
    ax_w.legend(loc="upper right", fontsize=6.6, frameon=False,
                handlelength=1.5, labelspacing=0.3, borderpad=0.1)
    ax_w.text(np.median(d_site) + 0.12, ax_w.get_ylim()[1] * 0.52,
              f"median {np.median(d_site):.2f} km\nvs "
              f"{np.median(d_rand):.2f} km random", ha="left", va="top",
              fontsize=6.4, color=ACCENT, linespacing=1.3)

    fig.legend(handles=[Line2D([0], [0], ls="none", marker=mk, ms=4.0,
                               color="#8a8a8a", mec="white", mew=0.5, label=t)
                        for t, mk in TYPES] +
               [Line2D([0], [0], ls="none", marker="o", ms=6.6,
                       color="#8a8a8a", mec="white", mew=0.5,
                       label="marker area $\\propto$ site area, 0.3-42 ha"),
                Line2D([0], [0], color="#6a6a6a", lw=0.7, ls=(0, (4, 2.5)),
                       label="Thiessen territory, first-tier centres"),
                Line2D([0], [0], color=ACCENT, lw=0.6, ls=(0, (1.6, 1.6)),
                       label="2.5 km catchment")],
               loc="lower left", bbox_to_anchor=(0.028, 0.098), ncol=4,
               fontsize=6.4, frameon=False, handletextpad=0.4,
               columnspacing=1.5, labelspacing=0.4)

    panel_labels_inches(fig, [ax_m, ax_r, ax_w], dx=-0.36, dy=0.16)

    print(f"{n_site} sites, largest {sorted_area[0]:.1f} ha, "
          f"rank-size slope {zslope:.3f}, deviation {dev:+.3f} dex")
    print(f"median distance to water: sites {np.median(d_site):.2f} km, "
          f"random {np.median(d_rand):.2f} km, KS D={ks.statistic:.3f} "
          f"p={ks.pvalue:.2e}")
    savefig(fig, "890_arq_site_distribution.png")


if __name__ == "__main__":
    main()
