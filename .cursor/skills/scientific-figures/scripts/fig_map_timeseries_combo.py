"""Map + time-series combination layout (Nature-style composite).

A large regional choropleth on the left is linked by colour-matched leader
lines to four stacked time-series panels on the right, one per highlighted
region. Each series carries a bootstrap-style confidence band and a decadal
trend annotation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch, Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.spatial import Voronoi
from pubstyle import set_pub_style, savefig

BASE_CMAP = LinearSegmentedColormap.from_list(
    "sand", ["#f6f3ec", "#e3e1d2", "#c9cfb6", "#a9bb9c", "#87a487"])
ACCENT = ["#2f6d8c", "#c0603f", "#7a6aa8", "#2f7f6a"]
YEARS = np.arange(1992, 2025)


def land_radius(theta):
    return (1.0 + 0.16 * np.cos(2 * theta + 2.2)
            + 0.09 * np.cos(3 * theta - 0.9)
            + 0.05 * np.cos(5 * theta + 1.3))


def clip_to_land(pts):
    d = np.asarray(pts, float)
    r = np.hypot(d[:, 0], d[:, 1])
    s = np.minimum(1.0, land_radius(np.arctan2(d[:, 1], d[:, 0]))
                   / np.maximum(r, 1e-9))
    return d * s[:, None]


def densify(poly, n=7):
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    t = np.linspace(0, 1, n, endpoint=False)[:, None, None]
    return (p[None] + (q - p)[None] * t).transpose(1, 0, 2).reshape(-1, 2)


def poly_centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * a)


def _cells(pts):
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    vor = Voronoi(np.vstack([pts, 5.0 * np.c_[np.cos(t), np.sin(t)]]))
    return [vor.vertices[vor.regions[vor.point_region[i]]]
            for i in range(len(pts))]


def make_regions(n=58, seed=5, relax=5):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1.2, 1.2, size=2)
        if np.hypot(*p) < 0.95 * land_radius(np.arctan2(p[1], p[0])):
            pts.append(p)
    pts = np.array(pts)
    for _ in range(relax):
        pts = np.array([poly_centroid(clip_to_land(densify(c, 4)))
                        for c in _cells(pts)])
    polys = [clip_to_land(densify(c, 8)) for c in _cells(pts)]
    return np.array([poly_centroid(p) for p in polys]), polys


def series(rng, base, slope, curv, noise):
    x = (YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0])
    mu = base + slope * x + curv * x ** 2
    wig = np.cumsum(rng.normal(0, noise, len(x)))
    wig -= np.linspace(0, wig[-1], len(x))
    y = mu + wig + rng.normal(0, noise * 0.7, len(x))
    ci = noise * 3.0 * (0.6 + 0.8 * x)
    return y, ci


def main():
    set_pub_style()
    cent, polys = make_regions()
    t = np.linspace(0, 2 * np.pi, 500)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]

    rng = np.random.default_rng(77)
    base = np.zeros(len(cent))
    for bx, by, s, w in [(-0.5, 0.4, 0.5, 1.0), (0.6, -0.3, 0.42, 0.85),
                         (0.15, 0.75, 0.3, 0.5), (-0.4, -0.6, 0.35, 0.7)]:
        base += w * np.exp(-(((cent[:, 0] - bx) ** 2 +
                              (cent[:, 1] - by) ** 2) / (2 * s ** 2)))
    base += rng.normal(0, 0.08, len(cent))
    base = 100 * (base - base.min()) / np.ptp(base)

    # four well-separated focus regions
    targets = np.array([[-0.62, 0.52], [0.62, 0.45], [0.66, -0.42],
                        [-0.55, -0.55]])
    sel = [int(np.argmin(((cent - t0) ** 2).sum(1))) for t0 in targets]
    names = ["Northwest\nuplands", "Northeast\nbasin",
             "Southeast\ncoast", "Southwest\nplateau"]

    fig = plt.figure(figsize=(10.6, 6.6))
    gs = GridSpec(4, 2, width_ratios=[1.55, 1], left=0.015, right=0.965,
                  top=0.895, bottom=0.085, wspace=0.13, hspace=0.24)

    axm = fig.add_subplot(gs[:, 0])
    pc = PolyCollection(polys, array=base, cmap=BASE_CMAP,
                        norm=Normalize(0, 100), edgecolors="white",
                        linewidths=0.5)
    axm.add_collection(pc)
    axm.plot(coast[:, 0], coast[:, 1], color="#3a3a3a", lw=1.0, zorder=5)
    for j, i in enumerate(sel):
        axm.add_patch(Polygon(polys[i], closed=True, facecolor=ACCENT[j],
                              edgecolor="white", lw=1.4, zorder=6, alpha=0.95))
        axm.text(cent[i, 0], cent[i, 1], "%d" % (j + 1), ha="center",
                 va="center", fontsize=8.5, fontweight="bold", color="white",
                 zorder=7)
    axm.set_aspect("equal")
    axm.set_xlim(coast[:, 0].min() - 0.07, coast[:, 0].max() + 0.07)
    axm.set_ylim(coast[:, 1].min() - 0.07, coast[:, 1].max() + 0.07)
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_visible(False)
    axm.text(0.0, 1.005, "(a)", transform=axm.transAxes, fontsize=11,
             fontweight="bold", va="bottom")
    axm.text(0.058, 1.008, "Long-term mean water stress index",
             transform=axm.transAxes, fontsize=9.5, fontweight="bold",
             va="bottom", color="#4a6b52")
    axm.annotate("N", xy=(0.045, 0.955), xytext=(0.045, 0.865),
                 xycoords="axes fraction", ha="center", fontsize=7,
                 color="#333",
                 arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    axm.plot([0.03, 0.19], [0.03, 0.03], transform=axm.transAxes,
             color="#333", lw=1.8, solid_capstyle="butt")
    axm.text(0.11, 0.048, "300 km", transform=axm.transAxes, fontsize=6,
             ha="center", color="#333")

    cax = fig.add_axes([0.075, 0.055, 0.20, 0.018])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("water stress index (0-100)", fontsize=6.8, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#777")

    # ---- stacked time series
    axes_ts = []
    for j in range(4):
        axt = fig.add_subplot(gs[j, 1])
        axes_ts.append(axt)
        y, ci = series(np.random.default_rng(200 + j),
                       base=[38, 52, 61, 30][j], slope=[16, 6, 24, -9][j],
                       curv=[6, -3, 9, -5][j], noise=[1.1, 1.4, 1.0, 1.3][j])
        axt.fill_between(YEARS, y - ci, y + ci, color=ACCENT[j], alpha=0.16,
                         lw=0)
        axt.plot(YEARS, y, color=ACCENT[j], lw=1.5)
        axt.plot(YEARS[-1], y[-1], "o", ms=4, color=ACCENT[j],
                 mec="white", mew=0.8, zorder=5)
        k = np.polyfit(YEARS, y, 1)[0] * 10
        axt.text(0.985, 0.86, "%+.1f dec$^{-1}$" % k, transform=axt.transAxes,
                 ha="right", va="top", fontsize=7, fontweight="bold",
                 color=ACCENT[j])
        axt.text(0.018, 0.86, "%d  %s" % (j + 1, names[j].replace("\n", " ")),
                 transform=axt.transAxes, ha="left", va="top", fontsize=7.4,
                 fontweight="bold", color="#333")
        axt.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 3.5)
        lo, hi = (y - ci).min(), (y + ci).max()
        axt.set_ylim(lo - 0.30 * (hi - lo), hi + 0.42 * (hi - lo))
        axt.tick_params(labelsize=6.5, length=2.5)
        axt.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        for sp in ("top", "right"):
            axt.spines[sp].set_visible(False)
        if j < 3:
            axt.set_xticklabels([])
        else:
            axt.set_xlabel("Year", fontsize=7.5, labelpad=1.5)
        if j == 0:
            axt.text(0.0, 1.16, "(b)", transform=axt.transAxes, fontsize=11,
                     fontweight="bold", va="bottom")
            axt.text(0.095, 1.165, "Regional trajectories (1992-2024)",
                     transform=axt.transAxes, fontsize=9.5, fontweight="bold",
                     va="bottom", color="#333")

    # ---- leader lines linking map polygons to panels
    for j, i in enumerate(sel):
        con = ConnectionPatch(xyA=(cent[i, 0], cent[i, 1]), coordsA=axm.transData,
                              xyB=(-0.012, 0.5), coordsB=axes_ts[j].transAxes,
                              color=ACCENT[j], lw=0.8, alpha=0.55,
                              linestyle=(0, (4, 2)),
                              connectionstyle="arc3,rad=%.2f" % (0.08 - 0.05 * j),
                              zorder=1)
        fig.add_artist(con)

    savefig(fig, "58_map_timeseries_combo.png")


if __name__ == "__main__":
    main()
