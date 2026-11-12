"""3x2 choropleth map grid over synthetic administrative regions.

Voronoi cells are relaxed with Lloyd's algorithm and clipped to a smooth
synthetic coastline, giving irregular "province" polygons. Each panel is one
survey year of the same indicator, sharing a single horizontal colorbar; a
small inset shows the per-panel distribution of regional values.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import Voronoi
from pubstyle import set_pub_style, savefig

CMAP = LinearSegmentedColormap.from_list(
    "teal_seq", ["#f4efe3", "#dbe6d6", "#a8ccbb", "#6aa894", "#357f73",
                 "#1b544f", "#0d2f2e"])
YEARS = [2000, 2005, 2010, 2015, 2020, 2024]


# ---------------------------------------------------------------- geometry
def land_radius(theta):
    """Smooth star-shaped coastline radius as a function of angle."""
    return (1.0 + 0.17 * np.cos(2 * theta + 0.7)
            + 0.10 * np.cos(3 * theta - 1.2)
            + 0.05 * np.cos(5 * theta + 2.1)
            + 0.03 * np.cos(7 * theta - 0.4))


def clip_to_land(pts):
    d = np.asarray(pts, float)
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    s = np.minimum(1.0, land_radius(th) / np.maximum(r, 1e-9))
    return d * s[:, None]


def densify(poly, n=7):
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    t = np.linspace(0, 1, n, endpoint=False)[:, None, None]
    seg = p[None] + (q - p)[None] * t          # (n, m, 2)
    return seg.transpose(1, 0, 2).reshape(-1, 2)


def poly_centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * a)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0


def _voronoi_cells(pts):
    ring_t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = 5.0 * np.c_[np.cos(ring_t), np.sin(ring_t)]
    vor = Voronoi(np.vstack([pts, ring]))
    cells = []
    for i in range(len(pts)):
        idx = vor.regions[vor.point_region[i]]
        cells.append(vor.vertices[idx])
    return cells


def make_regions(n=44, seed=7, relax=4):
    """Return (centroids, list of clipped polygons) tiling the landmass."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1.25, 1.25, size=2)
        th = np.arctan2(p[1], p[0])
        if np.hypot(*p) < 0.95 * land_radius(th):
            pts.append(p)
    pts = np.array(pts)
    for _ in range(relax):
        cells = [clip_to_land(densify(c, 5)) for c in _voronoi_cells(pts)]
        pts = np.array([poly_centroid(c) for c in cells])
    polys = [clip_to_land(densify(c, 8)) for c in _voronoi_cells(pts)]
    cent = np.array([poly_centroid(p) for p in polys])
    return cent, polys


def smooth_field(cent, rng, drift):
    """Spatially autocorrelated indicator with a drifting hotspot."""
    bumps = [(-0.55, 0.35, 0.45, 1.0), (0.60, -0.25, 0.38, 0.85),
             (0.10, 0.75, 0.30, 0.55), (-0.35, -0.70, 0.35, -0.60)]
    v = np.zeros(len(cent))
    for bx, by, s, w in bumps:
        v += w * np.exp(-(((cent[:, 0] - bx) ** 2 +
                           (cent[:, 1] - by) ** 2) / (2 * s ** 2)))
    hot = 0.95 * np.exp(-(((cent[:, 0] - 0.30 + 0.35 * drift) ** 2 +
                           (cent[:, 1] - 0.10 - 0.30 * drift) ** 2) / 0.10))
    v = v + hot * (0.35 + 1.1 * drift) + 0.9 * drift * cent[:, 0]
    v += rng.normal(0, 0.10, len(cent))
    return 18 + 26 * v


# ------------------------------------------------------------------- plot
def map_frame(ax, coast):
    ax.set_aspect("equal")
    pad = 0.08
    ax.set_xlim(coast[:, 0].min() - pad, coast[:, 0].max() + pad)
    ax.set_ylim(coast[:, 1].min() - pad, coast[:, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    set_pub_style()
    rng = np.random.default_rng(31)
    cent, polys = make_regions(n=46, seed=7)
    coast_t = np.linspace(0, 2 * np.pi, 400)
    coast = np.c_[np.cos(coast_t), np.sin(coast_t)] * land_radius(coast_t)[:, None]

    vals = np.array([smooth_field(cent, np.random.default_rng(100 + k), k / 5.0)
                     for k in range(6)])
    vmin, vmax = 6, 72
    vals = vmin + (vals - vals.min()) / np.ptp(vals) * (vmax - vmin)
    norm = Normalize(vmin, vmax)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.4))
    pc = None
    for k, ax in enumerate(axes.ravel()):
        v = vals[k]
        pc = PolyCollection(polys, array=v, cmap=CMAP, norm=norm,
                            edgecolors="white", linewidths=0.65)
        ax.add_collection(pc)
        ax.plot(coast[:, 0], coast[:, 1], color="#4a4a4a", lw=0.9)
        map_frame(ax, coast)

        ax.text(0.03, 1.015, "(%s)" % "abcdef"[k], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
        ax.text(0.99, 1.015, str(YEARS[k]), transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="right", va="bottom",
                color="#1b544f")
        ax.text(0.965, 0.955, "mean %.1f" % v.mean(), transform=ax.transAxes,
                fontsize=7, ha="right", va="top", color="#333",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#cccccc", lw=0.5, alpha=0.9))

        # inset distribution legend
        iax = ax.inset_axes([0.045, 0.045, 0.30, 0.155])
        cnt, edges = np.histogram(v, bins=9, range=(vmin, vmax))
        ctr = 0.5 * (edges[:-1] + edges[1:])
        iax.bar(ctr, cnt, width=(edges[1] - edges[0]) * 0.92,
                color=CMAP(norm(ctr)), edgecolor="white", linewidth=0.3)
        iax.set_xlim(vmin, vmax); iax.set_ylim(0, 15)
        iax.set_yticks([]); iax.set_xticks([vmin, vmax])
        iax.set_xticklabels(["%d" % vmin, "%d" % vmax], fontsize=6.0)
        iax.tick_params(length=1.4, pad=1)
        for sp in ("top", "right", "left"):
            iax.spines[sp].set_visible(False)
        iax.spines["bottom"].set_color("#999999")
        iax.spines["bottom"].set_linewidth(0.5)
        iax.patch.set_alpha(0)
        iax.set_title("n regions", fontsize=6.0, color="#555", pad=1.5)

        if k == 0:
            ax.annotate("N", xy=(0.055, 0.90), xytext=(0.055, 0.76),
                        xycoords="axes fraction", ha="center", fontsize=7,
                        color="#333",
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
            ax.plot([0.66, 0.94], [0.055, 0.055], transform=ax.transAxes,
                    color="#333", lw=1.8, solid_capstyle="butt")
            ax.text(0.80, 0.082, "500 km", transform=ax.transAxes,
                    fontsize=6.0, ha="center", color="#333")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.925, bottom=0.135,
                        wspace=0.05, hspace=0.14)
    cax = fig.add_axes([0.315, 0.082, 0.37, 0.022])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("Nitrogen-use intensity (kg N ha$^{-1}$ yr$^{-1}$)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")
    savefig(fig, "55_choropleth_grid.png")


if __name__ == "__main__":
    main()
