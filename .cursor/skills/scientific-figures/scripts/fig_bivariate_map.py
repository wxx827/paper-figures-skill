"""Bivariate choropleth map with a 3x3 colour-matrix legend.

Two spatially autocorrelated indicators are each split into terciles and
crossed into nine bivariate classes (Stevens blue-green scheme). The large
map carries the bivariate fill; the right column holds the 3x3 legend, a
tercile-gridded scatter of the two indicators, and two univariate mini-maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from scipy.spatial import Voronoi
from pubstyle import set_pub_style, savefig

# Stevens-style bivariate matrix: rows = variable B, cols = variable A
BIV = [["#e8e6e1", "#b9d5be", "#74ac80"],
       ["#b3bfd6", "#8fb0b0", "#5b9077"],
       ["#6b82b2", "#567991", "#2c5b59"]]
CMAP_A = LinearSegmentedColormap.from_list("a", ["#e8e6e1", "#74ac80"])
CMAP_B = LinearSegmentedColormap.from_list("b", ["#e8e6e1", "#6b82b2"])


# ---------------------------------------------------------------- geometry
def land_radius(theta):
    return (1.0 + 0.15 * np.cos(2 * theta - 0.4)
            + 0.09 * np.cos(3 * theta + 1.6)
            + 0.045 * np.cos(5 * theta - 0.9))


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


def make_regions(n=150, seed=11, relax=5):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1.2, 1.2, size=2)
        if np.hypot(*p) < 0.95 * land_radius(np.arctan2(p[1], p[0])):
            pts.append(p)
    pts = np.array(pts)
    for _ in range(relax):
        cells = [clip_to_land(densify(c, 4)) for c in _cells(pts)]
        pts = np.array([poly_centroid(c) for c in cells])
    polys = [clip_to_land(densify(c, 7)) for c in _cells(pts)]
    return np.array([poly_centroid(p) for p in polys]), polys


def gauss_field(cent, bumps, rng, noise=0.12):
    v = np.zeros(len(cent))
    for bx, by, s, w in bumps:
        v += w * np.exp(-(((cent[:, 0] - bx) ** 2 +
                           (cent[:, 1] - by) ** 2) / (2 * s ** 2)))
    v += rng.normal(0, noise, len(cent))
    return (v - v.min()) / np.ptp(v)


# ------------------------------------------------------------------- plot
def main():
    set_pub_style()
    cent, polys = make_regions()
    t = np.linspace(0, 2 * np.pi, 500)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]

    rng = np.random.default_rng(4)
    a = gauss_field(cent, [(-0.45, 0.30, 0.45, 1.0), (0.55, -0.35, 0.40, 0.9),
                           (0.25, 0.70, 0.28, 0.6)], rng)
    b = gauss_field(cent, [(-0.30, 0.15, 0.55, 1.0), (0.70, -0.10, 0.35, 0.8),
                           (-0.55, -0.65, 0.32, 0.7)], rng)
    a = 100 * a
    b = -1.0 + 5.5 * b

    qa = np.quantile(a, [1 / 3, 2 / 3])
    qb = np.quantile(b, [1 / 3, 2 / 3])
    ia = np.digitize(a, qa)
    ib = np.digitize(b, qb)
    face = [BIV[ib[i]][ia[i]] for i in range(len(a))]

    fig = plt.figure(figsize=(10.6, 6.5))
    gs = GridSpec(3, 2, width_ratios=[2.25, 1], height_ratios=[1.02, 1, 0.98],
                  left=0.015, right=0.965, top=0.885, bottom=0.055,
                  wspace=0.06, hspace=0.66)

    # ---- main bivariate map
    ax = fig.add_subplot(gs[:, 0])
    ax.add_collection(PolyCollection(polys, facecolors=face,
                                     edgecolors="white", linewidths=0.35))
    ax.plot(coast[:, 0], coast[:, 1], color="#3f3f3f", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(coast[:, 0].min() - 0.06, coast[:, 0].max() + 0.06)
    ax.set_ylim(coast[:, 1].min() - 0.06, coast[:, 1].max() + 0.10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.0, 1.005, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(0.055, 1.008, "Bivariate exposure classes", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="bottom", color="#2c5b59")

    hi = np.where((ia == 2) & (ib == 2))[0]
    if len(hi):
        c0 = cent[hi].mean(axis=0)
        ax.annotate("dual-high cluster\n(urban $\\times$ heat)",
                    xy=(c0[0], c0[1]),
                    xytext=(c0[0] - 0.22, c0[1] + 0.62), fontsize=7.2,
                    fontweight="bold", color="#1f3f3e", ha="center",
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec="#b8c4c2", lw=0.5, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color="#1f3f3e", lw=0.8,
                                    shrinkA=2, shrinkB=2))
    ax.annotate("N", xy=(0.045, 0.925), xytext=(0.045, 0.835),
                xycoords="axes fraction", ha="center", fontsize=7, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.plot([0.76, 0.94], [0.055, 0.055], transform=ax.transAxes,
            color="#333", lw=1.8, solid_capstyle="butt")
    ax.text(0.85, 0.075, "250 km", transform=ax.transAxes, fontsize=6,
            ha="center", color="#333")

    # ---- 3x3 legend
    axl = fig.add_subplot(gs[0, 1])
    img = np.array([[to_rgb(BIV[j][i]) for i in range(3)] for j in range(3)])
    axl.imshow(img, origin="lower", extent=(0, 3, 0, 3), interpolation="nearest")
    for k in range(1, 3):
        axl.axhline(k, color="white", lw=1.1)
        axl.axvline(k, color="white", lw=1.1)
    axl.set_xticks([0, 1, 2, 3])
    axl.set_yticks([0, 1, 2, 3])
    axl.set_xticklabels(["", "%.0f" % qa[0], "%.0f" % qa[1], ""], fontsize=6.2)
    axl.set_yticklabels(["", "%.1f" % qb[0], "%.1f" % qb[1], ""], fontsize=6.2)
    axl.tick_params(length=2, pad=1.5)
    for s in axl.spines.values():
        s.set_visible(False)
    axl.set_aspect("equal")
    axl.annotate("", xy=(3.35, 0), xytext=(0, 0), xycoords="data",
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color="#4a4a4a", lw=0.9))
    axl.annotate("", xy=(0, 3.35), xytext=(0, 0), xycoords="data",
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color="#4a4a4a", lw=0.9))
    axl.text(1.5, -0.75, "Urban fraction (%) $\\rightarrow$", ha="center",
             fontsize=7, color="#333")
    axl.text(-0.85, 1.5, "Heat anomaly (K) $\\rightarrow$", va="center",
             rotation=90, fontsize=7, color="#333")
    axl.text(-0.85, 3.45, "(b)  Bivariate legend", fontsize=9,
             fontweight="bold", va="bottom")

    # ---- scatter with tercile grid
    axs = fig.add_subplot(gs[1, 1])
    axs.scatter(a, b, s=13, c=face, edgecolor="#5a5a5a", linewidth=0.25,
                zorder=3)
    for q in qa:
        axs.axvline(q, color="#8a8a8a", lw=0.6, ls=(0, (3, 2)), zorder=1)
    for q in qb:
        axs.axhline(q, color="#8a8a8a", lw=0.6, ls=(0, (3, 2)), zorder=1)
    r = np.corrcoef(a, b)[0, 1]
    axs.text(0.97, 0.06, "r = %.2f" % r, transform=axs.transAxes, ha="right",
             fontsize=7.5, fontweight="bold", color="#2c5b59")
    axs.set_xlabel("Urban fraction (%)", fontsize=7.5, labelpad=1.5)
    axs.set_ylabel("Heat anomaly (K)", fontsize=7.5, labelpad=1.5)
    axs.tick_params(labelsize=6.5, length=2.5)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    axs.text(0.0, 1.04, "(c)  Joint distribution", transform=axs.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    # ---- univariate mini maps
    sub = gs[2, 1].subgridspec(1, 2, wspace=0.10)
    for j, (val, cm, lab) in enumerate([(a, CMAP_A, "Urban fraction"),
                                        (b, CMAP_B, "Heat anomaly")]):
        axm = fig.add_subplot(sub[0, j])
        n = (val - val.min()) / np.ptp(val)
        axm.add_collection(PolyCollection(polys, facecolors=cm(n),
                                          edgecolors="white", linewidths=0.15))
        axm.plot(coast[:, 0], coast[:, 1], color="#3f3f3f", lw=0.7)
        axm.set_aspect("equal")
        axm.set_xlim(coast[:, 0].min() - 0.05, coast[:, 0].max() + 0.05)
        axm.set_ylim(coast[:, 1].min() - 0.05, coast[:, 1].max() + 0.05)
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_visible(False)
        axm.set_title(lab, fontsize=7.2, color="#333", pad=2)
        if j == 0:
            axm.text(0.0, 1.14, "(d)  Univariate components",
                     transform=axm.transAxes, fontsize=9, fontweight="bold",
                     va="bottom")

    savefig(fig, "56_bivariate_map.png")


if __name__ == "__main__":
    main()
