"""3x2 grid of spatial clustering solutions with cluster-share bars.

Regions carry four spatially structured attributes; each panel maps a
different partition (k-means at several k, Ward linkage, Gaussian mixture)
using a shared discrete palette, with a 100% stacked bar underneath giving
the share of area in each cluster and a silhouette score badge on the map.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.spatial import Voronoi
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from pubstyle import set_pub_style, savefig

CLUSTER_COLORS = ["#3d6b8e", "#c98b3f", "#5c8f6b", "#a4536b", "#7a6aa8",
                  "#8c8577"]


def land_radius(theta):
    return (1.0 + 0.14 * np.cos(2 * theta - 1.5)
            + 0.10 * np.cos(3 * theta + 0.8)
            + 0.045 * np.cos(5 * theta + 2.6))


def clip_to_land(pts):
    d = np.asarray(pts, float)
    r = np.hypot(d[:, 0], d[:, 1])
    s = np.minimum(1.0, land_radius(np.arctan2(d[:, 1], d[:, 0]))
                   / np.maximum(r, 1e-9))
    return d * s[:, None]


def densify(poly, n=6):
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


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0


def _cells(pts):
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    vor = Voronoi(np.vstack([pts, 5.0 * np.c_[np.cos(t), np.sin(t)]]))
    return [vor.vertices[vor.regions[vor.point_region[i]]]
            for i in range(len(pts))]


def make_regions(n=120, seed=3, relax=5):
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
    polys = [clip_to_land(densify(c, 7)) for c in _cells(pts)]
    return np.array([poly_centroid(p) for p in polys]), polys


def features(cent, rng):
    specs = [[(-0.55, 0.45, 0.40, 1.0), (0.60, -0.30, 0.35, -0.8)],
             [(0.35, 0.60, 0.35, 1.0), (-0.45, -0.55, 0.40, 0.9)],
             [(0.70, 0.20, 0.30, 1.0), (-0.20, -0.10, 0.55, -0.7)],
             [(-0.70, -0.10, 0.35, 1.0), (0.25, -0.65, 0.32, 0.8)]]
    cols = []
    for sp in specs:
        v = np.zeros(len(cent))
        for bx, by, s, w in sp:
            v += w * np.exp(-(((cent[:, 0] - bx) ** 2 +
                               (cent[:, 1] - by) ** 2) / (2 * s ** 2)))
        v += rng.normal(0, 0.09, len(cent))
        cols.append((v - v.mean()) / v.std())
    return np.column_stack(cols)


def relabel(lab, X):
    """Reorder cluster ids by mean of the first feature for stable colours."""
    order = np.argsort([X[lab == c, 0].mean() for c in np.unique(lab)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[v] for v in lab])


def main():
    set_pub_style()
    cent, polys = make_regions()
    area = np.array([poly_area(p) for p in polys])
    t = np.linspace(0, 2 * np.pi, 400)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]
    X = features(cent, np.random.default_rng(9))
    Xs = np.column_stack([X, 0.9 * cent])       # mild spatial constraint

    runs = [("k-means", 3), ("k-means", 4), ("k-means", 5),
            ("Ward linkage", 5), ("Gaussian mixture", 5), ("k-means", 6)]

    fig = plt.figure(figsize=(10.8, 7.6))
    gs = GridSpec(2, 3, left=0.02, right=0.98, top=0.90, bottom=0.10,
                  wspace=0.06, hspace=0.20)

    for idx, (method, k) in enumerate(runs):
        r, c = divmod(idx, 3)
        sub = gs[r, c].subgridspec(2, 1, height_ratios=[1, 0.16], hspace=0.06)
        axm = fig.add_subplot(sub[0])
        axb = fig.add_subplot(sub[1])

        if method == "k-means":
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
        elif method == "Ward linkage":
            lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Xs)
        else:
            lab = GaussianMixture(k, covariance_type="full",
                                  random_state=0).fit_predict(Xs)
        lab = relabel(lab, Xs)
        sil = silhouette_score(Xs, lab)

        axm.add_collection(PolyCollection(
            polys, facecolors=[CLUSTER_COLORS[v] for v in lab],
            edgecolors="white", linewidths=0.3))
        axm.plot(coast[:, 0], coast[:, 1], color="#3a3a3a", lw=0.9)
        axm.set_aspect("equal")
        axm.set_xlim(coast[:, 0].min() - 0.06, coast[:, 0].max() + 0.06)
        axm.set_ylim(coast[:, 1].min() - 0.06, coast[:, 1].max() + 0.06)
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_visible(False)
        axm.text(0.0, 1.0, "(%s)" % "abcdef"[idx], transform=axm.transAxes,
                 fontsize=10, fontweight="bold", va="bottom")
        axm.text(0.108, 1.004, "%s, k = %d" % (method, k),
                 transform=axm.transAxes, fontsize=8.8, fontweight="bold",
                 va="bottom", color="#333")
        axm.text(0.99, 0.98, "silhouette\n%.2f" % sil, transform=axm.transAxes,
                 fontsize=6.8, ha="right", va="top", color="#666",
                 linespacing=1.35)

        share = np.array([area[lab == j].sum() for j in range(k)])
        share = 100 * share / share.sum()
        left = 0.0
        for j in range(k):
            axb.barh([0], [share[j]], left=left, height=0.62,
                     color=CLUSTER_COLORS[j], edgecolor="white", linewidth=0.7)
            if share[j] >= 9:
                axb.text(left + share[j] / 2, 0, "%.0f%%" % share[j],
                         ha="center", va="center", fontsize=6.2,
                         color="white", fontweight="bold")
            left += share[j]
        axb.set_xlim(0, 100); axb.set_ylim(-0.5, 0.5)
        axb.set_xticks([]); axb.set_yticks([])
        for sp in axb.spines.values():
            sp.set_visible(False)
        axb.text(0.0, 1.06, "share of mapped area (%)",
                 transform=axb.transAxes, fontsize=6.2, color="#777",
                 va="bottom")

    handles = [Patch(facecolor=CLUSTER_COLORS[j], edgecolor="white",
                     label="cluster %d" % (j + 1)) for j in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.035), frameon=False, handlelength=1.4,
               columnspacing=1.6, handleheight=0.9)
    savefig(fig, "61_cluster_map_grid.png")


if __name__ == "__main__":
    main()
