"""Delaunay triangulation and its Voronoi dual drawn from one point set.

Main panel: the same 56 sites carry both structures.  scipy's Delaunay gives
the triangle edges, scipy's Voronoi gives the cell walls, and the two are
plotted on one equal-aspect frame so the duality is visible - every Voronoi
vertex is the circumcentre of a Delaunay triangle, and three highlighted
circumcircles are drawn to show that no site falls inside them.

Auxiliary (both at 58 % of the main panel width): the minimum interior angle
of every triangle, for the Delaunay triangulation and for a triangulation
obtained from it by flipping one independent set of interior diagonals, which
is the empirical form of the max-min angle property; and the number of
Delaunay edges incident on each interior site, whose mean is pinned near the
value six that Euler's formula forces on a large planar triangulation.

Illustrative geometry, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.spatial import Delaunay, Voronoi

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
DEEP = "#134a76"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.02, 5.54, 4.44)
AUX_T_RECT = (7.62, 3.56, 3.16, 1.90)
AUX_B_RECT = (7.62, 1.02, 3.16, 1.90)

SEED = 20260813
NX, NY = 8, 7
BOX = (0.0, 6.0, 0.0, 6.0 * MAIN_RECT[3] / MAIN_RECT[2])


def sample_sites(rng):
    """Jittered lattice: locally even, globally irregular, no duplicates."""
    x0, x1, y0, y1 = BOX
    pad = 0.42
    gx = np.linspace(x0 + pad, x1 - pad, NX)
    gy = np.linspace(y0 + pad, y1 - pad, NY)
    gxx, gyy = np.meshgrid(gx, gy)
    dx = (gx[1] - gx[0]) * 0.30
    dy = (gy[1] - gy[0]) * 0.30
    pts = np.column_stack([gxx.ravel(), gyy.ravel()])
    pts = pts + rng.uniform(-1.0, 1.0, pts.shape) * np.array([dx, dy])
    return pts


def triangle_angles(pts, tris):
    """Interior angles (deg) of every triangle, shape (n_tri, 3)."""
    a = pts[tris[:, 0]]
    b = pts[tris[:, 1]]
    c = pts[tris[:, 2]]
    out = np.empty((len(tris), 3))
    for k, (p, q, r) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
        u = q - p
        v = r - p
        cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1)
                                       * np.linalg.norm(v, axis=1))
        out[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def circumcircle(p, q, r):
    """Circumcentre and radius of one triangle."""
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(ctr - np.asarray(p)))


def segments_cross(p1, p2, p3, p4):
    """True when open segments p1p2 and p3p4 properly intersect."""
    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1])
                       - (b[1] - a[1]) * (c[0] - a[0]))
    return (side(p1, p2, p3) * side(p1, p2, p4) < 0
            and side(p3, p4, p1) * side(p3, p4, p2) < 0)


def flip_once(pts, tri):
    """Flip an independent set of interior diagonals; returns new triangles."""
    simp = tri.simplices
    nbr = tri.neighbors
    used = np.zeros(len(simp), dtype=bool)
    out = []
    n_flip = 0
    for t in range(len(simp)):
        if used[t]:
            continue
        for k in range(3):
            s = int(nbr[t, k])
            if s < 0 or used[s] or used[t]:
                continue
            p = int(simp[t, k])
            a = int(simp[t, (k + 1) % 3])
            b = int(simp[t, (k + 2) % 3])
            q = int([v for v in simp[s] if v not in (a, b)][0])
            if segments_cross(pts[a], pts[b], pts[p], pts[q]):
                out.append([p, a, q])
                out.append([p, q, b])
                used[t] = True
                used[s] = True
                n_flip += 1
                break
    for t in range(len(simp)):
        if not used[t]:
            out.append(list(simp[t]))
    return np.asarray(out, dtype=int), n_flip


def clip_segment(p, q, box):
    """Liang-Barsky clip of segment pq to an axis-aligned box."""
    x0, x1, y0, y1 = box
    d = q - p
    t_lo, t_hi = 0.0, 1.0
    for pi, qi in ((-d[0], p[0] - x0), (d[0], x1 - p[0]),
                   (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            t_lo = max(t_lo, t)
        else:
            t_hi = min(t_hi, t)
        if t_lo > t_hi:
            return None
    return p + t_lo * d, p + t_hi * d


def voronoi_segments(vor, pts, box):
    """Finite and clipped-infinite Voronoi ridges inside ``box``."""
    centre = pts.mean(axis=0)
    span = max(box[1] - box[0], box[3] - box[2]) * 4.0
    segs = []
    for (i, j), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if v1 >= 0 and v2 >= 0:
            a, b = vor.vertices[v1], vor.vertices[v2]
        else:
            k = v1 if v1 >= 0 else v2
            t = pts[j] - pts[i]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            mid = 0.5 * (pts[i] + pts[j])
            n = n * np.sign(np.dot(mid - centre, n))
            a, b = vor.vertices[k], vor.vertices[k] + n * span
        clipped = clip_segment(np.asarray(a, float), np.asarray(b, float), box)
        if clipped is not None:
            segs.append(clipped)
    return segs


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    pts = sample_sites(rng)
    tri = Delaunay(pts)
    vor = Voronoi(pts)
    simp = tri.simplices
    n_pts = len(pts)
    n_tri = len(simp)

    edges = set()
    for s in simp:
        for u, v in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            edges.add((min(u, v), max(u, v)))
    edges = np.array(sorted(edges))
    n_edge = len(edges)

    hull = np.unique(tri.convex_hull)
    interior = np.setdiff1d(np.arange(n_pts), hull)
    deg = np.zeros(n_pts, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    deg_int = deg[interior]

    ang_del = triangle_angles(pts, simp)
    flipped, n_flip = flip_once(pts, tri)
    ang_flip = triangle_angles(pts, flipped)
    min_del = ang_del.min(axis=1)
    min_flip = ang_flip.min(axis=1)
    q10_del = float(np.percentile(min_del, 10.0))
    q10_flip = float(np.percentile(min_flip, 10.0))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the two dual structures on one frame ------------------------
    ax.set_aspect("equal")
    vsegs = voronoi_segments(vor, pts, BOX)
    ax.add_collection(LineCollection(vsegs, colors=RUST, linewidths=1.05,
                                     zorder=3, alpha=0.95))
    ax.add_collection(LineCollection([pts[e] for e in edges], colors=BLUE,
                                     linewidths=0.75, zorder=4, alpha=0.85))

    # three circumcircles: Voronoi vertices that are provably empty
    order = np.argsort(np.linalg.norm(pts[simp].mean(axis=1)
                                      - np.array([3.0, 2.5]), axis=1))
    picks, radii = [], []
    for idx in order:
        ctr, rad = circumcircle(*pts[simp[idx]])
        if not (BOX[0] + rad < ctr[0] < BOX[1] - rad
                and BOX[2] + rad < ctr[1] < BOX[3] - rad):
            continue
        if any(np.linalg.norm(ctr - c) < 1.5 for c, _ in picks):
            continue
        picks.append((ctr, rad))
        radii.append(rad)
        if len(picks) == 3:
            break

    d_all = np.linalg.norm(pts[None, :, :] - np.array([c for c, _ in picks])
                           [:, None, :], axis=2)
    slack = float((d_all / np.array(radii)[:, None]).min())

    for n, (ctr, rad) in enumerate(picks):
        ax.add_patch(Circle(tuple(ctr), rad, facecolor=lighten(ACCENT, 0.90),
                            edgecolor=ACCENT, lw=1.15, ls=(0, (4.0, 2.2)),
                            alpha=0.85, zorder=2))
        ax.plot([ctr[0]], [ctr[1]], marker="o", ms=4.4, mfc="white",
                mec=ACCENT, mew=1.2, zorder=8)
        if n == 0:
            ax.annotate("circumcentre = Voronoi vertex\n(circle interior "
                        "holds no site)",
                        xy=tuple(ctr), xytext=(ctr[0] + 0.30, ctr[1] + rad
                                               + 0.28),
                        fontsize=6.4, color=ACCENT, ha="left", va="bottom",
                        linespacing=1.4,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.82, pad=1.4),
                        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))

    ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", ms=4.2, mfc=DEEP,
            mec="white", mew=0.8, zorder=9)

    ax.set_xlim(BOX[0], BOX[1])
    ax.set_ylim(BOX[2], BOX[3])
    ax.set_xticks(np.arange(0, 6.1, 1.0))
    ax.set_yticks(np.arange(0, BOX[3] + 0.01, 1.0))
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless domain units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless domain units)", fontsize=8.5)
    ax.set_title("One point set, two dual structures: Delaunay triangles and "
                 "Voronoi cells", fontsize=9.0, fontweight="bold", pad=5.0)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc=DEEP, mec="white",
               mew=0.8, label=f"{n_pts} sites (vertices)"),
        Line2D([], [], color=BLUE, lw=0.9,
               label=f"Delaunay edges: {n_edge}, triangles: {n_tri}"),
        Line2D([], [], color=RUST, lw=1.05,
               label=f"Voronoi cell walls, {n_pts} cells"),
        Line2D([], [], color=ACCENT, lw=1.15, ls=(0, (4.0, 2.2)),
               label=f"3 circumcircles, closest site at "
                     f"{slack:.2f} x radius"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                    fontsize=6.5, handlelength=1.6, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.92, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.004, 0.004))
    leg.get_frame().set_linewidth(0.6)

    # ---- aux (b): the max-min angle property, measured ---------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    bins = np.arange(0.0, 66.0, 4.0)
    n_hist, _, _ = ax_t.hist(
        min_flip, bins=bins, color=lighten(RUST, 0.45), edgecolor="white",
        linewidth=0.6, zorder=3,
        label=f"after {n_flip} diagonal flips: mean {min_flip.mean():.1f} deg,"
              f" 10th pct {q10_flip:.1f} deg")
    ax_t.hist(min_del, bins=bins, histtype="step", color=BLUE, linewidth=1.5,
              zorder=5,
              label=f"Delaunay: mean {min_del.mean():.1f} deg, 10th pct "
                    f"{q10_del:.1f} deg")
    ax_t.axvline(q10_del, color=BLUE, lw=0.9, ls=(0, (3.0, 2.2)), zorder=6)
    ax_t.axvline(q10_flip, color=RUST, lw=0.9, ls=(0, (3.0, 2.2)), zorder=6)
    ax_t.set_ylim(0.0, float(n_hist.max()) * 1.42)
    ax_t.set_xlim(0.0, 64.0)
    ax_t.set_xticks([0, 15, 30, 45, 60])
    ax_t.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("minimum interior angle per triangle (deg)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("triangles (count)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Delaunay maximises the smallest angle", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper left", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.26, borderpad=0.10,
                bbox_to_anchor=(-0.012, 1.035))

    # ---- aux (c): incident-edge count at interior sites --------------------
    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    vals = np.arange(deg_int.min(), deg_int.max() + 1)
    counts = np.array([(deg_int == v).sum() for v in vals])
    ax_b.bar(vals, counts, width=0.66, color=lighten(BLUE, 0.30),
             edgecolor="white", linewidth=0.7, zorder=3)
    ax_b.axvline(deg_int.mean(), color=ACCENT, lw=1.1, zorder=5)
    ax_b.text(0.985, 0.97, f"mean {deg_int.mean():.2f} edges\nper interior "
              f"site", transform=ax_b.transAxes, ha="right", va="top",
              fontsize=6.1, color=ACCENT, linespacing=1.35)
    for v, c in zip(vals, counts):
        if c:
            ax_b.text(v, c + counts.max() * 0.035, f"{c}", ha="center",
                      va="bottom", fontsize=6.0, color=SUB)
    ax_b.set_xticks(vals)
    ax_b.set_xlim(vals[0] - 0.7, vals[-1] + 0.7)
    ax_b.set_ylim(0.0, counts.max() * 1.30)
    ax_b.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("incident Delaunay edges per site (count)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel(f"interior sites ({len(interior)} total)", fontsize=7.6,
                    labelpad=2.0)
    ax_b.set_title("Vertex valence clusters on six", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "972_cgx_delaunay_voronoi.png")


if __name__ == "__main__":
    main()
