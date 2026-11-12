"""Convex hull of a planar cloud, with the two quantities calipers extract.

Main panel: 260 sample points, their convex hull from scipy.spatial.ConvexHull,
the farthest pair of hull vertices (the set diameter) and the minimum-area
enclosing rectangle.  The rectangle is found by rotating the cloud into the
frame of every hull edge in turn and keeping the cheapest axis-aligned box, so
its side that lies flush against the hull is a real hull edge rather than a
drawn one.

Auxiliary (both at 58 % of the main panel width): the enclosing-box area swept
over all orientations from 0 to 180 degrees, with the hull-edge orientations
marked on the curve to show that the optimum is attained at one of them; and
how many vertices the hull keeps as the sample grows, for points drawn in a
square and in a disc, which separate into two visibly different growth laws.

Illustrative geometry, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

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

SEED = 481207
N_PTS = 260
ASPECT = MAIN_RECT[2] / MAIN_RECT[3]


def sample_cloud(rng, n=N_PTS):
    """Three overlapping anisotropic blobs; convex but not circular."""
    specs = ((0.34, [-0.9, -0.35], [[1.05, 0.42], [0.42, 0.34]]),
             (0.24, [0.85, 0.45], [[0.72, -0.30], [-0.30, 0.52]]),
             (0.14, [0.10, 1.05], [[0.46, 0.05], [0.05, 0.30]]))
    parts = [rng.multivariate_normal(mu, cov, size=int(round(frac * n)))
             for frac, mu, cov in specs]
    # a uniform-in-ellipse component so the hull is a polygon, not a triangle
    k = n - sum(len(p) for p in parts)
    r = np.sqrt(rng.uniform(0.0, 1.0, size=k))
    th = rng.uniform(0.0, 2 * np.pi, size=k)
    parts.append(np.column_stack([2.05 * r * np.cos(th),
                                  1.35 * r * np.sin(th)]))
    return np.vstack(parts)[:n]


def box_for_angle(pts, ang):
    """Extent of ``pts`` in a frame rotated by ``ang`` (rad): w, h, area."""
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    q = pts @ rot.T
    w = q[:, 0].max() - q[:, 0].min()
    h = q[:, 1].max() - q[:, 1].min()
    return w, h, w * h


def box_corners(pts, ang):
    """The four corners of the enclosing box at orientation ``ang``."""
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    q = pts @ rot.T
    x0, x1 = q[:, 0].min(), q[:, 0].max()
    y0, y1 = q[:, 1].min(), q[:, 1].max()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return corners @ np.linalg.inv(rot).T


def hull_growth(rng, ns, trials, shape):
    """Mean hull-vertex count against sample size, measured not asserted."""
    out = np.empty(len(ns))
    for i, n in enumerate(ns):
        k = np.empty(trials)
        for t in range(trials):
            if shape == "square":
                p = rng.uniform(-1.0, 1.0, size=(n, 2))
            else:
                r = np.sqrt(rng.uniform(0.0, 1.0, size=n))
                th = rng.uniform(0.0, 2 * np.pi, size=n)
                p = np.column_stack([r * np.cos(th), r * np.sin(th)])
            k[t] = len(ConvexHull(p).vertices)
        out[i] = k.mean()
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    pts = sample_cloud(rng)
    hull = ConvexHull(pts)
    hv = hull.vertices
    poly = pts[hv]
    n_hull = len(hv)

    # diameter: farthest pair, searched over hull vertices only
    d = np.linalg.norm(poly[:, None, :] - poly[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    diam = float(d[i, j])

    # rotating calipers: one candidate orientation per hull edge
    edge_ang = np.arctan2(np.roll(poly[:, 1], -1) - poly[:, 1],
                          np.roll(poly[:, 0], -1) - poly[:, 0]) % np.pi
    cand = np.array([box_for_angle(poly, a) for a in edge_ang])
    best = int(np.argmin(cand[:, 2]))
    a_best = float(edge_ang[best])
    w_best, h_best, area_best = cand[best]
    corners = box_corners(poly, a_best)

    # dense sweep for the auxiliary curve
    sweep = np.linspace(0.0, np.pi, 721)
    area_sweep = np.array([box_for_angle(poly, a)[2] for a in sweep])
    width_sweep = np.array([min(box_for_angle(poly, a)[:2]) for a in sweep])
    w_min = float(width_sweep.min())
    hull_area = float(hull.volume)
    fill = 100.0 * hull_area / area_best

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: cloud, hull, diameter, minimum-area rectangle ---------------
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5, alpha=0.5, zorder=0)

    ax.add_patch(Polygon(corners, closed=True,
                         facecolor=lighten(RUST, 0.90), edgecolor=RUST,
                         lw=1.4, zorder=2))
    ax.add_patch(Polygon(poly, closed=True, facecolor=lighten(BLUE, 0.82),
                         edgecolor=BLUE, lw=1.6, alpha=0.85, zorder=3))

    inside = np.ones(len(pts), dtype=bool)
    inside[hv] = False
    ax.plot(pts[inside, 0], pts[inside, 1], ls="none", marker="o", ms=2.9,
            mfc="#8fa9c4", mec="none", alpha=0.85, zorder=4)
    ax.plot(poly[:, 0], poly[:, 1], ls="none", marker="o", ms=5.0, mfc=DEEP,
            mec="white", mew=0.8, zorder=6)

    ax.plot(poly[[i, j], 0], poly[[i, j], 1], color=ACCENT, lw=1.6, zorder=7)
    for k in (i, j):
        ax.plot([poly[k, 0]], [poly[k, 1]], marker="o", ms=6.4, mfc=ACCENT,
                mec="white", mew=1.0, zorder=8)
    mid = 0.5 * (poly[i] + poly[j])
    rot_d = np.degrees(np.arctan2(poly[j, 1] - poly[i, 1],
                                  poly[j, 0] - poly[i, 0]))
    rot_d = (rot_d + 90.0) % 180.0 - 90.0
    ax.text(mid[0], mid[1] + 0.10, f"diameter {diam:.2f} units",
            ha="center", va="bottom", fontsize=6.6, color=ACCENT,
            rotation=rot_d,
            rotation_mode="anchor", zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      pad=1.0))

    # side labels on the optimal rectangle
    for k in range(4):
        e0 = corners[(k + 1) % 4] - corners[k]
        lab = corners[k] + 0.5 * e0
        out = lab - corners.mean(axis=0)
        out = 0.10 * out / np.linalg.norm(out)
        rot_e = np.degrees(np.arctan2(e0[1], e0[0]))
        rot_e = (rot_e + 90.0) % 180.0 - 90.0
        ax.text(lab[0] + out[0], lab[1] + out[1], f"{np.hypot(*e0):.2f} units",
                ha="center", va="center", fontsize=6.3, color=RUST,
                rotation=rot_e, rotation_mode="anchor", zorder=9)

    allpts = np.vstack([poly, corners])
    xc = 0.5 * (allpts[:, 0].min() + allpts[:, 0].max())
    yc = 0.5 * (allpts[:, 1].min() + allpts[:, 1].max())
    half_y = 0.5 * max(allpts[:, 1].max() - allpts[:, 1].min(),
                       (allpts[:, 0].max() - allpts[:, 0].min()) / ASPECT)
    half_y *= 1.22
    half_x = half_y * ASPECT
    yc = yc + 0.13 * half_y          # leave the legend a clear top-left corner
    ax.set_xlim(xc - half_x, xc + half_x)
    ax.set_ylim(yc - half_y, yc + half_y)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless length units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless length units)", fontsize=8.5)
    ax.set_title("Convex hull, farthest pair and minimum-area enclosing "
                 "rectangle", fontsize=9.0, fontweight="bold", pad=5.0)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc="#8fa9c4",
               mec="none", label=f"{N_PTS} sample points, "
                                 f"{N_PTS - n_hull} interior"),
        Line2D([], [], color=BLUE, lw=1.6,
               label=f"convex hull: {n_hull} vertices, area "
                     f"{hull_area:.2f} units^2"),
        Line2D([], [], color=ACCENT, lw=1.6,
               label=f"diameter (farthest pair) {diam:.2f} units"),
        Line2D([], [], color=RUST, lw=1.4,
               label=f"min-area box {w_best:.2f} x {h_best:.2f} units at "
                     f"{np.degrees(a_best):.1f} deg"),
        Line2D([], [], ls="none",
               label=f"hull fills {fill:.0f} % of that box"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    fontsize=6.5, handlelength=1.6, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.92, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.004, 0.996))
    leg.get_frame().set_linewidth(0.6)

    # ---- aux (b): the caliper objective over orientation -------------------
    deg = np.degrees(sweep)
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.fill_between(deg, area_best, area_sweep, facecolor="#eef1f6",
                      edgecolor="none", zorder=1)
    ax_t.plot(deg, area_sweep, color=BLUE, lw=1.4, zorder=4,
              label="enclosing-box area over orientation")
    ax_t.plot(np.degrees(edge_ang), cand[:, 2], ls="none", marker="o", ms=3.0,
              mfc=RUST, mec="white", mew=0.5, zorder=5,
              label=f"{n_hull} hull-edge orientations")
    ax_t.plot([np.degrees(a_best)], [area_best], marker="*", ms=9.0,
              mfc=ACCENT, mec="white", mew=0.7, ls="none", zorder=6,
              label=f"optimum {area_best:.2f} units^2 at "
                    f"{np.degrees(a_best):.1f} deg")
    ax_t.axhline(area_best, color=ACCENT, lw=0.8, ls=(0, (3.0, 2.2)),
                 zorder=3)
    ax_t.set_xlim(0.0, 180.0)
    ax_t.set_ylim(area_best * 0.90, area_sweep.max() * 1.30)
    ax_t.set_xticks([0, 45, 90, 135, 180])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("box orientation (deg)", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("enclosing area (units^2)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("The optimum sits on a hull edge", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper right", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.26, borderpad=0.10,
                bbox_to_anchor=(1.015, 1.045))

    # ---- aux (c): how the hull grows with the sample -----------------------
    ns = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    k_sq = hull_growth(rng, ns, 14, "square")
    k_di = hull_growth(rng, ns, 14, "disc")
    ax_b.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.plot(ns, k_sq, color=BLUE, lw=1.3, marker="o", ms=3.0, mfc="white",
              mew=0.9, zorder=4, label="uniform in a square")
    ax_b.plot(ns, k_di, color=TEAL, lw=1.3, marker="s", ms=3.0, mfc="white",
              mew=0.9, zorder=4, label="uniform in a disc")
    c_sq = k_sq[-1] / np.log(ns[-1])
    c_di = k_di[-1] / ns[-1] ** (1.0 / 3.0)
    ax_b.plot(ns, c_sq * np.log(ns), color=BLUE, lw=0.8, ls=(0, (3.0, 2.2)),
              zorder=3, label="fitted log n")
    ax_b.plot(ns, c_di * ns ** (1.0 / 3.0), color=TEAL, lw=0.8,
              ls=(0, (3.0, 2.2)), zorder=3, label="fitted n^(1/3)")
    ax_b.plot([N_PTS], [n_hull], marker="*", ms=9.0, mfc=ACCENT, mec="white",
              mew=0.7, ls="none", zorder=6,
              label=f"panel (a): {n_hull} of {N_PTS}")
    ax_b.set_xscale("log")
    ax_b.set_xlim(13.0, 11000.0)
    ax_b.set_ylim(0.0, max(k_sq.max(), k_di.max()) * 1.42)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("sample size (points)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("hull vertices (count)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Hull size grows with the shape of the support",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.0, ncol=2,
                handlelength=1.3, labelspacing=0.24, columnspacing=1.0,
                borderpad=0.10, bbox_to_anchor=(-0.012, 1.045))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "973_cgx_convex_hull_calipers.png")


if __name__ == "__main__":
    main()
