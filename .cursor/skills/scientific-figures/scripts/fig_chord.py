"""Chord diagram: land-cover transition flows as bezier ribbons.

Pure-matplotlib chord diagram (no external chord library).  Arcs around the
circle are sized by the total area entering and leaving each class; ribbons
join the pairs and are coloured by the source class.  The transition matrix is
sparse and directional, the way an observed land-cover change matrix is.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pubstyle import set_pub_style, savefig

NODES = ["Cropland", "Forest", "Grassland", "Wetland", "Water", "Urban"]
COLORS = ["#c9a227", "#1f6b4d", "#7cb342", "#4fa5a0", "#3d6b8e", "#a4503f"]

# rows = source, columns = target, units of 10^3 km^2 converted 1985-2020
FLOWS = np.array([
    [0, 12, 9, 1, 1, 26],       # from cropland
    [18, 0, 14, 2, 1, 9],       # from forest
    [22, 16, 0, 3, 1, 7],       # from grassland
    [2, 3, 2, 0, 5, 1],         # from wetland
    [1, 1, 1, 4, 0, 1],         # from water
    [1, 0, 0, 0, 0, 0],         # from urban
], dtype=float)


def bezier(p0, p1, p2, n=48):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def main():
    set_pub_style()
    n = len(NODES)
    totals = FLOWS.sum(axis=1) + FLOWS.sum(axis=0)
    gap = 0.05
    total_all = totals.sum()

    angles, start = [], np.pi / 2
    for i in range(n):
        span = totals[i] / total_all * (2 * np.pi - n * gap)
        angles.append((start, start - span))
        start -= span + gap

    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.set_aspect("equal")
    ax.axis("off")
    R = 1.0

    for i, (a0, a1) in enumerate(angles):
        th = np.linspace(a0, a1, 60)
        ax.plot(1.035 * np.cos(th), 1.035 * np.sin(th), color=COLORS[i],
                lw=7, solid_capstyle="butt", zorder=5)
        mid = (a0 + a1) / 2
        cx, sy = np.cos(mid), np.sin(mid)
        ha = "left" if cx > 0.15 else "right" if cx < -0.15 else "center"
        va = "center" if abs(sy) < 0.85 else ("bottom" if sy > 0 else "top")
        ax.text(1.10 * cx, 1.10 * sy, f"{NODES[i]}\n{totals[i]:.0f}",
                ha=ha, va=va, fontsize=8, fontweight="bold",
                color=COLORS[i], linespacing=1.3, zorder=8)

    offs = [a[0] for a in angles]
    spanper = [(angles[i][0] - angles[i][1]) / max(totals[i], 1e-9)
               for i in range(n)]
    pairs = [(FLOWS[i, j], i, j) for i in range(n) for j in range(n)
             if FLOWS[i, j] > 0]
    pairs.sort(reverse=True)                 # widest ribbons underneath

    for v, i, j in pairs:
        w_i, w_j = v * spanper[i], v * spanper[j]
        a_i0, a_i1 = offs[i], offs[i] - w_i
        offs[i] -= w_i
        a_j0, a_j1 = offs[j], offs[j] - w_j
        offs[j] -= w_j
        p = {k: R * np.array([np.cos(a), np.sin(a)]) for k, a in
             [("i0", a_i0), ("i1", a_i1), ("j0", a_j0), ("j1", a_j1)]}
        c1 = bezier(p["i1"], np.array([0.0, 0.0]), p["j0"])
        c2 = bezier(p["j1"], np.array([0.0, 0.0]), p["i0"])
        arc_j = np.linspace(a_j0, a_j1, 16)
        arc_i = np.linspace(a_i1, a_i0, 16)
        poly = np.vstack([
            c1,
            R * np.column_stack([np.cos(arc_j), np.sin(arc_j)]),
            c2,
            R * np.column_stack([np.cos(arc_i), np.sin(arc_i)]),
        ])
        ax.add_patch(patches.Polygon(poly, closed=True, facecolor=COLORS[i],
                                     alpha=0.42, lw=0.4, edgecolor="white",
                                     zorder=2))

    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.34, 1.34)
    ax.set_title("Land-cover transition flows, 1985-2020", fontsize=11,
                 fontweight="bold", pad=2)
    savefig(fig, "17_chord.png")


if __name__ == "__main__":
    main()
