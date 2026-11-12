"""A bounding volume hierarchy over a flat scene, and what one ray costs.

Main panel: 48 triangles, the axis-aligned boxes of every node of a median
split hierarchy shaded by depth, and one ray traced through it.  The boxes the
traversal actually opened are redrawn in the accent colour, the leaves it had
to test primitive by primitive are filled, and the first hit is marked, so the
picture and the counters in the legend come from the same traversal.

Auxiliary (both at 58 % of the main panel width): the number of primitive
tests 600 random rays need with the hierarchy, against the flat cost of
testing every triangle; and how the average cost splits between box tests and
primitive tests as the leaf size the builder stops at is varied, which is the
usual shallow optimum between a deep tree and a bushy one.

Illustrative geometry, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

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
GOLD = "#c9a227"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.02, 5.54, 4.44)
AUX_T_RECT = (7.62, 3.56, 3.16, 1.90)
AUX_B_RECT = (7.62, 1.02, 3.16, 1.90)
ASPECT = MAIN_RECT[2] / MAIN_RECT[3]

SEED = 90714
N_TRI = 48
HALF_Y = 2.30
HALF_X = HALF_Y * ASPECT
LEAF = 4
RAY_O = np.array([-HALF_X + 0.06, -1.62])
RAY_D = np.array([0.94, 0.34])
N_RAYS = 600


def make_scene(rng, n=N_TRI):
    """Randomly placed, randomly oriented small triangles."""
    tris = []
    for _ in range(n):
        c = np.array([rng.uniform(-HALF_X + 0.35, HALF_X - 0.35),
                      rng.uniform(-HALF_Y + 0.30, HALF_Y - 0.30)])
        r = rng.uniform(0.13, 0.30)
        a0 = rng.uniform(0.0, 2 * np.pi)
        ang = a0 + np.array([0.0, 2.1, 4.3]) + rng.normal(0.0, 0.20, 3)
        rr = r * (1.0 + rng.normal(0.0, 0.16, 3))
        tris.append(c + np.column_stack([rr * np.cos(ang), rr * np.sin(ang)]))
    return np.array(tris)


def tri_bounds(tris):
    return np.concatenate([tris.min(axis=1), tris.max(axis=1)], axis=1)


class Node:
    __slots__ = ("lo", "hi", "left", "right", "prims", "depth")

    def __init__(self, lo, hi, depth):
        self.lo = lo
        self.hi = hi
        self.left = None
        self.right = None
        self.prims = None
        self.depth = depth


def build(bounds, idx, depth=0, leaf=LEAF):
    lo = bounds[idx, :2].min(axis=0)
    hi = bounds[idx, 2:].max(axis=0)
    node = Node(lo, hi, depth)
    if len(idx) <= leaf:
        node.prims = idx
        return node
    cen = 0.5 * (bounds[idx, :2] + bounds[idx, 2:])
    axis = int(np.argmax(cen.max(axis=0) - cen.min(axis=0)))
    order = idx[np.argsort(cen[:, axis])]
    half = len(order) // 2
    node.left = build(bounds, order[:half], depth + 1, leaf)
    node.right = build(bounds, order[half:], depth + 1, leaf)
    return node


def walk(node, out=None):
    out = [] if out is None else out
    out.append(node)
    if node.left is not None:
        walk(node.left, out)
        walk(node.right, out)
    return out


def ray_box(o, d, lo, hi, t_max):
    """Slab test; returns entry distance or None."""
    t0, t1 = 0.0, t_max
    for k in (0, 1):
        if abs(d[k]) < 1e-12:
            if o[k] < lo[k] or o[k] > hi[k]:
                return None
            continue
        a = (lo[k] - o[k]) / d[k]
        b = (hi[k] - o[k]) / d[k]
        if a > b:
            a, b = b, a
        t0 = max(t0, a)
        t1 = min(t1, b)
        if t0 > t1:
            return None
    return t0


def ray_tri(o, d, tri):
    """Nearest crossing of the ray with the three edges of a triangle."""
    best = np.inf
    for k in range(3):
        p = tri[k]
        q = tri[(k + 1) % 3]
        e = q - p
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        rx, ry = p[0] - o[0], p[1] - o[1]
        t = (rx * (-e[1]) - ry * (-e[0])) / den
        u = (d[0] * ry - d[1] * rx) / den
        if t > 1e-9 and 0.0 <= u <= 1.0:
            best = min(best, t)
    return best


def trace(root, tris, o, d, t_max=np.inf):
    """Traversal with counters; also records the nodes it opened."""
    stack = [root]
    best = t_max
    hit = -1
    n_box = 0
    n_prim = 0
    visited = []
    leaves = []
    while stack:
        node = stack.pop()
        n_box += 1
        t_enter = ray_box(o, d, node.lo, node.hi, best)
        if t_enter is None:
            continue
        visited.append(node)
        if node.prims is not None:
            leaves.append(node)
            for i in node.prims:
                n_prim += 1
                t = ray_tri(o, d, tris[i])
                if t < best:
                    best = t
                    hit = int(i)
        else:
            stack.append(node.right)
            stack.append(node.left)
    return dict(t=best, hit=hit, n_box=n_box, n_prim=n_prim,
                visited=visited, leaves=leaves)


def random_rays(rng, n):
    """Rays launched inward from the frame border."""
    out = []
    for _ in range(n):
        side = rng.integers(0, 4)
        u = rng.uniform(-1.0, 1.0)
        if side == 0:
            o = np.array([-HALF_X, u * HALF_Y])
        elif side == 1:
            o = np.array([HALF_X, u * HALF_Y])
        elif side == 2:
            o = np.array([u * HALF_X, -HALF_Y])
        else:
            o = np.array([u * HALF_X, HALF_Y])
        target = np.array([rng.uniform(-HALF_X, HALF_X),
                           rng.uniform(-HALF_Y, HALF_Y)])
        d = target - o
        out.append((o, d / np.linalg.norm(d)))
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    tris = make_scene(rng)
    bounds = tri_bounds(tris)
    idx = np.arange(len(tris))
    root = build(bounds, idx)
    nodes = walk(root)
    depth_max = max(n.depth for n in nodes)
    n_leaf = sum(1 for n in nodes if n.prims is not None)

    # pick the demonstration ray by measurement: the deepest traversal among
    # candidate rays that still travel a good way before their first hit
    pick_rng = np.random.default_rng(SEED + 1)
    best_ray, best_res = (RAY_O, RAY_D / np.linalg.norm(RAY_D)), None
    for o, dd in random_rays(pick_rng, 500):
        r = trace(root, tris, o, dd)
        if r["hit"] < 0 or r["t"] < 3.2:
            continue
        if best_res is None or r["n_prim"] > best_res["n_prim"]:
            best_ray, best_res = (o, dd), r
    ray_o, d = best_ray
    res = best_res if best_res is not None else trace(root, tris, ray_o, d)
    hit_pt = ray_o + res["t"] * d
    vis_ids = {id(n) for n in res["visited"]}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: hierarchy and one traversal ---------------------------------
    ax.set_aspect("equal")
    for node in sorted(nodes, key=lambda n: n.depth):
        f = node.depth / max(depth_max, 1)
        col = lighten(DEEP, 0.20 + 0.62 * f)
        ax.add_patch(Rectangle(tuple(node.lo), *(node.hi - node.lo),
                               facecolor="none", edgecolor=col,
                               lw=1.25 - 0.75 * f, alpha=0.85,
                               zorder=2 + node.depth))
    for node in res["leaves"]:
        ax.add_patch(Rectangle(tuple(node.lo), *(node.hi - node.lo),
                               facecolor=lighten(ACCENT, 0.86),
                               edgecolor="none", zorder=1))
    for node in res["visited"]:
        ax.add_patch(Rectangle(tuple(node.lo), *(node.hi - node.lo),
                               facecolor="none", edgecolor=ACCENT, lw=1.15,
                               zorder=12))

    tested = set()
    for node in res["leaves"]:
        tested.update(int(i) for i in node.prims)
    for k, tri in enumerate(tris):
        if k == res["hit"]:
            fc, ec, lw = lighten(ACCENT, 0.30), ACCENT, 1.1
        elif k in tested:
            fc, ec, lw = lighten(GOLD, 0.55), GOLD, 0.8
        else:
            fc, ec, lw = "#dfe5ee", "#9aa9bd", 0.7
        ax.add_patch(Polygon(tri, closed=True, facecolor=fc, edgecolor=ec,
                             lw=lw, zorder=14 if k == res["hit"] else 10))

    end = ray_o + (res["t"] if res["hit"] >= 0 else 8.0) * d
    ax.annotate("", xy=tuple(end), xytext=tuple(ray_o),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                                shrinkA=0, shrinkB=0), zorder=16)
    far = ray_o + 8.0 * d
    ax.plot([end[0], far[0]], [end[1], far[1]], color=INK, lw=0.7,
            ls=(0, (3.0, 2.6)), alpha=0.5, zorder=15)
    ax.plot([hit_pt[0]], [hit_pt[1]], marker="o", ms=6.6, mfc=ACCENT,
            mec="white", mew=1.1, zorder=17)
    ax.text(hit_pt[0] + 0.10, hit_pt[1] + 0.12,
            f"first hit at distance {res['t']:.2f} units",
            ha="left", va="bottom", fontsize=6.5, color=ACCENT, zorder=18,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                      pad=1.4))
    lab_o = ray_o + 0.34 * d + 0.10 * np.array([-d[1], d[0]])
    ax.text(lab_o[0], lab_o[1], "ray origin",
            ha="right" if d[0] >= 0 else "left", va="center",
            fontsize=6.4, color=INK, zorder=18,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                      pad=1.0))

    ax.set_xlim(-HALF_X, HALF_X)
    ax.set_ylim(-HALF_Y, HALF_Y)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless length units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless length units)", fontsize=8.5)
    ax.set_title(f"Bounding volume hierarchy over {N_TRI} triangles, and the "
                 f"nodes one ray opens", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    handles = [
        Line2D([], [], color=lighten(DEEP, 0.20), lw=1.2,
               label=f"{len(nodes)} node boxes, {n_leaf} leaves, depth "
                     f"{depth_max}, leaf cap {LEAF}"),
        Line2D([], [], color=ACCENT, lw=1.15,
               label=f"{len(res['visited'])} boxes opened of "
                     f"{res['n_box']} tested"),
        Line2D([], [], color=GOLD, lw=1.0,
               label=f"{res['n_prim']} triangles tested of {N_TRI} "
                     f"({100.0 * res['n_prim'] / N_TRI:.0f} %)"),
        Line2D([], [], color=INK, lw=1.4, label="ray, first hit marked"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    fontsize=6.5, handlelength=1.6, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.92, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.004, 0.996))
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(40)

    # ---- aux (b): cost over many rays --------------------------------------
    rays = random_rays(rng, N_RAYS)
    prim_tests = np.array([trace(root, tris, o, dd)["n_prim"]
                           for o, dd in rays], dtype=float)
    box_tests = np.array([trace(root, tris, o, dd)["n_box"]
                          for o, dd in rays], dtype=float)

    ax_t.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.hist(prim_tests, bins=np.arange(-0.5, prim_tests.max() + 2.5, 2.0),
              color=lighten(BLUE, 0.35), edgecolor="white", linewidth=0.6,
              zorder=3, label=f"hierarchy, mean {prim_tests.mean():.1f} tests")
    ax_t.axvline(prim_tests.mean(), color=BLUE, lw=1.2, zorder=5)
    ax_t.axvline(N_TRI, color=ACCENT, lw=1.2, ls=(0, (3.0, 2.2)), zorder=5)
    ax_t.text(N_TRI - 1.2, ax_t.get_ylim()[1] * 0.62,
              f"test every triangle:\n{N_TRI} tests per ray", ha="right",
              va="top", fontsize=6.1, color=ACCENT, linespacing=1.35)
    ax_t.set_xlim(-1.0, N_TRI * 1.06)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("primitive tests per ray (count)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel(f"rays of {N_RAYS} (count)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title(f"Speed-up {N_TRI / prim_tests.mean():.1f} x on primitive "
                   f"tests", fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper right", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(1.015, 1.045))

    # ---- aux (c): where the cost goes, against leaf size -------------------
    caps = [1, 2, 4, 8, 16, 32]
    mb, mp, nn = [], [], []
    for cap in caps:
        r = build(bounds, idx, leaf=cap)
        stats = [trace(r, tris, o, dd) for o, dd in rays]
        mb.append(np.mean([s["n_box"] for s in stats]))
        mp.append(np.mean([s["n_prim"] for s in stats]))
        nn.append(len(walk(r)))
    mb = np.array(mb)
    mp = np.array(mp)
    tot = mb + mp
    xpos = np.arange(len(caps))

    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.bar(xpos, mb, width=0.62, color=lighten(DEEP, 0.42),
             edgecolor="white", linewidth=0.7, zorder=3, label="box tests")
    ax_b.bar(xpos, mp, width=0.62, bottom=mb, color=lighten(GOLD, 0.30),
             edgecolor="white", linewidth=0.7, zorder=3,
             label="primitive tests")
    best = int(np.argmin(tot))
    for k in xpos:
        ax_b.text(k, tot[k] + tot.max() * 0.035, f"{tot[k]:.0f}", ha="center",
                  va="bottom", fontsize=6.0,
                  color=ACCENT if k == best else SUB,
                  fontweight="bold" if k == best else "normal")
    ax_b.plot([best], [tot[best] + tot.max() * 0.20], marker="v", ms=5.0,
              color=ACCENT, ls="none", zorder=6)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([f"{c}\n({n} nodes)" for c, n in zip(caps, nn)],
                         fontsize=6.2, linespacing=1.3)
    ax_b.set_ylim(0.0, tot.max() * 1.30)
    ax_b.set_xlim(-0.62, len(caps) - 0.38)
    ax_b.tick_params(axis="y", labelsize=6.6)
    ax_b.set_xlabel("primitives per leaf the builder stops at (count)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("mean tests per ray (count)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title(f"Cheapest at {caps[best]} primitives per leaf",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper center", frameon=False, fontsize=6.0, ncol=2,
                handlelength=1.3, labelspacing=0.24, columnspacing=1.2,
                borderpad=0.10, bbox_to_anchor=(0.50, 1.035))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "976_cgx_bvh_ray_traversal.png")


if __name__ == "__main__":
    main()
