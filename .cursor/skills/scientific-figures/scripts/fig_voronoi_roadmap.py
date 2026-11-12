"""Generalised Voronoi roadmap: maximum-clearance routing versus RRT.

Obstacle boundaries are sampled at 0.20 m and passed to scipy's Voronoi
tessellation; ridges that stay in free space approximate the generalised
Voronoi diagram (the medial axis), which is by construction the locus of
points equidistant from two obstacles.  Dijkstra on that roadmap therefore
returns the safest corridor, and the clearance profile in panel (d) shows
how much margin it buys over a raw RRT solution of similar length.
"""
import heapq

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import Voronoi
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

W, H = 20.0, 14.0
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = np.array([1.3, 1.5])
GOAL = np.array([18.7, 12.6])
ROB_R = 0.16
MIN_CLEAR = 0.28            # roadmap edges must keep at least this margin

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"
VOR_C = "#0072b2"
RRT_C = "#d55e00"


def clearance(x, y):
    """Distance to the nearest obstacle surface or wall (vectorised)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.minimum.reduce([x, W - x, y, H - y]).astype(float)
    for x0, y0, w, h in RECTS:
        dx = np.maximum.reduce([x0 - x, x - (x0 + w), np.zeros_like(x)])
        dy = np.maximum.reduce([y0 - y, y - (y0 + h), np.zeros_like(y)])
        d = np.minimum(d, np.hypot(dx, dy))
    for cx, cy, r in CIRCLES:
        d = np.minimum(d, np.hypot(x - cx, y - cy) - r)
    return d


def seg_clear(p, q, n=16):
    t = np.linspace(0, 1, n)
    return clearance(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t).min()


def boundary_samples(step=0.20):
    pts = []
    for x0, y0, w, h in RECTS:
        for a, b in [((x0, y0), (x0 + w, y0)), ((x0 + w, y0), (x0 + w, y0 + h)),
                     ((x0 + w, y0 + h), (x0, y0 + h)), ((x0, y0 + h), (x0, y0))]:
            L = np.hypot(b[0] - a[0], b[1] - a[1])
            t = np.linspace(0, 1, max(int(L / step), 2), endpoint=False)
            pts.append(np.column_stack([a[0] + (b[0] - a[0]) * t,
                                        a[1] + (b[1] - a[1]) * t]))
    for cx, cy, r in CIRCLES:
        a = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * r / step), 12),
                        endpoint=False)
        pts.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    for a, b in [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)),
                 ((0, H), (0, 0))]:
        L = np.hypot(b[0] - a[0], b[1] - a[1])
        t = np.linspace(0, 1, int(L / step), endpoint=False)
        pts.append(np.column_stack([a[0] + (b[0] - a[0]) * t,
                                    a[1] + (b[1] - a[1]) * t]))
    return np.vstack(pts)


def build_roadmap():
    vor = Voronoi(boundary_samples())
    V = vor.vertices
    edges = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        p, q = V[a], V[b]
        if not (0 <= p[0] <= W and 0 <= p[1] <= H and
                0 <= q[0] <= W and 0 <= q[1] <= H):
            continue
        if np.hypot(*(p - q)) > 2.5:
            continue
        if seg_clear(p, q) < MIN_CLEAR:
            continue
        edges.append((a, b))
    used = sorted({i for e in edges for i in e})
    remap = {v: k for k, v in enumerate(used)}
    nodes = V[used]
    adj = [[] for _ in nodes]
    for a, b in edges:
        i, j = remap[a], remap[b]
        w = float(np.hypot(*(nodes[i] - nodes[j])))
        adj[i].append((j, w))
        adj[j].append((i, w))
    return nodes, adj, [(remap[a], remap[b]) for a, b in edges]


def dijkstra(nodes, adj, si, gi):
    n = len(nodes)
    dist = np.full(n, np.inf)
    par = np.full(n, -1, int)
    dist[si] = 0.0
    heap = [(0.0, si)]
    done = np.zeros(n, bool)
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == gi:
            break
        for v, w in adj[u]:
            if d + w < dist[v] - 1e-12:
                dist[v], par[v] = d + w, u
                heapq.heappush(heap, (dist[v], v))
    if not done[gi]:
        return None
    seq, cur = [], gi
    while cur != -1:
        seq.append(nodes[cur])
        cur = par[cur]
    return np.array(seq[::-1])


def attach(nodes, adj, p, min_clear=MIN_CLEAR):
    """Add p as a roadmap node linked to the nearest reachable vertices."""
    d = np.hypot(*(nodes - p).T)
    order = np.argsort(d)
    nodes2 = np.vstack([nodes, p])
    adj2 = [list(a) for a in adj] + [[]]
    k = len(nodes2) - 1
    added = 0
    for i in order[:80]:
        if seg_clear(p, nodes[i]) < min_clear * 0.6:
            continue
        w = float(d[i])
        adj2[k].append((i, w))
        adj2[i].append((k, w))
        added += 1
        if added >= 3:
            break
    return nodes2, adj2, k


def rrt(seed=5, n_iter=6000, eta=1.0, bias=0.10, tol=0.6):
    rng = np.random.default_rng(seed)
    nodes = [START.copy()]
    par = [-1]
    for _ in range(n_iter):
        q = GOAL.copy() if rng.random() < bias else \
            np.array([rng.uniform(0, W), rng.uniform(0, H)])
        P = np.array(nodes)
        d = np.hypot(*(P - q).T)
        k = int(np.argmin(d))
        if d[k] > eta:
            q = P[k] + (q - P[k]) * (eta / d[k])
        if clearance(q[0], q[1]) < ROB_R or seg_clear(P[k], q) < ROB_R:
            continue
        nodes.append(q)
        par.append(k)
        if np.hypot(*(q - GOAL)) < tol and seg_clear(q, GOAL) > ROB_R:
            seq, cur = [GOAL], len(nodes) - 1
            while cur != -1:
                seq.append(nodes[cur])
                cur = par[cur]
            return np.array(seq[::-1])
    return None


def resample(path, step=0.05):
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.arange(0, s[-1], step)
    return (np.column_stack([np.interp(u, s, path[:, 0]),
                             np.interp(u, s, path[:, 1])]), u, s[-1])


def draw_obstacles(ax):
    for x0, y0, w, h in RECTS:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=OBS_C, ec="none", zorder=5))
    for cx, cy, r in CIRCLES:
        ax.add_patch(Circle((cx, cy), r, fc=OBS_C, ec="none", zorder=5))


def map_frame(ax, xlabel=True, ylabel=True):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.tick_params(length=2.4, labelsize=6.6)
    if xlabel:
        ax.set_xlabel("x (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_yticklabels([])


def main():
    set_pub_style(base_font=8)
    nodes, adj, edges = build_roadmap()
    n2, a2, si = attach(nodes, adj, START)
    n3, a3, gi = attach(n2, a2, GOAL)
    vpath = dijkstra(n3, a3, si, gi)
    rpath = rrt()

    vr, vu, vlen = resample(vpath)
    rr, ru, rlen = resample(rpath)
    vc, rc = clearance(vr[:, 0], vr[:, 1]), clearance(rr[:, 0], rr[:, 1])

    gx = np.linspace(0, W, 420)
    gy = np.linspace(0, H, 300)
    GX, GY = np.meshgrid(gx, gy)
    CL = np.clip(clearance(GX, GY), 0, None)

    fig = plt.figure(figsize=(10.2, 9.3))
    gs = fig.add_gridspec(3, 2, left=0.062, right=0.945, top=0.888,
                          bottom=0.055, wspace=0.14, hspace=0.24,
                          height_ratios=[1.0, 1.0, 0.46])

    # (a) Voronoi diagram + retained roadmap --------------------------------
    ax_a = ax = fig.add_subplot(gs[0, 0])
    segs = [[nodes[i], nodes[j]] for i, j in edges]
    ax.add_collection(LineCollection(segs, colors=VOR_C, linewidths=0.55,
                                     alpha=0.9, zorder=3))
    ax.plot(nodes[:, 0], nodes[:, 1], ls="none", marker="o", ms=0.7,
            mfc=VOR_C, mec="none", alpha=0.5, zorder=4)
    draw_obstacles(ax)
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    map_frame(ax, xlabel=False)
    ax.set_title("Generalised Voronoi diagram of the obstacles",
                 fontsize=8.6, fontweight="bold", pad=3)
    ax.text(0.982, 0.035, "%d roadmap vertices\n%d edges kept "
            "(clearance > %.2f m)" % (len(nodes), len(edges), MIN_CLEAR),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (b) roadmap query vs RRT ---------------------------------------------
    ax_b = ax = fig.add_subplot(gs[0, 1])
    ax.add_collection(LineCollection(segs, colors="#b9cfe0", linewidths=0.45,
                                     zorder=2))
    draw_obstacles(ax)
    ax.plot(rpath[:, 0], rpath[:, 1], color="white", lw=3.0, zorder=6)
    ax.plot(rpath[:, 0], rpath[:, 1], color=RRT_C, lw=1.8, zorder=7,
            label="RRT path")
    ax.plot(vpath[:, 0], vpath[:, 1], color="white", lw=3.2, zorder=7)
    ax.plot(vpath[:, 0], vpath[:, 1], color=VOR_C, lw=2.0, zorder=8,
            label="Voronoi roadmap path")
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    map_frame(ax, xlabel=False, ylabel=False)
    ax.set_title("Dijkstra on the roadmap versus a raw RRT solution",
                 fontsize=8.6, fontweight="bold", pad=3)
    ax.text(0.982, 0.035, "Voronoi  %.2f m,  min clearance %.2f m\n"
            "RRT      %.2f m,  min clearance %.2f m\n"
            "straight-line bound %.2f m"
            % (vlen, vc.min(), rlen, rc.min(),
               float(np.hypot(*(GOAL - START)))), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, linespacing=1.5,
            family="DejaVu Sans Mono", zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (c) clearance field ---------------------------------------------------
    ax_c = ax = fig.add_subplot(gs[1, 0])
    cf = ax.contourf(GX, GY, CL, levels=np.linspace(0, 2.6, 14),
                     cmap="YlGnBu", extend="max")
    ax.contour(GX, GY, CL, levels=[ROB_R, 0.5, 1.0, 1.5, 2.0],
               colors="white", linewidths=0.4, alpha=0.7)
    draw_obstacles(ax)
    ax.plot(vpath[:, 0], vpath[:, 1], color="white", lw=2.6, zorder=7)
    ax.plot(vpath[:, 0], vpath[:, 1], color=VOR_C, lw=1.6, zorder=8)
    ax.plot(rpath[:, 0], rpath[:, 1], color="white", lw=2.4, zorder=6)
    ax.plot(rpath[:, 0], rpath[:, 1], color=RRT_C, lw=1.4, zorder=7)
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    map_frame(ax)
    ax.set_title("Clearance field with both solutions overlaid",
                 fontsize=8.6, fontweight="bold", pad=3)
    add_colorbar(fig, cf, ax, label="distance to nearest obstacle (m)",
                 fraction=0.036, pad=0.02, aspect=16)

    # (d) safety tube: maximal free disks along each solution ---------------
    ax_d = ax = fig.add_subplot(gs[1, 1])
    draw_obstacles(ax)
    every = int(1.60 / 0.05)
    for P, C, col, fc in [(rr, rc, RRT_C, "#f4d3b8"),
                          (vr, vc, VOR_C, "#c4dcee")]:
        for i in range(0, len(P), every):
            ax.add_patch(Circle(P[i], C[i], fc=fc, ec=col, lw=0.35,
                                alpha=0.30, zorder=3))
    for P, C, col in [(rr, rc, RRT_C), (vr, vc, VOR_C)]:
        ax.plot(P[:, 0], P[:, 1], color="white", lw=3.2,
                solid_capstyle="round", zorder=5.5)
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.9,
                solid_capstyle="round", zorder=6)
        k = int(np.argmin(C))
        ax.add_patch(Circle(P[k], C[k], fc="none", ec=col, lw=1.4, zorder=7))
        ax.plot(*P[k], marker="v", ms=4.6, mfc=col, mec="white", mew=0.7,
                ls="none", zorder=8)
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    map_frame(ax, ylabel=False)
    ax.set_title("Maximal free disks swept along each solution",
                 fontsize=8.6, fontweight="bold", pad=3)
    ax.text(0.982, 0.035, "disks every 1.6 m; the outlined disk\n"
            "marks the tightest point of each tube",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            linespacing=1.45, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (e) subordinate strip: the same clearance, quantified -----------------
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.set_facecolor("#f4f5f7")
    ax_e.plot(vu, vc, color=VOR_C, lw=1.3, label="Voronoi roadmap path")
    ax_e.plot(ru, rc, color=RRT_C, lw=1.3, label="RRT path")
    ax_e.axhline(ROB_R, color=G_C, lw=0.9, ls=(0, (2.5, 1.6)))
    ax_e.fill_between([0, max(vu[-1], ru[-1])], 0, ROB_R, color=G_C,
                      alpha=0.10, lw=0)
    ax_e.text(0.35, ROB_R + 0.07, "robot radius %.2f m" % ROB_R,
              fontsize=6.0, color=G_C, va="bottom")
    for u, c, col in [(vu, vc, VOR_C), (ru, rc, RRT_C)]:
        k = int(np.argmin(c))
        ax_e.plot(u[k], c[k], marker="v", ms=4.2, mfc=col, mec="white",
                  mew=0.6, zorder=5)
    ax_e.set_xlim(0, max(vu[-1], ru[-1]))
    ax_e.set_ylim(0, max(vc.max(), rc.max()) * 1.20)
    ax_e.set_xlabel("arc length along path (m)", fontsize=7.0, labelpad=1.2)
    ax_e.set_ylabel("clearance (m)", fontsize=7.0, labelpad=1.2)
    ax_e.tick_params(labelsize=6.2, length=2.2)
    ax_e.grid(True, lw=0.4, color="#e2e4e6")
    ax_e.set_axisbelow(True)
    for s in ax_e.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#9aa0a6")
    ax_e.legend(loc="upper right", fontsize=6.2, frameon=False,
                handlelength=1.6, ncol=2, columnspacing=1.4)
    ax_e.set_title("Supporting statistic  |  clearance measured along the "
                   "two paths of panel (d)", fontsize=7.4, style="italic",
                   color="#555555", pad=2.5)
    ax_e.text(0.008, 0.94, "mean clearance   Voronoi %.2f m   RRT %.2f m"
              % (vc.mean(), rc.mean()), transform=ax_e.transAxes, ha="left",
              va="top", fontsize=6.2, family="DejaVu Sans Mono", zorder=10,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                        lw=0.5, alpha=0.95))

    for lab, a in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(a, "(%s)" % lab, x=-0.06, y=1.02, size=10)
    add_panel_label(ax_e, "(e)", x=-0.028, y=1.06, size=9, weight="bold")

    handles = [Line2D([], [], color=VOR_C, lw=1.8,
                      label="Voronoi roadmap / path"),
               Line2D([], [], color=RRT_C, lw=1.8, label="RRT path"),
               Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "169_voronoi_roadmap.png")


if __name__ == "__main__":
    main()
