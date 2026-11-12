"""Six planners on one workspace: explored set, path quality and runtime.

Dijkstra, A* and greedy best-first run on the same 0.20 m 8-connected grid;
RRT, RRT* and PRM run in the continuous configuration space with a 0.16 m
inflated robot.  Every panel shades the region the planner actually touched
(closed cells, or tree / roadmap edges) and reports the measured wall-clock
time, so the classic completeness-optimality-effort trade-off is visible in
a single glance.
"""
import heapq
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

W, H = 20.0, 14.0
RES = 0.20
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = np.array([1.3, 1.5])
GOAL = np.array([18.7, 12.6])
ROB_R = 0.16

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"
S2 = np.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, S2), (-1, 1, S2), (1, -1, S2), (1, 1, S2)]


# ---- geometry --------------------------------------------------------------
def collision(x, y, pad=ROB_R):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x < pad) | (x > W - pad) | (y < pad) | (y > H - pad)
    for x0, y0, w, h in RECTS:
        m |= ((x >= x0 - pad) & (x <= x0 + w + pad) &
              (y >= y0 - pad) & (y <= y0 + h + pad))
    for cx, cy, r in CIRCLES:
        m |= (x - cx) ** 2 + (y - cy) ** 2 <= (r + pad) ** 2
    return m


def seg_free(p, Q, n=14):
    Q = np.atleast_2d(Q)
    t = np.linspace(0.0, 1.0, n)[:, None]
    X = p[0] + (Q[:, 0] - p[0])[None, :] * t
    Y = p[1] + (Q[:, 1] - p[1])[None, :] * t
    return ~collision(X, Y).any(axis=0)


def build_grid(res=RES):
    nx, ny = int(round(W / res)), int(round(H / res))
    xs, ys = (np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    return collision(X, Y), xs, ys


def path_len(p):
    return float(np.hypot(*np.diff(np.asarray(p), axis=0).T).sum())


# ---- planners --------------------------------------------------------------
def grid_search(occ, start, goal, mode):
    ny, nx = occ.shape
    g = np.full((ny, nx), np.inf)
    par = np.full((ny, nx, 2), -1, np.int32)
    closed = np.zeros((ny, nx), bool)

    def h(i, j):
        dy, dx = abs(i - goal[0]), abs(j - goal[1])
        return (dx + dy) + (S2 - 2.0) * min(dx, dy)

    g[start] = 0.0
    heap = [(0.0, 0.0, start)]
    cnt = 0
    while heap:
        _, gc, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        cnt += 1
        if (i, j) == goal:
            break
        for di, dj, c in MOVES:
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx) or occ[ii, jj]:
                continue
            if di and dj and (occ[i, jj] or occ[ii, j]):
                continue
            ng = gc + c
            if ng < g[ii, jj] - 1e-12:
                g[ii, jj] = ng
                par[ii, jj] = (i, j)
                key = ng if mode == "dijkstra" else (
                    h(ii, jj) if mode == "greedy" else ng + h(ii, jj))
                heapq.heappush(heap, (key, ng, (ii, jj)))
    seq, cur = [], goal
    while cur != start:
        seq.append(cur)
        cur = (int(par[cur][0]), int(par[cur][1]))
    seq.append(start)
    cells = np.array(seq[::-1])
    xy = np.column_stack([(cells[:, 1] + 0.5) * RES, (cells[:, 0] + 0.5) * RES])
    return dict(closed=closed, path=xy, n=cnt)


def rrt(n_iter=4000, seed=5, star=False, eta=1.10, bias=0.10, tol=0.70):
    rng = np.random.default_rng(seed)
    nodes = np.zeros((n_iter + 1, 2))
    nodes[0] = START
    parent = np.full(n_iter + 1, -1, int)
    cost = np.zeros(n_iter + 1)
    n, goal_nodes = 1, []
    for _ in range(n_iter):
        q = GOAL.copy() if rng.random() < bias else \
            np.array([rng.uniform(0, W), rng.uniform(0, H)])
        d = np.linalg.norm(nodes[:n] - q, axis=1)
        k = int(np.argmin(d))
        if d[k] > eta:
            q = nodes[k] + (q - nodes[k]) * (eta / d[k])
        if collision(q[0], q[1]) or not seg_free(nodes[k], q[None])[0]:
            continue
        if star:
            r = min(26.0 * np.sqrt(np.log(n + 1) / (n + 1)), 2.6)
            dist = np.linalg.norm(nodes[:n] - q, axis=1)
            near = np.flatnonzero(dist <= r)
            ok = seg_free(q, nodes[near]) if near.size else np.array([], bool)
            near, dn = near[ok], dist[near][ok]
            if near.size == 0:
                near = np.array([k])
                dn = np.array([np.linalg.norm(nodes[k] - q)])
            b = int(np.argmin(cost[near] + dn))
            nodes[n], parent[n] = q, near[b]
            cost[n] = cost[near[b]] + dn[b]
        else:
            nodes[n], parent[n] = q, k
            cost[n] = cost[k] + np.linalg.norm(nodes[k] - q)
        new = n
        n += 1
        if star:
            for idx in np.flatnonzero(cost[new] + dn - cost[near] < -1e-9):
                j = int(near[idx])
                if j == 0 or j == parent[new]:
                    continue
                delta = cost[j] - (cost[new] + dn[idx])
                if delta <= 1e-9:
                    continue
                parent[j], cost[j] = new, cost[j] - delta
                stack = [j]
                while stack:
                    u = stack.pop()
                    for c in np.flatnonzero(parent[:n] == u):
                        if c != u:
                            cost[c] -= delta
                            stack.append(int(c))
        if np.linalg.norm(q - GOAL) <= tol and seg_free(q, GOAL[None])[0]:
            goal_nodes.append(new)
            if not star:
                break
    best, bc = -1, np.inf
    for i in goal_nodes:
        c = cost[i] + np.linalg.norm(nodes[i] - GOAL)
        if c < bc:
            best, bc = i, c
    seq, cur = [GOAL], best
    while cur != -1:
        seq.append(nodes[cur].copy())
        cur = parent[cur]
    edges = [[nodes[i], nodes[parent[i]]] for i in range(1, n)]
    return dict(edges=edges, path=np.array(seq[::-1]), n=n)


def prm(n_samples=520, k=12, seed=4):
    rng = np.random.default_rng(seed)
    pts = [START, GOAL]
    while len(pts) < n_samples:
        q = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        if not collision(q[0], q[1]):
            pts.append(q)
    P = np.array(pts)
    n = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in np.argsort(D[i])[1:k + 1]:
            j = int(j)
            if j <= i and i in [e[0] for e in adj[j]]:
                continue
            if seg_free(P[i], P[j][None])[0]:
                adj[i].append((j, D[i, j]))
                adj[j].append((i, D[i, j]))
                edges.append([P[i], P[j]])
    dist = np.full(n, np.inf)
    par = np.full(n, -1, int)
    dist[0] = 0.0
    heap = [(0.0, 0)]
    done = np.zeros(n, bool)
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == 1:
            break
        for v, w in adj[u]:
            if d + w < dist[v] - 1e-12:
                dist[v], par[v] = d + w, u
                heapq.heappush(heap, (dist[v], v))
    seq, cur = [], 1
    while cur != -1:
        seq.append(P[cur])
        cur = par[cur]
    return dict(edges=edges, path=np.array(seq[::-1]), n=n)


# ---- drawing ---------------------------------------------------------------
def draw_obstacles(ax):
    for x0, y0, w, h in RECTS:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=OBS_C, ec="none", zorder=4))
    for cx, cy, r in CIRCLES:
        ax.add_patch(Circle((cx, cy), r, fc=OBS_C, ec="none", zorder=4))


def frame(ax, row, col):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.tick_params(length=2.2, labelsize=6.2)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=7.2, labelpad=1.2)
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=7.2, labelpad=1.2)
    else:
        ax.set_yticklabels([])


def main():
    set_pub_style(base_font=8)
    occ, xs, ys = build_grid()
    si = (int(START[1] / RES), int(START[0] / RES))
    gi = (int(GOAL[1] / RES), int(GOAL[0] / RES))
    ext = [0, W, 0, H]

    runs = []
    for name, mode in [("Dijkstra", "dijkstra"), ("A*", "astar"),
                       ("Greedy best-first", "greedy")]:
        t0 = time.perf_counter()
        r = grid_search(occ, si, gi, mode)
        r.update(name=name, ms=1e3 * (time.perf_counter() - t0),
                 unit="cells expanded", kind="grid")
        runs.append(r)
    for name, star in [("RRT", False), ("RRT*", True)]:
        t0 = time.perf_counter()
        r = rrt(star=star, n_iter=3000 if star else 4000)
        r.update(name=name, ms=1e3 * (time.perf_counter() - t0),
                 unit="tree nodes", kind="sample")
        runs.append(r)
    t0 = time.perf_counter()
    r = prm()
    r.update(name="PRM", ms=1e3 * (time.perf_counter() - t0),
             unit="roadmap nodes", kind="sample")
    runs.append(r)

    cols = ["#0072b2", "#009e73", "#d55e00", "#7b3294", "#c48b1a", "#b23a48"]
    opt = min(path_len(r["path"]) for r in runs)

    fig, axes = plt.subplots(2, 3, figsize=(11.3, 6.35))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.865, bottom=0.085,
                        wspace=0.05, hspace=0.12)

    for k, r in enumerate(runs):
        row, col = divmod(k, 3)
        ax = axes[row, col]
        if r["kind"] == "grid":
            img = np.zeros(occ.shape + (4,))
            img[r["closed"] & ~occ] = mpl.colors.to_rgba(cols[k], 0.28)
            ax.imshow(img, extent=ext, origin="lower",
                      interpolation="nearest", zorder=1)
        else:
            ax.add_collection(LineCollection(
                r["edges"], colors=cols[k], linewidths=0.36, alpha=0.55,
                zorder=1))
        draw_obstacles(ax)
        L = path_len(r["path"])
        ax.plot(r["path"][:, 0], r["path"][:, 1], color="white", lw=3.0,
                solid_capstyle="round", zorder=6)
        ax.plot(r["path"][:, 0], r["path"][:, 1], color=cols[k], lw=1.9,
                solid_capstyle="round", zorder=7)
        ax.plot(*START, marker="^", ms=7.5, mfc=S_C, mec="white", mew=0.9,
                ls="none", zorder=9)
        ax.plot(*GOAL, marker="*", ms=12, mfc=G_C, mec="white", mew=0.9,
                ls="none", zorder=9)
        ax.text(0.982, 0.035,
                "%s  %d\npath  %.2f m  (%.0f%% of best found)\ntime  %.0f ms"
                % (r["unit"], r["n"], L, 100 * L / opt, r["ms"]),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.3, linespacing=1.4, zorder=10,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                          lw=0.6, alpha=0.94))
        frame(ax, row, col)
        add_panel_label(ax, "(%s)" % "abcdef"[k], x=-0.04, y=1.005, size=9.5)
        ax.set_title(r["name"], fontsize=8.6, fontweight="bold", pad=3,
                     color=cols[k])

    handles = [Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc="#9ec7e8",
                      mec="none", label="explored set / graph"),
               Line2D([], [], color="#555555", lw=2.0, label="returned path"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "165_planner_comparison.png")


if __name__ == "__main__":
    main()
