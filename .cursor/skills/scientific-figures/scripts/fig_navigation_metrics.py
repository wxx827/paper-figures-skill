"""Navigation benchmark: success, optimality, runtime and an overall profile.

Every number in this figure is measured, not assumed: 40 feasible random
worlds (10 seeds x 4 clutter levels) are generated, screened for solvability
with Dijkstra, and then attacked by six planners under a fixed budget.  Grid
searches are resolution-complete and always return a path; the sampling
planners degrade as clutter grows, which is exactly what the success heatmap
and the runtime curves expose.
"""
import heapq
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar, get_cmap

W, H = 20.0, 14.0
RES = 0.10
ROB_R = 0.22
START = np.array([1.0, 1.0])
GOAL = np.array([19.0, 13.0])
WALL_X = [4.0, 8.0, 12.0, 16.0]
WALL_T = 0.5
N_CLUTTER = 6
LEVELS = [2.4, 1.6, 1.1, 0.8]          # gap width through each wall (m)
LEVEL_NAMES = ["wide\n2.4 m", "moderate\n1.6 m", "narrow\n1.1 m",
               "very narrow\n0.8 m"]
N_SEEDS = 10
RRT_BUDGET, RRTSTAR_BUDGET, PRM_NODES = 2500, 2500, 500

PLANNERS = ["Dijkstra", "A*", "Greedy", "RRT", "RRT*", "PRM"]
PCOL = {"Dijkstra": "#0072b2", "A*": "#009e73", "Greedy": "#d55e00",
        "RRT": "#7b3294", "RRT*": "#c48b1a", "PRM": "#b23a48"}
S2 = np.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, S2), (-1, 1, S2), (1, -1, S2), (1, 1, S2)]


# ---- random worlds ---------------------------------------------------------
def clearance(env, x, y):
    rects, circs = env
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.minimum.reduce([x, W - x, y, H - y]).astype(float)
    for x0, y0, w, h in rects:
        dx = np.maximum.reduce([x0 - x, x - (x0 + w), np.zeros_like(x)])
        dy = np.maximum.reduce([y0 - y, y - (y0 + h), np.zeros_like(y)])
        d = np.minimum(d, np.hypot(dx, dy))
    for cx, cy, r in circs:
        d = np.minimum(d, np.hypot(x - cx, y - cy) - r)
    return d


def seg_free(env, p, Q, n=12):
    Q = np.atleast_2d(Q)
    t = np.linspace(0.0, 1.0, n)[:, None]
    X = p[0] + (Q[:, 0] - p[0])[None, :] * t
    Y = p[1] + (Q[:, 1] - p[1])[None, :] * t
    return clearance(env, X, Y).min(axis=0) > ROB_R


def obst_dist(shape, p):
    """Distance from one obstacle to a point, ignoring the arena walls."""
    if len(shape) == 4:
        x0, y0, w, h = shape
        dx = max(x0 - p[0], p[0] - (x0 + w), 0.0)
        dy = max(y0 - p[1], p[1] - (y0 + h), 0.0)
        return float(np.hypot(dx, dy))
    cx, cy, r = shape
    return float(np.hypot(p[0] - cx, p[1] - cy) - r)


def sample_env(gap, rng):
    """Four walls, each pierced by one randomly placed gap, plus clutter."""
    rects, circs = [], []
    for x in WALL_X:
        yc = rng.uniform(1.2 + gap / 2, H - 1.2 - gap / 2)
        lo, hi = yc - gap / 2, yc + gap / 2
        if lo > 0.02:
            rects.append((x, 0.0, WALL_T, lo))
        if hi < H - 0.02:
            rects.append((x, hi, WALL_T, H - hi))
    for _ in range(N_CLUTTER):
        r = rng.uniform(0.35, 0.85)
        circs.append((rng.uniform(r, W - r), rng.uniform(r, H - r), r))
    keep_r = [q for q in rects
              if obst_dist(q, START) > 0.7 and obst_dist(q, GOAL) > 0.7]
    keep_c = [q for q in circs
              if obst_dist(q, START) > 0.7 and obst_dist(q, GOAL) > 0.7]
    return (keep_r, keep_c)


def make_grid(env):
    nx, ny = int(round(W / RES)), int(round(H / RES))
    xs, ys = (np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    return clearance(env, X, Y) < ROB_R


def grid_search(occ, s, g, mode):
    ny, nx = occ.shape
    gc = np.full((ny, nx), np.inf)
    par = np.full((ny, nx, 2), -1, np.int32)
    closed = np.zeros((ny, nx), bool)

    def h(i, j):
        dy, dx = abs(i - g[0]), abs(j - g[1])
        return (dx + dy) + (S2 - 2.0) * min(dx, dy)

    gc[s] = 0.0
    heap = [(0.0, 0.0, s)]
    n_exp = 0
    while heap:
        _, c, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        n_exp += 1
        if (i, j) == g:
            break
        for di, dj, w in MOVES:
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx) or occ[ii, jj]:
                continue
            if di and dj and (occ[i, jj] or occ[ii, j]):
                continue
            if c + w < gc[ii, jj] - 1e-12:
                gc[ii, jj] = c + w
                par[ii, jj] = (i, j)
                key = (c + w) if mode == "dijkstra" else (
                    h(ii, jj) if mode == "greedy" else c + w + h(ii, jj))
                heapq.heappush(heap, (key, c + w, (ii, jj)))
    if not closed[g]:
        return None, n_exp
    seq, cur = [], g
    while cur != s:
        seq.append(cur)
        cur = (int(par[cur][0]), int(par[cur][1]))
    seq.append(s)
    cells = np.array(seq[::-1])
    return np.column_stack([(cells[:, 1] + 0.5) * RES,
                            (cells[:, 0] + 0.5) * RES]), n_exp


def rrt(env, rng, star, budget, eta=1.0, bias=0.10, tol=0.7):
    nodes = np.zeros((budget + 1, 2))
    nodes[0] = START
    par = np.full(budget + 1, -1, int)
    cost = np.zeros(budget + 1)
    n, goals = 1, []
    for _ in range(budget):
        q = GOAL.copy() if rng.random() < bias else \
            np.array([rng.uniform(0, W), rng.uniform(0, H)])
        d = np.hypot(*(nodes[:n] - q).T)
        k = int(np.argmin(d))
        if d[k] > eta:
            q = nodes[k] + (q - nodes[k]) * (eta / d[k])
        if clearance(env, q[0], q[1]) <= ROB_R or \
                not seg_free(env, nodes[k], q[None])[0]:
            continue
        if star:
            r = min(24.0 * np.sqrt(np.log(n + 1) / (n + 1)), 2.4)
            dist = np.hypot(*(nodes[:n] - q).T)
            near = np.flatnonzero(dist <= r)
            ok = seg_free(env, q, nodes[near]) if near.size else \
                np.zeros(0, bool)
            near, dn = near[ok], dist[near][ok]
            if near.size == 0:
                near, dn = np.array([k]), np.array([d[k]])
            b = int(np.argmin(cost[near] + dn))
            nodes[n], par[n], cost[n] = q, near[b], cost[near[b]] + dn[b]
        else:
            nodes[n], par[n], cost[n] = q, k, cost[k] + d[k]
        new = n
        n += 1
        if star:
            for idx in np.flatnonzero(cost[new] + dn - cost[near] < -1e-9):
                j = int(near[idx])
                if j == 0 or j == par[new]:
                    continue
                delta = cost[j] - (cost[new] + dn[idx])
                if delta <= 1e-9:
                    continue
                par[j], cost[j] = new, cost[j] - delta
                stack = [j]
                while stack:
                    u = stack.pop()
                    for c in np.flatnonzero(par[:n] == u):
                        if c != u:
                            cost[c] -= delta
                            stack.append(int(c))
        if np.hypot(*(q - GOAL)) <= tol and seg_free(env, q, GOAL[None])[0]:
            goals.append(new)
            if not star:
                break
    if not goals:
        return None, n
    best = min(goals, key=lambda i: cost[i] + np.hypot(*(nodes[i] - GOAL)))
    seq, cur = [GOAL], best
    while cur != -1:
        seq.append(nodes[cur].copy())
        cur = par[cur]
    return np.array(seq[::-1]), n


def prm(env, rng, n_nodes=PRM_NODES, k=10):
    pts = [START, GOAL]
    tries = 0
    while len(pts) < n_nodes and tries < n_nodes * 40:
        tries += 1
        q = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        if clearance(env, q[0], q[1]) > ROB_R:
            pts.append(q)
    P = np.array(pts)
    n = len(P)
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    adj = [[] for _ in range(n)]
    for i in range(n):
        cand = np.argsort(D[i])[1:k + 1]
        ok = seg_free(env, P[i], P[cand])
        for j, good in zip(cand, ok):
            if good:
                adj[i].append((int(j), D[i, j]))
                adj[int(j)].append((i, D[i, j]))
    dist = np.full(n, np.inf)
    par = np.full(n, -1, int)
    dist[0] = 0.0
    heap, done = [(0.0, 0)], np.zeros(n, bool)
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
    if not done[1]:
        return None, n
    seq, cur = [], 1
    while cur != -1:
        seq.append(P[cur])
        cur = par[cur]
    return np.array(seq[::-1]), n


# ---- path quality metrics --------------------------------------------------
def path_stats(env, P):
    seg = np.hypot(*np.diff(P, axis=0).T)
    L = float(seg.sum())
    hd = np.arctan2(*np.diff(P, axis=0).T[::-1])
    turn = float(np.abs((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi).sum())
    t = np.linspace(0, 1, 6)[:, None]
    X = (P[:-1, 0][None, :] + np.diff(P[:, 0])[None, :] * t).ravel()
    Y = (P[:-1, 1][None, :] + np.diff(P[:, 1])[None, :] * t).ravel()
    return L, turn / max(L, 1e-6), float(clearance(env, X, Y).mean())


def benchmark():
    rows = []
    for li, gap in enumerate(LEVELS):
        made = 0
        seed = 0
        while made < N_SEEDS and seed < 600:
            seed += 1
            rng = np.random.default_rng(1000 * li + seed)
            env = sample_env(gap, rng)
            occ = make_grid(env)
            s = (int(START[1] / RES), int(START[0] / RES))
            g = (int(GOAL[1] / RES), int(GOAL[0] / RES))
            if occ[s] or occ[g]:
                continue
            ref, _ = grid_search(occ, s, g, "dijkstra")
            if ref is None:
                continue                       # keep only feasible instances
            made += 1
            for name in PLANNERS:
                r = np.random.default_rng(7000 + 13 * seed)
                t0 = time.perf_counter()
                if name == "Dijkstra":
                    P, _ = grid_search(occ, s, g, "dijkstra")
                elif name == "A*":
                    P, _ = grid_search(occ, s, g, "astar")
                elif name == "Greedy":
                    P, _ = grid_search(occ, s, g, "greedy")
                elif name == "RRT":
                    P, _ = rrt(env, r, False, RRT_BUDGET)
                elif name == "RRT*":
                    P, _ = rrt(env, r, True, RRTSTAR_BUDGET)
                else:
                    P, _ = prm(env, r)
                ms = 1e3 * (time.perf_counter() - t0)
                if P is None or len(P) < 2:
                    rows.append((li, made, name, 0, np.nan, np.nan, np.nan,
                                 ms))
                else:
                    L, turn, clr = path_stats(env, P)
                    rows.append((li, made, name, 1, L, turn, clr, ms))
    return rows


def radar_axes(fig, rect, labels):
    ax = fig.add_subplot(rect, polar=True)
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang), labels, fontsize=6.8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.0,
                       color="#888888")
    ax.set_rlabel_position(198)
    ax.tick_params(pad=1.0)
    ax.grid(True, lw=0.45, color="#dddddd")
    ax.spines["polar"].set_linewidth(0.7)
    ax.spines["polar"].set_color("#bbbbbb")
    return ax, ang


def main():
    set_pub_style(base_font=8)
    rows = benchmark()
    lv = np.array([r[0] for r in rows])
    inst = np.array([r[1] for r in rows])
    nm = np.array([r[2] for r in rows])
    ok = np.array([r[3] for r in rows], float)
    L = np.array([r[4] for r in rows], float)
    turn = np.array([r[5] for r in rows], float)
    clr = np.array([r[6] for r in rows], float)
    ms = np.array([r[7] for r in rows], float)

    best = {}
    for li in range(len(LEVELS)):
        for ii in np.unique(inst[lv == li]):
            m = (lv == li) & (inst == ii) & (ok == 1)
            best[(li, ii)] = np.nanmin(L[m]) if m.any() else np.nan
    ratio = np.array([L[i] / best[(lv[i], inst[i])] if ok[i] else np.nan
                      for i in range(len(rows))])

    succ = np.array([[ok[(nm == p) & (lv == li)].mean()
                      for li in range(len(LEVELS))] for p in PLANNERS])

    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.965, top=0.855,
                          bottom=0.105, wspace=0.26, hspace=0.36,
                          width_ratios=[1.0, 1.12])

    # (a) success-rate heatmap ---------------------------------------------
    ax_a = ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(100 * succ, cmap=get_cmap("cb_diverging", reverse=True), vmin=0, vmax=100,
                   aspect="auto")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVEL_NAMES, fontsize=6.4)
    ax.set_yticks(range(len(PLANNERS)))
    ax.set_yticklabels(PLANNERS, fontsize=7.0)
    for i in range(len(PLANNERS)):
        for j in range(len(LEVELS)):
            v = 100 * succ[i, j]
            ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=6.8,
                    fontweight="bold",
                    color="white" if v < 45 else "#1a1a1a")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")
    ax.tick_params(length=0)
    ax.set_xlabel("width of the gap through each wall", fontsize=7.6,
                  labelpad=2.0)
    ax.set_title("Success rate over %d feasible worlds per cell (%%)"
                 % N_SEEDS, fontsize=8.6, fontweight="bold", pad=3)
    add_colorbar(fig, im, ax, label="success rate (%)", fraction=0.036,
                 pad=0.02, aspect=16, ticks=[0, 50, 100])

    # (b) optimality ratio boxplots ----------------------------------------
    ax_b = ax = fig.add_subplot(gs[0, 1])
    data = [ratio[(nm == p) & (ok == 1)] for p in PLANNERS]
    data = [d[np.isfinite(d)] for d in data]
    bp = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                    medianprops=dict(color="#1a1a1a", lw=1.0),
                    whiskerprops=dict(lw=0.7, color="#666666"),
                    capprops=dict(lw=0.7, color="#666666"),
                    boxprops=dict(lw=0.6, edgecolor="#444444"))
    for patch, p in zip(bp["boxes"], PLANNERS):
        patch.set_facecolor(PCOL[p])
        patch.set_alpha(0.55)
    rngj = np.random.default_rng(3)
    for i, (d, p) in enumerate(zip(data, PLANNERS)):
        ax.plot(i + 1 + rngj.normal(0, 0.055, d.size), d, ls="none",
                marker="o", ms=1.7, mfc=PCOL[p], mec="none", alpha=0.55,
                zorder=4)
    ax.axhline(1.0, color="#333333", lw=0.8, ls=(0, (2.5, 1.6)))
    ax.text(0.4, 1.005, "best solution found in each world", fontsize=6.2,
            color="#555555", va="bottom")
    ax.set_xticklabels(PLANNERS, fontsize=7.0)
    ax.set_ylabel("path length / best in world (-)", fontsize=7.6,
                  labelpad=1.5)
    ax.set_ylim(0.95, max(1.05, np.nanpercentile(
        np.concatenate(data), 99) * 1.05))
    ax.grid(True, axis="y", lw=0.4, color="#e8e8e8")
    ax.set_axisbelow(True)
    ax.set_title("Path-length optimality across all successful runs",
                 fontsize=8.6, fontweight="bold", pad=3)

    # (c) runtime vs clutter ------------------------------------------------
    ax_c = ax = fig.add_subplot(gs[1, 0])
    xs = np.array(LEVELS, float)
    for p in PLANNERS:
        med = [np.nanmedian(ms[(nm == p) & (lv == li)])
               for li in range(len(LEVELS))]
        lo = [np.nanpercentile(ms[(nm == p) & (lv == li)], 25)
              for li in range(len(LEVELS))]
        hi = [np.nanpercentile(ms[(nm == p) & (lv == li)], 75)
              for li in range(len(LEVELS))]
        ax.plot(xs, med, color=PCOL[p], lw=1.3, marker="o", ms=3.2,
                mfc="white", mew=0.9, label=p)
        ax.fill_between(xs, lo, hi, color=PCOL[p], alpha=0.14, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.1f" % v for v in xs])
    ax.invert_xaxis()                       # difficulty increases rightwards
    ax.set_xlabel("gap width (m)   $\\longrightarrow$   harder",
                  fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("planning time (ms, log scale)", fontsize=7.6, labelpad=1.5)
    ax.grid(True, which="both", lw=0.4, color="#ececec")
    ax.set_axisbelow(True)
    for p in PLANNERS:
        y = np.nanmedian(ms[(nm == p) & (lv == len(LEVELS) - 1)])
        ax.annotate(p, xy=(xs[-1], y), xytext=(-2, 0),
                    textcoords="offset points", fontsize=6.0, color=PCOL[p],
                    ha="right", va="center", fontweight="bold")
    ax.set_title("Single-core planning time against passage difficulty",
                 fontsize=8.6, fontweight="bold", pad=3)

    # (d) overall profile ---------------------------------------------------
    labels = ["success\nrate", "path\noptimality", "trajectory\nsmoothness",
              "safety\nmargin", "real-time\ncapability"]
    ax_d = ax, ang = radar_axes(fig, gs[1, 1], labels)
    ax_d = ax
    prof = {}
    for p in PLANNERS:
        m = (nm == p)
        mo = m & (ok == 1)
        prof[p] = np.array([
            ok[m].mean(),
            np.nanmean(1.0 / ratio[mo]),
            float(np.exp(-np.nanmean(turn[mo]) / 1.0)),
            float(min(np.nanmean(clr[mo]) / 1.2, 1.0)),
            float(np.clip(1 - np.log10(max(np.nanmedian(ms[m]), 2.0) / 2.0)
                          / np.log10(2000 / 2.0), 0, 1))])
    cl = np.concatenate([ang, ang[:1]])
    for p in PLANNERS:
        v = np.concatenate([prof[p], prof[p][:1]])
        ax.plot(cl, v, color=PCOL[p], lw=1.2, label=p)
        ax.fill(cl, v, color=PCOL[p], alpha=0.055)
    ax.set_title("Normalised capability profile", fontsize=8.6,
                 fontweight="bold", pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.235), ncol=3,
              fontsize=6.4, frameon=False, handlelength=1.5,
              columnspacing=1.4)

    for a in (ax_b, ax_c):                  # match the boxed heatmap panel
        for s in a.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.7)
            s.set_color("#333333")
    for lab, a, xo in (("a", ax_a, -0.20), ("b", ax_b, -0.10),
                       ("c", ax_c, -0.14), ("d", ax_d, -0.06)):
        add_panel_label(a, "(%s)" % lab, x=xo, y=1.05, size=10)

    handles = [Line2D([], [], color=PCOL[p], lw=2.0, label=p)
               for p in PLANNERS]
    savefig(fig, "172_navigation_metrics.png")


if __name__ == "__main__":
    main()
