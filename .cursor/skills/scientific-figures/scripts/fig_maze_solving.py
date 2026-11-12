"""Maze search: BFS, DFS and A* on an identical braided maze.

The maze is carved by randomised depth-first search (recursive backtracker)
on a 27 x 19 cell lattice; 9% of the interior walls are then knocked out so
that the maze is *braided* and admits several distinct solutions.  All three
searches run on the same 4-connected free grid and log their expansion
order, which makes the difference between an optimal breadth-first sweep, a
depth-first plunge and a heuristic-guided A* immediately legible.
"""
import heapq
from collections import deque

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

NC, NR = 27, 19             # maze cells
BRAID = 0.09                # fraction of interior walls removed
S_C = "#1f8a4c"
G_C = "#b23a48"
OBS_C = "#3a3a3a"
ALG_C = {"BFS": "#0072b2", "DFS": "#d55e00", "A*": "#009e73"}


def generate_maze(nc=NC, nr=NR, seed=6, braid=BRAID):
    rng = np.random.default_rng(seed)
    wall = np.ones((2 * nr + 1, 2 * nc + 1), bool)
    seen = np.zeros((nr, nc), bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    wall[1, 1] = False
    while stack:
        r, c = stack[-1]
        opts = [(r + dr, c + dc, dr, dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < nr and 0 <= c + dc < nc
                and not seen[r + dr, c + dc]]
        if not opts:
            stack.pop()
            continue
        r2, c2, dr, dc = opts[int(rng.integers(len(opts)))]
        wall[2 * r + 1 + dr, 2 * c + 1 + dc] = False
        wall[2 * r2 + 1, 2 * c2 + 1] = False
        seen[r2, c2] = True
        stack.append((r2, c2))

    cand = [(2 * r + 1, 2 * c + 2) for r in range(nr) for c in range(nc - 1)]
    cand += [(2 * r + 2, 2 * c + 1) for r in range(nr - 1) for c in range(nc)]
    cand = [p for p in cand if wall[p]]
    pick = rng.choice(len(cand), int(braid * len(cand)), replace=False)
    for i in pick:
        wall[cand[i]] = False
    return wall


NB = ((-1, 0), (1, 0), (0, -1), (0, 1))


def search(wall, start, goal, mode):
    ny, nx = wall.shape
    order = np.full((ny, nx), -1, np.int32)
    par = {start: None}
    cnt = 0
    if mode == "BFS":
        q = deque([start])
        while q:
            u = q.popleft()
            if order[u] >= 0:
                continue
            order[u] = cnt
            cnt += 1
            if u == goal:
                break
            for dr, dc in NB:
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < ny and 0 <= v[1] < nx and not wall[v] \
                        and v not in par:
                    par[v] = u
                    q.append(v)
    elif mode == "DFS":
        st = [start]
        while st:
            u = st.pop()
            if order[u] >= 0:
                continue
            order[u] = cnt
            cnt += 1
            if u == goal:
                break
            for dr, dc in NB:
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < ny and 0 <= v[1] < nx and not wall[v] \
                        and order[v] < 0:
                    par[v] = u
                    st.append(v)
    else:
        g = {start: 0}
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        heap = [(h(start), 0, start)]
        while heap:
            _, gc, u = heapq.heappop(heap)
            if order[u] >= 0:
                continue
            order[u] = cnt
            cnt += 1
            if u == goal:
                break
            for dr, dc in NB:
                v = (u[0] + dr, u[1] + dc)
                if not (0 <= v[0] < ny and 0 <= v[1] < nx) or wall[v]:
                    continue
                if gc + 1 < g.get(v, 1 << 30):
                    g[v] = gc + 1
                    par[v] = u
                    heapq.heappush(heap, (gc + 1 + h(v), gc + 1, v))
    seq, cur = [], goal
    while cur is not None:
        seq.append(cur)
        cur = par[cur]
    return dict(order=order, path=np.array(seq[::-1]), n=cnt)


def frame(ax, wall, row):
    ny, nx = wall.shape
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 18, 36])
    ax.tick_params(length=2.2, labelsize=6.2)
    if row == 1:
        ax.set_xlabel("grid cell (x)", fontsize=7.2, labelpad=1.2)
    else:
        ax.set_xticklabels([])


def markers(ax, s, g):
    ax.plot(s[1], s[0], marker="^", ms=7, mfc=S_C, mec="white", mew=0.8,
            ls="none", zorder=9)
    ax.plot(g[1], g[0], marker="*", ms=11, mfc=G_C, mec="white", mew=0.8,
            ls="none", zorder=9)


def main():
    set_pub_style(base_font=8)
    wall = generate_maze()
    ny, nx = wall.shape
    start, goal = (1, 1), (2 * NR - 1, 2 * NC - 1)
    runs = {m: search(wall, start, goal, m) for m in ("BFS", "DFS", "A*")}
    free = int((~wall).sum())

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.3))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.855, bottom=0.085,
                        wspace=0.10, hspace=0.16)
    wall_cmap = mpl.colors.ListedColormap(["white", OBS_C])

    # (a) the maze ----------------------------------------------------------
    ax = axes[0, 0]
    ax.imshow(wall, origin="lower", cmap=wall_cmap, vmin=0, vmax=1,
              interpolation="nearest")
    markers(ax, start, goal)
    frame(ax, wall, 0)
    ax.set_ylabel("grid cell (y)", fontsize=7.2, labelpad=1.2)
    ax.set_title("Braided maze (%d x %d cells)" % (NC, NR), fontsize=8.4,
                 fontweight="bold", pad=3)
    ax.text(0.982, 0.02, "recursive backtracker + %.0f%% wall removal\n"
            "%d free cells, multiple distinct solutions" % (100 * BRAID, free),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
            linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (b)-(d) expansion order ----------------------------------------------
    for k, m in enumerate(("BFS", "DFS", "A*")):
        ax = axes.ravel()[k + 1]
        r = runs[m]
        img = np.ones((ny, nx, 4))
        vis = r["order"] >= 0
        img[vis] = mpl.cm.plasma(r["order"][vis] / max(r["n"] - 1, 1))
        img[wall] = mpl.colors.to_rgba(OBS_C)
        ax.imshow(img, origin="lower", interpolation="nearest")
        markers(ax, start, goal)
        frame(ax, wall, 0 if k < 2 else 1)
        if k == 2:
            ax.set_ylabel("grid cell (y)", fontsize=7.2, labelpad=1.2)
        ax.set_title("%s expansion order" % m, fontsize=8.4,
                     fontweight="bold", pad=3, color=ALG_C[m])
        ax.text(0.982, 0.02, "%d cells expanded (%.0f%% of free space)"
                % (r["n"], 100 * r["n"] / free), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.0, zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb",
                          lw=0.6, alpha=0.94))

    # (e) all three paths ---------------------------------------------------
    ax = axes[1, 1]
    ax.imshow(wall, origin="lower", cmap=wall_cmap, vmin=0, vmax=1,
              interpolation="nearest")
    styles = {"BFS": (2.6, "-", 1.0), "DFS": (1.6, (0, (4, 2)), 0.95),
              "A*": (0.9, "-", 1.0)}
    for m in ("BFS", "DFS", "A*"):
        p = runs[m]["path"]
        lw, ls, al = styles[m]
        ax.plot(p[:, 1], p[:, 0], color=ALG_C[m], lw=lw, ls=ls, alpha=al,
                solid_capstyle="round", zorder=6 + list(styles).index(m))
    markers(ax, start, goal)
    frame(ax, wall, 1)
    ax.set_title("Returned paths superimposed", fontsize=8.4,
                 fontweight="bold", pad=3)
    ax.text(0.982, 0.02, "BFS and A* coincide (both optimal);\n"
            "DFS commits to a longer corridor", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.0, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (f) effort and quality ------------------------------------------------
    ax = axes[1, 2]
    ax.remove()
    cell = fig.add_gridspec(2, 3, left=0.035, right=0.985, top=0.855,
                            bottom=0.085, wspace=0.10, hspace=0.16)[1, 2]
    pos = cell.get_position(fig)
    panel_bg = Rectangle(
        (pos.x0 - 0.030, pos.y0 - 0.052), pos.width + 0.050,
        pos.height + 0.108, transform=fig.transFigure, fc="#f4f5f7",
        ec="#d3d7db", lw=0.7, zorder=-5, clip_on=False)
    fig.add_artist(panel_bg)
    sub = cell.subgridspec(2, 1, hspace=0.42)
    names = ["BFS", "DFS", "A*"]
    ax1 = fig.add_subplot(sub[0])
    vals = [runs[m]["n"] for m in names]
    bars = ax1.bar(names, vals, color=[ALG_C[m] for m in names], width=0.6,
                   edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, str(v),
                 ha="center", fontsize=6.6, fontweight="bold")
    ax1.set_ylim(0, max(vals) * 1.22)
    ax1.set_ylabel("cells expanded", fontsize=7.2, labelpad=1.5)
    ax1.tick_params(labelsize=6.6, length=2.2)
    ax1.grid(True, axis="y", lw=0.4, color="#e4e4e4")
    ax1.set_axisbelow(True)
    ax1.set_title("Supporting statistic  |  effort and path quality",
                  fontsize=7.2, style="italic", color="#555555", pad=3)

    ax2 = fig.add_subplot(sub[1])
    lens = [len(runs[m]["path"]) - 1 for m in names]
    bars = ax2.bar(names, lens, color=[ALG_C[m] for m in names], width=0.6,
                   edgecolor="white", linewidth=0.6)
    best = min(lens)
    for b, v in zip(bars, lens):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(lens) * 0.03,
                 "%d  (%.0f%%)" % (v, 100 * v / best), ha="center",
                 fontsize=6.2, fontweight="bold")
    ax2.axhline(best, color="#666666", lw=0.7, ls=(0, (2.5, 1.6)))
    ax2.set_ylim(0, max(lens) * 1.26)
    ax2.set_ylabel("path length (cells)", fontsize=7.2, labelpad=1.5)
    ax2.tick_params(labelsize=6.6, length=2.2)
    ax2.grid(True, axis="y", lw=0.4, color="#e4e4e4")
    ax2.set_axisbelow(True)
    for a in (ax1, ax2):
        a.set_facecolor("#fbfbfc")
        for s in a.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color("#9aa0a6")

    labs = list("abcde")
    for k, ax in enumerate([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0],
                            axes[1, 1]]):
        add_panel_label(ax, "(%s)" % labs[k], x=-0.035, y=1.005, size=9.5)
    add_panel_label(ax1, "(f)", x=-0.235, y=1.02, size=9.5)

    sm = mpl.cm.ScalarMappable(cmap="plasma",
                               norm=mpl.colors.Normalize(0, 1))
    cb = add_colorbar(fig, sm, axes[0, 2], label="normalised expansion order",
                      fraction=0.032, pad=0.015, aspect=15,
                      ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["first", "", "last"], fontsize=6.0)

    handles = [Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="entrance S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="exit G"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="wall cell")] + \
              [Line2D([], [], color=ALG_C[m], lw=1.8, label="%s path" % m)
               for m in names]
    savefig(fig, "170_maze_solving.png")


if __name__ == "__main__":
    main()
