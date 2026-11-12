"""A* grid search anatomy: environment, expansion wavefront, optimal path.

A real 8-connected A* is run on a 0.20 m occupancy grid of a cluttered
20 x 14 m workspace.  Panel (b) colours every closed cell by its expansion
rank, so the heuristic-biased, goal-elongated wavefront is directly visible
and can be contrasted with the concentric growth of uninformed search; the
still-open frontier is drawn in amber.  Panel (c) overlays iso-cost-to-come
contours on the recovered minimum-cost path.
"""
import heapq

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

# ---- shared workspace ------------------------------------------------------
W, H = 20.0, 14.0
RES = 0.20
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = (1.3, 1.5)
GOAL = (18.7, 12.6)

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"


def build_occupancy(res=RES):
    nx, ny = int(round(W / res)), int(round(H / res))
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), bool)
    for x0, y0, w, h in RECTS:
        occ |= (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)
    for cx, cy, r in CIRCLES:
        occ |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    return occ, xs, ys


def to_idx(p, res=RES):
    return (int(p[1] / res), int(p[0] / res))


S2 = np.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, S2), (-1, 1, S2), (1, -1, S2), (1, 1, S2)]


def grid_search(occ, start, goal, mode="astar", eps=1.0):
    """8-connected A* / Dijkstra / greedy best-first with expansion logging."""
    ny, nx = occ.shape
    g = np.full((ny, nx), np.inf)
    order = np.full((ny, nx), -1, np.int32)
    par = np.full((ny, nx, 2), -1, np.int32)
    closed = np.zeros((ny, nx), bool)
    opened = np.zeros((ny, nx), bool)

    def h(i, j):
        dy, dx = abs(i - goal[0]), abs(j - goal[1])
        return (dx + dy) + (S2 - 2.0) * min(dx, dy)

    g[start] = 0.0
    opened[start] = True
    heap = [(eps * h(*start), 0.0, start)]
    cnt = 0
    while heap:
        _, gc, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        order[i, j] = cnt
        cnt += 1
        if (i, j) == goal:
            break
        for di, dj, c in MOVES:
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx) or occ[ii, jj]:
                continue
            if di and dj and (occ[i, jj] or occ[ii, j]):
                continue          # forbid corner cutting through diagonals
            ng = gc + c
            if ng < g[ii, jj] - 1e-12:
                g[ii, jj] = ng
                par[ii, jj] = (i, j)
                opened[ii, jj] = True
                key = ng if mode == "dijkstra" else (
                    h(ii, jj) if mode == "greedy" else ng + eps * h(ii, jj))
                heapq.heappush(heap, (key, ng, (ii, jj)))

    path = []
    if closed[goal]:
        cur = goal
        while cur != start:
            path.append(cur)
            cur = (int(par[cur][0]), int(par[cur][1]))
        path.append(start)
        path = path[::-1]
    return dict(g=g, order=order, closed=closed, opened=opened,
                path=np.array(path, int).reshape(-1, 2), n=cnt)


# ---- drawing helpers -------------------------------------------------------
def frame(ax, xlabel=True, ylabel=False):
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
    if ylabel:
        ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)


def start_goal(ax, size=8.0):
    ax.plot(*START, marker="^", ms=size, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    ax.plot(*GOAL, marker="*", ms=size * 1.65, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=8)


def cells_to_xy(cells, res=RES):
    return (cells[:, 1] + 0.5) * res, (cells[:, 0] + 0.5) * res


def main():
    set_pub_style(base_font=8)
    occ, xs, ys = build_occupancy()
    si, gi = to_idx(START), to_idx(GOAL)
    res = grid_search(occ, si, gi, mode="astar")
    dij = grid_search(occ, si, gi, mode="dijkstra")

    ny, nx = occ.shape
    ext = [0, W, 0, H]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.55))
    fig.subplots_adjust(left=0.045, right=0.965, top=0.79, bottom=0.14,
                        wspace=0.20)

    # (a) raw environment ---------------------------------------------------
    ax = axes[0]
    ax.imshow(np.where(occ, 1.0, 0.0), extent=ext, origin="lower",
              cmap=mpl.colors.ListedColormap(["white", OBS_C]),
              vmin=0, vmax=1, interpolation="nearest")
    for gx in np.arange(0, W + 1e-9, 1.0):
        ax.axvline(gx, color="#e2e2e2", lw=0.3, zorder=0)
    for gy in np.arange(0, H + 1e-9, 1.0):
        ax.axhline(gy, color="#e2e2e2", lw=0.3, zorder=0)
    start_goal(ax)
    frame(ax, ylabel=True)
    ax.set_title("Workspace and occupancy grid", fontsize=8.4,
                 fontweight="bold", pad=3)
    ax.text(0.5, -0.235, r"$100\times70$ cells, 0.20 m resolution, "
            "8-connected", transform=ax.transAxes, ha="center", fontsize=6.8,
            color="#666")
    ax.annotate("S", xy=START, xytext=(START[0] + 0.55, START[1] + 0.5),
                fontsize=7.6, fontweight="bold", color=S_C)
    ax.annotate("G", xy=GOAL, xytext=(GOAL[0] - 1.45, GOAL[1] - 0.35),
                fontsize=7.6, fontweight="bold", color=G_C)

    # (b) expansion order ---------------------------------------------------
    ax = axes[1]
    img = np.ones((ny, nx, 4))
    vis = res["order"] >= 0
    img[vis] = mpl.cm.viridis(res["order"][vis] / max(res["n"] - 1, 1))
    front = res["opened"] & ~res["closed"] & ~occ
    img[front] = mpl.colors.to_rgba("#c9a227")
    img[occ] = mpl.colors.to_rgba(OBS_C)
    ax.imshow(img, extent=ext, origin="lower", interpolation="nearest")
    start_goal(ax)
    frame(ax)
    ax.set_title("Expansion order of the closed set", fontsize=8.4,
                 fontweight="bold", pad=3)
    sm = mpl.cm.ScalarMappable(cmap="viridis",
                               norm=mpl.colors.Normalize(0, res["n"]))
    cb = add_colorbar(fig, sm, ax, label="expansion rank (nodes)",
                      fraction=0.036, pad=0.02, aspect=18)
    cb.set_ticks([0, res["n"] // 2, res["n"]])
    ax.text(0.5, -0.235, "wavefront is stretched toward G by the octile "
            "heuristic", transform=ax.transAxes, ha="center", fontsize=6.8,
            color="#666")
    ax.text(0.975, 0.03, "A*: %d nodes expanded\nDijkstra: %d nodes\n"
            "%.0f%% fewer expansions" % (res["n"], dij["n"],
                                         100 * (1 - res["n"] / dij["n"])),
            transform=ax.transAxes, va="bottom", ha="right", fontsize=6.6,
            linespacing=1.45, zorder=9,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.93))

    # (c) path + cost-to-come field ----------------------------------------
    ax = axes[2]
    ax.imshow(np.where(occ, 1.0, 0.0), extent=ext, origin="lower",
              cmap=mpl.colors.ListedColormap(["white", OBS_C]),
              vmin=0, vmax=1, interpolation="nearest")
    gm = np.ma.masked_invalid(np.where(occ, np.nan, dij["g"] * RES))
    X, Y = np.meshgrid(xs, ys)
    cf = ax.contourf(X, Y, gm, levels=np.arange(0, 34, 1.5), cmap="Blues_r",
                     alpha=0.55, zorder=0.5)
    ax.contour(X, Y, gm, levels=np.arange(0, 34, 3.0), colors="#3b5b92",
               linewidths=0.45, alpha=0.85, zorder=1)
    px, py = cells_to_xy(res["path"])
    ax.plot(px, py, color="white", lw=3.4, solid_capstyle="round", zorder=6)
    ax.plot(px, py, color=G_C, lw=2.0, solid_capstyle="round", zorder=7,
            label="A* optimal path")
    start_goal(ax)
    frame(ax)
    ax.set_title("Minimum-cost path and cost-to-come field", fontsize=8.4,
                 fontweight="bold", pad=3)
    cb = add_colorbar(fig, cf, ax, label="cost-to-come (m)", fraction=0.036,
                      pad=0.02, aspect=18)
    cost = res["g"][gi] * RES
    straight = np.hypot(GOAL[0] - START[0], GOAL[1] - START[1])
    ax.text(0.975, 0.03, "path cost  %.2f m\nEuclidean bound  %.2f m\n"
            "detour ratio  %.2f" % (cost, straight, cost / straight),
            transform=ax.transAxes, va="bottom", ha="right", fontsize=6.6,
            linespacing=1.45, zorder=9,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.93))
    ax.text(0.5, -0.235, "iso-cost contours every 3 m bend around obstacles",
            transform=ax.transAxes, ha="center", fontsize=6.8, color="#666")

    for ax, lab in zip(axes, "abc"):
        add_panel_label(ax, "(%s)" % lab, x=-0.055, y=1.035, size=10)

    handles = [Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], color=G_C, lw=2.2, label="optimal path"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc="#c9a227",
                      mec="none", label="open set (frontier)"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "163_astar_expansion.png")


if __name__ == "__main__":
    main()
