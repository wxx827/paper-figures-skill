"""One value function seen three ways: surface, contour with a rollout, Q per action.

All three panels are driven by a single value-iteration solve on a 2-D
navigation task with two cost ridges, and all three are drawn through the *same*
normalisation - V(s) and every Q(s, a) map share one colour bar, because they
are the same quantity in the same units.  The trajectory in panel (b) is a
greedy rollout of the solved policy, not a hand-drawn curve.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers "3d")

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import shared_norm, panel_label_inches

N = 76
EXTENT = (0.0, 10.0)
GOAL = (8.4, 7.9)
START = (1.1, 1.4)
GAMMA = 0.99
GOAL_R = 10.0        # terminal reward; without it "stay put" ties with walking
# eight-connected moves keep the greedy rollout from degenerating into a
# staircase along the domain border
MOVES = [(1, 0), (1, 1), (0, 1), (-1, 1),
         (-1, 0), (-1, -1), (0, -1), (1, -1)]
CARDINAL = {0: "$\\uparrow$ north", 2: "$\\rightarrow$ east",
            4: "$\\downarrow$ south", 6: "$\\leftarrow$ west"}
GOAL_C = "#b23a48"
START_C = "#2e7d5b"
PATH_C = "#1a1a1a"


def cost_field(X, Y, rng):
    """Per-step cost: two ridges plus low-frequency texture (no ties)."""
    c = 0.030 * np.ones_like(X)
    c += 0.90 * np.exp(-(((X - 4.2) ** 2) / 0.9 + ((Y - 5.6) ** 2) / 9.0))
    c += 0.65 * np.exp(-(((X - 7.3) ** 2) / 3.2 + ((Y - 2.2) ** 2) / 0.8))
    # clearance from the domain walls, otherwise every optimal path hugs a border
    lo, hi = EXTENT
    for d in (X - lo, hi - X, Y - lo, hi - Y):
        c += 0.30 * np.exp(-d / 0.9)
    for _ in range(9):
        cx, cy = rng.uniform(0.5, 9.5, 2)
        c += rng.uniform(0.02, 0.07) * np.exp(
            -(((X - cx) ** 2) + ((Y - cy) ** 2)) / rng.uniform(2.5, 7.0))
    return c


def solve(rng):
    xs = np.linspace(*EXTENT, N)
    X, Y = np.meshgrid(xs, xs)
    cost = cost_field(X, Y, rng)
    gi = (int(np.argmin(np.abs(xs - GOAL[1]))),
          int(np.argmin(np.abs(xs - GOAL[0]))))
    v = np.zeros((N, N))
    q = np.zeros((len(MOVES), N, N))
    for _ in range(1500):
        for a, (dy, dx) in enumerate(MOVES):
            step = np.hypot(dy, dx)
            vn = np.roll(np.roll(v, -dy, axis=0), -dx, axis=1)
            cn = np.roll(np.roll(cost, -dy, axis=0), -dx, axis=1)
            if dy:                          # stepping off the grid stays put
                k = -1 if dy == 1 else 0
                vn[k, :], cn[k, :] = v[k, :], cost[k, :]
            if dx:
                k = -1 if dx == 1 else 0
                vn[:, k], cn[:, k] = v[:, k], cost[:, k]
            q[a] = -cn * step + GAMMA ** step * vn
        nv = q.max(axis=0)
        nv[gi] = GOAL_R
        if np.abs(nv - v).max() < 1e-9:
            v = nv
            break
        v = nv
    return xs, X, Y, v, q, gi


def rollout(xs, q, gi):
    """Greedy path from START, lightly smoothed for display."""
    j = int(np.argmin(np.abs(xs - START[0])))
    i = int(np.argmin(np.abs(xs - START[1])))
    pts = [(xs[j], xs[i])]
    for _ in range(600):
        dy, dx = MOVES[int(np.argmax(q[:, i, j]))]
        i, j = int(np.clip(i + dy, 0, N - 1)), int(np.clip(j + dx, 0, N - 1))
        pts.append((xs[j], xs[i]))
        if (i, j) == gi:
            break
    print(f"rollout: {len(pts) - 1} steps, reached goal: {(i, j) == gi}")
    p = np.asarray(pts)
    k = np.ones(5) / 5.0
    sm = np.column_stack([np.convolve(p[:, c], k, mode="valid") for c in (0, 1)])
    return np.vstack([p[0], sm, p[-1]]), len(pts) - 1


def tile(ax, img, norm, cmap, title=None):
    ax.imshow(img, origin="lower", extent=(*EXTENT, *EXTENT), norm=norm,
              cmap=cmap, interpolation="bilinear")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    if title:
        ax.set_title(title, fontsize=7.2, color="#1a1a1a", pad=2.6)
    return ax


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4360)
    xs, X, Y, v, q, gi = solve(rng)
    path, n_step = rollout(xs, q, gi)
    cmap = get_cmap("cb_sequential_blue")
    norm = shared_norm([v] + [q[a] for a in CARDINAL], robust=(1.5, 100))

    fig = plt.figure(figsize=(10.4, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 0.95, 1.0],
                          left=0.015, right=0.985, top=0.865, bottom=0.215,
                          wspace=0.16)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axc = fig.add_subplot(gs[0, 1])
    sub = gs[0, 2].subgridspec(2, 2, hspace=0.30, wspace=0.10)
    axq = [fig.add_subplot(sub[i // 2, i % 2]) for i in range(4)]

    # ---- (a) surface ------------------------------------------------------
    st = 2
    ax3.plot_surface(X[::st, ::st], Y[::st, ::st], v[::st, ::st], cmap=cmap,
                     norm=norm, linewidth=0.15, edgecolor="white",
                     antialiased=True, rstride=1, cstride=1, alpha=0.99)
    floor = norm.vmin * 1.30
    ax3.contour(X, Y, v, levels=10, zdir="z", offset=floor, colors="#9a9a9a",
                linewidths=0.4)
    ax3.set_zlim(floor, norm.vmax * 1.08)
    ax3.set_xlim(*EXTENT)
    ax3.set_ylim(*EXTENT)
    ax3.view_init(elev=30, azim=-124)
    ax3.set_box_aspect((1, 1, 0.66))
    ax3.set_xlabel("$s_x$ (m)", fontsize=7, labelpad=-4)
    ax3.set_ylabel("$s_y$ (m)", fontsize=7, labelpad=-4)
    ax3.set_zlabel("$V(s)$ (return)", fontsize=7, labelpad=4)
    ax3.tick_params(labelsize=6.2, pad=-1.5)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("#d9d9d9")
    ax3.grid(color="#e8e8e8", linewidth=0.4)
    ax3.set_title("$V(s)$ as a surface", fontsize=8.6, fontweight="bold",
                  pad=-4)

    # ---- (b) contour + rollout -------------------------------------------
    cf = axc.contourf(X, Y, v, levels=16, cmap=cmap, norm=norm)
    cf.set_edgecolor("face")
    cl = axc.contour(X, Y, v, levels=7, colors="white", linewidths=0.5,
                     alpha=0.7)
    for t in axc.clabel(cl, fmt="%.0f", fontsize=6.0, inline=True,
                        colors="#1a1a1a"):
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    axc.plot(path[:, 0], path[:, 1], color="white", lw=2.8, alpha=0.9,
             solid_capstyle="round", zorder=5)
    axc.plot(path[:, 0], path[:, 1], color=PATH_C, lw=1.5, zorder=6,
             solid_capstyle="round")
    step = max(len(path) // 6, 1)
    anchor = path[::step]
    d = np.diff(anchor, axis=0)
    axc.quiver(anchor[:-1, 0], anchor[:-1, 1], d[:, 0], d[:, 1], color=PATH_C,
               angles="xy", scale_units="xy", scale=1.0, width=0.009,
               headwidth=4.0, headlength=4.6, zorder=7)
    axc.plot([START[0]], [START[1]], marker="o", ms=6.5, mfc=START_C,
             mec="white", mew=1.0, zorder=8)
    axc.plot([GOAL[0]], [GOAL[1]], marker="*", ms=13, mfc=GOAL_C, mec="white",
             mew=0.9, zorder=8)
    for (px, py), name, col, dy in (((START), "start", START_C, -0.5),
                                    ((GOAL), "goal", GOAL_C, -0.6)):
        t = axc.text(px, py + dy, name, fontsize=6.6, color=col,
                     fontweight="bold", ha="center", va="top", zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white")])
    axc.set_aspect("equal")
    axc.set_xlim(*EXTENT)
    axc.set_ylim(*EXTENT)
    axc.set_xlabel("$s_x$ (m)", fontsize=8)
    axc.set_ylabel("$s_y$ (m)", fontsize=8)
    axc.tick_params(labelsize=7.0)
    axc.set_title("Top view with the greedy rollout", fontsize=8.6,
                  fontweight="bold", pad=4)

    # ---- (c) Q(s, a) faceted by action ------------------------------------
    greedy = q.argmax(axis=0)
    for ax, (a, name) in zip(axq, CARDINAL.items()):
        tile(ax, q[a], norm, cmap, title=name)
        # outline where this action is the greedy one: the four tiles are
        # otherwise near-identical, because Q differs from V by one cell
        ax.contour(X, Y, (greedy == a).astype(float), levels=[0.5],
                   colors=GOAL_C, linewidths=0.8, zorder=4)
        ax.plot([GOAL[0]], [GOAL[1]], marker="*", ms=6.0, mfc=GOAL_C,
                mec="white", mew=0.6, zorder=5)
        t = ax.text(0.045, 0.045, f"greedy on {(greedy == a).mean():.0%}",
                    transform=ax.transAxes, fontsize=6.0, color="#1a1a1a",
                    ha="left", va="bottom", zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    cax = fig.add_axes([0.355, 0.108, 0.30, 0.026])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("state value / action value (discounted return)",
                 fontsize=7.0, color="#1a1a1a")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")

    panel_label_inches(fig, ax3, "(a)", dx=0.02, dy=0.02)
    panel_label_inches(fig, axc, "(b)", dx=-0.34, dy=0.10)
    panel_label_inches(fig, axq[0], "(c)", dx=-0.26, dy=0.16)

    v0 = v[int(np.argmin(np.abs(xs - START[1]))),
           int(np.argmin(np.abs(xs - START[0])))]
    savefig(fig, "436_rl_value_function.png")


if __name__ == "__main__":
    main()
