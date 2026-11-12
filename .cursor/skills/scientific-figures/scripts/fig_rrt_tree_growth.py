"""RRT* tree growth and asymptotic optimality over four sampling budgets.

A full RRT* (steer, choose-parent, rewire) is executed for 3000 samples in
the same 20 x 14 m workspace used by the other planning figures.  Snapshots
of the *entire* tree are deep-copied at 100 / 500 / 1500 / 3000 samples so
that each panel shows the true parent structure at that instant, together
with the best solution found so far.  The inset in panel (d) tracks the
monotonically decreasing solution cost that certifies asymptotic optimality.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

W, H = 20.0, 14.0
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = np.array([1.3, 1.5])
GOAL = np.array([18.7, 12.6])

ROB_R = 0.16          # robot inflation radius (m)
ETA = 1.10            # steer step (m)
GAMMA = 26.0          # RRT* connection-radius constant
GOAL_TOL = 0.70       # radius of the goal region (m)
GOAL_BIAS = 0.12

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"


def collision(x, y, pad=ROB_R):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x < pad) | (x > W - pad) | (y < pad) | (y > H - pad)
    for x0, y0, w, h in RECTS:
        m |= ((x >= x0 - pad) & (x <= x0 + w + pad) &
              (y >= y0 - pad) & (y <= y0 + h + pad))
    for cx, cy, r in CIRCLES:
        m |= (x - cx) ** 2 + (y - cy) ** 2 <= (r + pad) ** 2
    return m


def seg_free(p, Q, n=14):
    """Vectorised straight-line collision test from p to every row of Q."""
    Q = np.atleast_2d(Q)
    t = np.linspace(0.0, 1.0, n)[:, None]
    X = p[0] + (Q[:, 0] - p[0])[None, :] * t
    Y = p[1] + (Q[:, 1] - p[1])[None, :] * t
    return ~collision(X, Y).any(axis=0)


def rrt_star(n_iter=3000, snaps=(100, 500, 1500, 3000), seed=7):
    rng = np.random.default_rng(seed)
    nodes = np.zeros((n_iter + 1, 2))
    nodes[0] = START
    parent = np.full(n_iter + 1, -1, int)
    cost = np.zeros(n_iter + 1)
    n = 1
    goal_nodes = []
    hist = np.full(n_iter + 1, np.nan)
    out = {}

    for it in range(1, n_iter + 1):
        if rng.random() < GOAL_BIAS:
            q = GOAL.copy()
        else:
            q = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        d = np.linalg.norm(nodes[:n] - q, axis=1)
        k = int(np.argmin(d))
        if d[k] > ETA:
            q = nodes[k] + (q - nodes[k]) * (ETA / d[k])
        accept = not (collision(q[0], q[1]) or
                      not seg_free(nodes[k], q[None])[0])
        if accept:
            r = min(GAMMA * np.sqrt(np.log(n + 1) / (n + 1)), ETA * 2.4)
            dist = np.linalg.norm(nodes[:n] - q, axis=1)
            near = np.flatnonzero(dist <= r)
            if near.size == 0:
                near = np.array([k])
            ok = seg_free(q, nodes[near])
            near, dn = near[ok], dist[near][ok]
            if near.size == 0:
                near = np.array([k])
                dn = np.array([np.linalg.norm(nodes[k] - q)])

            cand = cost[near] + dn
            b = int(np.argmin(cand))
            nodes[n] = q
            parent[n] = near[b]
            cost[n] = cand[b]
            new = n
            n += 1

            # rewire: adopt the new node as parent whenever it lowers the cost
            improve = cost[new] + dn - cost[near]
            for idx in np.flatnonzero(improve < -1e-9):
                j = int(near[idx])
                if j == parent[new] or j == 0:
                    continue
                delta = cost[j] - (cost[new] + dn[idx])
                if delta <= 1e-9:
                    continue
                parent[j] = new
                cost[j] -= delta
                stack = [j]
                while stack:                  # propagate to the subtree
                    u = stack.pop()
                    for c in np.flatnonzero(parent[:n] == u):
                        if c == u:
                            continue
                        cost[c] -= delta
                        stack.append(int(c))

            if (np.linalg.norm(q - GOAL) <= GOAL_TOL and
                    seg_free(q, GOAL[None])[0]):
                goal_nodes.append(new)

        hist[it] = min([cost[i] + np.linalg.norm(nodes[i] - GOAL)
                        for i in goal_nodes], default=np.nan)

        if it in snaps:
            best, bc = -1, np.inf
            for i in goal_nodes:
                c = cost[i] + np.linalg.norm(nodes[i] - GOAL)
                if c < bc:
                    best, bc = i, c
            path = None
            if best >= 0:
                seq, cur = [GOAL], best
                while cur != -1:
                    seq.append(nodes[cur].copy())
                    cur = parent[cur]
                path = np.array(seq[::-1])
            out[it] = dict(nodes=nodes[:n].copy(), parent=parent[:n].copy(),
                           n=n, path=path, cost=bc)
    return out, hist


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
    ax.tick_params(length=2.4, labelsize=6.6)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_yticklabels([])


def main():
    set_pub_style(base_font=8)
    snaps = (100, 500, 1500, 3000)
    out, hist = rrt_star(snaps=snaps)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.15))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.075,
                        wspace=0.06, hspace=0.10)

    for k, it in enumerate(snaps):
        r, c = divmod(k, 2)
        ax = axes[r, c]
        s = out[it]
        nodes, par = s["nodes"], s["parent"]

        segs = [[nodes[i], nodes[par[i]]] for i in range(1, s["n"])]
        ax.add_collection(LineCollection(segs, colors="#9aa3ab", linewidths=0.35,
                                         alpha=0.85, zorder=2))
        ax.plot(nodes[1:s["n"], 0], nodes[1:s["n"], 1], ls="none", marker="o",
                ms=0.9, mfc="#5c6670", mec="none", alpha=0.75, zorder=3)
        draw_obstacles(ax)
        ax.add_patch(Circle(GOAL, GOAL_TOL, fc="none", ec=G_C, lw=0.7,
                            ls=(0, (2.5, 1.6)), zorder=5))

        if s["path"] is not None:
            ax.plot(s["path"][:, 0], s["path"][:, 1], color="white", lw=3.2,
                    solid_capstyle="round", zorder=6)
            ax.plot(s["path"][:, 0], s["path"][:, 1], color=G_C, lw=1.9,
                    solid_capstyle="round", zorder=7)
            txt = "%d samples\n%d tree nodes\nbest cost  %.2f m" % (
                it, s["n"], s["cost"])
        else:
            txt = "%d samples\n%d tree nodes\nno solution yet" % (it, s["n"])

        ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
                ls="none", zorder=9)
        ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
                ls="none", zorder=9)
        ax.text(0.018, 0.975, txt, transform=ax.transAxes, va="top",
                ha="left", fontsize=6.8, linespacing=1.45, zorder=10,
                bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#bbbbbb",
                          lw=0.6, alpha=0.94))
        frame(ax, r, c)
        add_panel_label(ax, "(%s)" % "abcd"[k], x=-0.045, y=1.005, size=10)
        ax.set_title("after %d samples" % it, fontsize=8.4,
                     fontweight="bold", pad=3)

    # convergence inset in the last panel -----------------------------------
    ax = axes[1, 1]
    ax.add_patch(Rectangle((13.55, 0.25), 6.20, 4.70, fc="white",
                           ec="#aaaaaa", lw=0.6, zorder=11))
    ins = ax.inset_axes([0.725, 0.065, 0.225, 0.235], zorder=12)
    it = np.arange(len(hist))
    good = np.isfinite(hist)
    ins.plot(it[good], hist[good], color="#3b5b92", lw=1.0)
    for s in snaps:
        if np.isfinite(hist[s]):
            ins.plot(s, hist[s], marker="o", ms=2.4, mfc=G_C, mec="white",
                     mew=0.4, zorder=5)
    ins.set_xlim(0, len(hist))
    lo = np.nanmin(hist)
    ins.set_ylim(lo - 0.25, np.nanmax(hist[good]) + 0.35)
    ins.axhline(lo, color=G_C, lw=0.55, ls=(0, (2.5, 1.6)))
    ins.set_xlabel("samples", fontsize=6.0, labelpad=0.6)
    ins.set_ylabel("cost (m)", fontsize=6.0, labelpad=0.8)
    ins.set_xticks([0, 1500, 3000])
    ins.tick_params(labelsize=6.0, length=1.6, pad=1.0)
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#888888")
    ins.set_title("solution cost vs. samples", fontsize=6.0, pad=1.4,
                  color="#333")

    handles = [Line2D([], [], color="#9aa3ab", lw=0.9, label="RRT* tree edge"),
               Line2D([], [], color=G_C, lw=2.0, label="best path so far"),
               Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], color=G_C, lw=0.9, ls=(0, (2.5, 1.6)),
                      label="goal region (r = 0.70 m)"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "164_rrt_tree_growth.png")


if __name__ == "__main__":
    main()
