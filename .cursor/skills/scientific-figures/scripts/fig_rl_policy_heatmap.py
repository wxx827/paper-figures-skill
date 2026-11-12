"""Greedy policy arrows over a state-visitation heat map in a grid world.

The policy is not decorative: value iteration is run on the maze, the softmax
policy pi(a|s) is read off the resulting Q values, and the visitation map is the
occupancy of 600 sampled rollouts from the start cell.  Arrow direction is the
greedy action, arrow length is max_a pi(a|s), so a short arrow really is an
undecided state - which the attached entropy panel confirms cell by cell.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import safe_colorbar, shared_norm, axes_inches, panel_label_inches

H, W = 11, 16
START, GOAL = (1, 1), (9, 14)          # (row, col); row is the y axis
GAMMA, STEP_COST = 0.97, 0.045
BETA_0 = 2.0                           # exploration policy used to collect data
BETA_LO, BETA_HI = 0.9, 9.0            # trained policy: sharp only where visited
N_ROLLOUT, MAX_STEP = 600, 260
# (dy, dx) for up / down / left / right
MOVES = np.array([(1, 0), (-1, 0), (0, -1), (0, 1)], dtype=float)

WALL_C = "#3a3a3a"
START_C = "#2e7d5b"
GOAL_C = "#b23a48"


def maze():
    w = np.zeros((H, W), dtype=bool)
    w[0:7, 4] = True                 # long wall, gap along the top
    w[4:11, 9] = True                # second wall, gap along the bottom
    w[7:9, 11:14] = True             # blocking slab short of the goal
    w[2, 6:8] = True
    w[START] = w[GOAL] = False
    return w


def value_iteration(walls):
    v = np.zeros((H, W))
    free = ~walls
    q = np.zeros((4, H, W))
    for _ in range(800):
        for a, (dy, dx) in enumerate(MOVES.astype(int)):
            nxt = np.roll(np.roll(v, -dy, axis=0), -dx, axis=1)
            valid = np.roll(np.roll(free, -dy, axis=0), -dx, axis=1)
            if dy == 1:
                valid[-1, :] = False
            if dy == -1:
                valid[0, :] = False
            if dx == 1:
                valid[:, -1] = False
            if dx == -1:
                valid[:, 0] = False
            # bumping into a wall or the border keeps the agent in place
            q[a] = -STEP_COST + GAMMA * np.where(valid, nxt, v)
        nv = q.max(axis=0)
        nv[GOAL] = 1.0
        nv[walls] = 0.0
        if np.abs(nv - v).max() < 1e-10:
            v = nv
            break
        v = nv
    return v, q


def softmax_policy(q, walls, beta):
    """Boltzmann policy on advantages rescaled by their typical magnitude.

    ``beta`` may be a scalar or a per-state map: a policy trained on its own
    rollouts only becomes sharp where it has collected data, which is exactly
    the coupling this figure is about.
    """
    adv = q - q.max(axis=0, keepdims=True)
    gap = np.abs(adv[adv < 0])
    scale = float(np.median(gap)) if gap.size else 1.0
    p = np.exp(np.asarray(beta) * adv / max(scale, 1e-9))
    p /= p.sum(axis=0, keepdims=True)
    p[:, walls] = np.nan
    return p


def rollouts(p, walls, rng):
    vis = np.zeros((H, W))
    for _ in range(N_ROLLOUT):
        r, c = START
        for _ in range(MAX_STEP):
            vis[r, c] += 1.0
            if (r, c) == GOAL:
                break
            a = rng.choice(4, p=p[:, r, c])
            dy, dx = MOVES[a].astype(int)
            nr, nc = r + dy, c + dx
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                r, c = nr, nc
    return vis / vis.sum() * 100.0        # % of all visited steps


def draw_cells(ax, walls):
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    for r, c in zip(*np.where(walls)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=WALL_C,
                               edgecolor="white", linewidth=0.4, zorder=4))


def mark_endpoints(ax, size=9.0, fs=6.4):
    ax.plot([START[1]], [START[0]], marker="o", ms=size, mfc=START_C,
            mec="white", mew=1.0, zorder=8)
    ax.plot([GOAL[1]], [GOAL[0]], marker="*", ms=size * 1.7, mfc=GOAL_C,
            mec="white", mew=0.9, zorder=8)
    for (r, c), name, col in (((START), "start", START_C),
                              ((GOAL), "goal", GOAL_C)):
        t = ax.text(c, r - 0.75, name, ha="center", va="top", fontsize=fs,
                    color=col, fontweight="bold", zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white")])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4350)
    walls = maze()
    v, q = value_iteration(walls)

    # pass 1: exploration policy -> where the agent actually gathers data
    explore = softmax_policy(q, walls, BETA_0)
    seen = np.log1p(rollouts(explore, walls, rng))
    seen /= seen.max()
    # pass 2: the trained policy is sharp only on that support
    beta = BETA_LO + (BETA_HI - BETA_LO) * seen
    p = softmax_policy(q, walls, beta)
    vis = rollouts(p, walls, rng)

    pf = np.nan_to_num(p, nan=0.0)
    conf = pf.max(axis=0)                             # max_a pi(a|s)
    greedy = pf.argmax(axis=0)
    ent = -np.nansum(p * np.log2(np.clip(p, 1e-12, 1)), axis=0)
    ent[walls] = np.nan
    conf[walls] = np.nan

    fig = plt.figure(figsize=(9.0, 5.0))
    mw, mh = 4.60, 4.60 * H / W                       # main grid, equal aspect
    ax = axes_inches(fig, (0.50, 0.72, mw, mh))
    aw = 0.58 * mw
    axe = axes_inches(fig, (6.05, 0.72 + mh - aw * H / W, aw, aw * H / W))

    norm = shared_norm([vis[vis > 0]], robust=(1, 98))
    im = ax.imshow(vis, cmap=get_cmap("cb_sequential_blue"), norm=norm,
                   origin="lower", interpolation="nearest", zorder=1)
    draw_cells(ax, walls)

    # arrows: ink on pale cells, white on saturated ones, so both stay legible
    rows, cols = np.where(~walls)
    shade = norm(np.clip(vis, norm.vmin, norm.vmax))
    lo_c, hi_c = np.nanmin(conf), np.nanmax(conf)
    for light, col, alp in ((False, "#1a1a1a", 0.85), (True, "white", 0.95)):
        sel = [(r, c) for r, c in zip(rows, cols)
               if ((r, c) != GOAL) and ((shade[r, c] > 0.62) == light)]
        if not sel:
            continue
        rr = np.array([s[0] for s in sel])
        cc = np.array([s[1] for s in sel])
        length = 0.26 + 0.54 * (conf[rr, cc] - lo_c) / (hi_c - lo_c + 1e-12)
        d = MOVES[greedy[rr, cc]]
        ax.quiver(cc - d[:, 1] * length / 2, rr - d[:, 0] * length / 2,
                  d[:, 1] * length, d[:, 0] * length, color=col, alpha=alp,
                  angles="xy", scale_units="xy", scale=1.0, width=0.0042,
                  headwidth=3.6, headlength=4.0, headaxislength=3.4, zorder=6)

    mark_endpoints(ax)
    ax.set_title("Greedy action and its confidence over visitation density",
                 fontsize=9.0, fontweight="bold", pad=5)
    safe_colorbar(fig, im, ax, label="state visitation (% of steps)",
                  width=0.11, pad=0.09)

    ime = axe.imshow(ent, cmap=get_cmap("cb_sequential_warm"), origin="lower",
                     interpolation="nearest", vmin=0.0, vmax=2.0, zorder=1)
    draw_cells(axe, walls)
    mark_endpoints(axe, size=6.0, fs=6.0)
    axe.set_title("Action entropy $H[\\pi_\\theta(\\cdot|s)]$", fontsize=8.5,
                  fontweight="bold", pad=5)
    safe_colorbar(fig, ime, axe, side="bottom", label="entropy (bits, max 2)",
                  width=0.10, pad=0.10, labelsize=6.5)

    for a, lab in ((ax, "(a)"), (axe, "(b)")):
        panel_label_inches(fig, a, lab, dx=-0.26, dy=0.14)

    handles = [
        Line2D([], [], color="#1a1a1a", lw=1.0, marker=">", ms=4,
               label="greedy action, length $\\propto \\max_a \\pi_\\theta(a|s)$"),
        Line2D([], [], color="none", marker="s", ms=6, mfc=WALL_C,
               mec="white", label="obstacle"),
        Line2D([], [], color="none", marker="o", ms=6, mfc=START_C,
               mec="white", label="start"),
        Line2D([], [], color="none", marker="*", ms=9, mfc=GOAL_C,
               mec="white", label="goal"),
    ]
    busy = vis > np.nanpercentile(vis[vis > 0], 70)
    savefig(fig, "435_rl_policy_heatmap.png")


if __name__ == "__main__":
    main()
