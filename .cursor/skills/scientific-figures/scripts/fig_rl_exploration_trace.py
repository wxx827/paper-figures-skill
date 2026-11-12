"""Eight snapshots of the agent's path, from random walk to a committed route.

Each tile is one episode rolled out with the epsilon of that training stage, so
the tiles and the epsilon curve underneath are the *same* numbers: the schedule
drives the simulation rather than being drawn next to it.  Coverage is the share
of arena cells any of the twelve episodes of that stage touched, and the path is
coloured by step index so the direction of travel is readable without arrows.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig, get_palette, style_axes
from figkit import panel_label_inches

ARENA = 10.0
GOAL = np.array([8.6, 8.4])
START = np.array([1.2, 1.3])
OBSTACLES = [(4.3, 3.0, 1.15), (5.6, 6.9, 1.35), (2.4, 6.6, 0.95),
             (8.0, 4.4, 0.90)]
N_STAGE, N_EP, MAX_STEP = 8, 12, 320
COV_BINS = 22
EPS0, EPS1 = 0.98, 0.04
TOTAL_STEPS = 2.0e6            # env steps spanned by the schedule

EPS_C = "#3b5b92"
COV_C = "#2e7d5b"
GOAL_C = "#b23a48"
START_C = "#2e7d5b"
OBST_C = "#3a3a3a"


def blocked(p):
    for cx, cy, r in OBSTACLES:
        if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < r * r:
            return True
    return not (0.15 <= p[0] <= ARENA - 0.15 and 0.15 <= p[1] <= ARENA - 0.15)


def episode(rng, eps, skill):
    """Epsilon-greedy walk: random direction with prob eps, else toward goal.

    ``skill`` (0 -> 1 over training) sharpens the greedy branch as well, so an
    early tile is messy for two reasons at once, exactly as in a real run.
    """
    p = START.copy()
    pts = [p.copy()]
    heading = rng.uniform(0, 2 * np.pi)
    for _ in range(MAX_STEP):
        if rng.random() < eps:
            heading += rng.normal(0, 1.5)      # momentum keeps it a walk
            d = np.array([np.cos(heading), np.sin(heading)])
        else:
            d = GOAL - p
            d = d / (np.linalg.norm(d) + 1e-9)
            d = d + rng.normal(0, 0.50 - 0.42 * skill, 2)
            for cx, cy, r in OBSTACLES:        # slide around obstacles
                off = p - np.array([cx, cy])
                dist = np.linalg.norm(off)
                if dist < r + 0.9:
                    d = d + (0.6 + 1.1 * skill) * off / (dist ** 2 + 1e-6)
            d = d / (np.linalg.norm(d) + 1e-9)
            heading = np.arctan2(d[1], d[0])
        nxt = p + 0.26 * d
        if blocked(nxt):
            heading += rng.uniform(1.6, 4.7)
            continue
        p = nxt
        pts.append(p.copy())
        if np.linalg.norm(p - GOAL) < 0.45:
            break
    return np.asarray(pts)


def draw_arena(ax):
    ax.set_xlim(0, ARENA)
    ax.set_ylim(0, ARENA)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    for cx, cy, r in OBSTACLES:
        ax.add_patch(Circle((cx, cy), r, facecolor=OBST_C, edgecolor="white",
                            linewidth=0.5, zorder=2))
    ax.plot(*START, marker="o", ms=4.6, mfc=START_C, mec="white", mew=0.8,
            zorder=6)
    ax.plot(*GOAL, marker="*", ms=9.0, mfc=GOAL_C, mec="white", mew=0.7,
            zorder=6)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4370)

    ramp = get_palette("cb_sequential_blue", 7)[1:]
    traj_cmap = mpl.colors.LinearSegmentedColormap.from_list("traj", ramp)

    frac = np.arange(N_STAGE) / (N_STAGE - 1)
    eps = EPS0 * (EPS1 / EPS0) ** (frac ** 1.7)
    skill = frac ** 0.75
    stage_steps = np.linspace(0.0, TOTAL_STEPS, N_STAGE)

    shown, coverage, reached, lengths = [], [], [], []
    for e, sk in zip(eps, skill):
        cells = set()
        best = None
        hits = 0
        for k in range(N_EP):
            tr = episode(rng, e, sk)
            for x, y in tr:
                cells.add((int(x / ARENA * COV_BINS),
                           int(y / ARENA * COV_BINS)))
            done = np.linalg.norm(tr[-1] - GOAL) < 0.5
            hits += int(done)
            if k == 0:
                best = tr
        shown.append(best)
        coverage.append(100.0 * len(cells) / (COV_BINS * COV_BINS))
        reached.append(100.0 * hits / N_EP)
        lengths.append(len(best))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.60],
                          left=0.055, right=0.925, top=0.885, bottom=0.085,
                          hspace=0.30, wspace=0.10)
    tiles = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(N_STAGE)]
    axb = fig.add_subplot(gs[2, :])

    for k, (ax, tr) in enumerate(zip(tiles, shown)):
        draw_arena(ax)
        seg = np.stack([tr[:-1], tr[1:]], axis=1)
        lc = LineCollection(seg, cmap=traj_cmap, linewidths=0.95, zorder=4,
                            capstyle="round",
                            array=np.linspace(0, 1, len(seg)))
        ax.add_collection(lc)
        ax.set_title(f"$\\epsilon = {eps[k]:.2f}$   "
                     f"({stage_steps[k] / 1e6:.2f}M steps)", fontsize=7.6,
                     fontweight="bold", pad=2.8)
        done = np.linalg.norm(tr[-1] - GOAL) < 0.5
        stamp = ax.text(0.045, 0.045,
                        f"{lengths[k]} steps, {'goal' if done else 'timeout'}",
                        transform=ax.transAxes, fontsize=6.2, color="#555555",
                        ha="left", va="bottom", zorder=7)
        stamp.set_path_effects(
            [pe.withStroke(linewidth=2.0, foreground="white")])
        panel_label_inches(fig, ax, f"({'abcdefgh'[k]})", dx=-0.22, dy=0.06)

    cax = fig.add_axes([0.940, 0.455, 0.013, 0.40])
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=traj_cmap,
                                            norm=mpl.colors.Normalize(0, 1)),
                      cax=cax)
    cb.set_label("step within episode (start $\\rightarrow$ end)", fontsize=6.8,
                 color="#1a1a1a")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["0", "T"])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")

    # ---- schedule and coverage, shared x with the snapshot stages ----------
    x = stage_steps / 1e6
    dense = np.linspace(0, TOTAL_STEPS, 400) / 1e6
    eps_dense = EPS0 * (EPS1 / EPS0) ** ((dense / dense[-1]) ** 1.7)
    axb.plot(dense, eps_dense, color=EPS_C, lw=2.0, zorder=4)
    axb.plot(x, eps, marker="o", ms=3.6, ls="none", mfc="white", mec=EPS_C,
             mew=1.1, zorder=5)
    axb.set_ylabel("exploration rate $\\epsilon$ (-)", fontsize=8, color=EPS_C)
    axb.tick_params(axis="y", labelsize=7.0, colors=EPS_C)
    axb.set_ylim(0, 1.06)
    axb.set_xlim(-0.05, dense[-1] + 0.05)
    axb.set_xlabel("Environment steps ($\\times 10^6$)", fontsize=8)
    axb.grid(axis="y", color="#eeeeee", lw=0.7)
    axb.set_axisbelow(True)

    axr = axb.twinx()
    axr.plot(x, coverage, color=COV_C, lw=1.6, marker="s", ms=3.4,
             mfc="white", mec=COV_C, mew=1.0, zorder=5)
    axr.plot(x, reached, color="#c9a227", lw=1.3, ls=(0, (3.2, 1.6)),
             marker="^", ms=3.4, mfc="white", mec="#c9a227", mew=1.0, zorder=5)
    axr.set_ylabel("state coverage / success (%)", fontsize=8, color="#333333")
    axr.tick_params(axis="y", labelsize=7.0, colors="#333333")
    axr.set_ylim(0, 105)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color("#333333")
    axb.spines["left"].set_color(EPS_C)

    for k, xv in enumerate(x):
        axb.axvline(xv, color="#d9d9d9", lw=0.6, ls=(0, (2.0, 2.0)), zorder=1)
        axb.text(xv, 1.03, f"({'abcdefgh'[k]})", transform=axb.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=6.0, color="#777777")
    style_axes(axb)
    panel_label_inches(fig, axb, "(i)", dx=-0.42, dy=0.06)
    axb.set_title("Schedule that produced the eight tiles above", fontsize=8.4,
                  fontweight="bold", pad=13)

    handles = [
        plt.Line2D([], [], color=EPS_C, lw=2.0, label="$\\epsilon$ schedule"),
        plt.Line2D([], [], color=COV_C, lw=1.6, marker="s", ms=3.4,
                   mfc="white", label="state coverage"),
        plt.Line2D([], [], color="#c9a227", lw=1.3, ls=(0, (3.2, 1.6)),
                   marker="^", ms=3.4, mfc="white",
                   label=f"success rate ({N_EP} episodes / stage)"),
    ]
    axb.legend(handles=handles, loc="center right", frameon=False,
               fontsize=7.0, bbox_to_anchor=(1.0, 0.55))

    savefig(fig, "437_rl_exploration_trace.png")


if __name__ == "__main__":
    main()
