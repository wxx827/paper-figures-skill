"""Multi-agent interaction graph across training: four time slices plus stats.

One simulation drives every mark on this figure.  ``rollout`` returns the
positions of 12 agents at 25 checkpoints; an edge exists whenever two agents
are inside the interaction radius, and its weight (hence its line width) is
how far inside.  The two curves underneath are the mean degree and the average
local clustering coefficient of *those* graphs, so the four arenas are four
samples of the same series - the script asserts that the curve value at each
snapshot equals the value recomputed from the edge list that panel draws.

Change the dynamics and both rows change together; that is the point.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
EDGE = "#7794ac"
CARD = "#dfe6ee"          # clearly below the 0.92 ink threshold, never between
TEAM = [BLUE[0], BLUE[2], BLUE[4]]
TEAM_NAME = ["scouts", "carriers", "blockers"]

N_AGENTS = 12
N_STEPS = 25
STEP_K = 10               # thousand environment steps between checkpoints
RADIUS = 0.265
SNAPS = (0, 8, 16, 24)


def rollout(seed=1588):
    """Positions of every agent at every checkpoint, in a unit arena."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.12, 0.88, (N_AGENTS, 2))
    hubs = np.array([[0.27, 0.70], [0.72, 0.66], [0.50, 0.24]])
    team = np.repeat(np.arange(3), N_AGENTS // 3)
    off = rng.uniform(-1.0, 1.0, (N_AGENTS, 2)) * 0.105
    phase = rng.uniform(0, 2 * np.pi, N_AGENTS)
    P = np.empty((N_STEPS, N_AGENTS, 2))
    for t in range(N_STEPS):
        f = (t / (N_STEPS - 1.0)) ** 1.35
        wob = 0.030 * np.stack([np.cos(phase + 0.55 * t),
                                np.sin(phase + 0.55 * t)], axis=1)
        P[t] = (1 - f) * start + f * (hubs[team] + off) + (0.35 + f) * wob
    return np.clip(P, 0.045, 0.955), team


def edges_of(pos):
    """(i, j, weight) for every pair inside the radius; weight in (0, 1]."""
    out = []
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            d = float(np.hypot(*(pos[i] - pos[j])))
            if d <= RADIUS:
                out.append((i, j, 1.0 - d / RADIUS))
    return out


def stats(ed):
    """Mean degree (2E/N) and average local clustering of one edge list."""
    A = np.zeros((N_AGENTS, N_AGENTS), bool)
    for i, j, _ in ed:
        A[i, j] = A[j, i] = True
    deg = A.sum(axis=1)
    loc = []
    for i in range(N_AGENTS):
        nb = np.flatnonzero(A[i])
        k = nb.size
        loc.append(0.0 if k < 2 else
                   A[np.ix_(nb, nb)].sum() / (k * (k - 1.0)))
    return 2.0 * len(ed) / N_AGENTS, float(np.mean(loc))


def lw_of(w):
    return 0.42 + 3.05 * w


def main():
    set_pub_style(base_font=8)
    P, team = rollout()
    graphs = [edges_of(P[t]) for t in range(N_STEPS)]
    series = np.array([stats(g) for g in graphs])
    steps = np.arange(N_STEPS) * STEP_K
    mdeg, clus = series[:, 0], series[:, 1]

    fig = plt.figure(figsize=(11.0, 4.6))
    y0, ph = 0.478, 0.385
    pw = ph * 4.6 / 11.0
    gap = 0.019
    x_left = 0.043

    for k, t in enumerate(SNAPS):
        ax = fig.add_axes([x_left + k * (pw + gap), y0, pw, ph])
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor=CARD, edgecolor=BLUE[5],
                                   linewidth=0.7, zorder=0))
        ed = graphs[t]
        for i, j, w in ed:
            ax.plot([P[t, i, 0], P[t, j, 0]], [P[t, i, 1], P[t, j, 1]],
                    color=EDGE, lw=lw_of(w), alpha=0.42 + 0.44 * w,
                    solid_capstyle="round", zorder=2)
        for a in range(N_AGENTS):
            ax.plot([P[t, a, 0]], [P[t, a, 1]], marker="o", ls="none",
                    ms=6.4, mfc=TEAM[team[a]], mec="white", mew=0.9, zorder=4)

        if k == 0:                       # show what the radius means, once
            a0 = int(np.argmin(P[t, :, 1]))
            ax.add_patch(Circle(P[t, a0], RADIUS, facecolor="none",
                                edgecolor=BLUE[2], lw=0.7, ls=(0, (3, 2)),
                                zorder=3))
            ax.text(P[t, a0, 0], P[t, a0, 1] + RADIUS + 0.030,
                    f"radius r = {RADIUS:.3f}", ha="center", va="bottom",
                    fontsize=6.4, color=BLUE[1], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=BLUE[5], lw=0.5, alpha=0.95))

        # the panel prints the arithmetic it wants the reader to trust
        d_shown, c_shown = stats(ed)
        assert abs(d_shown - mdeg[t]) < 1e-9 and abs(c_shown - clus[t]) < 1e-9
        ax.text(0.5, -0.075, f"$t$ = {steps[t]:d}k steps", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.4, fontweight="bold",
                color=INK)
        ax.text(0.5, -0.205,
                f"2 x {len(ed):d} edges / {N_AGENTS:d} agents = {d_shown:.2f}\n"
                f"clustering C = {c_shown:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.6, color=COLORS["sub"],
                linespacing=1.45)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.030, 0.965, f"({'abcd'[k]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.6, fontweight="bold",
                color=BLUE[1])

    # ---- the two graph statistics, over the whole run ----------------------
    x_right = x_left + 4 * pw + 3 * gap
    axc = fig.add_axes([x_left, 0.112, x_right - x_left, 0.208])
    axd = axc.twinx()

    axc.plot(steps, mdeg, color=BLUE[1], lw=1.7, marker="o", ms=3.0,
             mfc="white", mew=0.9, zorder=4, label="mean degree $2E/N$")
    axd.plot(steps, clus, color=BLUE[3], lw=1.7, ls=(0, (4.5, 2.0)),
             marker="s", ms=3.0, mfc="white", mew=0.9, zorder=3,
             label="clustering coefficient $C$")

    for k, t in enumerate(SNAPS):        # explicit endpoints, never axvline
        axc.plot([steps[t], steps[t]], [0.0, mdeg[t]], color="#a8b7c4",
                 lw=0.8, ls=(0, (2.2, 1.8)), zorder=2)
        axc.text(steps[t], mdeg[t] + 0.22, f"({'abcd'[k]})", ha="center",
                 va="bottom", fontsize=6.8, fontweight="bold", color=BLUE[1],
                 zorder=6,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                           alpha=0.92))

    axc.set_xlim(steps[0] - 4, steps[-1] + 4)
    axc.set_ylim(0, max(mdeg) * 1.32)
    axd.set_ylim(0, 1.0)
    axc.set_xlabel("environment steps (thousands)", fontsize=7.6,
                   color=COLORS["sub"], labelpad=1.5)
    axc.set_ylabel("mean degree", fontsize=7.4, color=BLUE[1], labelpad=2.0)
    axd.set_ylabel("clustering $C$", fontsize=7.4, color=BLUE[2], labelpad=2.0)
    axc.tick_params(labelsize=7.0, length=2.6)
    axd.tick_params(labelsize=7.0, length=2.6, colors=BLUE[2])
    axc.grid(axis="y", color="#e6eaee", lw=0.6)
    axc.set_axisbelow(True)
    axc.spines["top"].set_visible(False)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_color(BLUE[3])
    fig.text(x_left, 0.352, "Both curves are computed from the graphs above - "
             "one graph per checkpoint, the four labelled ones are (a)-(d).",
             ha="left", va="top", fontsize=6.8, color=COLORS["muted"])

    # ---- key column --------------------------------------------------------
    keys = [Line2D([], [], marker="o", ls="none", ms=6.4, mfc=TEAM[i],
                   mec="white", mew=0.9, label=f"{TEAM_NAME[i]} "
                   f"({int((team == i).sum()):d} agents)") for i in range(3)]
    keys += [Line2D([], [], color=EDGE, lw=lw_of(0.15), alpha=0.49,
                    label="weak link, $w$ = 0.15"),
             Line2D([], [], color=EDGE, lw=lw_of(0.85), alpha=0.79,
                    label="strong link, $w$ = 0.85")]
    leg = fig.legend(handles=keys, loc="upper left",
                     bbox_to_anchor=(x_right + 0.030, 0.840), frameon=False,
                     fontsize=7.2, handletextpad=0.6, labelspacing=0.62,
                     handlelength=1.7)
    for txt in leg.get_texts():
        txt.set_color(INK)

    peak = int(np.argmax(mdeg))
    fig.text(x_right + 0.078, 0.335,
             f"{N_AGENTS:d} agents, {N_STEPS:d} checkpoints\n"
             f"{STEP_K:d}k steps apart.  Link when\n"
             f"$d \\leq$ {RADIUS:.3f} arena widths;\n"
             f"$w = 1 - d/r$ sets the width.\n"
             f"Peak degree {mdeg[peak]:.2f} at "
             f"{steps[peak]:d}k steps,\n"
             f"then {mdeg[-1]:.2f} once the three\n"
             f"coalitions stop mixing.",
             ha="left", va="top", fontsize=6.7, color=COLORS["sub"],
             linespacing=1.6)


    savefig(fig, "1183_wm_mas_interaction_graph.png")


if __name__ == "__main__":
    main()
