"""Behaviour cloning from a family of human demonstrations.

The main panel carries the claim: eighteen teleoperated demonstrations of the
same reach-around task, and the single policy rollout learned from them.  The
policy is literally the mean of the plotted demonstrations after arc-length
resampling, so the thick line cannot drift away from the family it is drawn
over.  The two side panels are the evidence: a dynamic-time-warping cost
matrix showing how far apart two demonstrations are once time is realigned,
and how the success rate grows as demonstrations are added.  Every quoted
number is measured from those same arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"        # demonstrations, model side
GREEN = "#2e7d5b"       # measured success
RED = "#b23a48"         # the learned policy - the one emphasis
GREY = "#808080"

N_DEMO = 18
N_STEP = 90
START = np.array([0.06, 0.09])
GOAL = np.array([0.66, 0.47])
OBST = (0.28, 0.05, 0.14, 0.26)      # x, y, w, h in m
SEQ_CMAP = get_cmap("cb_sequential_blue")


def bezier(p0, p1, p2, p3, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1
            + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * p3)


def demos(seed=3):
    rng = np.random.default_rng(seed)
    c1 = np.array([0.18, 0.44])
    c2 = np.array([0.46, 0.44])
    out = []
    for _ in range(N_DEMO):
        j1 = c1 + rng.normal(0.0, [0.045, 0.055])
        j2 = c2 + rng.normal(0.0, [0.055, 0.040])
        s0 = START + rng.normal(0.0, 0.010, 2)
        s3 = GOAL + rng.normal(0.0, 0.008, 2)
        p = bezier(s0, j1, j2, s3, N_STEP)
        wob = rng.normal(0.0, 0.0045, (N_STEP, 2))
        k = np.exp(-0.5 * (np.arange(-12, 13) / 4.5) ** 2)
        k /= k.sum()
        for d in range(2):
            p[:, d] += np.convolve(wob[:, d], k, mode="same")
        out.append(p)
    return np.array(out)


def arclength_resample(p, n=N_STEP):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(s, d, p[:, k]) for k in range(2)])


def retime(p, gamma):
    """Re-sample an arc-length path onto a non-uniform speed profile.

    Demonstrations differ in *timing* as much as in shape; feeding two
    arc-length paths to DTW would align them perfectly by construction and
    the warping path would be an uninformative diagonal.
    """
    u = np.linspace(0.0, 1.0, len(p)) ** gamma * (len(p) - 1)
    idx = np.arange(len(p))
    return np.column_stack([np.interp(u, idx, p[:, k]) for k in range(2)])


def dtw(a, b):
    """Cumulative cost matrix and the backtracked warping path."""
    na, nb = len(a), len(b)
    loc = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((na, nb), np.inf)
    acc[0, 0] = loc[0, 0]
    for i in range(1, na):
        acc[i, 0] = acc[i - 1, 0] + loc[i, 0]
    for j in range(1, nb):
        acc[0, j] = acc[0, j - 1] + loc[0, j]
    for i in range(1, na):
        for j in range(1, nb):
            acc[i, j] = loc[i, j] + min(acc[i - 1, j], acc[i, j - 1],
                                        acc[i - 1, j - 1])
    i, j, path = na - 1, nb - 1, [(na - 1, nb - 1)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j],
                               acc[i, j - 1]]))
            i, j = [(i - 1, j - 1), (i - 1, j), (i, j - 1)][k]
        path.append((i, j))
    return acc, np.array(path[::-1])


def success_curve(seed=8):
    rng = np.random.default_rng(seed)
    n = np.array([1, 2, 3, 5, 8, 12, 18, 26, 40, 60])
    trials = 60
    p = 0.94 * (1.0 - np.exp(-n / 6.2))
    hits = rng.binomial(trials, np.clip(p, 0.0, 1.0))
    rate = hits / trials
    se = np.sqrt(rate * (1.0 - rate) / trials)
    return n, rate, se, trials


def traj_panel(ax, paths, policy):
    ax.add_patch(Rectangle(OBST[:2], OBST[2], OBST[3], facecolor="#d9d9d9",
                           edgecolor=GREY, linewidth=0.8, zorder=2))
    ax.text(OBST[0] + OBST[2] / 2, OBST[1] + OBST[3] + 0.012, "obstacle",
            ha="center", va="bottom", fontsize=6.8, color=SUB)

    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.0, alpha=0.28, zorder=3)
    ax.plot([], [], color=BLUE, lw=1.0, alpha=0.55,
            label=f"{len(paths)} human demonstrations")
    ax.plot(policy[:, 0], policy[:, 1], color=RED, lw=2.4, zorder=6,
            label="learned policy rollout")

    ax.plot(*START, marker="o", ms=6.0, mfc="white", mec=INK, mew=1.1,
            zorder=7)
    ax.plot(*GOAL, marker="*", ms=11.0, color=GREEN, zorder=7)
    ax.text(START[0] + 0.015, START[1] - 0.008, "start", fontsize=7.0,
            color=INK, ha="left", va="top")
    ax.text(GOAL[0] - 0.012, GOAL[1] + 0.014, "goal", fontsize=7.0,
            color=GREEN, ha="right", va="bottom")

    dev = np.linalg.norm(paths - policy[None, :, :], axis=2)
    band = float(dev.mean()) * 1e3
    spread = float(dev.max(axis=0).mean()) * 1e3
    ax.text(0.98, 0.03, f"mean demo-policy offset {band:.0f} mm\n"
                        f"widest demo spread {spread:.0f} mm",
            transform=ax.transAxes, fontsize=6.8, color=SUB, ha="right",
            va="bottom", linespacing=1.4)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left",
              handlelength=1.8, borderaxespad=0.3)
    ax.set_title("Demonstration family and the cloned rollout", fontsize=9.0,
                 fontweight="bold", color=INK)
    return band


def dtw_panel(fig, ax, acc, path):
    im = ax.imshow(acc, origin="lower", aspect="auto", cmap=SEQ_CMAP,
                   extent=[0, acc.shape[1], 0, acc.shape[0]])
    ax.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, color=RED, lw=1.3, zorder=4)
    ax.plot([0, acc.shape[1]], [0, acc.shape[0]], color="white", lw=0.8,
            ls=(0, (3, 2.2)), zorder=3)
    ax.text(0.96 * acc.shape[1], 0.06 * acc.shape[0],
            f"DTW distance {acc[-1, -1]:.2f} m", ha="right", va="bottom",
            fontsize=6.6, color="white")
    ax.set_xlabel("demo B step index (-)", fontsize=7.4)
    ax.set_ylabel("demo A step index (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Time alignment (DTW)", fontsize=8.2,
                 fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax, label="cumulative cost (m)", width=0.10,
                  pad=0.06, labelsize=6.5)


def curve_panel(ax, n, rate, se, trials):
    ax.fill_between(n, 100 * (rate - se), 100 * (rate + se), color=GREEN,
                    alpha=0.18, linewidth=0.0, zorder=2)
    ax.plot(n, 100 * rate, color=GREEN, lw=1.5, marker="o", ms=3.2,
            mfc="white", mew=0.9, zorder=4)

    plateau = 100 * float(rate[-3:].mean())
    need = float(np.interp(0.90 * plateau, 100 * rate, n))
    ax.axhline(plateau, color=GREY, lw=0.8, ls=(0, (3, 2.2)), zorder=3)
    ax.axvline(need, color=RED, lw=0.9, ls=(0, (2.6, 2.0)), zorder=3)
    ax.text(need + 1.5, 20.0, f"{need:.0f} demos reach\n90 % of the plateau",
            fontsize=6.6, color=RED, ha="left", va="bottom", linespacing=1.35)
    ax.text(1.05, plateau + 2.0, f"plateau {plateau:.0f} %", ha="left",
            va="bottom", fontsize=6.6, color=SUB)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 70)
    ax.set_ylim(0, 108)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("demonstrations used $N$ (-)", fontsize=7.4)
    ax.set_ylabel("task success (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.7)
    ax.set_title(f"Success, {trials} trials per budget", fontsize=8.2,
                 fontweight="bold", color=INK)
    return plateau, need


def main():
    set_pub_style()
    paths = np.array([arclength_resample(p) for p in demos()])
    policy = paths.mean(axis=0)
    acc, path = dtw(retime(paths[0], 1.7)[::3], retime(paths[7], 0.65)[::3])
    n, rate, se, trials = success_curve()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_dtw = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, 1])

    ax_main.set_aspect("equal")
    xlo, xhi = shared_limits([paths[..., 0], [START[0], GOAL[0]]], pad=0.10)
    ylo, yhi = shared_limits([paths[..., 1], [START[1], GOAL[1]]], pad=0.12)
    ax_main.set_xlim(xlo, xhi)
    ax_main.set_ylim(ylo, yhi)
    ax_main.grid(lw=0.4, alpha=0.55)
    ax_main.set_axisbelow(True)

    band = traj_panel(ax_main, paths, policy)
    dtw_panel(fig, ax_dtw, acc, path)
    plateau, need = curve_panel(ax_curve, n, rate, se, trials)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.4)
    fig.canvas.draw()
    ax_main.apply_aspect()
    panel_labels_inches(fig, [ax_main, ax_dtw, ax_curve], dx=-0.42, dy=0.10)
    savefig(fig, "585_rob_imitation_learning.png")


if __name__ == "__main__":
    main()
