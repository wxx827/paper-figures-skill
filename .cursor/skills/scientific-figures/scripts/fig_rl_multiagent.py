"""Four agents, two teams: paths, joint return, policy similarity, alignment.

Panels (c) and (d) are two views of one simulated quantity - a 32-dim policy
vector per agent that drifts toward its team's direction over training - so the
final similarity matrix and the alignment curve cannot disagree.  The encounter
markers in (a) are computed from the trajectories, not placed by hand.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import panel_label_inches, safe_colorbar

ARENA_X, ARENA_Y = 14.0, 9.0
AGENTS = ["A1", "A2", "B1", "B2"]
TEAM = [0, 0, 1, 1]
COL = ["#b23a48", "#c0603f", "#3b5b92", "#3d6b8e"]
TEAM_C = ["#b23a48", "#3b5b92"]
STARTS = np.array([[1.0, 1.2], [1.5, 7.8], [12.9, 1.3], [12.3, 7.9]])
TARGETS = np.array([[9.4, 5.6], [10.1, 4.2], [4.4, 4.0], [3.7, 5.6]])
N_STEP, MEET_R = 190, 1.10
DIM, N_CKPT, N_SEED = 32, 40, 6
STEPS = np.linspace(0, 5.0, N_CKPT)          # millions of env steps
GRID = "#eeeeee"


def trajectories(rng):
    pos = STARTS.copy()
    path = [pos.copy()]
    for t in range(N_STEP):
        nxt = pos.copy()
        for i in range(4):
            d = TARGETS[i] - pos[i]
            d = d / (np.linalg.norm(d) + 1e-9)
            d = d + rng.normal(0, 0.30, 2)
            for j in range(4):               # collision avoidance
                if i == j:
                    continue
                off = pos[i] - pos[j]
                dist = np.linalg.norm(off)
                if dist < 1.0:
                    d = d + 0.45 * off / (dist ** 2 + 1e-6)
            d = d / (np.linalg.norm(d) + 1e-9)
            nxt[i] = np.clip(pos[i] + 0.10 * d,
                             [0.3, 0.3], [ARENA_X - 0.3, ARENA_Y - 0.3])
        pos = nxt
        path.append(pos.copy())
    return np.asarray(path)                  # (T, 4, 2)


def encounters(path):
    out = []
    for t in range(path.shape[0]):
        for i in range(4):
            for j in range(i + 1, 4):
                if np.linalg.norm(path[t, i] - path[t, j]) < MEET_R:
                    out.append((t, i, j, (path[t, i] + path[t, j]) / 2))
    keep = []
    for e in out:                            # thin out consecutive frames
        if not keep or e[0] - keep[-1][0] > 8 or (e[1], e[2]) != (keep[-1][1],
                                                                  keep[-1][2]):
            keep.append(e)
    return keep


def policy_vectors(rng):
    """w_i(t): individual noise that slowly rotates onto the team direction."""
    team_dir = rng.normal(0, 1, (2, DIM))
    team_dir /= np.linalg.norm(team_dir, axis=1, keepdims=True)
    own = rng.normal(0, 1, (4, DIM))
    own /= np.linalg.norm(own, axis=1, keepdims=True)
    w = np.empty((N_CKPT, 4, DIM))
    drift = rng.normal(0, 0.06, (N_CKPT, 4, DIM)).cumsum(axis=0)
    for t, f in enumerate(np.linspace(0, 1, N_CKPT) ** 0.8):
        for i in range(4):
            v = (1 - 0.85 * f) * own[i] + 0.85 * f * team_dir[TEAM[i]] \
                + 0.10 * drift[t, i]
            w[t, i] = v / np.linalg.norm(v)
    return w


def cosine_matrix(w):
    return w @ w.T


def returns(rng):
    """Per-agent returns over 6 seeds; the joint return is their sum."""
    out = np.empty((N_SEED, N_CKPT, 4))
    for s in range(N_SEED):
        for i in range(4):
            asym = 42.0 + 6.0 * TEAM[i] + rng.normal(0, 2.4)
            rate = 1.1 + rng.normal(0, 0.22)
            base = asym * (1 - np.exp(-rate * STEPS / STEPS[-1] * 2.6))
            e = np.zeros(N_CKPT)
            for k in range(1, N_CKPT):
                e[k] = 0.72 * e[k - 1] + rng.normal(0, 1)
            out[s, :, i] = base + e * 2.1 * (1.2 - 0.6 * STEPS / STEPS[-1])
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4390)
    path = trajectories(rng)
    meets = encounters(path)
    w = policy_vectors(rng)
    ret = returns(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.72],
                          height_ratios=[1.0, 0.72], left=0.065, right=0.95,
                          top=0.885, bottom=0.085, hspace=0.36, wspace=0.26)
    axt = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axm = fig.add_subplot(gs[1, 1])

    # ---- (a) trajectories and encounters ----------------------------------
    for i in range(4):
        axt.plot(path[:, i, 0], path[:, i, 1], color=COL[i], lw=1.5,
                 solid_capstyle="round", zorder=4)
        axt.plot(*path[0, i], marker="o", ms=5.0, mfc="white", mec=COL[i],
                 mew=1.3, zorder=6)
        axt.plot(*path[-1, i], marker="D", ms=4.6, mfc=COL[i], mec="white",
                 mew=0.9, zorder=6)
        axt.text(path[0, i, 0] + 0.28, path[0, i, 1] + 0.28, AGENTS[i],
                 fontsize=6.8, color=COL[i], fontweight="bold", zorder=7)
    for t, i, j, mid in meets:
        axt.add_patch(Circle(mid, 0.42, facecolor="none", edgecolor="#8a8a8a",
                             lw=0.7, ls=(0, (2.0, 1.6)), zorder=3))
    axt.set_xlim(0, ARENA_X)
    axt.set_ylim(0, ARENA_Y)
    axt.set_aspect("equal")
    axt.set_xlabel("arena $x$ (m)", fontsize=8)
    axt.set_ylabel("arena $y$ (m)", fontsize=8)
    axt.tick_params(labelsize=7.0)
    axt.grid(color=GRID, lw=0.7)
    axt.set_axisbelow(True)
    axt.set_title(f"Paths and {len(meets)} close encounters "
                  f"($d < {MEET_R}$ m)", fontsize=8.8, fontweight="bold",
                  pad=4)
    style_axes(axt)

    # ---- (b) joint return --------------------------------------------------
    joint = ret.sum(axis=2)
    med = np.median(joint, axis=0)
    q1, q3 = np.percentile(joint, [25, 75], axis=0)
    axr.fill_between(STEPS, q1, q3, color="#3b5b92", alpha=0.20, lw=0, zorder=3)
    for s in range(N_SEED):
        axr.plot(STEPS, joint[s], color="#3b5b92", lw=0.6, alpha=0.30, zorder=2)
    axr.plot(STEPS, med, color="#3b5b92", lw=2.0, zorder=6, label="joint (sum)")
    for i in range(4):
        axr.plot(STEPS, np.median(ret[:, :, i], axis=0), color=COL[i], lw=1.3,
                 zorder=4, label=AGENTS[i])
    axr.set_xlim(0, STEPS[-1])
    axr.set_ylim(0, max(q3.max() * 1.08, 1.0))
    axr.set_xlabel("Environment steps ($\\times 10^6$)", fontsize=8)
    axr.set_ylabel("Episode return", fontsize=8)
    axr.set_title("Joint return and its four contributors", fontsize=8.4,
                  fontweight="bold", pad=4)
    axr.grid(color=GRID, lw=0.7)
    axr.set_axisbelow(True)
    axr.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper left",
               columnspacing=1.0, handlelength=1.6)
    axr.tick_params(labelsize=7.0)
    style_axes(axr)

    # ---- (c) alignment over training --------------------------------------
    within, cross = [], []
    for t in range(N_CKPT):
        m = cosine_matrix(w[t])
        wi = [m[i, j] for i in range(4) for j in range(i + 1, 4)
              if TEAM[i] == TEAM[j]]
        cr = [m[i, j] for i in range(4) for j in range(i + 1, 4)
              if TEAM[i] != TEAM[j]]
        within.append(np.mean(wi))
        cross.append(np.mean(cr))
    within, cross = np.asarray(within), np.asarray(cross)
    index = within - cross
    axc.axhspan(0, 2, color="#2e7d5b", alpha=0.06, lw=0, zorder=0)
    axc.axhspan(-2, 0, color="#b23a48", alpha=0.06, lw=0, zorder=0)
    axc.axhline(0, color="#808080", lw=0.9, zorder=2)
    axc.plot(STEPS, within, color="#2e7d5b", lw=1.3, label="within team")
    axc.plot(STEPS, cross, color="#c9a227", lw=1.3, label="across teams")
    axc.plot(STEPS, index, color="#3b5b92", lw=2.0, zorder=6,
             label="alignment index (within $-$ across)")
    axc.set_xlim(0, STEPS[-1])
    axc.set_ylim(-1.05, 1.35)
    axc.set_xlabel("Environment steps ($\\times 10^6$)", fontsize=8)
    axc.set_ylabel("policy cosine similarity (-)", fontsize=8)
    axc.set_title("Teams co-adapt while the two teams diverge", fontsize=8.4,
                  fontweight="bold", pad=4)
    axc.text(0.015, 0.96, "cooperative", transform=axc.transAxes, fontsize=6.2,
             color="#2e7d5b", va="top")
    axc.text(0.015, 0.04, "competitive", transform=axc.transAxes, fontsize=6.2,
             color="#b23a48", va="bottom")
    axc.grid(axis="y", color=GRID, lw=0.7)
    axc.set_axisbelow(True)
    axc.legend(frameon=False, fontsize=6.8, loc="lower right",
               handlelength=1.7)
    axc.tick_params(labelsize=7.0)
    style_axes(axc)

    # ---- (d) final similarity matrix --------------------------------------
    m = cosine_matrix(w[-1])
    im = axm.imshow(m, cmap=get_cmap("cb_diverging"), vmin=-1, vmax=1)
    axm.set_xticks(range(4), AGENTS, fontsize=7.0)
    axm.set_yticks(range(4), AGENTS, fontsize=7.0)
    for i in range(4):
        for j in range(4):
            axm.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                     fontsize=6.4,
                     color="white" if abs(m[i, j]) > 0.62 else "#1a1a1a")
        axm.get_xticklabels()[i].set_color(TEAM_C[TEAM[i]])
        axm.get_yticklabels()[i].set_color(TEAM_C[TEAM[i]])
    axm.axhline(1.5, color="#1a1a1a", lw=1.0)      # team block boundary
    axm.axvline(1.5, color="#1a1a1a", lw=1.0)
    axm.set_title("Policy similarity at 5.0M steps", fontsize=8.4,
                  fontweight="bold", pad=4)
    safe_colorbar(fig, im, axm, label="cosine similarity (-)", width=0.10,
                  pad=0.07, labelsize=6.6)

    for ax, lab in ((axt, "(a)"), (axr, "(b)"), (axc, "(c)"), (axm, "(d)")):
        panel_label_inches(fig, ax, lab, dx=-0.40, dy=0.08)

    handles = [Line2D([], [], color=COL[i], lw=1.5, label=f"{AGENTS[i]}")
               for i in range(4)]
    handles += [Line2D([], [], color="none", marker="o", ms=5, mfc="white",
                       mec="#555555", label="start"),
                Line2D([], [], color="none", marker="D", ms=4.6,
                       mfc="#555555", mec="white", label="end"),
                Line2D([], [], color="#8a8a8a", lw=0.7, ls=(0, (2.0, 1.6)),
                       marker="o", ms=6, mfc="none",
                       label=f"encounter ($d<{MEET_R}$ m)")]
    savefig(fig, "439_rl_multiagent.png")


if __name__ == "__main__":
    main()
