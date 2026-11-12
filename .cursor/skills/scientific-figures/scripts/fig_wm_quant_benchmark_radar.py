"""Capability profile of five world models: radar plus the same numbers as a table.

The radar reads the shape of each model at a glance; the heatmap table on the
right carries the values that the radar can only suggest.  Both panels show the
identical 5 x 7 matrix, share one colour / legend key, and the best entry in
each capability column is set in bold inside a dark cell border.

Synthetic data lives in ``gen_scores`` only -- swap that one function for a real
loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, add_colorbar

METHODS = ["PPO-pixel", "IRIS", "DreamerV3", "Genie-style", "Ours (WM-8)"]
# cool-blue ramp for the baselines, one warm accent reserved for Ours
COL = {"PPO-pixel": "#5f9dc6", "IRIS": "#8e6bbf", "DreamerV3": "#2a7f7a",
       "Genie-style": "#3b5b92", "Ours (WM-8)": "#b23a48"}
LW = {m: (2.4 if m == "Ours (WM-8)" else 1.6) for m in METHODS}

AXES_SHORT = ["Dynamics\nfidelity", "Long-horizon\nstability", "Object\npermanence",
              "3D\nconsistency", "Action\ncontrollability", "Novel-view\nsynthesis",
              "Sim-to-real\ntransfer"]
AXES_TABLE = ["Dyn.\nfid.", "Long-hor.\nstab.", "Object\nperm.", "3D\ncons.",
              "Action\nctrl.", "Novel\nview", "Sim2real"]

# method -> (overall level, per-axis tilt over the seven capabilities)
PROFILE = {
    "PPO-pixel":   (38.0, [0.6, -1.0, -0.9, -0.7, 1.1, -1.2, 0.2]),
    "IRIS":        (54.0, [0.9, -0.4, 0.3, -0.9, 0.2, -0.6, -0.3]),
    "DreamerV3":   (65.0, [0.7, 0.5, 0.1, -0.6, 0.9, -1.0, 0.4]),
    # Genie-style is the one baseline that still wins novel-view synthesis
    "Genie-style": (61.0, [1.0, -0.2, 0.6, 0.8, -1.1, 2.6, -0.9]),
    "Ours (WM-8)": (82.0, [0.5, 0.9, 0.7, 0.8, 0.4, -2.4, 0.5]),
}
SPREAD = 7.0


def gen_scores(seed=1501):
    """(5, 7) matrix of capability scores in 0-100."""
    rng = np.random.default_rng(seed)
    m = np.empty((len(METHODS), len(AXES_SHORT)))
    for i, name in enumerate(METHODS):
        level, tilt = PROFILE[name]
        m[i] = np.clip(level + SPREAD * np.asarray(tilt)
                       + rng.normal(0.0, 2.6, len(tilt)), 8.0, 97.0)
    return m


def radar(ax, scores):
    n = scores.shape[1]
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    loop = np.concatenate([ang, ang[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", ""], fontsize=6.6,
                       color="#8a8a8a")
    ax.set_rlabel_position(90.0 / n)
    ax.set_xticks(ang)
    ax.set_xticklabels(AXES_SHORT, fontsize=8.2, color="#1a1a1a")
    ax.tick_params(axis="x", pad=7.0)
    ax.grid(color="#dcdcdc", lw=0.7)
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)

    for i, name in enumerate(METHODS):
        hot = name == "Ours (WM-8)"
        v = np.concatenate([scores[i], scores[i, :1]])
        ax.fill(loop, v, color=COL[name], alpha=0.13 if hot else 0.06, lw=0,
                zorder=2)
        ax.plot(loop, v, color=COL[name], lw=LW[name], zorder=5 if hot else 3,
                solid_joinstyle="round")
        ax.plot(ang, scores[i], ls="none", marker="o", ms=3.4 if hot else 2.6,
                mfc="white", mec=COL[name], mew=1.1 if hot else 0.8,
                zorder=6 if hot else 4)


def table(fig, ax, scores, norm, cmap):
    im = ax.imshow(scores, cmap=cmap, norm=norm, aspect="auto", zorder=2)
    best = scores.argmax(axis=0)

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            v = scores[i, j]
            top = i == best[j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.4,
                    color="white" if norm(v) > 0.62 else "#1a1a1a",
                    fontweight="bold" if top else "normal", zorder=4)
            if top:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#1a1a1a", lw=1.5, zorder=5))

    ax.set_xticks(range(len(AXES_TABLE)))
    ax.set_xticklabels(AXES_TABLE, fontsize=7.2)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=8.4)
    for lab, name in zip(ax.get_yticklabels(), METHODS):
        lab.set_color(COL[name])
        if name == "Ours (WM-8)":
            lab.set_fontweight("bold")
    ax.tick_params(length=0, pad=3.0)
    ax.set_xticks(np.arange(-0.5, scores.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, scores.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4, zorder=3)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main():
    set_pub_style(base_font=8)
    scores = gen_scores()
    # one colour range for the table, computed before anything is drawn
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    cmap = get_cmap("cb_sequential_blue")

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], left=0.045,
                          right=0.955, top=0.845, bottom=0.145, wspace=0.30)
    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    ax_t = fig.add_subplot(gs[0, 1])

    radar(ax_r, scores)
    im = table(fig, ax_t, scores, norm, cmap)

    ax_r.set_title("Capability profile", fontsize=10.0, fontweight="bold",
                   pad=22.0)
    ax_t.set_title("Same matrix, exact scores", fontsize=10.0,
                   fontweight="bold", pad=8.0)

    cb = add_colorbar(fig, im, ax_t, label="capability score (0-100)",
                      orientation="horizontal", fraction=0.052, pad=0.13,
                      shrink=0.78, ticks=[20, 40, 60, 80])
    cb.ax.set_position(cb.ax.get_position())

    handles = [Line2D([], [], color=COL[m], lw=LW[m], marker="o", ms=4.0,
                      mfc="white", mec=COL[m], mew=1.1, label=m)
               for m in METHODS]
    handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1a1a1a",
                             lw=1.5, label="best on that capability"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                     bbox_to_anchor=(0.5, 0.012), frameon=False, fontsize=9.0,
                     handlelength=1.8, columnspacing=1.9, handletextpad=0.6)
    leg.get_texts()[4].set_fontweight("bold")
    leg.get_texts()[5].set_color("#555555")
    leg.get_texts()[5].set_fontsize(8.0)

    savefig(fig, "1096_wm_quant_benchmark_radar.png")


if __name__ == "__main__":
    main()
