"""Curriculum progress: what the sampler serves, and what the team solves.

A teacher-student loop generates both layers of this figure.  The student's
skill grows while it practises; the teacher promotes the difficulty target one
level whenever the success rate crosses 80%, and the task mixture is a kernel
around that target.  So the promotion markers are events of the run, not
hand-placed rules, and the success curve is the very signal the teacher reacts
to.

Every column of the stacked area is rounded with the largest-remainder method
and then asserted to sum to exactly 100.0%, and the two call-outs print their
own column's five numbers plus the total, so the reader can add them up.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

LEVELS = ["L1  2 agents, 1 goal", "L2  3 agents, 2 goals",
          "L3  4 agents, 3 goals", "L4  6 agents, 4 goals",
          "L5  8 agents, 6 goals"]
LEVEL_COL = [BLUE[7], BLUE[5], BLUE[3], BLUE[2], BLUE[0]]
NL = len(LEVELS)
T, STEP_K = 201, 2.5
PROMOTE = 80.0                       # success rate the teacher promotes on
WIDTH = 0.80                         # kernel width of the task sampler


def pct_column(w):
    """Round a weight vector to one decimal so the parts sum to exactly 100.0.

    Naive rounding leaves columns at 99.9 or 100.1, and a stacked chart whose
    printed parts do not add up is exactly the kind of thing a reader checks.
    """
    raw = 1000.0 * w / w.sum()
    down = np.floor(raw)
    short = int(round(raw.sum() - down.sum()))
    order = np.argsort(-(raw - down))
    down[order[:short]] += 1.0
    return down / 10.0


def curriculum(seed=1592):
    """Teacher-student loop; returns the mixture, the success rate, events."""
    rng = np.random.default_rng(seed)
    lv = np.arange(NL, dtype=float)
    skill, target = 0.15, 0.0
    d = 0.0
    share = np.zeros((T, NL))
    succ = np.zeros(T)
    events = []
    for t in range(T):
        d += 0.10 * (target - d)
        w = np.exp(-0.5 * ((lv - d) / WIDTH) ** 2)
        share[t] = pct_column(w)
        s = 100.0 / (1.0 + np.exp(-2.6 * (skill - d - 0.05)))
        succ[t] = float(np.clip(s + rng.normal(0.0, 1.5), 0.0, 100.0))
        if succ[t] >= PROMOTE and target < NL - 1 and t > 2:
            events.append((t, int(target), int(target) + 1))
            target += 1.0
        skill += 0.042 * (1.0 - 0.7 * succ[t] / 100.0)
    return share, succ, events


def main():
    set_pub_style(base_font=8)
    share, succ, events = curriculum()
    steps = np.arange(T) * STEP_K
    assert np.all(np.abs(share.sum(axis=1) - 100.0) < 1e-9), "columns != 100%"
    assert len(events) == NL - 1, "the curriculum must reach the last level"

    fig = plt.figure(figsize=(10.4, 6.0))
    # the stack fills the canvas: the old 0.148 bottom margin held nothing but
    # white, and `bbox_inches="tight"` cropped it back off, delivering 0.513
    ax = fig.add_axes([0.062, 0.088, 0.655, 0.700])

    ax.stackplot(steps, share.T, colors=LEVEL_COL, edgecolor="#ffffff",
                 linewidth=0.5, zorder=2)
    for i in range(NL - 1):                # hairline between the pale bands
        ax.plot(steps, share[:, :i + 1].sum(axis=1), color="#b9c6d2", lw=0.5,
                zorder=3)
    ax.plot(steps, succ, color=ACCENT, lw=2.1, zorder=6)
    ax.plot(steps, succ, color="white", lw=3.6, zorder=5, alpha=0.55)
    ax.plot([steps[0], steps[-1]], [PROMOTE, PROMOTE], color=ACCENT, lw=0.9,
            ls=(0, (4, 2)), zorder=7)
    ax.text(steps[-1] - 3, PROMOTE + 1.6,
            f"promotion threshold {PROMOTE:.0f}%", ha="right", va="bottom",
            fontsize=6.8, color=ACCENT, zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.94))

    for k, (t, a, b) in enumerate(events):
        ax.plot([steps[t], steps[t]], [0, 100], color="white", lw=1.9,
                zorder=7)
        ax.plot([steps[t], steps[t]], [0, 100], color=BLUE[0], lw=0.9,
                ls=(0, (3, 2)), zorder=8)
        ax.text(steps[t], 102.0 + 6.5 * (k % 2),
                f"L{a + 1:d} $\\rightarrow$ L{b + 1:d}  at "
                f"{steps[t]:.0f}k", ha="center", va="bottom", fontsize=6.9,
                fontweight="bold", color=BLUE[0], zorder=9, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=BLUE[5],
                          lw=0.6, alpha=0.97))

    # ---- two columns that print their own arithmetic ----------------------
    for ti, (tx, ty) in ((int(0.30 * T), (0.045, 0.470)),
                         (int(0.86 * T), (0.560, 0.130))):
        parts = share[ti]
        shown = [f"{v:.1f}" for v in parts]
        assert abs(sum(float(s) for s in shown) - 100.0) < 1e-9
        ax.plot([steps[ti], steps[ti]], [0, 100], color=BLUE[1], lw=0.8,
                zorder=7)
        txt = ("  +  ".join(shown[:3]) + "\n+  " + "  +  ".join(shown[3:]) +
               "  =  100.0%")
        ax.text(tx, ty, f"mixture at {steps[ti]:.0f}k steps\n" + txt,
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
                color=INK, zorder=9, linespacing=1.55,
                bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=BLUE[4],
                          lw=0.7, alpha=0.96))

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("environment steps (thousands)", fontsize=8.0,
                  color=COLORS["sub"], labelpad=2.0)
    ax.set_ylabel("share of sampled tasks (%)   /   success rate (%)",
                  fontsize=8.0, color=COLORS["sub"], labelpad=3.0)
    ax.tick_params(labelsize=7.2, length=2.8)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # ---- key + notes -------------------------------------------------------
    handles = [Patch(facecolor=LEVEL_COL[i], edgecolor="white", lw=0.5,
                     label=LEVELS[i]) for i in range(NL)][::-1]
    handles += [Line2D([], [], color=ACCENT, lw=2.1,
                       label="team success rate (%)"),
                Line2D([], [], color=ACCENT, lw=0.9, ls=(0, (4, 2)),
                       label=f"promotion threshold ({PROMOTE:.0f}%)"),
                Line2D([], [], color=BLUE[0], lw=0.9, ls=(0, (3, 2)),
                       label="promotion event"),
                Line2D([], [], color=BLUE[1], lw=0.8,
                       label="column printed in a call-out")]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.748, 0.788), frameon=False,
                     fontsize=7.2, handletextpad=0.7, labelspacing=0.72,
                     handlelength=1.5)
    for txt in leg.get_texts():
        txt.set_color(INK)

    tenth = int(np.argmax(share[:, NL - 1] >= 10.0))
    fig.text(0.748, 0.300,
             f"{NL:d} difficulty levels, {T:d} checkpoints\n"
             f"{STEP_K:.1f}k steps apart.  The sampler\n"
             f"is a Gaussian kernel of width\n"
             f"{WIDTH:.2f} levels around the current\n"
             f"target, renormalised to 100%.\n\n"
             f"The hardest level passes 10% of\n"
             f"the mixture at {steps[tenth]:.0f}k and ends at\n"
             f"{share[-1, NL - 1]:.1f}% of the sampled tasks,\n"
             f"with the team at {succ[-1]:.0f}% success.",
             ha="left", va="top", fontsize=6.8, color=COLORS["sub"],
             linespacing=1.6)

    # x=0.028, not 0.050: at 11 pt bold this headline is 10.06 in long and the
    # old inset pushed its right end 0.19 in past the declared canvas

    savefig(fig, "1187_wm_mas_curriculum.png")


if __name__ == "__main__":
    main()
