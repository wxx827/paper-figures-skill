"""Five-step peg-in-hole assembly: the sequence and what each step costs.

The strip is drawn on an inch-style equal-aspect axes so the five stages are
the same size and the action arrows between them have the same length; each
stage is built from the same primitives - fixture plate, bore, bracket,
two-finger gripper - moved rather than redrawn, which is what makes the
sequence read as one motion.  The panel underneath is the price list for that
motion: bar height is measured cycle time, the overlaid markers are the peak
contact force each step reaches, and the takt time in the caption is the sum
of the bars actually plotted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle

from pubstyle import set_pub_style, savefig
from figkit import arrow, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"        # the bracket being assembled
GREEN = "#2e7d5b"       # measured force / accepted
GOLD = "#c9a227"        # fastener
RED = "#b23a48"         # the one emphasis: the insertion step
GREY = "#808080"
PLATE = "#c8bcae"
TOOL = "#3a3a3a"

STEPS = ["Pick", "Align", "Insert", "Screw", "Verify"]
ACTIONS = ["transfer", "servo", "push", "torque"]
CYCLE = np.array([2.4, 1.8, 3.6, 4.1, 1.5])        # s per step
PEAK_F = np.array([6.0, 3.2, 41.0, 18.0, 5.5])     # N per step
SCREW_TAU = 1.85                                    # N m at the screw step

CW, GAP = 3.0, 0.62                                 # stage cell width / gap
YB = 0.75                                           # fixture top height


def gripper(ax, cx, cy, open_w, holding=False):
    for s in (-1.0, 1.0):
        ax.add_patch(Rectangle((cx + s * open_w / 2 - 0.09, cy), 0.18, 0.62,
                               facecolor=TOOL, edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((cx - open_w / 2 - 0.09, cy + 0.62),
                           open_w + 0.18, 0.17, facecolor=TOOL,
                           edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((cx - 0.11, cy + 0.79), 0.22, 0.55,
                           facecolor=TOOL, edgecolor="none", zorder=6))
    if holding:
        ax.add_patch(Rectangle((cx - open_w / 2 - 0.02, cy + 0.18),
                               open_w + 0.04, 0.10, facecolor=GREEN,
                               edgecolor="none", zorder=7))


def fixture(ax, cx):
    ax.add_patch(Rectangle((cx - 1.30, YB - 0.55), 2.60, 0.55,
                           facecolor=PLATE, edgecolor="#8a7f74",
                           linewidth=0.8, zorder=3))
    ax.add_patch(Rectangle((cx - 0.30, YB - 0.55), 0.60, 0.55,
                           facecolor="white", edgecolor="#8a7f74",
                           linewidth=0.8, zorder=4))


def bracket(ax, x, y, color=BLUE, alpha=1.0):
    ax.add_patch(Rectangle((x - 0.24, y), 0.48, 1.05, facecolor=color,
                           edgecolor="white", linewidth=0.7, zorder=5,
                           alpha=alpha))
    ax.add_patch(Rectangle((x - 0.55, y + 1.05), 1.10, 0.26, facecolor=color,
                           edgecolor="white", linewidth=0.7, zorder=5,
                           alpha=alpha))


def screw(ax, x, y):
    ax.add_patch(Polygon([(x - 0.10, y), (x + 0.10, y), (x + 0.06, y - 0.34),
                          (x - 0.06, y - 0.34)], closed=True,
                         facecolor=GOLD, edgecolor="none", zorder=7))
    ax.add_patch(Rectangle((x - 0.22, y), 0.44, 0.14, facecolor=GOLD,
                           edgecolor="none", zorder=7))


def stage(ax, k, cx):
    fixture(ax, cx)
    if k == 0:
        bracket(ax, cx - 0.85, YB)
        gripper(ax, cx - 0.85, YB + 1.55, 0.86)
        ax.annotate("", xy=(cx - 0.85, YB + 1.40), xytext=(cx - 0.85,
                                                           YB + 2.10),
                    arrowprops=dict(arrowstyle="-|>", color=TOOL, lw=0.9))
    elif k == 1:
        bracket(ax, cx, YB + 1.05)
        gripper(ax, cx, YB + 2.36, 0.58, holding=True)
        ax.plot([cx, cx], [YB - 0.60, YB + 1.05], color=RED, lw=0.8,
                ls=(0, (2.6, 2.0)), zorder=6)
        ax.text(cx - 0.22, YB + 0.45, "bore axis", fontsize=6.2, color=RED,
                ha="right", va="center")
    elif k == 2:
        bracket(ax, cx, YB - 0.18)
        gripper(ax, cx, YB + 1.13, 0.58, holding=True)
        arrow(ax, (cx + 0.95, YB + 1.30), (cx + 0.95, YB + 0.30), color=RED,
              lw=1.1, ms=9)
        ax.text(cx + 1.05, YB + 0.80, f"{PEAK_F[2]:.0f} N", fontsize=6.4,
                color=RED, ha="left", va="center")
    elif k == 3:
        bracket(ax, cx, YB - 0.38)
        screw(ax, cx, YB + 0.95)
        ax.add_patch(Rectangle((cx - 0.11, YB + 1.15), 0.22, 1.10,
                               facecolor=TOOL, edgecolor="none", zorder=6))
        ax.add_patch(Arc((cx, YB + 2.30), 1.10, 0.52, theta1=200.0,
                         theta2=345.0, color=TOOL, lw=1.1, zorder=6))
        arrow(ax, (cx + 0.40, YB + 2.08), (cx + 0.55, YB + 2.24),
              color=TOOL, lw=0.9, ms=8, zorder=6)
        ax.text(cx + 0.66, YB + 2.30,
                r"$\tau$ = " f"{SCREW_TAU:.2f}" r" N$\cdot$m", fontsize=6.4,
                color=SUB, ha="left", va="center")
    else:
        bracket(ax, cx, YB - 0.38)
        screw(ax, cx, YB + 0.95)
        ax.plot([cx - 0.62, cx - 0.62], [YB - 0.38, YB + 0.93], color=GREEN,
                lw=0.8, zorder=6)
        ax.plot([cx - 0.70, cx - 0.54], [YB + 0.93, YB + 0.93], color=GREEN,
                lw=0.8, zorder=6)
        ax.plot([cx - 0.70, cx - 0.54], [YB - 0.38, YB - 0.38], color=GREEN,
                lw=0.8, zorder=6)
        ax.text(cx - 0.78, YB + 0.62, "seating\ncheck", fontsize=6.2,
                color=GREEN, ha="right", va="center", linespacing=1.3)
    ax.text(cx, 4.45, f"{k + 1}  {STEPS[k]}", ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color=INK)


def strip_panel(ax):
    centers = [1.55 + i * (CW + GAP) for i in range(len(STEPS))]
    for k, cx in enumerate(centers):
        stage(ax, k, cx)
        if k < len(ACTIONS):
            x0 = cx + CW / 2 - 0.20
            x1 = centers[k + 1] - CW / 2 + 0.20
            arrow(ax, (x0, 3.55), (x1, 3.55), color=GREY, lw=1.0, ms=9)
            ax.text((x0 + x1) / 2.0, 3.70, ACTIONS[k], ha="center",
                    va="bottom", fontsize=6.8, color=SUB)
    ax.set_xlim(0.0, centers[-1] + CW / 2 + 0.15)
    ax.set_ylim(-0.25, 5.05)
    ax.set_aspect("equal")
    ax.axis("off")
    return centers


def cost_panel(ax, axr):
    x = np.arange(len(STEPS))
    ax.bar(x, CYCLE, width=0.56, color=BLUE, edgecolor="white",
           linewidth=0.7, zorder=3, label="cycle time")
    hot = int(np.argmax(CYCLE))
    ax.bar([hot], [CYCLE[hot]], width=0.56, color=RED, edgecolor="white",
           linewidth=0.7, zorder=4)
    for xi, v in zip(x, CYCLE):
        ax.text(xi, v - 0.14, f"{v:.1f} s", ha="center", va="top",
                fontsize=6.5, color="white", zorder=5)

    axr.plot(x, PEAK_F, color=GREEN, lw=1.3, marker="o", ms=4.0, mfc="white",
             mew=1.0, zorder=6, label="peak contact force")
    axr.set_ylim(0.0, PEAK_F.max() * 1.28)
    axr.set_ylabel("peak contact force $F$ (N)", color=GREEN, fontsize=7.6)
    axr.tick_params(axis="y", colors=GREEN, labelsize=7.0)
    axr.spines["top"].set_visible(False)

    ax.set_ylim(0.0, CYCLE.max() * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1} {s.lower()}" for i, s in enumerate(STEPS)])
    ax.set_ylabel("cycle time (s)", color=BLUE, fontsize=7.6)
    ax.tick_params(axis="y", colors=BLUE, labelsize=7.0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.45, alpha=0.65)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False,
              loc="upper left", ncol=2, handlelength=1.6, borderaxespad=0.3)
    ax.set_title("Cycle time and peak force per step", fontsize=8.5,
                 fontweight="bold", color=INK)
    return hot


def main():
    set_pub_style()

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.60], hspace=0.30)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_force = ax_cost.twinx()

    strip_panel(ax_strip)
    hot = cost_panel(ax_cost, ax_force)

    takt = float(CYCLE.sum())
    ax_cost.text(0.995, 0.90, f"takt time {takt:.1f} s",
                 transform=ax_cost.transAxes, ha="right", va="top",
                 fontsize=7.4, color=INK, fontweight="bold")

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.0)
    fig.canvas.draw()
    ax_strip.apply_aspect()
    panel_labels_inches(fig, [ax_strip, ax_cost], dx=-0.20, dy=0.06)
    savefig(fig, "586_rob_assembly_sequence.png")


if __name__ == "__main__":
    main()
