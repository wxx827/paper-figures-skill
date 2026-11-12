"""Benchmark construction workflow with a quality feedback loop.

Four stages -- task sampling, scene synthesis, annotation and quality control
-- each live in a rounded dashed container with a solid header bar and two to
three sub-modules carrying their own counters.  Accepted episodes leave the
last stage towards the release card; rejected ones travel back along a dashed
arc that re-enters scene synthesis, which is the point of the figure.

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pubstyle import set_pub_style, savefig, lighten

BLUE = "#3b5b92"
ORANGE = "#c0603f"
GREEN = "#2e7d5b"
RED = "#b23a48"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
FAINT = "#8a8a8a"
LINE = "#d5d5d5"

R = 0.06
LW = 0.9
DASH = (0, (6, 3))
SEED = 1484  # original id; figure number is 1079

# Rejection is the only free parameter of the loop; the accept share and the
# expected number of synthesis rounds per released task both follow from it.
REJECT = 0.124
ACCEPT = 1.0 - REJECT
ROUNDS = 1.0 / ACCEPT


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def container(ax, x0, x1, y0, y1, color, rounding=0.14, zorder=1):
    w, h = x1 - x0, y1 - y0
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=color, alpha=0.04, edgecolor="none",
                                zorder=zorder))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor="none", edgecolor=color,
                                linewidth=1.0, linestyle=DASH,
                                zorder=zorder + 0.5))


def header(ax, cx, cy, w, h, text, color, zorder=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor=color, edgecolor=color,
                                linewidth=0.8, zorder=zorder))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.4,
            fontweight="bold", color="white", zorder=zorder + 1)


def sub_item(ax, cx, cy, w, h, label, metric, color, zorder=3):
    """Gradient module row: bold label on the left, mono counter on the right."""
    tint = lighten(color, 0.80)
    base = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={R}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(base)
    cmap = LinearSegmentedColormap.from_list("g", ["white", tint])
    im = ax.imshow(np.linspace(0, 1, 128).reshape(-1, 1),
                   extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
                   origin="lower", cmap=cmap, vmin=0, vmax=1,
                   zorder=zorder + 0.2, interpolation="bilinear")
    im.set_clip_path(base)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="none", edgecolor=color,
                                linewidth=LW, zorder=zorder + 0.4))
    ax.text(cx - w / 2 + 0.13, cy, label, ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=INK, zorder=zorder + 1)
    ax.text(cx + w / 2 - 0.13, cy, metric, ha="right", va="center",
            fontsize=6.2, family="monospace", color=MUTED, zorder=zorder + 1)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.0, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def plate_text(ax, x, y, text, color=MUTED, fontsize=6.8, weight="normal",
               zorder=9):
    return ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                   color=color, fontweight=weight, zorder=zorder,
                   bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                             edgecolor="none"))


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 4.60
Y0, Y1 = 1.68, 3.88
ROWS = (3.08, 2.56, 2.04)
HEAD_Y = 3.62
STAGES = [
    (0.45, 3.11, BLUE, "Stage 1 · Task sampling",
     [("Capability taxonomy", "12 axes"),
      ("Difficulty stratifier", "5 tiers"),
      ("Seed task sampler", "6.0k")]),
    (3.41, 6.07, ORANGE, "Stage 2 · Scene synthesis",
     [("Asset retrieval", "38k mesh"),
      ("Layout + physics solver", "MuJoCo"),
      ("Multi-view renderer", "8 views")]),
    (6.37, 9.04, GREEN, "Stage 3 · Annotation",
     [("Program-generated GT", "exact"),
      ("VLM auto-caption", "3 votes"),
      ("Human spot-check", "5 %")]),
    (9.34, 12.00, RED, "Stage 4 · Quality control",
     [("Solvability check", "91.4 %"),
      ("Ambiguity filter", "96.8 %"),
      ("Difficulty calibration", "IRT")]),
]


def draw_stage(ax, x0, x1, color, title, items):
    cx = (x0 + x1) / 2.0
    container(ax, x0, x1, Y0, Y1, color)
    header(ax, cx, HEAD_Y, (x1 - x0) - 0.24, 0.34, title, color)
    for y, (label, metric) in zip(ROWS, items):
        sub_item(ax, cx, y, (x1 - x0) - 0.34, 0.42, label, metric, color)
    return cx


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 4.36, "Skill-driven benchmark construction workflow",
            ha="left", va="center", fontsize=12.0, fontweight="bold",
            color=INK)
    ax.text(0.45, 4.11, "every episode must clear quality control; rejected "
                        "episodes are sent back to scene synthesis instead of "
                        "being dropped", ha="left", va="center", fontsize=7.8,
            color=MUTED)

    cxs = [draw_stage(ax, *s) for s in STAGES]

    for i in range(3):
        arrow(ax, (STAGES[i][1], ROWS[1]), (STAGES[i + 1][0], ROWS[1]), lw=1.2)

    # accepted episodes leave towards the release card
    card_x, card_y = 11.00, 0.90
    arrow(ax, (card_x, Y0), (card_x, card_y + 0.28), color=GREEN, lw=1.2)
    plate_text(ax, card_x + 0.02, 1.34, f"accept  {100 * ACCEPT:.1f} %",
               color=GREEN, fontsize=6.6, weight="bold")
    ax.add_patch(FancyBboxPatch((10.05, card_y - 0.28), 1.90, 0.56,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="#f5f0fa", edgecolor=PURPLE,
                                linewidth=1.0, zorder=4))
    ax.text(card_x, card_y + 0.10, "Benchmark v1.0", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=PURPLE, zorder=6)
    ax.text(card_x, card_y - 0.11, "4,820 tasks  ·  12 capabilities",
            ha="center", va="center", fontsize=6.2, color=MUTED, zorder=6)

    # rejected episodes flow back into stage 2
    arrow(ax, (cxs[3] - 0.60, Y0), (cxs[1], Y0), color=RED, lw=1.1, ls=DASH,
          rad=-0.30, ms=13)
    plate_text(ax, 7.35, 0.86, f"reject  {100 * REJECT:.1f} %  ·  re-synthesise "
                               "the scene with a new seed", color=RED,
               fontsize=6.8, weight="bold")

    # flow legend, bottom left
    ax.add_patch(FancyBboxPatch((0.45, 0.50), 3.85, 0.86,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f5f5f5", edgecolor=LINE,
                                linewidth=0.8, zorder=1))
    rows = [(1.16, GREEN, "-",
             r"accepted episode  $\rightarrow$  release package"),
            (0.92, RED, DASH,
             r"rejected episode  $\rightarrow$  Stage 2 re-synthesis")]
    for y, col, ls, txt in rows:
        arrow(ax, (0.66, y), (1.06, y), color=col, ls=ls, lw=1.1, ms=11,
              zorder=4)
        ax.text(1.20, y, txt, ha="left", va="center", fontsize=6.8,
                color=INK, zorder=4)
    ax.text(1.20, 0.68, f"mean {ROUNDS:.2f} synthesis rounds per released task",
            ha="left", va="center", fontsize=6.4, color=MUTED, zorder=4)

    savefig(fig, "1079_wm_arch_benchmark_loop.png")


if __name__ == "__main__":
    main()
