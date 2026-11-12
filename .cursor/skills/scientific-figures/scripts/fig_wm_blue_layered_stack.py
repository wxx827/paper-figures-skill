"""Wm blue layered stack: the five-layer embodied world-model stack.

Five full-width bars, darkest at the bottom and lightest at the top, so the
blue ramp itself encodes the abstraction axis and the reader never has to
consult a legend.  Each bar carries its layer name in the ladder's own
contrast colour (white on the dark steps, ink on the light ones, switched on
relative luminance), and hangs one line of prose plus two or three component
chips off to the right.  A single thin arrow on the left states what the ramp
means.

Drawn on an inch canvas; bar heights and gaps are physical, not fractional.
"""
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas, anchors

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

W, H = 10.4, 5.2
X0, BAR_W = 1.05, 2.90         # bar span
X_TEXT = 4.22                  # where the prose column starts
Y_BOT, Y_TOP = 0.35, 4.47
BAR_H, GAP = 0.68, 0.16

X_NOTE = 10.05                 # right-aligned failure-mode column
X_RULE = 8.78

# Bottom-to-top: name, ladder step, prose, component chips, failure mode.
LAYERS = [
    ("Sensing", 0, "Raw multi-modal streams arriving at 30-200 Hz",
     ["RGB-D", "IMU", "Tactile"], "latency, calibration drift"),
    ("Representation", 1, "Compressed latent state with object-centric slots",
     ["Tokeniser", "Slot encoder", "Episodic memory"], "posterior collapse"),
    ("Dynamics", 2, "Learned transition model rolled out in latent space",
     ["Transition head", "Reward head", "Uncertainty"],
     "compounding rollout error"),
    ("Planning", 3, "Search or an amortised policy over imagined rollouts",
     ["CEM / MPPI", "Value head"], "mis-specified objective"),
    ("Control", 4, "Torque-level tracking at the actuator rate",
     ["Inverse dynamics", "Safety filter"], "sim-to-real torque gap"),
]
BOTTLENECK = 2                 # the one layer allowed the warm accent


def text_on(color):
    """Black-or-white type by relative luminance (BRIEF_WM_BLUE.md, blue A)."""
    r, g, b = mcolors.to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else INK


def bar(ax, y, name, color, index):
    ax.add_patch(FancyBboxPatch(
        (X0, y - BAR_H / 2), BAR_W, BAR_H,
        boxstyle="round,pad=0,rounding_size=0.07", facecolor=color,
        edgecolor="none", zorder=4))
    fg = text_on(color)
    ax.text(X0 + 0.20, y, f"L{index}", ha="left", va="center", color=fg,
            fontsize=7.0, alpha=0.75, zorder=6)
    ax.text(X0 + BAR_W / 2 + 0.14, y, name, ha="center", va="center",
            color=fg, fontsize=10.0, fontweight="bold", zorder=6)
    return anchors(X0 + BAR_W / 2, y, BAR_W, BAR_H)


def chips(ax, x, y, labels):
    for lab in labels:
        w = 0.062 * len(lab) + 0.26
        ax.add_patch(FancyBboxPatch(
            (x, y - 0.12), w, 0.24,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor=BLUE[6],
            edgecolor=BLUE[5], linewidth=0.7, zorder=4))
        ax.text(x + w / 2, y, lab, ha="center", va="center", fontsize=6.5,
                color=INK, zorder=6)
        x += w + 0.16


def abstraction_axis(ax):
    ax.add_patch(FancyArrowPatch((0.62, Y_BOT - 0.05), (0.62, Y_TOP + 0.05),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.1, color=INK, shrinkA=0,
                                 shrinkB=0, zorder=5))
    ax.text(0.40, (Y_BOT + Y_TOP) / 2, "increasing abstraction", rotation=90,
            ha="center", va="center", fontsize=7.4, color=INK)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)

    ax.text(0.15, 5.00, "A five-layer stack for embodied world models",
            fontsize=10.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(0.15, 4.74, "Each layer consumes the one below it; the blue ramp "
            "is the abstraction axis, not a separate variable",
            fontsize=7.2, color="#555555", ha="left", va="center")

    ax.plot([X_RULE, X_RULE], [Y_BOT - 0.05, Y_TOP + 0.30], color="#e1e6ea",
            lw=0.8, zorder=1)
    ax.text(X_NOTE, Y_TOP + 0.16, "typical failure mode", ha="right",
            va="center", fontsize=6.8, color="#777777")

    step = BAR_H + GAP
    for k, (name, tone, prose, comps, note) in enumerate(LAYERS):
        y = Y_BOT + BAR_H / 2 + k * step
        bar(ax, y, name, BLUE[tone], k + 1)
        ax.text(X_TEXT, y + 0.17, prose, ha="left", va="center",
                fontsize=7.4, color=INK)
        chips(ax, X_TEXT, y - 0.17, comps)
        hot = k == BOTTLENECK
        ax.text(X_NOTE, y, note, ha="right", va="center", fontsize=6.8,
                color=ACCENT if hot else "#666666",
                style="italic" if hot else "normal")
        if hot:
            ax.plot([X_RULE - 0.14, X_RULE - 0.14],
                    [y - 0.16, y + 0.16], color=ACCENT, lw=2.0,
                    solid_capstyle="butt", zorder=5)

    abstraction_axis(ax)
    savefig(fig, "1101_wm_blue_layered_stack.png")


if __name__ == "__main__":
    main()
