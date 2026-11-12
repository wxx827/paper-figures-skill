"""Funnel chart: staged attrition drawn as true centred trapezoids.

Every stage is a trapezoid whose top edge matches the previous stage's width
and whose bottom edge matches its own, so the outline is a continuous funnel
rather than a stack of bars. One sequential colour ramp encodes depth in the
pipeline; each stage carries its absolute count and its share of the intake,
and the gap between stages is annotated with the step-to-step retention.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, PALETTES, savefig

STAGES = ["Samples collected", "Passed quality control", "Features extracted",
          "Used for model training", "Passed validation", "Final output"]
VALUES = [12480, 10870, 8990, 7240, 6120, 5240]
COLS = PALETTES["cb_sequential_blue"][::-1][:6]


def text_colour(face):
    r, g, b = mcolors.to_rgb(face)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else "#1a1a1a"


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    n = len(STAGES)
    total = VALUES[0]
    widths = [0.92 * v / total for v in VALUES]
    widths.append(widths[-1] * 0.86)          # slight taper on the last band
    band, gap = 1.0, 0.16

    for i, (stage, val, col) in enumerate(zip(STAGES, VALUES, COLS)):
        y_top = -(i * (band + gap))
        y_bot = y_top - band
        wt, wb = widths[i], widths[i + 1]
        ax.add_patch(Polygon([(-wt / 2, y_top), (wt / 2, y_top),
                              (wb / 2, y_bot), (-wb / 2, y_bot)],
                             closed=True, facecolor=col, edgecolor="white",
                             linewidth=1.4, zorder=3))
        tc = text_colour(col)
        ax.text(0, y_top - band * 0.38, stage, ha="center", va="center",
                fontsize=9, fontweight="bold", color=tc, zorder=4)
        ax.text(0, y_top - band * 0.72, f"{val:,} samples", ha="center",
                va="center", fontsize=8, color=tc, alpha=0.92, zorder=4)
        ax.text(0.60, y_top - band / 2, f"{val / total:.0%}", ha="left",
                va="center", fontsize=9, fontweight="bold", color="#333333")
        if i:
            keep = val / VALUES[i - 1] - 1
            ax.text(-0.60, y_top + gap / 2, f"{keep:+.1%}", ha="right",
                    va="center", fontsize=7.5, color="#b23a48")

    ax.text(0.60, 0.42, "of intake", ha="left", va="center", fontsize=7.5,
            color="#777777")
    ax.text(-0.60, 0.42, "step change", ha="right", va="center", fontsize=7.5,
            color="#777777")
    ax.set_xlim(-0.86, 0.86)
    ax.set_ylim(-(n * (band + gap)), 0.85)
    ax.axis("off")
    ax.set_title("Sample attrition through the analysis pipeline",
                 fontsize=11, fontweight="bold", pad=4)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    savefig(fig, "50_funnel.png")


if __name__ == "__main__":
    main()
