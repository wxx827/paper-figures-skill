"""Waffle charts: three detectors, one square per percentage point.

A 10x10 grid of rounded tiles is the honest alternative to a pie - area is
counted rather than judged by angle, and 1 % is always the same size in every
panel. Tiles fill left to right, bottom to top, in a fixed category order that
is shared by all three panels, so a category always starts at the same place
unless its share really moved. Per-panel chips give the exact splits; the shared
strip at the bottom is the only place colours are named.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pubstyle import set_pub_style, savefig

CATS = ["Localisation", "Classification", "Duplicate", "Background", "Missed"]
COLORS = ["#3b5b92", "#b23a48", "#c9a227", "#2e7d5b", "#b3b3b3"]
MODELS = [("Faster R-CNN", [34, 26, 8, 19, 13], 4182),
          ("RetinaNet", [33, 27, 6, 21, 13], 3947),
          ("DETR", [41, 22, 4, 18, 15], 3106)]
INK = "#1a1a1a"
SUB = "#555555"


def panel_label(fig, ax, text, dx=-0.20, dy=0.02):
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def tile(ax, x, y, color, size=0.84, r=0.16, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), size, size, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=color, edgecolor="white", linewidth=0.5, zorder=z))


def draw_waffle(ax, name, shares, total):
    seq = np.repeat(np.arange(len(shares)), shares)
    for i, cat in enumerate(seq):
        row, col = divmod(i, 10)
        tile(ax, col * 1.0, row * 1.0, COLORS[cat])

    ax.text(5.0, 11.05, name, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(5.0, 10.35, f"{total:,} false predictions", ha="center",
            va="center", fontsize=6.4, color=SUB)
    for i, (c, s) in enumerate(zip(COLORS, shares)):
        x = i * 2.0 + 0.05
        tile(ax, x, -1.85, c, size=0.62, r=0.13, z=4)
        ax.text(x + 0.82, -1.54, f"{s}%", ha="left", va="center",
                fontsize=6.4, color=INK)
    ax.set_xlim(-0.35, 10.35)
    ax.set_ylim(-2.55, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style()

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.855,
                          bottom=0.135, wspace=0.10)
    axes = []
    for i, (name, shares, total) in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i])
        draw_waffle(ax, name, shares, total)
        axes.append(ax)

    for i, (c, n) in enumerate(zip(COLORS, CATS)):
        x = 0.155 + i * 0.161
        fig.patches.append(plt.Rectangle(
            (x, 0.072), 0.016, 0.024, transform=fig.transFigure,
            facecolor=c, edgecolor="white", linewidth=0.6, zorder=4))
        fig.text(x + 0.023, 0.084, n, ha="left", va="center", fontsize=7.0,
                 color=INK)

    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(fig, ax, lab)
    savefig(fig, "328_pie_waffle_chart.png")


if __name__ == "__main__":
    main()
