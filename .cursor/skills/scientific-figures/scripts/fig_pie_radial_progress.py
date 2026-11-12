"""Nine backbones as concentric radial progress rings, ranked best to worst.

Every cell holds three round-capped arcs on a light track - precision outside,
recall in the middle, mAP inside - all starting at 12 o'clock and running
clockwise, so ring length is directly comparable across the whole grid. The hole
carries the composite score that the grid is sorted by, and the read-out beside
each dial gives the exact numbers the arcs only approximate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

MODELS = ["ResNet-50", "ResNet-101", "ConvNeXt-T", "Swin-T", "ViT-B/16",
          "EfficientNet-B3", "YOLOv8-M", "DETR-R50", "RT-DETR"]
METRICS = ["Precision", "Recall", "mAP@0.5"]
COLORS = ["#3b5b92", "#2e7d5b", "#c9a227"]
RADII = [1.00, 0.74, 0.48]
TRACK = "#ececec"
LW = 5.5
INK = "#1a1a1a"
SUB = "#555555"


def arc(ax, r, frac, color, lw=LW, z=3):
    t = np.linspace(0, 2 * np.pi * float(np.clip(frac, 0, 1)), 200)
    ax.plot(r * np.sin(t), r * np.cos(t), lw=lw, color=color,
            solid_capstyle="round", zorder=z)


def draw_cell(ax, name, scores, rank):
    th = np.linspace(0, 2 * np.pi, 200)
    for r in RADII:
        ax.plot(r * np.cos(th), r * np.sin(th), lw=LW, color=TRACK, zorder=2)
    for r, v, c in zip(RADII, scores, COLORS):
        arc(ax, r, v, c)

    ax.text(0, 0.07, f"{np.mean(scores) * 100:.1f}", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=INK)
    ax.text(0, -0.13, "overall", ha="center", va="center", fontsize=6.0,
            color="#8a8a8a")
    ax.text(-1.30, 1.26, name, ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=INK)
    ax.text(3.32, 1.26, f"#{rank}", ha="right", va="center", fontsize=7.2,
            color="#8a8a8a", fontweight="bold")

    for i, (m, v, c) in enumerate(zip(METRICS, scores, COLORS)):
        y = 0.46 - i * 0.42
        ax.plot([1.45], [y], marker="o", ms=3.6, color=c, zorder=4)
        ax.text(1.63, y, m, ha="left", va="center", fontsize=6.8, color=SUB)
        ax.text(3.32, y, f"{v:.3f}", ha="right", va="center", fontsize=7.0,
                color=INK, fontweight="bold")
    ax.set_xlim(-1.34, 3.40)
    ax.set_ylim(-1.40, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style()
    rng = np.random.default_rng(331)
    scores = np.clip(rng.normal(0.83, 0.065, (len(MODELS), 3)), 0.58, 0.97)
    order = np.argsort(-scores.mean(axis=1))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 3, left=0.015, right=0.985, top=0.895,
                          bottom=0.115, wspace=0.06, hspace=0.16)
    for slot, idx in enumerate(order):
        ax = fig.add_subplot(gs[slot // 3, slot % 3])
        draw_cell(ax, MODELS[idx], scores[idx], slot + 1)

    handles = [Line2D([0], [0], color=c, lw=3.2, solid_capstyle="round",
                      label=f"{m}  ({p} ring)")
               for c, m, p in zip(COLORS, METRICS, ["outer", "middle",
                                                    "inner"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.045), columnspacing=2.4,
               handlelength=1.6, handletextpad=0.6)
    savefig(fig, "331_pie_radial_progress.png")


if __name__ == "__main__":
    main()
