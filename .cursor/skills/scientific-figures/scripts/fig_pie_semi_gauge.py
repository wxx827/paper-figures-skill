"""Four semicircular gauges with target markers, over a shared-scale history.

Each gauge is a half donut: a light track, a value arc, a triangular target
marker on the rim and a large centred number. Colour is driven by the target
test alone - amber when the metric misses its target, green when it clears it -
so the pass/fail read happens before any number is parsed. The 55 %-height
strip underneath plots the same four metrics over training on one shared y
range, so the gauges are snapshots of curves the reader can actually see.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from pubstyle import set_pub_style, savefig, lighten, style_axes

METRICS = [("Accuracy", 91.4, 90.0), ("Recall", 86.2, 88.0),
           ("F1 score", 88.7, 88.0), ("mAP@0.5", 79.5, 82.0)]
PASS, MISS = "#2e7d5b", "#c9a227"
TRACK = "#ececec"
INK = "#1a1a1a"
SUB = "#555555"
YLIM = (58.0, 100.0)


def panel_label(fig, ax, text, dx=-0.16, dy=0.02):
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def draw_gauge(ax, name, value, target):
    col = PASS if value >= target else MISS
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.27, facecolor=TRACK,
                       edgecolor="white", linewidth=1.4, zorder=2))
    ax.add_patch(Wedge((0, 0), 1.0, 180 - 1.8 * value, 180, width=0.27,
                       facecolor=col, edgecolor="white", linewidth=1.8,
                       zorder=3))

    deg = 180 - 1.8 * target
    a = np.deg2rad(deg)
    ax.plot([0.66 * np.cos(a), 1.04 * np.cos(a)],
            [0.66 * np.sin(a), 1.04 * np.sin(a)], lw=1.1, color="#333333",
            zorder=5)
    # rotate the triangle so it always points back at the hub
    ax.plot([1.13 * np.cos(a)], [1.13 * np.sin(a)], marker=(3, 0, deg + 90),
            ms=5.0, color="#333333", zorder=5, clip_on=False)

    ax.text(0, 0.33, f"{value:.1f}%", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    delta = value - target
    ax.text(0, 0.14, f"{delta:+.1f} vs target", ha="center", va="center",
            fontsize=6.2, color=col, fontweight="bold")
    ax.text(0, 1.30, name, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(-0.85, -0.12, "0", ha="center", va="center", fontsize=6.0,
            color="#8a8a8a")
    ax.text(0.85, -0.12, "100", ha="center", va="center", fontsize=6.0,
            color="#8a8a8a")
    ax.text(0, -0.30, f"target {target:.0f}%", ha="center", va="center",
            fontsize=6.4, color=SUB)
    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-0.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")
    return col


def draw_history(ax, rng, name, value, target, col, show_y):
    ep = np.arange(1, 41)
    start = value - rng.uniform(17, 24)
    curve = start + (value - start) * (1 - np.exp(-ep / 11.0))
    curve += rng.normal(0, 0.55, ep.size) * np.exp(-ep / 26.0)
    curve[-1] = value
    ax.fill_between(ep, YLIM[0], curve, color=lighten(col, 0.78), zorder=2)
    ax.plot(ep, curve, lw=1.4, color=col, zorder=4)
    ax.axhline(target, lw=0.8, ls=(0, (3, 2)), color="#8a8a8a", zorder=3)
    ax.plot([ep[-1]], [value], marker="o", ms=3.6, color=col,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 42)
    ax.set_yticks([60, 80, 100])
    ax.set_xticks([1, 20, 40])
    ax.tick_params(labelsize=6.2)
    ax.set_xlabel("epoch (-)", fontsize=6.8)
    if show_y:
        ax.set_ylabel("score (%)", fontsize=6.8)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(327)

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.90, 1.0],
                          left=0.075, right=0.985, top=0.865, bottom=0.175,
                          wspace=0.22, hspace=0.38)
    gauges, hists = [], []
    for i, (name, value, target) in enumerate(METRICS):
        ax_g = fig.add_subplot(gs[0, i])
        ax_h = fig.add_subplot(gs[1, i])
        col = draw_gauge(ax_g, name, value, target)
        draw_history(ax_h, rng, name, value, target, col, show_y=(i == 0))
        gauges.append(ax_g)
        hists.append(ax_h)

    b0, b1 = hists[0].get_position(), hists[-1].get_position()
    fig.patches.append(plt.Rectangle(
        (b0.x0 - 0.062, b0.y0 - 0.105), (b1.x1 + 0.010) - (b0.x0 - 0.062),
        b0.height + 0.155, transform=fig.transFigure, facecolor="#f5f5f5",
        edgecolor="none", zorder=-1))
    for ax in hists:
        ax.set_facecolor("none")

    for ax, lab in zip(gauges, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(fig, ax, lab)
    savefig(fig, "327_pie_semi_gauge.png")


if __name__ == "__main__":
    main()
