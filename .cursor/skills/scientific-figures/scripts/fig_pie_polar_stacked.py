"""Polar stacked ring: one sector per class, radial stacking per difficulty.

Angle selects the class, radius carries the count, and the four difficulty
bands stack outward inside each sector - so unlike a pie this keeps the
absolute magnitudes (sector length) and the internal mix (band lengths) in one
picture. Concentric guides make the radial scale readable. The 62 %-width panel
on the right renormalises every sector to 100 %, which is where the difficulty
mix can be compared without the class-size difference getting in the way.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, get_palette

CLASSES = ["Car", "Pedestrian", "Cyclist", "Bus", "Road sign", "Tree", "Pole"]
LEVELS = ["Easy", "Moderate", "Hard", "Ignored"]
COLORS = get_palette("cb_sequential_blue", 6)[1:5][::-1]
COUNTS = np.array([
    [46, 27, 16, 7],
    [31, 24, 19, 9],
    [14, 12, 11, 5],
    [19, 11, 6, 3],
    [26, 15, 8, 4],
    [34, 18, 9, 6],
    [12, 9, 7, 4],
], dtype=float)
RGRID = [25, 50, 75, 100]
INK = "#1a1a1a"
SUB = "#555555"


def panel_label(fig, ax, text, dx=-0.20, dy=0.02):
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def arc_text(ax, mid_deg, r, s, size, color, weight="normal"):
    rot = (-mid_deg) % 360.0
    if rot > 180:
        rot -= 360
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    ax.text(np.deg2rad(mid_deg), r, s, rotation=rot, rotation_mode="anchor",
            ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, zorder=6)


def draw_polar(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    n = len(CLASSES)
    step = 360.0 / n
    width = np.deg2rad(step * 0.80)
    th = np.linspace(0, 2 * np.pi, 240)
    for g in RGRID:
        ax.plot(th, np.full_like(th, g), lw=0.6, ls=(0, (3, 3)),
                color="#d5d5d5", zorder=1)

    for i in range(n):
        mid = i * step + step / 2.0
        bottom = 0.0
        for j, col in enumerate(COLORS):
            v = COUNTS[i, j]
            ax.bar(np.deg2rad(mid), v, width=width, bottom=bottom, color=col,
                   edgecolor="white", linewidth=1.6, zorder=3)
            bottom += v
        # names ride a fixed outer ring so they read as an axis; the totals
        # stay pinned to the end of their own sector
        arc_text(ax, mid, 122, CLASSES[i], 7.4, INK, weight="bold")
        arc_text(ax, mid, bottom + 8, f"{int(bottom)}k", 6.3, SUB)

    for g in RGRID:
        ax.text(np.deg2rad(-3.5), g, f"{g}", ha="center", va="center",
                fontsize=6.0, color="#8a8a8a", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))
    ax.text(np.deg2rad(0), -44, "instances\nper class\n(thousands)",
            ha="center", va="center", fontsize=6.4, color=SUB,
            linespacing=1.45, zorder=6)
    ax.set_ylim(-46, 132)
    ax.set_axis_off()


def draw_normalised(ax):
    y = np.arange(len(CLASSES))[::-1]
    norm = COUNTS / COUNTS.sum(axis=1, keepdims=True)
    left = np.zeros(len(CLASSES))
    for j, col in enumerate(COLORS):
        ax.barh(y, norm[:, j], left=left, height=0.62, color=col,
                edgecolor="white", linewidth=1.4, zorder=3)
        left += norm[:, j]
    hard = norm[:, 2] + norm[:, 3]
    for yi, h in zip(y, hard):
        ax.text(1.02, yi, f"{h * 100:.0f}%", ha="left", va="center",
                fontsize=6.4, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES, fontsize=7.0)
    ax.set_ylim(-0.7, len(CLASSES) - 0.3)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.2)
    ax.set_xlabel("share within class (%)", fontsize=6.8)
    ax.tick_params(length=2.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(1.02, len(CLASSES) - 0.35, "hard +\nignored", ha="left",
            va="center", fontsize=6.2, color=SUB, linespacing=1.35)
    ax.set_title("Renormalised to 100 % per class", fontsize=8.5,
                 fontweight="bold", color=INK, pad=6, loc="left")


def main():
    set_pub_style()

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62],
                          left=0.005, right=0.90, top=0.88, bottom=0.205,
                          wspace=0.05)
    ax_p = fig.add_subplot(gs[0, 0], projection="polar")
    ax_b = fig.add_subplot(gs[0, 1])

    bb = ax_b.get_position()
    fig.patches.append(plt.Rectangle(
        (bb.x0 - 0.055, bb.y0 - 0.055), bb.width + 0.135, bb.height + 0.135,
        transform=fig.transFigure, facecolor="#f5f5f5", edgecolor="none",
        zorder=-1))
    ax_b.set_facecolor("none")

    draw_polar(ax_p)
    draw_normalised(ax_b)

    handles = [Patch(facecolor=c, edgecolor="white", lw=0.8, label=n)
               for c, n in zip(COLORS, LEVELS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 0.062), columnspacing=1.6,
               handlelength=1.3, handletextpad=0.5)
    panel_label(fig, ax_p, "(a)", dx=0.22)
    panel_label(fig, ax_b, "(b)", dx=-0.42)
    savefig(fig, "329_pie_polar_stacked.png")


if __name__ == "__main__":
    main()
