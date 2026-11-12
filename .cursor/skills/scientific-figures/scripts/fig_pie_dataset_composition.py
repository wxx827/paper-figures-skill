"""Dataset composition donut - the canonical computer-vision paper figure.

Main panel: a 10-slice donut of annotated instances per object category, drawn
with a single-hue lightness ramp (largest slice darkest), 2 pt white separators
and every label pushed outside on a de-overlapped elbow leader. The hole carries
the grand total. Two subordinate panels at 56 % width carry the train/val/test
split donut and the per-class instance counts, sharing the main colour ramp.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from pubstyle import set_pub_style, savefig, PALETTES, style_axes

CLASSES = ["Car", "Pedestrian", "Building", "Vegetation", "Road sign",
           "Bicycle", "Traffic light", "Bus", "Truck", "Others"]
COUNTS = np.array([34210, 21870, 17640, 13920, 9880,
                   7460, 5930, 4380, 3120, 2996], dtype=float)
SPLIT = {"Train": 0.70, "Validation": 0.12, "Test": 0.18}
GREY = "#b3b3b3"
INK = "#1a1a1a"
SUB = "#555555"


def ramp(n, name="greens", lo=0.0, hi=0.88):
    """n colours along a house palette, dark -> light."""
    cmap = LinearSegmentedColormap.from_list(name, PALETTES[name])
    return [cmap(s) for s in np.linspace(lo, hi, n)]


def is_dark(color):
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.55


def panel_label(fig, ax, text, dx=-0.28, dy=0.08):
    """Panel letter at a fixed *inch* offset from the axes' top-left corner."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def place_outside_labels(ax, items, r_edge=1.0, r_stub=1.09, x_knee=1.14,
                         x_text=1.20, min_gap=0.29, y_span=1.24,
                         name_size=7.4, stat_size=6.8):
    """Draw de-overlapped outside labels with elbow leader lines.

    ``items`` is a list of ``(angle_deg, name, stat, colour, radial_offset)``.
    Labels are split into a left and a right column; within a column the anchor
    order along y is preserved while a greedy pass enforces ``min_gap``, which
    is what keeps the leaders from ever crossing each other.
    """
    for side in (+1, -1):
        grp = [it for it in items
               if (np.cos(np.deg2rad(it[0])) >= 0) == (side > 0)]
        if not grp:
            continue
        grp.sort(key=lambda it: -np.sin(np.deg2rad(it[0])))
        ys = [r_edge * np.sin(np.deg2rad(it[0])) for it in grp]
        # top-down pass: push each label below the previous one
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        # bottom-up repair if the column ran off the canvas
        ys[-1] = max(ys[-1], -y_span)
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = max(ys[i], ys[i + 1] + min_gap)
        ys[0] = min(ys[0], y_span)
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)

        for (ang, name, stat, col, off), y in zip(grp, ys):
            a = np.deg2rad(ang)
            px, py = (r_edge + off) * np.cos(a), (r_edge + off) * np.sin(a)
            sx, sy = (r_stub + off) * np.cos(a), (r_stub + off) * np.sin(a)
            kx = side * x_knee
            ax.plot([px, sx, kx, side * (x_text - 0.02)], [py, sy, y, y],
                    lw=0.7, color="#9a9a9a", solid_joinstyle="round",
                    clip_on=False, zorder=1)
            ax.plot([side * (x_text - 0.02)], [y], marker="o", ms=2.1,
                    color=col, clip_on=False, zorder=2)
            ha = "left" if side > 0 else "right"
            tx = side * x_text
            ax.text(tx, y + 0.075, name, ha=ha, va="center",
                    fontsize=name_size, color=INK, clip_on=False)
            ax.text(tx, y - 0.085, stat, ha=ha, va="center",
                    fontsize=stat_size, color=SUB, clip_on=False)


def draw_main_donut(ax, cols):
    total = COUNTS.sum()
    pct = COUNTS / total * 100
    explode = [0.035 if i == 0 else 0.0 for i in range(len(CLASSES))]
    wedges, _ = ax.pie(
        COUNTS, radius=1.0, colors=cols, startangle=90, counterclock=False,
        explode=explode,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2.2))
    wedges[0].set_linewidth(2.6)

    items = []
    for w, name, p, c, col, e in zip(wedges, CLASSES, pct, COUNTS, cols,
                                     explode):
        ang = (w.theta1 + w.theta2) / 2.0
        items.append((ang, name, f"{p:.1f}%  ({int(c):,})", col, e))
        if p >= 9.0:
            a = np.deg2rad(ang)
            rr = 0.81 + e
            ax.text(rr * np.cos(a), rr * np.sin(a), f"{p:.1f}%",
                    ha="center", va="center", fontsize=7.2,
                    fontweight="bold",
                    color="white" if is_dark(col) else INK, zorder=4)
    place_outside_labels(ax, items)

    ax.text(0, 0.13, f"{int(total):,}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0, -0.05, "annotated instances", ha="center", va="center",
            fontsize=7.0, color=SUB)
    ax.text(0, -0.20, "10 categories | 24,118 images", ha="center",
            va="center", fontsize=6.5, color="#8a8a8a")
    ax.text(0, 1.36, "Instance share by object category", ha="center",
            va="center", fontsize=9.0, fontweight="bold", color=INK)
    ax.set_xlim(-1.74, 1.74)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_split_donut(ax, total):
    labels = list(SPLIT)
    frac = np.array([SPLIT[k] for k in labels])
    cols = ["#3b5b92", "#7f9ac0", "#c3cee0"]
    wedges, _ = ax.pie(
        frac, radius=1.0, colors=cols, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.0))
    for w, f, col in zip(wedges, frac, cols):
        a = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        ax.text(0.79 * np.cos(a), 0.79 * np.sin(a), f"{f * 100:.0f}%",
                ha="center", va="center", fontsize=7.0, fontweight="bold",
                color="white" if is_dark(col) else INK)
    ax.text(0, 0.07, "Split", ha="center", va="center", fontsize=8.0,
            fontweight="bold", color=INK)
    ax.text(0, -0.12, "instance-level", ha="center", va="center",
            fontsize=6.2, color="#8a8a8a")
    for i, (k, f, col) in enumerate(zip(labels, frac, cols)):
        y = -1.28 - i * 0.24
        ax.add_patch(plt.Rectangle((-0.86, y - 0.055), 0.16, 0.11,
                                   facecolor=col, edgecolor="white", lw=0.7))
        ax.text(-0.62, y, f"{k}", ha="left", va="center", fontsize=6.5,
                color=INK)
        ax.text(0.92, y, f"{f * total / 1000:.1f}k", ha="right", va="center",
                fontsize=6.5, color=SUB)
    ax.text(0, 1.24, "Train / val / test", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=INK)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.92, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_count_bars(ax, cols):
    y = np.arange(len(CLASSES))[::-1]
    ax.barh(y, COUNTS / 1000.0, color=cols, edgecolor="white", lw=0.9,
            height=0.72, zorder=3)
    for yi, v in zip(y, COUNTS / 1000.0):
        ax.text(v + 0.9, yi, f"{v:.1f}", va="center", ha="left",
                fontsize=6.2, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES, fontsize=6.5)
    ax.set_ylim(-0.7, len(CLASSES) - 0.3)
    ax.set_xlim(0, COUNTS.max() / 1000.0 * 1.22)
    ax.set_xlabel("instances (thousands)", fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    style_axes(ax)
    ax.set_title("Per-class volume", fontsize=8.5, fontweight="bold",
                 color=INK, pad=4)


def main():
    set_pub_style()
    cols = ramp(len(CLASSES) - 1)
    cols.append(GREY)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.56],
                          height_ratios=[1.0, 0.92],
                          left=0.015, right=0.965, top=0.885, bottom=0.085,
                          wspace=0.28, hspace=0.52)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_split = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, 1])

    # grey card behind the two subordinate panels; axes must be transparent
    # and the patch must sit at zorder -1 or it would cover them
    b0, b1 = ax_split.get_position(), ax_bars.get_position()
    x0 = min(b0.x0, b1.x0) - 0.035
    fig.patches.append(plt.Rectangle(
        (x0, b1.y0 - 0.075), (max(b0.x1, b1.x1) + 0.028) - x0,
        (b0.y1 + 0.052) - (b1.y0 - 0.075), transform=fig.transFigure,
        facecolor="#f5f5f5", edgecolor="none", zorder=-1))
    for ax in (ax_split, ax_bars):
        ax.set_facecolor("none")

    draw_main_donut(ax_main, cols)
    draw_split_donut(ax_split, COUNTS.sum())
    draw_count_bars(ax_bars, cols)

    for ax, lab in ((ax_main, "(a)"), (ax_split, "(b)"), (ax_bars, "(c)")):
        panel_label(fig, ax, lab)
    savefig(fig, "323_pie_dataset_composition.png")


if __name__ == "__main__":
    main()
