"""Nested two-ring donut: four parent classes, eleven inherited subclasses.

Inner ring = parent classes in the house qualitative order; outer ring = their
subclasses, each tinted from the parent hue so the family is readable without a
key. Sub-class values ride on de-overlapped elbow leaders outside the disc, the
hole carries the grand total, and the right-hand card is an indented hierarchy
legend that spells out the parent/child tree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pubstyle import set_pub_style, savefig, lighten

TOTAL = 96540
INK = "#1a1a1a"
SUB = "#555555"

# parent -> (colour, [(subclass, share of the whole in %)])  order = descending
TREE = [
    ("Vehicles", "#3b5b92", [("Car", 18.4), ("Truck", 6.2),
                             ("Bus", 3.4), ("Motorcycle", 2.1)]),
    ("Nature", "#2e7d5b", [("Vegetation", 18.3), ("Sky", 8.7)]),
    ("Infrastructure", "#c9a227", [("Building", 12.7), ("Road sign", 5.8),
                                   ("Traffic light", 3.1)]),
    ("People", "#b23a48", [("Pedestrian", 14.8), ("Cyclist", 4.6),
                           ("Rider", 1.9)]),
]


def is_dark(color):
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.55


def panel_label(fig, ax, text, dx=-0.28, dy=0.08):
    """Panel letter at a fixed *inch* offset from the axes' top-left corner."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def place_outside_labels(ax, items, r_edge=1.0, r_stub=1.09, x_knee=1.16,
                         x_text=1.22, min_gap=0.235, y_span=1.26,
                         name_size=7.2, stat_size=6.6):
    """Elbow leader labels that cannot overlap and cannot cross.

    ``items`` = ``(angle_deg, name, stat, colour)``.  Anchors are sorted by y
    inside each column and a greedy top-down / bottom-up pass enforces a
    minimum spacing while preserving that order, so leaders stay nested.
    """
    for side in (+1, -1):
        grp = [it for it in items
               if (np.cos(np.deg2rad(it[0])) >= 0) == (side > 0)]
        if not grp:
            continue
        grp.sort(key=lambda it: -np.sin(np.deg2rad(it[0])))
        ys = [r_edge * np.sin(np.deg2rad(it[0])) for it in grp]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        ys[-1] = max(ys[-1], -y_span)
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = max(ys[i], ys[i + 1] + min_gap)
        ys[0] = min(ys[0], y_span)
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)

        for (ang, name, stat, col), y in zip(grp, ys):
            a = np.deg2rad(ang)
            ax.plot([r_edge * np.cos(a), r_stub * np.cos(a),
                     side * x_knee, side * (x_text - 0.02)],
                    [r_edge * np.sin(a), r_stub * np.sin(a), y, y],
                    lw=0.7, color="#9a9a9a", clip_on=False, zorder=1)
            ax.plot([side * (x_text - 0.02)], [y], marker="o", ms=2.0,
                    color=col, clip_on=False, zorder=2)
            ha = "left" if side > 0 else "right"
            ax.text(side * x_text, y + 0.065, name, ha=ha, va="center",
                    fontsize=name_size, color=INK, clip_on=False)
            ax.text(side * x_text, y - 0.075, stat, ha=ha, va="center",
                    fontsize=stat_size, color=SUB, clip_on=False)


def flatten():
    parents, pcols, subs, scols, svals = [], [], [], [], []
    for name, col, kids in TREE:
        parents.append((name, sum(v for _, v in kids)))
        pcols.append(col)
        for i, (kn, kv) in enumerate(kids):
            subs.append(kn)
            svals.append(kv)
            scols.append(lighten(col, 0.10 + 0.50 * i / max(len(kids) - 1, 1)))
    return parents, pcols, subs, scols, np.array(svals)


def draw_nested(ax, parents, pcols, subs, scols, svals):
    outer, _ = ax.pie(
        svals, radius=1.0, colors=scols, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor="white", linewidth=2.0))
    inner, _ = ax.pie(
        [v for _, v in parents], radius=0.66, colors=pcols, startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor="white", linewidth=2.4))

    for w, (_, v), col in zip(inner, parents, pcols):
        a = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        ax.text(0.51 * np.cos(a), 0.51 * np.sin(a), f"{v:.1f}%",
                ha="center", va="center", fontsize=7.4, fontweight="bold",
                color="white" if is_dark(col) else INK, zorder=4)

    items = []
    for w, name, v, col in zip(outer, subs, svals, scols):
        ang = (w.theta1 + w.theta2) / 2.0
        items.append((ang, name, f"{v:.1f}%  ({int(v / 100 * TOTAL):,})", col))
    place_outside_labels(ax, items)

    ax.text(0, 0.10, f"{TOTAL:,}", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    ax.text(0, -0.09, "instances", ha="center", va="center", fontsize=7.0,
            color=SUB)
    ax.text(0, 1.40, "Two-level class taxonomy", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=INK)
    ax.text(0, -1.40, "inner ring: 4 parent classes    "
            "outer ring: 11 subclasses", ha="center", va="center",
            fontsize=6.6, color="#8a8a8a")
    ax.set_xlim(-1.80, 1.80)
    ax.set_ylim(-1.48, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_tree_legend(ax, parents, pcols):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n_rows = len(TREE) + sum(len(k) for _, _, k in TREE)
    step = 0.855 / (n_rows + len(TREE) * 0.55)
    spine = 0.105
    y = 0.895
    for (pname, pcol, kids), (_, ptot) in zip(TREE, parents):
        ax.add_patch(plt.Rectangle((0.055, y - 0.021), 0.058, 0.042,
                                   facecolor=pcol, edgecolor="white", lw=0.8,
                                   zorder=3))
        ax.text(0.140, y, pname, ha="left", va="center", fontsize=7.6,
                fontweight="bold", color=INK)
        ax.text(0.965, y, f"{ptot:.1f}%", ha="right", va="center",
                fontsize=7.4, fontweight="bold", color=INK)
        kid_y = [y - step * (i + 1) for i in range(len(kids))]
        ax.plot([spine, spine], [y - 0.028, kid_y[-1]], lw=0.7,
                color="#c4c4c4", zorder=1)
        for i, ((kn, kv), ky) in enumerate(zip(kids, kid_y)):
            col = lighten(pcol, 0.10 + 0.50 * i / max(len(kids) - 1, 1))
            ax.plot([spine, 0.155], [ky, ky], lw=0.7, color="#c4c4c4",
                    zorder=1)
            ax.add_patch(plt.Rectangle((0.160, ky - 0.016), 0.045, 0.032,
                                       facecolor=col, edgecolor="white",
                                       lw=0.7, zorder=3))
            ax.text(0.225, ky, kn, ha="left", va="center", fontsize=6.9,
                    color=SUB)
            ax.text(0.965, ky, f"{kv:.1f}%", ha="right", va="center",
                    fontsize=6.9, color=SUB)
        y = kid_y[-1] - step * 1.55
    ax.text(0.055, 0.965, "Class hierarchy", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=INK)


def main():
    set_pub_style()
    parents, pcols, subs, scols, svals = flatten()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.56],
                          left=0.015, right=0.985, top=0.885, bottom=0.055,
                          wspace=0.10)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    bb = ax_leg.get_position()
    fig.patches.append(plt.Rectangle(
        (bb.x0 - 0.005, bb.y0 + 0.055), bb.width * 0.99, bb.height * 0.90,
        transform=fig.transFigure, facecolor="#f5f5f5", edgecolor="none",
        zorder=-1))
    ax_leg.set_facecolor("none")

    draw_nested(ax_pie, parents, pcols, subs, scols, svals)
    draw_tree_legend(ax_leg, parents, pcols)

    for ax, lab in ((ax_pie, "(a)"), (ax_leg, "(b)")):
        panel_label(fig, ax, lab)
    savefig(fig, "324_pie_nested_donut.png")


if __name__ == "__main__":
    main()
