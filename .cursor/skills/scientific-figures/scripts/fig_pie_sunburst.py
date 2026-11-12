"""Three-level sunburst built by hand on a polar axis, plus an icicle twin.

Each ring is a set of polar bars whose angular span is inherited from the parent
node, so the tree is exact rather than eyeballed. Ring 1 and ring 2 carry text
rotated along the arc, leaf names radiate outward, and the hole holds the root.
The right-hand card unrolls the identical tree into a three-row icicle chart,
which is the easiest way to check that the angles really do nest.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
SUB = "#555555"
L1_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227", "#b23a48"]

TREE = {
    "Vehicles": {"Road": {"Car": 12, "Truck": 6, "Bus": 4},
                 "Rail": {"Tram": 4, "Train": 3},
                 "Air": {"Drone": 3, "Plane": 2}},
    "Living": {"People": {"Adult": 11, "Child": 4},
               "Animals": {"Dog": 5, "Bird": 4},
               "Cyclists": {"Rider": 4}},
    "Static": {"Buildings": {"House": 8, "Tower": 3},
               "Signage": {"Sign": 7, "Light": 5}},
    "Terrain": {"Ground": {"Roadway": 6, "Sidewalk": 4},
                "Nature": {"Tree": 3, "Grass": 2}},
}
RINGS = [(0.32, 0.26), (0.58, 0.24), (0.82, 0.22)]   # (inner radius, width)


def is_dark(color):
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.55


def panel_label(fig, ax, text, dx=-0.28, dy=0.08):
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def layout():
    """Flatten the tree into (level, start%, span%, name, colour) records."""
    total = sum(sum(sum(l.values()) for l in g.values())
                for g in TREE.values())
    out, cursor = [], 0.0
    for gi, (g_name, mids) in enumerate(TREE.items()):
        g_val = sum(sum(l.values()) for l in mids.values())
        g_col = L1_COLORS[gi]
        out.append((0, cursor, g_val / total * 100, g_name, g_col))
        c1 = cursor
        for mi, (m_name, leaves) in enumerate(mids.items()):
            m_val = sum(leaves.values())
            m_col = lighten(g_col, 0.16 + 0.14 * mi)
            out.append((1, c1, m_val / total * 100, m_name, m_col))
            c2 = c1
            for li, (l_name, l_val) in enumerate(leaves.items()):
                l_col = lighten(m_col, 0.16 + 0.11 * li)
                out.append((2, c2, l_val / total * 100, l_name, l_col))
                c2 += l_val / total * 100
            c1 += m_val / total * 100
        cursor += g_val / total * 100
    return out, total


def arc_text(ax, mid_deg, r, s, size, color, weight="normal"):
    """Text laid along the arc, auto-flipped so it never reads upside down."""
    rot = (-mid_deg) % 360.0
    if rot > 180:
        rot -= 360
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    ax.text(np.deg2rad(mid_deg), r, s, rotation=rot, rotation_mode="anchor",
            ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, zorder=5)


def radial_text(ax, mid_deg, r, s, size, color):
    """Leaf label radiating outward from the rim."""
    sa = 90.0 - mid_deg
    if np.cos(np.deg2rad(sa)) >= 0:
        rot, ha = sa, "left"
    else:
        rot, ha = sa + 180, "right"
    ax.text(np.deg2rad(mid_deg), r, s, rotation=rot, rotation_mode="anchor",
            ha=ha, va="center", fontsize=size, color=color, clip_on=False,
            zorder=5)


def draw_sunburst(ax, records):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for level, start, span, name, col in records:
        r0, w = RINGS[level]
        ax.bar(np.deg2rad(start * 3.6 + span * 1.8), w,
               width=np.deg2rad(span * 3.6), bottom=r0, color=col,
               edgecolor="white", linewidth=2.0, align="center", zorder=3)
        mid = (start + span / 2.0) * 3.6
        if level == 0:
            arc_text(ax, mid, r0 + w / 2.0, f"{name}\n{span:.0f}%", 7.6,
                     "white" if is_dark(col) else INK, weight="bold")
        elif level == 1 and span * 3.6 >= 20:
            arc_text(ax, mid, r0 + w / 2.0, name, 6.8,
                     "white" if is_dark(col) else INK)
        elif level == 2:
            radial_text(ax, mid, RINGS[2][0] + RINGS[2][1] + 0.035,
                        f"{name}  {span:.0f}%", 6.4, SUB)
    ax.text(0, 0, "Annotated\ncorpus", ha="center", va="center", fontsize=8.4,
            fontweight="bold", color=INK, linespacing=1.4, zorder=6)
    # headroom for the radiating leaf labels, so nothing spills off-canvas
    ax.set_ylim(0, 1.42)
    ax.set_axis_off()


def draw_icicle(ax, records):
    ax.set_xlim(-1.5, 101.5)
    ax.set_ylim(-0.35, 3.05)
    ax.axis("off")
    names = ["Level 1  parent", "Level 2  group", "Level 3  leaf"]
    for level in range(3):
        y = 2.0 - level
        for lv, start, span, name, col in records:
            if lv != level:
                continue
            ax.add_patch(plt.Rectangle((start, y), span, 0.62,
                                       facecolor=col, edgecolor="white",
                                       lw=1.8, zorder=3))
            txt = f"{name}\n{span:.0f}%" if level < 2 else name
            if span >= (9 if level < 2 else 5.5):
                ax.text(start + span / 2.0, y + 0.31, txt, ha="center",
                        va="center", fontsize=6.3, linespacing=1.35,
                        color="white" if is_dark(col) else INK, zorder=4)
        ax.text(0, y + 0.74, names[level], ha="left", va="center",
                fontsize=6.8, color=SUB)
    for x in (0, 25, 50, 75, 100):
        ax.plot([x, x], [-0.20, -0.06], lw=0.7, color="#aaaaaa", zorder=1)
        ax.text(x, -0.30, f"{x}%", ha="center", va="center", fontsize=6.2,
                color=SUB)
    ax.text(0, 2.98, "Same tree, unrolled", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=INK)


def main():
    set_pub_style()
    records, _ = layout()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58],
                          left=0.02, right=0.975, top=0.885, bottom=0.075,
                          wspace=0.14)
    ax_sun = fig.add_subplot(gs[0, 0], projection="polar")
    ax_ice = fig.add_subplot(gs[0, 1])

    bb = ax_ice.get_position()
    fig.patches.append(plt.Rectangle(
        (bb.x0 - 0.018, bb.y0 - 0.012), bb.width + 0.036, bb.height + 0.024,
        transform=fig.transFigure, facecolor="#f5f5f5", edgecolor="none",
        zorder=-1))
    ax_ice.set_facecolor("none")

    draw_sunburst(ax_sun, records)
    draw_icicle(ax_ice, records)

    panel_label(fig, ax_sun, "(a)", dx=0.10)
    panel_label(fig, ax_ice, "(b)", dx=-0.20)
    savefig(fig, "325_pie_sunburst.png")


if __name__ == "__main__":
    main()
