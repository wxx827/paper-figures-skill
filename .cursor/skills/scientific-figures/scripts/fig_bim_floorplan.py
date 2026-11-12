"""Architectural floor plan of one office storey, coloured by room function.

The main panel is a drafted plan rather than a plotted one: rooms are polygons
tinted by their programme, partitions are drawn with the conventional double
line, doors carry a quarter-circle swing arc and windows are breaks in the
wall.  Every room label, every area on the bar panels and the net-to-gross
figure in the footer are computed from the same rectangle list that draws the
plan, so the drawing and the statistics cannot drift apart.  The two smaller
panels on the right carry the programme evidence: how the floor area is split
between functions, and how densely each function is occupied.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

from pubstyle import set_pub_style, lighten, savefig
from figkit import add_scale_bar, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
WALL = "#3a3a3a"

# Programme colours: house qualitative order, with the neutral grey reserved
# for circulation so that the served spaces read as the figure/ground subject.
FUNC = {
    "Office": "#3b5b92",
    "Meeting": "#2e7d5b",
    "Circulation": "#8a8a8a",
    "Sanitary": "#c9a227",
    "Plant": "#8e6bbf",
}
ORDER = ["Office", "Meeting", "Circulation", "Sanitary", "Plant"]

# name, function, x0, y0, width, height (m), design occupants
ROOMS = [
    ("Open office W", "Office", 0.0, 0.0, 10.0, 10.4, 14),
    ("Office B", "Office", 10.0, 0.0, 6.5, 10.4, 9),
    ("Meeting A", "Meeting", 16.5, 0.0, 7.0, 10.4, 18),
    ("Office C", "Office", 23.5, 0.0, 6.5, 5.2, 4),
    ("Meeting B", "Meeting", 23.5, 5.2, 6.5, 5.2, 8),
    ("Corridor", "Circulation", 0.0, 10.4, 30.0, 2.8, 3),
    ("Office D", "Office", 0.0, 13.2, 8.0, 10.8, 12),
    ("Office E", "Office", 8.0, 13.2, 7.0, 10.8, 10),
    ("Meeting C", "Meeting", 15.0, 13.2, 5.5, 10.8, 14),
    ("WC-W", "Sanitary", 20.5, 13.2, 2.5, 10.8, 2),
    ("WC-M", "Sanitary", 23.0, 13.2, 2.5, 10.8, 2),
    ("Plant room", "Plant", 25.5, 13.2, 4.5, 10.8, 1),
]

BX, BY = 30.0, 24.0            # building envelope, metres
T_EXT, T_INT = 0.35, 0.20      # wall thicknesses, metres
LINE = 0.045                   # drawn thickness of one wall face, metres

# hinge x, hinge y, span direction (deg); leaf swings 90 deg counter-clockwise
DOORS = [
    (5.0, 10.4, 180.0), (13.0, 10.4, 180.0), (20.0, 10.4, 180.0),
    (26.5, 10.4, 180.0), (28.5, 5.2, 180.0),
    (4.0, 13.2, 0.0), (11.0, 13.2, 0.0), (17.0, 13.2, 0.0),
    (21.0, 13.2, 0.0), (23.5, 13.2, 0.0), (26.5, 13.2, 0.0),
]
DOOR_W = 1.0

# x0, y0, x1, y1 of the glazed opening, along an exterior wall
WINDOWS = [
    (1.5, 0.0, 5.0, 0.0), (6.5, 0.0, 9.0, 0.0), (11.0, 0.0, 15.5, 0.0),
    (17.5, 0.0, 22.5, 0.0), (24.5, 0.0, 28.5, 0.0),
    (1.5, 24.0, 6.0, 24.0), (8.5, 24.0, 13.5, 24.0),
    (15.5, 24.0, 19.5, 24.0), (26.0, 24.0, 29.0, 24.0),
    (0.0, 2.0, 0.0, 6.5), (0.0, 15.5, 0.0, 21.0), (0.0, 10.9, 0.0, 12.7),
    (30.0, 1.5, 30.0, 4.0), (30.0, 6.5, 30.0, 9.5),
    (30.0, 10.9, 30.0, 12.7), (30.0, 15.0, 30.0, 20.0),
]


def wall_segment(ax, p0, p1, t, zorder=3):
    """Double-line wall: a dark poche rectangle with a white core inside."""
    (x0, y0), (x1, y1) = p0, p1
    horiz = abs(x1 - x0) >= abs(y1 - y0)
    lo, hi = (min(x0, x1), max(x0, x1)) if horiz else (min(y0, y1), max(y0, y1))
    fixed = y0 if horiz else x0
    if horiz:
        ax.add_patch(Rectangle((lo, fixed - t / 2), hi - lo, t,
                               facecolor=WALL, edgecolor="none",
                               zorder=zorder))
        core = t - 2 * LINE
        # The core stops short of each end so that T- and L-junctions stay
        # solid instead of being punched open by the crossing wall's core.
        ax.add_patch(Rectangle((lo + T_EXT / 2, fixed - core / 2),
                               max(hi - lo - T_EXT, 0.0), core,
                               facecolor="white", edgecolor="none",
                               zorder=zorder + 1))
    else:
        ax.add_patch(Rectangle((fixed - t / 2, lo), t, hi - lo,
                               facecolor=WALL, edgecolor="none",
                               zorder=zorder))
        core = t - 2 * LINE
        ax.add_patch(Rectangle((fixed - core / 2, lo + T_EXT / 2), core,
                               max(hi - lo - T_EXT, 0.0),
                               facecolor="white", edgecolor="none",
                               zorder=zorder + 1))


def collect_walls():
    """Unique wall centrelines with their thickness, taken from the rooms."""
    seen = {}
    for _, _, x, y, w, h, _ in ROOMS:
        edges = [((x, y), (x + w, y)), ((x, y + h), (x + w, y + h)),
                 ((x, y), (x, y + h)), ((x + w, y), (x + w, y + h))]
        for p0, p1 in edges:
            key = tuple(round(v, 3) for v in (*p0, *p1))
            on_ext = (p0[0] == p1[0] and p0[0] in (0.0, BX)) or \
                     (p0[1] == p1[1] and p0[1] in (0.0, BY))
            seen[key] = (p0, p1, T_EXT if on_ext else T_INT)
    return list(seen.values())


def draw_door(ax, hx, hy, ang, t):
    """Erase the wall over the opening, then draw the leaf and swing arc."""
    a = np.deg2rad(ang)
    dx, dy = np.cos(a), np.sin(a)
    if abs(dx) > abs(dy):
        ax.add_patch(Rectangle((min(hx, hx + dx * DOOR_W), hy - t / 2 - 0.01),
                               DOOR_W, t + 0.02, facecolor="white",
                               edgecolor="none", zorder=6))
    else:
        ax.add_patch(Rectangle((hx - t / 2 - 0.01, min(hy, hy + dy * DOOR_W)),
                               t + 0.02, DOOR_W, facecolor="white",
                               edgecolor="none", zorder=6))
    lx, ly = hx - dy * DOOR_W, hy + dx * DOOR_W          # leaf at ang + 90
    ax.plot([hx, lx], [hy, ly], color=WALL, lw=0.9, zorder=7)
    ax.add_patch(Arc((hx, hy), 2 * DOOR_W, 2 * DOOR_W, theta1=ang,
                     theta2=ang + 90.0, edgecolor=WALL, lw=0.55,
                     linestyle=(0, (2.2, 1.6)), zorder=7))


def draw_window(ax, x0, y0, x1, y1, t):
    horiz = abs(x1 - x0) >= abs(y1 - y0)
    if horiz:
        ax.add_patch(Rectangle((x0, y0 - t / 2 - 0.01), x1 - x0, t + 0.02,
                               facecolor="white", edgecolor="none", zorder=6))
        for off, lw in ((-t / 2, 0.8), (t / 2, 0.8), (0.0, 0.5)):
            ax.plot([x0, x1], [y0 + off, y0 + off], color=WALL, lw=lw,
                    zorder=7)
    else:
        ax.add_patch(Rectangle((x0 - t / 2 - 0.01, y0), t + 0.02, y1 - y0,
                               facecolor="white", edgecolor="none", zorder=6))
        for off, lw in ((-t / 2, 0.8), (t / 2, 0.8), (0.0, 0.5)):
            ax.plot([x0 + off, x0 + off], [y0, y1], color=WALL, lw=lw,
                    zorder=7)


def plan_panel(ax):
    for name, cat, x, y, w, h, _ in ROOMS:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=lighten(FUNC[cat], 0.74),
                               edgecolor="none", zorder=1))
    for p0, p1, t in collect_walls():
        wall_segment(ax, p0, p1, t)
    for hx, hy, ang in DOORS:
        draw_door(ax, hx, hy, ang, T_INT)
    for x0, y0, x1, y1 in WINDOWS:
        draw_window(ax, x0, y0, x1, y1, T_EXT)

    for name, cat, x, y, w, h, _ in ROOMS:
        rot = 90.0 if w < 3.5 else 0.0
        cx, cy = x + w / 2, y + h / 2
        off = 0.62 if rot == 0.0 else 0.0
        ax.text(cx - (0.0 if rot == 0.0 else 0.55), cy + off, name,
                ha="center", va="center", rotation=rot, fontsize=6.8,
                color=INK, zorder=8)
        ax.text(cx + (0.0 if rot == 0.0 else 0.62), cy - off,
                f"{w * h:.1f} m$^2$", ha="center", va="center", rotation=rot,
                fontsize=6.2, color=SUB, zorder=8)

    ax.annotate("", xy=(32.6, 23.4), xytext=(32.6, 19.6),
                arrowprops=dict(arrowstyle="-|>", color=WALL, lw=1.0,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(32.6, 23.7, "N", ha="center", va="bottom", fontsize=7.4,
            color=INK, fontweight="bold")
    add_scale_bar(ax, 5.0, "5 m", loc=(0.025, 0.030), height=0.014, n_seg=2,
                  color=WALL, alt="white", text_color=INK, halo="white",
                  fontsize=6.4)

    ax.set_xlim(-3.0, 34.5)
    ax.set_ylim(-3.6, 26.6)
    ax.set_aspect("equal")
    ax.axis("off")

    handles = [Rectangle((0, 0), 1, 1, facecolor=lighten(FUNC[c], 0.74),
                         edgecolor=WALL, linewidth=0.5) for c in ORDER]
    ax.legend(handles, ORDER, loc="upper center", bbox_to_anchor=(0.5, 1.005),
              ncol=5, frameon=False, fontsize=7.0, handlelength=1.3,
              handleheight=0.85, columnspacing=1.1, borderaxespad=0.0)


def area_panel(ax, areas, total):
    cats = ORDER[::-1]
    vals = [areas[c] for c in cats]
    ax.barh(np.arange(len(cats)), [100.0 * v / total for v in vals],
            height=0.62, color=[FUNC[c] for c in cats], edgecolor="white",
            linewidth=0.6)
    for i, (c, v) in enumerate(zip(cats, vals)):
        ax.text(100.0 * v / total + 1.4, i,
                f"{100.0 * v / total:.1f} %  ({v:.0f} m$^2$)", ha="left",
                va="center", fontsize=6.4, color=SUB)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=6.8)
    ax.set_xlim(0.0, 78.0)
    ax.set_xlabel("share of gross floor area (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Programme split by function", fontsize=8.2,
                 fontweight="bold", color=INK)


def density_panel(ax, areas, people):
    cats = ORDER[::-1]
    dens = [100.0 * people[c] / areas[c] for c in cats]
    ax.barh(np.arange(len(cats)), dens, height=0.62,
            color=[FUNC[c] for c in cats], edgecolor="white", linewidth=0.6)
    for i, (c, d) in enumerate(zip(cats, dens)):
        ax.text(d + 0.5, i, f"{d:.1f}  ({people[c]:.0f} p)", ha="left",
                va="center", fontsize=6.4, color=SUB)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=6.8)
    ax.set_xlim(0.0, max(dens) * 1.55)
    ax.set_xlabel("occupant density (persons per 100 m$^2$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Design occupancy", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    areas = {c: 0.0 for c in ORDER}
    people = {c: 0.0 for c in ORDER}
    for _, cat, _, _, w, h, n in ROOMS:
        areas[cat] += w * h
        people[cat] += n
    total = sum(areas.values())
    net = total - areas["Circulation"] - areas["Plant"]

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.58)
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_dens = fig.add_subplot(gs[1, 1])

    plan_panel(ax_plan)
    area_panel(ax_area, areas, total)
    density_panel(ax_dens, areas, people)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    ax_plan.apply_aspect()
    panel_labels_inches(fig, [ax_plan, ax_area, ax_dens], dx=-0.34, dy=0.10)
    savefig(fig, "588_bim_floorplan.png")


if __name__ == "__main__":
    main()
