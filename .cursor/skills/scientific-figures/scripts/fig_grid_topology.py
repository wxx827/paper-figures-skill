"""Single-line transmission network topology of a three-area power system.

Buses are drawn by function (square = generation, circle = load, triangle =
substation / switching station), branches are coloured by voltage class and
scaled in width by thermal capacity.  Control areas sit on pale background
cards; inter-area corridors carry a white casing plus their aggregate
transfer capability.

Layout note: the schematic owns the full canvas and the four legend blocks
live in a narrow right-hand rail, so no key ever covers the network.  Node
coordinates are laid out on a 12.4 x 8.6 frame whose aspect matches the axes
box, which keeps ``aspect='equal'`` from letterboxing the drawing.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig

# ---- shared energy colour system (kept identical across the energy set) ----
FUEL = {
    "coal":    "#5b5f64",
    "gas":     "#cf8447",
    "nuclear": "#8478a6",
    "wind":    "#4f9e91",
    "solar":   "#dcb35c",
    "hydro":   "#4a7f9e",
}
INK = "#333333"

# voltage classes: one sequential blue ramp keeps the branch layer unified
KV = {
    500: dict(color="#12446e", label="500 kV"),
    230: dict(color="#4e8fb5", label="230 kV"),
    115: dict(color="#a9bcc9", label="115 kV"),
}

# bus id -> (x, y, kind, label offset)
BUSES = {
    "N1": (1.25, 7.60, "gen",  (0.00,  0.42)),
    "N2": (4.20, 7.75, "gen",  (0.00,  0.42)),
    "N3": (2.70, 6.55, "sub",  (0.00, -0.46)),
    "N4": (1.05, 5.55, "load", (-0.48, 0.05)),
    "N5": (4.55, 5.60, "load", (0.00,  0.42)),
    "E1": (7.45, 7.55, "gen",  (0.00,  0.42)),
    "E2": (8.90, 6.95, "sub",  (0.05,  0.44)),
    "E3": (11.30, 7.70, "load", (0.00, 0.42)),
    "E4": (7.05, 5.60, "load", (0.00, -0.46)),
    "E5": (9.85, 5.60, "sub",  (0.54, -0.10)),
    "E6": (11.55, 6.35, "gen", (0.10, -0.46)),
    "S1": (2.95, 2.95, "sub",  (0.40, -0.30)),
    "S2": (2.10, 1.05, "gen",  (0.00, -0.46)),
    "S3": (4.70, 0.85, "load", (0.00, -0.46)),
    "S4": (6.30, 2.95, "load", (0.00,  0.42)),
    "S5": (8.90, 1.05, "gen",  (0.00, -0.46)),
    "S6": (9.75, 2.90, "sub",  (0.48, -0.10)),
}

# (from, to, kV, capacity MW)
BRANCHES = [
    ("N1", "N3", 500, 1200), ("N2", "N3", 500,  900),
    ("N3", "N4", 230,  600), ("N3", "N5", 230,  550),
    ("N4", "N5", 115,  250),
    ("E1", "E2", 500, 1400), ("E2", "E3", 230,  700),
    ("E2", "E4", 230,  650), ("E2", "E5", 500, 1100),
    ("E5", "E3", 115,  300), ("E4", "E5", 115,  280),
    ("E6", "E3", 115,  320), ("E6", "E5", 230,  620),
    ("S1", "S2", 230,  600), ("S2", "S3", 115,  300),
    ("S1", "S4", 230,  550), ("S3", "S4", 115,  260),
    ("S4", "S6", 230,  700), ("S5", "S6", 500, 1000),
    ("S3", "S5", 115,  320),
]

# inter-area corridors: (from, to, kV, capacity MW)
TIES = [
    ("N3", "E1", 500, 1500), ("N5", "E4", 230,  700),
    ("N4", "S1", 230,  650), ("E5", "S6", 500, 1200),
]

# Area names sit on a tab just *above* each card, which keeps them clear of
# every bus label inside.
AREAS = {
    "N": dict(box=(0.30, 4.95, 5.05, 3.40), face="#eef2f6", edge="#b9c6d2",
              name="AREA NORTH", tab_x=0.46),
    "E": dict(box=(6.45, 4.95, 5.65, 3.40), face="#f5f1ea", edge="#d5c8b4",
              name="AREA EAST", tab_x=6.61),
    # nudged inboard: both north-south corridors cross this card's corners
    "S": dict(box=(1.50, 0.35, 9.20, 3.20), face="#eef4ef", edge="#bacfbf",
              name="AREA SOUTH", tab_x=4.30),
}

MARK = {"gen": ("s", FUEL["gas"], 9.0), "load": ("o", "#f7f7f7", 7.6),
        "sub": ("^", FUEL["wind"], 9.2)}


def lw_for(cap):
    """Branch stroke width in points, linear in thermal rating."""
    return 0.75 + 2.35 * cap / 1500.0


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(7.6, 4.95))
    gs = fig.add_gridspec(1, 2, left=0.012, right=0.988, top=0.918,
                          bottom=0.030, wspace=0.02,
                          width_ratios=[1.0, 0.235])
    ax = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[0, 1])

    # ---- control-area cards ----------------------------------------------
    for spec in AREAS.values():
        x, y, w, h = spec["box"]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.26",
            facecolor=spec["face"], edgecolor=spec["edge"], lw=0.8,
            ls=(0, (5, 2.5)), zorder=0))
        ax.text(spec["tab_x"], y + h + 0.13, spec["name"], fontsize=7.0,
                color="#69747e", fontweight="bold", ha="left", va="bottom",
                zorder=1)

    # ---- branches ---------------------------------------------------------
    for a, b, kv, cap in BRANCHES:
        xa, ya = BUSES[a][:2]
        xb, yb = BUSES[b][:2]
        ax.plot([xa, xb], [ya, yb], color=KV[kv]["color"], lw=lw_for(cap),
                solid_capstyle="round", zorder=2)

    # inter-area corridors: white casing so they read as crossing a boundary
    for a, b, kv, cap in TIES:
        xa, ya = BUSES[a][:2]
        xb, yb = BUSES[b][:2]
        ax.plot([xa, xb], [ya, yb], color="white", lw=lw_for(cap) + 2.4,
                solid_capstyle="round", zorder=2.4)
        ax.plot([xa, xb], [ya, yb], color=KV[kv]["color"], lw=lw_for(cap),
                solid_capstyle="round", zorder=2.6)
        ax.text((xa + xb) / 2, (ya + yb) / 2, f"{cap:,} MW", fontsize=6.1,
                color=KV[kv]["color"], ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor=KV[kv]["color"], lw=0.5, alpha=0.95))

    # ---- buses ------------------------------------------------------------
    for bid, (x, y, kind, (dx, dy)) in BUSES.items():
        m, fc, ms = MARK[kind]
        ax.plot([x], [y], marker=m, markersize=ms, markerfacecolor=fc,
                markeredgecolor=INK, markeredgewidth=0.85, zorder=6,
                linestyle="none")
        ax.text(x + dx, y + dy, bid, fontsize=6.4, color="#40484f",
                ha="center", va="center", zorder=7)

    # a couple of anchoring annotations
    ax.annotate("Hydro plant\n1.2 GW", xy=(1.25, 7.60), xytext=(0.62, 6.82),
                fontsize=6.2, color="#69747e", ha="left", va="top",
                linespacing=1.35, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#a6b1ba"))
    ax.annotate("Metro load centre\n2.4 GW peak", xy=(7.05, 5.60),
                xytext=(6.35, 4.55), fontsize=6.2, color="#69747e",
                ha="center", va="top", linespacing=1.35, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#a6b1ba"))

    ax.set_xlim(-0.05, 12.35)
    ax.set_ylim(0.02, 8.98)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Three-area transmission network \u2014 17 buses, "
                 "24 branches, 4 inter-area corridors",
                 fontsize=9, fontweight="bold", loc="left", x=0.005, pad=5)

    # ---- legend rail ------------------------------------------------------
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")

    def rail_title(y, text):
        axl.text(0.0, y, text, fontsize=6.7, color="#69747e",
                 fontweight="bold", ha="left", va="center")

    node_h = [Line2D([], [], marker=MARK[k][0], color="none",
                     markerfacecolor=MARK[k][1], markeredgecolor=INK,
                     markeredgewidth=0.85, markersize=MARK[k][2] * 0.85,
                     label=lab)
              for k, lab in [("gen", "Generation"), ("load", "Load bus"),
                             ("sub", "Substation")]]
    rail_title(0.965, "BUS TYPE")
    l1 = axl.legend(handles=node_h, loc="upper left",
                    bbox_to_anchor=(-0.06, 0.935), frameon=False,
                    fontsize=6.8, handletextpad=0.5, labelspacing=0.5,
                    borderpad=0.1)
    axl.add_artist(l1)
    l1.set_clip_on(False)

    kv_h = [Line2D([], [], color=KV[k]["color"], lw=2.0, label=KV[k]["label"])
            for k in (500, 230, 115)]
    rail_title(0.700, "VOLTAGE CLASS")
    l2 = axl.legend(handles=kv_h, loc="upper left",
                    bbox_to_anchor=(-0.06, 0.670), frameon=False,
                    fontsize=6.8, handletextpad=0.5, labelspacing=0.5,
                    handlelength=1.9, borderpad=0.1)
    axl.add_artist(l2)
    l2.set_clip_on(False)

    rail_title(0.435, "THERMAL RATING")
    for i, cap in enumerate((250, 700, 1500)):
        y = 0.372 - i * 0.070
        axl.plot([0.02, 0.30], [y, y], color="#7f8c96", lw=lw_for(cap),
                 solid_capstyle="round", clip_on=False)
        axl.text(0.37, y, f"{cap:,} MW", fontsize=6.8, color="#1a1a1a",
                 ha="left", va="center")

    rail_title(0.150, "CORRIDOR")
    axl.add_patch(FancyBboxPatch(
        (-0.02, 0.045), 0.36, 0.075,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor="#eef2f6", edgecolor="none", zorder=1, clip_on=False))
    axl.plot([0.02, 0.30], [0.082, 0.082], color="white", lw=4.6,
             solid_capstyle="round", clip_on=False, zorder=2)
    axl.plot([0.02, 0.30], [0.082, 0.082], color=KV[500]["color"], lw=2.2,
             solid_capstyle="round", clip_on=False, zorder=3)
    axl.text(0.37, 0.082, "Inter-area tie", fontsize=6.8, color="#1a1a1a",
             ha="left", va="center")

    total = sum(c for *_, c in BRANCHES) + sum(c for *_, c in TIES)
    print(f"buses={len(BUSES)} branches={len(BRANCHES) + len(TIES)} "
          f"total branch rating={total:,} MW")
    savefig(fig, "193_grid_topology.png")


if __name__ == "__main__":
    main()
