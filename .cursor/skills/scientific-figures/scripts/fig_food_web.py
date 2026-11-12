"""Trophic food-web network with a subordinate biomass-pyramid panel.

Nodes are laid out on discrete trophic levels (producers at the base, top
predators at the crown); marker area encodes standing biomass and hue encodes
functional group.  Curved arrows run prey -> predator with width proportional
to energy flux.  The right column carries a compact key and a trophic pyramid
scaled to ~35 % of the web panel so the hierarchy stays obvious.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig, style_axes


GROUPS = {
    "Primary producer": "#5aa17f",
    "Detrital pool": "#9a8f7d",
    "Herbivore": "#c08f3e",
    "Mesopredator": "#3d8b93",
    "Top predator": "#a2593c",
}
INK = "#1a1a1a"

#            name            x     TL  biomass  group
NODES = [
    ("Phytoplankton", 0.17, 1, 46.0, "Primary producer"),
    ("Macroalgae",    0.44, 1, 30.0, "Primary producer"),
    ("Seagrass",      0.70, 1, 24.0, "Primary producer"),
    ("Detritus",      0.93, 1, 38.0, "Detrital pool"),
    ("Zooplankton",   0.13, 2, 14.0, "Herbivore"),
    ("Amphipods",     0.39, 2, 9.0, "Herbivore"),
    ("Snails",        0.64, 2, 11.0, "Herbivore"),
    ("Polychaetes",   0.89, 2, 7.0, "Herbivore"),
    ("Shrimp",        0.22, 3, 4.5, "Mesopredator"),
    ("Small fish",    0.51, 3, 6.5, "Mesopredator"),
    ("Crabs",         0.80, 3, 3.8, "Mesopredator"),
    ("Piscivores",    0.37, 4, 1.6, "Top predator"),
    ("Seabirds",      0.68, 4, 0.9, "Top predator"),
]

EDGES = [
    ("Phytoplankton", "Zooplankton", 1.00),
    ("Phytoplankton", "Amphipods", 0.35),
    ("Macroalgae", "Snails", 0.72),
    ("Macroalgae", "Amphipods", 0.50),
    ("Seagrass", "Snails", 0.55),
    ("Seagrass", "Polychaetes", 0.30),
    ("Detritus", "Polychaetes", 0.85),
    ("Detritus", "Amphipods", 0.45),
    ("Zooplankton", "Small fish", 0.90),
    ("Zooplankton", "Shrimp", 0.42),
    ("Amphipods", "Small fish", 0.58),
    ("Amphipods", "Shrimp", 0.52),
    ("Snails", "Crabs", 0.65),
    ("Polychaetes", "Crabs", 0.48),
    ("Polychaetes", "Shrimp", 0.30),
    ("Small fish", "Piscivores", 1.00),
    ("Small fish", "Seabirds", 0.70),
    ("Shrimp", "Piscivores", 0.45),
    ("Crabs", "Seabirds", 0.50),
    ("Crabs", "Piscivores", 0.28),
]

TL_NAMES = {1: "Producers / detritus", 2: "Primary consumers",
            3: "Secondary consumers", 4: "Apex predators"}


def panel_tag(fig, ax, label, dx=-0.052, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def marker_area(biomass):
    return 34.0 + 13.5 * biomass


def main():
    set_pub_style(base_font=8)

    pos = {n[0]: (n[1], float(n[2])) for n in NODES}
    grp = {n[0]: n[4] for n in NODES}
    bio = {n[0]: n[3] for n in NODES}

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.34],
                  height_ratios=[1.0, 0.80], hspace=0.40, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_pyr = fig.add_subplot(gs[1, 1])

    # ---- trophic bands ---------------------------------------------------
    for tl in (1, 2, 3, 4):
        ax.axhspan(tl - 0.34, tl + 0.34,
                   color="#f5f5f1" if tl % 2 else "#fafaf7", lw=0, zorder=0)
        ax.text(-0.075, tl, f"TL {tl}", fontsize=6.8, color="#6f6f68",
                ha="center", va="center", fontweight="bold", zorder=5)
        ax.text(1.075, tl, TL_NAMES[tl], fontsize=6.4, color="#8d8d85",
                ha="center", va="center", rotation=90, zorder=5)

    # ---- energy-flux arrows ----------------------------------------------
    for src, dst, flux in EDGES:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        r0 = np.sqrt(marker_area(bio[src])) / 2.0 + 1.2
        r1 = np.sqrt(marker_area(bio[dst])) / 2.0 + 2.4
        rad = 0.13 if x1 >= x0 else -0.13
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=6.0 + 2.6 * flux,
            linewidth=0.45 + 1.85 * flux, color=GROUPS[grp[src]],
            alpha=0.42, shrinkA=r0, shrinkB=r1, zorder=2,
            joinstyle="round", capstyle="round")
        ax.add_patch(arrow)

    # ---- nodes -----------------------------------------------------------
    for name, x, tl, b, g in NODES:
        ax.scatter([x], [tl], s=marker_area(b), color=GROUPS[g],
                   edgecolor="white", linewidth=0.9, zorder=4)

    # ---- labels: below the node, white halo so arrows never fight text ---
    for name, x, tl, b, g in NODES:
        dy = np.sqrt(marker_area(b)) / 2.0 + 5.5   # points
        ax.annotate(name, xy=(x, tl), xytext=(0, -dy),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6.6, color="#33332f", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.4,
                                                foreground="white")])

    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(0.45, 4.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Coastal food web: 13 compartments, 20 trophic links",
                 fontsize=8.5, loc="left", pad=6)

    # ---- key panel -------------------------------------------------------
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")
    ax_key.text(0.0, 0.99, "Functional group", fontsize=7.2,
                fontweight="bold", va="top", color=INK)
    for i, (label, col) in enumerate(GROUPS.items()):
        y = 0.865 - 0.088 * i
        ax_key.scatter([0.055], [y], s=46, color=col, edgecolor="white",
                       linewidth=0.8, clip_on=False)
        ax_key.text(0.145, y, label, fontsize=6.8, va="center",
                    color="#33332f")

    ax_key.text(0.0, 0.375, "Standing biomass", fontsize=7.2,
                fontweight="bold", va="top", color=INK)
    for i, val in enumerate((10, 25, 45)):
        x = 0.13 + 0.30 * i
        ax_key.scatter([x], [0.215], s=marker_area(val), color="#c9c9c2",
                       edgecolor="white", linewidth=0.8, clip_on=False)
        ax_key.text(x, 0.075, f"{val}", fontsize=6.4, ha="center",
                    va="center", color="#55554f")
    ax_key.text(0.99, 0.075, "g m$^{-2}$", fontsize=6.4, ha="right",
                va="center", color="#55554f")

    # ---- subordinate trophic pyramid -------------------------------------
    tl_biomass = {1: 138.0, 2: 41.0, 3: 14.8, 4: 2.5}
    tl_col = {1: "#5aa17f", 2: "#c08f3e", 3: "#3d8b93", 4: "#a2593c"}
    ax_pyr.set_facecolor("#f6f6f2")
    span = max(tl_biomass.values())
    for tl in (1, 2, 3, 4):
        w = np.sqrt(tl_biomass[tl] / span)          # width ~ sqrt(biomass)
        ax_pyr.barh(tl, w, height=0.62, left=-w / 2.0, color=tl_col[tl],
                    edgecolor="white", linewidth=0.7, zorder=3)
        if w >= 0.42:
            ax_pyr.text(0.0, tl, f"{tl_biomass[tl]:.1f}", ha="center",
                        va="center", fontsize=6.3, color="white",
                        fontweight="bold", zorder=5)
        else:
            ax_pyr.text(w / 2.0 + 0.035, tl, f"{tl_biomass[tl]:.1f}",
                        ha="left", va="center", fontsize=6.3,
                        color=tl_col[tl], fontweight="bold", zorder=5)
        ax_pyr.text(-0.66, tl, f"TL {tl}", ha="left", va="center",
                    fontsize=6.4, color="#6f6f68")
    ax_pyr.set_xlim(-0.68, 0.68)
    ax_pyr.set_ylim(0.45, 4.55)
    ax_pyr.set_xticks([])
    ax_pyr.set_yticks([])
    for side in ("top", "right", "left"):
        ax_pyr.spines[side].set_visible(False)
    ax_pyr.spines["bottom"].set_visible(False)
    ax_pyr.set_title("Biomass pyramid\nwidth $\\propto\\sqrt{g\\,m^{-2}}$",
                     fontsize=7.2, loc="left", pad=4, linespacing=1.35)
    ax_pyr.text(0.5, -0.045, "transfer efficiency $\\approx$ 11 %",
                transform=ax_pyr.transAxes, ha="center", va="top",
                fontsize=6.3, color="#77776f", style="italic")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.055)

    panel_tag(fig, ax, "(a)", dx=-0.070)
    panel_tag(fig, ax_pyr, "(b)", dx=-0.070, dy=0.042)

    savefig(fig, "224_food_web.png")


if __name__ == "__main__":
    main()
