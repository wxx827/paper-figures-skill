"""USDA soil texture triangle with a survey sample cloud laid over it.

The twelve USDA classes are drawn as true ternary polygons (sand at the
lower-left corner, silt at the lower-right, clay at the apex) and shaded by
their own mean clay content, so the fill is a monotone ramp rather than
twelve arbitrary hues: pale sands at the bottom left, dark clays at the top.
Survey samples are placed by their sand-silt-clay composition and coloured by
organic matter, and every sample is classified by a point-in-polygon test
against the same polygons that are drawn, so the count bars on the right
cannot disagree with the picture.

The lower right panel gives the hydraulic reason the triangle matters:
available water capacity is derived from the field capacity and wilting point
printed beside it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, savefig, get_cmap, darken
from figkit import halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
SQ3 = np.sqrt(3.0) / 2.0

# USDA class polygons as (sand %, clay %) vertices; silt is the remainder.
CLASSES = [
    ("Clay", [(0, 100), (0, 60), (20, 40), (45, 40), (45, 55)],
     "Clay", (0.0, 0.0), 6.8),
    ("Silty clay", [(0, 60), (0, 40), (20, 40)],
     "Silty\nclay", (0.0, 0.0), 6.2),
    ("Sandy clay", [(45, 35), (45, 55), (65, 35)],
     "Sandy\nclay", (0.0, 0.005), 6.2),
    ("Clay loam", [(20, 27), (20, 40), (45, 40), (45, 27)],
     "Clay loam", (0.0, 0.0), 6.5),
    ("Silty clay loam", [(0, 27), (0, 40), (20, 40), (20, 27)],
     "Silty clay\nloam", (0.0, 0.0), 6.2),
    ("Sandy clay loam", [(45, 27), (45, 35), (65, 35), (80, 20), (52, 20)],
     "Sandy clay\nloam", (0.0, 0.0), 6.2),
    ("Loam", [(23, 27), (45, 27), (52, 20), (52, 7), (43, 7)],
     "Loam", (0.0, 0.0), 6.8),
    ("Silt loam", [(0, 27), (23, 27), (50, 0), (20, 0), (8, 12), (0, 12)],
     "Silt loam", (0.02, 0.012), 6.8),
    ("Silt", [(0, 12), (8, 12), (20, 0), (0, 0)], "Silt", (0.0, 0.0), 6.5),
    ("Sandy loam",
     [(52, 20), (52, 7), (43, 7), (50, 0), (70, 0), (85, 15), (80, 20)],
     "Sandy loam", (-0.01, -0.012), 6.5),
    ("Loamy sand", [(70, 0), (85, 0), (90, 10), (85, 15)],
     "Loamy\nsand", (0.018, 0.016), 6.0),
    ("Sand", [(85, 0), (100, 0), (90, 10)], "Sand", (-0.012, 0.004), 6.2),
]

# representative hydraulic properties; AWC is derived, never typed in
RETENTION = [
    ("Sand", 10.0, 4.0, 210.0),
    ("Sandy loam", 21.0, 9.0, 44.0),
    ("Loam", 28.0, 12.0, 13.0),
    ("Clay loam", 34.0, 19.0, 4.3),
    ("Clay", 42.0, 27.0, 1.2),
]

# sampling clusters: (sand, silt, clay) centre and Dirichlet concentration
CLUSTERS = [((0.34, 0.42, 0.24), 70.0, 46), ((0.62, 0.26, 0.12), 55.0, 38),
            ((0.20, 0.44, 0.36), 60.0, 26), ((0.80, 0.14, 0.06), 90.0, 20)]


def tern(sand, clay):
    """(sand %, clay %) -> cartesian, sand corner lower-left, clay at apex."""
    sand = np.asarray(sand, dtype=float)
    clay = np.asarray(clay, dtype=float)
    return (100.0 - sand - 0.5 * clay) / 100.0, clay * SQ3 / 100.0


def draw_axes_frame(ax):
    """Three labelled ternary edges with outward ticks every 10 percent."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQ3]])
    ax.add_patch(Polygon(corners, closed=True, facecolor="none",
                         edgecolor=RULE, lw=1.0, zorder=6))
    for v in range(10, 100, 10):
        # sand: bottom edge, reading right to left
        x, y = tern(v, 0.0)
        ax.plot([x, x], [y, y - 0.014], color=RULE, lw=0.6, zorder=6)
        ax.text(x, y - 0.030, f"{v}", ha="center", va="top", fontsize=6.4,
                color=SUB)
        # clay: left edge (silt = 0)
        x, y = tern(100.0 - v, v)
        ax.plot([x, x - 0.013], [y, y + 0.008], color=RULE, lw=0.6, zorder=6)
        ax.text(x - 0.022, y + 0.012, f"{v}", ha="right", va="center",
                fontsize=6.4, color=SUB)
        # silt: right edge (sand = 0)
        x, y = tern(0.0, 100.0 - v)
        ax.plot([x, x + 0.013], [y, y + 0.008], color=RULE, lw=0.6, zorder=6)
        ax.text(x + 0.022, y + 0.012, f"{v}", ha="left", va="center",
                fontsize=6.4, color=SUB)

    ax.text(0.5, -0.088, "Sand (%)  " + "\u2190", ha="center", va="top",
            fontsize=8.0, color=INK)
    ax.text(0.175, 0.470, "\u2192  Clay (%)", ha="center", va="center",
            rotation=60, fontsize=8.0, color=INK)
    ax.text(0.828, 0.470, "Silt (%)  " + "\u2190", ha="center", va="center",
            rotation=-60, fontsize=8.0, color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(601)

    # ---- samples -----------------------------------------------------------
    comps, om = [], []
    for centre, conc, n in CLUSTERS:
        comps.append(rng.dirichlet(np.array(centre) * conc, size=n) * 100.0)
        om.append(rng.gamma(3.2, 0.62, size=n) + 0.35)
    comps = np.vstack(comps)
    om = np.concatenate(om)
    sand, silt, clay = comps[:, 0], comps[:, 1], comps[:, 2]

    paths = {name: Path(np.array(v, dtype=float))
             for name, v, _, _, _ in CLASSES}
    pts = np.column_stack([sand, clay])
    assign = np.full(sand.size, "", dtype=object)
    for name, p in paths.items():
        inside = p.contains_points(pts, radius=0.5)
        assign[inside & (assign == "")] = name
    counts = {name: int(np.sum(assign == name)) for name, *_ in CLASSES}

    # ---- class fills: one monotone ramp keyed to mean clay -----------------
    cmap_fill = get_cmap("cb_sequential_warm")
    mean_clay = {name: float(np.mean([v[1] for v in verts]))
                 for name, verts, _, _, _ in CLASSES}
    cmax = max(mean_clay.values())
    fill = {name: cmap_fill(0.10 + 0.62 * mean_clay[name] / cmax)
            for name in mean_clay}

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.72],
                          left=0.045, right=0.975, top=0.860, bottom=0.128,
                          wspace=0.16, hspace=0.42)
    ax_t = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    # ---- triangle ----------------------------------------------------------
    for name, verts, label, (ox, oy), fs in CLASSES:
        v = np.array(verts, dtype=float)
        xy = np.column_stack(tern(v[:, 0], v[:, 1]))
        ax_t.add_patch(Polygon(xy, closed=True, facecolor=fill[name],
                               edgecolor="white", lw=0.9, zorder=2))
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        # halo + high zorder: the class names must stay readable where the
        # sample cloud is densest, which is exactly over the loam fields
        halo_text(ax_t, cx + ox, cy + oy, label, ha="center", va="center",
                  fontsize=fs, color=darken(fill[name], 0.66), halo="white",
                  halo_lw=1.8, linespacing=1.15, zorder=9)

    draw_axes_frame(ax_t)
    sx, sy = tern(sand, clay)
    sc = ax_t.scatter(sx, sy, c=om, cmap=get_cmap("cb_sequential_green"),
                      s=17, linewidths=0.35, edgecolors="white", zorder=7,
                      vmin=float(om.min()), vmax=float(om.max()))
    ax_t.set_xlim(-0.055, 1.055)
    ax_t.set_ylim(-0.135, 0.925)
    ax_t.set_aspect("equal")
    ax_t.axis("off")
    ax_t.set_title("USDA texture classes with survey samples", fontsize=9.0,
                   fontweight="bold", pad=6)
    # the upper-left corner outside the triangle is dead space on any ternary
    # plot; parking the colorbar there keeps the figure footprint compact
    cax = ax_t.inset_axes([0.035, 0.800, 0.255, 0.028])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2, color=RULE,
                      pad=1.6)
    cb.set_label("soil organic matter (%)", fontsize=6.6, color=INK,
                 labelpad=2.0)

    # ---- sample counts per class ------------------------------------------
    order = sorted([c for c in CLASSES if counts[c[0]] > 0],
                   key=lambda c: counts[c[0]])
    yy = np.arange(len(order))
    vals = [counts[c[0]] for c in order]
    ax_b.barh(yy, vals, height=0.68,
              color=[fill[c[0]] for c in order],
              edgecolor=[darken(fill[c[0]], 0.35) for c in order], lw=0.6,
              zorder=3)
    for y, c, v in zip(yy, order, vals):
        ax_b.text(v + 0.7, y, f"{c[0]}  ({v})", ha="left", va="center",
                  fontsize=6.4, color=INK, zorder=4)
    ax_b.set_yticks([])
    ax_b.set_ylim(-0.7, len(order) - 0.3)
    ax_b.set_xlim(0, max(vals) * 2.30)
    ax_b.set_xlabel("samples in class (n)", fontsize=7.4)
    ax_b.tick_params(axis="x", labelsize=6.5)
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x", lw=0.5, color="#e4e4e4", zorder=0)
    ax_b.set_title("Sample count by texture class", fontsize=8.5,
                   fontweight="bold")

    # ---- water retention table --------------------------------------------
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_ylim(0.0, 1.0)
    ax_r.axis("off")
    cols = [0.005, 0.455, 0.625, 0.800, 0.995]
    heads = ["Texture class", "FC\n(% vol)", "WP\n(% vol)",
             "AWC\n(mm m$^{-1}$)", "K$_{sat}$\n(mm h$^{-1}$)"]
    for x, h, ha in zip(cols, heads, ["left"] + ["right"] * 4):
        ax_r.text(x, 0.935, h, ha=ha, va="top", fontsize=6.5, color=INK,
                  fontweight="bold", linespacing=1.25)
    ax_r.plot([0.0, 1.0], [0.755, 0.755], color=RULE, lw=0.8)
    for i, (name, fc_v, wp_v, ksat) in enumerate(RETENTION):
        y = 0.680 - i * 0.140
        if i % 2 == 0:
            ax_r.add_patch(plt.Rectangle((0.0, y - 0.052), 1.0, 0.108,
                                         facecolor="#f5f5f5",
                                         edgecolor="none", zorder=0))
        awc = (fc_v - wp_v) * 10.0
        ax_r.text(cols[0], y, name, ha="left", va="center", fontsize=6.6,
                  color=INK)
        for x, txt in zip(cols[1:], [f"{fc_v:.0f}", f"{wp_v:.0f}",
                                     f"{awc:.0f}", f"{ksat:g}"]):
            ax_r.text(x, y, txt, ha="right", va="center", fontsize=6.6,
                      color=SUB)
    ax_r.plot([0.0, 1.0], [0.008, 0.008], color="#cccccc", lw=0.6)
    ax_r.set_title("Water retention by texture class", fontsize=8.5,
                   fontweight="bold")

    panel_labels_inches(fig, [ax_t, ax_b, ax_r], dx=-0.26, dy=0.14)

    top_name, top_n = max(counts.items(), key=lambda kv: kv[1])
    loamy = sum(counts[k] for k in counts if "loam" in k.lower())

    print(f"n={sand.size}  classes hit={sum(v > 0 for v in counts.values())}")
    print("counts:", {k: v for k, v in counts.items() if v})
    print(f"mean clay={clay.mean():.2f}%  mean OM={om.mean():.2f}%")
    savefig(fig, "601_soil_texture_triangle.png")


if __name__ == "__main__":
    main()
