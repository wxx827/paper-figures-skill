"""Milestone timeline of 3D scene representations, voxels to 4D Gaussians.

A single horizontal arrow carries six milestone cards hung alternately above
and below it.  Each card is the house two-tone block -- a saturated header
strip naming the representation over a white body holding one sentence of
characterisation -- and the three representation families (explicit geometry,
neural field, splatting primitive) each own a hue, so the drift from discrete
grids to continuous fields to point primitives reads as a colour gradient
along the axis rather than as six unrelated boxes.

Everything is drawn on an inch canvas: corner radii, card gaps and stem
lengths are in inches on paper, which is what keeps six hand-placed cards
looking like one grid.
"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from pubstyle import COLORS, darken, savefig, set_pub_style
from figkit import inch_canvas

FW, FH = 10.4, 6.4

# The house blue ladder, darkening with time: the ramp itself is the story,
# so the six milestones take six consecutive rungs rather than six hues.
LADDER = ["#a9c4de", "#8fb0d6", "#6f96c4", "#5b83b5", "#43689b", "#2f4f7f"]
BODY_FILL = "#eef3f9"

# Families are read off the ladder in pairs; the key uses each pair's midpoint.
FAMILY = [("#9dbadb", "Explicit geometry"),
          ("#658dbd", "Neural field"),
          ("#39598b", "Point primitive")]

# year, short name, three-line characterisation
MILESTONES = [
    (2013, "Voxel Grid",
     "Regular occupancy\ncells; memory cost\ngrows as $O(n^3)$"),
    (2016, "Triangle Mesh",
     "Explicit surface;\nfast to raster, hard\nto optimise topology"),
    (2019, "Implicit SDF",
     "Continuous level set;\nwatertight surface at\nany query resolution"),
    (2020, "NeRF",
     "Volume rendering of\na radiance field;\nhours of fitting"),
    (2023, "3D Gaussians",
     "Anisotropic splats\nrasterised at real-\ntime frame rates"),
    (2025, "4D Gaussians",
     "Time-conditioned\nsplats; deformable\ndynamic scenes"),
]

AXIS_Y = 3.13          # centre line of the timeline band
BAND_H = 0.30
CARD_W, CARD_H = 1.60, 1.34
HEAD_H = 0.36
STEM = 0.50
ROUND = 0.075


def ink_on(color):
    """Legible text colour for a header strip of the given blue."""
    r, g, b = mpl.colors.to_rgb(color)
    return "#1f2c3a" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.56 else "white"


def two_tone_card(ax, cx, cy, w, h, head, body, color, head_h=HEAD_H,
                  head_fs=8.8, body_fs=6.8, z=4):
    """Deep header strip over a pale-blue body, hairline outline, soft drop."""
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    box = f"round,pad=0,rounding_size={ROUND}"
    ax.add_patch(FancyBboxPatch((x0 + 0.035, y0 - 0.042), w, h, boxstyle=box,
                                facecolor="#d2dceb", edgecolor="none",
                                mutation_aspect=1.0, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box,
                                facecolor=BODY_FILL, edgecolor="none",
                                mutation_aspect=1.0, zorder=z))
    hy = y0 + h - head_h
    ax.add_patch(FancyBboxPatch((x0, hy), w, head_h, boxstyle=box,
                                facecolor=color, edgecolor="none",
                                mutation_aspect=1.0, zorder=z + 1))
    ax.add_patch(Rectangle((x0, hy), w, ROUND, facecolor=color,
                           edgecolor="none", zorder=z + 2))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box, facecolor="none",
                                edgecolor=color, linewidth=0.9,
                                mutation_aspect=1.0, zorder=z + 3))
    ax.text(cx, hy + head_h / 2.0, head, ha="center", va="center",
            color=ink_on(color), fontsize=head_fs, fontweight="bold",
            zorder=z + 4)
    ax.text(cx, y0 + (h - head_h) / 2.0, body, ha="center", va="center",
            color="#3a4756", fontsize=body_fs, linespacing=1.62, zorder=z + 4)


def main():
    set_pub_style()
    fig, ax = inch_canvas(FW, FH)

    band_x0, band_x1 = 0.30, 9.72
    tip = 10.16
    half = BAND_H / 2.0

    # ---- the timeline itself: one arrow, painted with the same ladder -------
    shaft = [(band_x0, AXIS_Y + half), (band_x1, AXIS_Y + half),
             (band_x1, AXIS_Y + half * 1.75), (tip, AXIS_Y),
             (band_x1, AXIS_Y - half * 1.75), (band_x1, AXIS_Y - half),
             (band_x0, AXIS_Y - half)]
    clip = Polygon(shaft, closed=True, facecolor="none", edgecolor="none",
                   zorder=2)
    ax.add_patch(clip)
    ramp = LinearSegmentedColormap.from_list("band", [LADDER[1], LADDER[5]])
    im = ax.imshow(np.linspace(0.0, 1.0, 512).reshape(1, -1),
                   extent=[band_x0, tip, AXIS_Y - half * 1.75,
                           AXIS_Y + half * 1.75], cmap=ramp,
                   interpolation="bilinear", zorder=2)
    im.set_clip_path(clip)
    ax.text(band_x0 + 0.16, AXIS_Y, "time", ha="left", va="center",
            color="#1f2c3a", fontsize=7.0, style="italic", zorder=4)

    # ---- milestone cards, alternating above / below -------------------------
    xs = [1.22 + 1.556 * i for i in range(len(MILESTONES))]
    for i, ((yr, name, body), cx) in enumerate(zip(MILESTONES, xs)):
        col = LADDER[i]
        up = i % 2 == 0
        if up:
            y_stem0, y_stem1 = AXIS_Y + half, AXIS_Y + half + STEM
            cy = y_stem1 + CARD_H / 2.0
            yr_y, yr_va = AXIS_Y - half - 0.10, "top"
        else:
            y_stem0, y_stem1 = AXIS_Y - half, AXIS_Y - half - STEM
            cy = y_stem1 - CARD_H / 2.0
            yr_y, yr_va = AXIS_Y + half + 0.10, "bottom"

        ax.plot([cx, cx], [y_stem0, y_stem1], color=col, lw=1.5, zorder=3,
                solid_capstyle="butt")
        ax.add_patch(Circle((cx, AXIS_Y), 0.078, facecolor="white",
                            edgecolor=col, linewidth=1.3, zorder=5))
        ax.text(cx, yr_y, str(yr), ha="center", va=yr_va, fontsize=8.2,
                fontweight="bold", color=darken(col, 0.34), zorder=5)
        two_tone_card(ax, cx, cy, CARD_W, CARD_H, name, body, col)

    # ---- titles and family key ---------------------------------------------
    ax.text(0.30, 6.02, "Three decades of scene representation",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=COLORS["ink"])
    ax.text(0.30, 5.72, "Explicit grids gave way to continuous neural fields, "
            "and those in turn to rasterisable point primitives",
            ha="left", va="center", fontsize=8.0, color=COLORS["sub"])

    handles = [Line2D([], [], marker="s", linestyle="none", markersize=7.0,
                      markerfacecolor=c, markeredgecolor="white",
                      markeredgewidth=0.6, label=lab)
               for c, lab in FAMILY]
    leg = ax.legend(handles=handles, loc="upper right",
                    bbox_to_anchor=(1.0 - 0.30 / FW, 6.16 / FH), ncol=1,
                    fontsize=7.8, handletextpad=0.55, labelspacing=0.55,
                    borderaxespad=0.0)
    leg.set_zorder(9)
    ax.text(0.30, 0.40, "The blue deepens with time, so the ramp alone reads "
            "left to right; years are indicative rather than exact",
            ha="left", va="center", fontsize=6.8, color=COLORS["muted"])

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1081_wm_concept_repr_timeline.png")


if __name__ == "__main__":
    main()
