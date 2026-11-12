"""Four-stage energy-flow (Sankey) diagram, primary supply to end use.

Ribbons are cubic-bezier polygons whose width is the annual energy in TWh.
Primary sources keep their fuel colour into the conversion node; downstream
the merged electricity trunk takes a single slate colour, because a mixed
stream has no single fuel identity.  Rejected energy peels away downward in
warm grey at each stage and is labelled with the stage efficiency, so the
narrowing of the useful stream can be read straight off the picture.

Sources are ordered by conversion efficiency (best at the top), which keeps
the useful ribbons roughly parallel and stops the diagram from braiding.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

from pubstyle import set_pub_style, savefig

FUEL = {
    "Nuclear": "#8478a6",
    "Coal":    "#5b5f64",
    "Hydro":   "#4a7f9e",
    "Wind":    "#4f9e91",
    "Solar":   "#dcb35c",
    "Gas":     "#cf8447",
}
TRUNK = "#6f93a8"
LOSS = "#bdb5aa"
SECTOR_C = ["#8fb0c2", "#79a0b6", "#638fa6", "#4e7d93"]
INK = "#333333"
GREY = "#7d858c"

# source -> (primary energy TWh, conversion efficiency), best first
SOURCES = [
    ("Wind",    95.0, 1.00),
    ("Solar",   62.0, 1.00),
    ("Hydro",   78.0, 0.90),
    ("Gas",    240.0, 0.52),
    ("Coal",   320.0, 0.38),
    ("Nuclear", 180.0, 0.33),
]
TD_EFF = 0.94
SECTORS = [("Industry", 205.0), ("Buildings", 168.0),
           ("Transport", 72.0), ("Other", 56.0)]

XCOL = [0.000, 0.345, 0.690, 1.000]
NODE_W = 0.019
TOP = 0.925
GAP = 0.010


def ribbon(ax, x0, y0t, y0b, x1, y1t, y1b, color, alpha=0.55, z=2):
    """Filled cubic-bezier band between two vertical slices."""
    xm = (x0 + x1) / 2.0
    verts = [(x0, y0t),
             (xm, y0t), (xm, y1t), (x1, y1t),
             (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b),
             (x0, y0t)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, lw=0,
                           alpha=alpha, zorder=z))


def node(ax, x, ytop, h, color, z=5):
    ax.add_patch(Rectangle((x, ytop - h), NODE_W, h, facecolor=color,
                           edgecolor="white", lw=0.5, zorder=z))


def main():
    set_pub_style(base_font=8)

    primary = sum(e for _, e, _ in SOURCES)
    useful = [e * f for _, e, f in SOURCES]
    losses = [e * (1 - f) for _, e, f in SOURCES]
    gen = sum(useful)
    conv_loss = sum(losses)
    delivered = gen * TD_EFF
    td_loss = gen - delivered

    scale = 0.815 / primary
    fig = plt.figure(figsize=(7.7, 4.55))
    ax = fig.add_axes([0.005, 0.015, 0.99, 0.885])
    ax.set_xlim(-0.315, 1.435)
    ax.set_ylim(0.020, 1.048)
    ax.axis("off")

    # ---- column 0: primary sources ---------------------------------------
    src_top, src_slice = TOP, {}
    for (name, e, _), u, l in zip(SOURCES, useful, losses):
        h = e * scale
        node(ax, XCOL[0], src_top, h, FUEL[name])
        src_slice[name] = (src_top, src_top - u * scale, src_top - h)
        ax.text(XCOL[0] - 0.014, src_top - h / 2, f"{name}   {e:.0f}",
                fontsize=7.0, color=INK, ha="right", va="center", zorder=6)
        src_top -= h + GAP

    # ---- column 1: generation + conversion losses ------------------------
    gen_h, cl_h = gen * scale, conv_loss * scale
    node(ax, XCOL[1], TOP, gen_h, TRUNK)
    cl_top = 0.060 + cl_h
    node(ax, XCOL[1], cl_top, cl_h, LOSS)

    gy, ly = TOP, cl_top
    for (name, _, _), u, l in zip(SOURCES, useful, losses):
        st, smid, sb = src_slice[name]
        if u > 0:
            ribbon(ax, XCOL[0] + NODE_W, st, smid, XCOL[1], gy,
                   gy - u * scale, FUEL[name], 0.58)
            gy -= u * scale
        if l > 0:
            ribbon(ax, XCOL[0] + NODE_W, smid, sb, XCOL[1], ly,
                   ly - l * scale, FUEL[name], 0.30)
            ly -= l * scale

    # ---- column 2: delivered + T&D losses --------------------------------
    d_h, td_h = delivered * scale, td_loss * scale
    node(ax, XCOL[2], TOP, d_h, TRUNK)
    td_top = TOP - d_h - 0.020
    node(ax, XCOL[2], td_top, td_h, LOSS)
    ribbon(ax, XCOL[1] + NODE_W, TOP, TOP - d_h, XCOL[2], TOP, TOP - d_h,
           TRUNK, 0.50)
    ribbon(ax, XCOL[1] + NODE_W, TOP - d_h, TOP - gen_h, XCOL[2], td_top,
           td_top - td_h, LOSS, 0.55)

    # ---- column 3: end-use sectors ---------------------------------------
    sy, ey = TOP, TOP
    for (name, e), col in zip(SECTORS, SECTOR_C):
        h = e * scale
        node(ax, XCOL[3], ey, h, col)
        ribbon(ax, XCOL[2] + NODE_W, sy, sy - h, XCOL[3], ey, ey - h, col,
               0.55)
        ax.text(XCOL[3] + NODE_W + 0.014, ey - h / 2,
                f"{name}   {e:.0f}", fontsize=7.0, color=INK, ha="left",
                va="center", zorder=6)
        sy -= h
        ey -= h + GAP

    # ---- stage headers ----------------------------------------------------
    # name over value on two lines: a single line is wider than the column
    # pitch and the four headers ran into each other
    # name / total / stage efficiency on three lines: a single line is wider
    # than the column pitch and the four headers ran into each other
    heads = [
        (XCOL[0], "PRIMARY SUPPLY", f"{primary:.0f} TWh", "", GREY),
        (XCOL[1], "GENERATION", f"{gen:.0f} TWh",
         f"{100 * gen / primary:.0f}% of supply", "#4e7186"),
        (XCOL[2], "DELIVERED", f"{delivered:.0f} TWh",
         f"{100 * TD_EFF:.0f}% of generation", "#4e7186"),
        (XCOL[3] + NODE_W, "END USE",
         f"{sum(e for _, e in SECTORS):.0f} TWh", "", GREY),
    ]
    for x, name, val, eff, col in heads:
        ax.text(x, TOP + 0.078, name, fontsize=7.0, color=col,
                fontweight="bold", ha="left", va="bottom")
        ax.text(x, TOP + 0.047, val, fontsize=6.8, color=GREY, ha="left",
                va="bottom")
        if eff:
            ax.text(x, TOP + 0.016, eff, fontsize=6.5, color=col,
                    ha="left", va="bottom")

    # ---- loss labels ------------------------------------------------------
    ax.text(XCOL[1], 0.044,
            f"Conversion losses   {conv_loss:.0f} TWh  "
            f"({100 * conv_loss / primary:.0f}% of primary supply)",
            fontsize=6.8, color="#8c8375", ha="left", va="top")
    ax.annotate(f"T&D losses   {td_loss:.0f} TWh  "
                f"({100 * (1 - TD_EFF):.0f}% of generation)",
                xy=(XCOL[2] + NODE_W, td_top - td_h / 2),
                xytext=(XCOL[2] + 0.075, td_top - 0.075), fontsize=6.8,
                color="#8c8375", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#b3aa9c"))

    ax.text(XCOL[3] + NODE_W + 0.014, 0.30,
            f"Overall system efficiency\n"
            f"{100 * delivered / primary:.0f}%   "
            f"({primary:.0f} \u2192 {delivered:.0f} TWh)",
            fontsize=6.8, color="#4e7186", ha="left", va="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f2f6f8",
                      edgecolor="#cfdde4", lw=0.7))


    print(f"primary={primary:.0f}  generated={gen:.0f} "
          f"({100 * gen / primary:.1f}%)  delivered={delivered:.0f} "
          f"({100 * delivered / primary:.1f}%)")
    print(f"conversion losses={conv_loss:.0f}  T&D losses={td_loss:.0f}  "
          f"end use={sum(e for _, e in SECTORS):.0f}")
    savefig(fig, "199_energy_sankey.png")


if __name__ == "__main__":
    main()
