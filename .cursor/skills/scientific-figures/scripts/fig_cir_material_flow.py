"""Material flow analysis of an engineering metal, drawn as a closed loop.

A linear Sankey answers "where does the material go"; a circular-economy
Sankey has to answer "how much of it comes back", which a left-to-right
diagram structurally cannot show.  So the recycled fraction leaves the
end-of-life column as an ordinary ribbon and is then carried by a single wide
green band that runs beneath the whole diagram and re-enters production from
below - the only flow on the figure that travels right to left, which is
exactly the point.

Ribbon width is the annual flow in Mt/a; nodes also carry the standing stock
where one exists.  All rates printed on the figure are derived from the flow
table at the top of the file, so the widths and the percentages cannot drift
apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

from pubstyle import set_pub_style, savefig, darken

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
MAT = "#a1907c"          # primary material, warm neutral
MAT_L = "#c4b7a6"        # material immobilised in the in-use stock
LOSS = "#b3b3b3"         # tailings, dissipative and reprocessing losses
LANDFILL = "#9c948a"
BURN = "#c0603f"
RECY = "#2e7d5b"         # the one saturated colour on the figure

# ---- flow table (Mt per year) ---------------------------------------------
EXTRACTED = 100.0
TAILINGS = 18.0
STOCK_ADD = 26.0
LANDFILLED = 34.0
INCINERATED = 12.0
DISSIPATED = 10.0
RECYCLED = 34.0
# standing stocks (Mt), reported at the node rather than on the ribbon
STOCKS = {"Primary extraction": ("economic reserves", 4300.0),
          "Use phase": ("in-use stock", 1240.0),
          "Landfill": ("landfilled stock", 860.0)}

XCOL = [0.000, 0.240, 0.480, 0.720, 0.960]
NODE_W = 0.014
TOP = 0.900
GAP = 0.028
LOSS_GAP = 0.070


def ribbon(ax, x0, y0t, y0b, x1, y1t, y1b, color, alpha=0.52, z=2):
    """Filled cubic-bezier band between two vertical slices."""
    xm = 0.5 * (x0 + x1)
    verts = [(x0, y0t), (xm, y0t), (xm, y1t), (x1, y1t), (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0t)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, lw=0,
                           alpha=alpha, zorder=z))


def node(ax, x, ytop, h, color, z=5):
    ax.add_patch(Rectangle((x, ytop - h), NODE_W, h, facecolor=color,
                           edgecolor="white", lw=0.5, zorder=z))


def stack(entries, scale, top):
    """Lay a column of nodes out top-down; returns {name: (top, bottom)}."""
    y, out = top, {}
    for name, val, extra in entries:
        y -= extra
        out[name] = (y, y - val * scale)
        y -= val * scale + GAP
    return out


def main():
    set_pub_style(base_font=9)

    primary = EXTRACTED - TAILINGS
    prod_in = primary + RECYCLED
    to_eol = prod_in - STOCK_ADD
    circularity = RECYCLED / prod_in
    eol_rate = RECYCLED / to_eol
    landfill_rate = LANDFILLED / to_eol
    balance = TAILINGS + STOCK_ADD + LANDFILLED + INCINERATED + DISSIPATED

    scale = 0.480 / prod_in
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.010, 0.078, 0.982, 0.792])
    ax.set_xlim(-0.175, 1.330)
    ax.set_ylim(-0.150, 1.000)
    ax.axis("off")

    # ---- node columns ------------------------------------------------------
    c0 = stack([("Primary extraction", EXTRACTED, 0.0)], scale, TOP)
    c1 = stack([("Tailings", TAILINGS, 0.0),
                ("Production", prod_in, LOSS_GAP)], scale, TOP)
    ptop = c1["Production"][0]
    c2 = stack([("Use phase", prod_in, 0.0)], scale, ptop)
    c3 = stack([("Net stock addition", STOCK_ADD, 0.0),
                ("End of life", to_eol, LOSS_GAP)], scale, ptop)
    etop = c3["End of life"][0]
    c4 = stack([("Landfill", LANDFILLED, 0.0),
                ("Incineration", INCINERATED, 0.0),
                ("Dissipative loss", DISSIPATED, 0.0),
                ("Recycling", RECYCLED, LOSS_GAP)], scale, etop)

    COL = {"Primary extraction": MAT, "Tailings": LOSS, "Production": MAT,
           "Use phase": MAT, "Net stock addition": MAT_L,
           "End of life": MAT, "Landfill": LANDFILL,
           "Incineration": BURN, "Dissipative loss": LOSS,
           "Recycling": RECY}
    FLOW = {"Primary extraction": EXTRACTED, "Tailings": TAILINGS,
            "Production": prod_in, "Use phase": prod_in,
            "Net stock addition": STOCK_ADD, "End of life": to_eol,
            "Landfill": LANDFILLED, "Incineration": INCINERATED,
            "Dissipative loss": DISSIPATED, "Recycling": RECYCLED}

    for col, x in ((c0, XCOL[0]), (c1, XCOL[1]), (c2, XCOL[2]),
                   (c3, XCOL[3]), (c4, XCOL[4])):
        for name, (t, b) in col.items():
            node(ax, x, t, t - b, COL[name])

    # ---- ribbons -----------------------------------------------------------
    y = c0["Primary extraction"][0]
    for name, v in (("Tailings", TAILINGS), ("Production", primary)):
        h = v * scale
        ribbon(ax, XCOL[0] + NODE_W, y, y - h, XCOL[1], c1[name][0],
               c1[name][0] - h, COL[name],
               alpha=0.62 if name == "Tailings" else 0.52)
        y -= h

    ribbon(ax, XCOL[1] + NODE_W, *c1["Production"], XCOL[2], *c2["Use phase"],
           MAT)

    y = c2["Use phase"][0]
    for name, v in (("Net stock addition", STOCK_ADD), ("End of life", to_eol)):
        h = v * scale
        ribbon(ax, XCOL[2] + NODE_W, y, y - h, XCOL[3], c3[name][0],
               c3[name][0] - h, COL[name],
               alpha=0.60 if name == "Net stock addition" else 0.52)
        y -= h

    y = c3["End of life"][0]
    for name in ("Landfill", "Incineration", "Dissipative loss", "Recycling"):
        h = FLOW[name] * scale
        ribbon(ax, XCOL[3] + NODE_W, y, y - h, XCOL[4], c4[name][0],
               c4[name][0] - h, COL[name],
               alpha=0.62 if name == "Dissipative loss" else 0.55)
        y -= h

    # ---- the return loop ---------------------------------------------------
    # A constant-width band is easiest to draw as a very thick rounded
    # polyline, so the width has to be converted from data units to points
    # once the axes geometry is final.
    band_h = RECYCLED * scale
    fw, fh = fig.get_size_inches()
    ax_h_in = ax.get_position().height * fh
    y_span = float(np.diff(ax.get_ylim())[0])
    lw_pt = band_h * ax_h_in * 72.0 / y_span

    x_rec = XCOL[4] + NODE_W / 2.0
    x_prod = XCOL[1] + NODE_W / 2.0
    y_rail = c4["Recycling"][1] - 0.035 - band_h / 2.0
    y_prod_b = c1["Production"][1]
    ax.plot([x_rec, x_rec, x_prod, x_prod],
            [c4["Recycling"][1], y_rail, y_rail, y_prod_b],
            lw=lw_pt, color=RECY, alpha=0.55, solid_capstyle="butt",
            solid_joinstyle="round", zorder=3)
    for xc in (0.88, 0.46, 0.33):
        ax.plot([xc], [y_rail], marker="<", ms=6.5, mfc="white", mec="none",
                alpha=0.95, zorder=4)
    ax.plot([x_prod], [y_prod_b - 0.055], marker="^", ms=6.5, mfc="white",
            mec="none", alpha=0.95, zorder=4)
    ax.text(0.615, y_rail, f"secondary material  {RECYCLED:.0f} Mt/a", ha="center",
            va="center", fontsize=7.4, fontweight="bold", color="white",
            zorder=5)

    # ---- labels ------------------------------------------------------------
    t, b = c0["Primary extraction"]
    ax.text(XCOL[0] - 0.016, 0.5 * (t + b) + 0.030, "Primary extraction",
            fontsize=7.6, fontweight="bold", color=INK, ha="right",
            va="center", zorder=7)
    ax.text(XCOL[0] - 0.016, 0.5 * (t + b) + 0.002,
            f"{EXTRACTED:.0f} Mt/a", fontsize=7.0, color=SUB, ha="right",
            va="center", zorder=7)
    ax.text(XCOL[0] - 0.016, 0.5 * (t + b) - 0.028,
            f"{STOCKS['Primary extraction'][0]}\n"
            f"{STOCKS['Primary extraction'][1]:,.0f} Mt", fontsize=6.4,
            color=MUTED, ha="right", va="top", linespacing=1.4, zorder=7)

    subline = {"Use phase": f"{STOCKS['Use phase'][0]} "
                            f"{STOCKS['Use phase'][1]:,.0f} Mt",
               "Net stock addition": f"{100.0 * STOCK_ADD / prod_in:.0f}% of "
                                     f"production",
               "End of life": "discarded products entering waste management"}
    for col, x, keys in ((c1, XCOL[1], ("Production",)),
                         (c2, XCOL[2], ("Use phase",)),
                         (c3, XCOL[3], ("Net stock addition", "End of life"))):
        for name in keys:
            nt = col[name][0]
            sub = subline.get(name)
            ax.text(x, nt + (0.040 if sub else 0.014),
                    f"{name}   {FLOW[name]:.0f} Mt/a", fontsize=7.0,
                    fontweight="bold", color=darken(COL[name], 0.34),
                    ha="left", va="bottom", zorder=7)
            if sub:
                ax.text(x, nt + 0.014, sub, fontsize=6.4, color="#6d6459",
                        ha="left", va="bottom", zorder=7)

    tt, tb = c1["Tailings"]
    ax.text(XCOL[1] + NODE_W + 0.014, 0.5 * (tt + tb) + 0.013,
            f"Tailings   {TAILINGS:.0f} Mt/a", fontsize=7.0,
            fontweight="bold", color=darken(LOSS, 0.42), ha="left",
            va="center", zorder=7)
    ax.text(XCOL[1] + NODE_W + 0.014, 0.5 * (tt + tb) - 0.013,
            f"{100.0 * TAILINGS / EXTRACTED:.0f}% of ore never leaves the "
            f"mine site", fontsize=6.4, color=MUTED, ha="left", va="center",
            zorder=7)

    for name in ("Landfill", "Incineration", "Dissipative loss", "Recycling"):
        nt, nb = c4[name]
        share = FLOW[name] / to_eol
        note = f"{100.0 * share:.0f}% of end-of-life flow"
        if name == "Landfill":
            note += (f", {STOCKS['Landfill'][0]} "
                     f"{STOCKS['Landfill'][1]:,.0f} Mt")
        ax.text(XCOL[4] + NODE_W + 0.014, 0.5 * (nt + nb) + 0.014,
                f"{name}   {FLOW[name]:.0f} Mt/a", fontsize=7.0,
                fontweight="bold", color=darken(COL[name], 0.30), ha="left",
                va="center", zorder=7)
        ax.text(XCOL[4] + NODE_W + 0.014, 0.5 * (nt + nb) - 0.014, note,
                fontsize=6.4, color=MUTED, ha="left", va="center", zorder=7)

    heads = [(XCOL[0], "EXTRACTION", "virgin ore"),
             (XCOL[1], "PRODUCTION", "primary + secondary feed"),
             (XCOL[2], "USE", "products in service"),
             (XCOL[3], "END OF LIFE", "discarded each year"),
             (XCOL[4], "FATE", "recovered or lost")]
    for x, name, sub in heads:
        ax.text(x, 0.985, name, fontsize=7.2, fontweight="bold",
                color="#8a7f70", ha="left", va="top")
        ax.text(x, 0.950, sub, fontsize=6.5, color=MUTED, ha="left", va="top")

    ax.text(0.350, 0.155, f"Circularity rate {100.0 * circularity:.1f}%",
            fontsize=8.6, fontweight="bold", color=darken(RECY, 0.20),
            ha="left", va="top", zorder=7)
    ax.text(0.350, 0.117, textwrap.fill(
        f"secondary material as a share of the {prod_in:.0f} Mt/a entering "
        f"production; the remaining {100.0 * (1.0 - circularity):.1f}% must "
        f"still be mined.", 62),
        fontsize=6.6, color=SUB, ha="left", va="top", linespacing=1.5,
        zorder=7)


    print(f"in={EXTRACTED:.0f}  out={balance:.0f}  prod_in={prod_in:.0f}  "
          f"circularity={circularity:.4f}  eol_rate={eol_rate:.4f}  "
          f"lw_pt={lw_pt:.1f}")
    savefig(fig, "627_cir_material_flow.png")


if __name__ == "__main__":
    main()
