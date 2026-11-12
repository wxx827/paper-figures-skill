"""Four-stage catchment water balance drawn as a Sankey diagram.

Ribbon width is the annual volume in 10^8 m3.  The stream is partitioned
twice - precipitation into evapotranspiration, infiltration and surface
runoff, then infiltration into recharge and deep percolation - and finally
allocated from the two stores (river channel and aquifer) to four use
sectors.  Ribbons are coloured by where the water *goes*, which is the only
question the diagram is meant to answer; every departure that never reaches a
user (evapotranspiration, deep percolation, outflow to the sea) peels away
downward in grey and is labelled with its share of precipitation.

Every volume printed on the figure is derived from the same three tables at
the top of the file, so the ribbon widths and the percentages cannot drift
apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

from pubstyle import set_pub_style, savefig, darken

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
LOSS = "#b3b3b3"
COL = {
    "Precipitation": "#4a7f9e",
    "Surface runoff": "#3b5b92",
    "Infiltration": "#2a7f7a",
    "Evapotranspiration": LOSS,
    "River channel": "#3b5b92",
    "Groundwater": "#2a7f7a",
    "Deep percolation": LOSS,
    "Agriculture": "#2e7d5b",
    "Industry": "#3d6b8e",
    "Domestic": "#c9a227",
    "Ecological flow": "#6bbf8a",
    "Outflow to sea": LOSS,
}

PRECIP = 328.0                                   # 10^8 m3 a-1
PARTITION = [("Surface runoff", 0.180), ("Infiltration", 0.240),
             ("Evapotranspiration", 0.580)]
DEEP_SHARE = 0.280                               # of infiltration
# sector demand (10^8 m3 a-1) and the share of it met from groundwater
SECTORS = [("Agriculture", 42.0, 0.45), ("Industry", 14.5, 0.30),
           ("Domestic", 9.5, 0.55), ("Ecological flow", 7.0, 0.10)]

XCOL = [0.000, 0.305, 0.615, 0.925]
NODE_W = 0.016
TOP = 0.870
GAP = 0.030
LOSS_GAP = 0.075


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


def stack(x, entries, scale):
    """Lay a column of nodes out top-down; returns {name: (top, bottom)}."""
    y, out = TOP, {}
    for name, val, extra in entries:
        y -= extra
        out[name] = (y, y - val * scale)
        y -= val * scale + GAP
    return out


def main():
    set_pub_style(base_font=8)

    # ---- volumes -----------------------------------------------------------
    part = {n: PRECIP * f for n, f in PARTITION}
    infil = part["Infiltration"]
    deep = infil * DEEP_SHARE
    recharge = infil - deep
    river = part["Surface runoff"]

    gw_use = {n: d * g for n, d, g in SECTORS}
    sw_use = {n: d * (1.0 - g) for n, d, g in SECTORS}
    demand = sum(d for _, d, _ in SECTORS)
    gw_out = recharge - sum(gw_use.values())
    sw_out = river - sum(sw_use.values())
    outflow = gw_out + sw_out

    scale = 0.720 / PRECIP
    fig = plt.figure(figsize=(10.4, 5.8))
    ax = fig.add_axes([0.005, 0.012, 0.990, 0.852])
    ax.set_xlim(-0.235, 1.240)
    ax.set_ylim(0.000, 1.020)
    ax.axis("off")

    # ---- nodes -------------------------------------------------------------
    c0 = stack(XCOL[0], [("Precipitation", PRECIP, 0.0)], scale)
    c1 = stack(XCOL[1], [("Surface runoff", river, 0.0),
                         ("Infiltration", infil, 0.0),
                         ("Evapotranspiration", part["Evapotranspiration"],
                          LOSS_GAP)], scale)
    c2 = stack(XCOL[2], [("River channel", river, 0.0),
                         ("Groundwater", recharge, 0.0),
                         ("Deep percolation", deep, LOSS_GAP)], scale)
    c3_entries = [(n, d, 0.0) for n, d, _ in SECTORS]
    c3_entries.append(("Outflow to sea", outflow, LOSS_GAP))
    c3 = stack(XCOL[3], c3_entries, scale)

    for col, x in ((c0, XCOL[0]), (c1, XCOL[1]), (c2, XCOL[2]),
                   (c3, XCOL[3])):
        for name, (t, b) in col.items():
            node(ax, x, t, t - b, COL[name])

    # ---- stage 1: precipitation -> partition -------------------------------
    y = c0["Precipitation"][0]
    for name, _ in PARTITION:
        h = part[name] * scale
        ribbon(ax, XCOL[0] + NODE_W, y, y - h, XCOL[1], c1[name][0],
               c1[name][1], COL[name])
        y -= h

    # ---- stage 2: partition -> stores --------------------------------------
    ribbon(ax, XCOL[1] + NODE_W, *c1["Surface runoff"], XCOL[2],
           *c2["River channel"], COL["River channel"])
    it, ib = c1["Infiltration"]
    ribbon(ax, XCOL[1] + NODE_W, it, it - recharge * scale, XCOL[2],
           *c2["Groundwater"], COL["Groundwater"])
    ribbon(ax, XCOL[1] + NODE_W, it - recharge * scale, ib, XCOL[2],
           *c2["Deep percolation"], LOSS, alpha=0.62)

    # ---- stage 3: stores -> users ------------------------------------------
    src_y = {"River channel": c2["River channel"][0],
             "Groundwater": c2["Groundwater"][0]}
    dst_y = {n: c3[n][0] for n in c3}
    for name in [n for n, _, _ in SECTORS] + ["Outflow to sea"]:
        for src, table in (("River channel", sw_use),
                           ("Groundwater", gw_use)):
            v = (table[name] if name != "Outflow to sea"
                 else (sw_out if src == "River channel" else gw_out))
            if v <= 0:
                continue
            h = v * scale
            ribbon(ax, XCOL[2] + NODE_W, src_y[src], src_y[src] - h,
                   XCOL[3], dst_y[name], dst_y[name] - h, COL[name],
                   alpha=0.62 if name == "Outflow to sea" else 0.52)
            src_y[src] -= h
            dst_y[name] -= h

    # ---- labels ------------------------------------------------------------
    def pct(v):
        return 100.0 * v / PRECIP

    t, b = c0["Precipitation"]
    ax.text(XCOL[0] - 0.014, 0.5 * (t + b) + 0.016, "Precipitation",
            fontsize=7.6, fontweight="bold", color=INK, ha="right",
            va="center", zorder=7)
    ax.text(XCOL[0] - 0.014, 0.5 * (t + b) - 0.016,
            f"{PRECIP:.0f} \u00d7 10$^{8}$ m$^{3}$ a$^{{-1}}$", fontsize=6.8,
            color=SUB, ha="right", va="center", zorder=7)

    for col, x in ((c1, XCOL[1]), (c2, XCOL[2])):
        for name, (nt, _) in col.items():
            v = {"Surface runoff": river, "Infiltration": infil,
                 "Evapotranspiration": part["Evapotranspiration"],
                 "River channel": river, "Groundwater": recharge,
                 "Deep percolation": deep}[name]
            ax.text(x, nt + 0.012,
                    f"{name}   {v:.1f}  ({pct(v):.0f}%)", fontsize=6.8,
                    color=darken(COL[name], 0.30), ha="left", va="bottom",
                    zorder=7)

    for name, d, g in SECTORS:
        nt, nb = c3[name]
        ax.text(XCOL[3] + NODE_W + 0.012, 0.5 * (nt + nb),
                f"{name}   {d:.1f}  ({pct(d):.1f}%)", fontsize=6.8,
                color=darken(COL[name], 0.28), ha="left", va="center",
                zorder=7)
    nt, nb = c3["Outflow to sea"]
    ax.text(XCOL[3] + NODE_W + 0.012, 0.5 * (nt + nb),
            f"Outflow to sea   {outflow:.1f}  ({pct(outflow):.1f}%)",
            fontsize=6.8, color="#8c8c8c", ha="left", va="center", zorder=7)

    heads = [(XCOL[0], "PRECIPITATION", f"{PRECIP:.0f} \u00d7 10$^{8}$ m$^{3}$"),
             (XCOL[1], "PARTITION", "surface / subsurface / vapour"),
             (XCOL[2], "STORES", f"{river + recharge:.1f} available"),
             (XCOL[3], "WITHDRAWAL", f"{demand:.1f} to users")]
    for x, name, sub in heads:
        ax.text(x, 0.985, name, fontsize=7.2, fontweight="bold",
                color="#4e7186", ha="left", va="top")
        ax.text(x, 0.952, sub, fontsize=6.6, color=MUTED, ha="left",
                va="top")

    ax.text(XCOL[1] + NODE_W + 0.020, 0.5 * sum(c1["Evapotranspiration"]),
            "consumed as vapour\nbefore it can be used:\n"
            f"{pct(part['Evapotranspiration']):.0f}% of precipitation",
            fontsize=6.8, color="#8c8c8c", ha="left", va="center",
            linespacing=1.5, zorder=7)

    # the lower right of any Sankey is dead space once the losses have peeled
    # away; the reading notes go there instead of in a bottom footnote band
    ax.text(XCOL[2] - 0.055, 0.360, textwrap.fill(
        "Ribbon width is volume in 10^8 m3 per year and ribbons are coloured "
        "by destination; grey ribbons are departures that never reach a "
        "user. Percentages are shares of precipitation. Groundwater supplies "
        f"{100 * sum(gw_use.values()) / demand:.0f}% of total withdrawal and "
        "the river channel the rest, so the aquifer carries a load out of "
        "proportion to the recharge that sustains it.", 74),
        ha="left", va="top", fontsize=6.6, color=MUTED, linespacing=1.55,
        zorder=7)

    print(f"P={PRECIP:.1f}  ET={part['Evapotranspiration']:.1f}  "
          f"infil={infil:.1f}  runoff={river:.1f}")
    print(f"recharge={recharge:.1f}  deep={deep:.1f}  demand={demand:.1f}  "
          f"outflow={outflow:.1f}  (gw {gw_out:.1f} + sw {sw_out:.1f})")
    print(f"balance check: {part['Evapotranspiration'] + deep + demand + outflow:.1f}"
          f" vs {PRECIP:.1f}")
    savefig(fig, "603_wat_balance_sankey.png")


if __name__ == "__main__":
    main()
