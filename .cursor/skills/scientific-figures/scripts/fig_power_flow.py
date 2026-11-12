"""AC power-flow solution drawn on the three-area network of figure 193.

Branches are coloured by loading (share of thermal rating) on a green-amber-
red ramp, a triangular marker at each mid-span gives the flow direction, and
the two branches above 100 % are redrawn as thick red dashes with their
loading spelt out.  Bus voltages are printed in per unit next to each node
and turn red when they leave the 0.95-1.05 band.

Layout note: the schematic keeps the whole left canvas; the colour scale, the
key and the loading histogram are stacked in a narrow right rail, and the
histogram is deliberately small so it never competes with the network.  Node
labels are offset in *points*, so the drawing needs no aspect lock and always
fills its axes.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
GREY = "#8d959c"
ALERT = "#b23a48"
CARD = "#f3f5f6"
CARD_E = "#d3d9de"

# bus id -> (x, y, kind, voltage p.u., label offset in points)
BUSES = {
    "N1": (1.25, 7.60, "gen",  1.031, (0, 12)),
    "N2": (4.20, 7.75, "gen",  1.024, (0, 12)),
    "N3": (2.70, 6.55, "sub",  1.012, (0, -13)),
    "N4": (1.05, 5.55, "load", 0.987, (-17, 2)),
    "N5": (4.55, 5.60, "load", 0.979, (0, 12)),
    "E1": (7.45, 7.55, "gen",  1.042, (0, 12)),
    "E2": (8.90, 6.95, "sub",  1.018, (2, 13)),
    "E3": (11.30, 7.70, "load", 0.968, (0, 12)),
    "E4": (7.05, 5.60, "load", 0.961, (0, -13)),
    "E5": (9.85, 5.60, "sub",  0.994, (19, -3)),
    "E6": (11.55, 6.35, "gen", 1.028, (5, -13)),
    "S1": (2.95, 2.95, "sub",  1.003, (15, -10)),
    "S2": (2.10, 1.15, "gen",  1.021, (0, -13)),
    "S3": (4.70, 1.00, "load", 0.952, (0, -13)),
    # label pushed low-right: the violation call-out arrives from above
    "S4": (6.30, 2.95, "load", 0.938, (13, -8)),
    "S5": (8.90, 1.15, "gen",  1.014, (0, -13)),
    "S6": (9.75, 2.90, "sub",  0.981, (17, -3)),
}

# (from, to, MW rating, loading %, +1 = flows from -> to)
BRANCHES = [
    ("N1", "N3", 1200,  78, +1), ("N2", "N3",  900,  62, +1),
    ("N3", "N4",  600,  71, +1), ("N3", "N5",  550,  58, +1),
    ("N4", "N5",  250,  34, -1),
    ("E1", "E2", 1400,  84, +1), ("E2", "E3",  700,  66, +1),
    ("E2", "E4",  650,  91, +1), ("E2", "E5", 1100,  73, +1),
    ("E5", "E3",  300,  47, +1), ("E4", "E5",  280,  29, -1),
    ("E6", "E3",  320,  52, +1), ("E6", "E5",  620,  44, +1),
    ("S1", "S2",  600,  55, -1), ("S2", "S3",  300,  38, +1),
    ("S1", "S4",  550,  63, +1), ("S3", "S4",  260,  41, +1),
    ("S4", "S6",  700,  69, -1), ("S5", "S6", 1000,  88, +1),
    ("S3", "S5",  320,  45, -1),
]
TIES = [
    ("N3", "E1", 1500,  96, -1), ("N5", "E4",  700, 108, +1),
    ("N4", "S1",  650,  74, +1), ("E5", "S6", 1200, 113, -1),
]

AREAS = [
    ((0.30, 4.95, 5.05, 3.40), "AREA NORTH", 0.46),
    ((6.45, 4.95, 5.65, 3.40), "AREA EAST", 6.61),
    ((1.50, 0.35, 9.20, 3.20), "AREA SOUTH", 4.30),
]

SHAPE = {"gen": "s", "load": "o", "sub": "^"}
SIZE = {"gen": 8.4, "load": 7.2, "sub": 8.6}
VLO, VHI = 0.95, 1.05
LOAD_MAX = 120.0


def darken(color, f=0.34):
    r, g, b = mpl.colors.to_rgb(color)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def lw_for(cap):
    return 0.95 + 2.15 * cap / 1500.0


def main():
    set_pub_style(base_font=8)
    cmap = LinearSegmentedColormap.from_list(
        "loading", ["#4a8f68", "#8fae52", "#dcb35c", "#d0803f", "#b23a48"],
        N=256)
    norm = Normalize(0.0, LOAD_MAX)

    fig = plt.figure(figsize=(7.8, 4.9))
    gs = fig.add_gridspec(1, 2, left=0.012, right=0.988, top=0.905,
                          bottom=0.035, wspace=0.03,
                          width_ratios=[1.0, 0.30])
    ax = fig.add_subplot(gs[0, 0])
    rail = gs[0, 1].subgridspec(3, 1, height_ratios=[0.115, 0.455, 0.43],
                                hspace=0.50)
    axcb = fig.add_subplot(rail[0])
    axk = fig.add_subplot(rail[1])
    axh = fig.add_subplot(rail[2])

    # ---- neutral area cards ----------------------------------------------
    for (x, y, w, h), name, tx in AREAS:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.26",
            facecolor=CARD, edgecolor=CARD_E, lw=0.8, ls=(0, (5, 2.5)),
            zorder=0))
        ax.text(tx, y + h + 0.13, name, fontsize=7.0, color="#8b959d",
                fontweight="bold", ha="left", va="bottom", zorder=1)

    # ---- branches, arrows and overload call-outs --------------------------
    over = []
    for a, b, cap, pct, sgn in BRANCHES + TIES:
        xa, ya = BUSES[a][:2]
        xb, yb = BUSES[b][:2]
        col = cmap(norm(pct))
        base_lw = lw_for(cap)
        if pct > 100:
            over.append((a, b, pct))
            ax.plot([xa, xb], [ya, yb], color="white", lw=base_lw + 2.6,
                    solid_capstyle="round", zorder=2.2)
            ax.plot([xa, xb], [ya, yb], color=ALERT, lw=base_lw + 1.1,
                    ls=(0, (4.5, 2.0)), solid_capstyle="butt", zorder=2.6)
        else:
            ax.plot([xa, xb], [ya, yb], color=col, lw=base_lw,
                    solid_capstyle="round", zorder=2)

        dx, dy = (xb - xa) * sgn, (yb - ya) * sgn
        rot = np.degrees(np.arctan2(dy, dx)) - 90.0
        ax.plot([(xa + xb) / 2], [(ya + yb) / 2], marker=(3, 0, rot),
                markersize=5.4, color=ALERT if pct > 100 else darken(col),
                markeredgecolor="white", markeredgewidth=0.55, zorder=5,
                linestyle="none")

    for a, b, pct in over:
        xa, ya = BUSES[a][:2]
        xb, yb = BUSES[b][:2]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ux, uy = xb - xa, yb - ya
        n = np.hypot(ux, uy)
        ax.text(mx - uy / n * 0.62, my + ux / n * 0.62, f"{pct:.0f}%",
                fontsize=6.4, color=ALERT, fontweight="bold", ha="center",
                va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor=ALERT, lw=0.6, alpha=0.95))

    # ---- buses with per-unit voltages -------------------------------------
    nv = 0
    for bid, (x, y, kind, v, (ox, oy)) in BUSES.items():
        bad = not (VLO <= v <= VHI)
        nv += bad
        ax.plot([x], [y], marker=SHAPE[kind], markersize=SIZE[kind],
                markerfacecolor=ALERT if bad else "#fbfbfb",
                markeredgecolor=ALERT if bad else INK,
                markeredgewidth=1.0 if bad else 0.85, zorder=6,
                linestyle="none")
        ha = "left" if ox > 4 else "right" if ox < -4 else "center"
        va = "bottom" if oy > 4 else "top" if oy < -4 else "center"
        ax.annotate(f"{v:.3f}", xy=(x, y), xytext=(ox, oy),
                    textcoords="offset points", ha=ha, va=va, fontsize=6.3,
                    color=ALERT if bad else "#5f6a72",
                    fontweight="bold" if bad else "normal", zorder=7)

    ax.annotate(f"{BUSES['S4'][3]:.3f} p.u.\nbelow the 0.95 limit",
                xy=(6.30, 2.95), xytext=(7.45, 4.22), fontsize=6.3,
                color=ALERT, ha="left", va="bottom", linespacing=1.35,
                zorder=8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=ALERT))

    allp = [p for *_, p, _ in BRANCHES + TIES]
    ax.set_xlim(-0.05, 12.35)
    ax.set_ylim(0.02, 8.98)
    ax.axis("off")
    ax.set_title("Power-flow solution \u2014 branch loading, flow direction "
                 "and bus voltages", fontsize=9, fontweight="bold",
                 loc="left", x=0.005, pad=13)
    ax.text(0.005, 1.012, f"{len(over)} of {len(allp)} branches above their "
            f"rating   \u00b7   {nv} bus outside {VLO:.2f}\u2013{VHI:.2f} "
            f"p.u.   \u00b7   mean loading {np.mean(allp):.0f}%",
            transform=ax.transAxes, fontsize=6.8, color="#6d7880",
            ha="left", va="bottom")

    # ---- rail 1: colour scale ---------------------------------------------
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axcb,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.2, color=INK,
                      pad=1.5)
    cb.set_ticks([0, 30, 60, 90, 120])
    cb.set_label("Branch loading (% of rating)", fontsize=6.8, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")

    # ---- rail 2: key -------------------------------------------------------
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    axk.axis("off")
    axk.text(0.0, 0.985, "BUS TYPE", fontsize=6.7, color="#6d7880",
             fontweight="bold", ha="left", va="center")
    node_h = [Line2D([], [], marker=SHAPE[k], color="none",
                     markerfacecolor="#fbfbfb", markeredgecolor=INK,
                     markeredgewidth=0.85, markersize=SIZE[k] * 0.85,
                     label=lab)
              for k, lab in (("gen", "Generation"), ("load", "Load bus"),
                             ("sub", "Substation"))]
    l1 = axk.legend(handles=node_h, loc="upper left",
                    bbox_to_anchor=(-0.06, 0.945), frameon=False,
                    fontsize=6.8, handletextpad=0.5, labelspacing=0.5,
                    borderpad=0.1)
    axk.add_artist(l1)
    l1.set_clip_on(False)

    axk.text(0.0, 0.545, "FLOW & VIOLATIONS", fontsize=6.7, color="#6d7880",
             fontweight="bold", ha="left", va="center")
    flow_h = [
        Line2D([], [], marker=(3, 0, -90), color="none",
               markerfacecolor="#4a5a52", markeredgecolor="white",
               markeredgewidth=0.55, markersize=5.4,
               label="Flow direction"),
        Line2D([], [], color=ALERT, lw=2.4, ls=(0, (4.5, 2.0)),
               label="Overloaded (>100%)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=ALERT,
               markeredgecolor=ALERT, markersize=6.2,
               label="Voltage violation"),
    ]
    l2 = axk.legend(handles=flow_h, loc="upper left",
                    bbox_to_anchor=(-0.06, 0.505), frameon=False,
                    fontsize=6.8, handletextpad=0.5, labelspacing=0.62,
                    handlelength=1.9, borderpad=0.1)
    axk.add_artist(l2)
    l2.set_clip_on(False)

    # ---- rail 3: loading histogram (subordinate) --------------------------
    bins = np.arange(0, 121, 20)
    cnt, _ = np.histogram(allp, bins=bins)
    ctr = bins[:-1] + 10
    axh.bar(ctr, cnt, width=17.0, color=[cmap(norm(c)) for c in ctr], lw=0,
            zorder=3)
    axh.axvline(100, color=ALERT, lw=0.8, ls=(0, (3, 2)), zorder=4)
    axh.set_xlim(0, 120)
    axh.set_xticks([0, 60, 120])
    axh.set_ylim(0, cnt.max() * 1.28)
    axh.set_yticks([0, cnt.max()])
    axh.tick_params(labelsize=6.4)
    axh.set_xlabel("Loading (%)", fontsize=6.8, labelpad=1.5)
    axh.set_ylabel("Branches", fontsize=6.8, labelpad=2)
    axh.set_title("Loading distribution", fontsize=7.0, fontweight="bold",
                  loc="left", x=0.0, pad=3)
    axh.grid(axis="y", lw=0.4, alpha=0.5)
    style_axes(axh)

    print(f"branches={len(allp)}  overloaded={len(over)}  "
          f"mean loading={np.mean(allp):.1f}%  max={max(allp)}%")
    print(f"voltage range {min(v for *_, v, _ in BUSES.values()):.3f}"
          f"\u2013{max(v for *_, v, _ in BUSES.values()):.3f} p.u.  "
          f"violations={nv}")
    savefig(fig, "198_power_flow.png")


if __name__ == "__main__":
    main()
