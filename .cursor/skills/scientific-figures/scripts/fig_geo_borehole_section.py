"""Borehole correlation section along a 1.6 km site-investigation traverse.

Seven boreholes are drawn at their true chainage.  Equivalent units are joined
between adjacent holes by ornamented polygons, so the panel reads as a
continuous cross-section rather than a row of logs.  Where a unit is missing
from a hole the wedge tapers to a point part-way along the intervening span,
which is how a pinch-out is drawn by hand.  The plan inset gives the traverse
geometry; vertical exaggeration is computed from the axes box and stated on
the figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, lighten, darken, add_panel_label

# ---- house lithology set, shared across the geology figures ---------------
LITH = [
    ("Made ground",        lighten("#c0603f", 0.74), "xx"),
    ("Alluvial sand",      lighten("#c9a227", 0.46), "...."),
    ("Silty clay",         lighten("#2e7d5b", 0.70), "----"),
    ("Terrace gravel",     lighten("#c0603f", 0.52), "oo"),
    ("Weathered mudstone", lighten("#8e6bbf", 0.72), "--.."),
    ("Sandstone bedrock",  lighten("#3b5b92", 0.62), "\\\\"),
]

WELLS = ["BH-01", "BH-02", "BH-03", "BH-04", "BH-05", "BH-06", "BH-07"]
CHAIN = np.array([0.0, 210.0, 430.0, 760.0, 1080.0, 1340.0, 1600.0])
# depth below section datum to the ground surface at each hole (m)
SURFACE = np.array([18.0, 14.0, 11.0, 8.0, 10.0, 15.0, 20.0])
# drilled depth below datum
TD = np.array([74.0, 90.0, 100.0, 105.0, 101.0, 94.0, 80.0])

THICK = {
    "Made ground":        np.array([3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 2.0]),
    "Alluvial sand":      np.array([0.0, 7.0, 14.0, 19.0, 12.0, 5.0, 0.0]),
    "Silty clay":         np.array([9.0, 11.0, 10.0, 8.0, 12.0, 14.0, 13.0]),
    "Terrace gravel":     np.array([0.0, 0.0, 5.0, 9.0, 7.0, 4.0, 2.0]),
    "Weathered mudstone": np.array([14.0, 16.0, 15.0, 13.0, 17.0, 18.0, 16.0]),
}

ZMAX = 120.0
PINCH = 0.35          # fraction of the span a missing unit stays absent
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def interp_thickness(xg, t):
    """Piecewise-linear thickness with pinch-outs held off the borehole."""
    out = np.zeros_like(xg)
    for i in range(len(CHAIN) - 1):
        x0, x1 = CHAIN[i], CHAIN[i + 1]
        m = (xg >= x0) & (xg <= x1)
        u = (xg[m] - x0) / (x1 - x0)
        t0, t1 = t[i], t[i + 1]
        if t0 <= 0 < t1:
            v = np.clip((u - PINCH) / (1.0 - PINCH), 0.0, 1.0) * t1
        elif t1 <= 0 < t0:
            v = np.clip((1.0 - u - PINCH) / (1.0 - PINCH), 0.0, 1.0) * t0
        else:
            v = t0 + (t1 - t0) * u
        out[m] = v
    return out


def main():
    set_pub_style(**{"hatch.linewidth": 0.45})
    from scipy.interpolate import PchipInterpolator

    xg = np.linspace(CHAIN[0], CHAIN[-1], 900)
    surf = PchipInterpolator(CHAIN, SURFACE)(xg)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.46],
                          width_ratios=[1.0, 0.52], left=0.062, right=0.985,
                          top=0.885, bottom=0.075, hspace=0.42, wspace=0.10)
    ax = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_k = fig.add_subplot(gs[1, 1])

    # ---- correlated section ----------------------------------------------
    top = surf.copy()
    tops_at_well = {}
    twell = SURFACE.copy()
    for name, fc, hatch in LITH[:-1]:
        thick = interp_thickness(xg, THICK[name])
        base = top + thick
        ax.fill_between(xg, top, base, facecolor=fc, hatch=hatch,
                        edgecolor=darken(fc, 0.42), linewidth=0.55, zorder=2)
        ax.plot(xg, base, color=darken(fc, 0.5), lw=0.7, zorder=3)
        tops_at_well[name] = (twell.copy(), twell + THICK[name])
        twell = twell + THICK[name]
        top = base

    name, fc, hatch = LITH[-1]
    ax.fill_between(xg, top, ZMAX, facecolor=fc, hatch=hatch,
                    edgecolor=darken(fc, 0.42), linewidth=0.55, zorder=2)
    tops_at_well[name] = (twell.copy(), np.full_like(twell, ZMAX))

    ax.plot(xg, surf, color=RULE, lw=1.4, zorder=6)

    # ---- borehole tracks --------------------------------------------------
    bw = 26.0
    for i, w in enumerate(WELLS):
        x = CHAIN[i]
        for name, fc, hatch in LITH:
            z0, z1 = tops_at_well[name][0][i], tops_at_well[name][1][i]
            z1 = min(z1, TD[i])
            if z1 - z0 < 0.05:
                continue
            ax.add_patch(Rectangle((x - bw / 2, z0), bw, z1 - z0,
                                   facecolor=fc, edgecolor=INK, linewidth=0.8,
                                   hatch=hatch, zorder=7))
        ax.plot([x, x], [TD[i], TD[i] + 4.5], color=INK, lw=0.9, zorder=7)
        ax.plot([x - 9, x + 9], [TD[i] + 4.5] * 2, color=INK, lw=1.2,
                zorder=7)
        ax.plot([x], [SURFACE[i]], marker="v", ms=5.0, mfc="white",
                mec=INK, mew=0.9, zorder=8, clip_on=False)
        ax.text(x, SURFACE[i] - 6.5, w, ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color=INK, zorder=8)
        ax.text(x + 22, TD[i] + 3.0, f"TD {TD[i]:.0f} m", ha="left",
                va="center", fontsize=6.2, color=SUB, zorder=8,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.82))

    # pinch-out call-out, located from the interpolated thickness itself
    tg = interp_thickness(xg, THICK["Alluvial sand"])
    i_p = int(np.argmax(tg > 0.05))
    ax.annotate("sand pinches out\nagainst valley margin",
                xy=(xg[i_p], surf[i_p] + 4.0), xytext=(150.0, 62.0),
                fontsize=6.8, color=RULE, ha="left", va="center",
                linespacing=1.3, zorder=9,
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=RULE, lw=0.8,
                                connectionstyle="arc3,rad=-0.3"))

    ax.set_xlim(-125.0, CHAIN[-1] + 125.0)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_xlabel("Chainage along section A-A' (m)")
    ax.set_ylabel("Depth below section datum (m)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.text(0.012, 0.955, "A", transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", color=INK, ha="left", va="top")
    ax.text(0.988, 0.955, "A'", transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", color=INK, ha="right", va="top")

    fig.canvas.draw()
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    ve = ((CHAIN[-1] + 250.0) / (bb.width * fw)) / (ZMAX / (bb.height * fh))
    ax.text(0.988, 0.045, f"vertical exaggeration x{ve:.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=MUTED, zorder=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

    # ---- plan of the traverse (auxiliary) --------------------------------
    east = np.array([412.0, 604.0, 812.0, 1128.0, 1436.0, 1690.0, 1944.0])
    north = np.array([1180.0, 1236.0, 1284.0, 1352.0, 1424.0, 1478.0, 1532.0])
    off_e = np.array([760.0, 1320.0, 1780.0])
    off_n = np.array([1420.0, 1160.0, 1300.0])

    ax_p.plot(east, north, color=RULE, lw=1.1, ls=(0, (5, 2)), zorder=2)
    ax_p.plot(off_e, off_n, marker="o", ls="none", ms=3.6, mfc="white",
              mec=MUTED, mew=0.8, zorder=3, label="off-section hole")
    ax_p.plot(east, north, marker="o", ls="none", ms=4.6, mfc="#b23a48",
              mec="white", mew=0.7, zorder=4, label="section hole")
    for e, n, w in zip(east, north, WELLS):
        ax_p.text(e, n + 34, w.split("-")[1], ha="center", va="bottom",
                  fontsize=6.0, color=SUB)
    ax_p.text(east[0] - 55, north[0] - 18, "A", fontsize=8.0,
              fontweight="bold", color=INK, ha="right", va="top")
    ax_p.text(east[-1] + 55, north[-1] + 8, "A'", fontsize=8.0,
              fontweight="bold", color=INK, ha="left", va="bottom")
    ax_p.annotate("", xy=(560.0, 1560.0), xytext=(560.0, 1450.0),
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0))
    ax_p.text(560.0, 1576.0, "N", ha="center", va="bottom", fontsize=6.8,
              color=RULE, fontweight="bold")
    ax_p.set_aspect("equal")
    ax_p.set_xlim(300.0, 2060.0)
    ax_p.set_ylim(1080.0, 1640.0)
    ax_p.set_xlabel("Local easting (m)")
    ax_p.set_ylabel("Local northing (m)")
    ax_p.tick_params(labelsize=6.5)
    ax_p.xaxis.label.set_size(7.2)
    ax_p.yaxis.label.set_size(7.2)
    ax_p.grid(color="#e6e6e6", lw=0.5)
    ax_p.set_axisbelow(True)
    ax_p.legend(loc="lower right", frameon=False, fontsize=6.5,
                handletextpad=0.4, borderaxespad=0.2)
    ax_p.set_title("Traverse plan", fontsize=8.5, fontweight="bold")

    # ---- ornament key -----------------------------------------------------
    ax_k.axis("off")
    ax_k.set_xlim(0, 1)
    ax_k.set_ylim(0, 1)
    ax_k.set_title("Unit ornament", fontsize=8.5, fontweight="bold",
                   loc="left", x=0.0)
    for j, (name, fc, hatch) in enumerate(LITH):
        y = 0.94 - j * 0.145
        ax_k.add_patch(Rectangle((0.02, y - 0.052), 0.16, 0.100,
                                 facecolor=fc, edgecolor=darken(fc, 0.42),
                                 linewidth=0.6, hatch=hatch,
                                 transform=ax_k.transAxes))
        ax_k.text(0.215, y, name, ha="left", va="center", fontsize=7.0,
                  color=INK)
    n_pinch = sum(int(np.any(v <= 0)) for v in THICK.values())
    ax_k.text(0.02, 0.04, f"{len(WELLS)} holes, {len(LITH)} units, "
              f"{n_pinch} units pinching out", ha="left", va="bottom",
              fontsize=6.5, color=MUTED)

    add_panel_label(ax, "(a)", x=-0.055, y=1.045)
    add_panel_label(ax_p, "(b)", x=-0.115, y=1.045)

    savefig(fig, "405_geo_borehole_section.png")


if __name__ == "__main__":
    main()
