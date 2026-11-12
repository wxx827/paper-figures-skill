"""Chip floorplan: functional blocks tiled across a die with standard-cell
row texture, routing channels and a magnified SRAM sub-array, beside a
subordinate column of area share and placement utilisation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PatchCollection

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import image_axes, panel_labels_inches, corner_stamp

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
CHANNEL = "#f0f0f0"
TRACE = "#c2c8d0"

# Block classes: logic on house blue, memory on house green, IO on house gold,
# analog on house purple.  Five classes, matching DESIGN_SYSTEM section 2.1.
KIND = {
    "cpu":  ("CPU logic",   "#3b5b92"),
    "gpu":  ("GPU logic",   lighten("#3b5b92", 0.30)),
    "sram": ("SRAM array",  "#2e7d5b"),
    "io":   ("IO / PHY",    "#c9a227"),
    "pll":  ("PLL / analog", "#8e6bbf"),
}

DIE_W, DIE_H = 5200.0, 4400.0          # um
RING = 220.0                            # IO ring thickness, um

# name, x0, y0, w, h, kind, cell-row pitch (um), placement utilisation
BLOCKS = [
    ("CPU core 0", 380, 3260, 940, 760, "cpu", 38, 0.82),
    ("CPU core 1", 1410, 3260, 940, 760, "cpu", 38, 0.80),
    ("CPU core 2", 380, 2410, 940, 760, "cpu", 38, 0.79),
    ("CPU core 3", 1410, 2410, 940, 760, "cpu", 38, 0.81),
    ("L3 SRAM", 2440, 2410, 900, 1610, "sram", 26, 0.93),
    ("GPU cluster", 3430, 2410, 1390, 1610, "gpu", 46, 0.71),
    ("L2 SRAM bank A", 380, 1430, 1180, 800, "sram", 26, 0.91),
    ("L2 SRAM bank B", 1650, 1430, 1180, 800, "sram", 26, 0.90),
    ("NoC fabric", 2920, 1430, 1900, 800, "cpu", 54, 0.62),
    ("PLL", 380, 380, 560, 900, "pll", 90, 0.48),
    ("DDR PHY", 1030, 380, 1800, 900, "io", 70, 0.66),
    ("SerDes PHY", 2920, 380, 1900, 900, "io", 70, 0.64),
]

# Magnifier crop, sized so the inset in the caption band below the die is
# filled exactly (see floorplan_panel): 900 / 477 = the inset's paper aspect.
ZOOM = (2440, 3420, 900, 477)           # x0, y0, w, h in um - the L3 SRAM
ZOOM_C = "#b23a48"


def cell_rows(ax, x0, y0, w, h, pitch, color="white", alpha=0.32, lw=0.35):
    """Hairlines standing in for standard-cell rows inside a logic block."""
    ys = np.arange(y0 + pitch, y0 + h - pitch * 0.4, pitch)
    if ys.size == 0:
        return
    segs = [[(x0 + 8, y), (x0 + w - 8, y)] for y in ys]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=5))


def sram_banks(ax, x0, y0, w, h, color, pitch):
    """Sub-bank tiling plus bit-line hairlines, the SRAM density signature."""
    nx = max(2, int(round(w / 235.0)))
    ny = max(2, int(round(h / 215.0)))
    gap = 15.0
    bw = (w - gap * (nx - 1)) / nx
    bh = (h - gap * (ny - 1)) / ny
    rects = []
    for i in range(nx):
        for j in range(ny):
            bx = x0 + i * (bw + gap)
            by = y0 + j * (bh + gap)
            rects.append(Rectangle((bx, by), bw, bh))
            cell_rows(ax, bx, by, bw, bh, pitch, color=lighten(color, 0.72),
                      alpha=0.75, lw=0.32)
    ax.add_collection(PatchCollection(
        rects, facecolors="none", edgecolors=lighten(color, 0.42),
        linewidths=0.45, zorder=4))


def draw_die(ax, labels=True):
    """Paint the whole die: IO ring, routing channels, blocks, textures."""
    ax.add_patch(Rectangle((0, 0), DIE_W, DIE_H, facecolor="#fafafa",
                           edgecolor=RULE, linewidth=0.9, zorder=1))
    gold = KIND["io"][1]
    for r in [(0, 0, DIE_W, RING), (0, DIE_H - RING, DIE_W, RING),
              (0, RING, RING, DIE_H - 2 * RING),
              (DIE_W - RING, RING, RING, DIE_H - 2 * RING)]:
        ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor=gold,
                               edgecolor="none", alpha=0.85, zorder=2))
    # bond pads punched into the ring
    pads = []
    for x in np.arange(150, DIE_W - 100, 190):
        pads += [Rectangle((x, 52), 96, 96), Rectangle((x, DIE_H - 148), 96, 96)]
    for y in np.arange(330, DIE_H - 300, 190):
        pads += [Rectangle((52, y), 96, 96), Rectangle((DIE_W - 148, y), 96, 96)]
    ax.add_collection(PatchCollection(pads, facecolors=darken(gold, 0.28),
                                      edgecolors="none", zorder=3))

    ax.add_patch(Rectangle((RING, RING), DIE_W - 2 * RING, DIE_H - 2 * RING,
                           facecolor=CHANNEL, edgecolor="none", zorder=3))

    # routing channels between the block rows and columns
    for xc in (1365, 2395, 3385):
        for k in range(-3, 4):
            ax.plot([xc + k * 11, xc + k * 11], [300, DIE_H - 300],
                    color=TRACE, lw=0.35, zorder=3.5)
    for yc in (2320, 1355):
        for k in range(-3, 4):
            ax.plot([300, DIE_W - 300], [yc + k * 11, yc + k * 11],
                    color=TRACE, lw=0.35, zorder=3.5)

    for name, x0, y0, w, h, kind, pitch, _u in BLOCKS:
        col = KIND[kind][1]
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=col,
                               edgecolor="white", linewidth=0.7, zorder=4))
        if kind == "sram":
            sram_banks(ax, x0, y0, w, h, col, pitch)
        else:
            cell_rows(ax, x0, y0, w, h, pitch)
        if labels:
            fs = 7.0 if w * h > 9.0e5 else 6.3
            ax.text(x0 + w / 2, y0 + h / 2 + (24 if w * h > 6e5 else 0), name,
                    ha="center", va="center", color="white", fontsize=fs,
                    fontweight="bold", zorder=7)
            if w * h > 6e5:
                ax.text(x0 + w / 2, y0 + h / 2 - 62,
                        f"{w * h / 1e6:.2f} mm$^2$", ha="center", va="center",
                        color="white", fontsize=6.0, alpha=0.92, zorder=7)


def floorplan_panel(fig, ax):
    draw_die(ax, labels=True)
    # The band below the die holds the magnifier, so the inset never covers a
    # functional block; it also soaks up the whitespace an equal-aspect panel
    # would otherwise leave above and below the die.
    ax.set_xlim(-150, DIE_W + 150)
    ax.set_ylim(-1500, DIE_H + 150)
    image_axes(ax, color="#8a8a8a", lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)

    zx, zy, zw, zh = ZOOM
    ax.add_patch(Rectangle((zx, zy), zw, zh, facecolor="none",
                           edgecolor=ZOOM_C, linewidth=1.2, zorder=8))

    loc = [0.60, 0.020, 0.39, 0.188]
    axi = ax.inset_axes(loc)
    draw_die(axi, labels=False)
    axi.set_xlim(zx, zx + zw)
    axi.set_ylim(zy, zy + zh)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_aspect("auto")
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.4)
        s.set_color(ZOOM_C)
    axi.set_zorder(9)
    axi.patch.set_visible(True)
    ax.annotate("", xy=(loc[0] + 0.02, loc[1] + loc[3]),
                xycoords="axes fraction", xytext=(zx + zw * 0.35, zy),
                textcoords="data",
                arrowprops=dict(arrowstyle="-", color=ZOOM_C, lw=0.7,
                                alpha=0.9), zorder=8)
    ax.text(0.015, 0.150, "Magnified L3 SRAM (red box)", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, color=INK,
            fontweight="bold")
    ax.text(0.015, 0.082, "32 sub-banks of 256 \u00d7 128 bit\n"
            "bit-cell pitch 0.26 \u00b5m  \u00b7  6T cell",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.4,
            color=SUB, linespacing=1.45)
    ax.set_title("Die floorplan  \u00b7  5200 \u00d7 4400 \u00b5m",
                 fontsize=9.0, fontweight="bold", pad=5)


def area_table():
    """Area per block class in um^2, plus the leftover routing channel area."""
    tot = {k: 0.0 for k in KIND}
    for _n, _x, _y, w, h, kind, _p, _u in BLOCKS:
        tot[kind] += w * h
    ring = DIE_W * DIE_H - (DIE_W - 2 * RING) * (DIE_H - 2 * RING)
    tot["io"] += ring
    used = sum(tot.values())
    return tot, DIE_W * DIE_H - used


def area_panel(ax, tot, free):
    die = DIE_W * DIE_H
    items = sorted(tot.items(), key=lambda kv: kv[1])
    names = [KIND[k][0] for k, _v in items] + ["Routing channel"]
    vals = [v for _k, v in items] + [free]
    cols = [KIND[k][1] for k, _v in items] + ["#b3b3b3"]
    y = np.arange(len(vals))
    ax.barh(y, [100.0 * v / die for v in vals], height=0.62, color=cols,
            edgecolor="white", linewidth=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(100.0 * v / die + 0.9, yi,
                f"{100.0 * v / die:.1f}%   {v / 1e6:.2f} mm$^2$", ha="left",
                va="center", fontsize=6.2, color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlim(0, 100.0 * max(vals) / die * 1.62)
    ax.set_ylim(-0.62, len(vals) - 0.38)
    ax.set_xlabel("Share of die area (%)", fontsize=7.5)
    ax.set_title("Area budget by block class", fontsize=8.5, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def util_panel(ax):
    rows = [(n, u, KIND[k][1], w * h) for n, _x, _y, w, h, k, _p, u in BLOCKS]
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    mean_u = float(np.average([r[1] for r in rows],
                              weights=[r[3] for r in rows]))
    ax.barh(y, [100 * r[1] for r in rows], height=0.66,
            color=[r[2] for r in rows], edgecolor="white", linewidth=0.6,
            zorder=3)
    ax.axvline(100 * mean_u, color="#808080", lw=0.9, ls=(0, (4, 2.5)),
               zorder=4)
    ax.text(100 * mean_u + 1.4, -0.12, f"die mean {100 * mean_u:.1f}%",
            ha="left", va="center", fontsize=6.2, color=SUB)
    for yi, r in zip(y, rows):
        ax.text(100 * r[1] - 1.6, yi, f"{100 * r[1]:.0f}", ha="right",
                va="center", fontsize=6.1, color="white", fontweight="bold",
                zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.62, len(rows) - 0.34)
    ax.set_xlabel("Placement utilisation (%)", fontsize=7.5)
    ax.set_title("Cell utilisation per block", fontsize=8.5, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    return mean_u


def main():
    set_pub_style()
    tot, free = area_table()
    die = DIE_W * DIE_H
    top_kind = max(tot, key=lambda k: tot[k])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86], wspace=0.24, hspace=0.42,
                          left=0.035, right=0.905, top=0.885, bottom=0.085)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_util = fig.add_subplot(gs[1, 1])

    floorplan_panel(fig, ax_main)
    area_panel(ax_area, tot, free)
    mean_u = util_panel(ax_util)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white",
                         linewidth=0.6, label=lab)
               for lab, c in KIND.values()]


    panel_labels_inches(fig, [ax_main, ax_area, ax_util], dx=-0.24, dy=0.08)
    savefig(fig, "498_semi_layout_floorplan.png")


if __name__ == "__main__":
    main()
