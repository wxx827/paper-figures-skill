"""Bi-temporal change detection over a 6 x 6 km peri-urban scene.

Main row: the two dates and the dNDVI change map, all three on the identical
footprint and the identical stretch, with the detected change polygons
outlined on the change map and echoed faintly on the later date.

Auxiliary row (58 % of the image tiles): area gained and lost per class as a
diverging bar chart, and the full class-transition matrix in square
kilometres. Both are counted from the same two classified rasters that
generate the imagery, so the panels cannot disagree with the map.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, binary_opening

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    shared_norm, image_axes, add_scale_bar)

N = 220                       # pixels per side
GSD = 30.0                    # m
KM = N * GSD / 1000.0         # scene width in km
PIX_KM2 = (GSD / 1000.0) ** 2
EXT = [0.0, KM, 0.0, KM]

# Scene rendering colours: land-cover semantics, not chart colours.
CLASSES = [
    ("Forest", "#2e7d5b", 0.79),
    ("Cropland", "#c9a227", 0.54),
    ("Water", "#3b5b92", -0.14),
    ("Built-up", "#8a8a8a", 0.13),
]
LOSS = "#7b3294"              # matches the low end of cb_diverging_pg
GAIN = "#3f8a4f"              # matches the high end
DTHR = 0.16                   # |dNDVI| threshold for a change polygon


def field(rng, sigma):
    z = gaussian_filter(rng.normal(size=(N, N)), sigma, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-12)


def disc(xx, yy, cx, cy, rx, ry, rot=0.0):
    ca, sa = np.cos(rot), np.sin(rot)
    u = (xx - cx) * ca + (yy - cy) * sa
    v = -(xx - cx) * sa + (yy - cy) * ca
    return (u / rx) ** 2 + (v / ry) ** 2 < 1.0


def build_scene(rng):
    yy, xx = np.mgrid[0:N, 0:N] * (GSD / 1000.0)

    river = np.abs(xx - (1.55 + 0.55 * np.sin(yy / 1.15))) < 0.13
    lake = disc(xx, yy, 5.10, 1.35, 0.85, 0.52, 0.35)
    water = river | lake

    core = disc(xx, yy, 4.35, 4.75, 1.05, 0.80, -0.4)
    built = (core | (field(rng, 2.2) > 1.55)) & ~water
    relief = field(rng, 11.0)
    forest = (relief > 0.42) & ~built & ~water
    cls1 = np.where(water, 2, np.where(built, 3, np.where(forest, 0, 1)))

    cls2 = cls1.copy()
    # clear-cut on the northern ridge
    cut = disc(xx, yy, 2.20, 5.05, 0.92, 0.60, 0.5) & (cls1 == 0)
    cls2[cut] = 1
    # urban expansion ring around the existing core
    grow = (disc(xx, yy, 4.35, 4.75, 1.62, 1.28, -0.4) & ~core
            & (field(rng, 1.6) > -0.15) & (cls1 == 1))
    cls2[grow] = 3
    # reservoir draw-down exposes the lake margin
    dry = lake & ~disc(xx, yy, 5.02, 1.30, 0.58, 0.34, 0.35)
    cls2[dry] = 1
    # plantation established on former cropland
    plant = disc(xx, yy, 5.55, 4.05, 0.62, 0.72, 0.0) & (cls1 == 1)
    cls2[plant] = 0

    # shared surface texture, so both dates look like the same landscape
    tex = 0.5 + 0.5 * np.clip(0.6 * field(rng, 1.4) + 0.4 * field(rng, 4.0),
                              -2.2, 2.2) / 2.2
    return cls1, cls2, tex


def render(cls, tex):
    """Natural-looking composite from the class map and a brightness field."""
    rgb = np.zeros((N, N, 3))
    for code, (_, hexcol, _) in enumerate(CLASSES):
        base = np.array([int(hexcol[i:i + 2], 16) / 255.0
                         for i in (1, 3, 5)])
        m = cls == code
        rgb[m] = base
    shade = (0.68 + 0.58 * tex)[:, :, None]
    return np.clip(rgb * shade, 0.0, 1.0)


def ndvi_of(cls, tex, rng, offset):
    v = np.zeros((N, N))
    for code, (_, _, mean) in enumerate(CLASSES):
        v[cls == code] = mean
    v = v + 0.11 * (tex - 0.5) + offset + 0.018 * field(rng, 1.0)
    return np.clip(v, -0.4, 1.0)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(759)

    cls1, cls2, tex = build_scene(rng)
    rgb1, rgb2 = render(cls1, tex), render(cls2, tex)
    nd1 = ndvi_of(cls1, tex, rng, 0.0)
    nd2 = ndvi_of(cls2, tex, rng, -0.025)
    dnd = gaussian_filter(nd2 - nd1, 0.8)

    changed = binary_opening(np.abs(dnd) > DTHR, np.ones((3, 3)))
    xg = (np.arange(N) + 0.5) * GSD / 1000.0

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    tiles = [axes_inches(fig, (x0, 2.98, 2.62, 2.62))
             for x0 in (0.52, 3.32, 6.12)]
    ax_bar = axes_inches(fig, (1.02, 0.98, 4.05, 1.52))
    ax_cm = axes_inches(fig, (6.62, 0.98, 1.52, 1.52))

    # ---- (a)-(c) the three rasters ---------------------------------------
    for ax, img, title in zip(tiles[:2], (rgb1, rgb2),
                              ("2019-07-14 (T1)", "2024-07-09 (T2)")):
        ax.imshow(img, extent=EXT, origin="lower", interpolation="bilinear")
        image_axes(ax, color="#333333", lw=0.7)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    tiles[1].contour(xg, xg, changed.astype(float), levels=[0.5],
                     colors="#f5f5f5", linewidths=0.6, alpha=0.85)

    norm = shared_norm([dnd], robust=(1, 99), symmetric=True)
    ax = tiles[2]
    im = ax.imshow(dnd, extent=EXT, origin="lower", cmap=get_cmap(
        "cb_diverging_pg"), norm=norm, interpolation="bilinear")
    ax.contour(xg, xg, changed.astype(float), levels=[0.5], colors="#1a1a1a",
               linewidths=0.7)
    image_axes(ax, color="#333333", lw=0.7)
    ax.set_title("Change map (T2 $-$ T1)", fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, im, ax, label="$\\Delta$NDVI (-)", width=0.11,
                  pad=0.07, labelsize=6.5, extend="both")

    add_scale_bar(tiles[0], 1.0, label="1 km", loc=(0.045, 0.055),
                  height=0.020, fontsize=6.2)
    for ax in tiles:
        ax.set_xlim(0, KM)
        ax.set_ylim(0, KM)

    legend_items = [Patch(facecolor=c, edgecolor="white", lw=0.4, label=n)
                    for n, c, _ in CLASSES]
    legend_items.append(Line2D([], [], color="#1a1a1a", lw=0.9,
                               label="Change polygon"))
    tiles[0].legend(handles=legend_items, loc="upper center", ncol=5,
                    fontsize=7.0, frameon=False, handlelength=1.2,
                    columnspacing=1.1, handletextpad=0.5,
                    bbox_to_anchor=(1.545, 1.215))

    pct = 100.0 * changed.mean()
    ax.text(0.035, 0.965, "%.1f %% of the scene changed\n(%.2f km$^2$)"
            % (pct, changed.sum() * PIX_KM2), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color="#1a1a1a",
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#cccccc",
                      lw=0.5, alpha=0.92))

    # ---- transition accounting -------------------------------------------
    names = [n for n, _, _ in CLASSES]
    nc = len(names)
    trans = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            trans[i, j] = np.count_nonzero((cls1 == i) & (cls2 == j))
    trans *= PIX_KM2
    gained = trans.sum(axis=0) - np.diag(trans)
    lost = trans.sum(axis=1) - np.diag(trans)

    # ---- (d) gain / loss per class ---------------------------------------
    ypos = np.arange(nc)[::-1]
    ax_bar.barh(ypos, gained, height=0.52, color=GAIN, edgecolor="white",
                linewidth=0.4, zorder=3, label="Gained")
    ax_bar.barh(ypos, -lost, height=0.52, color=LOSS, edgecolor="white",
                linewidth=0.4, zorder=3, label="Lost")
    net = gained - lost
    ax_bar.plot(net, ypos, ls="none", marker="D", ms=3.6, mfc="white",
                mec="#1a1a1a", mew=0.9, zorder=5, label="Net")
    ax_bar.axvline(0.0, color="#333333", lw=0.8, zorder=4)
    for y, g, l in zip(ypos, gained, lost):
        if g > 0:
            ax_bar.text(g + 0.04, y, "+%.2f" % g, ha="left", va="center",
                        fontsize=6.2, color="#2c6b3d")
        if l > 0:
            ax_bar.text(-l - 0.04, y, "%.2f" % -l, ha="right", va="center",
                        fontsize=6.2, color="#5d2470")
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(names, fontsize=6.8)
    lim = 1.35 * max(gained.max(), lost.max())
    ax_bar.set_xlim(-lim, lim)
    ax_bar.set_xlabel("Area change (km$^2$)", fontsize=7.4, labelpad=1.0)
    ax_bar.set_title("Area gained and lost per class", fontsize=8.5,
                     fontweight="bold")
    ax_bar.tick_params(labelsize=6.6)
    style_axes(ax_bar)
    ax_bar.grid(axis="x", lw=0.5, alpha=0.45)
    ax_bar.legend(loc="upper right", fontsize=6.4, frameon=False, ncol=3,
                  handlelength=1.1, columnspacing=0.9,
                  bbox_to_anchor=(1.0, 1.03))

    # ---- (e) transition matrix -------------------------------------------
    off = trans - np.diag(np.diag(trans))
    imc = ax_cm.imshow(off, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                       vmax=float(off.max()), interpolation="nearest")
    ax_cm.set_aspect("equal")
    short = [n[:5] for n in names]
    ax_cm.set_xticks(range(nc))
    ax_cm.set_yticks(range(nc))
    ax_cm.set_xticklabels(short, fontsize=6.2, rotation=36, ha="right")
    ax_cm.set_yticklabels(short, fontsize=6.2)
    ax_cm.tick_params(width=0.7, length=2.0)
    for s in ax_cm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    for i in range(nc):
        for j in range(nc):
            if i == j:
                ax_cm.text(j, i, "%.1f" % trans[i, j], ha="center",
                           va="center", fontsize=6.0, color="#777777")
            else:
                ax_cm.text(j, i, "%.2f" % trans[i, j], ha="center",
                           va="center", fontsize=6.0,
                           color=("white" if off[i, j] > 0.55 * off.max()
                                  else "#1a1a1a"))
    ax_cm.set_xlabel("Class at T2", fontsize=7.0, labelpad=1.0)
    ax_cm.set_ylabel("Class at T1", fontsize=7.0, labelpad=1.0)
    ax_cm.set_title("Transitions (km$^2$)", fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, imc, ax_cm, label="off-diagonal area (km$^2$)",
                  width=0.10, pad=0.07, labelsize=6.3)

    panel_labels_inches(fig, tiles + [ax_bar, ax_cm], dx=-0.16, dy=0.14)


    savefig(fig, "759_rs_change_detection.png")


if __name__ == "__main__":
    main()
