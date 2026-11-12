"""Species-richness hotspot map with a conservation gap analysis.

Left: a synthetic landmass carrying a gridded species-richness surface, the
top-decile hotspot contour and five protected-area boundaries, finished with a
north arrow and a scale bar.  Right: a compact key above a subordinate bar
chart giving the share of each taxon's range that falls inside the reserve
network, referenced against the 30 % policy target.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes


DARK = "#1b5e4b"
OCHRE = "#c08f3e"
PA_EDGE = "#33332f"
RUST = "#a2593c"
OCEAN = "#eaf0f1"
INK = "#1a1a1a"

RICH_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "rich", ["#f6f4ea", "#e0e9d2", "#b6d3ae", "#7fb489", "#448f68",
             "#1b6449", "#0a3a2b"])

TAXA = [
    ("Vascular plants", 41.2),
    ("Birds", 36.5),
    ("Mammals", 28.4),
    ("Reptiles", 22.1),
    ("Amphibians", 18.6),
    ("Freshwater fish", 12.3),
]

PAS = [
    ("PA-1", 66.0, 157.0, 30.0),
    ("PA-2", 152.0, 173.0, 24.0),
    ("PA-3", 188.0, 107.0, 27.0),
    ("PA-4", 98.0, 83.0, 21.0),
    ("PA-5", 134.0, 129.0, 17.0),
]


def panel_tag(fig, ax, label, dx=-0.052, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def blob(cx, cy, r, rng, rough=0.16, n=260):
    """Closed, smoothly irregular outline around (cx, cy)."""
    th = np.linspace(0, 2 * np.pi, n)
    rr = np.ones_like(th)
    for f in (2, 3, 5, 7):
        rr += rough * rng.uniform(0.35, 1.0) / f * \
            np.cos(f * th + rng.uniform(0, 2 * np.pi))
    rr *= r
    return th, rr, cx + rr * np.cos(th), cy + rr * np.sin(th)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260230)

    # ---- landmass ---------------------------------------------------------
    lcx, lcy = 128.0, 126.0
    th, rr, lx, ly = blob(lcx, lcy, 94.0, rng, rough=0.22, n=420)
    order = np.argsort(np.arctan2(ly - lcy, lx - lcx))
    th_sorted = np.arctan2(ly - lcy, lx - lcx)[order]
    rr_sorted = rr[order]

    nx, ny = 300, 280
    gx = np.linspace(0, 262, nx)
    gy = np.linspace(0, 252, ny)
    XX, YY = np.meshgrid(gx, gy)
    ang = np.arctan2(YY - lcy, XX - lcx)
    dist = np.hypot(XX - lcx, YY - lcy)
    bound = np.interp(ang, th_sorted, rr_sorted, period=2 * np.pi)
    land = dist <= bound

    cores = [(70, 155, 1.00, 30, 24), (152, 170, 0.86, 27, 21),
             (188, 109, 0.74, 26, 22), (98, 83, 0.58, 24, 19),
             (132, 129, 0.42, 20, 16)]
    z = np.zeros_like(XX)
    for cx, cy, amp, sx, sy in cores:
        z += amp * np.exp(-(((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))
    z += 0.42 * gaussian_filter(rng.normal(0, 1, XX.shape), 13)
    z = (z - z.min()) / (z.max() - z.min())
    rich = 24.0 + 158.0 * z
    rich_m = np.ma.masked_where(~land, rich)

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.40],
                  height_ratios=[0.66, 1.0], hspace=0.36, wspace=0.52)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 1])

    ax_map.set_facecolor(OCEAN)
    im = ax_map.pcolormesh(gx, gy, rich_m, cmap=RICH_CMAP, vmin=24, vmax=182,
                           shading="gouraud", rasterized=True, zorder=2)
    ax_map.plot(lx, ly, color="#7f8f92", lw=0.9, zorder=4)

    hot = np.percentile(rich[land], 90)
    ax_map.contour(gx, gy, np.ma.filled(rich_m, np.nan), levels=[hot],
                   colors=[OCHRE], linewidths=1.0, linestyles=[(0, (3, 1.8))],
                   zorder=5)

    for name, cx, cy, r in PAS:
        _t, _r, px, py = blob(cx, cy, r, rng, rough=0.20, n=200)
        ax_map.add_patch(Polygon(np.column_stack([px, py]), closed=True,
                                 facecolor="white", alpha=0.16,
                                 edgecolor="none", zorder=6))
        ax_map.add_patch(Polygon(np.column_stack([px, py]), closed=True,
                                 facecolor="none", edgecolor=PA_EDGE,
                                 linewidth=1.0, zorder=7))
        ax_map.text(cx, cy, name, fontsize=6.2, ha="center", va="center",
                    color="#22221f", fontweight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])

    # north arrow
    ax_map.annotate("", xy=(246, 240), xytext=(246, 222),
                    arrowprops=dict(arrowstyle="-|>", color="#33332f",
                                    lw=1.0), zorder=9)
    ax_map.text(246, 216, "N", fontsize=6.8, ha="center", va="top",
                color="#33332f", fontweight="bold", zorder=9)

    # scale bar on a backing plate
    ax_map.add_patch(Rectangle((176, 6), 78, 22, facecolor="white",
                               alpha=0.80, edgecolor="none", zorder=9))
    ax_map.plot([184, 234], [15.5, 15.5], color="#2b2b28", lw=2.6,
                solid_capstyle="butt", zorder=10)
    ax_map.plot([184, 209], [15.5, 15.5], color="white", lw=1.4,
                solid_capstyle="butt", zorder=11)
    ax_map.text(209, 19.0, "50 km", fontsize=6.2, ha="center", va="bottom",
                color="#2b2b28", zorder=11)

    ax_map.set_xlim(0, 262)
    ax_map.set_ylim(0, 252)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xticks(np.arange(0, 262, 60))
    ax_map.set_yticks(np.arange(0, 252, 50))
    ax_map.set_xlabel("Easting (km)")
    ax_map.set_ylabel("Northing (km)")
    ax_map.tick_params(length=2.5)
    for s in ax_map.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#5a5a54")
    ax_map.set_title("Vascular-plant richness, 5 km grid cells",
                     fontsize=8.5, loc="left", pad=5)

    # inset cax so the bar always tracks the equal-aspect map box
    cax = ax_map.inset_axes([1.028, 0.0, 0.028, 1.0])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("Species richness (spp. per cell)", fontsize=6.8,
                 labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#5a5a54")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2, pad=1.5)
    cb.set_ticks([25, 75, 125, 175])
    cb.ax.axhline(hot, color=OCHRE, lw=1.1, ls=(0, (2.5, 1.6)))

    # ---- key --------------------------------------------------------------
    ax_key.axis("off")
    handles = [
        Line2D([], [], color=PA_EDGE, lw=1.0, label="Protected area boundary"),
        Line2D([], [], color=OCHRE, lw=1.0, ls=(0, (3, 1.8)),
               label=f"Hotspot ($\\geq$ {hot:.0f} spp.)"),
        Line2D([], [], color="#7f8f92", lw=0.9, label="Coastline"),
    ]
    lk = ax_key.legend(handles=handles, loc="upper left", fontsize=6.6,
                       frameon=False, handlelength=1.6, labelspacing=0.5,
                       handletextpad=0.6, borderaxespad=0.0,
                       bbox_to_anchor=(-0.03, 1.03))
    lk.set_clip_on(False)
    ax_key.text(-0.03, 0.28,
                "5 reserves, 1 842 km$^2$\n"
                "= 14.6 % of land area\n"
                "1 240 species records",
                transform=ax_key.transAxes, fontsize=6.4, va="top",
                color="#6f6f66", linespacing=1.55)

    # ---- subordinate gap analysis -----------------------------------------
    ax_gap.set_facecolor("#f6f6f2")
    names = [t[0] for t in TAXA]
    vals = np.array([t[1] for t in TAXA])
    cols = [RICH_CMAP(0.82 - 0.10 * i) for i in range(len(TAXA))]
    ypos = np.arange(len(TAXA))[::-1]
    ax_gap.barh(ypos, vals, height=0.62, color=cols, edgecolor="white",
                linewidth=0.6, zorder=3)
    for yy, v in zip(ypos, vals):
        ax_gap.text(v + 1.4, yy, f"{v:.1f}", va="center", ha="left",
                    fontsize=6.1, color="#4a4a44")
    ax_gap.axvline(30.0, color=RUST, lw=1.0, ls=(0, (3.5, 2.0)), zorder=5)
    ax_gap.text(31.0, -0.52, "30 % target", fontsize=6.2, color=RUST,
                ha="left", va="bottom", fontweight="bold")

    ax_gap.set_yticks(ypos)
    ax_gap.set_yticklabels(names, fontsize=6.4)
    ax_gap.set_xlim(0, 54)
    ax_gap.set_ylim(-0.62, len(TAXA) - 0.30)
    ax_gap.set_xticks([0, 20, 40])
    ax_gap.tick_params(labelsize=6.2, length=2.0)
    ax_gap.set_xlabel("Range inside reserves (%)", fontsize=6.8, labelpad=2)
    ax_gap.set_title("Conservation gap by taxon", fontsize=7.4, loc="left",
                     pad=4)
    ax_gap.grid(axis="x", lw=0.45, color="#e4e4de")
    style_axes(ax_gap)

    fig.subplots_adjust(left=0.072, right=0.985, top=0.875, bottom=0.105)

    fig.canvas.draw()
    panel_tag(fig, ax_map, "(a)", dx=-0.056)
    panel_tag(fig, ax_gap, "(b)", dx=-0.056, dy=0.040)

    savefig(fig, "230_biodiversity_hotspot.png")


if __name__ == "__main__":
    main()
