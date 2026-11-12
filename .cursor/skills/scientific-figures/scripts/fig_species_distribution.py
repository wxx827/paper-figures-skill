"""Species distribution model output.

Left: a continuous habitat-suitability surface with the binarisation contour
and observed occurrence records overlaid, plus a scale bar and a horizontal
colorbar.  Right: a 2x2 block of marginal response curves (all four locked to
one 0-1 suitability axis) sitting above a permutation variable-importance bar
chart.  Both right-hand blocks are deliberately small relative to the map.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes


DARK = "#1b5e4b"
MID = "#5aa17f"
OCHRE = "#c08f3e"
OCC = "#a2593c"
INK = "#1a1a1a"

SUIT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "suit", ["#f4f2ea", "#dce7d5", "#a9cdb4", "#6aab8b", "#2f7f5f", "#12513c"])

VARS = [
    ("Mean ann. temp.", 24.6),
    ("Precipitation", 21.3),
    ("Canopy cover", 18.9),
    ("Dist. to water", 14.2),
    ("Soil pH", 11.5),
    ("Slope", 9.5),
]


def panel_tag(fig, ax, label, dx=-0.050, dy=0.016, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def suitability_field(rng, nx=240, ny=190):
    x = np.linspace(0, 120, nx)
    y = np.linspace(0, 95, ny)
    xx, yy = np.meshgrid(x, y)
    cores = [(32, 62, 1.00, 22, 15), (78, 40, 0.88, 26, 18),
             (100, 74, 0.62, 15, 12), (18, 22, 0.55, 17, 13),
             (60, 80, 0.45, 13, 10)]
    z = np.zeros_like(xx)
    for cx, cy, amp, sx, sy in cores:
        z += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    z += 0.55 * gaussian_filter(rng.normal(0, 1, xx.shape), 11)
    z -= 0.30 * np.exp(-(((xx - 55) / 9) ** 2))          # unsuitable corridor
    z = (z - z.min()) / (z.max() - z.min())
    return x, y, z


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260228)

    x, y, z = suitability_field(rng)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.64],
                  height_ratios=[1.0, 0.50], wspace=0.34, hspace=0.62)
    ax_map = fig.add_subplot(gs[:, 0])
    inner = gs[0, 1].subgridspec(2, 2, hspace=0.60, wspace=0.16)
    resp_axes = [fig.add_subplot(inner[i // 2, i % 2]) for i in range(4)]
    ax_imp = fig.add_subplot(gs[1, 1])

    # ---- suitability map -------------------------------------------------
    im = ax_map.pcolormesh(x, y, z, cmap=SUIT_CMAP, vmin=0, vmax=1,
                           shading="gouraud", rasterized=True, zorder=1)
    thr = 0.42
    ax_map.contour(x, y, z, levels=[thr], colors=[OCHRE], linewidths=1.0,
                   linestyles=[(0, (4, 2))], zorder=3)

    p = np.clip(z, 0, None) ** 4
    p = p / p.sum()
    idx = rng.choice(p.size, size=92, replace=False, p=p.ravel())
    xy = np.array(np.unravel_index(idx, z.shape))
    px = x[xy[1]] + rng.normal(0, 0.4, idx.size)
    py = y[xy[0]] + rng.normal(0, 0.4, idx.size)
    ax_map.scatter(px, py, s=9, color=OCC, edgecolor="white", linewidth=0.4,
                   alpha=0.95, zorder=4)

    ax_map.set_xlim(0, 120)
    ax_map.set_ylim(0, 95)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("Easting (km)")
    ax_map.set_ylabel("Northing (km)")
    ax_map.set_xticks(np.arange(0, 120, 30))
    ax_map.set_yticks(np.arange(0, 96, 20))
    ax_map.tick_params(length=2.5)
    for s in ax_map.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#5a5a54")
    ax_map.set_title("Maxent habitat suitability, 1 km grid",
                     fontsize=8.5, loc="left", pad=5)

    # scale bar on a translucent backing plate
    ax_map.add_patch(Rectangle((84.5, 2.0), 33.0, 13.0, facecolor="white",
                               alpha=0.78, edgecolor="none", zorder=5))
    ax_map.plot([88, 108], [5.6, 5.6], color="#2b2b28", lw=2.4,
                solid_capstyle="butt", zorder=6)
    ax_map.plot([88, 98], [5.6, 5.6], color="white", lw=1.3,
                solid_capstyle="butt", zorder=7)
    ax_map.text(98, 8.0, "20 km", fontsize=6.2, ha="center", va="bottom",
                color="#2b2b28", zorder=7)

    key = [
        Line2D([], [], color=OCHRE, lw=1.0, ls=(0, (4, 2)),
               label=f"threshold {thr:.2f}"),
        Line2D([], [], color="none", marker="o", ms=3.2, mfc=OCC,
               mec="white", mew=0.4, label="occurrence record"),
    ]
    lk = ax_map.legend(handles=key, loc="lower left", fontsize=6.2,
                       frameon=True, facecolor="white", edgecolor="#d3d3cc",
                       framealpha=0.9, handlelength=1.5, borderpad=0.4,
                       labelspacing=0.3, borderaxespad=0.45)
    lk.get_frame().set_linewidth(0.6)
    lk.set_clip_on(False)

    ax_map.text(0.022, 0.975,
                "AUC = $\\bf{0.91}$ $\\pm$ 0.02\nTSS = 0.74\n"
                "$n$ = 92 occurrences",
                transform=ax_map.transAxes, va="top", ha="left",
                fontsize=6.4, linespacing=1.35, color="#33332f", zorder=8,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#d3d3cc", linewidth=0.6, alpha=0.93))

    cb = fig.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.105,
                      fraction=0.046, aspect=34)
    cb.set_label("Habitat suitability (0-1)", fontsize=7, labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#5a5a54")
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.2, pad=1.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.axvline(thr, color=OCHRE, lw=1.1, ls=(0, (2.5, 1.6)))

    # ---- marginal response curves ----------------------------------------
    def gauss(t, mu, sd):
        return np.exp(-0.5 * ((t - mu) / sd) ** 2)

    curves = [
        ("Mean annual temp. ($^{\\circ}$C)", np.linspace(2, 22, 160),
         lambda t: 0.10 + 0.85 * gauss(t, 12.5, 3.6)),
        ("Precipitation (mm)", np.linspace(400, 1600, 160),
         lambda t: 0.08 + 0.82 / (1 + np.exp(-(t - 880) / 130.0))),
        ("Canopy cover (%)", np.linspace(0, 100, 160),
         lambda t: 0.07 + 0.86 * gauss(t, 72.0, 21.0)),
        ("Distance to water (km)", np.linspace(0, 12, 160),
         lambda t: 0.06 + 0.84 * np.exp(-t / 3.4)),
    ]

    for k, (ax, (lab, tt, fn)) in enumerate(zip(resp_axes, curves)):
        ax.set_facecolor("#fbfbf8")
        vals = fn(tt)
        band = 0.055 + 0.035 * np.abs(np.gradient(vals)) / \
            (np.abs(np.gradient(vals)).max() + 1e-9)
        ax.fill_between(tt, vals - band, vals + band, color=MID, alpha=0.22,
                        lw=0)
        ax.plot(tt, vals, color=DARK, lw=1.3)
        ax.axhline(thr, color=OCHRE, lw=0.7, ls=(0, (2.5, 1.8)), alpha=0.8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(tt[0], tt[-1])
        ax.set_xlabel(lab, fontsize=6.3, labelpad=1.5)
        ax.tick_params(labelsize=6.0, length=2.0, pad=1.5)
        if k % 2 == 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Suit.", fontsize=6.3, labelpad=1.5)
        ax.grid(axis="y", lw=0.45, color="#ebebe6")
        style_axes(ax)

    resp_axes[1].set_xticks([400, 800, 1200, 1600])
    resp_axes[2].set_xticks([0, 25, 50, 75, 100])

    # ---- variable importance ---------------------------------------------
    ax_imp.set_facecolor("#f6f6f2")
    names = [v[0] for v in VARS]
    imp = np.array([v[1] for v in VARS])
    cols = [SUIT_CMAP(0.88 - 0.11 * i) for i in range(len(VARS))]
    ypos = np.arange(len(VARS))[::-1]
    ax_imp.barh(ypos, imp, height=0.62, color=cols, edgecolor="white",
                linewidth=0.6, zorder=3)
    for yy, v in zip(ypos, imp):
        ax_imp.text(v + 0.7, yy, f"{v:.1f}", va="center", ha="left",
                    fontsize=6.0, color="#4a4a44")
    ax_imp.set_yticks(ypos)
    ax_imp.set_yticklabels(names, fontsize=6.3)
    ax_imp.set_xlim(0, 30)
    ax_imp.set_ylim(-0.62, len(VARS) - 0.38)
    ax_imp.set_xticks([0, 10, 20, 30])
    ax_imp.tick_params(labelsize=6.0, length=2.0)
    ax_imp.set_xlabel("Permutation importance (%)", fontsize=6.6,
                      labelpad=1.5)
    ax_imp.grid(axis="x", lw=0.45, color="#e4e4de")
    style_axes(ax_imp)

    fig.subplots_adjust(left=0.072, right=0.985, top=0.875, bottom=0.115)

    fig.canvas.draw()
    panel_tag(fig, ax_map, "(a)", dx=-0.056)
    b0 = resp_axes[0].get_position()
    fig.text(b0.x0 - 0.040, b0.y1 + 0.052, "(b)", fontsize=10,
             fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(b0.x0 + 0.028, b0.y1 + 0.055, "Marginal response curves",
             fontsize=7.4, ha="left", va="bottom", color=INK)
    panel_tag(fig, ax_imp, "(c)", dx=-0.040, dy=0.020)

    savefig(fig, "228_species_distribution.png")


if __name__ == "__main__":
    main()
