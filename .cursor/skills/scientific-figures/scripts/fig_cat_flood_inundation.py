"""Figure 724 - Fluvial inundation depth for a 100-year flood.

A synthetic valley is built from a downstream bank slope, a narrow channel
incision along a meandering thalweg, a broad floodplain swale and a handful of
terraces.  Flooding is a horizontal water surface referenced to the bank
profile: depth is the water surface minus the ground, clipped at zero, with
the channel itself masked out so the map shows overbank water only.

Panel (a) is that depth field at the 100-year stage, with the terrain contours
showing through on dry ground and every building point tested against the same
grid.  Panel (b) is the depth-damage relationship applied to those depths and
panel (c) re-runs the identical model at eight return periods, so the marked
point on panel (c) is the area quoted on panel (a).
"""
import textwrap
import warnings

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, savefig, get_palette, lighten, darken,
                      COLORS)
from figkit import (add_scale_bar, axes_inches, halo_text, panel_label_inches,
                    safe_colorbar)

FW, FH = 10.4, 6.4
MAIN_X, MAIN_W, MAIN_TOP = 0.62, 5.30, 5.50
AUX_X, AUX_W, AUX_GAP = 7.05, 2.95, 0.66

X1, Y1 = 12.0, 10.64                # map extent, km
NX, NY = 300, 266
DESIGN_T = 100.0                    # return period drawn on the map, years
PERIODS = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
DEPTH_LEVELS = [0.10, 0.30, 0.60, 1.00, 1.50, 2.20, 3.20]
CHANNEL_HALF = 0.22                 # masked channel half-width, km
DEEP = 1.00                         # "deep water" threshold, m

ACCENT = "#b23a48"
BLUES = get_palette("cb_sequential_blue", 8)[1:7]
RIVER = "#134a76"

# Terraces sit inside the floodplain band, so they show up as dry islands.
TERRACES = [(2.2, 7.55, 0.85, 0.70, 2.3), (5.3, 4.65, 0.95, 0.75, 2.6),
            (8.3, 6.45, 0.80, 0.70, 2.0), (10.5, 2.15, 0.90, 0.80, 2.4),
            (6.9, 6.40, 0.70, 0.60, 1.7)]

OCCUPANCY = [
    ("Residential, 1 storey", 0.62, 0.86),
    ("Residential, 2 storey", 0.44, 0.74),
    ("Commercial", 0.34, 0.94),
]


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def thalweg(x):
    """River centreline northing for a given easting, km."""
    return 5.15 + 1.45 * np.sin(x / 2.35) + 0.62 * np.sin(x / 1.05 + 0.7)


def bank_level(x):
    """Floodplain bank elevation, m above datum."""
    return 42.0 - 1.15 * x


def terrain(mx, my):
    """Ground elevation, m above datum.

    The valley walls are what bound the flood: ground rises away from the
    thalweg, so a given stage wets a band rather than the whole map.
    """
    d = np.abs(my - thalweg(mx))
    z = bank_level(mx)
    z += 5.00 * (1.0 - np.exp(-(d / 2.30) ** 2))      # valley walls
    z -= 0.90 * np.exp(-(d / 1.10) ** 2)              # floodplain swale
    z -= 6.00 * np.exp(-(d / 0.22) ** 2)              # channel incision
    for tx, ty, sx, sy, amp in TERRACES:
        z += amp * np.exp(-(((mx - tx) / sx) ** 2 + ((my - ty) / sy) ** 2))
    z += 0.45 * np.sin(mx * 1.9) * np.cos(my * 1.55)
    z += 0.28 * np.cos(mx * 3.1 + my * 2.2)
    return z


def stage(period):
    """Overbank stage above the bank profile for a return period, m."""
    return 0.18 + 0.44 * np.log(period)


def depth_field(mx, my, z, period):
    """Overbank inundation depth, m; the channel itself is masked to NaN."""
    d = np.clip(bank_level(mx) + stage(period) - z, 0.0, None)
    d[np.abs(my - thalweg(mx)) < CHANNEL_HALF] = np.nan
    d[d < DEPTH_LEVELS[0]] = np.nan
    return d


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7240)

    gx = np.linspace(0.0, X1, NX)
    gy = np.linspace(0.0, Y1, NY)
    mx, my = np.meshgrid(gx, gy)
    z = terrain(mx, my)
    cell = (X1 / (NX - 1)) * (Y1 / (NY - 1))          # km^2

    dep = depth_field(mx, my, z, DESIGN_T)
    wet = np.isfinite(dep)
    area = float(wet.sum() * cell)
    area_deep = float((wet & (dep >= DEEP)).sum() * cell)
    dmax = float(np.nanmax(dep))
    iy, ix = np.unravel_index(np.nanargmax(np.where(wet, dep, -1.0)),
                              dep.shape)

    areas = np.array([np.isfinite(depth_field(mx, my, z, T)).sum() * cell
                      for T in PERIODS])
    areas_deep = np.array([
        np.nansum(depth_field(mx, my, z, T) >= DEEP) * cell for T in PERIODS])

    # ---- buildings ---------------------------------------------------------
    # settlement follows the valley, so most buildings sit near the river
    n_near, n_far = 150, 90
    bx = np.concatenate([rng.uniform(0.3, X1 - 0.3, n_near),
                         rng.uniform(0.3, X1 - 0.3, n_far)])
    by = np.concatenate([thalweg(bx[:n_near]) + rng.normal(0.0, 1.55, n_near),
                         rng.uniform(0.3, Y1 - 0.3, n_far)])
    keep = ((np.abs(by - thalweg(bx)) > CHANNEL_HALF + 0.06)
            & (by > 0.3) & (by < Y1 - 0.3))
    bx, by = bx[keep], by[keep]
    bd = np.clip(bank_level(bx) + stage(DESIGN_T) - terrain(bx, by), 0.0, None)
    hit = bd >= DEPTH_LEVELS[0]

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    main_h = MAIN_W * Y1 / X1
    main_y = MAIN_TOP - main_h
    ax = axes_inches(fig, (MAIN_X, main_y, MAIN_W, main_h))
    aux_h = (main_h - AUX_GAP) / 2.0
    ax_d = axes_inches(fig, (AUX_X, main_y + aux_h + AUX_GAP, AUX_W, aux_h))
    ax_a = axes_inches(fig, (AUX_X, main_y, AUX_W, aux_h))

    # ---- (a) inundation depth ----------------------------------------------
    ax.set_facecolor("#f5f5f5")           # dry land, so the map is not blank
    ct = ax.contour(mx, my, z, levels=np.arange(20.0, 50.0, 2.0),
                    colors="#b3b3b3", linewidths=0.45, zorder=2)
    ax.clabel(ct, ct.levels[::2], fmt="%.0f", fontsize=6.0, inline=True,
              inline_spacing=3, colors="#8a8a8a")

    cmap = mpl.colors.ListedColormap(BLUES)
    cmap.set_over(darken(RIVER, 0.30))
    norm = mpl.colors.BoundaryNorm(DEPTH_LEVELS, cmap.N)
    cf = ax.contourf(mx, my, dep, levels=DEPTH_LEVELS, cmap=cmap, norm=norm,
                     extend="max", zorder=3)
    rasterize(cf)
    cf.set_edgecolor("face")

    rx = np.linspace(0.0, X1, 400)
    ax.plot(rx, thalweg(rx), color=RIVER, lw=2.2, solid_capstyle="round",
            zorder=5)

    ax.scatter(bx[~hit], by[~hit], s=7.0, marker="s", facecolor="white",
               edgecolor="#777777", linewidth=0.5, zorder=6)
    ax.scatter(bx[hit], by[hit], s=9.0, marker="s", facecolor=ACCENT,
               edgecolor="white", linewidth=0.45, zorder=7)

    ax.plot([gx[ix]], [gy[iy]], marker="v", ms=5.0, mfc="white",
            mec=darken(RIVER, 0.30), mew=1.2, zorder=8)
    halo_text(ax, float(np.clip(gx[ix], 0.11 * X1, 0.89 * X1)),
              gy[iy] + 0.24,
              f"max depth {dmax:.2f} m", fontsize=6.4, ha="center",
              va="bottom", color=darken(RIVER, 0.30), halo="white",
              halo_lw=2.4, zorder=9)

    ax.set_xlim(0.0, X1)
    ax.set_ylim(0.0, Y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title(f"Inundation depth at the {DESIGN_T:.0f}-year flood",
                 fontsize=8.8, fontweight="bold", pad=4.0)
    add_scale_bar(ax, 2.0, "2 km", loc=(0.035, 0.045), height=0.020,
                  fontsize=6.2, n_seg=2)
    safe_colorbar(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                  label="inundation depth (m)", width=0.12, pad=0.08,
                  labelsize=6.8, ticks=DEPTH_LEVELS, extend="max")
    ax.legend(handles=[
        Line2D([0], [0], color=RIVER, lw=2.2, label="river channel"),
        Line2D([0], [0], color="#b3b3b3", lw=0.45,
               label="ground elevation (m)"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=3.2,
               markerfacecolor=ACCENT, markeredgecolor="white",
               markeredgewidth=0.45,
               label=f"building flooded ({int(hit.sum())})"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=3.0,
               markerfacecolor="white", markeredgecolor="#777777",
               markeredgewidth=0.5,
               label=f"building dry ({int((~hit).sum())})"),
    ], loc="upper left", frameon=True, framealpha=0.92, facecolor="white",
        edgecolor="#cccccc", borderpad=0.42, fontsize=6.3, handlelength=1.8,
        labelspacing=0.32)
    halo_text(ax, 0.972, 0.045,
              f"inundated area {area:.1f} km$^2$, of which "
              f"{area_deep:.1f} km$^2$ deeper than {DEEP:.1f} m",
              transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
              color="#1a1a1a", halo="white", halo_lw=2.6, zorder=9)

    # ---- (b) depth-damage --------------------------------------------------
    dd = np.linspace(0.0, DEPTH_LEVELS[-1], 220)
    for lo, hi, col in zip([0.0] + DEPTH_LEVELS[:-1], DEPTH_LEVELS, BLUES):
        ax_d.axvspan(lo, hi, facecolor=col, alpha=0.20, lw=0.0, zorder=0)
    dcols = [darken(ACCENT, 0.34), ACCENT, lighten(ACCENT, 0.36)]
    mean_depth = float(np.nanmean(dep))
    for (name, k, cap), col in zip(OCCUPANCY, dcols):
        curve = cap * (1.0 - np.exp(-k * dd))
        ax_d.plot(dd, curve, color=col, lw=1.6, zorder=5, label=name)
        v = cap * (1.0 - np.exp(-k * mean_depth))
        ax_d.plot([mean_depth], [v], marker="o", ms=3.4, mfc="white",
                  mec=col, mew=1.1, zorder=6)
    ax_d.axvline(mean_depth, color="#333333", lw=0.8, ls=(0, (2.6, 1.8)),
                 zorder=4)
    right = mean_depth < 0.5 * DEPTH_LEVELS[-1]
    ax_d.text(mean_depth + (0.07 if right else -0.07), 0.03,
              f"mean flooded depth {mean_depth:.2f} m", fontsize=6.2,
              color="#333333", ha="left" if right else "right", va="bottom")
    ax_d.set_xlim(0.0, DEPTH_LEVELS[-1])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlabel("water depth (m)", fontsize=7.4)
    ax_d.set_ylabel("damage ratio (-)", fontsize=7.2)
    ax_d.tick_params(labelsize=6.6)
    ax_d.grid(axis="y", lw=0.5, alpha=0.42)
    ax_d.set_title("Depth-damage relationship", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_d.legend(loc="lower right", frameon=False, fontsize=6.3,
                handlelength=1.7, labelspacing=0.26)
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    # ---- (c) area against return period ------------------------------------
    ax_a.plot(PERIODS, areas, color=RIVER, lw=1.7, marker="o", ms=3.2,
              mfc="white", mew=1.0, zorder=5, label="total inundated")
    ax_a.plot(PERIODS, areas_deep, color=BLUES[2], lw=1.7,
              ls=(0, (4.4, 2.0)), marker="o", ms=3.2, mfc="white", mew=1.0,
              zorder=5, label=f"deeper than {DEEP:.1f} m")
    ax_a.fill_between(PERIODS, areas_deep, areas, facecolor=BLUES[1],
                      alpha=0.22, lw=0.0, zorder=2)
    k = PERIODS.index(DESIGN_T)
    ax_a.plot([DESIGN_T], [areas[k]], marker="o", ms=5.2, mfc="white",
              mec=ACCENT, mew=1.4, zorder=7)
    ax_a.annotate(f"{areas[k]:.1f} km$^2$ on panel (a)",
                  xy=(DESIGN_T, areas[k]), xytext=(-6.0, 8.0),
                  textcoords="offset points", fontsize=6.3, color=ACCENT,
                  ha="right", va="bottom")
    ax_a.set_xscale("log")
    ax_a.set_xlim(1.7, 620.0)
    ax_a.set_ylim(0.0, areas.max() * 1.24)
    ax_a.set_xticks(PERIODS)
    ax_a.set_xticklabels([f"{T:.0f}" for T in PERIODS])
    ax_a.minorticks_off()
    ax_a.set_xlabel("return period (yr)", fontsize=7.4)
    ax_a.set_ylabel("inundated area (km$^2$)", fontsize=7.2)
    ax_a.tick_params(labelsize=6.6)
    ax_a.grid(axis="y", lw=0.5, alpha=0.42)
    ax_a.set_title("Inundated area against return period", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.3,
                handlelength=1.9, labelspacing=0.26)
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)

    ax.apply_aspect()
    for a, lab in ((ax, "(a)"), (ax_d, "(b)"), (ax_a, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.30 if a is ax else -0.52,
                           dy=0.09)

    growth = areas[-1] / areas[0]

    savefig(fig, "724_cat_flood_inundation.png")


if __name__ == "__main__":
    main()
