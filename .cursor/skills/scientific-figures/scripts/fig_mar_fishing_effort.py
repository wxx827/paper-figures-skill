"""Where a fleet actually fishes: gridded effort over a shelf and its break.

The map is a 10 km fishing-ground grid carrying vessel-hours per cell.  Three
things are laid over it because effort is only interpretable against them: the
coastline (effort has to stop somewhere), the 200 m isobath (the shelf break
is where the demersal fleet concentrates) and a no-take marine protected area
whose boundary is visible as a hole in the effort field rather than as a
caption.

The two side panels are derived from the very same grid, not invented next to
it.  Monthly effort is the annual total redistributed by a seasonal weight, so
the twelve bars sum to the number quoted on the map, and the depth-band panel
bins every wet cell by its own modelled depth and splits it between three gear
types with a depth-dependent mix - so the statement "trawlers own the shelf,
longliners own the slope" is a measurement of the plotted field.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import (axes_inches, panel_labels_inches, add_scale_bar,
                    halo_text, safe_colorbar)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
LAND, LAND_EDGE = "#e4e2dc", "#9a9a9a"
MPA_COL = "#2e7d5b"
ACCENT = "#b23a48"
GEARS = ["trawler", "longliner", "purse seiner"]
GEAR_COL = {"trawler": "#3b5b92", "longliner": "#c9a227",
            "purse seiner": "#8e6bbf"}
SEASON_COL = "#e69a54"

FW, FH = 10.4, 6.4
MAIN = (0.66, 1.06, 4.40, 4.10)
AUX1 = (6.68, 3.10, 2.72, 2.06)
AUX2 = (6.68, 1.06, 2.72, 1.44)

CELL = 10.0                      # km
NX, NY = 32, 30                  # -> 320 x 300 km domain
MONTHS = ["J", "F", "M", "A", "M", "J",
          "J", "A", "S", "O", "N", "D"]
DEPTH_BANDS = [(0.0, 50.0, "0-50"), (50.0, 150.0, "50-150"),
               (150.0, 400.0, "150-400"), (400.0, 1200.0, ">400")]


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def coast_x(y):
    """Easting (km) of the shoreline at northing ``y``."""
    return (52.0 + 26.0 * np.sin(2.0 * np.pi * y / 340.0)
            + 11.0 * np.cos(2.0 * np.pi * y / 118.0))


def mpa_polygon():
    th = np.linspace(0.0, 2.0 * np.pi, 260)
    r = 46.0 + 7.0 * np.sin(3.0 * th) + 4.0 * np.cos(5.0 * th + 0.7)
    return 158.0 + r * np.cos(th), 214.0 + r * np.sin(th)


def build_grid(rng):
    xe = np.arange(NX + 1) * CELL
    ye = np.arange(NY + 1) * CELL
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)

    d = X - coast_x(Y)                                    # km from shore
    depth = (10.0 + 150.0 * np.clip(d / 120.0, 0.0, 1.0)
             + 700.0 * logistic((d - 170.0) / 16.0)
             + 26.0 * np.sin(2.0 * np.pi * Y / 96.0)
             * np.clip(d / 80.0, 0.0, 1.0))
    depth = depth + rng.normal(0.0, 5.0, depth.shape)
    wet = d > 0.0
    depth = np.where(wet, np.clip(depth, 4.0, None), np.nan)

    grounds = [(126.0, 92.0, 34.0, 1.00), (196.0, 176.0, 26.0, 0.86),
               (108.0, 232.0, 22.0, 0.58), (232.0, 62.0, 24.0, 0.52),
               (168.0, 132.0, 18.0, 0.44)]
    eff = np.zeros_like(X)
    for gx, gy, gs, amp in grounds:
        eff += amp * np.exp(-0.5 * (((X - gx) / gs) ** 2
                                    + ((Y - gy) / (gs * 1.25)) ** 2))
    # shelf-break ribbon: the fleet tracks the 200 m contour
    eff += 0.55 * np.exp(-0.5 * ((depth - 200.0) / 55.0) ** 2)
    eff *= np.exp(-np.clip(d - 20.0, 0.0, None) / 260.0)   # steaming cost
    eff = eff * np.exp(rng.normal(0.0, 0.22, eff.shape))

    px, py = mpa_polygon()
    ang = np.arctan2(Y - 214.0, X - 158.0)
    rad = np.hypot(X - 158.0, Y - 214.0)
    rlim = 46.0 + 7.0 * np.sin(3.0 * ang) + 4.0 * np.cos(5.0 * ang + 0.7)
    inside = rad < rlim
    eff = np.where(inside, eff * 0.06, eff)
    eff = np.where(wet, eff, np.nan)
    eff = eff / np.nanmax(eff) * 1850.0                    # vessel-hours
    return dict(xe=xe, ye=ye, X=X, Y=Y, depth=depth, eff=eff, wet=wet,
                inside=inside, mpa=(px, py), grounds=grounds)


def gear_split(depth):
    """Effort share per gear as a function of depth (m); columns sum to 1."""
    tr = np.exp(-0.5 * ((np.log(depth) - np.log(70.0)) / 0.75) ** 2)
    ll = logistic((depth - 230.0) / 70.0)
    ps = np.exp(-0.5 * ((np.log(depth) - np.log(38.0)) / 0.55) ** 2) * 0.9
    tot = tr + ll + ps
    return tr / tot, ll / tot, ps / tot


def draw_map(ax, g):
    eff = np.ma.masked_invalid(g["eff"])
    im = ax.pcolormesh(g["xe"], g["ye"], eff, cmap=get_cmap(
        "cb_sequential_warm"), shading="flat", vmin=0.0,
        vmax=float(np.nanmax(g["eff"])), zorder=2)

    yy = np.linspace(0.0, NY * CELL, 400)
    ax.fill_betweenx(yy, 0.0, coast_x(yy), facecolor=LAND, edgecolor="none",
                     zorder=3)
    ax.plot(coast_x(yy), yy, color=LAND_EDGE, lw=1.0, zorder=4)
    ax.text(6.0, 292.0, "land", fontsize=6.6, color=MUTED, ha="left",
            va="top", zorder=5)

    cs = ax.contour(g["X"], g["Y"], np.nan_to_num(g["depth"], nan=0.0),
                    levels=[200.0], colors="#3d6b8e", linewidths=0.9,
                    linestyles=[(0, (5.0, 2.4))], zorder=5)
    halo_text(ax, 192.0, 30.0, "200 m", ha="left", va="center", fontsize=6.2,
              color="#3d6b8e", halo="white", halo_lw=2.2, zorder=6)

    px, py = g["mpa"]
    ax.plot(px, py, color=MPA_COL, lw=1.6, zorder=6)
    ax.plot(px, py, color="white", lw=3.0, zorder=5, alpha=0.75)
    halo_text(ax, 158.0, 264.0, "no-take MPA", ha="center", va="bottom",
              fontsize=6.6, color=MPA_COL, halo="white", halo_lw=2.2,
              zorder=7)

    # Label the two busiest cells the field actually contains, not the
    # hotspot centres that were used to build it.
    eff = np.where(np.isfinite(g["eff"]), g["eff"], -1.0)
    j1, i1 = np.unravel_index(int(np.argmax(eff)), eff.shape)
    far = (np.hypot(g["X"] - g["X"][j1, i1], g["Y"] - g["Y"][j1, i1]) > 80.0)
    j2, i2 = np.unravel_index(int(np.argmax(np.where(far, eff, -1.0))),
                              eff.shape)
    picks = [((j1, i1), "busiest ground", (16.0, -10.0), "left", "top"),
             ((j2, i2), "secondary ground", (0.0, -22.0), "center", "top")]
    for (j, i), name, off, ha, va in picks:
        gx, gy = g["X"][j, i], g["Y"][j, i]
        halo_text(ax, gx + off[0], gy + off[1],
                  "{:s}\n{:.0f} vessel-hours".format(name, g["eff"][j, i]),
                  ha=ha, va=va, fontsize=6.2, color=INK, halo="white",
                  halo_lw=2.2, zorder=8, linespacing=1.25)
        ax.plot([gx], [gy], marker="o", ms=4.2, mfc="none", mec=ACCENT,
                mew=1.1, zorder=8)

    add_scale_bar(ax, 50.0, "50 km", loc=(0.030, 0.035), height=0.016,
                  n_seg=2, color="#3a3a3a", alt="white", fontsize=6.2,
                  text_color=INK, halo="white")

    ax.set_xlim(0.0, NX * CELL)
    ax.set_ylim(0.0, NY * CELL)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Annual effort on a 10 km fishing-ground grid",
                 fontsize=8.5, fontweight="bold")
    return im


def draw_season(ax, total):
    w = np.array([0.42, 0.38, 0.58, 0.86, 1.18, 1.44,
                  1.52, 1.36, 1.10, 0.82, 0.56, 0.40])
    share = w / w.sum()
    monthly = share * total / 1000.0
    ax.bar(np.arange(12), monthly, width=0.72, color=SEASON_COL,
           edgecolor="#a04c18", linewidth=0.5, zorder=3)
    k = int(np.argmax(monthly))
    ax.bar([k], [monthly[k]], width=0.72, color=ACCENT, edgecolor="#8c2f2f",
           linewidth=0.5, zorder=4)
    ax.text(k, monthly[k] + 0.06 * monthly.max(),
            "peak {:.0%} of year".format(share[k]), ha="center", va="bottom",
            fontsize=6.4, color=ACCENT, zorder=5)
    q3 = share[5:8].sum()
    ax.text(0.035, 0.94, "Jun-Aug carries {:.0%} of annual effort"
            .format(q3), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color=SUB)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTHS, fontsize=6.8)
    ax.set_ylim(0.0, monthly.max() * 1.30)
    ax.set_xlabel("month", fontsize=7.8)
    ax.set_ylabel("effort (10$^3$ vessel-hours)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Seasonal distribution of effort", fontsize=8.5,
                 fontweight="bold")
    return share


def draw_depth_bands(ax, g):
    depth = g["depth"]
    eff = g["eff"]
    ok = np.isfinite(depth) & np.isfinite(eff)
    tr, ll, ps = gear_split(np.where(ok, depth, 1.0))
    labels, mat = [], []
    for lo, hi, name in DEPTH_BANDS:
        m = ok & (depth >= lo) & (depth < hi)
        e = eff[m]
        mat.append([float(np.sum(e * tr[m])), float(np.sum(e * ll[m])),
                    float(np.sum(e * ps[m]))])
        labels.append(name)
    mat = np.array(mat)
    frac = mat / mat.sum()
    ypos = np.arange(len(labels))[::-1]                # shallow band on top
    left = np.zeros(len(labels))
    for j, gear in enumerate(GEARS):
        ax.barh(ypos, frac[:, j] * 100.0, left=left * 100.0, height=0.62,
                color=GEAR_COL[gear], edgecolor="white", linewidth=0.6,
                zorder=3)
        left = left + frac[:, j]
    for k, y in enumerate(ypos):
        ax.text(frac[k].sum() * 100.0 + 1.2, y,
                "{:.0f}%".format(frac[k].sum() * 100.0), ha="left",
                va="center", fontsize=6.2, color=SUB, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.62, len(labels) - 0.38)
    ax.set_xlim(0.0, frac.sum(axis=1).max() * 100.0 * 1.22)
    ax.set_xlabel("share of fleet effort (%)", fontsize=7.8)
    ax.set_ylabel("depth band (m)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Working depth by gear type", fontsize=8.5,
                 fontweight="bold")
    return frac


def main():
    set_pub_style()
    rng = np.random.default_rng(691)
    g = build_grid(rng)
    total = float(np.nansum(g["eff"]))

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX1)
    ax_d = axes_inches(fig, AUX2)

    im = draw_map(ax_m, g)
    safe_colorbar(fig, im, ax_m,
                  label="effort (vessel-hours per 100 km$^2$)",
                  width=0.11, pad=0.09)
    share = draw_season(ax_s, total)
    frac = draw_depth_bands(ax_d, g)

    handles = [Patch(facecolor=GEAR_COL[k], edgecolor="white", label=k)
               for k in GEARS]

    panel_labels_inches(fig, [ax_m, ax_s, ax_d], dx=-0.44, dy=0.20)

    inside_share = float(np.nansum(g["eff"][g["inside"]]) / total)
    shelf = float(np.nansum(g["eff"][np.isfinite(g["depth"])
                                     & (g["depth"] < 200.0)]) / total)

    print("total={:.0f} vh  inside={:.4f}  shelf={:.3f}".format(
        total, inside_share, shelf))
    savefig(fig, "691_mar_fishing_effort.png")


if __name__ == "__main__":
    main()
