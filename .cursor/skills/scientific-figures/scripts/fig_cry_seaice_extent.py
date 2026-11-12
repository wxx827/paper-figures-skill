"""Figure 613 - Arctic sea-ice concentration at the September minimum.

The polar panel is drawn on a Lambert azimuthal *equal-area* disk, so counting
grid cells is a legitimate way to measure area: the extent quoted on the map
is the sum of the cells whose concentration reaches the 15 % threshold, scaled
by the true area of the cap north of 45 deg N.

That property is what ties the three panels together.  The seasonal-cycle
family in panel (b) is generated first; the ice-edge latitude offset of each
mapped field is then solved by bisection so that the field reproduces the
September value it is supposed to show - the current year for the solid edge,
the 1991-2020 median for the dashed one.  Nothing on the map is a free
parameter, and no number in the figure is typed in.
"""
import textwrap

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_label_inches, safe_colorbar, shared_limits

R_EARTH = 6371.0                     # km
LAT_EDGE = 50.0                      # latitude at the rim of the disk
LAT_CIRCLES = (80.0, 70.0, 60.0)
YEARS = np.arange(1979, 2025)
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
SEPT = 9.0                           # month index used for the minimum
MARCH = 3.0                          # month index used for the maximum
Y_ANOM = 2012                        # the one strongly anomalous summer

ICE_LINE = "#b23a48"                 # house red, the one emphasis: this year
MED_LINE = "#333333"                 # 1991-2020 median edge
LAND = "#e3e0da"
LAND_EDGE = "#b3b3b3"


def seasonal_cycle(t, year, rng_state):
    """Extent in 10^6 km2 for fractional month ``t`` of ``year``."""
    base = 11.05 + 4.45 * np.cos(2 * np.pi * (t - 2.2) / 12.0)
    summer = np.clip(np.cos(2 * np.pi * (t - SEPT) / 12.0), 0.0, None)
    decline = (0.020 + 0.032 * summer) * (year - YEARS[0])
    return base - decline + rng_state * (0.45 + 0.55 * summer)


def polar_grid(n=420):
    """Equal-area disk: returns x, y, latitude, longitude and the cell area."""
    g = np.linspace(-1.0, 1.0, n)
    xg, yg = np.meshgrid(g, g)
    r = np.hypot(xg, yg)
    s = np.sin(np.deg2rad(90.0 - LAT_EDGE) / 2.0)
    colat = 2.0 * np.degrees(np.arcsin(np.clip(r * s, 0.0, 1.0)))
    lat = 90.0 - colat
    lon = np.degrees(np.arctan2(xg, -yg))
    inside = r <= 1.0
    cap = 2 * np.pi * R_EARTH ** 2 * (1.0 - np.cos(np.deg2rad(90.0 - LAT_EDGE)))
    cell = (2.0 / (n - 1)) ** 2 * cap / np.pi          # km2 per grid cell
    return xg, yg, r, lat, lon, inside, cell


def coast_lat(lon):
    """Stylised circum-Arctic coastline, open at the Atlantic and Bering gates."""
    l = np.deg2rad(lon)
    base = (59.5 + 2.6 * np.sin(l + 0.7) + 1.8 * np.sin(3 * l + 2.1)
            + 1.1 * np.sin(5 * l + 0.3))
    d_atl = ((lon + 180.0) % 360.0) - 180.0
    d_pac = ((lon - 190.0 + 180.0) % 360.0) - 180.0
    return (base - 21.0 * np.exp(-0.5 * (d_atl / 22.0) ** 2)
            - 19.0 * np.exp(-0.5 * (d_pac / 7.0) ** 2))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6130)

    # ---- seasonal-cycle family --------------------------------------------
    t = np.linspace(0.5, 12.5, 145)
    wobble = rng.normal(0.0, 0.34, YEARS.size)
    wobble[YEARS == Y_ANOM] -= 1.15          # the one exceptional melt season
    cycles = np.array([seasonal_cycle(t, y, w)
                       for y, w in zip(YEARS, wobble)])
    sept = np.array([seasonal_cycle(SEPT, y, w)
                     for y, w in zip(YEARS, wobble)])
    march = np.array([seasonal_cycle(MARCH, y, w)
                      for y, w in zip(YEARS, wobble)])
    ref = (YEARS >= 1991) & (YEARS <= 2020)
    ext_med = float(np.median(sept[ref]))
    ext_cur = float(sept[-1])
    i_rec = int(np.argmin(sept))
    slope = float(np.polyfit(YEARS, sept, 1)[0]) * 10.0     # per decade
    slope_m = float(np.polyfit(YEARS, march, 1)[0]) * 10.0
    fit = np.poly1d(np.polyfit(YEARS, sept, 1))
    fit_m = np.poly1d(np.polyfit(YEARS, march, 1))

    # ---- polar field, tuned to reproduce those two numbers -----------------
    xg, yg, r, lat, lon, inside, cell = polar_grid()
    land = inside & (lat < coast_lat(lon))
    tex = gaussian_filter(rng.normal(size=lat.shape), 7.0)
    tex = 0.86 + 0.30 * tex / (tex.std() or 1.0)

    def conc_field(shift):
        edge = (66.0 + shift + 5.5 * np.sin(np.deg2rad(lon) + 1.1)
                + 3.0 * np.sin(np.deg2rad(3 * lon) + 0.4))
        c = np.clip((lat - edge) / 5.0, 0.0, 1.0) * tex
        c = np.clip(c, 0.0, 1.0)
        return np.where(inside & ~land, c, np.nan)

    def extent_of(c):
        return float(np.nansum(c >= 0.15)) * cell / 1.0e6   # 10^6 km2

    def solve(target, lo=-14.0, hi=14.0):
        for _ in range(38):
            mid = 0.5 * (lo + hi)
            if extent_of(conc_field(mid)) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    c_cur = conc_field(solve(ext_cur))
    c_med = conc_field(solve(ext_med))
    ext_map = extent_of(c_cur)

    # ---- layout -------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.020, right=0.960, top=0.888, bottom=0.118,
                          wspace=0.20, hspace=0.46)
    ax = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    # ---- (a) the ice pack ---------------------------------------------------
    # Open water reads as a mid blue rather than the near-black end of the
    # ramp: at the September minimum most of the disk is ice-free, and a very
    # dark ocean would make the panel far heavier than its neighbours.
    ice_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ice", get_palette("cb_sequential_blue", 9, reverse=True)[2:])
    pm = ax.pcolormesh(xg, yg, 100.0 * c_cur, cmap=ice_cmap,
                       vmin=0.0, vmax=100.0, shading="auto", rasterized=True,
                       zorder=2)
    ax.contourf(xg, yg, np.where(inside, land.astype(float), 0.0),
                levels=[0.5, 1.5], colors=[LAND], zorder=3)
    ax.contour(xg, yg, np.where(inside, land.astype(float), 0.0),
               levels=[0.5], colors=[LAND_EDGE], linewidths=0.6, zorder=4)

    for lat_c in LAT_CIRCLES:
        rr = (np.sin(np.deg2rad(90.0 - lat_c) / 2.0)
              / np.sin(np.deg2rad(90.0 - LAT_EDGE) / 2.0))
        th = np.linspace(0, 2 * np.pi, 361)
        ax.plot(rr * np.sin(th), rr * np.cos(th), color="#8a8a8a", lw=0.5,
                ls=(0, (3, 3)), zorder=5)
        ax.text(rr * np.sin(np.deg2rad(66.0)) * 0.99,
                rr * np.cos(np.deg2rad(66.0)) * 0.99,
                f"{lat_c:.0f}$^\\circ$N", fontsize=6.0, color="#555555",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.75))
    for lon_s, name in ((0.0, "0$^\\circ$"), (90.0, "90$^\\circ$E"),
                        (180.0, "180$^\\circ$"), (270.0, "90$^\\circ$W")):
        a = np.deg2rad(lon_s)
        ax.plot([0, np.sin(a)], [0, -np.cos(a)], color="#8a8a8a", lw=0.5,
                ls=(0, (3, 3)), zorder=5)
        ax.text(1.09 * np.sin(a), -1.09 * np.cos(a), name, fontsize=6.4,
                color="#555555", ha="center", va="center")

    ax.contour(xg, yg, np.nan_to_num(c_med), levels=[0.15],
               colors=[MED_LINE], linewidths=1.1, linestyles=[(0, (4, 2.2))],
               zorder=7)
    ax.contour(xg, yg, np.nan_to_num(c_cur), levels=[0.15],
               colors=[ICE_LINE], linewidths=1.7, zorder=8)
    th = np.linspace(0, 2 * np.pi, 721)
    ax.plot(np.sin(th), np.cos(th), color=COLORS["rule"], lw=0.9, zorder=9)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"September {YEARS[-1]} concentration, extent "
                 f"{ext_map:.2f} $\\times$ 10$^6$ km$^2$", fontsize=8.8,
                 fontweight="bold", pad=4.0)
    safe_colorbar(fig, pm, ax, label="sea-ice concentration (%)",
                  width=0.12, pad=0.05, shrink=0.80, labelsize=6.8,
                  ticks=[0, 20, 40, 60, 80, 100])
    ax.legend(handles=[
        Line2D([0], [0], color=ICE_LINE, lw=1.7,
               label=f"{YEARS[-1]} ice edge (15 %)"),
        Line2D([0], [0], color=MED_LINE, lw=1.1, ls=(0, (4, 2.2)),
               label="1991-2020 median edge"),
        Line2D([0], [0], color=LAND, lw=4.0, label="land"),
    ], loc="lower left", bbox_to_anchor=(-0.03, -0.015), frameon=False,
        fontsize=6.6, handlelength=1.9)

    # ---- (b) the seasonal cycle family --------------------------------------
    for k, y in enumerate(YEARS[:-1]):
        if k == i_rec:
            continue
        ax_s.plot(t, cycles[k], color="#cccccc", lw=0.6, zorder=2)
    ax_s.plot(t, cycles[i_rec], color=MED_LINE, lw=1.2, ls=(0, (4, 2.2)),
              zorder=4, label=f"record low {YEARS[i_rec]}")
    ax_s.plot(t, cycles[-1], color=ICE_LINE, lw=1.9, zorder=5,
              label=f"{YEARS[-1]}")
    ax_s.plot([SEPT], [sept[-1]], marker="o", ms=3.8, mfc="white",
              mec=ICE_LINE, mew=1.1, zorder=6)
    ax_s.axvline(SEPT, color=COLORS["baseline"], lw=0.7, ls=(0, (2, 2)),
                 zorder=1)
    ax_s.set_xlim(0.5, 12.5)
    ax_s.set_xticks(np.arange(1, 13))
    ax_s.set_xticklabels(MONTH_LABELS)
    ax_s.set_xlabel("month", fontsize=7.6)
    ax_s.set_ylabel("extent (10$^6$ km$^2$)", fontsize=7.6)
    ax_s.tick_params(labelsize=6.6)
    ax_s.grid(axis="y", lw=0.5, alpha=0.6)
    ax_s.legend(loc="upper right", frameon=False, fontsize=6.4,
                handlelength=1.7, labelspacing=0.28)
    ax_s.set_title(f"Annual cycle, {YEARS[0]}-{YEARS[-1]}", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)

    # ---- (c) the September minimum trend ------------------------------------
    for series, fitted, col, name, sl in (
            (march, fit_m, "#777777", "March maximum", slope_m),
            (sept, fit, ICE_LINE, "September minimum", slope)):
        ax_m.plot(YEARS, series, marker="o", ms=2.4, lw=0.8, color="#aaaaaa",
                  mfc="#999999", mec="none", zorder=3)
        ax_m.plot(YEARS, fitted(YEARS), color=col, lw=1.6, zorder=4)
        ax_m.text(YEARS[-1] + 0.8, fitted(YEARS[-1]),
                  f"{name}\n{sl:+.2f} per decade", fontsize=6.2, color=col,
                  ha="left", va="center", linespacing=1.3)
    ax_m.plot([YEARS[i_rec]], [sept[i_rec]], marker="v", ms=4.6, mfc="white",
              mec=MED_LINE, mew=1.0, zorder=5)
    ax_m.annotate(f"record low {YEARS[i_rec]}",
                  xy=(YEARS[i_rec], sept[i_rec]),
                  xytext=(YEARS[i_rec] - 2.0, sept[i_rec] + 2.6),
                  fontsize=6.3, color=MED_LINE, ha="center", va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=0.7,
                                  color=COLORS["rule"], shrinkA=1, shrinkB=3))
    ax_m.set_xlim(YEARS[0] - 1.0, YEARS[-1] + 15.0)
    ax_m.set_xticks([1980, 1990, 2000, 2010, 2020])
    ax_m.set_xlabel("year", fontsize=7.6)
    ax_m.set_ylabel("extent (10$^6$ km$^2$)", fontsize=7.6)
    ax_m.tick_params(labelsize=6.6)
    ax_m.grid(axis="y", lw=0.5, alpha=0.6)
    ax_m.set_title("Seasonal extremes (10$^6$ km$^2$ per decade)",
                   fontsize=8.5, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax_m.spines[s].set_visible(False)

    # Same quantity in both aux panels, so one range covers both (section 13.1).
    lo, hi = shared_limits([cycles, sept, march], pad=0.06)
    ax_s.set_ylim(lo, hi)
    ax_m.set_ylim(lo, hi)

    panel_label_inches(fig, ax, "(a)", dx=-0.10, dy=0.09)
    panel_label_inches(fig, ax_s, "(b)", dx=-0.42, dy=0.09)
    panel_label_inches(fig, ax_m, "(c)", dx=-0.42, dy=0.09)


    savefig(fig, "613_cry_seaice_extent.png")


if __name__ == "__main__":
    main()
