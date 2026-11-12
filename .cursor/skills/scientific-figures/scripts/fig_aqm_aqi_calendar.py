"""A year of daily air quality index, day by day, with what drove it.

The main panel is the calendar heat map every air quality report opens with:
one square per day, split into two half-year bands so the squares stay large
enough to read, coloured by the daily index.  The index itself is not invented
- four pollutant concentrations are generated, each is converted to a
sub-index through the piecewise-linear breakpoint table, and the day's AQI is
the largest of them, exactly as the standard defines it.

Because the calendar and both companion panels are reductions of that same
daily table they cannot disagree: the exceedance count printed on the calendar
is the number of coloured squares at or above the unhealthy-for-sensitive-
groups class, the monthly stack re-bins those same squares, and the primary
pollutant shares are the argmax of the same sub-index matrix, so they sum to
one hundred per cent by construction.

Illustrative synthetic monitoring year, not a real station record.
"""
import datetime as dt
import textwrap

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.66, 1.06, 5.45, 4.06)
AUX_T_RECT = (7.42, 3.34, 3.16, 1.78)
AUX_B_RECT = (7.42, 1.06, 3.16, 1.78)

YEAR = 2023

# AQI classes: clean end in house green, dirty end in house red and brown.
CAT_EDGE = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
CAT_COL = ["#2e7d5b", "#a9c48d", "#e0b366", "#d98c5f", "#b23a48", "#7a2f33"]
CAT_NAME = ["good", "moderate", "unhealthy (sensitive)", "unhealthy",
            "very unhealthy", "hazardous"]

# Breakpoint tables: (C_low, C_high, I_low, I_high) per pollutant.
BREAKS = {
    "PM2.5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
              (250.5, 500.4, 301, 500)],
    "PM10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "O3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
           (86, 105, 151, 200), (106, 200, 201, 300)],
    "NO2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300)],
}
UNITS = {"PM2.5": "$\\mu g/m^3$", "PM10": "$\\mu g/m^3$", "O3": "ppb",
         "NO2": "ppb"}
POL_COL = {"PM2.5": "#b23a48", "PM10": "#c9a227", "O3": "#3b5b92",
           "NO2": "#8c5a3c"}


def sub_index(conc, pol):
    """Piecewise-linear AQI sub-index for one pollutant, index units."""
    conc = np.asarray(conc, dtype=float)
    out = np.zeros_like(conc)
    for c_lo, c_hi, i_lo, i_hi in BREAKS[pol]:
        m = (conc >= c_lo) & (conc <= c_hi)
        out[m] = i_lo + (i_hi - i_lo) * (conc[m] - c_lo) / (c_hi - c_lo)
    top = BREAKS[pol][-1]
    out[conc > top[1]] = top[3]
    return out


def ar1(rng, n, rho, sigma):
    """Unit-variance-scaled AR(1) sequence, for day-to-day persistence."""
    e = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def synth_year(rng):
    """Daily concentrations for one calendar year, plus dates."""
    d0 = dt.date(YEAR, 1, 1)
    dates = [d0 + dt.timedelta(days=int(i))
             for i in range((dt.date(YEAR + 1, 1, 1) - d0).days)]
    n = len(dates)
    doy = np.arange(n, dtype=float)
    wk = np.array([d.weekday() for d in dates])
    weekend = (wk >= 5).astype(float)

    winter = np.cos(2.0 * np.pi * (doy - 8.0) / 365.0)
    summer = np.cos(2.0 * np.pi * (doy - 196.0) / 365.0)

    pm25 = np.exp(np.log(21.0 + 12.0 * winter) + ar1(rng, n, 0.72, 0.30)
                  - 0.10 * weekend)
    dust = np.zeros(n)
    for start in rng.choice(np.arange(60, 130), size=4, replace=False):
        span = np.arange(start, min(start + 4, n))
        dust[span] += rng.uniform(70.0, 150.0)
    pm10 = 1.75 * pm25 + dust + np.abs(ar1(rng, n, 0.55, 6.0))
    o3 = np.clip(33.0 + 19.0 * summer + ar1(rng, n, 0.60, 7.5)
                 + 4.0 * weekend, 4.0, None)
    no2 = np.clip(21.0 + 9.0 * winter + ar1(rng, n, 0.55, 5.0)
                  - 3.0 * weekend, 3.0, None)
    return dates, {"PM2.5": pm25, "PM10": pm10, "O3": o3, "NO2": no2}


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1009)

    dates, conc = synth_year(rng)
    n = len(dates)
    pols = list(BREAKS.keys())
    sub = np.vstack([sub_index(conc[p], p) for p in pols])
    aqi = sub.max(axis=0)
    primary = np.array(pols)[np.argmax(sub, axis=0)]
    cat = np.clip(np.searchsorted(CAT_EDGE, aqi, side="right") - 1, 0, 5)
    n_exceed = int((aqi > 100.0).sum())

    # ---- calendar grid: two half-year bands of seven weekday rows ----------
    n_col, gap = 27, 2
    grid = np.full((7 * 2 + gap, n_col), np.nan)
    off = dates[0].weekday()
    band_of, col_of = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for i, d in enumerate(dates):
        week = (i + off) // 7
        band = 0 if week < n_col else 1
        col = week - band * n_col
        row = d.weekday() + band * (7 + gap)
        grid[row, col] = aqi[i]
        band_of[i], col_of[i] = band, col

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax_c, ax_d):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the calendar -------------------------------------------------
    cmap = mpl.colors.ListedColormap(CAT_COL)
    norm = mpl.colors.BoundaryNorm(CAT_EDGE, cmap.N)
    ax.set_aspect("equal")
    xg = np.arange(n_col + 1, dtype=float)
    yg = np.arange(grid.shape[0] + 1, dtype=float)
    im = ax.pcolormesh(xg, yg, np.ma.masked_invalid(grid), cmap=cmap,
                       norm=norm, edgecolors="white", linewidth=0.55,
                       zorder=3)
    ax.set_xlim(0.0, n_col)
    ax.set_ylim(grid.shape[0], 0.0)
    ax.set_xticks([])
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5)
    wd = ["Mon", "", "Wed", "", "Fri", "", "Sun"]
    ax.set_yticklabels(wd + [""] * gap + wd, fontsize=6.4)
    ax.tick_params(length=0.0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
              "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        sel = np.array([d.month == m for d in dates])
        for b in (0, 1):
            s = sel & (band_of == b)
            if s.sum() < 12:
                continue
            xm = float(np.median(col_of[s])) + 0.5
            ax.text(xm, b * (7 + gap) - 0.45, months[m - 1], ha="center",
                    va="bottom", fontsize=6.6, color=SUB)
    ax.set_title(f"Calendar of daily air quality index, {YEAR}: "
                 f"{n_exceed} days above AQI 100", fontsize=9.0,
                 fontweight="bold", pad=21.0)
    cb = safe_colorbar(fig, im, ax, label="daily AQI (index)", side="bottom",
                       width=0.13, pad=0.52, shrink=0.94, labelsize=7.0,
                       ticks=CAT_EDGE)
    cb.ax.set_xticklabels([f"{v:.0f}" for v in CAT_EDGE], fontsize=6.6)
    short = ["good", "moderate", "USG", "unhealthy", "very unhealthy",
             "hazardous"]
    dark = [True, False, False, True, True, True]
    for c in range(6):
        cb.ax.text(0.5 * (CAT_EDGE[c] + CAT_EDGE[c + 1]), 0.5, short[c],
                   ha="center", va="center", fontsize=6,
                   color="white" if dark[c] else INK)

    worst = int(np.argmax(aqi))
    ax.plot([col_of[worst] + 0.5], [dates[worst].weekday()
                                    + band_of[worst] * (7 + gap) + 0.5],
            marker="o", ms=6.0, mfc="none", mec=INK, mew=1.1, zorder=6)
    ax.text(0.0, grid.shape[0] + 0.55,
            f"{YEAR} daily AQI, weeks 1-27 in the upper band, 28-53 in the "
            f"lower one; circled is the worst day, "
            f"{dates[worst].strftime('%d %b')}, AQI {aqi[worst]:.0f} "
            f"({primary[worst]})",
            ha="left", va="top", fontsize=6.4, color=MUTED, clip_on=False)

    # ---- aux (b): the same squares re-binned by month -----------------------
    mon_idx = np.array([d.month for d in dates])
    bottom = np.zeros(12)
    for c in range(6):
        vals = np.array([int(((mon_idx == m) & (cat == c)).sum())
                         for m in range(1, 13)], dtype=float)
        ax_c.bar(np.arange(12), vals, bottom=bottom, width=0.80,
                 color=CAT_COL[c], edgecolor="white", linewidth=0.5,
                 zorder=4, label=CAT_NAME[c])
        bottom += vals
    ax_c.set_xlim(-0.7, 11.7)
    ax_c.set_ylim(0.0, 44.0)
    ax_c.set_xticks(np.arange(12))
    ax_c.set_xticklabels([m[0] for m in months], fontsize=6.6)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_ylabel("days in month (days)", fontsize=7.5)
    ax_c.set_xlabel(f"month of {YEAR}", fontsize=7.5, labelpad=1.5)
    ax_c.set_title("Every square, re-binned by month", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    worst_m = int(np.argmax([((mon_idx == m) & (aqi > 100.0)).sum()
                             for m in range(1, 13)]))
    ax_c.text(0.5, 0.975,
              f"{months[worst_m]} carries "
              f"{int(((mon_idx == worst_m + 1) & (aqi > 100)).sum())} of the "
              f"{n_exceed} exceedance days",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.3,
              color=SUB)

    # ---- aux (c): who drove the exceedances ---------------------------------
    ex = aqi > 100.0
    share = np.array([100.0 * (primary[ex] == p).sum() / max(ex.sum(), 1)
                      for p in pols])
    order = np.argsort(share)
    ypos = np.arange(len(pols))
    ax_d.barh(ypos, share[order], height=0.62,
              color=[POL_COL[pols[i]] for i in order], edgecolor="white",
              linewidth=0.6, zorder=4)
    for y, i in zip(ypos, order):
        peak = conc[pols[i]][ex].max() if ex.sum() else 0.0
        ax_d.text(share[i] + 1.4, y, f"{share[i]:.1f} %  (peak "
                  f"{peak:.0f} {UNITS[pols[i]]})", ha="left", va="center",
                  fontsize=6.3, color=INK)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([pols[i] for i in order], fontsize=7.0)
    ax_d.set_xlim(0.0, 100.0)
    ax_d.set_ylim(-0.65, len(pols) - 0.35)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("share of the exceedance days (%)", fontsize=7.5,
                    labelpad=1.5)
    ax_d.set_title("Which pollutant set the index", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_d.text(0.985, 0.055,
              "shares are the argmax of the four sub-indices\non the "
              f"{n_exceed} exceedance days, summing to "
              f"{share.sum():.0f} %",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=6.3,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.54, dy=0.30)
    foot = add_footer(fig, "Synthetic data, seed=1009")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "1009_aqm_aqi_calendar.png")


if __name__ == "__main__":
    main()
