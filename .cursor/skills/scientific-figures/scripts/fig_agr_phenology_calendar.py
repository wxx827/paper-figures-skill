"""Figure 402 - Cropping calendar for one rotation region, with thermal time.

Rows are crops, columns are the calendar year, and each band is a phenological
stage.  Two things lift it above a plain Gantt chart.  The accumulated growing
degree days sit behind the bands on their own right-hand axis, so the reader
can see that the same stage occupies very different amounts of thermal time in
March and in July; and the side panel integrates that curve over each crop's
own sowing-to-harvest window, so the thermal requirement printed there is
measured off the very curve drawn behind the calendar - not quoted from a book.

Winter crops wrap the year boundary; their bands are split and drawn at both
ends of the axis rather than silently clipped.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import (set_pub_style, savefig, lighten, darken, style_axes,
                      COLORS)
from figkit import panel_labels_inches

SEED = 12
T_BASE = 10.0                 # base temperature for the degree-day sum, deg C
DAYS = 365

# Ordinal phenology ramp (domain convention, DESIGN_SYSTEM section 2.2): the
# crop greens up, turns gold as it fills, then browns off.  Every hue is
# derived from a house colour so the strip sits in the library's tonal family.
STAGES = [
    ("Sowing", lighten("#c0603f", 0.62)),
    ("Emergence", lighten("#2e7d5b", 0.66)),
    ("Vegetative growth", "#6bbf8a"),
    ("Heading / flowering", "#2e7d5b"),
    ("Grain or tuber filling", "#c9a227"),
    ("Maturity and harvest", darken("#c9a227", 0.42)),
]

# crop -> six (start, end) day-of-year windows, one per stage; a window whose
# start exceeds its end runs through 31 December.
CROPS = {
    "Winter wheat": [(288, 296), (297, 320), (321, 105), (106, 130),
                     (131, 160), (161, 172)],
    "Winter rapeseed": [(265, 272), (273, 300), (301, 75), (76, 105),
                        (106, 135), (136, 148)],
    "Spring potato": [(60, 70), (71, 95), (96, 120), (121, 140), (141, 170),
                      (171, 182)],
    "Cotton": [(105, 115), (116, 135), (136, 180), (181, 215), (216, 270),
               (271, 295)],
    "Single-season rice": [(120, 133), (134, 155), (156, 200), (201, 218),
                           (219, 255), (256, 268)],
    "Summer maize": [(160, 166), (167, 180), (181, 210), (211, 225),
                     (226, 260), (261, 272)],
    "Soybean": [(150, 158), (159, 172), (173, 200), (201, 220), (221, 255),
                (256, 268)],
}

# Field operations: marker shape carries the meaning, all in one ink colour so
# that the stage ramp keeps the only colour channel in the panel.
OPS = [("Sowing pass", "v"), ("Nitrogen top-dressing", "o"),
       ("Irrigation", "s"), ("Crop protection spray", "D"),
       ("Harvest", "^")]
OP_DOY = {
    "Winter wheat": [(292, 0), (318, 1), (95, 1), (118, 2), (128, 3),
                     (168, 4)],
    "Winter rapeseed": [(268, 0), (305, 1), (88, 3), (144, 4)],
    "Spring potato": [(65, 0), (100, 1), (128, 2), (135, 3), (178, 4)],
    "Cotton": [(110, 0), (150, 1), (190, 2), (205, 3), (290, 4)],
    "Single-season rice": [(126, 0), (162, 1), (175, 2), (208, 3), (264, 4)],
    "Summer maize": [(163, 0), (196, 1), (214, 2), (232, 3), (268, 4)],
    "Soybean": [(154, 0), (188, 1), (212, 2), (230, 3), (264, 4)],
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MLEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MSTART = np.concatenate([[1], 1 + np.cumsum(MLEN)[:-1]])


def temperature(seed=SEED, n=DAYS):
    """Daily mean air temperature for a warm-temperate site, deg C."""
    rng = np.random.default_rng(seed)
    doy = np.arange(1, n + 1)
    base = 14.8 - 13.2 * np.cos(2 * np.pi * (doy - 18) / n)
    weather = np.convolve(rng.standard_normal(n + 20), np.ones(9) / 9.0,
                          "same")[10:10 + n]
    return doy, base + 2.6 * weather


def segments(win):
    """(start, end) day window -> list of (x0, width) pieces on a 1..365 axis."""
    s, e = win
    if s <= e:
        return [(s, e - s + 1)]
    return [(s, DAYS - s + 1), (1, e)]


def cycle_days(stages):
    """Every day from the sowing pass to the end of maturity, as a mask."""
    s = stages[0][0]
    e = stages[-1][1]
    d = np.arange(1, DAYS + 1)
    return (d >= s) | (d <= e) if s > e else (d >= s) & (d <= e)


def main():
    set_pub_style(base_font=8)
    doy, tmean = temperature()
    gdd_daily = np.clip(tmean - T_BASE, 0.0, None)
    gdd_cum = np.cumsum(gdd_daily)

    names = list(CROPS)
    thermal = np.array([gdd_daily[cycle_days(CROPS[c])].sum() for c in names])
    span = np.array([int(cycle_days(CROPS[c]).sum()) for c in names])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.30],
                          height_ratios=[1.0, 0.34],
                          left=0.108, right=0.930, top=0.845, bottom=0.098,
                          wspace=0.075, hspace=0.16)
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_gdd = fig.add_subplot(gs[0, 1], sharey=ax_cal)
    ax_t = fig.add_subplot(gs[1, 0], sharex=ax_cal)

    # ---- (a) the calendar --------------------------------------------------
    ax_bg = ax_cal.twinx()
    ax_bg.fill_between(doy, 0.0, gdd_cum, color=lighten("#c9a227", 0.84),
                       lw=0.0, zorder=0)
    ax_bg.plot(doy, gdd_cum, color=lighten("#c9a227", 0.25), lw=1.0, zorder=1)
    ax_bg.set_ylim(0.0, gdd_cum[-1] * 1.06)
    # No right-hand ticks: the curve is context, and the numbers that matter
    # are integrated per crop in panel (b).  A second tick column here would
    # also have to squeeze into the gutter in front of that panel.
    ax_bg.set_yticks([])
    ax_bg.set_zorder(0)
    ax_cal.set_zorder(2)
    ax_cal.patch.set_visible(False)
    for s in ("top", "left", "right"):
        ax_bg.spines[s].set_visible(False)

    for i, crop in enumerate(names):
        for k, win in enumerate(CROPS[crop]):
            ax_cal.broken_barh(segments(win), (i - 0.30, 0.60),
                               facecolors=STAGES[k][1], edgecolor="white",
                               linewidth=0.5, zorder=4)
        for d, op in OP_DOY[crop]:
            ax_cal.plot([d], [i + 0.44], marker=OPS[op][1], ms=3.4,
                        mfc="white", mec=COLORS["rule"], mew=0.7, zorder=6,
                        clip_on=False)

    ax_cal.text(6, -0.52, "background fill: accumulated growing degree days "
                          f"above {T_BASE:.0f} \u00b0C",
                fontsize=6.4, color=darken("#c9a227", 0.34), ha="left",
                va="center", zorder=5)

    for m in MSTART[1:]:
        ax_cal.axvline(m, color=COLORS["grid"], lw=0.5, zorder=1)
    ax_cal.set_xlim(1, DAYS)
    ax_cal.set_ylim(len(names) - 0.5, -0.72)
    ax_cal.set_yticks(range(len(names)))
    ax_cal.set_yticklabels(names, fontsize=7.5)
    ax_cal.set_xticks(MSTART + MLEN / 2.0)
    ax_cal.set_xticklabels(MONTHS, fontsize=7.0)
    ax_cal.tick_params(axis="x", length=0, labelbottom=False)
    ax_cal.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax_cal.spines[s].set_visible(False)
    ax_cal.set_title("Phenological stages and field operations, "
                     "day of year 1 to 365", fontsize=8.8, fontweight="bold",
                     pad=4.0)

    # ---- (b) thermal time actually integrated under that curve -------------
    ax_gdd.barh(np.arange(len(names)), thermal, height=0.60,
                color=lighten("#3b5b92", 0.30), edgecolor="white",
                linewidth=0.6, zorder=3)
    for i, (v, d) in enumerate(zip(thermal, span)):
        ax_gdd.text(v + thermal.max() * 0.025, i, f"{v:,.0f}".replace(",", " "),
                    fontsize=6.5, va="center", ha="left", color=COLORS["sub"])
        ax_gdd.text(thermal.max() * 0.035, i, f"{d} d", fontsize=6.2,
                    va="center", ha="left", color="white", zorder=5)
    ax_gdd.set_xlim(0, thermal.max() * 1.30)
    ax_gdd.set_xlabel("thermal time of the cycle (\u00b0C\u00b7d)",
                      fontsize=7.6)
    ax_gdd.tick_params(axis="x", labelsize=6.8)
    ax_gdd.tick_params(axis="y", labelleft=False, length=0)
    ax_gdd.set_xticks([0, 1000, 2000, 3000])
    ax_gdd.grid(axis="x", linewidth=0.5, alpha=0.7)
    ax_gdd.set_title("Sowing to harvest", fontsize=8.8, fontweight="bold",
                     pad=4.0)
    for s in ("top", "right", "left"):
        ax_gdd.spines[s].set_visible(False)

    # ---- (c) the daily temperature the calendar is built on ----------------
    ax_t.fill_between(doy, T_BASE, tmean, where=tmean >= T_BASE,
                      color=lighten("#c0603f", 0.68), lw=0.0, zorder=2)
    ax_t.plot(doy, tmean, color=darken("#c0603f", 0.18), lw=0.9, zorder=3)
    ax_t.axhline(T_BASE, color=COLORS["baseline"], lw=0.8, ls="--", zorder=4)
    ax_t.text(DAYS - 4, T_BASE + 0.8, f"base {T_BASE:.0f} \u00b0C",
              fontsize=6.4, color=COLORS["sub"], ha="right", va="bottom")
    for m in MSTART[1:]:
        ax_t.axvline(m, color=COLORS["grid"], lw=0.5, zorder=1)
    ax_t.set_xlim(1, DAYS)
    ax_t.set_ylim(min(tmean.min() - 2.0, 0.0), tmean.max() + 3.0)
    ax_t.set_xticks(MSTART + MLEN / 2.0)
    ax_t.set_xticklabels(MONTHS, fontsize=7.0)
    ax_t.tick_params(axis="x", length=0, labelbottom=True)
    ax_t.tick_params(axis="y", labelsize=6.8)
    ax_t.set_ylabel("daily mean\nair temperature (\u00b0C)", fontsize=7.4,
                    linespacing=1.25)
    style_axes(ax_t)

    # ---- one shared key for both colour and marker channels ----------------
    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=n)
               for n, c in STAGES]
    handles += [Line2D([0], [0], marker=m, ls="none", mfc="white",
                       mec=COLORS["rule"], mew=0.7, ms=3.8, label=n)
                for n, m in OPS]

    # Panels (a) and (c) carry long crop names / a two-line y label, panel (b)
    # has no y ticks at all - DESIGN_SYSTEM section 4 allows the offset to
    # follow the label column in exactly this case.
    panel_labels_inches(fig, [ax_cal, ax_t], labels=["(a)", "(c)"], dx=-0.80,
                        dy=0.09)
    panel_labels_inches(fig, [ax_gdd], labels=["(b)"], dx=-0.14, dy=0.09)
    hottest = names[int(np.argmax(thermal))]
    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).

    savefig(fig, "402_agr_phenology_calendar.png")


if __name__ == "__main__":
    main()
