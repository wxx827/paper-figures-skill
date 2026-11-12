"""Calendar heatmap: multi-year GitHub-style week x weekday grids.

One row of gapped rounded cells per year; colour encodes the daily value.
Calendar starts are the real ones (1 Jan 2018 was a Monday) and 2020 is
handled as a leap year, so the month labels line up with the cells.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pubstyle import set_pub_style, savefig

CMAP = LinearSegmentedColormap.from_list(
    "grn", ["#eef0f2", "#c7e8cd", "#7fc99a", "#3f9c6b", "#1d6b45"])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "", "Wed", "", "Fri", "", "Sun"]
# 1 Jan fell on Monday in 2018, Tuesday in 2019, Wednesday in 2020
YEARS = [(2018, 0, 365), (2019, 1, 365), (2020, 2, 366)]
ROW_GAP = 8.6


def month_days(leap):
    d = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return d


def draw_year(ax, vals, start_weekday, y0, norm):
    cell = 0.84
    for d, v in enumerate(vals):
        pos = d + start_weekday
        x, wd = pos // 7, pos % 7
        y = y0 + (6 - wd)
        ax.add_patch(FancyBboxPatch(
            (x + (1 - cell) / 2, y + (1 - cell) / 2), cell, cell,
            boxstyle="round,pad=0,rounding_size=0.2", linewidth=0,
            facecolor=CMAP(norm(v))))


def main():
    set_pub_style()
    rng = np.random.default_rng(151)
    norm = Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    n_years = len(YEARS)
    for yi, (yr, sw, ndays) in enumerate(YEARS):
        t = np.arange(ndays)
        vals = np.clip(0.48 + 0.34 * np.sin(2 * np.pi * (t - 80) / ndays)
                       + 0.02 * yi + rng.normal(0, 0.10, ndays), 0, 1)
        y0 = (n_years - 1 - yi) * ROW_GAP
        draw_year(ax, vals, sw, y0, norm)

        ax.text(-3.2, y0 + 3.5, str(yr), fontsize=10, fontweight="bold",
                va="center", ha="right", color="#333333")
        for wd, lab in enumerate(WEEKDAYS):
            if lab:
                ax.text(-0.7, y0 + (6 - wd) + 0.5, lab, fontsize=6,
                        va="center", ha="right", color="#8a8a8a")

        cum = 0
        for m, md in enumerate(month_days(ndays == 366)):
            wk = (cum + sw) / 7.0
            if yi == 0:
                ax.text(wk + 0.25, n_years * ROW_GAP - 1.25, MONTHS[m],
                        fontsize=7.5, ha="left", color="#555555")
            cum += md
        ax.text(54.2, y0 + 3.5, f"mean {vals.mean():.2f}", fontsize=7,
                va="center", ha="left", color="#666666")

    ax.set_xlim(-4.2, 60)
    ax.set_ylim(-1.4, n_years * ROW_GAP)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Daily NDVI calendar, 2018-2020", fontsize=10.5,
                 fontweight="bold", pad=12)

    cax = fig.add_axes([0.36, 0.045, 0.26, 0.020])
    cb = fig.colorbar(ScalarMappable(cmap=CMAP, norm=norm), cax=cax,
                      orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("Daily NDVI (unitless)", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.90, bottom=0.13)
    savefig(fig, "25_calendar_heatmap.png")


if __name__ == "__main__":
    main()
