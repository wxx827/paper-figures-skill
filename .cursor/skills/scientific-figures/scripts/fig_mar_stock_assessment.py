"""Fishery stock assessment read off a Kobe plot with its supporting series.

The main panel is the standard Kobe phase plane: stock status B/B_MSY on the
horizontal axis, fishing pressure F/F_MSY on the vertical one, with the four
quadrants shaded green (healthy), gold (one reference point breached) and red
(overfished while overfishing continues).  The trajectory is one simulated
stock walked from 1975 to 2024; marker shade tracks the year, so the loop the
reader follows - green corner, up into overfishing, left into the red, then
back down and right as the rebuilding plan bites - is the whole argument.

Every quoted number is computed from the same arrays that are plotted: the
peak-pressure and minimum-biomass years are located with argmax/argmin, catch
is F times biomass so the MSY line in panel (b) really is F_MSY * B_MSY, and
the Beverton-Holt curve in panel (c) comes from a curve_fit on the plotted
stock-recruit pairs.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, darken,
                      annotate_box)
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREEN, GOLD, RED, BLUE = "#2e7d5b", "#c9a227", "#b23a48", "#3b5b92"
DEEP = "#134a76"
CATCH_FILL, CATCH_EDGE = "#9cc3dd", "#5f9dc6"

FW, FH = 10.4, 6.4
MAIN = (0.80, 1.06, 4.84, 4.10)
AUX1 = (6.52, 3.10, 2.78, 2.06)
AUX2 = (6.52, 1.06, 2.78, 1.44)

B_MSY = 420.0            # kt of spawning biomass at maximum sustainable yield
F_MSY = 0.19             # per year
YEAR0, YEAR1 = 1975, 2024
XMAX, YMAX = 2.35, 2.05


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def beverton_holt(s, a, b):
    """Recruits (millions) from spawning biomass (kt)."""
    return a * s / (b + s)


def build_stock(rng):
    years = np.arange(YEAR0, YEAR1 + 1)
    t = (years - years[0]) / (years[-1] - years[0])
    b = (1.95 - 1.85 * logistic((t - 0.40) / 0.13)
         + 0.95 * logistic((t - 0.72) / 0.11))
    f = (0.45 + 1.55 * logistic((t - 0.35) / 0.13)
         - 1.25 * logistic((t - 0.70) / 0.12))
    b = b * np.exp(rng.normal(0.0, 0.035, b.size))
    f = f * np.exp(rng.normal(0.0, 0.055, f.size))
    biomass = b * B_MSY
    catch = f * F_MSY * biomass
    recruits = beverton_holt(biomass, 1500.0, 300.0)
    recruits = recruits * np.exp(rng.normal(0.0, 0.20, recruits.size))
    return dict(year=years, b=b, f=f, biomass=biomass, catch=catch,
                recruits=recruits)


def draw_kobe(ax, st):
    quads = [((0.0, 1.0), (1.0, YMAX - 1.0), RED,
              "overfished and\noverfishing", "left", "top", (0.04, 0.965)),
             ((1.0, 1.0), (XMAX - 1.0, YMAX - 1.0), GOLD,
              "overfishing", "right", "bottom", (0.965, 0.505)),
             ((0.0, 0.0), (1.0, 1.0), GOLD,
              "overfished,\nrebuilding", "left", "bottom", (0.04, 0.035)),
             ((1.0, 0.0), (XMAX - 1.0, 1.0), GREEN,
              "healthy", "right", "bottom", (0.965, 0.035))]
    for (x0, y0), (w, h), col, name, ha, va, loc in quads:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=lighten(col, 0.80),
                               edgecolor="none", zorder=0))
        ax.text(loc[0], loc[1], name, transform=ax.transAxes, ha=ha, va=va,
                fontsize=6.8, color=darken(col, 0.22), linespacing=1.3,
                zorder=4)
    ax.axvline(1.0, color="#808080", lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)
    ax.axhline(1.0, color="#808080", lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)

    ax.plot(st["b"], st["f"], color="#4a4a4a", lw=1.0, alpha=0.85, zorder=5)
    cmap = get_cmap("cb_sequential_blue")
    shade = np.linspace(0.28, 1.0, st["year"].size)
    ax.scatter(st["b"], st["f"], s=15.0, c=cmap(shade), edgecolors="white",
               linewidths=0.35, zorder=6)

    k_hi = int(np.argmax(st["f"]))
    k_lo = int(np.argmin(st["b"]))
    marks = [(0, "start " + str(int(st["year"][0])), (14, 10), "left"),
             (k_hi, "peak F " + str(int(st["year"][k_hi])), (-14, 16),
              "right"),
             (k_lo, "lowest B " + str(int(st["year"][k_lo])), (-14, -16),
              "right"),
             (st["year"].size - 1, "end " + str(int(st["year"][-1])),
              (16, -12), "left")]
    for k, lab, off, ha in marks:
        ax.annotate(lab, xy=(st["b"][k], st["f"][k]), xytext=off,
                    textcoords="offset points", fontsize=6.6, color=INK,
                    ha=ha, zorder=8,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=RULE,
                                    shrinkA=0.0, shrinkB=2.0))

    annotate_box(ax, "\n".join([
        "terminal year " + str(int(st["year"][-1])),
        "B/B$_{MSY}$ = " + "{:.2f}".format(st["b"][-1]),
        "F/F$_{MSY}$ = " + "{:.2f}".format(st["f"][-1]),
        "years overfished  {:d} of {:d}".format(
            int(np.sum(st["b"] < 1.0)), st["b"].size)]),
        loc="upper right", fontsize=6.6, margin=0.028)

    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_xlabel("stock status  B/B$_{MSY}$ (-)", fontsize=8.0)
    ax.set_ylabel("fishing pressure  F/F$_{MSY}$ (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Kobe phase plane, 1975-2024 trajectory", fontsize=8.5,
                 fontweight="bold")


def draw_series(ax, st):
    msy = F_MSY * B_MSY
    ax.bar(st["year"], st["catch"], width=0.82, color=CATCH_FILL,
           edgecolor=CATCH_EDGE, linewidth=0.4, zorder=3)
    ax.axhline(msy, color=RED, lw=1.0, ls=(0, (4.0, 2.2)), zorder=5)
    ax.text(st["year"][0] + 0.5, msy + 3.0,
            "MSY {:.0f} kt".format(msy), fontsize=6.4, color=RED,
            ha="left", va="bottom", zorder=6)
    ax.set_xlim(st["year"][0] - 1.0, st["year"][-1] + 1.0)
    ax.set_ylim(0.0, max(st["catch"].max(), msy) * 1.32)
    ax.set_ylabel("catch (kt/yr)", fontsize=7.8, color=darken(CATCH_EDGE, 0.2))
    ax.tick_params(labelsize=6.8)
    ax.tick_params(axis="y", colors=darken(CATCH_EDGE, 0.25))

    axb = ax.twinx()
    axb.plot(st["year"], st["biomass"], color=DEEP, lw=1.5, zorder=6)
    axb.axhline(B_MSY, color=DEEP, lw=0.9, ls=(0, (2.6, 2.0)), alpha=0.75,
                zorder=4)
    axb.text(st["year"][0] + 0.5, B_MSY + 16.0, "B$_{MSY}$", fontsize=6.4,
             color=DEEP, ha="left", va="bottom", zorder=6)
    axb.set_ylim(0.0, st["biomass"].max() * 1.62)
    axb.set_ylabel("spawning biomass (kt)", fontsize=7.8, color=DEEP)
    axb.tick_params(labelsize=6.8, colors=DEEP)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(True)
    axb.spines["right"].set_color(DEEP)
    axb.spines["right"].set_linewidth(0.8)

    ax.set_title("Catch and biomass against reference points", fontsize=8.5,
                 fontweight="bold")
    return msy


def draw_recruitment(ax, st):
    par, _ = curve_fit(beverton_holt, st["biomass"], st["recruits"],
                       p0=[1500.0, 300.0], maxfev=20000)
    cmap = get_cmap("cb_sequential_blue")
    shade = np.linspace(0.28, 1.0, st["year"].size)
    ax.scatter(st["biomass"], st["recruits"], s=11.0, c=cmap(shade),
               edgecolors="white", linewidths=0.3, zorder=4)
    grid = np.linspace(0.0, st["biomass"].max() * 1.06, 220)
    ax.plot(grid, beverton_holt(grid, *par), color=RULE, lw=1.3, zorder=5)
    ax.axvline(B_MSY, color="#808080", lw=0.9, ls=(0, (4.0, 2.4)), zorder=3)
    r_at_msy = beverton_holt(B_MSY, *par)
    ax.text(B_MSY + 12.0, ax.get_ylim()[0], "B$_{MSY}$", fontsize=6.4,
            color=MUTED, ha="left", va="bottom", zorder=6)
    ax.text(0.035, 0.94,
            "Beverton-Holt  R$_{max}$ = " + "{:.0f}".format(par[0])
            + " million,  R(B$_{MSY}$) = " + "{:.0f}".format(r_at_msy),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=SUB)
    ax.set_xlim(0.0, st["biomass"].max() * 1.06)
    ax.set_xlabel("spawning biomass (kt)", fontsize=7.8)
    ax.set_ylabel("recruits (millions)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Stock-recruit relationship", fontsize=8.5,
                 fontweight="bold")
    return par, r_at_msy


def main():
    set_pub_style()
    rng = np.random.default_rng(690)
    st = build_stock(rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_k = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX1)
    ax_r = axes_inches(fig, AUX2)

    draw_kobe(ax_k, st)
    msy = draw_series(ax_s, st)
    par, r_at_msy = draw_recruitment(ax_r, st)

    panel_labels_inches(fig, [ax_k, ax_s, ax_r], dx=-0.44, dy=0.20)

    over = float(np.mean(st["f"] > 1.0))
    peak_yr = int(st["year"][int(np.argmax(st["catch"]))])

    print("terminal b={:.3f} f={:.3f} msy={:.1f} bh_a={:.0f} bh_b={:.0f} "
          "r_msy={:.0f}".format(st["b"][-1], st["f"][-1], msy, par[0], par[1],
                                r_at_msy))
    savefig(fig, "690_mar_stock_assessment.png")


if __name__ == "__main__":
    main()
