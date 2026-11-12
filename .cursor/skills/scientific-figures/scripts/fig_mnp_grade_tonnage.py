"""Grade-tonnage curves cut from one lognormal deposit grade distribution.

The main panel is the curve pair that decides whether a deposit is a mine:
as the cut-off grade rises, the tonnage above cut-off falls and the average
grade of what is left rises.  The two curves must move in opposite
directions, and here they have to, because both are integrals of the *same*
block-model grade distribution above the same cut-off - nothing is drawn
independently.  The two smaller panels follow that same integral: contained
metal, which is tonnage times grade and therefore peaks at an interior
cut-off, and the sensitivity of that peak to the metal price through the
break-even cut-off.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"        # average grade, right axis
BLUE = "#3b5b92"       # tonnage, left axis
GREEN = "#2e7d5b"      # contained metal
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

SEED = 833_2026
N_BLOCK = 240_000              # blocks in the model
BLOCK_MT = 0.0125              # million tonnes per block
LOG_MU, LOG_SIGMA = np.log(0.42), 0.62   # lognormal grade, % Cu
CUTOFF = np.linspace(0.05, 1.60, 220)    # cut-off grades, % Cu
CUT_USED = 0.35                # cut-off currently applied, % Cu

PRICE_BASE = 9000.0            # USD per tonne of copper
RECOVERY = 0.88                # metallurgical recovery
PAYABLE = 0.965                # smelter payable fraction
COST_MINE_MILL = 21.0          # USD per tonne of ore processed
PRICES = [6500.0, 9000.0, 11500.0]
MILL_MT = 1200.0               # ore the plant can treat over the mine life


def deposit(seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.lognormal(LOG_MU, LOG_SIGMA, N_BLOCK)


def grade_tonnage(grades, cutoffs):
    """Tonnage, average grade and contained metal above each cut-off."""
    order = np.sort(grades)
    n_above = order.size - np.searchsorted(order, cutoffs, side="left")
    csum = np.concatenate([[0.0], np.cumsum(order)])
    total = csum[-1]
    metal_above = total - csum[order.size - n_above]

    tonnes = n_above * BLOCK_MT
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(n_above > 0, metal_above / np.maximum(n_above, 1),
                       np.nan)
    metal_kt = tonnes * avg / 100.0 * 1000.0        # kt of contained copper
    return tonnes, avg, metal_kt


def breakeven_cutoff(price):
    """Cut-off at which revenue from one tonne of ore covers its cost."""
    return COST_MINE_MILL / (price * RECOVERY * PAYABLE / 100.0)


def main_panel(ax, tonnes, avg, at_cut):
    ax_r = ax.twinx()
    ax.fill_between(CUTOFF, 0.0, tonnes, color=lighten(BLUE, 0.86),
                    linewidth=0.0, zorder=1)
    ln_t, = ax.plot(CUTOFF, tonnes, color=BLUE, lw=2.1, zorder=5,
                    label="tonnage above cut-off (left axis)")
    ln_g, = ax_r.plot(CUTOFF, avg, color=RED, lw=2.1, zorder=5,
                      label="average grade above cut-off (right axis)")

    ax.axvline(CUT_USED, color=INK, lw=1.2, ls=(0, (4, 2.4)), zorder=6)
    ax.plot([CUT_USED], [at_cut["tonnes"]], marker="o", ms=5.4, color=BLUE,
            mfc="white", mew=1.5, zorder=8)
    ax_r.plot([CUT_USED], [at_cut["avg"]], marker="o", ms=5.4, color=RED,
              mfc="white", mew=1.5, zorder=8)
    ax.annotate(f"{at_cut['tonnes']:.0f} Mt of ore",
                xy=(CUT_USED, at_cut["tonnes"]),
                xytext=(CUT_USED + 0.10, at_cut["tonnes"] + 620.0),
                ha="left", va="bottom", fontsize=7.0, color=BLUE, zorder=9,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax_r.annotate(f"{at_cut['avg']:.2f} % Cu average",
                  xy=(CUT_USED, at_cut["avg"]),
                  xytext=(CUT_USED + 0.12, at_cut["avg"] - 0.20),
                  ha="left", va="top", fontsize=7.0, color=RED, zorder=9,
                  arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(CUT_USED - 0.02, tonnes.max() * 0.985,
            f"cut-off in use  {CUT_USED:.2f} % Cu", ha="right", va="top",
            fontsize=7.0, color=INK, rotation=90.0, zorder=9)

    ax.text(0.99, tonnes.max() * 0.60,
            f"reserve at the cut-off in use\n{at_cut['tonnes']:.0f} Mt at "
            f"{at_cut['avg']:.2f} % Cu\n{at_cut['metal']:.0f} kt contained "
            "copper", ha="left", va="top", fontsize=6.8, color=INK,
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.95))

    ax.set_xlim(CUTOFF[0], CUTOFF[-1])
    ax.set_ylim(0.0, tonnes.max() * 1.06)
    ax.set_xlabel("cut-off grade (% Cu)")
    ax.set_ylabel("ore tonnage above cut-off (Mt)", color=BLUE)
    ax.tick_params(labelsize=7.2)
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["left"].set_color(BLUE)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", lw=0.45, alpha=0.5)

    ax_r.set_ylim(0.0, np.nanmax(avg) * 1.10)
    ax_r.set_ylabel("average grade above cut-off (% Cu)", color=RED)
    ax_r.tick_params(labelsize=7.2, colors=RED)
    ax_r.spines["right"].set_color(RED)
    ax_r.spines["top"].set_visible(False)
    ax.legend(handles=[ln_t, ln_g], fontsize=6.9, frameon=True,
              framealpha=0.93, loc="upper right", handlelength=1.8,
              borderaxespad=0.4, edgecolor="#cccccc")
    ax.set_title("Raising the cut-off trades tonnage for grade - the two "
                 "curves are one integral", fontsize=9.0, fontweight="bold",
                 color=INK)


def recovered_metal(tonnes, avg):
    """Copper the plant can actually make: the mill caps how much ore runs."""
    milled = np.minimum(tonnes, MILL_MT)
    return milled * avg / 100.0 * 1000.0 * RECOVERY


def metal_panel(ax, tonnes, avg, produced, i_peak):
    ax.fill_between(CUTOFF, 0.0, produced, color=lighten(GREEN, 0.85),
                    linewidth=0.0, zorder=1)
    ax.plot(CUTOFF, produced, color=GREEN, lw=1.8, zorder=5)
    ax.plot([CUTOFF[i_peak]], [produced[i_peak]], marker="o", ms=5.0,
            color=GREEN, mfc="white", mew=1.4, zorder=7)
    ax.axvline(CUTOFF[i_peak], color=GREEN, lw=0.9, ls=(0, (3, 2.2)),
               zorder=4)
    ax.axvline(CUT_USED, color=INK, lw=0.9, ls=(0, (4, 2.4)), zorder=4)

    ax.annotate(f"peak {produced[i_peak]:.0f} kt Cu\n"
                f"at {CUTOFF[i_peak]:.2f} % Cu cut-off\n"
                f"({MILL_MT:.0f} Mt plant limit)",
                xy=(CUTOFF[i_peak], produced[i_peak]),
                xytext=(CUTOFF[i_peak] + 0.14, produced[i_peak] * 0.94),
                ha="left", va="top", fontsize=6.3, color=GREEN, zorder=8,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    ax.text(CUT_USED + 0.03, produced.max() * 0.08, "cut-off in use",
            ha="left", va="bottom", fontsize=6.2, color=INK, rotation=90.0)

    ax.set_xlim(CUTOFF[0], CUTOFF[-1])
    ax.set_ylim(0.0, produced.max() * 1.16)
    ax.set_xlabel("cut-off grade (% Cu)", fontsize=7.4)
    ax.set_ylabel("copper produced (kt)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.55)
    ax.set_title("Grade gain beats tonnage loss, then stops",
                 fontsize=8.2, fontweight="bold", color=INK)


def price_panel(ax, tonnes, avg):
    milled = np.minimum(tonnes, MILL_MT)
    cols = [lighten(PURPLE, 0.42), PURPLE, "#5f4a86"]
    for price, col in zip(PRICES, cols):
        value = milled * 1e6 * (avg / 100.0 * price * RECOVERY * PAYABLE
                                - COST_MINE_MILL) / 1e9
        ax.plot(CUTOFF, value, color=col, lw=1.6, zorder=5,
                label=f"{price / 1000:.1f} k$/t Cu")
        i_top = int(np.nanargmax(value))
        ax.plot([CUTOFF[i_top]], [value[i_top]], marker="o", ms=4.4,
                color=col, mfc="white", mew=1.3, zorder=7)
        ax.axvline(breakeven_cutoff(price), color=col, lw=0.8,
                   ls=(0, (2, 2.2)), alpha=0.8, zorder=3)

    ax.axvline(CUT_USED, color=INK, lw=0.9, ls=(0, (4, 2.4)), zorder=4)
    ax.axhline(0.0, color="#333333", lw=0.8, zorder=4)
    be_base = breakeven_cutoff(PRICE_BASE)
    ax.annotate("dotted: break-even cut-off,\n"
                f"{be_base:.2f} % Cu at {PRICE_BASE / 1000:.1f} k$/t",
                xy=(be_base, 4.0), xytext=(be_base + 0.20, 22.0),
                ha="left", va="top", fontsize=6.2, color=SUB, zorder=8,
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor="none", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))

    ax.set_xlim(CUTOFF[0], CUTOFF[-1])
    ax.set_xlabel("cut-off grade (% Cu)", fontsize=7.4)
    ax.set_ylabel("undiscounted margin (bn USD)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.55)
    ax.legend(fontsize=6.2, frameon=False, loc="upper right",
              handlelength=1.4, borderaxespad=0.35, labelspacing=0.30)
    ax.set_title("Price scenarios move the optimum cut-off", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    grades = deposit()
    tonnes, avg, metal_kt = grade_tonnage(grades, CUTOFF)
    produced = recovered_metal(tonnes, avg)
    i_used = int(np.argmin(np.abs(CUTOFF - CUT_USED)))
    i_peak = int(np.nanargmax(produced))
    at_cut = dict(tonnes=tonnes[i_used], avg=avg[i_used],
                  metal=metal_kt[i_used])

    fig = plt.figure(figsize=(10.6, 6.6))
    ax_main = axes_inches(fig, (0.92, 0.94, 5.28, 4.66))
    ax_metal = axes_inches(fig, (7.40, 3.64, 2.76, 1.96))
    ax_price = axes_inches(fig, (7.40, 0.94, 2.76, 1.96))

    main_panel(ax_main, tonnes, avg, at_cut)
    metal_panel(ax_metal, tonnes, avg, produced, i_peak)
    price_panel(ax_price, tonnes, avg)

    panel_labels_inches(fig, [ax_main, ax_metal, ax_price], dx=-0.58,
                        dy=0.10)
    savefig(fig, "833_mnp_grade_tonnage.png")


if __name__ == "__main__":
    main()
