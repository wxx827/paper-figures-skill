"""Vaccination coverage by age band and dose, and the gap left to the target.

Main panel: coverage heat map, age band x dose, annotated with the value in
each cell. Auxiliary panels: population-weighted cumulative uptake over the
campaign year for each dose, and the remaining distance to the vaccination
target implied by a herd-immunity threshold of 1 - 1/R0 divided by vaccine
effectiveness. Every number is read off the same coverage array.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from figkit import panel_label_inches, safe_colorbar
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, get_palette

AGES = ["5-11", "12-17", "18-29", "30-44", "45-59", "60-69", "70-79", "80+"]
POP = np.array([0.085, 0.070, 0.165, 0.205, 0.195, 0.130, 0.095, 0.055])
DOSES = ["Dose 1", "Dose 2", "Booster 1", "Booster 2"]

R0 = 3.5
VE = 0.85
TARGET = 100.0 * (1.0 - 1.0 / R0) / VE

INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
ALERT = "#b23a48"
GREEN = "#2e7d5b"
DASH = (0, (4.0, 2.4))

N_WEEK = 53
START = "2021-01-04"

# per-age ceiling for dose 1, then dose-specific multipliers and timing
CEIL1 = np.array([0.62, 0.74, 0.80, 0.86, 0.90, 0.94, 0.96, 0.95])
MULT = np.array([1.00, 0.955, 0.700, 0.360])
T50_AGE = np.array([31.0, 27.0, 24.0, 21.0, 17.0, 12.0, 8.5, 6.5])
T50_DOSE = np.array([0.0, 4.5, 20.0, 33.0])
BOOSTER2_ELIG = np.array([0.10, 0.15, 0.35, 0.55, 0.80, 1.0, 1.0, 1.0])


def coverage_curves():
    """coverage[age, dose, week] as a fraction, from logistic uptake."""
    t = np.arange(N_WEEK, dtype=float)
    cov = np.zeros((len(AGES), len(DOSES), N_WEEK))
    for a in range(len(AGES)):
        for d in range(len(DOSES)):
            ceiling = CEIL1[a] * MULT[d]
            if d == 3:
                ceiling *= BOOSTER2_ELIG[a]
            t50 = T50_AGE[a] + T50_DOSE[d]
            cov[a, d] = ceiling / (1.0 + np.exp(-(t - t50) / 3.4))
    return cov


def main():
    set_pub_style(base_font=9)
    cov = coverage_curves()
    final = 100.0 * cov[:, :, -1]
    weeks = pd.date_range(START, periods=N_WEEK, freq="7D")

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.78],
                          height_ratios=[1.0, 0.74], hspace=0.55, wspace=0.30,
                          left=0.088, right=0.985, top=0.845, bottom=0.205)
    ax = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # ---- (a) coverage heat map ---------------------------------------------
    im = ax.imshow(final, cmap=get_cmap("cb_sequential_green"), aspect="auto",
                   vmin=0.0, vmax=100.0, origin="upper")
    ax.set_xticks(range(len(DOSES)))
    ax.set_xticklabels(DOSES, fontsize=7.0)
    ax.set_yticks(range(len(AGES)))
    ax.set_yticklabels(AGES, fontsize=7.0)
    ax.set_ylabel("Age band (years)", fontsize=8.0)
    ax.tick_params(length=2.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    for a in range(len(AGES)):
        for d in range(len(DOSES)):
            v = final[a, d]
            ax.text(d, a, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if v > 55 else INK)
    ax.set_title("Coverage by age band and dose (%)", fontsize=9.0,
                 fontweight="bold", loc="left", pad=4)
    safe_colorbar(fig, im, ax, label="coverage (% of age band)",
                  side="bottom", width=0.10, pad=0.30, shrink=1.0)

    # ---- (b) cumulative uptake over the campaign ---------------------------
    cols = get_palette("cb_sequential_green", 5)[1:]
    nat = 100.0 * np.tensordot(POP, cov, axes=(0, 0))       # dose x week
    for d in range(len(DOSES)):
        axb.plot(weeks, nat[d], color=cols[d], lw=1.6, zorder=4,
                 label=DOSES[d])
    axb.axhline(TARGET, color=RULE, ls=DASH, lw=0.9, zorder=5)
    axb.text(weeks[-2], TARGET + 2.5,
             f"target {TARGET:.0f}% = $(1-1/R_0)$ / VE", fontsize=6.4,
             color=RULE, va="bottom", ha="right")
    axb.set_ylim(0, 100)
    axb.set_xlim(weeks[0], weeks[-1])
    axb.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axb.set_ylabel("Population covered (%)", fontsize=7.6)
    axb.set_xlabel("Campaign month (2021)", fontsize=7.6)
    axb.set_title("National uptake by dose", fontsize=8.5, fontweight="bold",
                  loc="left", pad=4)
    axb.tick_params(labelsize=6.6)
    # framed: the only free space overlaps the target rule (DESIGN_SYSTEM 6.1)
    axb.legend(loc="upper left", fontsize=6.4, frameon=True, framealpha=0.92,
               facecolor="white", edgecolor="#cccccc", handlelength=1.3,
               labelspacing=0.22, borderpad=0.35, borderaxespad=0.3)
    style_axes(axb)

    # ---- (c) distance left to the target -----------------------------------
    gap = TARGET - final[:, 1]
    ypos = np.arange(len(AGES))
    colr = [ALERT if g > 0 else GREEN for g in gap]
    axc.barh(ypos, gap, height=0.68, color=colr, zorder=3)
    axc.axvline(0.0, color=RULE, lw=0.8, zorder=4)
    for y, g in zip(ypos, gap):
        off = 0.9 if g > 0 else -0.9
        axc.text(g + off, y, f"{g:+.0f}", va="center",
                 ha="left" if g > 0 else "right", fontsize=6.2, color=INK)
    axc.invert_yaxis()
    axc.set_yticks(ypos)
    axc.set_yticklabels(AGES, fontsize=6.4)
    axc.set_xlim(gap.min() * 1.45, gap.max() * 1.35)
    axc.set_xlabel("Distance to target (percentage points)", fontsize=7.6)
    axc.set_title("Gap in the primary series", fontsize=8.5,
                  fontweight="bold", loc="left", pad=4)
    axc.tick_params(axis="x", labelsize=6.6)
    axc.tick_params(axis="y", length=0)
    n_short = int((gap > 0).sum())
    axc.text(0.985, 0.06, f"{n_short} of {len(AGES)} bands still short",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=6.4,
             color=MUTED)
    style_axes(axc)

    panel_label_inches(fig, ax, "(a)", dx=-0.44, dy=0.07)
    panel_label_inches(fig, axb, "(b)", dx=-0.44, dy=0.07)
    panel_label_inches(fig, axc, "(c)", dx=-0.44, dy=0.07)

    savefig(fig, "428_epi_vaccine_coverage.png")


if __name__ == "__main__":
    main()
