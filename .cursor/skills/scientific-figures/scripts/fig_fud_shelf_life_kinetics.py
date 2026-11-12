"""Shelf-life prediction: one Arrhenius law drawn three ways.

A single rate law generates the whole page.  Five storage temperatures each
get a first-order decay of retained ascorbic acid, k(T) = A exp(-Ea/RT); the
shelf life at each temperature is then found by interpolating the *plotted*
curve against the acceptance limit rather than by inverting the formula, so
the marked crossings sit exactly where the eye puts them.  Those five rate
constants are regressed on 1/T in the upper right panel, which recovers the
activation energy and the Q10 that were used to make them - the fit is a
consistency check on the main panel, not an independent claim - and the same
fit, with its covariance propagated through, draws the predicted shelf-life
curve and its 95 % band in the lower right panel, where the five measured
shelf lives are overlaid to show they fall on it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

# Cold to hot: the dark end of the house blue sequential ramp running into the
# dark end of the warm one.  A single-hue ramp leaves the coldest cell almost
# white, and the coldest cell is the one the reader came for.
TEMP_COLOURS = ["#134a76", "#5f9dc6", "#c9a227", "#cc722f", "#8c4a12"]

RED = "#b23a48"
GREEN = "#2e7d5b"
BLUE = "#3b5b92"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

R_GAS = 8.314                     # J / (mol K)
EA_TRUE = 68000.0                 # J / mol
K_REF = 0.0165                    # 1/day at the reference temperature
T_REF = 293.15                    # K
C0 = 42.0                         # mg / 100 g ascorbic acid at packing
LIMIT_FRAC = 0.50                 # acceptance limit, fraction of C0
TEMPS_C = np.array([4.0, 10.0, 20.0, 30.0, 37.0])
T_MAX_D = 260.0
N_REP = 3                         # analytical replicates per sampling point
CV_ASSAY = 0.070                  # relative SD of the titration
SAMPLE_D = np.array([0.0, 7.0, 14.0, 30.0, 60.0, 90.0, 150.0, 210.0, 260.0])

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.30, 3.61, 2.68, 1.89)
AUX_B = (7.30, 1.02, 2.68, 1.82)


def rate(temp_c):
    """First-order loss constant in 1/day from the Arrhenius law."""
    tk = temp_c + 273.15
    return K_REF * np.exp(-EA_TRUE / R_GAS * (1.0 / tk - 1.0 / T_REF))


def estimate_rates(rng):
    """Measured replicates per cell, and the k each cell's own log fit gives.

    Taking k from the simulated titrations rather than from ``rate`` is what
    makes the Arrhenius regression a fit with residuals - and therefore the
    confidence band in the third panel a band with a width.
    """
    k_hat, reps = [], []
    for tc in TEMPS_C:
        true = C0 * np.exp(-rate(tc) * SAMPLE_D)
        r = true[None, :] * rng.normal(1.0, CV_ASSAY, (N_REP, SAMPLE_D.size))
        mean = r.mean(axis=0)
        keep = mean > 0.10 * C0          # below this the log is all noise
        slope = np.polyfit(SAMPLE_D[keep], np.log(mean[keep]), 1)[0]
        k_hat.append(-float(slope))
        reps.append(r)
    return np.array(k_hat), reps


def cross(x, y, level):
    """First x where a monotone decreasing y drops through ``level``."""
    idx = np.flatnonzero(y <= level)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(-level, [-y[i - 1], -y[i]], [x[i - 1], x[i]]))


def label_ladder(values, min_gap):
    """Push end-of-curve label heights apart without changing their order."""
    order = np.argsort(values)
    y = np.array(values, dtype=float)
    for j in range(1, order.size):
        a, b = order[j - 1], order[j]
        y[b] = max(y[b], y[a] + min_gap)
    return y


def main_panel(ax, days, curves, life, colours, reps):
    limit = LIMIT_FRAC * C0
    ends = label_ladder([c[-1] for c in curves], 0.058 * C0)
    ax.axhline(limit, color=RED, lw=1.2, ls=(0, (5.0, 2.2)), zorder=5)
    ax.text(T_MAX_D - 3.0, limit + 0.9,
            "acceptance limit {:.0f} % of $C_0$ = {:.1f} mg/100 g".format(
                100.0 * LIMIT_FRAC, limit),
            ha="right", va="bottom", fontsize=6.6, color=RED, zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.90))

    for i, tc in enumerate(TEMPS_C):
        col = colours[i]
        ax.plot(days, curves[i], color=col, lw=1.6, zorder=4)
        rep = reps[i]
        ax.errorbar(SAMPLE_D, rep.mean(axis=0), yerr=rep.std(axis=0, ddof=1),
                    fmt="o", ms=3.2, mfc="white", mec=col, mew=0.9,
                    ecolor=col, elinewidth=0.7, capsize=1.8, zorder=6)
        if np.isfinite(life[i]):
            ax.plot([life[i], life[i]], [0.0, limit], color=col, lw=0.7,
                    ls=(0, (2.0, 1.8)), zorder=3)
            ax.plot([life[i]], [limit], marker="v", ms=5.0, mfc=col,
                    mec="white", mew=0.8, zorder=7)
            ax.annotate("{:.0f} d".format(life[i]), xy=(life[i], limit),
                        xytext=(0.0, -9.0 - 11.0 * (i % 2)),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6.4, color=col, zorder=7)
        ax.text(T_MAX_D + 4.0, ends[i], "{:.0f} $^\\circ$C".format(tc),
                ha="left", va="center", fontsize=6.8, color=col,
                fontweight="bold", zorder=7, clip_on=False)

    ax.set_xlim(-4.0, T_MAX_D)
    ax.set_ylim(0.0, C0 * 1.14)
    ax.set_xlabel("storage time (days)")
    ax.set_ylabel("retained ascorbic acid (mg/100 g)")
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=SUB, lw=1.6, label="first-order fit"),
               Line2D([], [], color=SUB, lw=0.0, marker="o", ms=3.6,
                      mfc="white", mec=SUB, mew=0.9,
                      label="mean of {:d} replicates $\\pm$ 1 SD".format(
                          N_REP)),
               Line2D([], [], color=SUB, lw=0.0, marker="v", ms=4.6, mfc=SUB,
                      mec="white", label="shelf life at the limit")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=False,
              bbox_to_anchor=(0.995, 0.995), handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.30)
    ax.set_title("Vitamin C loss at five storage temperatures",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def arrhenius_panel(ax, k_obs, colours, slope, icept, ea_fit, q10):
    inv = 1000.0 / (TEMPS_C + 273.15)
    grid = np.linspace(inv.min() - 0.03, inv.max() + 0.03, 60)
    ax.plot(grid, icept + slope * grid, color=INK, lw=1.2, zorder=4)
    for i, tc in enumerate(TEMPS_C):
        ax.plot([inv[i]], [np.log(k_obs[i])], marker="o", ms=5.0,
                mfc=colours[i], mec="white", mew=0.8, zorder=6)
        ax.annotate("{:.0f} $^\\circ$C".format(tc),
                    xy=(inv[i], np.log(k_obs[i])), xytext=(0.0, -8.0),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6, color=colours[i], zorder=7)
    ax.text(0.035, 0.055,
            "$E_a$ = $-R\\,$slope = {:.1f} kJ/mol\n"
            "$Q_{{10}}$ at 20 $^\\circ$C = {:.2f}".format(ea_fit / 1000.0,
                                                          q10),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6,
            color=INK, linespacing=1.40, zorder=8)
    ax.set_xlabel("$10^3/T$ (1/K)", fontsize=7.4)
    ax.set_ylabel("$\\ln k$ ($k$ in 1/day)", fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Arrhenius plot of the same five rates", fontsize=8.0,
                 fontweight="bold", color=INK)


def prediction_panel(ax, life, colours, slope, icept, cov):
    tc = np.linspace(0.0, 40.0, 200)
    inv = 1000.0 / (tc + 273.15)
    ln_k = icept + slope * inv
    ln_sl = np.log(-np.log(LIMIT_FRAC)) - ln_k
    var = cov[0, 0] * inv ** 2 + 2.0 * cov[0, 1] * inv + cov[1, 1]
    band = 1.96 * np.sqrt(np.maximum(var, 0.0))
    ax.fill_between(tc, np.exp(ln_sl - band), np.exp(ln_sl + band),
                    color=lighten(BLUE, 0.78), lw=0.0, zorder=2)
    for sgn in (-1.0, 1.0):
        ax.plot(tc, np.exp(ln_sl + sgn * band), color=lighten(BLUE, 0.42),
                lw=0.6, zorder=3)
    ax.plot(tc, np.exp(ln_sl), color=BLUE, lw=1.5, zorder=4)
    for i, t in enumerate(TEMPS_C):
        ax.plot([t], [life[i]], marker="o", ms=4.6, mfc=colours[i],
                mec="white", mew=0.8, zorder=6)
    j = int(np.argmin(np.abs(tc - 25.0)))
    ax.annotate("25 $^\\circ$C abuse:\n{:.0f} days".format(np.exp(ln_sl[j])),
                xy=(25.0, np.exp(ln_sl[j])), xytext=(9.0, 6.0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.4, color=INK, linespacing=1.30, zorder=7,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    ax.set_yscale("log")
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(4.0, 520.0)
    ax.set_xlabel("storage temperature ($^\\circ$C)", fontsize=7.4)
    ax.set_ylabel("predicted shelf life (days)", fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=BLUE, lw=1.5, label="Arrhenius fit"),
               Line2D([], [], color=lighten(BLUE, 0.82), lw=5.0,
                      label="95 % band"),
               Line2D([], [], color=SUB, lw=0.0, marker="o", ms=4.0,
                      mfc=SUB, mec="white", label="from panel (a)")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.2, frameon=False,
              handlelength=1.3, borderaxespad=0.2, labelspacing=0.22)
    ax.set_title("Shelf life against temperature", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(914)
    colours = TEMP_COLOURS

    days = np.linspace(0.0, T_MAX_D, 1400)
    k_obs, reps = estimate_rates(rng)
    curves = [C0 * np.exp(-k * days) for k in k_obs]
    life = np.array([cross(days, c, LIMIT_FRAC * C0) for c in curves])

    inv = 1000.0 / (TEMPS_C + 273.15)
    (slope, icept), cov = np.polyfit(inv, np.log(k_obs), 1, cov=True)
    ea_fit = -slope * R_GAS * 1000.0
    q10 = float(np.exp(ea_fit * 10.0 / (R_GAS * T_REF * (T_REF + 10.0))))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_arr = axes_inches(fig, AUX_T)
    ax_pred = axes_inches(fig, AUX_B)

    main_panel(ax_main, days, curves, life, colours, reps)
    arrhenius_panel(ax_arr, k_obs, colours, slope, icept, ea_fit, q10)
    prediction_panel(ax_pred, life, colours, slope, icept, cov)

    panel_labels_inches(fig, [ax_main, ax_arr, ax_pred], dx=-0.62, dy=0.14)
    savefig(fig, "914_fud_shelf_life_kinetics.png")


if __name__ == "__main__":
    main()
