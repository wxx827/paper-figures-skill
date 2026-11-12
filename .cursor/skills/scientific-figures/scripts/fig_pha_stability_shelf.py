"""Shelf life from a confidence bound, not from a regression line.

Three registration batches are assayed on the ICH long-term schedule.  The
pooled least-squares line through all of them is drawn, but the retest period
is taken where the one-sided 95 % lower confidence bound of that line meets the
95 % of label claim release limit - always earlier than where the line itself
meets it, and the gap between the two crossings is annotated because it is the
single most common mistake in reading this plot.  The impurity panel and the
degradation-rate panel are regressed from the same simulated assays, so the
accelerated condition that fails on impurities is the same one that shows the
steepest slope.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
ORANGE = "#c0603f"
GREY = "#808080"
HAIR = "#cccccc"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

MONTHS_LT = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0])
MONTHS_ACC = np.array([0.0, 1.0, 2.0, 3.0, 6.0])
LSL_ASSAY = 95.0            # lower release limit, % of label claim
IMP_SPEC = 0.50             # total-impurity specification, %
IMP_ID = 0.10               # identification threshold, %
X_MAX = 48.0
ASSAY_SD = 0.75

# label, batch intercept offset (%), colour
BATCHES = [("batch PX-2301", 0.30, "#134a76"),
           ("batch PX-2302", 0.00, "#3b5b92"),
           ("batch PX-2303", -0.28, "#7fa7cc")]

# label, months sampled, true slope (%/month), colour
CONDITIONS = [("5 $^\\circ$C", MONTHS_LT, -0.048, "#cfe1ee"),
              ("25 $^\\circ$C / 60 %RH", MONTHS_LT, -0.142, "#5f9dc6"),
              ("30 $^\\circ$C / 65 %RH", MONTHS_LT, -0.262, "#2e74a8"),
              ("40 $^\\circ$C / 75 %RH", MONTHS_ACC, -0.615, "#134a76")]

C0_ASSAY = 100.35
IMP0, IMP_RATE_LT, IMP_RATE_ACC = 0.075, 0.0072, 0.0545


def assay_points(rng, months, slope, offset=0.0, sd=ASSAY_SD):
    return C0_ASSAY + offset + slope * months + rng.normal(0.0, sd,
                                                           months.size)


def pooled_fit(x, y):
    """OLS plus everything the one-sided confidence bound needs."""
    n = x.size
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (icept + slope * x)
    s = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    xbar = float(x.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    return dict(slope=float(slope), icept=float(icept), s=s, n=n, xbar=xbar,
                sxx=sxx, tcrit=float(stats.t.ppf(0.95, n - 2)))


def lower_bound(fit, xg):
    """One-sided 95 % lower confidence bound of the fitted mean response."""
    se = fit["s"] * np.sqrt(1.0 / fit["n"] +
                            (xg - fit["xbar"]) ** 2 / fit["sxx"])
    return fit["icept"] + fit["slope"] * xg - fit["tcrit"] * se


def cross_down(x, y, level):
    """First x where a decreasing y drops through ``level``."""
    idx = np.flatnonzero(y <= level)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(-level, [-y[i - 1], -y[i]], [x[i - 1], x[i]]))


def main_panel(ax, lt, fit, acc_pts, acc_fit, xg, low, t_shelf, t_line):
    ax.axhline(LSL_ASSAY, color=RED, lw=1.25, ls=(0, (5.0, 2.2)), zorder=6)
    ax.text(X_MAX - 0.6, LSL_ASSAY + 0.10,
            "release limit {:.1f} % of label claim (quality standard)".format(
                LSL_ASSAY),
            ha="right", va="bottom", fontsize=6.6, color=RED, zorder=9,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.92))

    ax.fill_between(xg, low, fit["icept"] + fit["slope"] * xg,
                    color="#e8eef5", lw=0.0, zorder=1)
    ax.plot(xg, fit["icept"] + fit["slope"] * xg, color=BLUE, lw=1.7,
            zorder=5)
    ax.plot(xg, low, color=BLUE, lw=1.1, ls=(0, (3.0, 1.8)), zorder=5)

    for (name, _off, col), y in zip(BATCHES, lt):
        ax.plot(MONTHS_LT, y, ls="none", marker="o", ms=4.0, mfc="white",
                mec=col, mew=1.1, zorder=7)

    ax.plot(MONTHS_ACC, acc_pts, ls="none", marker="s", ms=3.8, mfc="white",
            mec=ORANGE, mew=1.0, zorder=7)
    xa = np.linspace(0.0, 9.0, 40)
    ax.plot(xa, acc_fit["icept"] + acc_fit["slope"] * xa, color=ORANGE,
            lw=1.3, ls=(0, (1.8, 1.6)), zorder=6)
    ax.annotate("accelerated 40 $^\\circ$C / 75 %RH\n"
                "{:.2f} % per month".format(abs(acc_fit["slope"])),
                xy=(6.0, acc_pts[-1]), xytext=(9.5, 96.3), fontsize=6.4,
                color=ORANGE, ha="left", va="center", linespacing=1.35,
                zorder=9,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7,
                                shrinkA=1, shrinkB=3))

    for t, col, lab, dy in ((t_shelf, BLUE, "shelf life", 0.0),
                            (t_line, GREY, "line crossing", 0.0)):
        if not np.isfinite(t):
            continue
        ax.plot([t, t], [93.2, LSL_ASSAY], color=col, lw=0.8,
                ls=(0, (2.0, 1.8)), zorder=4)
        ax.plot([t], [LSL_ASSAY], marker="v", ms=6.0, mfc=col, mec="white",
                mew=0.9, zorder=10)
    ax.annotate("retest period {:.0f} months\n(lower bound crosses here)"
                .format(t_shelf), xy=(t_shelf, LSL_ASSAY),
                xytext=(t_shelf - 2.0, 97.4), fontsize=6.8, color=BLUE,
                ha="right", va="bottom", fontweight="bold", linespacing=1.35,
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.90),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8,
                                shrinkA=1, shrinkB=3))
    ax.annotate("the fitted line alone\nwould claim {:.0f} months"
                .format(t_line), xy=(t_line, LSL_ASSAY),
                xytext=(t_line + 1.4, 93.55), fontsize=6.4, color=SUB,
                ha="left", va="bottom", linespacing=1.35, zorder=10,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=1, shrinkB=3))

    handles = [Line2D([], [], color=BLUE, lw=1.7,
                      label="pooled regression, {:d} points".format(fit["n"])),
               Line2D([], [], color=BLUE, lw=1.1, ls=(0, (3.0, 1.8)),
                      label="one-sided 95 % lower confidence bound"),
               Line2D([], [], color=RED, lw=1.25, ls=(0, (5.0, 2.2)),
                      label="release limit (specification)")]
    handles += [Line2D([], [], color=col, lw=0.0, marker="o", ms=4.0,
                       mfc="white", mec=col, mew=1.1, label=name)
                for name, _o, col in BATCHES]
    handles.append(Line2D([], [], color=ORANGE, lw=0.0, marker="s", ms=3.8,
                          mfc="white", mec=ORANGE, mew=1.0,
                          label="accelerated condition"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False,
              bbox_to_anchor=(0.004, 0.004), handlelength=1.8,
              borderaxespad=0.0, labelspacing=0.28)

    ax.set_xlim(-1.2, X_MAX)
    ax.set_ylim(93.2, 101.9)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xlabel("storage time (months)")
    ax.set_ylabel("assay (% of label claim)")
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Long-term stability at 25 $^\\circ$C / 60 %RH, three "
                 "registration batches", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def impurity_panel(ax, imp_lt, imp_acc):
    ax.axhline(IMP_SPEC, color=RED, lw=1.2, ls=(0, (5.0, 2.2)), zorder=5)
    ax.text(35.4, IMP_SPEC + 0.012, "specification {:.2f} %".format(IMP_SPEC),
            ha="right", va="bottom", fontsize=6.0, color=RED, zorder=7)
    ax.axhline(IMP_ID, color=GREY, lw=1.0, ls=(0, (1.4, 1.6)), zorder=5)
    ax.text(35.4, IMP_ID + 0.014,
            "identification threshold {:.2f} %".format(IMP_ID), ha="right",
            va="bottom", fontsize=6.0, color=SUB, zorder=7)

    ax.plot(MONTHS_LT, imp_lt, color=BLUE, lw=1.4, marker="o", ms=3.0,
            mfc="white", mec=BLUE, mew=0.9, zorder=6,
            label="25 $^\\circ$C / 60 %RH")
    ax.plot(MONTHS_ACC, imp_acc, color=ORANGE, lw=1.4, ls=(0, (2.6, 1.8)),
            marker="s", ms=3.0, mfc="white", mec=ORANGE, mew=0.9, zorder=6,
            label="40 $^\\circ$C / 75 %RH")
    ax.set_xlim(-1.2, 37.0)
    ax.set_ylim(0.0, 0.62)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("storage time (months)", fontsize=7.2)
    ax.set_ylabel("total impurities (%)", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.1, frameon=False, handlelength=1.5,
              borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Impurity growth against two limit types", fontsize=8.0,
                 fontweight="bold", color=INK)


def rate_panel(ax, rates):
    y = np.arange(len(CONDITIONS))[::-1]
    vals = [abs(r) for r in rates]
    cols = [c[3] for c in CONDITIONS]
    ax.barh(y, vals, height=0.56, color=cols, edgecolor="white", linewidth=0.8,
            zorder=4)
    for yi, v, col in zip(y, vals, cols):
        ax.text(v + 0.012, yi, "{:.3f}".format(v), ha="left", va="center",
                fontsize=6.3, color=INK, fontweight="bold", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in CONDITIONS], fontsize=6.4)
    ax.set_xlim(0.0, max(vals) * 1.30)
    ax.set_ylim(-0.62, len(CONDITIONS) - 0.34)
    ax.set_xlabel("degradation rate (% of label claim per month)",
                  fontsize=7.2)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Slope of each storage condition", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Warn if the footnote box overlaps an in-view tick or axis label."""
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
    rng = np.random.default_rng(9451)

    slope_lt = CONDITIONS[1][2]
    lt = [assay_points(rng, MONTHS_LT, slope_lt, off)
          for _n, off, _c in BATCHES]
    x_all = np.tile(MONTHS_LT, len(BATCHES))
    y_all = np.concatenate(lt)
    fit = pooled_fit(x_all, y_all)

    # The accelerated series is simulated once and reused, so the slope
    # annotated in the main panel is the bar drawn in the rate panel.
    acc_pts = assay_points(rng, MONTHS_ACC, CONDITIONS[3][2], -0.10)
    acc_fit = pooled_fit(MONTHS_ACC, acc_pts)

    xg = np.linspace(0.0, X_MAX, 2400)
    low = lower_bound(fit, xg)
    line = fit["icept"] + fit["slope"] * xg
    t_shelf = cross_down(xg, low, LSL_ASSAY)
    t_line = cross_down(xg, line, LSL_ASSAY)

    rates = []
    for k, (_name, months, slope, _col) in enumerate(CONDITIONS):
        if k == 1:                       # the pooled long-term fit, reused
            rates.append(fit["slope"])
        elif k == 3:                     # the accelerated series, reused
            rates.append(acc_fit["slope"])
        else:
            pts = assay_points(rng, months, slope)
            rates.append(float(np.polyfit(months, pts, 1)[0]))

    imp_lt = np.clip(IMP0 + IMP_RATE_LT * MONTHS_LT +
                     rng.normal(0.0, 0.011, MONTHS_LT.size), 0.0, None)
    imp_acc = np.clip(IMP0 + IMP_RATE_ACC * MONTHS_ACC +
                      rng.normal(0.0, 0.014, MONTHS_ACC.size), 0.0, None)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_imp = axes_inches(fig, AUX_T)
    ax_rate = axes_inches(fig, AUX_B)

    main_panel(ax_main, lt, fit, acc_pts, acc_fit, xg, low, t_shelf, t_line)
    impurity_panel(ax_imp, imp_lt, imp_acc)
    rate_panel(ax_rate, rates)



    panel_labels_inches(fig, [ax_main, ax_imp, ax_rate], dx=-0.62, dy=0.14)
    savefig(fig, "945_pha_stability_shelf.png")


if __name__ == "__main__":
    main()
