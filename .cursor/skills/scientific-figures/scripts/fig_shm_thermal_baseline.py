"""Four years of expansion-joint monitoring, before and after thermal removal.

Main panel: the daily joint opening with the air temperature behind it and the
thermal regression drawn through it, and beneath, on the same axis but offset
so the two can be compared day for day, the residual left once that regression
is subtracted, with the control limits set on the first two years and the
exceedances picked out.  Auxiliary (both at 58 % of the main panel width): the
opening against effective temperature year by year, and the monthly mean
residual as a control chart.

The point of the figure is that the raw record hides the fault completely: a
few thousandths of a millimetre per day, starting in the fourth year, is a
fraction of the seasonal swing and only the residual shows it.  Every number
on the figure is computed rather than asserted - the model
is least squares over the baseline window, the limits are three standard
deviations of the baseline residual, the alarm days are the first crossings of
those limits, and the detection lags are the gap back to the day the drift was
injected.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GREY = "#9a9a9a"

NDAY = 1461
T_MEAN, T_AMP, T_PHASE = 14.0, 12.0, 110.0
TAU = 3.0                 # days, structural thermal lag
K_TRUE = -0.62            # mm per degree C
NOISE = 0.30              # mm
T_ON = 1150               # day the drift starts
DRIFT = 0.0055            # mm per day
BASE_END = 730            # baseline window for the model and the limits
OFFSET = -17.0            # mm, where the residual trace is drawn

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def series(rng):
    """Air temperature, effective temperature and joint opening."""
    t = np.arange(NDAY, dtype=float)
    weather = gaussian_filter1d(rng.normal(0.0, 1.0, NDAY), 4.0)
    weather *= 2.4 / (weather.std() + 1e-12)
    t_air = T_MEAN + T_AMP * np.sin(2.0 * np.pi * (t - T_PHASE) / 365.25) \
        + weather

    t_eff = np.empty(NDAY)
    t_eff[0] = t_air[0]
    for i in range(1, NDAY):
        t_eff[i] = t_eff[i - 1] + (t_air[i] - t_eff[i - 1]) / TAU

    drift = np.where(t >= T_ON, DRIFT * (t - T_ON), 0.0)
    disp = K_TRUE * (t_eff - T_MEAN) + drift + rng.normal(0.0, NOISE, NDAY)
    return t, t_air, t_eff, disp, drift


def fit_thermal(t_eff, disp):
    """Least-squares thermal model over the baseline window."""
    b = slice(0, BASE_END)
    k, c = np.polyfit(t_eff[b], disp[b], 1)
    pred = c + k * t_eff
    res = disp - pred
    sd = float(res[b].std(ddof=1))
    ss = 1.0 - np.var(res[b]) / np.var(disp[b])
    return k, c, pred, res, sd, float(ss)


def monthly(t, res):
    """Monthly mean residual, its day index and the number of samples."""
    edges = np.arange(0, NDAY + 1, 30)
    mid = 0.5 * (edges[:-1] + edges[1:])
    val = np.array([res[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    return mid, val, 30


def main_panel(ax, t, t_air, disp, pred, res, sd, alarm):
    axt = ax.twinx()
    axt.plot(t, t_air, color=lighten(BLUE, 0.72), lw=0.7, zorder=1)
    axt.set_ylim(-38.0, 46.0)
    axt.set_ylabel("air temperature (degrees C)", color=GREY, fontsize=8.0)
    axt.tick_params(axis="y", colors=GREY, labelsize=6.6)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_color(GREY)

    ax.plot(t, disp, color=BLUE, lw=0.6, alpha=0.85, zorder=4)
    ax.plot(t, pred, color=INK, lw=1.0, ls=(0, (4.0, 2.0)), zorder=5)

    ax.axhline(OFFSET, color=GREY, lw=0.7, zorder=3)
    for s in (-3.0, 3.0):
        ax.axhline(OFFSET + s * sd, color=RED, lw=0.8, ls=(0, (3.0, 2.0)),
                   zorder=4)
    ax.plot(t, OFFSET + res, color=ORANGE, lw=0.6, zorder=6)
    hot = np.abs(res) > 3.0 * sd
    ax.plot(t[hot], OFFSET + res[hot], lw=0.0, marker="o", ms=1.7, mfc=RED,
            mec="none", zorder=7)

    ax.axvspan(0, BASE_END, facecolor=TEAL, alpha=0.07, lw=0.0, zorder=0)
    ax.text(BASE_END / 2.0, 9.4, "baseline window for the model and the "
            "limits", fontsize=6.3, color=TEAL, ha="center", va="top",
            zorder=8)
    ax.axvline(T_ON, color=RED, lw=0.9, ls=(0, (2.5, 2.0)), zorder=5)
    ax.annotate("drift of {:.4f} mm/day starts here".format(DRIFT),
                xy=(T_ON, 6.0), xytext=(T_ON - 40.0, 9.0), fontsize=6.4,
                color=RED, ha="right", va="top", zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7, shrinkA=2,
                                shrinkB=3))
    ax.annotate("first daily exceedance on day {:d},\n{:d} days after the "
                "drift began".format(alarm, alarm - T_ON),
                xy=(alarm, OFFSET + res[alarm]),
                xytext=(alarm - 90.0, OFFSET - 5.2), fontsize=6.4, color=RED,
                ha="right", va="top", zorder=9, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7, shrinkA=2,
                                shrinkB=3))
    ax.text(24.0, OFFSET + 1.75, "residual, offset by {:.0f} mm"
            .format(OFFSET), fontsize=6.4, color=ORANGE, ha="left",
            va="bottom", zorder=8)

    ax.set_xlim(0.0, NDAY)
    ax.set_ylim(-24.0, 10.5)
    ax.set_xticks(np.arange(0, NDAY, 365))
    ax.set_yticks([-22, -20, -18, -16, -14, -8, -4, 0, 4, 8])
    ax.set_xlabel("days since installation")
    ax.set_ylabel("joint opening (mm), and the same residual offset below")
    ax.grid(axis="x", lw=0.5, alpha=0.30)
    for side in ("top",):
        ax.spines[side].set_visible(False)
    ax.set_title("Raw record, thermal model and what is left over",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    handles = [Line2D([], [], color=BLUE, lw=0.9, label="measured opening"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (4.0, 2.0)),
                      label="thermal model"),
               Line2D([], [], color=ORANGE, lw=0.9, label="residual"),
               Line2D([], [], color=RED, lw=0.8, ls=(0, (3.0, 2.0)),
                      label="3 sigma limits"),
               Line2D([], [], color=lighten(BLUE, 0.72), lw=0.9,
                      label="air temperature")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.3, ncol=3,
              frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.94, handlelength=1.7, columnspacing=1.0,
              borderaxespad=0.35)


def scatter_panel(ax, t, t_eff, disp, k, c):
    cols = [lighten(TEAL, 0.45), TEAL, BLUE, RED]
    for y in range(4):
        m = (t >= 365 * y) & (t < 365 * (y + 1))
        ax.plot(t_eff[m], disp[m], lw=0.0, marker="o", ms=1.7, mfc=cols[y],
                mec="none", alpha=0.75, zorder=3,
                label="year {:d}".format(y + 1))
    xg = np.linspace(t_eff.min(), t_eff.max(), 40)
    ax.plot(xg, c + k * xg, color=INK, lw=1.3, zorder=5)
    ax.text(0.035, 0.06, "baseline fit {:.3f} mm per degree C"
            .format(k), transform=ax.transAxes, fontsize=6.2, color=INK,
            ha="left", va="bottom", zorder=7)
    ax.set_xlabel("effective structure temperature (degrees C)", fontsize=6.8)
    ax.set_ylabel("joint opening (mm)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=0.8, borderaxespad=0.3, labelspacing=0.22, ncol=2,
              columnspacing=0.8, markerscale=2.2)
    ax.set_title("Year four sits above its own thermal line", fontsize=8.0,
                 fontweight="bold", color=INK)


def chart_panel(ax, mid, val, lim, alarm_m):
    ok = np.abs(val) <= lim
    ax.axhspan(-lim, lim, facecolor=TEAL, alpha=0.10, lw=0.0, zorder=1)
    ax.axhline(0.0, color=GREY, lw=0.8, zorder=3)
    for s in (-lim, lim):
        ax.axhline(s, color=RED, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.plot(mid, val, color=SUB, lw=0.8, zorder=5)
    ax.plot(mid[ok], val[ok], lw=0.0, marker="o", ms=3.0, mfc="white",
            mec=SUB, mew=0.8, zorder=6)
    ax.plot(mid[~ok], val[~ok], lw=0.0, marker="o", ms=3.6, mfc=RED,
            mec="white", mew=0.6, zorder=7)
    ax.axvline(T_ON, color=RED, lw=0.9, ls=(0, (2.5, 2.0)), zorder=4)
    ax.annotate("monthly mean crosses on day {:.0f},\nonly {:.0f} days after "
                "the drift began".format(alarm_m, alarm_m - T_ON),
                xy=(alarm_m, lim), xytext=(alarm_m - 60.0, lim + 0.62),
                fontsize=6.0, color=RED, ha="right", va="bottom", zorder=8,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7, shrinkA=2,
                                shrinkB=3))
    ax.text(30.0, -lim - 0.06, "3 sigma of the monthly mean, {:.2f} mm"
            .format(lim), fontsize=6.0, color=RED, ha="left", va="top",
            zorder=8)
    ax.set_xlim(0.0, NDAY)
    ax.set_ylim(-1.05, 1.85)
    ax.set_xticks(np.arange(0, NDAY, 365))
    ax.set_xlabel("days since installation", fontsize=6.8)
    ax.set_ylabel("monthly mean residual (mm)", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Averaging finds it a month sooner", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
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
    rng = np.random.default_rng(19870)

    t, t_air, t_eff, disp, drift = series(rng)
    k, c, pred, res, sd, r2 = fit_thermal(t_eff, disp)

    over = np.where((t >= T_ON) & (np.abs(res) > 3.0 * sd))[0]
    alarm = int(over[0])

    mid, val, n_m = monthly(t, res)
    lim = 3.0 * sd / np.sqrt(n_m)
    over_m = np.where((mid >= T_ON) & (np.abs(val) > lim))[0]
    alarm_m = float(mid[over_m[0]])

    swing = float(disp[:BASE_END].max() - disp[:BASE_END].min())
    print("k {:.3f}  sd {:.3f}  R2 {:.3f}  alarm {:d} / {:.0f}"
          .format(k, sd, r2, alarm, alarm_m))
    assert alarm > T_ON and alarm_m > T_ON, "no alarm before the fault"
    assert alarm_m < alarm, "averaging must detect sooner"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sc = axes_inches(fig, AUX_T)
    ax_cc = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, t_air, disp, pred, res, sd, alarm)
    scatter_panel(ax_sc, t, t_eff, disp, k, c)
    chart_panel(ax_cc, mid, val, lim, alarm_m)



    panel_labels_inches(fig, [ax_main, ax_sc, ax_cc], dx=-0.62, dy=0.14)
    savefig(fig, "989_shm_thermal_baseline.png")


if __name__ == "__main__":
    main()
