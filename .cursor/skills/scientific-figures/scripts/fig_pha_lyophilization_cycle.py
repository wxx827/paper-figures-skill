"""Freeze-drying cycle: the shelf drives it, the product temperature limits it.

One 42 h cycle is generated once and read three ways.  The upper panel carries
the shelf set point and four product thermocouples that lag it, each vial
position holding its own sublimation plateau against the collapse temperature;
the lower panel carries the two vacuum gauges, whose convergence is what
actually declares the end of primary drying, detected by interpolating the
plotted difference rather than by naming an hour.  The margin bars are the
distance from each position's own plateau maximum to the collapse temperature,
so the position drawn crossing the red rule in the upper panel is the one with
a negative bar, and the moisture panel reads its endpoint off a fitted decay.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
RED = "#b23a48"
ORANGE = "#c0603f"
GREY = "#808080"

FIG_W, FIG_H = 11.0, 6.5
MAIN_T = (0.88, 3.12, 4.62, 2.40)
MAIN_B = (0.88, 1.02, 4.62, 1.80)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

T_END = 42.0
T_FREEZE, T_PRIMARY = 5.0, 30.0     # phase boundaries in hours
T_COLLAPSE = -32.0                  # collapse temperature, deg C
P_PRIMARY, P_SECONDARY = 100.0, 60.0    # chamber set points, mTorr
CONV_TOL = 5.0                      # gauge convergence tolerance, mTorr
MOIST_SPEC = 1.00                   # residual moisture specification, % w/w

SHELF_KNOTS_T = [0.0, 1.0, 2.6, T_FREEZE, 6.2, T_PRIMARY, 32.2, T_END]
SHELF_KNOTS_Y = [20.0, 20.0, -45.0, -45.0, -20.0, -20.0, 30.0, 30.0]

# label, sublimation plateau (deg C), colour
POSITIONS = [("centre vials", -37.8, "#134a76"),
             ("middle vials", -36.4, "#3b5b92"),
             ("edge vials", -34.6, "#7fa7cc"),
             ("door-front vials", -31.6, ORANGE)]

MOIST_HOURS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
N_VIAL = 3


def shelf_profile(t):
    return np.interp(t, SHELF_KNOTS_T, SHELF_KNOTS_Y)


def product_profile(t, shelf, plateau, t_dry_end, rng, tau=0.13):
    """Thermocouple trace: supercooling, nucleation, plateau, then catch-up."""
    raw = np.where(t < t_dry_end, plateau + 0.055 * (t - T_FREEZE), shelf)
    raw = np.where(t < T_FREEZE - 0.4, shelf, raw)
    # Supercooling to well below the equilibrium point, then the nucleation
    # exotherm snapping the vial back toward the freezing point.
    nuc = (t >= 0.5) & (t <= 3.4)
    raw[nuc] = np.interp(t[nuc], [0.5, 1.9, 1.96, 2.5, 3.4],
                         [14.0, -16.0, -1.2, -14.0, shelf_profile(3.4)])
    y = np.empty_like(raw)
    y[0] = 20.0
    a = (t[1] - t[0]) / (tau + t[1] - t[0])
    for i in range(1, t.size):
        y[i] = y[i - 1] + a * (raw[i] - y[i - 1])
    return y + rng.normal(0.0, 0.085, t.size)


def gauges(t, rng):
    """Capacitance manometer set point and the Pirani gauge that trails it."""
    cm = np.where(t < T_PRIMARY, P_PRIMARY, P_SECONDARY)
    cm = np.where(t < T_FREEZE - 0.2, np.nan, cm)
    ratio = 1.0 + 0.62 / (1.0 + np.exp((t - 26.5) / 1.10))
    pir = cm * ratio + rng.normal(0.0, 0.9, t.size)
    return cm, pir


def convergence_time(t, cm, pir):
    """First hour where the two gauges agree to within CONV_TOL mTorr."""
    d = np.abs(pir - cm)
    ok = np.isfinite(d) & (t > T_FREEZE + 2.0)
    idx = np.flatnonzero(ok & (d <= CONV_TOL))
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return float(np.interp(-CONV_TOL, [-d[i - 1], -d[i]], [t[i - 1], t[i]]))


def decay(h, floor, amp, tau):
    return floor + amp * np.exp(-np.asarray(h, dtype=float) / tau)


def phase_bands(ax, label=False):
    spans = [(0.0, T_FREEZE, "#f2f5f8", "freezing"),
             (T_FREEZE, T_PRIMARY, "#eef3f0", "primary drying"),
             (T_PRIMARY, T_END, "#faf5ea", "secondary drying")]
    for x0, x1, col, name in spans:
        ax.axvspan(x0, x1, color=col, zorder=0)
        if label:
            ax.text(0.5 * (x0 + x1), 0.965, name, ha="center", va="top",
                    fontsize=6.8, color=SUB, fontweight="bold", zorder=6,
                    transform=ax.get_xaxis_transform())
    for x in (T_FREEZE, T_PRIMARY):
        ax.axvline(x, color="#c8ced4", lw=0.8, zorder=1)


def temperature_panel(ax, t, shelf, prod, t_conv):
    phase_bands(ax, label=True)
    ax.axhline(T_COLLAPSE, color=RED, lw=1.25, ls=(0, (5.0, 2.2)), zorder=6)
    ax.text(T_END - 0.4, T_COLLAPSE + 5.5,
            "collapse temperature $T_c$ = {:.0f} $^\\circ$C".format(
                T_COLLAPSE),
            ha="right", va="bottom", fontsize=6.5, color=RED, zorder=9,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.92))
    ax.plot(t, shelf, color=INK, lw=1.9, zorder=5)
    for (name, _p, col), y in zip(POSITIONS, prod):
        ax.plot(t, y, color=col, lw=1.15, zorder=4)

    ax.axvline(t_conv, color=TEAL, lw=1.1, ls=(0, (3.2, 1.8)), zorder=7)
    ax.annotate("primary drying ends {:.1f} h\n({:.1f} h before the shelf "
                "ramp)".format(t_conv, T_PRIMARY - t_conv),
                xy=(t_conv, -12.0), xytext=(t_conv - 1.4, 4.0),
                fontsize=6.4, color=TEAL, ha="right", va="center",
                linespacing=1.30, zorder=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7,
                                shrinkA=1, shrinkB=2))

    handles = [Line2D([], [], color=INK, lw=1.9, label="shelf set point")]
    handles += [Line2D([], [], color=col, lw=1.15, label=name)
                for name, _p, col in POSITIONS]
    handles.append(Line2D([], [], color=RED, lw=1.25, ls=(0, (5.0, 2.2)),
                          label="collapse temperature (product limit)"))
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=False,
              bbox_to_anchor=(0.055, 0.905), handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.26, ncol=2,
              columnspacing=1.2)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(-52.0, 42.0)
    ax.set_xticks(np.arange(0, 43, 6))
    ax.set_yticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40])
    ax.set_ylabel("temperature ($^\\circ$C)", fontsize=8.0)
    ax.tick_params(labelbottom=False)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cycle recipe and four product thermocouples", fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def pressure_panel(ax, t, cm, pir, t_conv):
    phase_bands(ax)
    ax.plot(t, pir, color=GOLD, lw=1.3, zorder=5, label="Pirani gauge")
    ax.plot(t, cm, color=BLUE, lw=1.6, zorder=6,
            label="capacitance manometer (set point)")
    ax.fill_between(t, cm, pir, where=np.isfinite(cm), color=lighten(GOLD,
                    0.78), lw=0.0, zorder=3)
    ax.axvline(t_conv, color=TEAL, lw=1.1, ls=(0, (3.2, 1.8)), zorder=7)
    ax.plot([t_conv], [P_PRIMARY], marker="v", ms=6.0, mfc=TEAL, mec="white",
            mew=0.9, zorder=9)
    ax.annotate("gauges agree to {:.0f} mTorr".format(CONV_TOL),
                xy=(t_conv, P_PRIMARY), xytext=(t_conv - 1.4, 136.0),
                fontsize=6.3, color=TEAL, ha="right", va="center", zorder=9,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 200.0)
    ax.set_xticks(np.arange(0, 43, 6))
    ax.set_xlabel("time from cycle start (h)")
    ax.set_ylabel("chamber pressure\n(mTorr)", fontsize=7.6)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.3, frameon=False,
              handlelength=1.6, borderaxespad=0.3, labelspacing=0.26)


def margin_panel(ax, margins):
    y = np.arange(len(POSITIONS))[::-1]
    cols = [TEAL if m > 0 else RED for m in margins]
    ax.barh(y, margins, height=0.56,
            color=[lighten(c, 0.50) for c in cols], edgecolor=cols,
            linewidth=1.0, zorder=4)
    ax.axvline(0.0, color=RED, lw=1.2, zorder=6)
    ax.text(0.16, -0.55, "$T_c$", ha="left", va="bottom", fontsize=6.4,
            color=RED, fontweight="bold", zorder=7)
    for yi, m, c in zip(y, margins, cols):
        ax.text(m + (0.13 if m > 0 else -0.13), yi,
                "{:+.1f} $^\\circ$C".format(m), ha="left" if m > 0 else
                "right", va="center", fontsize=6.4, color=c,
                fontweight="bold", zorder=7)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in POSITIONS], fontsize=6.4)
    ax.set_xlim(-3.6, 6.4)
    ax.set_ylim(-0.62, len(POSITIONS) - 0.34)
    ax.set_xlabel("margin from the plateau maximum to $T_c$ ($^\\circ$C)",
                  fontsize=7.0)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Collapse margin by shelf position", fontsize=8.0,
                 fontweight="bold", color=INK)


def moisture_panel(ax, mean, sd, popt, t_spec):
    ax.axhline(MOIST_SPEC, color=RED, lw=1.2, ls=(0, (5.0, 2.2)), zorder=5)
    ax.text(11.6, MOIST_SPEC + 0.10,
            "specification {:.1f} % w/w".format(MOIST_SPEC), ha="right",
            va="bottom", fontsize=6.1, color=RED, zorder=7)
    hg = np.linspace(0.0, 12.0, 300)
    ax.plot(hg, decay(hg, *popt), color=BLUE, lw=1.5, zorder=4)
    ax.errorbar(MOIST_HOURS, mean, yerr=sd, ls="none", marker="o", ms=3.2,
                mfc="white", mec=BLUE, mew=0.9, ecolor=BLUE, elinewidth=0.7,
                capsize=1.8, zorder=6)
    ax.plot([t_spec, t_spec], [0.0, MOIST_SPEC], color=TEAL, lw=0.9,
            ls=(0, (2.2, 1.8)), zorder=5)
    ax.plot([t_spec], [MOIST_SPEC], marker="v", ms=5.6, mfc=TEAL, mec="white",
            mew=0.8, zorder=8)
    ax.annotate("{:.1f} h of secondary drying\nmeets the specification"
                .format(t_spec), xy=(t_spec, MOIST_SPEC),
                xytext=(t_spec + 1.0, 2.5), fontsize=6.2, color=TEAL,
                ha="left", va="center", linespacing=1.30, zorder=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(-0.5, 12.4)
    ax.set_ylim(0.0, 4.9)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("time into secondary drying (h)", fontsize=7.2)
    ax.set_ylabel("residual moisture (% w/w)", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Residual moisture, {:d} vials per point".format(N_VIAL),
                 fontsize=8.0, fontweight="bold", color=INK)


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
    rng = np.random.default_rng(9471)

    t = np.linspace(0.0, T_END, 4200)
    shelf = shelf_profile(t)
    cm, pir = gauges(t, rng)
    t_conv = convergence_time(t, cm, pir)

    prod = [product_profile(t, shelf, p, t_conv, rng)
            for _n, p, _c in POSITIONS]
    dry = (t >= T_FREEZE + 1.0) & (t <= t_conv)
    margins = [float(T_COLLAPSE - y[dry].max()) for y in prod]

    truth = decay(MOIST_HOURS, 0.45, 3.80, 2.40)
    reps = truth[None, :] * rng.normal(1.0, 0.075, (N_VIAL, MOIST_HOURS.size))
    mean, sd = reps.mean(axis=0), reps.std(axis=0, ddof=1)
    popt, _ = curve_fit(decay, MOIST_HOURS, mean, p0=[0.4, 3.8, 2.4],
                        bounds=([0.0, 0.5, 0.3], [2.0, 9.0, 12.0]),
                        maxfev=40000)
    hg = np.linspace(0.0, 12.0, 4000)
    fit = decay(hg, *popt)
    below = np.flatnonzero(fit <= MOIST_SPEC)
    t_spec = float(np.interp(-MOIST_SPEC, [-fit[below[0] - 1],
                                           -fit[below[0]]],
                             [hg[below[0] - 1], hg[below[0]]]))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_t = axes_inches(fig, MAIN_T)
    ax_p = axes_inches(fig, MAIN_B)
    ax_m = axes_inches(fig, AUX_T)
    ax_w = axes_inches(fig, AUX_B)

    temperature_panel(ax_t, t, shelf, prod, t_conv)
    pressure_panel(ax_p, t, cm, pir, t_conv)
    margin_panel(ax_m, margins)
    moisture_panel(ax_w, mean, sd, popt, t_spec)

    worst = POSITIONS[int(np.argmin(margins))][0]


    panel_labels_inches(fig, [ax_t, ax_p, ax_m, ax_w], dx=-0.62, dy=0.12)
    savefig(fig, "947_pha_lyophilization_cycle.png")


if __name__ == "__main__":
    main()
