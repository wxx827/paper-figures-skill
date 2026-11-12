"""The pool boiling curve, and the two turning points that make it famous.

One array of surface heat flux against wall superheat, built regime by regime
on a logarithmic grid and then rounded off, carries the whole sheet.  Free
convection gives way to nucleate boiling where the two laws cross; nucleate
boiling follows a cubic in superheat until the vapour blankets the surface; the
transition branch falls; film boiling picks up again once radiation through the
vapour film starts to matter.  The critical heat flux and the Leidenfrost
minimum are not annotated by hand - they are the maximum and the minimum of the
plotted array, and the burnout excursion is the superheat at which the film
branch can again carry the critical flux, found by interpolating the same curve.
The side panels are two other readings of it: the heat transfer coefficient the
curve implies, and the path a flux-controlled heater is forced to take, which is
where the hysteresis and the melted wall come from.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#134a76"
RED = "#b23a48"
EMBER = "#c0603f"
BROWN = "#7a4a2a"
GOLD = "#c9a227"
GREY = "#8a8a8a"

SEED = 1040
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

T_SAT = 373.15             # saturated water at 1 atm, K
SIGMA = 5.67e-8            # Stefan-Boltzmann, W/(m2 K4)
EMISS = 0.90               # emissivity of the heated wall
DT_CHF_SET = 30.0          # superheat where nucleate boiling is designed to peak
Q_CHF_SET = 1.26e6         # peak flux the nucleate law is scaled to, W/m2
DT_LEID_SET = 120.0        # superheat the transition branch is aimed at
Q_LEID_SET = 2.0e4         # flux at that point, W/m2
H_FILM = 167.0             # film-boiling conduction coefficient at 120 K, W/(m2 K)
SMOOTH = 5.0               # corner rounding, samples of the log grid


def q_convection(dt):
    return 1.30e3 * (dt / 5.0) ** 1.25


def q_nucleate(dt):
    return Q_CHF_SET * (dt / DT_CHF_SET) ** 3.0


def q_transition(dt):
    m = np.log(Q_LEID_SET / Q_CHF_SET) / np.log(DT_LEID_SET / DT_CHF_SET)
    return Q_CHF_SET * (dt / DT_CHF_SET) ** m


def q_film(dt):
    conv = H_FILM * (dt / DT_LEID_SET) ** 0.33 * dt
    rad = EMISS * SIGMA * ((T_SAT + dt) ** 4 - T_SAT ** 4)
    return conv + rad


def boiling_curve(n=1400):
    """Piecewise laws, joined where they cross, then rounded in log-log."""
    dt = np.logspace(np.log10(0.9), np.log10(3000.0), n)
    q_c, q_n = q_convection(dt), q_nucleate(dt)
    q_t, q_f = q_transition(dt), q_film(dt)
    i_onb = int(np.flatnonzero(q_n >= q_c)[0])
    i_leid = int(np.flatnonzero((dt > DT_CHF_SET) & (q_f >= q_t))[0])
    q = np.where(dt <= dt[i_onb], q_c,
                 np.where(dt <= DT_CHF_SET, q_n,
                          np.where(dt < dt[i_leid], q_t, q_f)))
    q = 10.0 ** gaussian_filter1d(np.log10(q), SMOOTH, mode="nearest")
    return dt, q, float(dt[i_onb])


def turning_points(dt, q):
    """Critical heat flux and Leidenfrost minimum, read off the array."""
    lo = np.flatnonzero(dt < 70.0)
    i_chf = lo[int(np.argmax(q[lo]))]
    hi = np.flatnonzero((dt > dt[i_chf]) & (dt < 600.0))
    i_min = hi[int(np.argmin(q[hi]))]
    film = np.flatnonzero(dt > dt[i_min])
    dt_burn = float(np.interp(q[i_chf], q[film], dt[film]))
    return int(i_chf), int(i_min), dt_burn


def main_panel(ax, dt, q, dt_onb, i_chf, i_min, dt_burn, slope):
    bands = ((0.9, dt_onb, "#eef2f7", "free\nconvection"),
             (dt_onb, dt[i_chf], "#fbe9e2", "nucleate boiling"),
             (dt[i_chf], dt[i_min], "#f3ded4", "transition\nboiling"),
             (dt[i_min], 3000.0, "#efe7de", "film boiling"))
    for x0, x1, col, name in bands:
        ax.axvspan(x0, x1, facecolor=col, edgecolor="none", zorder=1)
        ax.text(np.sqrt(x0 * x1), 3.1e6, name, fontsize=6.6, color=SUB,
                ha="center", va="top", linespacing=1.30, zorder=8)

    seg = ((dt <= dt_onb, BLUE, "-", 2.0),
           ((dt >= dt_onb) & (dt <= dt[i_chf]), RED, "-", 2.4),
           ((dt >= dt[i_chf]) & (dt <= dt[i_min]), EMBER, (0, (4.0, 2.0)), 1.8),
           (dt >= dt[i_min], BROWN, "-", 2.0))
    for m, col, ls, lw in seg:
        ax.loglog(dt[m], q[m], color=col, lw=lw, ls=ls, zorder=6)

    ax.plot([dt[i_chf]], [q[i_chf]], marker="o", ms=5.6, mfc="white", mec=RED,
            mew=1.6, zorder=10)
    ax.text(dt[i_chf] * 0.78, q[i_chf] * 1.28, "critical heat flux\n%.2f "
            "MW/m$^2$ at %.0f K" % (q[i_chf] / 1e6, dt[i_chf]), fontsize=6.7,
            color=RED, ha="right", va="bottom", linespacing=1.32,
            fontweight="bold", zorder=10)
    ax.plot([dt[i_min]], [q[i_min]], marker="o", ms=5.6, mfc="white",
            mec=BROWN, mew=1.6, zorder=10)
    ax.text(dt[i_min] * 1.16, q[i_min] * 0.42, "Leidenfrost minimum\n%.1f "
            "kW/m$^2$ at %.0f K" % (q[i_min] / 1e3, dt[i_min]), fontsize=6.7,
            color=BROWN, ha="left", va="top", linespacing=1.32,
            fontweight="bold", zorder=10)

    ax.annotate("", xy=(dt_burn, q[i_chf]), xytext=(dt[i_chf], q[i_chf]),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                ls=(0, (3.6, 1.8)), shrinkA=5, shrinkB=2),
                zorder=9)
    ax.plot([dt_burn], [q[i_chf]], marker="X", ms=6.0, color=RED, zorder=10)
    ax.text(dt_burn * 0.92, q[i_chf] * 0.62, "burnout: hold the flux at the "
            "peak and the wall\nmust jump to %.0f K of superheat, %.0f K "
            "away,\nwhich melts most heater materials"
            % (dt_burn, dt_burn - dt[i_chf]), fontsize=6.6, color=RED,
            ha="right", va="top", linespacing=1.34, zorder=10)

    i_a = int(np.argmin(np.abs(dt - 12.0)))
    ax.annotate("nucleate boiling follows a cubic:\nthe fitted slope here is "
                "%.2f decades\nof flux per decade of superheat" % slope,
                xy=(dt[i_a], q[i_a]), xytext=(1.35, 2.1e4), fontsize=6.6,
                color=RED, ha="left", va="top", linespacing=1.34, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=3,
                                shrinkB=4))
    ax.annotate("onset of nucleate boiling\nat %.1f K of superheat" % dt_onb,
                xy=(dt_onb, float(np.interp(dt_onb, dt, q))),
                xytext=(0.98, 3.4e2), fontsize=6.5, color=BLUE, ha="left",
                va="bottom", linespacing=1.32, zorder=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, shrinkA=3,
                                shrinkB=4))

    ax.set_xlim(0.9, 3000.0)
    ax.set_ylim(1.0e2, 5.0e6)
    ax.set_xlabel("wall superheat " + r"$\Delta T_e = T_w - T_{sat}$" + " (K)")
    ax.set_ylabel("surface heat flux (W/m$^2$)")
    ax.grid(lw=0.5, alpha=0.30, which="major")
    ax.grid(lw=0.4, alpha=0.16, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Saturated pool boiling of water at 1 atm on a heated wall",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def coefficient_panel(ax, dt, q, i_chf, i_min):
    h = q / dt
    i_h = int(np.argmax(h))
    ax.loglog(dt, h, color=DEEP, lw=1.7, zorder=6)
    ax.plot([dt[i_h]], [h[i_h]], marker="o", ms=4.6, mfc="white", mec=RED,
            mew=1.4, zorder=9)
    ax.axvline(dt[i_chf], color=RED, lw=0.8, ls=(0, (2.6, 1.8)), zorder=4)
    ax.axvline(dt[i_min], color=BROWN, lw=0.8, ls=(0, (2.6, 1.8)), zorder=4)
    ax.text(1.15, 1.05e5, "best coefficient\n%.1f kW/(m$^2$" r"$\cdot$"
            "K) at %.0f K" % (h[i_h] / 1e3, dt[i_h]), fontsize=6.1, color=RED,
            ha="left", va="top", linespacing=1.30, zorder=9)
    ax.text(dt[i_min] * 1.25, 22.0, "film boiling gives back\nbarely a "
            "hundredth of it", fontsize=6.0, color=BROWN, ha="left",
            va="bottom", linespacing=1.30, zorder=9)
    ax.set_xlim(0.9, 3000.0)
    ax.set_ylim(15.0, 1.2e5)
    ax.set_xlabel("wall superheat (K)", fontsize=6.8)
    ax.set_ylabel("h = q / " + r"$\Delta T_e$" + " (W/(m$^2\\cdot$K))",
                  fontsize=6.4)
    ax.grid(lw=0.5, alpha=0.36, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("What the same curve is worth as a coefficient",
                 fontsize=8.0, fontweight="bold", color=INK)
    return float(h[i_h]), float(dt[i_h])


def hysteresis_panel(ax, dt, q, i_chf, i_min, dt_burn):
    up = np.flatnonzero(dt <= dt[i_chf])
    down = np.flatnonzero(dt >= dt[i_min])
    ax.loglog(q[up], dt[up], color=RED, lw=1.8, zorder=6)
    ax.loglog(q[down], dt[down], color=BROWN, lw=1.8, zorder=6)
    ax.loglog(q[i_chf:i_min], dt[i_chf:i_min], color=EMBER, lw=1.1,
              ls=(0, (2.4, 2.0)), zorder=5)
    ax.annotate("", xy=(q[i_chf], dt_burn), xytext=(q[i_chf], dt[i_chf]),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                shrinkA=2, shrinkB=2), zorder=8)
    ax.annotate("", xy=(q[i_min], float(np.interp(q[i_min], q[up], dt[up]))),
                xytext=(q[i_min], dt[i_min]),
                arrowprops=dict(arrowstyle="-|>", color=BROWN, lw=1.2,
                                shrinkA=2, shrinkB=2), zorder=8)
    ax.text(q[i_chf] * 0.72, dt_burn * 1.10, "burnout", fontsize=6.2,
            color=RED, ha="right", va="bottom", fontweight="bold", zorder=9)
    ax.text(q[i_min] * 0.72, 34.0, "rewetting", fontsize=6.2, color=BROWN,
            ha="right", va="center", fontweight="bold", zorder=9)
    ax.text(9.0e5, 74.0, "unreachable", fontsize=6.0, color=EMBER,
            ha="right", va="bottom", zorder=9)
    ax.text(1.30e2, 3300.0, "raise the flux and the wall\nfollows the red "
            "branch to the\npeak, then jumps; lower it and\nit holds the "
            "brown film branch\ndown to %.0f kW/m$^2$" % (q[i_min] / 1e3),
            fontsize=6.0, color=SUB, ha="left", va="top", linespacing=1.30,
            zorder=9)
    ax.set_xlim(1.0e2, 5.0e6)
    ax.set_ylim(0.9, 4000.0)
    ax.set_xlabel("imposed surface heat flux (W/m$^2$)", fontsize=6.8)
    ax.set_ylabel("wall superheat (K)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.36, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("A flux-controlled heater cannot follow the curve",
                 fontsize=8.0, fontweight="bold", color=INK)


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
    np.random.default_rng(SEED)

    dt, q, dt_onb = boiling_curve()
    i_chf, i_min, dt_burn = turning_points(dt, q)

    fit = np.flatnonzero((dt > 8.0) & (dt < 22.0))
    slope = float(np.polyfit(np.log10(dt[fit]), np.log10(q[fit]), 1)[0])

    assert q[i_chf] > q[i_min] * 20.0, \
        "the critical flux must tower over the Leidenfrost minimum"
    assert dt_onb < dt[i_chf] < dt[i_min] < dt_burn, \
        "the turning points have to appear in the textbook order"
    assert np.all(np.diff(q[:i_chf]) > 0), \
        "everything before the peak has to be rising"
    assert np.all(np.diff(q[i_chf:i_min]) < 0), \
        "the transition branch has to fall"
    assert 2.4 < slope < 3.2, "nucleate boiling should stay close to a cubic"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_hy = axes_inches(fig, AUX_B)

    main_panel(ax_main, dt, q, dt_onb, i_chf, i_min, dt_burn, slope)
    h_max, dt_hmax = coefficient_panel(ax_h, dt, q, i_chf, i_min)
    hysteresis_panel(ax_hy, dt, q, i_chf, i_min, dt_burn)



    panel_labels_inches(fig, [ax_main, ax_h, ax_hy], dx=-0.62, dy=0.14)
    savefig(fig, "1040_ht_boiling_curve.png")


if __name__ == "__main__":
    main()
