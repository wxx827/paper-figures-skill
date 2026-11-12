"""Flowering time over the vernalization x photoperiod plane, three cultivars.

One two-factor model is evaluated on a single grid: a saturating exponential
response to days of cold at 5 degrees C and a logistic response to daylength.
The filled field is the winter cultivar; the two other cultivars appear only
as their 60 d isolines, so the reader can see how much cold each of them can
trade away for an hour of extra light.  Both quoted thresholds are measured on
the grid that is drawn - the critical photoperiod is where the numerical
derivative of flowering time with respect to daylength is steepest, and the
saturating cold requirement is where the vernalization term has spent 95 % of
its range.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
GOLD = "#c9a227"
BLUE = "#3b5b92"

SEED = 1033
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

V_MAX = 70.0          # days of cold at 5 degrees C
P_LO, P_HI = 8.0, 20.0    # photoperiod, h d-1
SAT_FRAC = 0.95       # definition of a satisfied cold requirement
ISO = 60.0            # the isoline the three cultivars are compared on

# name, floor (d), cold-removable days, cold time constant (d), daylength-
# removable days, critical photoperiod (h), logistic width (h), colour
CULTIVAR = [
    ("Winter type", 42.0, 46.0, 22.0, 30.0, 13.8, 1.10, "#0b3d2e"),
    ("Facultative", 40.0, 22.0, 14.0, 26.0, 13.0, 1.20, "#c9a227"),
    ("Spring type", 38.0, 6.0, 9.0, 22.0, 12.4, 1.30, "#b23a48"),
]

# Three glasshouse programmes plotted as operating points.
PROGRAMME = [(6.0, 11.0, "unlit autumn crop"),
             (28.0, 15.5, "cold store + 4 h top-up"),
             (58.0, 18.0, "full chill + long day")]


def days_to_flower(v, p, floor, dv, tau, dp, pc, k):
    """Days from sowing to first open flower on the two-factor model."""
    cold = dv * np.exp(-np.asarray(v, dtype=float) / tau)
    light = dp * (1.0 - 1.0 / (1.0 + np.exp(-(np.asarray(p, dtype=float)
                                              - pc) / k)))
    return floor + cold + light


def field(cv, vv, pp):
    _n, floor, dv, tau, dp, pc, k, _c = cv
    return days_to_flower(vv, pp, floor, dv, tau, dp, pc, k)


def critical_photoperiod(cv, p_axis, v_hold):
    """Daylength at which the drawn response falls fastest, read numerically."""
    line = field(cv, np.full_like(p_axis, v_hold), p_axis)
    slope = np.gradient(line, p_axis)
    return float(p_axis[int(np.argmin(slope))]), line, slope


def cold_requirement(cv, v_axis, p_hold):
    """Days of cold at which 95 % of the cold-removable delay has been spent."""
    line = field(cv, v_axis, np.full_like(v_axis, p_hold))
    span = line[0] - line[-1]
    target = line[0] - SAT_FRAC * span
    idx = int(np.argmax(line <= target))
    return float(v_axis[idx]), line


def main_panel(fig, ax, vv, pp, grids, v95, pcrit):
    levels = np.linspace(40.0, 120.0, 17)
    cf = ax.contourf(vv, pp, grids[0], levels=levels,
                     cmap=get_cmap("cb_sequential_green", reverse=True),
                     extend="both", zorder=2)
    cs = ax.contour(vv, pp, grids[0], levels=levels[::2], colors="#ffffff",
                    linewidths=0.55, alpha=0.8, zorder=3)
    for t in ax.clabel(cs, fmt="%.0f", fontsize=6.0, inline=True,
                       inline_spacing=4):
        t.set_path_effects([pe.withStroke(linewidth=1.7, foreground="#2f5343")])

    for cv, g in zip(CULTIVAR, grids):
        col = cv[7]
        ax.contour(vv, pp, g, levels=[ISO], colors=[col], linewidths=1.9,
                   zorder=6)

    halo = [pe.withStroke(linewidth=2.1, foreground="white")]
    ax.axhline(pcrit, color=BLUE, lw=1.0, ls=(0, (4.5, 2.6)), zorder=5)
    ax.text(1.0, pcrit + 0.16,
            "critical photoperiod {:.1f} h d$^{{-1}}$".format(pcrit),
            ha="left", va="bottom", fontsize=6.4, color=BLUE,
            fontweight="bold", zorder=9).set_path_effects(halo)
    ax.axvline(v95, color=RED, lw=1.0, ls=(0, (4.5, 2.6)), zorder=5)
    ax.text(v95 - 1.2, P_LO + 0.28,
            "{:.0f} d of cold covers {:.0f} % of\nthe winter-type "
            "requirement".format(v95, 100.0 * SAT_FRAC), ha="right",
            va="bottom", fontsize=6.4, color=RED, linespacing=1.34,
            zorder=9).set_path_effects(halo)

    for (v, p, label) in PROGRAMME:
        d = float(field(CULTIVAR[0], v, p))
        ax.plot([v], [p], marker="*", ms=9.5, mfc="white", mec=INK, mew=0.9,
                zorder=8)
        t = ax.annotate("{:s}\n{:.0f} d to flower".format(label, d),
                        xy=(v, p), xytext=(v + 3.0, p - 1.35), fontsize=6.2,
                        color=INK, ha="left", va="top", linespacing=1.32,
                        zorder=9,
                        arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                        shrinkA=2, shrinkB=4))
        t.set_path_effects(halo)

    handles = [Line2D([], [], color=c[7], lw=1.9,
                      label="{:s}, {:.0f} d isoline".format(c[0], ISO))
               for c in CULTIVAR]
    handles += [Line2D([], [], color=INK, ls="none", marker="*", ms=8.0,
                       mfc="white", mec=INK, label="glasshouse programme")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.2,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=1.7, borderaxespad=0.35,
                    labelspacing=0.24)
    leg.set_zorder(10)

    ax.set_xlim(0.0, V_MAX)
    ax.set_ylim(P_LO, P_HI)
    ax.set_xlabel("vernalization duration at 5 " + u"\u00b0" + "C (d)")
    ax.set_ylabel("photoperiod (h d$^{-1}$)")
    ax.set_title("Days from sowing to anthesis over the cold x daylength "
                 "plane", fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    safe_colorbar(fig, cf, ax, label="days to anthesis (d)", width=0.13,
                  pad=0.10, ticks=levels[::4])


def slice_panel(ax, p_axis, v_holds, pcrit):
    cv = CULTIVAR[0]
    shades = ["#a4d7b8", "#3d9469", "#0b3d2e"]
    for v_hold, col in zip(v_holds, shades):
        line = field(cv, np.full_like(p_axis, v_hold), p_axis)
        ax.plot(p_axis, line, color=col, lw=1.6, zorder=5,
                label="{:.0f} d cold".format(v_hold))
    ax.axvline(pcrit, color=BLUE, lw=0.9, ls=(0, (4.0, 2.4)), zorder=4)
    ax.text(pcrit - 0.25, 118.0, "critical\n{:.1f} h d$^{{-1}}$".format(pcrit),
            ha="right", va="top", fontsize=6.1, color=BLUE, linespacing=1.32,
            zorder=6)

    ax.set_xlim(P_LO, P_HI)
    ax.set_ylim(38.0, 124.0)
    ax.set_xlabel("photoperiod (h d$^{-1}$)", fontsize=6.8)
    ax.set_ylabel("days to anthesis (d)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.94, handlelength=1.3,
              borderaxespad=0.3, labelspacing=0.2)
    ax.set_title("Winter type sliced along daylength", fontsize=8.0,
                 fontweight="bold", color=INK)


def requirement_panel(ax, v_axis, reqs):
    y = np.arange(len(CULTIVAR))[::-1]
    vals = [r[0] for r in reqs]
    cols = [c[7] for c in CULTIVAR]
    ax.barh(y, vals, height=0.58, color=cols, edgecolor="white",
            linewidth=0.7, zorder=4)
    for yi, cv, v, r in zip(y, CULTIVAR, vals, reqs):
        saved = r[1][0] - r[1][-1]
        ax.text(v + 1.4, yi, "{:.0f} d cold saves {:.0f} d of crop time"
                .format(v, saved), ha="left", va="center", fontsize=6.0,
                color=SUB, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in CULTIVAR], fontsize=6.4)
    ax.set_xlim(0.0, max(vals) * 1.95)
    ax.set_ylim(-0.62, len(CULTIVAR) - 0.38)
    ax.set_xlabel("cold needed for {:.0f} % of the vernalization response (d)"
                  .format(100.0 * SAT_FRAC), fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.48)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("How much chilling each cultivar insists on", fontsize=8.0,
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
    np.random.default_rng(SEED)

    v_axis = np.linspace(0.0, V_MAX, 281)
    p_axis = np.linspace(P_LO, P_HI, 241)
    vv, pp = np.meshgrid(v_axis, p_axis)
    grids = [field(cv, vv, pp) for cv in CULTIVAR]

    for g in grids:
        assert np.all(np.diff(g, axis=1) <= 1e-9), "more cold must never delay"
        assert np.all(np.diff(g, axis=0) <= 1e-9), "more light must never delay"

    pcrit, _line, _slope = critical_photoperiod(CULTIVAR[0], p_axis, V_MAX)
    reqs = [cold_requirement(cv, v_axis, 16.0) for cv in CULTIVAR]
    v95 = reqs[0][0]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sl = axes_inches(fig, AUX_T)
    ax_req = axes_inches(fig, AUX_B)

    main_panel(fig, ax_main, vv, pp, grids, v95, pcrit)
    slice_panel(ax_sl, p_axis, [0.0, 20.0, 60.0], pcrit)
    requirement_panel(ax_req, v_axis, reqs)

    fast = float(grids[0].min())
    slow = float(grids[0].max())


    panel_labels_inches(fig, [ax_main, ax_sl, ax_req], dx=-0.62, dy=0.14)
    savefig(fig, "1033_hrt_vernalization_photoperiod.png")


if __name__ == "__main__":
    main()
