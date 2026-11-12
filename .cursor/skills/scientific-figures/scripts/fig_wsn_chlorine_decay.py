"""Residual chlorine down a transmission main, and the booster that saves it.

Chlorine decay is first order in the bulk and first order at the wall, so the
residual along a pipe is an exponential in *travel time*, not in distance: the
same kilometre costs more residual where the water is slow.  The main panel
carries both readings of that fact - concentration against distance for three
water temperatures, with the travel-time axis mirrored on top - and adds the
booster case, which is the identical decay law restarted at the dosing point.
Every compliance distance quoted is the root of the decay curve against the
minimum residual, the water-age panel integrates the same velocity profile
that produced the travel time, and the trihalomethane trace is a yield on the
chlorine actually consumed, so it cannot disagree with the curve above it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#134a76"
PALE = "#9cc3dd"
GREEN = "#2e7d5b"
BROWN = "#7a6a5d"
RED = "#b23a48"
ORANGE = "#c0603f"

SEED = 1005
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

L_MAIN = 24.0              # transmission main length, km
C0 = 1.05                  # residual leaving the treatment works, mg/L
C_MIN = 0.20               # minimum residual at the far tap, mg/L
C_MAX = 0.80               # taste and odour ceiling, mg/L
DIAM = 0.60                # main diameter, m
KB20 = 0.22                # bulk decay constant at 20 C, per hour
THETA = 1.07               # Arrhenius-like temperature factor, per degree
KW = 0.045                 # wall decay contribution, per hour
BOOST_KM = 13.5            # booster chlorination station, km
BOOST_TO = 0.62            # residual the booster restores, mg/L
TEMPS = (8.0, 15.0, 22.0)
DESIGN = 1                 # the booster is designed for the mean temperature
THM_YIELD = 78.0           # trihalomethane yield, ug per mg of chlorine used
HAIRLINE = "#cccccc"


def velocity(x_km):
    """Mean velocity down the main, m/s: it slows as branches draw water off."""
    return 1.05 - 0.030 * x_km + 0.00035 * x_km ** 2


def travel_time(x_km, n=4001):
    """Water age (h) as the integral of dx over the velocity that is drawn."""
    x = np.linspace(0.0, float(np.max(x_km)), n)
    v = velocity(x)
    t = np.concatenate([[0.0], np.cumsum(np.diff(x) * 1000.0 /
                                         (0.5 * (v[1:] + v[:-1])))]) / 3600.0
    return np.interp(x_km, x, t)


def k_total(temp):
    """First-order decay constant per hour at a given water temperature."""
    return KB20 * THETA ** (temp - 20.0) + KW


def residual(x_km, temp, c_start=C0, x_start=0.0):
    """Chlorine (mg/L) after the travel time between x_start and x_km."""
    dt = travel_time(np.atleast_1d(x_km)) - travel_time(np.atleast_1d(x_start))
    out = c_start * np.exp(-k_total(temp) * np.maximum(dt, 0.0))
    return out if np.ndim(x_km) else float(out[0])


def boosted(x_km, temp):
    """Two-stage profile: works dose, then the booster restarts the decay."""
    x = np.atleast_1d(np.asarray(x_km, dtype=float))
    up = residual(x, temp)
    dn = residual(x, temp, c_start=BOOST_TO, x_start=BOOST_KM)
    return np.where(x < BOOST_KM, up, dn)


def compliance_distance(temp, c_start=C0, x_start=0.0):
    """Distance where the residual crosses the minimum, found by bisection."""
    f = lambda x: residual(x, temp, c_start, x_start) - C_MIN
    if f(L_MAIN) > 0.0:
        return np.nan
    return brentq(f, x_start + 1e-6, L_MAIN, xtol=1e-9)


def node_ages(rng):
    """Twelve demand nodes hung off the main, with their own branch delay."""
    x = np.linspace(1.6, 23.0, 12)
    branch = 1.4 + 5.6 * rng.random(12) ** 1.7            # extra hours
    return x, travel_time(x) + branch


def main_panel(ax, x, curves, boost, dist):
    ax.axhspan(0.0, C_MIN, facecolor="#f6ded0", edgecolor="none", zorder=1)
    ax.axhspan(C_MAX, 1.20, facecolor="#f2eadb", alpha=0.55, edgecolor="none",
               zorder=1)
    ax.text(0.35, C_MIN - 0.02, "below the {:.2f} mg/L minimum residual"
            .format(C_MIN), fontsize=6.3, color=ORANGE, ha="left", va="top",
            zorder=6)
    ax.text(0.35, C_MAX + 0.025, "taste and odour ceiling {:.2f} mg/L"
            .format(C_MAX), fontsize=6.3, color=BROWN, ha="left",
            va="bottom", zorder=6)

    for k, (temp, c) in enumerate(zip(TEMPS, curves)):
        col = lighten(DEEP, 0.16 * k)
        ax.plot(x, c, color=col, lw=1.5, zorder=5)
        ax.text(L_MAIN - 0.3, c[-1] - 0.012,
                "{:.0f} C water, k = {:.3f} per h".format(temp, k_total(temp)),
                fontsize=6.2, color=col, ha="right", va="top", zorder=6)
        if np.isfinite(dist[k]):
            ax.plot([dist[k]], [C_MIN], marker="v", ms=5.0, mfc=col,
                    mec="white", mew=0.8, zorder=8)

    ax.plot(x, boost, color=GREEN, lw=1.5, ls=(0, (4.5, 2.2)), zorder=7)
    ax.axvline(BOOST_KM, color=GREEN, lw=0.8, ls=(0, (1.6, 1.8)), zorder=4)
    ax.annotate("booster at {:.1f} km re-doses to {:.2f} mg/L\n(design case, "
                "{:.0f} C water)".format(BOOST_KM, BOOST_TO, TEMPS[DESIGN]),
                xy=(BOOST_KM, BOOST_TO),
                xytext=(BOOST_KM - 1.2, BOOST_TO + 0.24), fontsize=6.5,
                color=GREEN, ha="right", va="bottom", linespacing=1.32,
                zorder=10, arrowprops=dict(arrowstyle="-", color=GREEN,
                                           lw=0.8, shrinkA=2, shrinkB=4))
    warm = int(np.argmax(TEMPS))
    ax.annotate("{:.0f} C water runs out of residual\nat {:.1f} km, "
                "{:.1f} km short of the end"
                .format(TEMPS[warm], dist[warm], L_MAIN - dist[warm]),
                xy=(dist[warm], C_MIN),
                xytext=(dist[warm] + 0.7, 0.64), fontsize=6.4,
                color=RED, ha="left", va="bottom", linespacing=1.32,
                zorder=10, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                           shrinkA=2, shrinkB=5))

    ax.set_xlim(0.0, L_MAIN)
    ax.set_ylim(0.0, 1.20)
    ax.set_xlabel("distance along the transmission main (km)")
    ax.set_ylabel("free chlorine residual (mg/L)")
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], color=DEEP, lw=1.5, label="single dose at the works"),
        Line2D([], [], color=GREEN, lw=1.5, ls=(0, (4.5, 2.2)),
               label="with booster chlorination"),
        Line2D([], [], color="none", marker="v", mfc=DEEP, mec="white",
               ms=5.0, label="compliance limit crossed"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, borderaxespad=0.3, labelspacing=0.28)

    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    ticks = np.arange(0.0, L_MAIN + 0.1, 4.0)
    axt.set_xticks(ticks)
    axt.set_xticklabels(["{:.1f}".format(t) for t in travel_time(ticks)],
                        fontsize=6.4)
    axt.set_xlabel("water age at the same section (h)", fontsize=7.4,
                   color=SUB, labelpad=3.0)
    axt.tick_params(axis="x", colors=SUB, labelsize=6.4)
    axt.spines["right"].set_visible(False)
    axt.spines["top"].set_color(HAIRLINE)
    ax.set_title("Decay along a {:.0f} km main at three water temperatures"
                 .format(L_MAIN), fontsize=9.0, fontweight="bold", color=INK,
                 pad=26.0)


def age_panel(ax, x, xnodes, ages, temp):
    ax.plot(x, travel_time(x), color=BLUE, lw=1.5, zorder=5)
    ax.plot(xnodes, ages, marker="o", ms=4.2, mfc=PALE, mec=DEEP, mew=0.8,
            ls="none", zorder=7)
    for xi, ai in zip(xnodes, ages):
        ax.plot([xi, xi], [travel_time(np.array([xi]))[0], ai], color=PALE,
                lw=0.8, zorder=4)
    worst = int(np.argmax(ages))
    ax.annotate("oldest water {:.1f} h".format(ages[worst]),
                xy=(xnodes[worst], ages[worst]),
                xytext=(xnodes[worst] - 1.0, ages[worst] + 1.6), fontsize=6.1,
                color=DEEP, ha="right", va="bottom", zorder=9,
                arrowprops=dict(arrowstyle="-", color=DEEP, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.text(0.6, travel_time(np.array([L_MAIN]))[0] * 0.55,
            "trunk travel time", fontsize=6.1, color=BLUE, ha="left",
            va="center", zorder=8)
    ax.set_xlim(0.0, L_MAIN)
    ax.set_ylim(0.0, float(ages.max()) * 1.22)
    ax.set_xlabel("distance along the main (km)", fontsize=6.8)
    ax.set_ylabel("water age (h)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Age at the trunk and at the branch nodes", fontsize=8.0,
                 fontweight="bold", color=INK)


def thm_panel(ax, x, curves, boost, temp_index):
    used = C0 - curves[temp_index]
    at_boost = residual(BOOST_KM, TEMPS[temp_index])
    used_b = np.where(x < BOOST_KM, C0 - boost,
                      (C0 - at_boost) + (BOOST_TO - boost))
    ax.plot(x, THM_YIELD * used, color=BROWN, lw=1.5, zorder=5)
    ax.plot(x, THM_YIELD * used_b, color=GREEN, lw=1.4, ls=(0, (4.5, 2.2)),
            zorder=6)
    ax.axhline(80.0, color=RED, lw=0.9, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(0.4, 82.0, "regulatory limit 80 ug/L", fontsize=6.1, color=RED,
            ha="left", va="bottom", zorder=7)
    ax.text(L_MAIN - 0.4, THM_YIELD * used[-1] - 1.5, "single dose",
            fontsize=6.1, color=BROWN, ha="right", va="top", zorder=7)
    ax.text(L_MAIN - 0.4, THM_YIELD * used_b[-1] + 2.0, "with booster",
            fontsize=6.1, color=GREEN, ha="right", va="bottom", zorder=7)
    ax.set_xlim(0.0, L_MAIN)
    ax.set_ylim(0.0, 110.0)
    ax.set_xlabel("distance along the main (km)", fontsize=6.8)
    ax.set_ylabel("trihalomethanes (ug/L)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Disinfection by-product formed on the way", fontsize=8.0,
                 fontweight="bold", color=INK)
    return float(THM_YIELD * used_b[-1])


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
    rng = np.random.default_rng(SEED)
    x = np.linspace(0.0, L_MAIN, 600)
    curves = [residual(x, t) for t in TEMPS]
    warm = int(np.argmax(TEMPS))
    boost = boosted(x, TEMPS[DESIGN])
    dist = np.array([compliance_distance(t) for t in TEMPS])

    assert np.all(np.diff([c[-1] for c in curves]) < 0.0), \
        "warmer water must decay faster"
    for t, d in zip(TEMPS, dist):
        if np.isfinite(d):
            assert abs(residual(d, t) - C_MIN) < 1e-9, "crossing is not a root"
    assert abs(boost[-1] - BOOST_TO *
               np.exp(-k_total(TEMPS[DESIGN]) *
                      (travel_time(np.array([L_MAIN]))[0] -
                       travel_time(np.array([BOOST_KM]))[0]))) < 1e-12, \
        "the booster leg must obey the same decay law"
    assert boost[-1] > C_MIN, "the booster has to hold the far-end residual"

    xnodes, ages = node_ages(rng)
    t_end = float(travel_time(np.array([L_MAIN]))[0])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_age = axes_inches(fig, AUX_T)
    ax_thm = axes_inches(fig, AUX_B)

    main_panel(ax_main, x, curves, boost, dist)
    age_panel(ax_age, x, xnodes, ages, TEMPS[DESIGN])
    thm_end = thm_panel(ax_thm, x, curves, boost, DESIGN)



    panel_labels_inches(fig, [ax_main, ax_age, ax_thm], dx=-0.62, dy=0.14)
    savefig(fig, "1005_wsn_chlorine_decay.png")


if __name__ == "__main__":
    main()
