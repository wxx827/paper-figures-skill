"""A psychrometric chart drawn from the saturation law, and two air paths on it.

Everything on the chart is generated from one vapour-pressure correlation: the
saturation curve, the relative-humidity family under it, the constant-enthalpy
rulings that cross them, and the dew point of the summer state, which is found
by asking at what temperature saturated air would hold the humidity that state
already carries.  Two processes are then drawn on the same field - a summer coil
that cools past the dew point, sheds moisture along the saturation curve and is
reheated to supply condition, and a winter path that preheats and then humidifies
along a line of constant enthalpy.  The side panels are the consequences: the
coil load split into the sensible and latent parts that the two legs of the path
represent, and the surface temperature at which a room at a given humidity starts
to condense on its cold surfaces.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#134a76"
BLUE = "#3b5b92"
PALE = "#9cc3dd"
RED = "#b23a48"
EMBER = "#c0603f"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
GREY = "#8a8a8a"
HAIR = "#cccccc"

DEG = r"$^\circ$C"

SEED = 1042
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

P_ATM = 101325.0           # barometric pressure, Pa
T_MAX, W_MAX = 45.0, 30.0  # chart limits, degC and g/kg dry air
M_AIR = 1.50               # supply airflow, kg/s of dry air
ROOM_DB = 26.0             # room dry bulb for the condensation panel, degC
SURF_T = 16.0              # chilled surface temperature, degC

STATE_1 = (32.0, 0.55)     # summer mixed air: dry bulb degC, relative humidity
STATE_2 = (11.0, 0.95)     # apparatus dew point leaving the coil
T_SUPPLY = 14.0            # after reheat, degC
STATE_4 = (4.0, 0.60)      # winter outdoor air
T_PREHEAT = 32.0           # after the sensible preheater, degC
T_ROOM_W = 22.0            # after the adiabatic humidifier, degC


def p_sat(t):
    """Saturation vapour pressure over water, Pa."""
    t = np.asarray(t, dtype=float)
    return 610.94 * np.exp(17.625 * t / (t + 243.04))


def humidity_ratio(t, rh):
    pw = rh * p_sat(t)
    return 0.621945 * pw / (P_ATM - pw)


def enthalpy(t, w):
    """Moist-air enthalpy per kg of dry air, kJ/kg."""
    return 1.006 * t + w * (2501.0 + 1.86 * t)


def dew_point(w):
    """Invert the saturation law: the temperature at which this air saturates."""
    pw = P_ATM * w / (0.621945 + w)
    return brentq(lambda t: p_sat(t) - pw, -20.0, 80.0, xtol=1e-8)


def rh_at(t, w):
    pw = P_ATM * w / (0.621945 + w)
    return float(pw / p_sat(t))


def exit_point(t, w):
    """Last point of a curve inside the chart box, for edge labelling."""
    ok = np.flatnonzero((w <= W_MAX) & (t <= T_MAX))
    i = int(ok[-1])
    return float(t[i]), float(w[i])


def chart_panel(ax, states, t_dp):
    t = np.linspace(0.0, T_MAX, 500)
    w_sat = 1e3 * humidity_ratio(t, 1.0)

    for h in range(20, 121, 10):
        w_h = 1e3 * (h - 1.006 * t) / (2501.0 + 1.86 * t)
        m = (w_h >= 0.0) & (w_h <= np.minimum(w_sat, W_MAX))
        if m.sum() < 3:
            continue
        ax.plot(t[m], w_h[m], color="#d8d2c8", lw=0.7, ls=(0, (4.0, 2.4)),
                zorder=2)
        if h in (40, 60, 80, 100):
            ax.text(t[m][0] - 0.45, w_h[m][0] + 0.55, "%d kJ/kg" % h,
                    fontsize=6.0, color="#a89a86", ha="right", va="bottom",
                    zorder=3)

    for rh in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        w_r = 1e3 * humidity_ratio(t, rh)
        m = w_r <= W_MAX
        ax.plot(t[m], w_r[m], color=PALE, lw=0.8, zorder=3)
        te, we = exit_point(t[m], w_r[m])
        if we > W_MAX - 0.4:
            ax.text(te - 0.25, W_MAX - 0.35, "%d %%" % (100 * rh),
                    fontsize=6.0, color=BLUE, ha="right", va="top", zorder=4)
        else:
            ax.text(T_MAX - 0.35, we + 0.30, "%d %%" % (100 * rh),
                    fontsize=6.0, color=BLUE, ha="right", va="bottom",
                    zorder=4)
    m = w_sat <= W_MAX
    ax.plot(t[m], w_sat[m], color=DEEP, lw=2.0, zorder=6)
    ax.text(t[m][-1] - 0.4, W_MAX - 0.35, "saturation, 100 %", fontsize=6.4,
            color=DEEP, ha="right", va="top", fontweight="bold", zorder=6)
    ax.fill_between(t[m], w_sat[m], W_MAX, color="#f3f6f9", edgecolor="none",
                    zorder=1)

    (t1, w1), (t2, w2), (t3, w3) = states[0], states[1], states[2]
    (t4, w4), (t5, w5), (t6, w6) = states[3], states[4], states[5]

    ax.plot([t1, t2], [w1, w2], color=BLUE, lw=2.0, zorder=7)
    ax.plot([t2, t3], [w2, w3], color=GOLD, lw=2.0, zorder=7)
    ax.plot([t4, t5], [w4, w5], color=RED, lw=2.0, zorder=7)
    hum_t = np.linspace(t6, t5, 60)
    hum_w = 1e3 * (enthalpy(t5, 1e-3 * w5) - 1.006 * hum_t) / \
        (2501.0 + 1.86 * hum_t)
    ax.plot(hum_t, hum_w, color=EMBER, lw=2.0, zorder=7)
    for (ta, wa), (tb, wb), col in (((t1, w1), (t2, w2), BLUE),
                                    ((t2, w2), (t3, w3), GOLD),
                                    ((t4, w4), (t5, w5), RED),
                                    ((t5, w5), (t6, w6), EMBER)):
        ax.annotate("", xy=(0.5 * (ta + tb), 0.5 * (wa + wb)),
                    xytext=(0.62 * ta + 0.38 * tb, 0.62 * wa + 0.38 * wb),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=8)

    ax.plot([t_dp, t1], [w1, w1], color=GREEN, lw=1.0, ls=(0, (2.6, 1.8)),
            zorder=6)
    ax.plot([t_dp], [w1], marker="o", ms=4.6, mfc="white", mec=GREEN, mew=1.3,
            zorder=9)
    ax.text(t_dp - 0.5, w1 + 0.45, "dew point %.1f " % t_dp + DEG +
            "\nsolved from the saturation law", fontsize=6.4, color=GREEN,
            ha="right", va="bottom", linespacing=1.30, zorder=9)

    tags = (("1  mixed air", t1, w1, 0.55, 0.75, "left"),
            ("2  off the coil", t2, w2, -0.55, 0.85, "right"),
            ("3  supply", t3, w3, 0.45, -1.10, "left"),
            ("4  winter outdoor", t4, w4, 0.55, 0.60, "left"),
            ("5  after preheat", t5, w5, 0.50, -1.30, "left"),
            ("6  room supply", t6, w6, 0.0, 0.80, "center"))
    for name, tt, ww, dx, dy, ha in tags:
        ax.plot([tt], [ww], marker="o", ms=5.2, mfc="white", mec=INK, mew=1.2,
                zorder=10)
        ax.text(tt + dx, ww + dy, name, fontsize=6.5, color=INK, ha=ha,
                va="bottom" if dy > 0 else "top", fontweight="bold",
                zorder=10)

    handles = [
        Line2D([], [], color=BLUE, lw=2.0,
               label="1-2 cool and dehumidify on the coil"),
        Line2D([], [], color=GOLD, lw=2.0, label="2-3 sensible reheat"),
        Line2D([], [], color=RED, lw=2.0, label="4-5 sensible preheat"),
        Line2D([], [], color=EMBER, lw=2.0,
               label="5-6 humidify at constant enthalpy"),
        Line2D([], [], color=PALE, lw=0.9, label="relative humidity"),
        Line2D([], [], color="#d8d2c8", lw=0.8, ls=(0, (4.0, 2.4)),
               label="constant enthalpy"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.2,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.0, borderaxespad=0.45,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xlim(0.0, T_MAX)
    ax.set_ylim(0.0, W_MAX)
    ax.set_xlabel("dry-bulb temperature (" + DEG + ")")
    ax.set_ylabel("humidity ratio (g of water per kg of dry air)")
    ax.grid(lw=0.5, alpha=0.26)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Moist air at %.1f kPa: two conditioning paths on one chart"
                 % (P_ATM / 1e3), fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def load_panel(ax, sens, lat, reheat, shr):
    ax.barh([1], [sens], height=0.46, color=BLUE, edgecolor="white",
            linewidth=0.8, zorder=5)
    ax.barh([1], [lat], left=[sens], height=0.46, color=PALE,
            edgecolor="white", linewidth=0.8, zorder=5)
    ax.barh([0], [reheat], height=0.46, color=GOLD, edgecolor="white",
            linewidth=0.8, zorder=5)
    ax.text(sens * 0.5, 1, "sensible\n%.1f kW" % sens, fontsize=6.2,
            color="white", ha="center", va="center", fontweight="bold",
            linespacing=1.24, zorder=8)
    ax.text(sens + lat * 0.5, 1, "latent\n%.1f kW" % lat, fontsize=6.2,
            color=INK, ha="center", va="center", fontweight="bold",
            linespacing=1.24, zorder=8)
    ax.text(reheat + 1.2, 0, "reheat %.1f kW, paid twice:\nonce to cool the "
            "air, once to warm it" % reheat, fontsize=6.1, color=SUB,
            ha="left", va="center", linespacing=1.28, zorder=8)
    ax.text(0.6, 1.42, "coil load %.1f kW at %.2f kg/s of dry air, "
            "sensible heat ratio %.2f" % (sens + lat, M_AIR, shr),
            fontsize=6.2, color=INK, ha="left", va="bottom", zorder=8)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["cooling coil", "reheat coil"], fontsize=6.6)
    ax.set_xlim(0.0, (sens + lat) * 1.10)
    ax.set_ylim(-0.62, 1.85)
    ax.set_xlabel("load (kW)", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.40)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("What the two legs of the path cost", fontsize=8.0,
                 fontweight="bold", color=INK)


def condensation_panel(ax, rh_grid, t_dp_grid, rh_crit):
    ax.axhspan(0.0, SURF_T, facecolor="#eef2f7", edgecolor="none", zorder=1)
    ax.plot(1e2 * rh_grid, t_dp_grid, color=DEEP, lw=1.8, zorder=6)
    ax.axhline(SURF_T, color=BLUE, lw=1.0, ls=(0, (3.4, 2.0)), zorder=5)
    ax.axvline(1e2 * rh_crit, color=RED, lw=1.0, ls=(0, (2.4, 1.8)), zorder=5)
    ax.fill_between(1e2 * rh_grid, SURF_T, t_dp_grid,
                    where=t_dp_grid >= SURF_T, color="#f6ded8",
                    edgecolor="none", zorder=2)
    ax.plot([1e2 * rh_crit], [SURF_T], marker="o", ms=5.0, mfc="white",
            mec=RED, mew=1.4, zorder=9)
    ax.text(1e2 * rh_crit - 1.6, 4.5, "%.0f %% relative humidity" % (
        1e2 * rh_crit), fontsize=6.2, color=RED, ha="right", va="bottom",
        fontweight="bold", zorder=9)
    ax.text(22.0, SURF_T - 1.2, "chilled surface at %.0f " % SURF_T + DEG,
            fontsize=6.1, color=BLUE, ha="left", va="top", zorder=9)
    ax.text(98.0, 16.5, "condensation on the surface", fontsize=6.2,
            color=RED, ha="right", va="bottom", fontweight="bold", zorder=9)
    ax.set_xlim(20.0, 100.0)
    ax.set_ylim(0.0, 27.0)
    ax.set_xlabel("relative humidity in a %.0f " % ROOM_DB + DEG + " room (%)",
                  fontsize=6.8)
    ax.set_ylabel("dew point (" + DEG + ")", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.36)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The same inversion, used as a condensation limit",
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

    w1 = humidity_ratio(*STATE_1)
    w2 = humidity_ratio(*STATE_2)
    w4 = humidity_ratio(*STATE_4)
    h5 = enthalpy(T_PREHEAT, w4)
    w6 = (h5 - 1.006 * T_ROOM_W) / (2501.0 + 1.86 * T_ROOM_W)
    states = [(STATE_1[0], 1e3 * w1), (STATE_2[0], 1e3 * w2),
              (T_SUPPLY, 1e3 * w2), (STATE_4[0], 1e3 * w4),
              (T_PREHEAT, 1e3 * w4), (T_ROOM_W, 1e3 * w6)]
    t_dp = dew_point(w1)

    h1, h2 = enthalpy(STATE_1[0], w1), enthalpy(STATE_2[0], w2)
    total = M_AIR * (h1 - h2)
    cp_moist = 1.006 + 1.86 * 0.5 * (w1 + w2)
    sens = M_AIR * cp_moist * (STATE_1[0] - STATE_2[0])
    lat = M_AIR * 2501.0 * (w1 - w2)
    shr = sens / total
    reheat = M_AIR * (1.006 + 1.86 * w2) * (T_SUPPLY - STATE_2[0])

    rh_grid = np.linspace(0.20, 1.00, 300)
    t_dp_grid = np.array([dew_point(humidity_ratio(ROOM_DB, r))
                          for r in rh_grid])
    rh_crit = float(p_sat(SURF_T) / p_sat(ROOM_DB))

    assert abs(sens + lat - total) < 0.02 * total, \
        "the sensible and latent legs have to add up to the coil load"
    assert STATE_2[0] < t_dp < STATE_1[0], \
        "the coil must reach below the dew point to dry the air"
    assert w6 > w4 and abs(enthalpy(T_ROOM_W, w6) - h5) < 1e-9, \
        "adiabatic humidification adds water at constant enthalpy"
    assert abs(dew_point(humidity_ratio(ROOM_DB, rh_crit)) - SURF_T) < 1e-6, \
        "the critical humidity is where the dew point meets the surface"
    assert rh_at(STATE_1[0], w1) > rh_at(T_ROOM_W, w6), \
        "the winter supply must be drier than the summer mixed air"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_load = axes_inches(fig, AUX_T)
    ax_con = axes_inches(fig, AUX_B)

    chart_panel(ax_main, states, t_dp)
    load_panel(ax_load, sens, lat, reheat, shr)
    condensation_panel(ax_con, rh_grid, t_dp_grid, rh_crit)



    panel_labels_inches(fig, [ax_main, ax_load, ax_con], dx=-0.62, dy=0.14)
    savefig(fig, "1042_ht_psychrometric_chart.png")


if __name__ == "__main__":
    main()
