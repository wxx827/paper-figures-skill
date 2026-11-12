"""Economic insulation thickness: where the two bills are the same size.

A hot furnace wall is insulated to a variable depth and both bills are drawn
against that depth on the same annual basis - the fuel the residual heat loss
burns, and the annualised capital of the insulation that stops it.  The
recommended thickness is the minimum of their sum, taken as the smallest value
of the plotted total rather than picked by eye, and the panel also marks where
the two component curves cross.  For a plane wall with a capital cost linear in
thickness those two points nearly coincide, and the script measures the gap
between them instead of asserting it.  The side panels carry the two
constraints that sit next to the money: the surface temperature that fixes a
minimum thickness for anyone who has to work beside the wall, and the critical
radius, which is the one geometry where the first millimetres of insulation make
the heat loss worse.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

DEG = r"$^\circ$C"

SEED = 1043
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

T_HOT = 320.0              # process side of the wall, degC
T_AMB = 25.0               # still air in the plant, degC
R_WALL = 0.020             # casing and inside film, m2 K/W
K_INS = 0.055              # calcium silicate, W/(m K)
H_OUT = 10.0               # outside film, convection plus radiation, W/(m2 K)
HOURS = 4000.0             # operating hours per year, h
FUEL = 9.0                 # delivered fuel, USD per GJ
ETA_BOILER = 0.80          # combustion to wall efficiency
COST_M3 = 500.0            # installed insulation, USD per m3
RATE, LIFE = 0.08, 15.0    # discount rate and economic life, years
T_SKIN_LIMIT = 60.0        # personnel protection limit, degC
T_MAX_MM = 400.0           # thickness swept, mm

CABLE_R1 = 0.0020          # bare cable radius, m
CABLE_K = 0.150            # cable jacket conductivity, W/(m K)
CABLE_H = 12.0             # outside film on the jacket, W/(m2 K)
PIPE_R1 = 0.0572           # DN100 pipe radius for contrast, m


def crf():
    """Capital recovery factor: capital turned into an annual charge."""
    f = (1.0 + RATE) ** LIFE
    return RATE * f / (f - 1.0)


def heat_flux(t_m):
    """Steady loss through the wall, W per m2."""
    return (T_HOT - T_AMB) / (R_WALL + t_m / K_INS + 1.0 / H_OUT)


def energy_cost(t_m):
    """Annual fuel bill for that loss, USD per m2 per year."""
    gj = heat_flux(t_m) * HOURS * 3600.0 * 1e-9 / ETA_BOILER
    return gj * FUEL


def capital_cost(t_m):
    """Annualised cost of the insulation itself, USD per m2 per year."""
    return COST_M3 * t_m * crf()


def skin_temperature(t_m):
    return T_AMB + heat_flux(t_m) / H_OUT


def cylinder_loss(r2, r1, k, h, dt=1.0):
    """Loss per metre of a lagged cylinder, W per m per K of driving span."""
    return 2.0 * np.pi * dt / (np.log(r2 / r1) / k + 1.0 / (h * r2))


def cost_panel(ax, t_mm, e_cost, c_cost, tot, i_opt, t_cross, band, payback):
    ax.axvspan(band[0], band[1], facecolor="#f0f2f4", edgecolor="none",
               zorder=1)
    ax.plot(t_mm, e_cost, color=RED, lw=2.0, zorder=6)
    ax.plot(t_mm, c_cost, color=BLUE, lw=2.0, zorder=6)
    ax.plot(t_mm, tot, color=INK, lw=2.4, zorder=7)

    ax.plot([t_mm[i_opt]], [tot[i_opt]], marker="o", ms=6.0, mfc="white",
            mec=INK, mew=1.6, zorder=10)
    ax.plot([t_mm[i_opt], t_mm[i_opt]], [0.0, tot[i_opt]], color=INK, lw=0.9,
            ls=(0, (2.4, 1.8)), zorder=5)
    e_at = float(np.interp(t_cross, t_mm, e_cost))
    ax.plot([t_cross], [e_at], marker="D", ms=5.0, mfc="white", mec=GREEN,
            mew=1.4, zorder=10)

    ax.annotate("cheapest wall: %.0f mm of insulation at\n%.1f USD per m$^2$ "
                "per year, %.1f USD of fuel\nplus %.1f USD of annualised "
                "capital"
                % (t_mm[i_opt], tot[i_opt], e_cost[i_opt], c_cost[i_opt]),
                xy=(t_mm[i_opt], tot[i_opt]), xytext=(t_mm[i_opt] + 26.0,
                                                      88.0),
                fontsize=6.7, color=INK, ha="left", va="top",
                linespacing=1.34, zorder=11,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=3,
                                shrinkB=5))
    ax.annotate("the two bills are equal at %.0f mm, only\n%.0f mm from the "
                "minimum: on a plane wall\nwith a linear capital cost the "
                "crossing is\nthe optimum to within a few per cent"
                % (t_cross, abs(t_cross - t_mm[i_opt])),
                xy=(t_cross, e_at), xytext=(t_cross + 22.0, 38.0),
                fontsize=6.6, color=GREEN, ha="left", va="top",
                linespacing=1.34, zorder=11,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8,
                                shrinkA=3, shrinkB=5))
    ax.text(0.5 * (band[0] + band[1]), 4.0, "within 5 %% of the minimum\n"
            "from %.0f to %.0f mm" % (band[0], band[1]), fontsize=6.3,
            color=SUB, ha="center", va="bottom", linespacing=1.30, zorder=9)
    ax.text(30.0, 97.5, "a bare wall burns %.0f USD per m$^2$ every year, "
            "off\nthe top of this axis; %.0f mm pays it back in %.2f years"
            % (e_cost[0], t_mm[i_opt], payback), fontsize=6.5, color=RED,
            ha="left", va="top", linespacing=1.32, zorder=9)

    handles = [Line2D([], [], color=RED, lw=2.0, label="fuel for the residual "
                      "heat loss"),
               Line2D([], [], color=BLUE, lw=2.0,
                      label="annualised insulation capital"),
               Line2D([], [], color=INK, lw=2.4, label="total annual cost")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=6.4,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.0, borderaxespad=0.45,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xlim(0.0, T_MAX_MM)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("insulation thickness (mm)")
    ax.set_ylabel("annual cost (USD per m$^2$ of wall per year)")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("A %.0f " % T_HOT + DEG + " wall at %.0f h/yr, fuel at %.0f "
                 "USD/GJ, insulation at %.0f USD/m$^3$"
                 % (HOURS, FUEL, COST_M3), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def skin_panel(ax, t_mm, flux, skin, t_safe, t_opt):
    ax.plot(t_mm, flux, color=EMBER, lw=1.8, zorder=6)
    ax.set_xlim(0.0, T_MAX_MM)
    ax.set_ylim(0.0, 1200.0)
    ax.set_xlabel("insulation thickness (mm)", fontsize=6.8)
    ax.set_ylabel("heat loss (W/m$^2$)", fontsize=6.6, color=EMBER)
    ax.tick_params(axis="y", colors=EMBER)
    ax.grid(lw=0.5, alpha=0.36)

    ax2 = ax.twinx()
    ax2.plot(t_mm, skin, color=DEEP, lw=1.8, zorder=6)
    ax2.axhline(T_SKIN_LIMIT, color=RED, lw=1.0, ls=(0, (3.0, 2.0)), zorder=5)
    ax2.plot([t_safe], [T_SKIN_LIMIT], marker="o", ms=4.4, mfc="white",
             mec=RED, mew=1.3, zorder=9)
    ax2.set_ylim(0.0, 330.0)
    ax2.set_ylabel("surface temperature (" + DEG + ")", fontsize=6.6,
                   color=DEEP)
    ax2.tick_params(axis="y", colors=DEEP)
    ax2.text(t_safe + 8.0, T_SKIN_LIMIT + 12.0, "%.0f " % T_SKIN_LIMIT + DEG +
             " touch limit\nneeds %.0f mm" % t_safe, fontsize=6.1, color=RED,
             ha="left", va="bottom", linespacing=1.28, zorder=9)
    ax2.text(T_MAX_MM - 8.0, 150.0, "the economic %.0f mm is far\nthicker "
             "than safety demands" % t_opt, fontsize=6.0, color=SUB,
             ha="right", va="bottom", linespacing=1.28, zorder=9)
    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    ax.set_title("What the thickness does to the wall itself", fontsize=8.0,
                 fontweight="bold", color=INK)


def critical_panel(ax, t_mm, q_cable, q_pipe, r_crit, t_peak):
    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)
    ax.fill_between(t_mm, 1.0, q_cable, where=q_cable >= 1.0,
                    color="#f7ecd2", edgecolor="none", zorder=2)
    ax.plot(t_mm, q_cable, color=GOLD, lw=1.8, zorder=6)
    ax.plot(t_mm, q_pipe, color=BLUE, lw=1.8, zorder=6)
    ax.axvline(t_peak, color=RED, lw=1.0, ls=(0, (2.6, 1.8)), zorder=5)
    ax.plot([t_peak], [q_cable.max()], marker="o", ms=4.6, mfc="white",
            mec=RED, mew=1.3, zorder=9)
    ax.text(t_peak + 1.6, 2.42, "critical radius k/h = %.1f mm, reached\n"
            "after %.1f mm of jacket: the cable runs\n%.0f %% hotter in "
            "loss before it improves"
            % (1e3 * r_crit, t_peak, 1e2 * (q_cable.max() - 1.0)),
            fontsize=6.1, color=RED, ha="left", va="top", linespacing=1.28,
            zorder=9)
    ax.text(58.0, 1.32, "2 mm cable, k = %.2f W/(m" % CABLE_K + r"$\cdot$" +
            "K)", fontsize=6.1, color="#9a7d1c", ha="right", va="bottom",
            zorder=9)
    ax.text(58.0, 0.24, "DN100 pipe, k = %.3f W/(m" % K_INS + r"$\cdot$" +
            "K)", fontsize=6.1, color=BLUE, ha="right", va="bottom", zorder=9)
    ax.text(1.5, 1.06, "worse than bare", fontsize=6.0, color="#9a7d1c",
            ha="left", va="bottom", zorder=9)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 2.60)
    ax.set_xlabel("thickness of the lagging (mm)", fontsize=6.8)
    ax.set_ylabel("heat loss / bare loss (dimensionless)", fontsize=6.4)
    ax.grid(lw=0.5, alpha=0.36)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Why small cylinders are the exception", fontsize=8.0,
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

    t_mm = np.linspace(0.0, T_MAX_MM, 1601)
    t_m = 1e-3 * t_mm
    e_cost = energy_cost(t_m)
    c_cost = capital_cost(t_m)
    tot = e_cost + c_cost
    i_opt = int(np.argmin(tot))

    diff = e_cost - c_cost
    j = int(np.flatnonzero(np.diff(np.sign(diff)) != 0)[0])
    t_cross = float(np.interp(0.0, [diff[j + 1], diff[j]],
                              [t_mm[j + 1], t_mm[j]]))

    ok = np.flatnonzero(tot <= 1.05 * tot[i_opt])
    band = (float(t_mm[ok[0]]), float(t_mm[ok[-1]]))

    flux = heat_flux(t_m)
    skin = skin_temperature(t_m)
    t_safe = float(np.interp(-T_SKIN_LIMIT, -skin, t_mm))

    r_crit = CABLE_K / CABLE_H
    lag_mm = np.linspace(1e-3, 60.0, 900)
    q_cable = cylinder_loss(CABLE_R1 + 1e-3 * lag_mm, CABLE_R1, CABLE_K,
                            CABLE_H) / (2.0 * np.pi * CABLE_R1 * CABLE_H)
    q_pipe = cylinder_loss(PIPE_R1 + 1e-3 * lag_mm, PIPE_R1, K_INS,
                           H_OUT) / (2.0 * np.pi * PIPE_R1 * H_OUT)
    t_peak = float(lag_mm[int(np.argmax(q_cable))])
    r_peak = CABLE_R1 + 1e-3 * t_peak

    assert np.allclose(tot, e_cost + c_cost), \
        "the total curve must be the two component curves added"
    assert abs(t_cross - t_mm[i_opt]) < 0.06 * t_mm[i_opt], \
        "on a plane wall the crossing has to land on the optimum"
    assert np.all(np.diff(flux) < 0.0), "insulation cannot raise a plane loss"
    assert abs(r_peak - r_crit) < 0.2e-3, \
        "the cable peak must sit at the critical radius k/h"
    assert q_cable.max() > 1.15 and q_cable[-1] < q_cable.max(), \
        "the thin cable has to get worse before it gets better"
    assert np.all(np.diff(q_pipe) < 0.0), \
        "a pipe already larger than the critical radius only improves"

    payback = (COST_M3 * t_m[i_opt]) / float(e_cost[0] - e_cost[i_opt])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_skin = axes_inches(fig, AUX_T)
    ax_crit = axes_inches(fig, AUX_B)

    cost_panel(ax_main, t_mm, e_cost, c_cost, tot, i_opt, t_cross, band,
               payback)
    skin_panel(ax_skin, t_mm, flux, skin, t_safe, t_mm[i_opt])
    critical_panel(ax_crit, lag_mm, q_cable, q_pipe, r_crit, t_peak)



    panel_labels_inches(fig, [ax_main, ax_skin, ax_crit], dx=-0.62, dy=0.14)
    savefig(fig, "1043_ht_insulation_economics.png")


if __name__ == "__main__":
    main()
