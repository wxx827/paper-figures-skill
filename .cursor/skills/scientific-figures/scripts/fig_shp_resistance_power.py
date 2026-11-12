"""Resistance decomposition and the power prediction that follows from it.

The main panel stacks the four resistance components against ship speed, so
the total is the height of the stack rather than a fifth curve that could
disagree with it.  The frictional part is the ITTC-1957 line evaluated on the
plotted Reynolds number, the residuary part carries the wave-making hump, and
the two smaller terms are appendage and still-air drag.  Effective power runs
on the right-hand axis as the product of the plotted total and the plotted
speed, and the service speed is marked with the share each component holds
there - shares that are computed from the stack at that column.

The upper companion panel walks the same speeds up the transmission chain,
effective to delivered to brake power, and shows where the sea margin and the
maximum continuous rating fall.  The lower panel converts brake power into
fuel, in tonnes per day and in tonnes per nautical mile, which is the pair of
curves behind every slow-steaming decision.

Illustrative synthetic data for a 180 m hull; not a model-basin report.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
STEEL = "#5b6b7a"
BLUE = "#2e74a8"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.06, 5.26, 4.34)
AUX_T = (7.46, 3.62, 3.05, 1.78)
AUX_B = (7.46, 1.06, 3.05, 1.72)

G = 9.81
RHO_W, RHO_A = 1025.0, 1.225     # kg/m^3
NU = 1.19e-6                     # kinematic viscosity, m^2/s
L_PP, B_MLD, T_DRAFT, C_B = 180.0, 32.0, 10.5, 0.72
S_WET = 1.70 * L_PP * T_DRAFT + C_B * L_PP * B_MLD      # wetted surface, m^2
A_FRONT, CD_AIR = 880.0, 0.80    # transverse area (m^2), air drag coefficient
KN = 0.514444                    # knots to m/s

ETA_D, ETA_S = 0.705, 0.980      # quasi-propulsive and shaft efficiency
SEA_MARGIN = 0.15                # dimensionless
MCR_FRACTION = 0.85              # brake power at service speed / MCR
SFOC = 172.0                     # specific fuel oil consumption, g/(kW h)
V_SERVICE = 15.5                 # knots

COMPONENTS = [("frictional $R_F$ (ITTC-1957)", "#134a76"),
              ("residuary $R_R$ (wave making)", "#5f9dc6"),
              ("appendage $R_{APP}$", "#a9cddc"),
              ("still-air $R_{AA}$", "#c9a227")]


def resistance(v_kn):
    """Component resistances in kN on the given speed array (knots)."""
    v = v_kn * KN
    rn = np.maximum(v * L_PP / NU, 1e3)
    cf = 0.075 / (np.log10(rn) - 2.0) ** 2
    fn = v / np.sqrt(G * L_PP)
    cr = (0.00022 + 0.00068 * np.exp(-((fn - 0.207) / 0.026) ** 2)
          + 0.00100 * (fn / 0.245) ** 8)
    r_f = 0.5 * RHO_W * S_WET * cf * v ** 2 / 1000.0
    r_r = 0.5 * RHO_W * S_WET * cr * v ** 2 / 1000.0
    r_app = 0.075 * r_f
    r_aa = 0.5 * RHO_A * CD_AIR * A_FRONT * v ** 2 / 1000.0
    return np.vstack([r_f, r_r, r_app, r_aa])


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)

    v = np.linspace(6.0, 20.0, 561)
    parts = resistance(v)
    total = parts.sum(axis=0)
    p_eff = total * v * KN                       # kN * m/s = kW
    p_del = p_eff / ETA_D
    p_brake = p_del / ETA_S
    p_inst = p_brake * (1.0 + SEA_MARGIN)

    i_s = int(np.argmin(np.abs(v - V_SERVICE)))
    share = parts[:, i_s] / total[i_s] * 100.0
    p_mcr = float(p_inst[i_s]) / MCR_FRACTION

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_p = axes_inches(fig, AUX_T)
    ax_f = axes_inches(fig, AUX_B)
    for a in (ax, ax_p, ax_f):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the resistance stack ----------------------------------------
    ax.stackplot(v, parts, colors=[c for _, c in COMPONENTS],
                 labels=[n for n, _ in COMPONENTS], edgecolor="white",
                 linewidth=0.45, zorder=4)
    ax.plot(v, total, color=INK, lw=1.5, zorder=7)
    ax.text(19.85, float(total[-1]) * 0.985, "total $R_T$", fontsize=6.8,
            color=INK, ha="right", va="top", zorder=9)

    ax.axvline(V_SERVICE, color=RED, lw=1.1, ls=(0, (4.0, 2.2)), zorder=8)
    ax.plot([V_SERVICE], [total[i_s]], marker="o", ms=5.0, mfc="white",
            mec=RED, mew=1.4, zorder=10)
    ax.annotate("service speed %.1f knots\n$R_T$ = %.0f kN, $P_E$ = "
                "%.0f kW\nshares %.0f / %.0f / %.0f / %.0f %%"
                % (V_SERVICE, total[i_s], p_eff[i_s], *share),
                xy=(V_SERVICE, total[i_s]), xytext=(0.055, 0.965),
                textcoords="axes fraction", fontsize=6.8, color=RED,
                ha="left", va="top", linespacing=1.5, zorder=11,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                                shrinkA=2, shrinkB=5))

    fn_s = V_SERVICE * KN / np.sqrt(G * L_PP)
    share_r = parts[1] / total
    band = v < 18.5
    i_h = int(np.argmax(np.where(band, share_r, -1.0)))
    ax.annotate("wave-making hump in the plotted share:\n%.1f knots, $F_n$ = "
                "%.3f, %.0f %% of $R_T$\n(%.0f %% at the service $F_n$ of "
                "%.3f)"
                % (v[i_h], v[i_h] * KN / np.sqrt(G * L_PP),
                   share_r[i_h] * 100.0, share_r[i_s] * 100.0, fn_s),
                xy=(v[i_h], float(parts[0, i_h] + parts[1, i_h] * 0.5)),
                xytext=(0.048, 0.720), textcoords="axes fraction",
                fontsize=6.8, color=INK, ha="left", va="top",
                linespacing=1.5, zorder=11,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                shrinkA=2, shrinkB=4))

    ax_e = ax.twinx()
    ax_e.plot(v, p_eff, color=GREEN, lw=1.8, ls=(0, (5.0, 2.2)), zorder=9)
    ax_e.set_ylim(0.0, float(p_eff.max()) * 1.16)
    ax_e.set_xlim(6.0, 20.0)
    ax_e.spines["top"].set_visible(False)
    ax_e.tick_params(labelsize=7.0, labelcolor=GREEN, length=3)
    ax_e.set_ylabel("effective power $P_E = R_T V$ (kW)", fontsize=8.4,
                    color=GREEN)
    ax_e.grid(False)

    ax.set_xlim(6.0, 20.0)
    ax.set_ylim(0.0, float(total.max()) * 1.16)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("ship speed V (knots)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("resistance (kN)", fontsize=8.4)
    ax.set_title("Resistance components stack to the total",
                 fontsize=9.2, fontweight="bold", pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=6.9,
              handlelength=1.5, labelspacing=0.30, borderaxespad=0.35,
              bbox_to_anchor=(0.005, 0.615))

    # ---- aux (b): the transmission chain -----------------------------------
    for arr, col, lab in ((p_eff, GREEN, "effective $P_E$"),
                          (p_del, BLUE, "delivered $P_D = P_E/\\eta_D$"),
                          (p_brake, STEEL, "brake $P_B = P_D/\\eta_S$"),
                          (p_inst, GOLD, "installed, 15 % sea margin")):
        ax_p.plot(v, arr / 1000.0, color=col, lw=1.7, zorder=6, label=lab)
    ax_p.axhline(p_mcr / 1000.0, color=RED, lw=1.1, ls=(0, (4.0, 2.2)),
                 zorder=7)
    ax_p.text(19.8, p_mcr / 1000.0 * 1.035,
              "MCR %.0f kW (%.0f %% at service speed)"
              % (p_mcr, MCR_FRACTION * 100.0), fontsize=6.2, color=RED,
              ha="right", va="bottom", zorder=8)
    ax_p.axvline(V_SERVICE, color=FAINT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax_p.set_xlim(6.0, 20.0)
    ax_p.set_ylim(0.0, float(p_inst.max()) / 1000.0 * 1.12)
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_xlabel("ship speed V (knots)", fontsize=7.5, labelpad=1.5)
    ax_p.set_ylabel("power (MW)", fontsize=7.5)
    ax_p.set_title("Effective to installed power", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_p.legend(loc="upper left", frameon=False, fontsize=6.2,
                handlelength=1.4, labelspacing=0.26, borderaxespad=0.30)

    # ---- aux (c): fuel ------------------------------------------------------
    t_day = SFOC * p_brake * 24.0 / 1e6
    t_nm = SFOC * p_brake / 1e6 / v
    ax_f.plot(v, t_day, color=STEEL, lw=1.9, zorder=6)
    ax_f.set_xlim(6.0, 20.0)
    ax_f.set_ylim(0.0, float(t_day.max()) * 1.22)
    ax_f.tick_params(labelsize=6.8)
    ax_f.set_xlabel("ship speed V (knots)", fontsize=7.5, labelpad=1.5)
    ax_f.set_ylabel("fuel burn (t/day)", fontsize=7.5, color=STEEL)
    ax_f.tick_params(axis="y", labelcolor=STEEL)

    ax_f2 = ax_f.twinx()
    ax_f2.plot(v, t_nm, color=RED, lw=1.9, ls=(0, (5.0, 2.2)), zorder=6)
    ax_f2.set_ylim(0.0, float(t_nm.max()) * 1.22)
    ax_f2.spines["top"].set_visible(False)
    ax_f2.tick_params(labelsize=6.8, labelcolor=RED, length=3)
    ax_f2.set_ylabel("fuel per distance (t/nm)", fontsize=7.5, color=RED)
    ax_f2.grid(False)

    v_slow = 12.0
    j = int(np.argmin(np.abs(v - v_slow)))
    for a, arr, col in ((ax_f, t_day, STEEL), (ax_f2, t_nm, RED)):
        a.plot([v[i_s], v[j]], [arr[i_s], arr[j]], linestyle="none",
               marker="o", ms=4.0, mfc="white", mec=col, mew=1.3, zorder=9)
    ax_f.text(0.035, 0.955,
              "slowing %.1f to %.1f knots cuts the daily burn %.0f %%\n"
              "and the burn per mile %.0f %%"
              % (V_SERVICE, v_slow, (1.0 - t_day[j] / t_day[i_s]) * 100.0,
                 (1.0 - t_nm[j] / t_nm[i_s]) * 100.0),
              transform=ax_f.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45, zorder=10)
    ax_f.set_title("Fuel is the cube of the decision", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_p, ax_f], dx=-0.64, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 999")
    check_layout(fig, foot, [ax, ax_p, ax_f])
    savefig(fig, "999_shp_resistance_power.png")


if __name__ == "__main__":
    main()
