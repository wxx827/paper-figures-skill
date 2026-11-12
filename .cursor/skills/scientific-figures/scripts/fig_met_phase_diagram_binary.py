"""Eutectic binary phase diagram with a lever-rule cooling path.

The main panel is the map: liquidus, solidus, solvus and the eutectic
isotherm of a model A-B system, with every two-phase field tinted and named.
A vertical dashed line follows one alloy down through the diagram, and the
tie lines drawn across it carry phase fractions that are evaluated from the
same boundary functions the curves are drawn from - change a boundary and
the printed fractions move with it.  The two smaller panels are the
supporting evidence for that path: the cooling curve it produces (with the
eutectic arrest whose length is set by the liquid fraction left at the
eutectic) and the phase inventory as a stacked area.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"        # liquid / high temperature
OCHRE = "#c0603f"      # liquid + beta field
BLUE = "#3b5b92"       # alpha solid solution
GREEN = "#2e7d5b"      # beta solid solution
GOLD = "#c9a227"       # alpha + beta eutectic microstructure
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"

T_A = 660.0            # melting point of pure A, deg C
T_B = 780.0            # melting point of pure B, deg C
T_E = 420.0            # eutectic temperature, deg C
C_E = 42.0             # eutectic composition, wt% B
C_AE = 12.0            # solubility limit of B in alpha at T_E
C_BE = 88.0            # composition of beta at T_E
C_A_RT = 1.5           # solvus intercepts at the bottom of the diagram
C_B_RT = 98.5
T_RT = 40.0
T_TOP = 810.0
C0 = 30.0              # alloy followed down the diagram, wt% B


def liquidus_a(t):
    s = np.clip((T_A - np.asarray(t, float)) / (T_A - T_E), 0.0, 1.0)
    return C_E * s ** 0.85


def solidus_a(t):
    s = np.clip((T_A - np.asarray(t, float)) / (T_A - T_E), 0.0, 1.0)
    return C_AE * s ** 1.15


def liquidus_b(t):
    u = np.clip((T_B - np.asarray(t, float)) / (T_B - T_E), 0.0, 1.0)
    return 100.0 - (100.0 - C_E) * u ** 0.85


def solidus_b(t):
    u = np.clip((T_B - np.asarray(t, float)) / (T_B - T_E), 0.0, 1.0)
    return 100.0 - (100.0 - C_BE) * u ** 1.15


def solvus_a(t):
    r = np.clip((np.asarray(t, float) - T_RT) / (T_E - T_RT), 0.0, 1.0)
    return C_A_RT + (C_AE - C_A_RT) * r ** 1.6


def solvus_b(t):
    r = np.clip((np.asarray(t, float) - T_RT) / (T_E - T_RT), 0.0, 1.0)
    return C_B_RT - (C_B_RT - C_BE) * r ** 1.6


# Liquidus temperature of the C0 alloy, obtained by inverting liquidus_a.
T_LIQ = T_A - (C0 / C_E) ** (1.0 / 0.85) * (T_A - T_E)


def phase_fractions(t):
    """Lever-rule fractions of liquid, alpha and beta for the C0 alloy."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    f_l = np.zeros_like(t)
    f_a = np.zeros_like(t)
    f_b = np.zeros_like(t)

    f_l[t >= T_LIQ] = 1.0

    m = (t < T_LIQ) & (t >= T_E)
    if m.any():
        cl, ca = liquidus_a(t[m]), solidus_a(t[m])
        x = np.clip((C0 - ca) / np.maximum(cl - ca, 1e-9), 0.0, 1.0)
        f_l[m], f_a[m] = x, 1.0 - x

    s = t < T_E
    if s.any():
        ca, cb = solvus_a(t[s]), solvus_b(t[s])
        y = np.clip((cb - C0) / np.maximum(cb - ca, 1e-9), 0.0, 1.0)
        f_a[s], f_b[s] = y, 1.0 - y
    return f_l, f_a, f_b


def boundary_curves(n=240):
    t_left = np.linspace(T_E, T_A, n)
    t_right = np.linspace(T_E, T_B, n)
    t_sol = np.linspace(T_RT, T_E, n)
    return {
        "liq_a": np.column_stack([liquidus_a(t_left), t_left]),
        "sol_a": np.column_stack([solidus_a(t_left), t_left]),
        "liq_b": np.column_stack([liquidus_b(t_right), t_right]),
        "sol_b": np.column_stack([solidus_b(t_right), t_right]),
        "solv_a": np.column_stack([solvus_a(t_sol), t_sol]),
        "solv_b": np.column_stack([solvus_b(t_sol), t_sol]),
    }


def cooling_curve(t_start=770.0, t_stop=150.0, rate=3.0, latent=95.0,
                  latent_e=155.0, n=1400):
    """Temperature against time for the C0 alloy, slowed by latent heat."""
    temp = np.linspace(t_start, t_stop, n)
    f_l, _, _ = phase_fractions(temp)
    f_s = 1.0 - f_l
    d_t = np.abs(np.diff(temp))
    d_fs = np.diff(f_s)
    step = d_t / rate * (1.0 + latent * np.clip(d_fs, 0.0, None) / d_t)

    plateau = latent_e * float(phase_fractions(T_E + 0.5)[0][0]) / rate
    i_e = int(np.argmin(np.abs(temp - T_E)))
    step[i_e] += plateau

    time = np.concatenate([[0.0], np.cumsum(step)])
    return time, temp, plateau, time[i_e]


def region(ax, pts, color, z=1):
    ax.add_patch(Polygon(np.asarray(pts, float), closed=True,
                         facecolor=color, edgecolor="none", zorder=z))


def tie_line(ax, t_hold, c_lo, c_hi, f_lo, lab_lo, lab_hi, color,
             hi_outside=False):
    """Horizontal tie line with end caps and the two lever-rule fractions."""
    ax.plot([c_lo, c_hi], [t_hold, t_hold], color=color, lw=0.9,
            ls=(0, (3.2, 2.0)), zorder=6)
    for c in (c_lo, c_hi):
        ax.plot([c, c], [t_hold - 11.0, t_hold + 11.0], color=color, lw=0.9,
                zorder=6)
    ax.plot([C0], [t_hold], marker="o", ms=3.4, color=INK, mfc="white",
            mew=0.9, zorder=7)
    chip = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.88)
    ax.text(c_lo + 1.4, t_hold + 14.0, f"{lab_lo} {f_lo * 100:.0f} %",
            ha="left", va="bottom", fontsize=6.6, color=color, zorder=8,
            bbox=chip)
    x_hi, ha_hi = ((c_hi + 1.6, "left") if hi_outside
                   else (c_hi - 1.4, "right"))
    ax.text(x_hi, t_hold + 14.0, f"{lab_hi} {(1.0 - f_lo) * 100:.0f} %",
            ha=ha_hi, va="bottom", fontsize=6.6, color=color, zorder=8,
            bbox=chip)


def main_panel(ax, cv):
    top = [(0.0, T_TOP), (100.0, T_TOP)]
    region(ax, np.vstack([cv["liq_a"], top, cv["liq_b"][::-1]]),
           lighten(RED, 0.74))
    region(ax, np.vstack([cv["sol_a"], cv["liq_a"][::-1]]),
           lighten(RED, 0.88))
    region(ax, np.vstack([cv["sol_b"], cv["liq_b"][::-1]]),
           lighten(OCHRE, 0.88))
    region(ax, np.vstack([cv["solv_a"], cv["sol_a"], [(0.0, T_RT)]]),
           lighten(BLUE, 0.78))
    region(ax, np.vstack([cv["solv_b"], cv["sol_b"], [(100.0, T_RT)]]),
           lighten(GREEN, 0.80))
    region(ax, np.vstack([cv["solv_a"], cv["solv_b"][::-1]]),
           lighten(GOLD, 0.86))

    ax.plot(cv["liq_a"][:, 0], cv["liq_a"][:, 1], color=RED, lw=1.7, zorder=5,
            label="liquidus")
    ax.plot(cv["liq_b"][:, 0], cv["liq_b"][:, 1], color=RED, lw=1.7, zorder=5)
    ax.plot(cv["sol_a"][:, 0], cv["sol_a"][:, 1], color=BLUE, lw=1.4, zorder=5,
            label="solidus")
    ax.plot(cv["sol_b"][:, 0], cv["sol_b"][:, 1], color=GREEN, lw=1.4,
            zorder=5)
    ax.plot(cv["solv_a"][:, 0], cv["solv_a"][:, 1], color=BLUE, lw=1.1,
            ls=(0, (5, 2.2)), zorder=5, label="solvus")
    ax.plot(cv["solv_b"][:, 0], cv["solv_b"][:, 1], color=GREEN, lw=1.1,
            ls=(0, (5, 2.2)), zorder=5)
    ax.plot([C_AE, C_BE], [T_E, T_E], color=INK, lw=1.8, zorder=5,
            label="eutectic isotherm")

    ax.plot([C_E], [T_E], marker="o", ms=5.0, color=INK, mfc=GOLD, mew=1.0,
            zorder=8)
    ax.annotate(f"eutectic point\n{C_E:.0f} wt% B,  {T_E:.0f}" +
                r" $^\circ$C",
                xy=(C_E, T_E), xytext=(C_E + 9.0, T_E - 78.0), ha="left",
                va="top", fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    ax.plot([C0, C0], [T_RT, T_TOP], color=INK, lw=1.1, ls=(0, (4, 2.4)),
            zorder=6)
    ax.text(C0 + 1.8, T_TOP - 12.0, f"alloy $C_0$ = {C0:.0f} wt% B",
            ha="left", va="top", fontsize=7.0, color=INK)
    ax.plot([C0], [T_LIQ], marker="v", ms=4.6, color=RED, zorder=8)
    ax.text(C0 + 2.4, T_LIQ + 7.0,
            f"$T_L$ = {T_LIQ:.0f}" + r" $^\circ$C", ha="left", va="bottom",
            fontsize=6.6, color=RED)

    t1 = 462.0
    f_l1 = float(phase_fractions(t1)[0][0])
    tie_line(ax, t1, solidus_a(t1), liquidus_a(t1), 1.0 - f_l1,
             r"$\alpha$", "L", BLUE, hi_outside=True)
    t2 = 235.0
    f_a2 = float(phase_fractions(t2)[1][0])
    tie_line(ax, t2, solvus_a(t2), solvus_b(t2), f_a2,
             r"$\alpha$", r"$\beta$", SUB)

    ax.text(62.0, 720.0, "L", fontsize=10.5, color=RED, ha="center",
            va="center", zorder=7)
    ax.text(4.6, 505.0, r"$\alpha$", fontsize=9.5, color=BLUE, ha="center",
            va="center", zorder=7)
    ax.text(95.6, 505.0, r"$\beta$", fontsize=9.5, color=GREEN, ha="center",
            va="center", zorder=7)
    ax.text(21.0, 447.0, r"L + $\alpha$", fontsize=8.4, color="#8c2f2f",
            ha="center", va="center", zorder=7)
    ax.text(78.0, 570.0, r"L + $\beta$", fontsize=8.4, color="#8c2f2f",
            ha="center", va="center", zorder=7)
    ax.text(50.0, 130.0, r"$\alpha$ + $\beta$", fontsize=9.5, color="#8a6a12",
            ha="center", va="center", zorder=7)

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(T_RT, T_TOP)
    ax.set_xlabel("composition (wt% B)")
    ax.set_ylabel(r"temperature ($^\circ$C)")
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(100, 801, 100))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.tick_params(labelsize=7.2)
    ax.legend(fontsize=6.8, frameon=True, framealpha=0.92, loc="upper left",
              handlelength=1.6, borderaxespad=0.35, edgecolor="#cccccc")
    ax.set_title("Model A-B system: eutectic at "
                 f"{C_E:.0f} wt% B and {T_E:.0f}" + r" $^\circ$C",
                 fontsize=9.0, fontweight="bold", color=INK)


def cooling_panel(ax, time, temp, plateau, t_at_e):
    ax.plot(time, temp, color=RED, lw=1.6, zorder=4)
    ax.axhline(T_E, color=INK, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
    ax.axhline(T_LIQ, color=GREY, lw=0.8, ls=(0, (2, 2.0)), zorder=2)

    i_l = int(np.argmin(np.abs(temp - T_LIQ)))
    ax.plot([time[i_l]], [T_LIQ], marker="v", ms=4.2, color=RED, zorder=6)
    ax.annotate(f"liquidus arrest  {T_LIQ:.0f}" + r" $^\circ$C",
                xy=(time[i_l], T_LIQ), xytext=(time[i_l] + 14.0,
                                               T_LIQ + 96.0),
                ha="left", va="bottom", fontsize=6.4, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.annotate(f"eutectic plateau  {plateau:.0f} s",
                xy=(t_at_e + plateau * 0.5, T_E),
                xytext=(t_at_e + plateau * 0.5, T_E - 128.0), ha="center",
                va="top", fontsize=6.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    ax.set_xlim(0.0, time[-1])
    ax.set_ylim(140.0, 790.0)
    ax.set_xlabel("time (s)", fontsize=7.4)
    ax.set_ylabel(r"temperature ($^\circ$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(r"Cooling curve along $C_0$", fontsize=8.2,
                 fontweight="bold", color=INK)


def fraction_panel(ax, temp_grid):
    f_l, f_a, f_b = phase_fractions(temp_grid)
    ax.stackplot(temp_grid, f_l, f_a, f_b,
                 colors=[lighten(RED, 0.42), lighten(BLUE, 0.42),
                         lighten(GREEN, 0.42)],
                 edgecolor="white", linewidth=0.4,
                 labels=["L", r"$\alpha$", r"$\beta$"])
    for t_mark, col in [(T_LIQ, GREY), (T_E, INK)]:
        ax.axvline(t_mark, color=col, lw=0.9, ls=(0, (3, 2.2)), zorder=5)
    f_e = float(phase_fractions(T_E + 0.5)[0][0])
    ax.annotate(f"{f_e * 100:.0f} % liquid left\nat the eutectic",
                xy=(T_E + 6.0, f_e * 0.5), xytext=(T_E - 118.0, 0.40),
                ha="left", va="center", fontsize=6.4, color=INK,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    ax.set_xlim(temp_grid.max(), temp_grid.min())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"temperature ($^\circ$C), cooling to the right",
                  fontsize=7.4)
    ax.set_ylabel("phase fraction (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=3,
              handlelength=1.2, borderaxespad=0.25, columnspacing=1.1)
    ax.set_title("Phase inventory from the lever rule", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    cv = boundary_curves()
    time, temp, plateau, t_at_e = cooling_curve()
    grid = np.linspace(150.0, 770.0, 900)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.80, 0.92, 5.55, 4.52))
    ax_cool = axes_inches(fig, (7.20, 3.55, 2.95, 1.89))
    ax_frac = axes_inches(fig, (7.20, 0.92, 2.95, 1.89))

    main_panel(ax_main, cv)
    cooling_panel(ax_cool, time, temp, plateau, t_at_e)
    fraction_panel(ax_frac, grid)

    f_e = float(phase_fractions(T_E + 0.5)[0][0])
    panel_labels_inches(fig, [ax_main, ax_cool, ax_frac], dx=-0.46, dy=0.10)
    savefig(fig, "828_met_phase_diagram_binary.png")


if __name__ == "__main__":
    main()
