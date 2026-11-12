"""Ellingham diagram for oxide formation, with the carbon reduction lines.

The main panel is the standard construction: standard free energy of
formation per mole of O2 against temperature, one straight line per
oxidation reaction, the slope carrying minus the entropy of reaction.  The
two carbon lines run the other way and cut across the metal lines; where
they cut is the temperature above which carbon reduces that oxide, and every
one of those temperatures is solved from the same line coefficients that are
drawn.  The nomographic rays that make this diagram unusual - oxygen
isobars and CO/CO2 ratios, all radiating from their origins at absolute zero
off the left of the frame - come from the ideal-gas relation rather than
being traced by hand.  The two smaller panels rank the reduction
temperatures and read the stability order off the lines at one temperature.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
OCHRE = "#c0603f"
TEAL = "#2a7f7a"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

T_LO, T_HI = 200.0, 2000.0          # temperature axis, deg C
G_LO, G_HI = -1300.0, 60.0          # free energy axis, kJ per mol O2
T_PICK = 1200.0                     # temperature for the stability ranking
T_ABS = 273.15
R_LN10 = 8.314e-3 * np.log(10.0)    # kJ / (mol K) per decade

# reaction label, short name, colour, intercept at 0 deg C (kJ/mol O2),
# slope (kJ/mol O2 per deg C), temperature at which the line is labelled
OXIDES = [
    (r"2 Ca + O$_2$ $\rightarrow$ 2 CaO", "CaO", PURPLE,
     -1265.3, 0.2176, 1950.0),
    (r"4/3 Al + O$_2$ $\rightarrow$ 2/3 Al$_2$O$_3$", "Al$_2$O$_3$", BLUE,
     -1113.5, 0.2118, 1950.0),
    (r"Ti + O$_2$ $\rightarrow$ TiO$_2$", "TiO$_2$", TEAL,
     -934.1, 0.1471, 900.0),
    (r"Si + O$_2$ $\rightarrow$ SiO$_2$", "SiO$_2$", GOLD,
     -861.8, 0.2059, 1950.0),
    (r"2 Mn + O$_2$ $\rightarrow$ 2 MnO", "MnO", GREEN,
     -758.2, 0.1941, 900.0),
    (r"2 Fe + O$_2$ $\rightarrow$ 2 FeO", "FeO", RED,
     -525.9, 0.1529, 1950.0),
    (r"2 Ni + O$_2$ $\rightarrow$ 2 NiO", "NiO", "#8c2f2f",
     -439.4, 0.1647, 1950.0),
    (r"4 Cu + O$_2$ $\rightarrow$ 2 Cu$_2$O", "Cu$_2$O", OCHRE,
     -314.1, 0.1471, 1950.0),
]

# The two carbon lines.  2 C + O2 -> 2 CO creates gas, so its entropy of
# reaction is positive and the line falls instead of rising - that single
# sign is why coke can reduce almost anything if you get hot enough.
CO_LINE = (r"2 C + O$_2$ $\rightarrow$ 2 CO", "2C+O$_2$", INK,
           -227.1, -0.1765, 1950.0)
CO2_LINE = (r"C + O$_2$ $\rightarrow$ CO$_2$", "C+O$_2$", SUB,
            -394.6, -0.0012, 1300.0)

P_RAYS = [(-1.0, 1000.0), (-8.0, 700.0), (-16.0, 1750.0), (-28.0, 1800.0)]
Q_RAYS = [(2.0, 1985.0), (-2.0, 1985.0)]


def line(t, icept, slope):
    return icept + slope * np.asarray(t, dtype=float)


def crossing(row, other):
    """Temperature (deg C) where two straight lines meet."""
    return (other[3] - row[3]) / (row[4] - other[4])


def reduction_temperatures():
    return [(row[1], row[2], crossing(row, CO_LINE)) for row in OXIDES]


def main_panel(ax, red_temps):
    t = np.linspace(T_LO, T_HI, 200)
    ax.axhline(0.0, color=HAIR, lw=0.8, zorder=1)
    nomograph(ax)

    for label, short, color, icept, slope, t_lab in OXIDES:
        ax.plot(t, line(t, icept, slope), color=color, lw=1.5, zorder=5)
        ax.text(t_lab, line(t_lab, icept, slope) + 13.0, label, ha="right",
                va="bottom", fontsize=6.3, color=color, zorder=7)

    for label, short, color, icept, slope, t_lab in (CO_LINE, CO2_LINE):
        ax.plot(t, line(t, icept, slope), color=color, lw=2.0,
                ls=(0, (6, 2.2)), zorder=6)
        ax.text(t_lab, line(t_lab, icept, slope) + 14.0, label, ha="right",
                va="bottom", fontsize=6.8, color=color, fontweight="bold",
                zorder=7)

    for short, color, t_x in red_temps:
        if not (T_LO + 20.0 < t_x < T_HI):
            continue
        g_x = line(t_x, CO_LINE[3], CO_LINE[4])
        ax.plot([t_x], [g_x], marker="o", ms=4.8, color=color, mfc="white",
                mew=1.3, zorder=9)
        ax.plot([t_x, t_x], [G_LO, g_x], color=color, lw=0.7,
                ls=(0, (2, 2.4)), alpha=0.8, zorder=4)
        ax.text(t_x, G_LO + 14.0, f"{short}\n{t_x:.0f}" + r" $^\circ$C",
                ha="center", va="bottom", fontsize=6.0, color=color,
                zorder=9)

    ax.axvline(T_PICK, color=GREY, lw=1.0, ls=(0, (5, 2.4)), zorder=4)
    ax.text(T_PICK - 16.0, G_HI - 18.0, f"{T_PICK:.0f}" + r" $^\circ$C",
            ha="right", va="top", fontsize=6.6, color=GREY, zorder=8)

    ax.text(T_LO + 30.0, G_HI - 16.0,
            "grey dotted:  oxygen isobars $p_{\\mathrm{O}_2}$ (atm)\n"
            "ochre dashed:  CO/CO$_2$ ratio rays",
            ha="left", va="top", fontsize=6.3, color=SUB, zorder=8,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.94))

    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(G_LO, G_HI)
    ax.set_xlabel(r"temperature ($^\circ$C)")
    ax.set_ylabel(r"$\Delta G^{\circ}$ of formation (kJ per mol O$_2$)")
    ax.set_xticks(np.arange(200, 2001, 200))
    ax.set_yticks(np.arange(-1200, 1, 200))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.tick_params(labelsize=7.2)
    ax.set_title("Oxide stability and the reach of carbon reduction",
                 fontsize=9.0, fontweight="bold", color=INK)


def nomograph(ax):
    """Oxygen isobars and CO/CO2 rays, both radiating from absolute zero."""
    chip = dict(boxstyle="round,pad=0.16", facecolor="white",
                edgecolor="none", alpha=0.88)

    for p, t_lab in P_RAYS:
        slope = R_LN10 * p
        g = slope * (np.array([T_LO, T_HI]) + T_ABS)
        if g[1] < G_LO:
            t_end = G_LO / slope - T_ABS
            g = np.array([slope * (T_LO + T_ABS), G_LO])
            xs = np.array([T_LO, t_end])
        else:
            xs = np.array([T_LO, T_HI])
        ax.plot(xs, g, color=GREY, lw=0.65, ls=(0, (1.5, 2.0)), zorder=2)
        ax.text(t_lab, slope * (t_lab + T_ABS), r"$10^{%.0f}$" % p,
                ha="center", va="center", fontsize=6.0, color=GREY,
                zorder=8, bbox=chip)

    g0 = line(-T_ABS, CO_LINE[3], CO_LINE[4])
    for q, t_lab in Q_RAYS:
        slope = CO_LINE[4] + 2.0 * R_LN10 * q
        xs = np.array([T_LO, T_HI])
        ax.plot(xs, g0 + slope * (xs + T_ABS), color=OCHRE, lw=0.65,
                ls=(0, (3.0, 2.0)), zorder=2)
        ax.text(t_lab, g0 + slope * (t_lab + T_ABS), r"$10^{%.0f}$" % q,
                ha="center", va="center", fontsize=6.0, color=OCHRE,
                zorder=8, bbox=chip)


def ranking_panel(ax, red_temps):
    rows = sorted(red_temps, key=lambda r: r[2])
    names = [r[0] for r in rows]
    temps = np.array([r[2] for r in rows])
    cols = [lighten(r[1], 0.32) for r in rows]

    y = np.arange(len(rows))[::-1]
    ax.barh(y, temps, height=0.62, color=cols, edgecolor="white",
            linewidth=0.7, zorder=3)
    for yi, t_x in zip(y, temps):
        ax.text(t_x - 60.0, yi, f"{t_x:.0f}", ha="right", va="center",
                fontsize=6.2, color=INK, zorder=5)
    ax.axvline(T_PICK, color=GREY, lw=1.0, ls=(0, (4, 2.2)), zorder=4)
    ax.text(T_PICK - 70.0, y[0] + 0.52, f"{T_PICK:.0f}" + r" $^\circ$C",
            ha="right", va="bottom", fontsize=6.2, color=GREY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_ylim(-0.62, y[0] + 1.05)
    ax.set_xlim(0.0, temps.max() * 1.14)
    ax.set_xlabel(r"carbon reduction temperature ($^\circ$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title(r"Where 2 C + O$_2$ $\rightarrow$ 2 CO takes over",
                 fontsize=8.2, fontweight="bold", color=INK)


def stability_panel(ax):
    vals = [(row[1], row[2], line(T_PICK, row[3], row[4])) for row in OXIDES]
    vals.append((CO_LINE[1], INK, line(T_PICK, CO_LINE[3], CO_LINE[4])))
    vals.sort(key=lambda v: v[2])
    names = [v[0] for v in vals]
    g = np.array([v[2] for v in vals])
    cols = [lighten(v[1], 0.32) for v in vals]

    x = np.arange(len(vals))
    ax.bar(x, g, width=0.66, color=cols, edgecolor="white", linewidth=0.7,
           zorder=3)
    i_c = names.index(CO_LINE[1])
    ax.bar([x[i_c]], [g[i_c]], width=0.66, color=lighten(INK, 0.55),
           edgecolor=INK, linewidth=0.9, zorder=4)
    for xi, gi in zip(x, g):
        ax.text(xi, gi - 30.0, f"{gi:.0f}", ha="center", va="top",
                fontsize=6.0, color=INK, zorder=5)
    ax.axhline(0.0, color="#333333", lw=0.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.2, rotation=40.0, ha="right")
    ax.set_ylim(G_LO, G_HI)
    ax.set_yticks(np.arange(-1200, 1, 300))
    ax.set_ylabel(r"$\Delta G^{\circ}$ (kJ/mol O$_2$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    below, above = names[i_c - 1], names[i_c + 1]
    ax.set_title(f"At {T_PICK:.0f}" + r" $^\circ$C carbon sits between "
                 + f"{below} and {above}", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    red_temps = reduction_temperatures()

    fig = plt.figure(figsize=(10.4, 6.7))
    ax_main = axes_inches(fig, (0.88, 0.98, 5.48, 4.76))
    ax_rank = axes_inches(fig, (7.30, 3.72, 2.86, 2.02))
    ax_stab = axes_inches(fig, (7.30, 1.16, 2.86, 1.72))

    main_panel(ax_main, red_temps)
    ranking_panel(ax_rank, red_temps)
    stability_panel(ax_stab)

    easy = min(red_temps, key=lambda r: r[2])
    hard = max(red_temps, key=lambda r: r[2])
    panel_labels_inches(fig, [ax_main, ax_rank, ax_stab], dx=-0.54, dy=0.10)
    savefig(fig, "830_met_ellingham.png")


if __name__ == "__main__":
    main()
