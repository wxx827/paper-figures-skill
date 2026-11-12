"""Why every reactor programme picks D-T: four cross sections on one axis.

The cross sections are astrophysical S-factors divided by the Gamow
penetration term, sigma(E) = S(E) / (E exp(B_G / sqrt(E))), with the
Bosch-Hale rational S(E) for D-T, both D-D branches and D-3He, and a
resonance-shaped S(E) for p-11B whose Gamow constant is computed from Z1 Z2
and the reduced mass rather than copied.  That single line of physics is
enough to reproduce the fact the figure exists to show: D-T peaks near 64 keV
with a cross section roughly two orders of magnitude above its rivals at the
temperatures a reactor can reach.

The lower main panel is the same four curves seen through a Maxwellian,
<sigma v> = sqrt(8 / pi m_r) T^-3/2 * integral sigma(E) E exp(-E/T) dE,
integrated numerically on a log energy grid, and the peak positions and the
D-T check value at 10 keV are read back off those arrays.  The supporting
panels are two consequences: the temperature at which each fuel's required
triple product is cheapest, and the split of the released energy between
neutrons, which only heat a blanket, and charged particles, which can heat
the plasma.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

RED = "#b23a48"          # D-T
ORANGE = "#c0603f"       # D-D
BLUE = "#3b5b92"         # D-3He
PURPLE = "#8e6bbf"       # p-11B
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

KEV = 1.602176634e-16
MB = 1.0e-31                     # m^2 per millibarn
AMU = 1.66053907e-27
C_BREM = 5.34e-37                # W m^3 keV^-1/2

E_CM = np.logspace(np.log10(1.0), np.log10(4000.0), 1400)     # keV
T_ION = np.logspace(np.log10(1.0), np.log10(500.0), 260)      # keV
E_VIEW = 1000.0                  # keV, right edge of the cross-section panel

# Bosch-Hale rational S-factors: (B_G, A1..A5, B1..B4), E in keV, S in keV mb
BH = {
    "DT": (34.3827, (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0),
           (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4)),
    "DDp": (31.3970, (5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10),
            (0.0, 0.0, 0.0, 0.0)),
    "DDn": (31.3970, (5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9),
            (0.0, 0.0, 0.0, 0.0)),
    "DHe3": (68.7508, (5.7501e6, 2.5226e3, 4.5566e1, 0.0, 0.0),
             (-3.1995e-3, -8.5530e-6, 5.9014e-8, 0.0)),
}

FUELS = [
    ("D-T", RED, 2.0141, 3.0160, 17590.0, 14060.0, "$^2$H + $^3$H"),
    ("D-D", ORANGE, 2.0141, 2.0141, 3650.0, 1225.0, "$^2$H + $^2$H"),
    ("D-$^3$He", BLUE, 2.0141, 3.0160, 18350.0, 0.0, "$^2$H + $^3$He"),
    ("p-$^{11}$B", PURPLE, 1.0073, 11.0093, 8680.0, 0.0, "$^1$H + $^{11}$B"),
]

FW, FH = 10.4, 6.4
TOPM = (0.90, 3.44, 4.78, 1.76)
BOTM = (0.90, 1.06, 4.78, 1.66)
AUX1 = (6.62, 3.32, 2.80, 1.88)
AUX2 = (6.62, 1.06, 2.80, 1.52)


# ------------------------------------------------------------------- physics
def gamow_bg(z1, z2, a1, a2):
    """Gamow constant sqrt(E_G) in keV^1/2 from charges and reduced mass."""
    mr = a1 * a2 / (a1 + a2) * AMU
    mrc2 = mr * (2.99792458e8) ** 2 / KEV        # keV
    return np.sqrt((np.pi / 137.035999 * z1 * z2) ** 2 * 2.0 * mrc2)


def bosch_hale(key, e):
    bg, a, b = BH[key]
    num = a[0] + e * (a[1] + e * (a[2] + e * (a[3] + e * a[4])))
    den = 1.0 + e * (b[0] + e * (b[1] + e * (b[2] + e * b[3])))
    return np.maximum(num / den, 0.0) / (e * np.exp(bg / np.sqrt(e)))


def sigma_pb11(e):
    """p-11B: Gamow term from first principles, S(E) shaped by resonances."""
    bg = gamow_bg(1.0, 5.0, 1.0073, 11.0093)
    s = (3.30e8 / (1.0 + ((e - 590.0) / 500.0) ** 2)
         + 0.95e8 * np.exp(-0.5 * ((e - 148.0) / 24.0) ** 2))
    return s / (e * np.exp(bg / np.sqrt(e)))


def cross_sections():
    out = {"D-T": bosch_hale("DT", E_CM),
           "D-D": bosch_hale("DDp", E_CM) + bosch_hale("DDn", E_CM),
           "D-$^3$He": bosch_hale("DHe3", E_CM),
           "p-$^{11}$B": sigma_pb11(E_CM)}
    return {k: np.clip(v, 1e-12, None) for k, v in out.items()}


def reactivity(sig_mb, mr_amu, temps=T_ION):
    """Maxwellian <sigma v> in m^3/s from a tabulated cross section."""
    sig = sig_mb * MB
    ej = E_CM * KEV
    out = np.empty(temps.size)
    for i, t in enumerate(temps):
        w = np.exp(-E_CM / t)
        out[i] = (np.sqrt(8.0 / (np.pi * mr_amu * AMU)) * (t * KEV) ** -1.5 *
                  np.trapezoid(sig * ej * w, ej))
    return out


def best_temperature(sv, e_charged_kev):
    """Temperature minimising the required triple product for this fuel."""
    heat = 0.25 * sv * e_charged_kev * KEV
    loss = C_BREM * np.sqrt(T_ION)
    net = heat - loss
    with np.errstate(divide="ignore", invalid="ignore"):
        ntt = np.where(net > 0.0, 3.0 * (T_ION * KEV) * T_ION / net, np.inf)
    k = int(np.argmin(ntt))
    return float(T_ION[k]), float(ntt[k])


def build():
    sig = cross_sections()
    rows = []
    for name, col, a1, a2, q_tot, e_n, formula in FUELS:
        mr = a1 * a2 / (a1 + a2)
        sv = reactivity(sig[name], mr)
        k = int(np.argmax(sig[name]))
        j = int(np.argmax(sv / T_ION ** 2))
        t_opt, ntt_opt = best_temperature(sv, q_tot - e_n)
        rows.append(dict(name=name, color=col, sigma=sig[name], sv=sv,
                         e_peak=float(E_CM[k]), s_peak=float(sig[name][k]),
                         q=q_tot, e_n=e_n, e_c=q_tot - e_n, formula=formula,
                         t_opt=t_opt, ntt_opt=ntt_opt,
                         t_eff=float(T_ION[j])))
    return rows


# -------------------------------------------------------------------- panels
def edge_labels(ax, x, items, min_decades):
    """Right-margin curve labels pushed apart in log space so none collide."""
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] * 10.0 ** min_decades)
    for y, (_y0, text, col) in zip(ys, items):
        ax.text(x, y, text, ha="left", va="center", fontsize=6.3, color=col,
                clip_on=False)


def sigma_panel(ax, rows):
    for r in rows:
        ax.plot(E_CM, r["sigma"], color=r["color"], lw=1.7, zorder=5)
    dt = rows[0]
    ax.plot([dt["e_peak"]], [dt["s_peak"]], marker="v", ms=4.8, color=RED,
            zorder=8)
    ax.annotate("D-T resonance: {:.0f} mb at E$_{{cm}}$ = {:.0f} keV,\n"
                "{:.0f}$\\times$ the D-D cross section there".format(
                    dt["s_peak"], dt["e_peak"],
                    dt["s_peak"] / float(np.interp(dt["e_peak"], E_CM,
                                                   rows[1]["sigma"]))),
                xy=(dt["e_peak"], dt["s_peak"]), xytext=(2.0, 2.1e3),
                ha="left", va="top", fontsize=6.4, color=RED,
                linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    edge_labels(ax, E_VIEW * 1.08,
                [(float(np.interp(E_VIEW, E_CM, r["sigma"])), r["name"],
                  r["color"]) for r in rows], 0.62)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, E_VIEW)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("centre-of-mass energy E$_{cm}$ (keV)", fontsize=7.6)
    ax.set_ylabel("cross section $\\sigma$ (mb;  1 barn = 1000 mb)",
                  fontsize=7.2)
    ax.tick_params(labelsize=6.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    ax.set_title("Fusion cross sections: D-T dominates at reactor energies",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)


def sv_panel(ax, rows):
    for r in rows:
        ax.plot(T_ION, r["sv"], color=r["color"], lw=1.7, zorder=5)
    edge_labels(ax, 545.0, [(float(r["sv"][-1]), r["name"], r["color"])
                            for r in rows], 0.52)
    dt = rows[0]
    sv10 = float(np.interp(10.0, T_ION, dt["sv"]))
    ax.plot([10.0], [sv10], marker="o", ms=4.4, mfc="white", mec=RED,
            mew=1.2, zorder=8)
    ax.annotate("$\\langle\\sigma v\\rangle_{{DT}}$ = {:.2f}$\\times$"
                "10$^{{-22}}$ $m^3 s^{{-1}}$\nat T = 10 keV".format(
                    sv10 * 1e22), xy=(10.0, sv10), xytext=(1.35, 6e-23),
                ha="left", va="top", fontsize=6.3, color=RED,
                linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.axvspan(8.0, 25.0, color=lighten(GOLD, 0.86), lw=0.0, zorder=1)
    ax.text(14.0, 4e-27, "reactor-relevant\nT$_i$ window", ha="center",
            va="bottom", fontsize=6.1, color="#8a6d10", linespacing=1.25,
            zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 500.0)
    ax.set_ylim(1e-27, 2e-21)
    ax.set_xlabel("ion temperature T$_i$ (keV)", fontsize=7.6)
    ax.set_ylabel("$\\langle\\sigma v\\rangle$ ($m^3 s^{-1}$)", fontsize=7.2)
    ax.tick_params(labelsize=6.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    ax.set_title("The same four fuels seen through a Maxwellian",
                 fontsize=8.6, fontweight="bold", color=INK, pad=4.0)
    return sv10


def optimum_panel(ax, rows):
    pos = np.arange(len(rows), dtype=float)
    vals = [r["t_opt"] for r in rows]
    ax.barh(pos, vals, height=0.62, color=[r["color"] for r in rows],
            edgecolor="white", lw=0.6, zorder=4)
    for p, r in zip(pos, rows):
        ax.text(r["t_opt"] * 1.10, p, "{:.0f} keV".format(r["t_opt"]),
                ha="left", va="center", fontsize=6.2, color=SUB, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(3.0, 900.0)
    ax.set_ylim(-0.62, len(rows) - 0.30)
    ax.set_yticks(pos)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("temperature of cheapest ignition (keV)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax.set_title("Where each fuel is easiest to ignite", fontsize=8.2,
                 fontweight="bold", color=INK)


def energy_panel(ax, rows):
    pos = np.arange(len(rows), dtype=float)
    neut = np.array([r["e_n"] / 1e3 for r in rows])
    chg = np.array([r["e_c"] / 1e3 for r in rows])
    ax.barh(pos, chg, height=0.62, color=lighten(RED, 0.30),
            edgecolor="white", lw=0.6, zorder=4, label="charged products")
    ax.barh(pos, neut, height=0.62, left=chg, color="#9a9a9a",
            edgecolor="white", lw=0.6, zorder=4, label="neutrons")
    for p, r, c, n in zip(pos, rows, chg, neut):
        ax.text(c + n + 0.45, p, "{:.1f} MeV,  {:.0f} % charged".format(
            r["q"] / 1e3, 100.0 * r["e_c"] / r["q"]), ha="left", va="center",
            fontsize=6.1, color=SUB, zorder=6)
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(-0.62, len(rows) + 0.22)
    ax.set_yticks(pos)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("energy released per reaction (MeV)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax.legend(loc="lower right", fontsize=6.2, frameon=False,
              bbox_to_anchor=(0.995, 0.02), handlelength=1.5,
              labelspacing=0.26, borderaxespad=0.0)
    ax.set_title("Neutrons heat a blanket, charged products heat the plasma",
                 fontsize=7.8, fontweight="bold", color=INK)


def main():
    set_pub_style()
    np.random.default_rng(923)
    rows = build()

    fig = plt.figure(figsize=(FW, FH))
    ax_s = axes_inches(fig, TOPM)
    ax_v = axes_inches(fig, BOTM)
    ax_o = axes_inches(fig, AUX1)
    ax_e = axes_inches(fig, AUX2)
    sigma_panel(ax_s, rows)
    sv10 = sv_panel(ax_v, rows)
    optimum_panel(ax_o, rows)
    energy_panel(ax_e, rows)

    dt, dd, dhe, pb = rows
    panel_label_inches(fig, ax_s, "(a)", dx=-0.66, dy=0.13)
    panel_label_inches(fig, ax_v, "(b)", dx=-0.66, dy=0.13)
    panel_label_inches(fig, ax_o, "(c)", dx=-0.62, dy=0.13)
    panel_label_inches(fig, ax_e, "(d)", dx=-0.62, dy=0.13)
    savefig(fig, "923_fus_reaction_crosssection.png")


if __name__ == "__main__":
    main()
