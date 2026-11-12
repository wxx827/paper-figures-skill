"""Gelation kinetics: the moduli crossover as a thermometer and a clock.

The main panel is the claim: storage and loss moduli followed through an
isothermal cure at three temperatures, where the crossover of the two curves
marks the gel point.  Each gel time printed on the figure is interpolated from
the plotted arrays, not quoted.  The two smaller panels show what those three
crossovers are worth: an Arrhenius plot over six cure temperatures whose slope
returns the activation energy that generated the kinetics in the first place,
and a frequency sweep taken at the gel point, where both moduli follow the
same power law and the fitted exponent is the Winter-Chambon relaxation
exponent.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

BLUE = "#3b5b92"        # coolest cure
GREEN = "#2e7d5b"       # reference cure, also the frequency sweep sample
RED = "#b23a48"         # hottest cure
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

R_GAS = 8.314           # J/(mol K)
EA_TRUE = 6.50e4        # J/mol, activation energy of the cure
K0 = 6.7e7              # 1/s, Arrhenius prefactor
P_GEL = 0.60            # extent of reaction at the chemical gel point
A_EL = 1.20e4           # Pa, elastic prefactor
B_VI = 4.00e2           # Pa, viscous prefactor
FLOOR = 0.09            # Pa, instrument noise floor
OMEGA = 6.28            # rad/s, frequency of the cure sweep

SHOWN = {45.0: BLUE, 55.0: GREEN, 65.0: RED}
ALL_T = [40.0, 45.0, 50.0, 55.0, 60.0, 65.0]


def rate(t_c):
    return K0 * np.exp(-EA_TRUE / (R_GAS * (t_c + 273.15)))


def cure(t, t_c, rng):
    p = 1.0 - np.exp(-rate(t_c) * t)
    x = p / P_GEL
    # Pre-gel clusters already carry a little elasticity; the divergent term
    # only switches on once the network percolates.
    gp = A_EL * np.maximum(x - 1.0, 0.0) ** 2.3 + 6.0 * x ** 8 + FLOOR
    gpp = B_VI * x ** 1.6 / (1.0 + 0.6 * np.maximum(x - 1.0, 0.0) ** 2) + FLOOR
    gp = gp * (1.0 + rng.normal(0.0, 0.02, t.size))
    gpp = gpp * (1.0 + rng.normal(0.0, 0.02, t.size))
    return gp, gpp


def crossover(x, a, b):
    """Where two positive curves meet, by log-log interpolation."""
    d = np.log10(a) - np.log10(b)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    i = int(idx[-1])
    f = d[i] / (d[i] - d[i + 1])
    lx = np.log10(x[i]) + f * (np.log10(x[i + 1]) - np.log10(x[i]))
    ly = np.log10(a[i]) + f * (np.log10(a[i + 1]) - np.log10(a[i]))
    return 10.0 ** lx, 10.0 ** ly


def cures(seed=137):
    rng = np.random.default_rng(seed)
    t = np.logspace(1.0, 4.3, 340)
    out = {}
    for t_c, color in SHOWN.items():
        gp, gpp = cure(t, t_c, rng)
        tg, gg = crossover(t, gp, gpp)
        out[t_c] = dict(t=t, gp=gp, gpp=gpp, tg=tg, gg=gg, color=color)
    return out


def arrhenius(seed=139):
    rng = np.random.default_rng(seed)
    t = np.logspace(1.0, 4.6, 600)
    temp, tgel = [], []
    for t_c in ALL_T:
        gp, gpp = cure(t, t_c, rng)
        tg, _ = crossover(t, gp, gpp)
        temp.append(t_c + 273.15)
        tgel.append(tg)
    temp = np.array(temp)
    tgel = np.array(tgel)
    inv = 1000.0 / temp
    slope, icpt = np.polyfit(inv, np.log(tgel), 1)
    return inv, tgel, float(slope), float(icpt), \
        float(slope * 1000.0 * R_GAS / 1000.0)


def gel_point_sweep(seed=149, n_exp=0.52):
    rng = np.random.default_rng(seed)
    w = np.logspace(-1.0, 2.0, 120)
    s = 260.0
    gp = s * w ** n_exp * (1.0 + rng.normal(0.0, 0.025, w.size))
    gpp = gp * np.tan(n_exp * np.pi / 2.0) / (1.0 +
                                              rng.normal(0.0, 0.0, w.size))
    gpp = gpp * (1.0 + rng.normal(0.0, 0.025, w.size))
    n_fit = float(np.polyfit(np.log10(w), np.log10(gp), 1)[0])
    tand = float(np.median(gpp / gp))
    return w, gp, gpp, n_fit, tand


def main_panel(ax, data):
    for t_c, d in data.items():
        ax.loglog(d["t"], d["gp"], color=d["color"], lw=1.9, zorder=5,
                  label=f"{t_c:.0f}" + r"$^\circ$C")
        ax.loglog(d["t"], d["gpp"], color=d["color"], lw=1.3, zorder=4,
                  ls=(0, (4.0, 2.2)))

    for t_c, d in data.items():
        ax.plot([d["tg"]], [d["gg"]], marker="o", ms=6.4, mfc="white",
                mec=d["color"], mew=1.5, zorder=7)

    ax.text(0.035, 0.975, r"solid: $G'$    dashed: $G''$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=MUTED)
    ax.text(0.035, 0.905, r"open circle: gel point, $G' = G''$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB)
    for i, (t_c, d) in enumerate(data.items()):
        ax.text(0.035, 0.828 - 0.058 * i,
                f"{t_c:.0f}" + r"$^\circ$C:  " + f"{d['tg']:.0f} s",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                color=d["color"])
    ax.set_xlim(3e1, 2.0e4)
    ax.set_ylim(5e-2, 3e4)
    ax.set_xlabel("cure time $t$ (s)")
    ax.set_ylabel(r"$G'$, $G''$ (Pa)")
    ax.tick_params(which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right", ncol=3,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.4,
              title="cure temperature", title_fontsize=7.0)
    ax.set_title(f"Isothermal cure followed at {OMEGA:.2f} rad/s, 1 % strain",
                 fontsize=9.0, fontweight="bold", color=INK)


def arrhenius_panel(ax, inv, tgel, slope, icpt, ea_fit):
    grid = np.linspace(inv.min() * 0.995, inv.max() * 1.005, 40)
    ax.semilogy(grid, np.exp(icpt + slope * grid), color=GREY, lw=1.2,
                zorder=3)
    ax.semilogy(inv, tgel, ls="none", marker="o", ms=4.0, mfc="white",
                mec=INK, mew=0.9, zorder=4)
    for t_c, color in SHOWN.items():
        i = ALL_T.index(t_c)
        ax.semilogy([inv[i]], [tgel[i]], ls="none", marker="o", ms=5.0,
                    mfc=color, mec="white", mew=0.8, zorder=6)
    ax.text(0.045, 0.94,
            r"$E_a$ = " + f"{ea_fit:.0f} kJ/mol\n" +
            r"(model input " + f"{EA_TRUE / 1e3:.0f} kJ/mol)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, linespacing=1.4)
    ax.set_xlabel(r"$1000/T$ (K$^{-1}$)", fontsize=7.4)
    ax.set_ylabel(r"$t_{gel}$ (s)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.set_title(f"Gel time over {len(ALL_T)} cure temperatures",
                 fontsize=8.2, fontweight="bold", color=INK)


def winter_panel(ax, w, gp, gpp, n_fit, tand):
    ax.loglog(w, gp, color=GREEN, lw=1.6, zorder=5, label=r"$G'$")
    ax.loglog(w[::7], gpp[::7], ls="none", marker="o", ms=3.4, mfc="white",
              mec=GREEN, mew=0.9, zorder=6, label=r"$G''$")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right", ncol=2,
              handlelength=1.3, columnspacing=0.9, borderaxespad=0.3)
    ax.text(0.045, 0.90,
            r"$G',G'' \propto \omega^{n}$" + "\n" +
            r"$n$ = " + f"{n_fit:.2f}" + "\n" +
            r"$\tan\delta$ = " + f"{tand:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=INK, linespacing=1.5)
    ax.set_xlim(8e-2, 1.1e2)
    ax.set_ylim(4e1, 6e3)
    ax.set_xlabel(r"angular frequency $\omega$ (rad/s)", fontsize=7.4)
    ax.set_ylabel(r"$G'$, $G''$ (Pa)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.set_title(r"Frequency sweep at the gel point, 55$^\circ$C",
                 fontsize=8.2, fontweight="bold", color=INK)


def main():
    set_pub_style()
    data = cures()
    inv, tgel, slope, icpt, ea_fit = arrhenius()
    w, gp, gpp, n_fit, tand = gel_point_sweep()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_arr = fig.add_subplot(gs[0, 1])
    ax_win = fig.add_subplot(gs[1, 1])

    main_panel(ax_main, data)
    arrhenius_panel(ax_arr, inv, tgel, slope, icpt, ea_fit)
    winter_panel(ax_win, w, gp, gpp, n_fit, tand)

    hot = data[max(SHOWN)]["tg"]
    cold = data[min(SHOWN)]["tg"]
    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_arr, ax_win], dx=-0.42, dy=0.12)
    savefig(fig, "737_sof_gelation_kinetics.png")


if __name__ == "__main__":
    main()
