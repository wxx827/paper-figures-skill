"""Polymer scaling laws: coil size, zero-shear viscosity, relaxation spectrum.

The main panel is the claim: radius of gyration against molecular weight for
one polymer measured in three solvents, where the only thing that changes is
the slope on the log-log plot.  Each Flory exponent printed on the figure is
the slope of a least-squares fit to the plotted points, shown next to the
theoretical value it is meant to reproduce, so a reader can see the agreement
rather than take it on trust.  The two smaller panels carry the dynamic
consequences: the zero-shear viscosity, whose slope changes from 1 to 3.4 at
a critical molecular weight recovered from the intersection of two fits, and
the relaxation spectrum, where entanglement adds a terminal peak that the
unentangled chain simply does not have.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

BLUE = "#3b5b92"        # good solvent / entangled melt
GREY = "#808080"        # theta solvent / unentangled melt
RED = "#b23a48"         # poor solvent, the collapsed globule
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

SOLVENTS = {
    "Good solvent": dict(color=BLUE, nu=0.588, rg1e6=62.0),
    r"$\theta$ solvent": dict(color=GREY, nu=0.500, rg1e6=41.0),
    "Poor solvent": dict(color=RED, nu=1.0 / 3.0, rg1e6=19.0),
}

M_REF = 1.0e6           # g/mol, the anchor of every prefactor
MC_TRUE = 3.2e4         # g/mol, critical molecular weight of the melt
ETA_C = 4.5e-4          # Pa s per (g/mol), unentangled prefactor


def coil_sizes(seed=101, n=11):
    rng = np.random.default_rng(seed)
    mw = np.logspace(4.0, 7.0, n)
    out = {}
    for name, s in SOLVENTS.items():
        pref = s["rg1e6"] / M_REF ** s["nu"]
        rg = pref * mw ** s["nu"] * (1.0 + rng.normal(0.0, 0.055, n))
        slope, icpt = np.polyfit(np.log10(mw), np.log10(rg), 1)
        out[name] = dict(mw=mw, rg=rg, color=s["color"], nu_fit=float(slope),
                         nu_th=s["nu"], icpt=float(icpt))
    return out


def zero_shear(seed=113, n=15):
    rng = np.random.default_rng(seed)
    mw = np.logspace(3.4, 6.4, n)
    eta = np.where(mw < MC_TRUE, ETA_C * mw,
                   ETA_C * MC_TRUE ** (-2.4) * mw ** 3.4)
    eta = eta * (1.0 + rng.normal(0.0, 0.07, n))
    lo, hi = mw < 0.6 * MC_TRUE, mw > 1.8 * MC_TRUE
    a1, b1 = np.polyfit(np.log10(mw[lo]), np.log10(eta[lo]), 1)
    a2, b2 = np.polyfit(np.log10(mw[hi]), np.log10(eta[hi]), 1)
    mc = 10.0 ** ((b2 - b1) / (a1 - a2))
    return dict(mw=mw, eta=eta, a1=float(a1), b1=float(b1), a2=float(a2),
                b2=float(b2), mc=float(mc))


def spectra(seed=127):
    tau = np.logspace(-6.0, 2.0, 320)
    tau_e, tau_d = 3.0e-4, 6.0
    rouse = 9.0e3 * (tau / tau_e) ** -0.5
    ent = rouse * np.exp(-tau / (2.2 * tau_e)) + 2.4e4 * np.exp(
        -0.5 * (np.log10(tau / tau_d) / 0.34) ** 2)
    une = rouse * np.exp(-tau / (2.0e-3))
    rng = np.random.default_rng(seed)
    ent = ent * (1.0 + rng.normal(0.0, 0.02, tau.size))
    une = une * (1.0 + rng.normal(0.0, 0.02, tau.size))
    peak = float(tau[int(np.argmax(ent * (tau > 0.1)))])
    return tau, dict(Entangled=(ent, BLUE), Unentangled=(une, GREY)), \
        tau_e, peak


def main_panel(ax, coils):
    grid = np.logspace(3.85, 7.15, 60)
    for name, d in coils.items():
        fit = 10.0 ** (d["icpt"] + d["nu_fit"] * np.log10(grid))
        ax.loglog(grid, fit, color=d["color"], lw=1.5, zorder=3)
        ax.loglog(d["mw"], d["rg"], ls="none", marker="o", ms=4.6,
                  mfc="white", mec=d["color"], mew=1.2, zorder=5,
                  label=name + r",  $\nu$ = " + f"{d['nu_fit']:.3f}  "
                  f"(theory {d['nu_th']:.3f})")

    ax.text(1.4e4, 78.0, r"$R_g \propto M_w^{\nu}$", fontsize=8.6,
            color=INK, ha="left", va="center")
    ax.set_xlim(7e3, 1.05e7)
    ax.set_ylim(2.0, 130.0)
    ax.set_xlabel(r"weight-average molar mass $M_w$ (g/mol)")
    ax.set_ylabel(r"radius of gyration $R_g$ (nm)")
    ax.tick_params(which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.4)
    ax.set_title("Static light scattering on one polymer in three solvents",
                 fontsize=9.0, fontweight="bold", color=INK)


def viscosity_panel(ax, zs):
    grid_lo = np.logspace(3.3, np.log10(zs["mc"]), 40)
    grid_hi = np.logspace(np.log10(zs["mc"]), 6.5, 40)
    ax.loglog(grid_lo, 10.0 ** (zs["b1"] + zs["a1"] * np.log10(grid_lo)),
              color=GREY, lw=1.3, zorder=3)
    ax.loglog(grid_hi, 10.0 ** (zs["b2"] + zs["a2"] * np.log10(grid_hi)),
              color=BLUE, lw=1.5, zorder=3)
    ax.loglog(zs["mw"], zs["eta"], ls="none", marker="o", ms=3.6,
              mfc="white", mec=INK, mew=0.9, zorder=5)
    ax.axvline(zs["mc"], color=RED, lw=1.0, ls=(0, (3.0, 2.2)), zorder=2)
    ax.annotate(r"$M_c$ = " + f"{zs['mc'] / 1e3:.0f} kg/mol\n" +
                r"$M_e$ = " + f"{zs['mc'] / 2e3:.0f} kg/mol",
                xy=(zs["mc"], 6e3), xytext=(2.4e3, 1.6e5),
                ha="left", va="center", fontsize=6.3, color=RED,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    x1 = 7.0e3
    y1 = 10.0 ** (zs["b1"] + zs["a1"] * np.log10(x1))
    ax.text(x1, 0.22 * y1, f"slope {zs['a1']:.2f}", fontsize=6.3, color=SUB,
            rotation=12, ha="center", va="top")
    x2 = 4.0e5
    y2 = 10.0 ** (zs["b2"] + zs["a2"] * np.log10(x2))
    ax.text(x2, 0.13 * y2, f"slope {zs['a2']:.2f}", fontsize=6.3, color=BLUE,
            rotation=40, ha="center", va="top")
    ax.set_xlim(1.6e3, 6.5e6)
    ax.set_ylim(3e-3, 2e6)
    ax.set_xlabel(r"$M_w$ (g/mol)", fontsize=7.4)
    ax.set_ylabel(r"$\eta_0$ (Pa$\cdot$s)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.set_title("Zero-shear viscosity of the melt", fontsize=8.2,
                 fontweight="bold", color=INK)


def spectrum_panel(ax, tau, curves, tau_e, peak):
    for name, (h, color) in curves.items():
        ax.loglog(tau, h, color=color, lw=1.4, zorder=4, label=name)
    ax.axvline(tau_e, color="#808080", lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(tau_e * 1.4, 6.0e1, r"$\tau_e$ = " + f"{tau_e * 1e3:.1f} ms",
            fontsize=6.3, color=SUB, ha="left", va="bottom")
    ax.axvline(peak, color=BLUE, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(peak * 0.75, 6.0e1, r"$\tau_d$ = " + f"{peak:.1f} s",
            fontsize=6.3, color=BLUE, ha="right", va="bottom")
    ax.set_xlim(tau[0], tau[-1])
    ax.set_ylim(1e0, 2e6)
    ax.set_xlabel(r"relaxation time $\tau$ (s)", fontsize=7.4)
    ax.set_ylabel(r"spectrum $H(\tau)$ (Pa)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right",
              handlelength=1.3, borderaxespad=0.2)
    ax.set_title("Relaxation spectrum", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    coils = coil_sizes()
    zs = zero_shear()
    tau, curves, tau_e, peak = spectra()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.46, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_eta = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, 1])

    main_panel(ax_main, coils)
    viscosity_panel(ax_eta, zs)
    spectrum_panel(ax_spec, tau, curves, tau_e, peak)

    good = coils["Good solvent"]
    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_eta, ax_spec], dx=-0.42, dy=0.12)
    savefig(fig, "736_sof_polymer_scaling.png")


if __name__ == "__main__":
    main()
