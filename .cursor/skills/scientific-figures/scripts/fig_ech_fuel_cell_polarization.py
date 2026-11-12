"""Polarisation and power density of a PEM fuel cell, loss region by loss region.

(a) Cell voltage against current density with the power density on a second
    axis.  The three classical regimes are shaded from the point where each
    loss term takes over the slope: activation while A/j exceeds the area
    resistance, ohmic in the linear middle, concentration once the limiting
    current is approached.  Both region edges and the peak power point are
    computed from the plotted curve, never typed in.
(b) Auxiliary panel: the same three overpotentials stacked under the reversible
    potential, so the voltage left for the load is the white gap on top.
(c) Auxiliary panel: polarisation under three operating conditions, sharing the
    voltage scale of panel (a).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

RNG = np.random.default_rng(70521)

E_REV = 1.23          # reversible cell potential, V
V_TOP = 1.30          # one voltage ruler for every panel of this figure
J0 = 1.0e-4           # exchange current density, A cm-2
A_TAFEL = 0.045       # Tafel prefactor, V
ASR = 0.12            # area specific resistance, ohm cm2
C_CONC = 0.045        # mass-transport prefactor, V
J_LIM = 1.90          # limiting current density, A cm-2

CELL = "#3b5b92"      # house blue: the cell voltage
POWER = "#8e6bbf"     # house purple: power density
ACT = "#2e7d5b"       # house green: activation loss
OHM = "#c9a227"       # house gold: ohmic loss
CONC = "#b23a48"      # house red: concentration loss

CONDS = [dict(name="80 $^\\circ$C, 2.5 bar", r=0.100, jl=2.15, a=0.042,
              col=ACT),
         dict(name="80 $^\\circ$C, 1.5 bar (baseline)", r=ASR, jl=J_LIM,
              a=A_TAFEL, col=CELL),
         dict(name="60 $^\\circ$C, 1.0 bar", r=0.170, jl=1.45, a=0.052,
              col=OHM)]


def losses(j, a=A_TAFEL, r=ASR, c=C_CONC, jl=J_LIM):
    """Activation, ohmic and concentration overpotentials in volts."""
    e_act = a * np.log(np.clip(j, 1e-6, None) / J0)
    e_ohm = r * j
    e_con = -c * np.log(np.clip(1.0 - j / jl, 1e-4, None))
    return e_act, e_ohm, e_con


def polarisation(j, **kw):
    return E_REV - sum(losses(j, **kw))


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axs = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    j = np.linspace(0.002, 0.995 * J_LIM, 700)
    v = polarisation(j)
    p = v * j
    e_act, e_ohm, e_con = losses(j)

    # Regime edges: where the slope contribution of one loss overtakes ohmic.
    j_act = A_TAFEL / ASR
    j_con = J_LIM - C_CONC / ASR
    k_pk = int(np.argmax(p))

    # ---------------- (a) polarisation + power ----------------------------
    bands = [(0.0, j_act, ACT, "activation"),
             (j_act, j_con, OHM, "ohmic"),
             (j_con, J_LIM, CONC, "transport")]
    for x0, x1, col, name in bands:
        ax.axvspan(x0, x1, color=col, alpha=0.085, lw=0, zorder=0)
        ax.axvline(x1, color=col, lw=0.7, ls=(0, (2.6, 2.2)), alpha=0.8,
                   zorder=1)
        ax.text(0.5 * (x0 + x1), 1.285, name, fontsize=7.2, color=col,
                ha="center", va="top", fontweight="bold")
        ax.text(0.5 * (x0 + x1), 1.195,
                "%.2f - %.2f A cm$^{-2}$" % (x0, x1), fontsize=6.2,
                color="#777777", ha="center", va="top")

    ax.plot(j, v, color=CELL, lw=2.0, zorder=6)
    ax.plot([j[0]], [v[0]], marker="o", ms=4.6, mfc="white", mec=CELL,
            mew=1.1, zorder=7)
    ax.text(j[0] + 0.03, v[0], "OCV = %.2f V" % v[0], fontsize=6.8,
            color=CELL, ha="left", va="center")

    axp = ax.twinx()
    axp.plot(j, p, color=POWER, lw=1.5, ls=(0, (4.0, 2.0)), zorder=5)
    axp.plot([j[k_pk]], [p[k_pk]], marker="D", ms=4.6, mfc=POWER,
             mec="white", mew=0.9, zorder=7)
    axp.annotate("peak %.2f W cm$^{-2}$\nat %.2f A cm$^{-2}$, %.2f V"
                 % (p[k_pk], j[k_pk], v[k_pk]),
                 xy=(j[k_pk], p[k_pk]), xytext=(j[k_pk] - 0.16, 0.40),
                 fontsize=6.8, color=POWER, ha="right", va="center",
                 linespacing=1.4,
                 arrowprops=dict(arrowstyle="-|>", color=POWER, lw=0.8,
                                 mutation_scale=8, shrinkB=3.0))
    axp.set_ylim(0.0, 1.05)
    axp.set_ylabel("Power density (W cm$^{-2}$)", color=POWER)
    axp.tick_params(axis="y", labelsize=7.2, colors=POWER)
    axp.spines["right"].set_visible(True)
    axp.spines["right"].set_color(POWER)
    axp.spines["top"].set_visible(False)

    k_mid = int(np.argmin(np.abs(j - 0.5 * (j_act + j_con))))
    ax.annotate("linear region, ASR = %.2f $\\Omega$ cm$^{2}$" % ASR,
                xy=(j[k_mid], v[k_mid]), xytext=(j[k_mid] - 0.05, 0.33),
                fontsize=6.8, color=OHM, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=OHM, lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    ax.axvline(J_LIM, color=CONC, lw=0.9, ls=(0, (1.4, 1.8)), zorder=2)
    ax.text(J_LIM - 0.032, 1.10, "$j_{lim}$", fontsize=7.0, color=CONC,
            ha="center", va="center", rotation=90, fontweight="bold")

    ax.set_xlim(0.0, J_LIM + 0.06)
    ax.set_ylim(0.0, V_TOP)
    ax.set_xlabel("Current density (A cm$^{-2}$)")
    ax.set_ylabel("Cell voltage (V)", color=CELL)
    ax.tick_params(axis="y", labelsize=7.2, colors=CELL)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.spines["left"].set_color(CELL)

    # ---------------- (b) loss breakdown ----------------------------------
    axs.stackplot(j, e_act, e_ohm, e_con, colors=[ACT, OHM, CONC],
                  alpha=0.85, edgecolor="white", linewidth=0.3, zorder=2,
                  labels=["$\\eta_{act}$", "$\\eta_{ohm}$", "$\\eta_{conc}$"])
    axs.plot(j, np.full_like(j, E_REV), color="#333333", lw=0.9, zorder=4)
    axs.text(J_LIM + 0.02, E_REV - 0.035, "$E_{rev}$ = %.2f V" % E_REV,
             fontsize=6.4, color="#333333", ha="right", va="top")
    axs.text(0.50, 0.72, "useful cell voltage", transform=axs.transAxes,
             fontsize=6.4, color=CELL, ha="center", va="center")
    axs.set_xlim(0.0, J_LIM + 0.06)
    axs.set_ylim(0.0, V_TOP)
    axs.set_xlabel("Current density (A cm$^{-2}$)", fontsize=8, labelpad=1.5)
    axs.set_ylabel("Overpotential (V)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5)
    axs.legend(loc="upper left", bbox_to_anchor=(0.015, 0.88), fontsize=6.4,
               ncol=3, handlelength=1.1, columnspacing=0.9,
               borderaxespad=0.0, handletextpad=0.4)

    # ---------------- (c) operating conditions ----------------------------
    vs = []
    for cond in CONDS:
        jc = np.linspace(0.002, 0.995 * cond["jl"], 500)
        vc = polarisation(jc, a=cond["a"], r=cond["r"], jl=cond["jl"])
        vs.append(vc)
        pk = int(np.argmax(vc * jc))
        lw = 1.8 if cond["col"] == CELL else 1.2
        axc.plot(jc, vc, color=cond["col"], lw=lw, zorder=4,
                 label="%s, %.2f W cm$^{-2}$" % (cond["name"], (vc * jc)[pk]))
        axc.plot([jc[pk]], [vc[pk]], marker="D", ms=3.4, mfc=cond["col"],
                 mec="white", mew=0.6, zorder=5)
    axc.set_xlim(0.0, 2.25)
    axc.set_ylim(0.0, V_TOP)      # same voltage ruler as panels (a) and (b)
    axc.set_xlabel("Current density (A cm$^{-2}$)", fontsize=8, labelpad=1.5)
    axc.set_ylabel("Cell voltage (V)", fontsize=8, labelpad=1.5)
    axc.tick_params(labelsize=6.5)
    axc.grid(color="#ededed", lw=0.5, zorder=0)
    axc.legend(loc="lower left", fontsize=6.3, handlelength=1.3,
               labelspacing=0.3, borderaxespad=0.35)
    axc.text(0.975, 0.955, "diamonds mark peak power",
             transform=axc.transAxes, fontsize=6.2, color="#777777",
             ha="right", va="top")

    panel_labels_inches(fig, [ax, axs, axc], dx=-0.34, dy=0.10)

    savefig(fig, "705_ech_fuel_cell_polarization.png")


if __name__ == "__main__":
    main()
