"""Water-electrolyser efficiency and specific energy against current density.

(a) Faradaic and voltage efficiency plotted separately with their product, the
    total efficiency, drawn as the emphasised curve; hydrogen output sits on a
    second axis.  Crossover kills the Faradaic term at low load and cell
    overpotential kills the voltage term at high load, so the product has an
    interior optimum that the figure locates from the curve itself.
(b) Auxiliary panel: specific energy consumption in kWh per kg of hydrogen on
    the same current-density scale, with the thermoneutral floor and the
    best operating point.
(c) Auxiliary panel: the same metric for four electrolyser technologies.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

RNG = np.random.default_rng(70625)

E_REV = 1.23           # reversible cell potential, V
E_TN = 1.48            # thermoneutral potential, V
TAFEL = 0.060          # V per decade
J0 = 1.0e-5            # A cm-2
ASR = 0.22             # ohm cm2
J_X = 0.020            # crossover current density, A cm-2
FE_MAX = 0.995
FARADAY = 96485.0      # C mol-1
M_H2 = 2.016e-3        # kg mol-1

E_SCALE = (0.0, 84.0)  # one kWh/kg ruler shared by panels (b) and (c)

FARAD = "#2e7d5b"      # house green: Faradaic efficiency
VOLTEF = "#c9a227"     # house gold: voltage efficiency
TOTAL = "#3b5b92"      # house blue: the emphasised total efficiency
RATE = "#8e6bbf"       # house purple: hydrogen output
LIMIT = "#b23a48"      # house red: thermodynamic floor

TECHS = [("Alkaline", 1.10, FARAD), ("PEM", 1.00, TOTAL),
         ("AEM", 1.05, VOLTEF), ("SOEC", 0.83, "#8e6bbf")]


def cell_voltage(j):
    return E_REV + TAFEL * np.log10(j / J0) + ASR * j


def faradaic(j):
    """Crossover-limited at low load, bubble-limited at high load."""
    return FE_MAX * (j / (j + J_X)) * (1.0 - 0.010 * j)


def specific_energy(j):
    """kWh per kg of hydrogen produced."""
    return cell_voltage(j) * 2.0 * FARADAY / (faradaic(j) * M_H2 * 3.6e6)


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axe = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axt = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    j = np.logspace(np.log10(0.02), np.log10(3.0), 500)
    fe = faradaic(j)
    ve = E_REV / cell_voltage(j)
    tot = fe * ve
    # mol s-1 cm-2 -> mL min-1 cm-2 at STP
    rate = fe * j / (2.0 * FARADAY) * 22414.0 * 60.0
    k_opt = int(np.argmax(tot))

    # ---------------- (a) efficiency terms --------------------------------
    ax.axvline(j[k_opt], color=TOTAL, lw=0.8, ls=(0, (2.6, 2.2)), zorder=2)
    ax.plot(j, fe * 100.0, color=FARAD, lw=1.35, ls=(0, (4.2, 2.0)), zorder=4,
            label="Faradaic efficiency $\\eta_F$")
    ax.plot(j, ve * 100.0, color=VOLTEF, lw=1.35, ls=(0, (1.6, 1.8)),
            zorder=4, label="voltage efficiency $\\eta_V$")
    ax.plot(j, tot * 100.0, color=TOTAL, lw=2.1, zorder=6,
            label="total efficiency $\\eta_F \\times \\eta_V$")
    ax.plot([j[k_opt]], [tot[k_opt] * 100.0], marker="o", ms=5.6, mfc="white",
            mec=TOTAL, mew=1.3, zorder=7)
    ax.annotate("optimum %.1f %% at %.2f A cm$^{-2}$"
                % (tot[k_opt] * 100.0, j[k_opt]),
                xy=(j[k_opt], tot[k_opt] * 100.0), xytext=(0.048, 41.0),
                fontsize=7.0, color=TOTAL, ha="left", va="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=TOTAL, lw=0.9,
                                mutation_scale=8, shrinkB=4.0))
    ax.text(0.031, 92.0, "H$_2$ crossover\ndominates", fontsize=6.8,
            color=FARAD, ha="left", va="top", linespacing=1.4)
    ax.text(2.10, 91.0, "cell overpotential\ndominates", fontsize=6.8,
            color=VOLTEF, ha="right", va="top", linespacing=1.4)

    axr = ax.twinx()
    axr.plot(j, rate, color=RATE, lw=1.3, ls=(0, (5.0, 2.0)), zorder=5)
    axr.text(0.33, rate.max() * 0.31, "H$_2$ output", fontsize=6.8,
             color=RATE, ha="left", va="center")
    axr.set_ylim(0.0, rate.max() * 1.12)
    axr.set_ylabel("H$_2$ output (mL min$^{-1}$ cm$^{-2}$)", color=RATE)
    axr.tick_params(axis="y", labelsize=7.2, colors=RATE)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(RATE)
    axr.spines["top"].set_visible(False)
    axr.set_xscale("log")

    ax.set_xscale("log")
    ax.set_xlim(0.02, 3.0)
    ax.set_ylim(30.0, 102.0)
    ax.set_xlabel("Current density (A cm$^{-2}$)")
    ax.set_ylabel("Efficiency (%)")
    ax.tick_params(labelsize=7.2, which="both")
    ax.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
              fontsize=7.0, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.0)

    # ---------------- (b) specific energy ---------------------------------
    se = specific_energy(j)
    k_min = int(np.argmin(se))
    se_floor = E_TN * 2.0 * FARADAY / (M_H2 * 3.6e6)
    axe.plot(j, se, color=TOTAL, lw=1.6, zorder=4)
    axe.axhline(se_floor, color=LIMIT, lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    axe.text(0.021, se_floor + 0.8, "thermoneutral floor, %.1f" % se_floor,
             fontsize=6.3, color=LIMIT, ha="left", va="bottom")
    axe.plot([j[k_min]], [se[k_min]], marker="o", ms=4.6, mfc="white",
             mec=TOTAL, mew=1.1, zorder=6)
    axe.annotate("best %.1f kWh kg$^{-1}$\nat %.2f A cm$^{-2}$"
                 % (se[k_min], j[k_min]),
                 xy=(j[k_min], se[k_min]), xytext=(0.30, 58.0), fontsize=6.4,
                 color=TOTAL, ha="left", va="center", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", color=TOTAL, lw=0.8,
                                 mutation_scale=7, shrinkB=3.0))
    axe.set_xscale("log")
    axe.set_xlim(0.02, 3.0)
    axe.set_ylim(*E_SCALE)
    axe.set_xlabel("Current density (A cm$^{-2}$)", fontsize=8, labelpad=1.5)
    axe.set_ylabel("Energy (kWh kg$^{-1}$ H$_2$)", fontsize=8, labelpad=1.5)
    axe.tick_params(labelsize=6.5, which="both")
    axe.grid(color="#ededed", lw=0.5, zorder=0)

    # ---------------- (c) technology comparison ---------------------------
    ref = specific_energy(np.array([1.0]))[0]
    vals = np.array([ref * f for _, f, _ in TECHS])
    cols = [c for _, _, c in TECHS]
    names = [n for n, _, _ in TECHS]
    order = np.argsort(vals)
    axt.bar(np.arange(len(TECHS)), vals[order], width=0.62,
            color=[cols[i] for i in order], edgecolor="white", linewidth=0.6,
            zorder=3)
    for x, i in enumerate(order):
        axt.text(x, vals[i] + 0.7, "%.1f" % vals[i], fontsize=6.4,
                 color="#555555", ha="center", va="bottom", zorder=4)
    axt.axhline(se_floor, color=LIMIT, lw=0.9, ls=(0, (3.4, 2.2)), zorder=5)
    axt.text(-0.42, 63.0, "dashed red: thermoneutral floor, %.1f" % se_floor,
             fontsize=6.3, color=LIMIT, ha="left", va="bottom")
    axt.set_xticks(np.arange(len(TECHS)))
    axt.set_xticklabels([names[i] for i in order], fontsize=6.8)
    axt.set_ylim(*E_SCALE)         # same energy ruler as panel (b)
    axt.set_ylabel("Energy (kWh kg$^{-1}$ H$_2$)", fontsize=8, labelpad=1.5)
    axt.set_xlabel("Technology at 1.0 A cm$^{-2}$", fontsize=8, labelpad=1.5)
    axt.tick_params(axis="y", labelsize=6.5)
    axt.tick_params(axis="x", length=0)
    axt.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    panel_labels_inches(fig, [ax, axe, axt], dx=-0.34, dy=0.10)

    savefig(fig, "706_ech_electrolyzer_efficiency.png")


if __name__ == "__main__":
    main()
