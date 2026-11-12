"""Long-term cycling of three lithium-ion cell builds.

(a) Main panel: capacity retention against cycle number for three builds,
    with coulombic efficiency of each build on the right axis.  The cycle
    life is read where each retention curve crosses 80 per cent, marked on
    the curve and dropped to the axis.
(b) Auxiliary panel: charge and discharge voltage curves of the best build at
    three representative cycles, showing the plateau shortening and the
    polarisation opening up as the cell ages.
(c) Auxiliary panel: rate capability as a staircase of C-rates, ending with a
    return to the starting rate so the recovered capacity is visible.

Panels (b) and (c) share one specific-capacity scale.  Cycle lives,
retentions and recoveries are all read back from the plotted arrays.
Illustrative values.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, shared_limits

RNG = np.random.default_rng(76701)

N_CYCLE = 800
Q_NOMINAL = 185.0        # first-discharge specific capacity, mAh g-1
EOL = 0.80               # end-of-life retention
SHOW = (1, 300, 700)     # cycles drawn in panel (b)
RATES = [0.2, 0.5, 1.0, 2.0, 5.0, 0.2]     # C-rate staircase
PER_RATE = 5             # cycles held at each rate

EOL_COL = "#b23a48"      # house red: the single end-of-life criterion

CELLS = [dict(name="baseline", sei=0.0100, lam=3.0e-4, ce=0.9955,
              col="#c9a227"),
         dict(name="coated cathode", sei=0.0072, lam=1.8e-4, ce=0.9975,
              col="#2e7d5b"),
         dict(name="coated + additive", sei=0.0050, lam=1.2e-4, ce=0.9990,
              col="#3b5b92")]


def retention(n, sei, lam):
    """Square-root SEI growth plus a linear loss of active material."""
    return np.exp(-sei * np.sqrt(n) - lam * n)


def ocv(x):
    """Open-circuit voltage of a layered-oxide cathode against state of charge."""
    return 3.80 - 0.25 * np.tanh((x - 0.5) * 3.0) - 0.35 * x ** 8


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.90, 1.02, 5.24, 4.62))
    axs = axes_inches(fig, (7.42, 3.74, 2.66, 1.90))
    axb = axes_inches(fig, (7.42, 1.02, 2.66, 1.90))
    axr = ax.twinx()
    axr.spines["top"].set_visible(False)

    cyc = np.arange(1, N_CYCLE + 1)
    for c in CELLS:
        ret = retention(cyc, c["sei"], c["lam"])
        c["ret"] = 100.0 * ret * (1.0 + RNG.normal(0.0, 0.0016, cyc.size))
        c["life"] = float(np.interp(100.0 * EOL, c["ret"][::-1], cyc[::-1]))
        c["eff"] = 100.0 * (c["ce"] - 0.035 * np.exp(-(cyc - 1) / 11.0)
                            + RNG.normal(0.0, 0.0004, cyc.size))

    # ---------------- (a) retention and efficiency ------------------------
    ax.axhline(100.0 * EOL, color=EOL_COL, lw=0.9, ls=(0, (3.0, 2.2)),
               zorder=2)
    ax.text(6.0, 100.0 * EOL + 0.7, "end of life, %.0f %% retention"
            % (100.0 * EOL), fontsize=6.6, color=EOL_COL, ha="left",
            va="bottom")
    for c in CELLS:
        ax.plot(cyc, c["ret"], color=c["col"], lw=1.6, zorder=5,
                label="%s,  %.0f cycles to %.0f %%"
                      % (c["name"], c["life"], 100.0 * EOL))
        ax.plot([c["life"], c["life"]], [55.0, 100.0 * EOL], color=c["col"],
                lw=0.7, ls=(0, (2.0, 2.0)), alpha=0.85, zorder=3)
        ax.plot([c["life"]], [100.0 * EOL], marker="o", ms=5.2, mfc="white",
                mec=c["col"], mew=1.2, zorder=8)
        axr.plot(cyc[::9], c["eff"][::9], ls="none", marker="o", ms=1.9,
                 mfc=c["col"], mec="none", alpha=0.75, zorder=4)

    gain = CELLS[-1]["life"] / CELLS[0]["life"]
    ax.annotate("%.1f$\\times$ the cycle life of the baseline" % gain,
                xy=(CELLS[-1]["life"], 100.0 * EOL),
                xytext=(CELLS[-1]["life"] - 40.0, 90.0), fontsize=6.8,
                color=CELLS[-1]["col"], ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color=CELLS[-1]["col"],
                                lw=0.8, mutation_scale=8, shrinkB=4.0))
    ax.text(0.985, 0.975, "small dots: coulombic efficiency, right axis",
            transform=ax.transAxes, fontsize=6.5, color="#777777",
            ha="right", va="top")

    ax.set_xlim(0.0, N_CYCLE + 8.0)
    ax.set_ylim(55.0, 103.0)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Capacity retention (%)")
    ax.tick_params(labelsize=7.2)
    ax.grid(color="#f2f2f2", lw=0.5, zorder=0)
    ax.legend(loc="lower left", fontsize=7.0, handlelength=1.4,
              labelspacing=0.32, borderaxespad=0.6)
    axr.set_ylim(95.8, 100.4)
    axr.set_ylabel("Coulombic efficiency (%)", fontsize=8, labelpad=2.5)
    axr.tick_params(axis="y", labelsize=7.0)

    # ---------------- (b) voltage curves ----------------------------------
    ref = CELLS[-1]
    x = np.linspace(0.002, 0.998, 320)
    for j, n in enumerate(SHOW):
        q_max = Q_NOMINAL * ref["ret"][n - 1] / 100.0
        pol = 0.028 + 6.5e-5 * n
        shade = 0.30 + 0.35 * j
        col = tuple(np.array(mpl.colors.to_rgb(ref["col"]))
                    * (1.0 - shade) + shade)
        axs.plot(q_max * x, ocv(x) - pol, color=col, lw=1.25, zorder=4,
                 label="cycle %d" % n)
        axs.plot(q_max * x, ocv(x) + pol, color=col, lw=1.25, ls=(0, (3.0,
                 2.0)), zorder=4)
        if j == len(SHOW) - 1:
            x_pol = 0.96
            axs.annotate("", xy=(q_max * x_pol, ocv(x_pol) + pol),
                         xytext=(q_max * x_pol, ocv(x_pol) - pol),
                         arrowprops=dict(arrowstyle="<->", color="#555555",
                                         lw=0.8, mutation_scale=6, shrinkA=0,
                                         shrinkB=0))
            axs.annotate("%.0f mV polarisation at cycle %d"
                         % (2000.0 * pol, n),
                         xy=(q_max * x_pol, ocv(x_pol)), xytext=(4.0, 3.10),
                         fontsize=6.2, color="#555555", ha="left",
                         va="bottom",
                         arrowprops=dict(arrowstyle="-|>", color="#555555",
                                         lw=0.7, mutation_scale=6,
                                         shrinkB=2.0))
    axs.set_ylim(3.05, 4.30)
    axs.set_xlabel("Specific capacity (mAh g$^{-1}$)", fontsize=8,
                   labelpad=1.5)
    axs.set_ylabel("Cell voltage (V)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5)
    axs.grid(color="#ededed", lw=0.5, zorder=0)
    axs.legend(loc="upper right", fontsize=6.3, handlelength=1.2,
               labelspacing=0.26, borderaxespad=0.3,
               title="%s, solid: discharge" % ref["name"],
               title_fontsize=6.3)

    # ---------------- (c) rate capability ---------------------------------
    steps = np.repeat(RATES, PER_RATE)
    idx = np.arange(1, steps.size + 1)
    cap = (Q_NOMINAL / (1.0 + (steps / 6.0) ** 1.1)
           * (1.0 - 3.0e-4 * idx) * (1.0 + RNG.normal(0.0, 0.004, idx.size)))
    for k, r in enumerate(RATES):
        sl = slice(k * PER_RATE, (k + 1) * PER_RATE)
        if k % 2:
            axb.axvspan(idx[sl][0] - 0.5, idx[sl][-1] + 0.5, color="#f5f5f5",
                        zorder=0)
        axb.text(idx[sl].mean(), cap[sl].mean() + 9.0, "%.1fC" % r,
                 fontsize=6.2, color="#555555", ha="center", va="bottom")
    axb.plot(idx, cap, color="#bbbbbb", lw=0.7, zorder=3)
    axb.plot(idx, cap, ls="none", marker="o", ms=3.4, mfc=ref["col"],
             mec="white", mew=0.5, zorder=5)
    recover = cap[-PER_RATE:].mean() / cap[:PER_RATE].mean()
    axb.annotate("%.1f %% of the initial capacity recovered" % (100.0 *
                                                                recover),
                 xy=(idx[-3], cap[-3]), xytext=(idx[-1] + 0.5, 18.0),
                 fontsize=6.3, color="#555555", ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7,
                                 mutation_scale=7, shrinkB=3.0))
    axb.set_xlim(0.2, idx[-1] + 0.8)
    axb.set_xlabel("Cycle number in the rate test", fontsize=8, labelpad=1.5)
    axb.set_ylabel("Specific capacity (mAh g$^{-1}$)", fontsize=8,
                   labelpad=1.5)
    axb.tick_params(labelsize=6.5)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    q_first = Q_NOMINAL * ref["ret"][SHOW[0] - 1] / 100.0
    lo, hi = shared_limits([cap, np.array([0.0, q_first])], pad=0.08)
    axs.set_xlim(max(lo, 0.0), hi)
    axb.set_ylim(max(lo, 0.0), hi)

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.40, dy=0.10)

    savefig(fig, "767_ec_battery_cycling.png")


if __name__ == "__main__":
    main()
