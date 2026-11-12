"""Tafel analysis of four oxygen-evolution catalysts.

(a) Overpotential against log current density.  Each data set is fitted over
    the same linear window; the fitted slope is reported in mV/dec and the
    exchange current density is read off where that fit extrapolates to zero
    overpotential, marked with an open circle on the baseline.  Points bend
    away from the fit at high current where mass transport takes over.
(b) Auxiliary panel: benchmark overpotential at 10 mA/cm2, taken from the same
    fits.
(c) Auxiliary panel: chronopotentiometric stability at 10 mA/cm2 over 24 h,
    sharing the overpotential scale of panel (b).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import axes_inches, panel_labels_inches, shared_limits

RNG = np.random.default_rng(70729)

J_BENCH = 10.0                 # benchmark current density, mA cm-2
FIT_LO, FIT_HI = 1.0, 60.0     # linear Tafel window, mA cm-2

CATS = [dict(name="NiFe LDH", b=42.0, j0=2.5e-3, drift=0.9, col="#3b5b92",
             lw=1.9),
        dict(name="IrO$_2$", b=58.0, j0=1.1e-3, drift=2.6, col="#2e7d5b",
             lw=1.2),
        dict(name="Co$_3$O$_4$", b=76.0, j0=6.0e-4, drift=5.4, col="#c9a227",
             lw=1.2),
        dict(name="NiO", b=118.0, j0=2.0e-4, drift=11.5, col="#8e6bbf",
             lw=1.2)]

FADE = "#b23a48"               # house red: degradation / transport limit


def overpotential(j, b, j0):
    """Tafel branch plus a mass-transport upturn at high current."""
    return b * np.log10(j / j0) + 46.0 * (j / 200.0) ** 1.6


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axs = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    j = np.logspace(-2.6, 2.10, 28)
    fitted = []
    for c in CATS:
        eta = overpotential(j, c["b"], c["j0"]) + RNG.normal(0.0, 3.2, j.size)
        w = (j >= FIT_LO) & (j <= FIT_HI)
        slope, icept = np.polyfit(np.log10(j[w]), eta[w], 1)
        fitted.append(dict(eta=eta, w=w, slope=slope, icept=icept,
                           j0=10.0 ** (-icept / slope),
                           eta10=slope * np.log10(J_BENCH) + icept))

    # ---------------- (a) Tafel plot --------------------------------------
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=2)
    ax.axvline(J_BENCH, color="#9a9a9a", lw=0.7, ls=(0, (2.4, 2.2)), zorder=2)
    ax.text(J_BENCH * 0.82, 782.0, "10 mA cm$^{-2}$ benchmark", fontsize=6.6,
            color="#777777", ha="right", va="top")
    ax.axvspan(FIT_HI, 3e2, color=FADE, alpha=0.06, lw=0, zorder=0)
    ax.text(2.6e2, 782.0, "transport\nlimited", fontsize=6.6, color=FADE,
            ha="right", va="top", linespacing=1.4)

    for c, f in zip(CATS, fitted):
        jl = np.logspace(np.log10(f["j0"]), np.log10(FIT_HI * 1.4), 60)
        ax.plot(jl, f["slope"] * np.log10(jl) + f["icept"], color=c["col"],
                lw=0.8, ls=(0, (3.0, 2.2)), alpha=0.85, zorder=3)
        jf = np.logspace(np.log10(FIT_LO), np.log10(FIT_HI), 30)
        ax.plot(jf, f["slope"] * np.log10(jf) + f["icept"], color=c["col"],
                lw=c["lw"] + 0.5, zorder=5)
        ax.plot(j, f["eta"], ls="none", marker="o", ms=3.4, mfc="white",
                mec=c["col"], mew=0.8, zorder=6,
                label="%s, $b$ = %.0f mV dec$^{-1}$" % (c["name"], f["slope"]))
        ax.plot([f["j0"]], [0.0], marker="o", ms=5.0, mfc="white",
                mec=c["col"], mew=1.2, zorder=7)

    best = int(np.argmin([f["eta10"] for f in fitted]))
    ax.annotate("$j_0$ = %.1f $\\times$ 10$^{-3}$ mA cm$^{-2}$"
                % (fitted[best]["j0"] * 1e3),
                xy=(fitted[best]["j0"], 0.0), xytext=(1.6e-4, 255.0),
                fontsize=6.8, color=CATS[best]["col"], ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-|>", color=CATS[best]["col"],
                                lw=0.8, mutation_scale=8, shrinkB=3.0))
    ax.annotate("$\\eta_{10}$ = %.0f mV" % fitted[best]["eta10"],
                xy=(J_BENCH, fitted[best]["eta10"]), xytext=(0.9, 300.0),
                fontsize=6.8, color=CATS[best]["col"], ha="right",
                va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CATS[best]["col"],
                                lw=0.8, mutation_scale=8, shrinkB=4.0))

    ax.set_xscale("log")
    ax.set_xlim(1.2e-4, 3.0e2)
    ax.set_ylim(-45.0, 800.0)
    ax.set_xlabel("Current density (mA cm$^{-2}$)")
    ax.set_ylabel("Overpotential $\\eta$ (mV)")
    ax.tick_params(labelsize=7.2, which="both")
    ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=7.0, handlelength=1.2,
              labelspacing=0.32, borderaxespad=0.4)
    annotate_box(ax, "Tafel slope $\\rightarrow$ rate-determining step\n"
                     "~40 mV dec$^{-1}$: second electron transfer\n"
                     "~60 mV dec$^{-1}$: chemical step after transfer\n"
                     "~120 mV dec$^{-1}$: first electron transfer",
                 loc="lower right", fontsize=6.5, color="#333333")

    # ---------------- (b) benchmark overpotential -------------------------
    eta10 = np.array([f["eta10"] for f in fitted])
    cols = [c["col"] for c in CATS]
    names = [c["name"] for c in CATS]
    order = np.argsort(eta10)
    axb.bar(np.arange(len(CATS)), eta10[order], width=0.62,
            color=[cols[i] for i in order], edgecolor="white", linewidth=0.6,
            zorder=3)
    for x, i in enumerate(order):
        axb.text(x, eta10[i] + 12.0, "%.0f" % eta10[i], fontsize=6.4,
                 color="#555555", ha="center", va="bottom", zorder=4)
    axb.set_xticks(np.arange(len(CATS)))
    axb.set_xticklabels([names[i] for i in order], fontsize=6.8)
    axb.set_xlabel("Overpotential at 10 mA cm$^{-2}$", fontsize=8,
                   labelpad=1.5)
    axb.set_ylabel("$\\eta_{10}$ (mV)", fontsize=8, labelpad=1.5)
    axb.tick_params(axis="y", labelsize=6.5)
    axb.tick_params(axis="x", length=0)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    # ---------------- (c) stability ---------------------------------------
    hrs = np.linspace(0.0, 24.0, 260)
    traces = []
    for c, f in zip(CATS, fitted):
        tr = (f["eta10"] * (1.0 + c["drift"] / 100.0 * (hrs / 24.0) ** 0.75)
              + RNG.normal(0.0, 1.6, hrs.size))
        traces.append(tr)
        axs.plot(hrs, tr, color=c["col"], lw=1.15, zorder=4)
    worst = int(np.argmax([c["drift"] for c in CATS]))
    for k in (best, worst):
        rise = (traces[k][-1] - traces[k][0]) / traces[k][0] * 100.0
        axs.text(24.6, traces[k][-1], "%s  +%.1f %%" % (names[k], rise),
                 fontsize=6.3, color=cols[k], ha="right", va="bottom")

    lim = (0.0, shared_limits([eta10, np.concatenate(traces)], pad=0.10)[1])
    axb.set_ylim(*lim)
    axs.set_ylim(*lim)
    axs.set_xlim(0.0, 25.0)
    axs.set_xlabel("Time at 10 mA cm$^{-2}$ (h)", fontsize=8, labelpad=1.5)
    axs.set_ylabel("$\\eta_{10}$ (mV)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5)
    axs.grid(color="#ededed", lw=0.5, zorder=0)
    axs.text(0.025, 0.955, "colours as in (a)", transform=axs.transAxes,
             fontsize=6.2, color="#777777", ha="left", va="top")

    panel_labels_inches(fig, [ax, axb, axs], dx=-0.34, dy=0.10)

    savefig(fig, "707_ech_tafel_analysis.png")


if __name__ == "__main__":
    main()
