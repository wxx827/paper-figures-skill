"""Turnover kinetics of a deactivating catalyst over a temperature series.

(a) Conversion against time on stream at five bed temperatures.  Each curve
    integrates a first-order rate whose site activity decays exponentially, so
    the hottest bed is fastest at first and then falls behind - the dashed red
    reference is the same 673 K run without deactivation and the shaded wedge
    is the conversion it loses.  An Arrhenius inset regresses the rate
    constants and reports the activation energy.
(b) Auxiliary panel: log-log rate orders in the two reactants.
(c) Auxiliary panel: normalised site activity against time on stream, with the
    half-life of the hottest bed.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches, shared_limits

RNG = np.random.default_rng(70417)

R_GAS = 8.314                 # J mol-1 K-1
T_REF = 623.0                 # K
K_REF = 0.055                 # first-order rate constant at T_REF, min-1
KD_REF = 0.020                # deactivation constant at T_REF, min-1
EA = 82.0e3                   # J mol-1
EA_D = 130.0e3                # J mol-1
SITE_RATIO = 850.0            # mol reactant fed per mol active site
TEMPS = np.array([573.0, 598.0, 623.0, 648.0, 673.0])

LOSS = "#b23a48"              # house red: what deactivation costs
ORDERS = [("reactant A", 0.86, 0.42, "#2e7d5b"),
          ("reactant B", 0.45, 0.30, "#c9a227")]


def arr(pre, ea, t):
    """Arrhenius scaling of ``pre`` measured at ``T_REF`` to temperature t."""
    return pre * np.exp(-ea / R_GAS * (1.0 / t - 1.0 / T_REF))


def conversion(t, k, kd):
    """First-order conversion with an exponentially decaying site count."""
    return 1.0 - np.exp(-(k / kd) * (1.0 - np.exp(-kd * t)))


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axo = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axd = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    cols = get_palette("cb_sequential_blue", 7)[2:7]
    t = np.linspace(0.0, 120.0, 400)
    ks = arr(K_REF, EA, TEMPS)
    kds = arr(KD_REF, EA_D, TEMPS)
    curves = [conversion(t, k, kd) for k, kd in zip(ks, kds)]

    # ---------------- (a) conversion vs time on stream --------------------
    i_hot = int(np.argmax(TEMPS))
    ideal = 1.0 - np.exp(-ks[i_hot] * t)
    ax.fill_between(t, curves[i_hot] * 100.0, ideal * 100.0, color=LOSS,
                    alpha=0.10, zorder=1)
    ax.plot(t, ideal * 100.0, color=LOSS, lw=1.0, ls=(0, (3.4, 2.4)),
            zorder=3, label="%d K, no deactivation" % TEMPS[i_hot])
    for tt, c, y in zip(TEMPS, cols, curves):
        ax.plot(t, y * 100.0, color=c, lw=1.55, zorder=5,
                label="%d K" % tt)

    gap = (ideal[-1] - curves[i_hot][-1]) * 100.0
    ax.annotate("%.0f %% conversion lost to\nsintering at %d K"
                % (gap, TEMPS[i_hot]),
                xy=(96.0, 0.5 * (ideal[-1] + curves[i_hot][-1]) * 100.0),
                xytext=(44.0, 104.0), fontsize=6.8, color=LOSS, ha="left",
                va="bottom", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=LOSS, lw=0.8,
                                mutation_scale=8, shrinkB=2.0))

    i_ref = int(np.argmin(np.abs(TEMPS - T_REF)))
    tof0 = SITE_RATIO * ks[i_ref] / 60.0
    ton = SITE_RATIO * curves[i_ref][-1]
    ax.text(0.025, 0.965,
            "at %d K:  TOF$_0$ = %.2f s$^{-1}$,  TON = %.0f after %.0f min"
            % (TEMPS[i_ref], tof0, ton, t[-1]),
            transform=ax.transAxes, fontsize=7.0, color="#1a1a1a",
            ha="left", va="top")

    ax.set_xlim(0.0, 124.0)
    ax.set_ylim(0.0, 132.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Time on stream (min)")
    ax.set_ylabel("Conversion (%)")
    ax.tick_params(labelsize=7.2)
    ax.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=6,
              fontsize=6.8, handlelength=1.3, columnspacing=0.9,
              borderaxespad=0.0)

    ins = ax.inset_axes([0.585, 0.085, 0.375, 0.345])
    inv_t = 1000.0 / TEMPS
    lnk = np.log(ks)
    slope, icept = np.polyfit(inv_t, lnk, 1)
    ea_fit = -slope * R_GAS          # slope is per 1000/T, so this is kJ mol-1
    xg = np.linspace(inv_t.min() - 0.03, inv_t.max() + 0.03, 30)
    ins.plot(xg, icept + slope * xg, color="#555555", lw=0.9,
             ls=(0, (3.0, 2.2)), zorder=2)
    for x, y, c in zip(inv_t, lnk, cols):
        ins.plot([x], [y], marker="o", ms=3.6, mfc=c, mec="white", mew=0.6,
                 ls="none", zorder=3)
    ins.set_xlabel("1000 / $T$ (K$^{-1}$)", fontsize=6.4, labelpad=1.5)
    ins.set_ylabel("ln($k$ / min$^{-1}$)", fontsize=6.4, labelpad=1.5)
    ins.tick_params(labelsize=6.0, length=2.0, pad=1.4)
    ins.text(0.94, 0.90, "$E_a$ = %.0f kJ mol$^{-1}$" % ea_fit,
             transform=ins.transAxes, ha="right", va="top", fontsize=6.4,
             color="#1a1a1a", fontweight="bold")
    for s in ins.spines.values():
        s.set_linewidth(0.6)
    ins.patch.set_alpha(0.94)

    # ---------------- (b) reaction orders ---------------------------------
    p = np.logspace(-1.0, 0.30, 9)
    rate_sets = []
    for name, order, pre, col in ORDERS:
        rate = pre * p ** order * np.exp(RNG.normal(0.0, 0.045, p.size))
        rate_sets.append(rate)
        n_fit, b_fit = np.polyfit(np.log10(p), np.log10(rate), 1)
        pg = np.logspace(-1.08, 0.36, 40)
        axo.plot(pg, 10.0 ** (b_fit + n_fit * np.log10(pg)), color=col,
                 lw=1.0, ls=(0, (3.2, 2.2)), zorder=3)
        axo.plot(p, rate, ls="none", marker="o", ms=3.6, mfc=col, mec="white",
                 mew=0.6, zorder=4,
                 label="%s, $n$ = %.2f" % (name, n_fit))
    axo.set_xscale("log")
    axo.set_yscale("log")
    axo.set_xlim(*shared_limits([p], pad=0.10, log=True))
    axo.set_ylim(*shared_limits(rate_sets, pad=0.16, log=True))
    axo.set_xlabel("Partial pressure (bar)", fontsize=8, labelpad=1.5)
    axo.set_ylabel("Rate (mmol g$^{-1}$ s$^{-1}$)", fontsize=8, labelpad=1.5)
    axo.tick_params(labelsize=6.5, which="both")
    axo.grid(which="major", color="#ededed", lw=0.5, zorder=0)
    axo.legend(loc="upper left", fontsize=6.5, handlelength=1.2,
               labelspacing=0.3, borderaxespad=0.35)

    # ---------------- (c) deactivation ------------------------------------
    ends = np.exp(-kds * t[-1]) * 100.0
    for tt, c, kd in zip(TEMPS, cols, kds):
        axd.plot(t, np.exp(-kd * t) * 100.0, color=c, lw=1.35, zorder=4)
    # Only the two extremes get an end label; the three intermediate curves
    # end within a few points of each other and would collide.
    for i in (int(np.argmin(TEMPS)), i_hot):
        axd.text(123.0, max(ends[i], 4.5), "%d K" % TEMPS[i], fontsize=6.2,
                 color=cols[i], ha="left", va="center")
    axd.text(0.975, 0.965, "colours as in (a)", transform=axd.transAxes,
             fontsize=6.2, color="#777777", ha="right", va="top")
    t_half = np.log(2.0) / kds[i_hot]
    axd.axhline(50.0, color="#808080", lw=0.7, ls=(0, (2.6, 2.2)), zorder=2)
    axd.annotate("$t_{1/2}$ = %.1f min at %d K" % (t_half, TEMPS[i_hot]),
                 xy=(t_half, 50.0), xytext=(30.0, 74.0), fontsize=6.5,
                 color=LOSS, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=LOSS, lw=0.7,
                                 mutation_scale=7, shrinkB=2.0))
    axd.set_xlim(0.0, 138.0)
    axd.set_ylim(0.0, 106.0)
    axd.set_xlabel("Time on stream (min)", fontsize=8, labelpad=1.5)
    axd.set_ylabel("Site activity (%)", fontsize=8, labelpad=1.5)
    axd.tick_params(labelsize=6.5)
    axd.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    panel_labels_inches(fig, [ax, axo, axd], dx=-0.34, dy=0.10)

    savefig(fig, "704_cat_turnover_kinetics.png")


if __name__ == "__main__":
    main()
