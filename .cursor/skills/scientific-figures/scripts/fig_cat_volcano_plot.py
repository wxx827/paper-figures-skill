"""Sabatier volcano: catalytic activity against an adsorption-energy descriptor.

(a) Volcano relation between the oxygen binding free energy and the turnover
    activity.  Two straight branches meet at the Sabatier optimum: the
    strong-binding side is limited by product removal, the weak-binding side by
    reactant activation.  Metals are scattered on the curve and labelled with
    their element symbols; element labels are staggered so that neighbouring
    points never collide.
(b) Auxiliary panel (55 % of the main width): the scaling relation between two
    adsorbate binding energies that makes a one-dimensional descriptor legal in
    the first place, with the least-squares line and its statistics.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

RNG = np.random.default_rng(70314)

G_OPT = 1.60            # Sabatier optimum of the descriptor, eV
A_STRONG = 3.05         # branch slope on the strong-binding side, dec/eV
A_WEAK = 2.35           # branch slope on the weak-binding side, dec/eV
LOG_TOF_MAX = 1.15      # apex activity, log10(TOF / s-1)

STRONG = "#2e7d5b"      # house green: binds too strongly
WEAK = "#c9a227"        # house gold: binds too weakly
BEST = "#3b5b92"        # house blue: the single emphasised catalyst
CURVE = "#8a8a8a"

METALS = [("W", 0.58), ("Mo", 0.74), ("Re", 0.96), ("Ru", 1.19),
          ("Co", 1.33), ("Ni", 1.45), ("Rh", 1.56), ("Pt$_3$Ni", 1.63),
          ("Ir", 1.76), ("Pd", 1.92), ("Cu", 2.16), ("Au", 2.44),
          ("Ag", 2.63)]


def volcano(x):
    """Two-branch Sabatier volcano, log10 turnover frequency in s-1."""
    return LOG_TOF_MAX - np.maximum(A_STRONG * (G_OPT - x),
                                    A_WEAK * (x - G_OPT))


def stagger(xs, ys, min_dx=0.17, off=0.155):
    """Alternate label offsets above / below whenever two points crowd."""
    order = np.argsort(xs)
    out = np.zeros_like(ys)
    side, prev = 1.0, -1e9
    for i in order:
        side = -side if xs[i] - prev < min_dx else 1.0
        out[i] = side * off
        prev = xs[i]
    return out


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.40, 4.62))
    axs = axes_inches(fig, (7.20, 1.02, 3.00, 4.62))

    names = [m[0] for m in METALS]
    gx = np.array([m[1] for m in METALS])
    gy = volcano(gx) + RNG.normal(0.0, 0.085, gx.size)

    k_best = int(np.argmax(gy))
    cols = [WEAK if g > G_OPT else STRONG for g in gx]
    cols[k_best] = BEST

    # ---------------- (a) the volcano -------------------------------------
    xs = np.linspace(0.35, 2.85, 400)
    ys = volcano(xs)
    x_apex = float(xs[int(np.argmax(ys))])

    ax.axvspan(0.35, G_OPT, color="#f5f5f5", zorder=0)
    ax.axvspan(G_OPT, 2.85, color="#f0f0f0", zorder=0)
    ax.plot(xs, ys, color=CURVE, lw=1.5, zorder=3)
    xe = 0.28                              # dashed continuation past the apex
    t1 = np.linspace(G_OPT, G_OPT + xe, 20)
    t2 = np.linspace(G_OPT - xe, G_OPT, 20)
    ax.plot(t1, LOG_TOF_MAX + A_STRONG * (t1 - G_OPT), color=STRONG, lw=0.8,
            ls=(0, (3.0, 2.4)), alpha=0.8, zorder=2)
    ax.plot(t2, LOG_TOF_MAX + A_WEAK * (G_OPT - t2), color=WEAK, lw=0.8,
            ls=(0, (3.0, 2.4)), alpha=0.8, zorder=2)

    for x, y, c in zip(gx, gy, cols):
        big = c == BEST
        ax.plot([x], [y], marker="o", ms=7.2 if big else 5.0,
                mfc=c, mec="white", mew=0.9 if big else 0.7,
                zorder=7 if big else 6, ls="none")

    dy = stagger(gx, gy)
    for x, y, d, c, n in zip(gx, gy, dy, cols, names):
        ax.text(x, y + d, n, fontsize=7.0 if c == BEST else 6.6, color=c,
                ha="center", va="bottom" if d > 0 else "top",
                fontweight="bold" if c == BEST else "normal", zorder=8)

    ax.axvline(x_apex, color="#777777", lw=0.7, ls=(0, (2.4, 2.0)), zorder=1)
    ax.annotate("Sabatier optimum\n$\\Delta G_{ads}$ = %.2f eV" % x_apex,
                xy=(x_apex, LOG_TOF_MAX), xytext=(x_apex + 0.46, 1.70),
                fontsize=7.0, color="#333333", ha="left", va="center",
                fontweight="bold", linespacing=1.35,
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    ax.annotate("%s: %.2f dec above Pd"
                % (names[k_best], gy[k_best] - gy[names.index("Pd")]),
                xy=(gx[k_best], gy[k_best]), xytext=(2.05, 0.72),
                fontsize=6.8, color=BEST, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=BEST, lw=0.8,
                                mutation_scale=8, shrinkB=4.0))

    ax.text(0.42, -3.55, "binding too strong\nproduct desorption limits\n"
            "slope $\\approx$ +%.1f dec eV$^{-1}$" % A_STRONG,
            fontsize=6.8, color=STRONG, ha="left", va="bottom",
            linespacing=1.45)
    ax.text(2.78, -3.55, "binding too weak\ndissociation limits\n"
            "slope $\\approx$ -%.1f dec eV$^{-1}$" % A_WEAK,
            fontsize=6.8, color=WEAK, ha="right", va="bottom",
            linespacing=1.45)

    ax.set_xlim(0.35, 2.85)
    ax.set_ylim(-3.65, 2.05)
    ax.set_xlabel("O* binding free energy $\\Delta G_{ads}$ (eV)")
    ax.set_ylabel("Activity, log$_{10}$(TOF / s$^{-1}$)")
    ax.tick_params(labelsize=7.2)
    ax.legend(handles=[
        Line2D([], [], color=CURVE, lw=1.5, label="Sabatier volcano"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc=STRONG,
               mec="white", label="strong-binding branch"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc=WEAK,
               mec="white", label="weak-binding branch"),
        Line2D([], [], ls="none", marker="o", ms=6.6, mfc=BEST,
               mec="white", label="best candidate")],
        loc="upper left", fontsize=7.0, handlelength=1.4, labelspacing=0.32,
        borderaxespad=0.4)

    # ---------------- (b) scaling relation --------------------------------
    g_oh = np.linspace(0.15, 1.75, len(METALS)) + RNG.normal(0, 0.05,
                                                             len(METALS))
    g_ooh = 0.86 * g_oh + 3.14 + RNG.normal(0.0, 0.075, g_oh.size)
    slope, icept = np.polyfit(g_oh, g_ooh, 1)
    pred = slope * g_oh + icept
    r2 = 1.0 - np.sum((g_ooh - pred) ** 2) / np.sum((g_ooh - g_ooh.mean()) ** 2)

    xf = np.linspace(g_oh.min() - 0.12, g_oh.max() + 0.12, 60)
    axs.plot(xf, slope * xf + icept, color="#555555", lw=1.1,
             ls=(0, (3.4, 2.4)), zorder=3)
    axs.fill_between(xf, slope * xf + icept - 0.16, slope * xf + icept + 0.16,
                     color="#e8e8e8", zorder=1)
    for x, y, c, n in zip(g_oh, g_ooh, cols, names):
        big = c == BEST
        axs.plot([x], [y], marker="o", ms=6.4 if big else 4.6, mfc=c,
                 mec="white", mew=0.8 if big else 0.6, ls="none",
                 zorder=6 if big else 5)
    axs.annotate(names[k_best], xy=(g_oh[k_best], g_ooh[k_best]),
                 xytext=(g_oh[k_best] + 0.10, g_ooh[k_best] - 0.34),
                 fontsize=6.6, color=BEST, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="-", color=BEST, lw=0.6,
                                 shrinkB=3.0))

    axs.text(0.04, 0.955,
             "$\\Delta G_{OOH}$ = %.2f $\\Delta G_{OH}$ + %.2f eV\n"
             "$R^2$ = %.3f,  n = %d" % (slope, icept, r2, g_oh.size),
             transform=axs.transAxes, fontsize=6.8, color="#333333",
             ha="left", va="top", linespacing=1.45)
    axs.set_xlabel("$\\Delta G_{OH}$ (eV)", fontsize=8)
    axs.set_ylabel("$\\Delta G_{OOH}$ (eV)", fontsize=8)
    axs.tick_params(labelsize=6.8)
    axs.grid(color="#ececec", lw=0.5, zorder=0)

    panel_labels_inches(fig, [ax, axs], dx=-0.34, dy=0.10)

    savefig(fig, "703_cat_volcano_plot.png")


if __name__ == "__main__":
    main()
