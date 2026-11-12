"""Sabatier volcano for the four-electron oxygen evolution reaction.

(a) Main panel: activity, as the logarithm of the exchange current density,
    against the binding-energy descriptor dG(O*) - dG(OH*).  The two branches
    are two different potential-determining steps - O* formation on the
    weak-binding side, OOH* formation on the strong-binding side - each a
    straight line in the descriptor, and their crossing is the volcano apex.
    Metals and alloys are scattered on the curve; element labels are pushed
    onto four alternating offset levels with leader lines so that crowded
    points never overprint one another.
(b) Auxiliary panel: the OOH*/OH* scaling relation that makes a single
    descriptor legal, with its least-squares line, equation and R-squared.
(c) Auxiliary panel: theoretical overpotential of the best surfaces against
    the floor that the same scaling relation imposes on any of them.

Every number printed on the figure is computed from the arrays that are
plotted.  Illustrative values, not measurements.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

RNG = np.random.default_rng(76201)

# Standard four-electron OER thermodynamics (Norskov-type analysis).
G_O2 = 4.92            # total free energy of the 4-electron reaction, eV
G_SCALE = 3.20         # universal dG(OOH*) - dG(OH*) offset, eV
U_EQ = 1.23            # equilibrium potential, V vs RHE
B_STRONG = 0.250       # effective Tafel-like decade width, strong side, V/dec
B_WEAK = 0.190         # effective decade width, weak-binding side, V/dec
LOG_J_TOP = 0.55       # apex activity, log10(j0 / mA cm-2)

STRONG = "#2e7d5b"     # house green: binds oxygen too strongly
WEAK = "#c9a227"       # house gold: binds oxygen too weakly
BEST = "#3b5b92"       # house blue: the single emphasised surface
CURVE = "#8a8a8a"
FLOOR = "#b23a48"      # house red: the scaling-imposed limit

SURFACES = [("Ru", 1.14), ("Rh", 1.26), ("Ir", 1.42), ("IrO$_2$", 1.51),
            ("Pt$_3$Ni", 1.58), ("RuO$_2$", 1.64), ("Co", 1.72),
            ("Ni", 1.83), ("NiFe", 1.90), ("Pd", 2.01), ("Cu", 2.14),
            ("Ag", 2.26), ("Au", 2.38)]


ETA_MIN = G_SCALE / 2.0 - U_EQ     # overpotential at the apex, V


def overpotential(x):
    """Theoretical OER overpotential in V for descriptor x = dG_O - dG_OH."""
    return np.maximum(x, G_SCALE - x) - U_EQ


def branch(x, kind):
    """One straight branch of the volcano, extended past the apex.

    Both branches pass through the apex activity by construction, so the
    crossing of the two limiting steps is the top of the volcano.
    """
    if kind == "strong":                       # OOH* formation limits
        return LOG_J_TOP - ((G_SCALE - x) - U_EQ - ETA_MIN) / B_STRONG
    return LOG_J_TOP - (x - U_EQ - ETA_MIN) / B_WEAK      # O* formation


def activity(x):
    """Volcano curve: the lower of the two branches at every descriptor value."""
    return np.minimum(branch(x, "strong"), branch(x, "weak"))


def stagger(xs, ys, steps=(0.26, -0.26, 0.50, -0.50)):
    """Send each label to whichever offset level has the most room left."""
    order = np.argsort(xs)
    last = np.full(len(steps), -1e9)
    out = np.zeros_like(ys)
    for i in order:
        k = int(np.argmax(xs[i] - last))
        out[i] = steps[k]
        last[k] = xs[i]
    return out


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 1.02, 5.32, 4.62))
    axs = axes_inches(fig, (7.32, 3.74, 2.88, 1.90))
    axb = axes_inches(fig, (7.32, 1.02, 2.88, 1.90))

    names = [s[0] for s in SURFACES]
    dx = np.array([s[1] for s in SURFACES])
    dy = activity(dx) + RNG.normal(0.0, 0.10, dx.size)
    eta = overpotential(dx)

    k_best = int(np.argmax(dy))
    cols = [WEAK if v > G_SCALE / 2.0 else STRONG for v in dx]
    cols[k_best] = BEST

    # ---------------- (a) the volcano -------------------------------------
    xs = np.linspace(0.95, 2.50, 400)
    ys = activity(xs)
    x_apex = float(xs[int(np.argmax(ys))])
    eta_floor = float(overpotential(np.array([x_apex]))[0])

    ax.axvspan(0.95, x_apex, color="#f5f5f5", zorder=0)
    ax.axvspan(x_apex, 2.50, color="#f0f0f0", zorder=0)

    xe = np.linspace(x_apex, 2.50, 40)
    ax.plot(xe, branch(xe, "strong"), color=STRONG, lw=0.8,
            ls=(0, (3.0, 2.4)), alpha=0.85, zorder=2)
    xw = np.linspace(0.95, x_apex, 40)
    ax.plot(xw, branch(xw, "weak"), color=WEAK, lw=0.8, ls=(0, (3.0, 2.4)),
            alpha=0.85, zorder=2)
    ax.plot(xs, ys, color=CURVE, lw=1.6, zorder=3)

    for x, y, c in zip(dx, dy, cols):
        big = c == BEST
        ax.plot([x], [y], marker="o", ms=7.4 if big else 5.0, mfc=c,
                mec="white", mew=1.0 if big else 0.7, ls="none",
                zorder=7 if big else 6)

    off = stagger(dx, dy)
    for x, y, d, c, n in zip(dx, dy, off, cols, names):
        up = d > 0
        ax.plot([x, x], [y + np.sign(d) * 0.09, y + d - np.sign(d) * 0.05],
                color="#b8b8b8", lw=0.6, zorder=4)
        ax.text(x, y + d, n, fontsize=7.0 if c == BEST else 6.6, color=c,
                ha="center", va="bottom" if up else "top",
                fontweight="bold" if c == BEST else "normal", zorder=8)

    ax.axvline(x_apex, color="#777777", lw=0.7, ls=(0, (2.4, 2.0)), zorder=1)
    ax.annotate("Sabatier apex\n$\\Delta G_{O*}-\\Delta G_{OH*}$ = %.2f eV\n"
                "$\\eta_{min}$ = %.2f V" % (x_apex, eta_floor),
                xy=(x_apex, LOG_J_TOP), xytext=(1.98, 1.06),
                fontsize=7.0, color="#333333", ha="left", va="center",
                linespacing=1.40,
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    ax.annotate("%s, %.2f dec above Pd" % (names[k_best],
                                           dy[k_best] - dy[names.index("Pd")]),
                xy=(dx[k_best], dy[k_best]), xytext=(1.00, 0.38),
                fontsize=6.8, color=BEST, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=BEST, lw=0.8,
                                mutation_scale=8, shrinkB=4.0))

    ax.text(1.00, -4.55, "O* bound too strongly\nOOH* formation limits\n"
            "slope +%.1f dec eV$^{-1}$" % (1.0 / B_STRONG), fontsize=6.8,
            color=STRONG, ha="left", va="bottom", linespacing=1.45)
    ax.text(2.45, -4.55, "O* bound too weakly\nO* formation limits\n"
            "slope -%.1f dec eV$^{-1}$" % (1.0 / B_WEAK), fontsize=6.8,
            color=WEAK, ha="right", va="bottom", linespacing=1.45)

    ax.set_xlim(0.95, 2.50)
    ax.set_ylim(-4.70, 1.55)
    ax.set_xlabel("Descriptor $\\Delta G_{O*}-\\Delta G_{OH*}$ (eV)")
    ax.set_ylabel("Activity, log$_{10}$($j_0$ / mA cm$^{-2}$)")
    ax.tick_params(labelsize=7.2)
    ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
    ax.legend(handles=[
        Line2D([], [], color=CURVE, lw=1.6, label="Sabatier volcano"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc=STRONG, mec="white",
               label="strong-binding branch"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc=WEAK, mec="white",
               label="weak-binding branch"),
        Line2D([], [], ls="none", marker="o", ms=6.8, mfc=BEST, mec="white",
               label="most active surface")],
        loc="upper left", fontsize=7.0, handlelength=1.4, labelspacing=0.32,
        borderaxespad=0.4)

    # ---------------- (b) scaling relation --------------------------------
    g_oh = 0.62 * dx + RNG.normal(0.0, 0.055, dx.size)
    g_ooh = g_oh + G_SCALE + RNG.normal(0.0, 0.070, dx.size)
    slope, icept = np.polyfit(g_oh, g_ooh, 1)
    pred = slope * g_oh + icept
    r2 = 1.0 - np.sum((g_ooh - pred) ** 2) / np.sum((g_ooh - g_ooh.mean()) ** 2)
    resid = float(np.std(g_ooh - pred))

    xf = np.linspace(g_oh.min() - 0.10, g_oh.max() + 0.10, 60)
    axs.fill_between(xf, slope * xf + icept - resid, slope * xf + icept + resid,
                     color="#ececec", zorder=1)
    axs.plot(xf, slope * xf + icept, color="#555555", lw=1.1,
             ls=(0, (3.4, 2.4)), zorder=3)
    for x, y, c in zip(g_oh, g_ooh, cols):
        big = c == BEST
        axs.plot([x], [y], marker="o", ms=5.6 if big else 3.8, mfc=c,
                 mec="white", mew=0.7, ls="none", zorder=5 if big else 4)
    axs.text(0.035, 0.955,
             "$\\Delta G_{OOH*}$ = %.2f $\\Delta G_{OH*}$ + %.2f eV\n"
             "$R^2$ = %.3f,  $n$ = %d,  s = %.2f eV"
             % (slope, icept, r2, g_oh.size, resid),
             transform=axs.transAxes, fontsize=6.4, color="#333333",
             ha="left", va="top", linespacing=1.45)
    axs.set_xlabel("$\\Delta G_{OH*}$ (eV)", fontsize=8, labelpad=1.5)
    axs.set_ylabel("$\\Delta G_{OOH*}$ (eV)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5)
    axs.grid(color="#ededed", lw=0.5, zorder=0)

    # ---------------- (c) theoretical limit -------------------------------
    order = np.argsort(eta)[:6]
    axb.bar(np.arange(order.size), eta[order], width=0.62,
            color=[cols[i] for i in order], edgecolor="white", linewidth=0.6,
            zorder=3)
    axb.axhline(eta_floor, color=FLOOR, lw=1.0, ls=(0, (3.0, 2.2)), zorder=5)
    axb.text(0.025, 0.955, "dashed: scaling floor %.2f V" % eta_floor,
             transform=axb.transAxes, fontsize=6.3, color=FLOOR, ha="left",
             va="top")
    for x, i in enumerate(order):
        axb.text(x, eta[i] + 0.012, "%.2f" % eta[i], fontsize=6.2,
                 color="#555555", ha="center", va="bottom", zorder=4)
    axb.set_xticks(np.arange(order.size))
    axb.set_xticklabels([names[i] for i in order], fontsize=6.6)
    axb.set_ylim(0.0, float(eta[order].max()) * 1.30)
    axb.set_xlabel("Six surfaces closest to the apex", fontsize=8, labelpad=1.5)
    axb.set_ylabel("Theoretical $\\eta$ (V)", fontsize=8, labelpad=1.5)
    axb.tick_params(axis="y", labelsize=6.5)
    axb.tick_params(axis="x", length=0)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.36, dy=0.10)

    savefig(fig, "762_ecat_volcano_plot.png")


if __name__ == "__main__":
    main()
