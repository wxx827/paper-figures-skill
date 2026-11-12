"""Free-energy staircase for the four-electron oxygen evolution reaction.

(a) Main panel: cumulative Gibbs free energy along the reaction path at the
    equilibrium potential, one staircase per catalyst plus the ideal catalyst
    whose four steps are all flat.  The best surface is repeated as a dashed
    staircase at its own onset potential, where every step has turned
    downhill.  The potential-determining step of each catalyst is marked with
    a double-headed arrow carrying its height in eV.
(b) Auxiliary panel: the four elementary-step energies of every catalyst at
    zero bias, against the 1.23 eV that an ideal catalyst would need.
(c) Auxiliary panel: the resulting overpotential split into the floor imposed
    by adsorbate scaling and the excess each surface adds on top of it.

All barriers, overpotentials and onset potentials are recomputed from the
adsorption energies that draw the staircases.  Illustrative values.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

G_TOTAL = 4.92          # free energy of 2 H2O -> O2 + 4 (H+ + e-), eV
U_EQ = 1.23             # equilibrium potential, V vs RHE
G_SCALE = 3.20          # dG(OOH*) - dG(OH*) fixed by scaling, eV
ETA_FLOOR = G_SCALE / 2.0 - U_EQ      # overpotential floor from scaling, V

STATES = ["2 H$_2$O", "OH*", "O*", "OOH*", "O$_2$"]
HALF = 0.27             # half width of a state level, reaction-coordinate units

IDEAL = "#8a8a8a"
PDS = "#b23a48"         # house red: the single emphasised feature

CATS = [dict(name="RuO$_2$", g_oh=0.85, dx=1.52, col="#3b5b92"),
        dict(name="IrO$_2$", g_oh=1.05, dx=1.75, col="#2e7d5b"),
        dict(name="Co$_3$O$_4$", g_oh=1.25, dx=1.90, col="#c9a227")]


def levels(g_oh, dx, u):
    """Cumulative free energy of the five states at potential ``u``."""
    g_o = g_oh + dx
    g_ooh = g_oh + G_SCALE
    raw = np.array([0.0, g_oh, g_o, g_ooh, G_TOTAL])
    return raw - u * np.arange(5.0)


def staircase(ax, y, color, lw=1.7, ls="-", alpha=1.0, zorder=4, label=None):
    """Level segments joined by dotted risers; returns the level artist."""
    art = None
    for i, v in enumerate(y):
        ln, = ax.plot([i - HALF, i + HALF], [v, v], color=color, lw=lw, ls=ls,
                      alpha=alpha, solid_capstyle="butt", zorder=zorder,
                      label=label if i == 0 else None)
        art = art or ln
    for i in range(len(y) - 1):
        ax.plot([i + HALF, i + 1 - HALF], [y[i], y[i + 1]], color=color,
                lw=0.8, ls=(0, (1.6, 1.8)), alpha=alpha * 0.85,
                zorder=zorder - 1)
    return art


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 1.02, 5.32, 4.62))
    axs = axes_inches(fig, (7.32, 3.74, 2.88, 1.90))
    axb = axes_inches(fig, (7.32, 1.02, 2.88, 1.90))

    for c in CATS:
        g0 = levels(c["g_oh"], c["dx"], 0.0)
        c["steps"] = np.diff(g0)                       # elementary steps, eV
        c["k_pds"] = int(np.argmax(c["steps"]))
        c["dg_pds"] = float(c["steps"].max())
        c["eta"] = c["dg_pds"] - U_EQ
        c["y_eq"] = levels(c["g_oh"], c["dx"], U_EQ)

    best = int(np.argmin([c["eta"] for c in CATS]))
    u_onset = U_EQ + CATS[best]["eta"]

    # ---------------- (a) the staircase -----------------------------------
    ax.axhline(0.0, color="#cccccc", lw=0.7, zorder=1)
    staircase(ax, np.zeros(5), IDEAL, lw=1.3, ls=(0, (4.0, 2.4)), zorder=3,
              label="ideal catalyst, four equal steps")
    for c in CATS:
        staircase(ax, c["y_eq"], c["col"], zorder=5,
                  label="%s, $\\eta$ = %.2f V" % (c["name"], c["eta"]))

    y_on = levels(CATS[best]["g_oh"], CATS[best]["dx"], u_onset)
    staircase(ax, y_on, CATS[best]["col"], lw=1.1, ls=(0, (3.0, 2.2)),
              alpha=0.85, zorder=4,
              label="%s at onset $U$ = %.2f V" % (CATS[best]["name"], u_onset))
    ax.text(4.0 + HALF + 0.06, y_on[-1], "every step downhill\nat $U$ = %.2f V"
            % u_onset, fontsize=6.5, color=CATS[best]["col"], ha="right",
            va="bottom", linespacing=1.40)

    for i, c in enumerate(CATS):
        k = c["k_pds"]
        y0, y1 = c["y_eq"][k], c["y_eq"][k + 1]
        # Two surfaces can share a potential-determining step; nudge the
        # arrows apart so the double heads stay readable.
        share = [j for j, d in enumerate(CATS) if d["k_pds"] == k]
        xa = k + 0.5 + (share.index(i) - (len(share) - 1) / 2.0) * 0.14
        ax.annotate("", xy=(xa, y1), xytext=(xa, y0),
                    arrowprops=dict(arrowstyle="<->", color=c["col"], lw=0.9,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        left = share.index(i) < (len(share) - 1) / 2.0
        ax.text(xa - 0.06 if left else xa + 0.06, y1 - 0.10,
                "$\\Delta G_{PDS}$ = %.2f eV" % (y1 - y0), fontsize=6.5,
                color=c["col"], ha="right" if left else "left", va="top")

    ax.text(0.02, 0.965, "double arrows mark the potential-determining step "
            "of each surface", transform=ax.transAxes, fontsize=6.6,
            color="#555555", ha="left", va="top")

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(STATES)
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-2.05, 1.35)
    ax.set_xlabel("Reaction coordinate at $U$ = %.2f V vs RHE" % U_EQ)
    ax.set_ylabel("Cumulative free energy $\\Delta G$ (eV)")
    ax.tick_params(labelsize=7.2)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#f2f2f2", lw=0.5, zorder=0)
    ax.legend(loc="lower left", fontsize=7.0, handlelength=1.6,
              labelspacing=0.34, borderaxespad=0.4)

    # ---------------- (b) elementary steps --------------------------------
    n = len(CATS)
    w = 0.20
    for i, c in enumerate(CATS):
        xs = np.arange(4) + (i - (n - 1) / 2.0) * w
        axs.bar(xs, c["steps"], width=w * 0.88, color=c["col"],
                edgecolor="white", linewidth=0.5, zorder=3)
        axs.plot(xs[c["k_pds"]], c["steps"][c["k_pds"]] + 0.13, marker="v",
                 ms=3.4, color=PDS, zorder=5)
    axs.axhline(U_EQ, color=IDEAL, lw=1.0, ls=(0, (3.0, 2.2)), zorder=4)
    axs.text(0.025, 0.955, "dashed: ideal %.2f eV per step;  red marker: PDS"
             % U_EQ, transform=axs.transAxes, fontsize=6.3, color="#555555",
             ha="left", va="top")
    axs.set_xticks(np.arange(4))
    axs.set_xticklabels(["%s$\\rightarrow$%s" % (STATES[i], STATES[i + 1])
                         for i in range(4)], fontsize=6.2)
    axs.set_ylim(0.0, max(c["steps"].max() for c in CATS) * 1.32)
    axs.set_xlabel("Elementary step at zero bias", fontsize=8, labelpad=1.5)
    axs.set_ylabel("$\\Delta G$ (eV)", fontsize=8, labelpad=1.5)
    axs.tick_params(axis="y", labelsize=6.5)
    axs.tick_params(axis="x", length=0)
    axs.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    # ---------------- (c) overpotential split -----------------------------
    eta = np.array([c["eta"] for c in CATS])
    xs = np.arange(n)
    axb.bar(xs, np.full(n, ETA_FLOOR), width=0.56, color="#d5d5d5",
            edgecolor="white", linewidth=0.6, zorder=3,
            label="scaling floor %.2f V" % ETA_FLOOR)
    axb.bar(xs, eta - ETA_FLOOR, width=0.56, bottom=ETA_FLOOR,
            color=[c["col"] for c in CATS], edgecolor="white", linewidth=0.6,
            zorder=3)
    for x, e in zip(xs, eta):
        axb.text(x, e + 0.018, "%.2f V" % e, fontsize=6.3, color="#555555",
                 ha="center", va="bottom", zorder=4)
    axb.set_xticks(xs)
    axb.set_xticklabels([c["name"] for c in CATS], fontsize=6.6)
    axb.set_ylim(0.0, float(eta.max()) * 1.32)
    axb.set_xlabel("Overpotential split", fontsize=8, labelpad=1.5)
    axb.set_ylabel("$\\eta$ (V)", fontsize=8, labelpad=1.5)
    axb.tick_params(axis="y", labelsize=6.5)
    axb.tick_params(axis="x", length=0)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    axb.legend(handles=[Line2D([], [], color="#d5d5d5", lw=5.0,
                               label="floor set by scaling, %.2f V"
                                     % ETA_FLOOR)],
               loc="upper left", fontsize=6.3, handlelength=1.0,
               borderaxespad=0.25)

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.36, dy=0.10)

    savefig(fig, "763_cat_free_energy_diagram.png")


if __name__ == "__main__":
    main()
