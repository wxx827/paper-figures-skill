"""Gibbs free-energy diagram for a catalytic cycle on three metal surfaces.

(a) Stepwise profile along the reaction coordinate: horizontal level segments
    joined by dashed connectors, one path per catalyst plus the uncatalysed
    gas-phase route.  The rate-determining step of the best catalyst is marked
    with a heavy double arrow carrying its dG(ddagger).  A twin axis restates
    the same energies in kJ/mol.
(b) Auxiliary panel (55 % of the main width): per-elementary-step barriers for
    the three catalysts, every number read back out of the level arrays that
    panel (a) is drawn from.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, arrow

RNG = np.random.default_rng(70213)

EV_TO_KJ = 96.485                      # 1 eV -> kJ/mol

STATES = ["A(g) + B(g)", "A* + B*", "TS1$^\\ddagger$", "AB*",
          "TS2$^\\ddagger$", "P*", "P(g)"]
IS_TS = [False, False, True, False, True, False, False]

# Level energies in eV, one entry per state in STATES.
PATHS = [
    dict(name="Pt$_3$Ni (111)", col="#3b5b92", lw=2.0, z=6,
         e=[0.00, -0.62, 0.16, -0.95, -0.33, -1.42, -0.85]),
    dict(name="Pd (111)", col="#2e7d5b", lw=1.25, z=5,
         e=[0.00, -0.45, 0.52, -0.70, 0.18, -1.10, -0.85]),
    dict(name="Cu (111)", col="#c9a227", lw=1.25, z=4,
         e=[0.00, -0.28, 0.86, -0.40, 0.60, -1.02, -0.85]),
]
# The uncatalysed route only has reactant, gas-phase transition state, product.
BARE = dict(name="uncatalysed", col="#b23a48", lw=1.15, idx=[0, 3, 6],
            e=[0.00, 2.35, -0.85])

STEPS = [("TS1: C-H scission", 1, 2),
         ("TS2: C-O coupling", 3, 4),
         ("P* desorption", 5, 6)]

HALF = 0.30            # half width of a normal level segment, x units
HALF_TS = 0.20         # transition states get a shorter rule


def draw_path(ax, xs, energies, col, lw, z, ts_flags=None):
    """Level segments plus dashed connectors for one reaction path."""
    ts_flags = ts_flags if ts_flags is not None else [False] * len(xs)
    for i, (x, e) in enumerate(zip(xs, energies)):
        hw = HALF_TS if ts_flags[i] else HALF
        ax.plot([x - hw, x + hw], [e, e], color=col, lw=lw,
                solid_capstyle="butt", zorder=z + 1)
        if i:
            hw0 = HALF_TS if ts_flags[i - 1] else HALF
            ax.plot([xs[i - 1] + hw0, x - hw], [energies[i - 1], e],
                    color=col, lw=0.7, ls=(0, (2.6, 2.0)), alpha=0.75,
                    zorder=z)


def main():
    set_pub_style(base_font=9)

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax = axes_inches(fig, (0.80, 1.02, 5.00, 4.62))
    axb = axes_inches(fig, (7.32, 1.02, 2.82, 4.62))

    # ---------------- (a) free-energy profile -----------------------------
    xs = np.arange(len(STATES), dtype=float)
    ylo, yhi = -1.75, 2.85

    for x in xs:
        ax.plot([x, x], [ylo, yhi], color="#e2e2e2", lw=0.5, ls=(0, (1, 2.6)),
                zorder=0)
    ax.axhline(0.0, color="#cccccc", lw=0.6, zorder=1)

    draw_path(ax, xs[BARE["idx"]], BARE["e"], BARE["col"], BARE["lw"], 3,
              [False, True, False])
    for p in PATHS:
        draw_path(ax, xs, p["e"], p["col"], p["lw"], p["z"], IS_TS)

    best = PATHS[0]
    barriers = [best["e"][j] - best["e"][i] for _, i, j in STEPS]
    k_rds = int(np.argmax(barriers))
    i_rds, j_rds = STEPS[k_rds][1], STEPS[k_rds][2]
    x_rds = 0.5 * (xs[i_rds] + xs[j_rds]) + 0.10
    arrow(ax, (x_rds, best["e"][i_rds]), (x_rds, best["e"][j_rds]),
          color=best["col"], lw=1.9, ms=9, style="<|-|>", zorder=8)
    ax.text(x_rds + 0.13, 0.5 * (best["e"][i_rds] + best["e"][j_rds]),
            "$\\Delta G^\\ddagger$ = %.2f eV\n(%.0f kJ mol$^{-1}$), RDS"
            % (barriers[k_rds], barriers[k_rds] * EV_TO_KJ),
            fontsize=6.8, color=best["col"], ha="left", va="center",
            fontweight="bold", linespacing=1.35)

    e_bare = BARE["e"][1] - BARE["e"][0]
    x_bare = xs[BARE["idx"][1]] + 0.34
    arrow(ax, (x_bare, BARE["e"][0]), (x_bare, BARE["e"][1]),
          color=BARE["col"], lw=1.0, ms=8, style="<|-|>", zorder=8)
    ax.text(x_bare + 0.13, 0.62 * BARE["e"][1],
            "$\\Delta G^\\ddagger$ = %.2f eV\nwithout catalyst" % e_bare,
            fontsize=6.6, color=BARE["col"], ha="left", va="center",
            linespacing=1.35)

    d_rxn = best["e"][-1] - best["e"][0]
    ax.annotate("$\\Delta G_{rxn}$ = %+.2f eV" % d_rxn,
                xy=(xs[-1], best["e"][-1]), xytext=(xs[-1] + 0.42, -0.42),
                fontsize=6.6, color="#555555", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.6,
                                shrinkA=1.0, shrinkB=1.0))

    for x, s in zip(xs, STATES):
        ax.text(x, ylo + 0.06, s, fontsize=6.4, color="#555555",
                ha="center", va="bottom")

    ax.set_xlim(-0.55, len(STATES) - 0.45)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Gibbs free energy (eV)")
    ax.set_xlabel("Reaction coordinate (-)", labelpad=3)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.2)

    axk = ax.twinx()
    axk.set_ylim(ylo * EV_TO_KJ, yhi * EV_TO_KJ)
    axk.set_ylabel("Gibbs free energy (kJ mol$^{-1}$)", fontsize=8,
                   color="#555555", labelpad=2)
    axk.tick_params(axis="y", labelsize=7.0, colors="#555555")
    axk.spines["right"].set_visible(True)
    axk.spines["right"].set_color("#9a9a9a")
    axk.spines["top"].set_visible(False)
    axk.spines["bottom"].set_visible(False)

    handles = [Line2D([], [], color=p["col"], lw=p["lw"], label=p["name"])
               for p in PATHS]
    handles.append(Line2D([], [], color=BARE["col"], lw=BARE["lw"],
                          label=BARE["name"]))
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, ncol=2,
              handlelength=1.5, columnspacing=1.1, labelspacing=0.3,
              borderaxespad=0.35)

    # ---------------- (b) elementary-step barriers ------------------------
    n_p = len(PATHS)
    ypos = np.arange(len(STEPS), dtype=float)
    h = 0.24
    all_b = []
    for k, p in enumerate(PATHS):
        vals = np.array([p["e"][j] - p["e"][i] for _, i, j in STEPS])
        all_b.append(vals)
        off = (k - (n_p - 1) / 2.0) * h
        axb.barh(ypos + off, vals, height=h * 0.88, color=p["col"],
                 edgecolor="white", linewidth=0.5,
                 alpha=1.0 if k == 0 else 0.88, zorder=3)
        for y, v in zip(ypos + off, vals):
            axb.text(v + 0.018, y, "%.2f" % v, fontsize=6.2,
                     color="#555555", ha="left", va="center", zorder=4)

    b_max = float(np.max(np.concatenate(all_b)))
    axb.set_xlim(0.0, b_max * 1.30)
    axb.set_ylim(-0.55, len(STEPS) - 0.45)
    axb.set_yticks(ypos)
    axb.set_yticklabels([s[0] for s in STEPS], fontsize=6.6)
    axb.invert_yaxis()
    axb.set_xlabel("Elementary-step barrier (eV)", fontsize=8)
    axb.tick_params(axis="x", labelsize=6.8)
    axb.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    axb.axvline(np.max(all_b[0]), color=best["col"], lw=0.8,
                ls=(0, (3.0, 2.2)), zorder=2)
    axb.text(np.max(all_b[0]) + 0.02, len(STEPS) - 0.52,
             "RDS of %s" % best["name"], fontsize=6.2, color=best["col"],
             ha="left", va="bottom")

    panel_labels_inches(fig, [ax, axb], dx=-0.34, dy=0.10)

    savefig(fig, "702_cat_energy_diagram.png")


if __name__ == "__main__":
    main()
