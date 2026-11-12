"""Effectiveness-NTU: what a capacity ratio costs, and what a shell cannot do.

Every curve on this sheet comes from the closed-form effectiveness relations,
evaluated on one NTU grid, so the three panels are three readings of the same
arrays.  The main panel is the counterflow family: raising the capacity ratio
from a condensing stream to a balanced pair drags the whole curve down, and the
marked design point is found by inverting the plotted Cr = 0.75 curve at the
target effectiveness rather than by quoting a number.  The upper side panel
fixes that capacity ratio and swaps the flow arrangement instead.  The lower one
takes each arrangement out to a very large NTU and reads the ceiling it converges
to: two of the four arrangements simply cannot reach the design effectiveness at
this capacity ratio, no matter how much surface is bought, which is the decision
the chart exists to make.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#134a76"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
EMBER = "#c0603f"
RED = "#b23a48"
GREY = "#8a8a8a"

SEED = 1039
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

CR_FAMILY = (0.00, 0.25, 0.50, 0.75, 1.00)
CR_DESIGN = 0.75
EPS_TARGET = 0.80
C_MIN = 12.0               # capacity rate of the limiting stream, kW/K
DT_INLET = 78.0            # inlet temperature difference, K
NTU_BIG = 80.0             # stand-in for "infinite surface"


def eps_counter(n, cr):
    n = np.asarray(n, dtype=float)
    if abs(cr - 1.0) < 1e-9:
        return n / (1.0 + n)
    e = np.exp(-n * (1.0 - cr))
    return (1.0 - e) / (1.0 - cr * e)


def eps_parallel(n, cr):
    n = np.asarray(n, dtype=float)
    return (1.0 - np.exp(-n * (1.0 + cr))) / (1.0 + cr)


def eps_crossflow(n, cr):
    """Both streams unmixed, the usual engineering approximation."""
    n = np.asarray(n, dtype=float)
    if cr < 1e-9:
        return 1.0 - np.exp(-n)
    return 1.0 - np.exp((n ** 0.22 / cr) * (np.exp(-cr * n ** 0.78) - 1.0))


def eps_shell(n, cr):
    """One shell pass, two (or any even number of) tube passes."""
    n = np.asarray(n, dtype=float)
    s = np.sqrt(1.0 + cr * cr)
    e = np.exp(-n * s)
    return 2.0 / (1.0 + cr + s * (1.0 + e) / (1.0 - e))


ARRANGEMENTS = (("counterflow", eps_counter, DEEP, "-"),
                ("crossflow, both unmixed", eps_crossflow, GREEN, "-"),
                ("1 shell pass, 2 tube passes", eps_shell, GOLD, "-"),
                ("parallel flow", eps_parallel, EMBER, (0, (4.4, 2.0))))


def family_panel(ax, ntu, ntu_design):
    cols = get_palette("cb_sequential_blue", 7)[2:7]
    handles = []
    for cr, col in zip(CR_FAMILY, cols):
        e = eps_counter(ntu, cr)
        ax.plot(ntu, e, color=col, lw=2.0, zorder=6)
        tag = {0.00: " (a condensing stream)",
               1.00: " (balanced streams)"}.get(cr, "")
        handles.append(Line2D([], [], color=col, lw=2.0,
                              label="Cr = %.2f%s" % (cr, tag)))
    e_par = eps_parallel(ntu, CR_DESIGN)
    ax.plot(ntu, e_par, color=EMBER, lw=1.3, ls=(0, (4.4, 2.0)), zorder=5)
    handles.append(Line2D([], [], color=EMBER, lw=1.3, ls=(0, (4.4, 2.0)),
                          label="parallel flow, Cr = %.2f" % CR_DESIGN))
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=2.0, borderaxespad=0.45,
                    labelspacing=0.30,
                    title="counterflow unless noted", title_fontsize=6.3)
    leg.set_zorder(12)

    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3.2, 2.0)), zorder=3)
    ax.text(0.08, 1.012, "thermodynamic ceiling, Q = Cmin(Th,in - Tc,in)",
            fontsize=6.3, color=SUB, ha="left", va="bottom", zorder=8)

    ax.plot([ntu_design, ntu_design], [0.0, EPS_TARGET], color=RED, lw=0.9,
            ls=(0, (2.2, 1.8)), zorder=5)
    ax.plot([0.0, ntu_design], [EPS_TARGET, EPS_TARGET], color=RED, lw=0.9,
            ls=(0, (2.2, 1.8)), zorder=5)
    ax.plot([ntu_design], [EPS_TARGET], marker="o", ms=5.4, mfc="white",
            mec=RED, mew=1.5, zorder=9)
    ax.annotate("design point: %.0f %% effectiveness at Cr = %.2f needs\n"
                "NTU = %.2f, so UA = %.1f kW/K on a %.1f kW/K stream"
                % (100.0 * EPS_TARGET, CR_DESIGN, ntu_design,
                   ntu_design * C_MIN, C_MIN),
                xy=(ntu_design, EPS_TARGET),
                xytext=(ntu_design + 0.30, 0.545), fontsize=6.6, color=RED,
                ha="left", va="top", linespacing=1.34, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=3,
                                shrinkB=4))

    for n_mark in (1.0, 2.0, 3.0, 4.0):
        e = float(eps_counter(n_mark, CR_DESIGN))
        ax.plot([n_mark], [e], marker="s", ms=3.0, mfc="white", mec=DEEP,
                mew=0.8, zorder=9)
    e1 = float(eps_counter(1.0, CR_DESIGN))
    e4 = float(eps_counter(4.0, CR_DESIGN))
    ax.text(0.12, 0.998, "the squares walk up Cr = %.2f one NTU at a "
            "time:\nthe first buys %.0f points of effectiveness and\nthe "
            "fourth buys %.0f, so surface has diminishing\nreturns and the "
            "target sits below the knee"
            % (CR_DESIGN, 100.0 * e1,
               100.0 * (e4 - eps_counter(3.0, CR_DESIGN))),
            fontsize=6.5, color=DEEP, ha="left", va="top", linespacing=1.34,
            zorder=10)

    ax.set_xlim(0.0, ntu[-1])
    ax.set_ylim(0.0, 1.09)
    ax.set_xlabel("number of transfer units NTU = UA / Cmin (dimensionless)")
    ax.set_ylabel("effectiveness " + r"$\varepsilon$" + " = Q / Qmax "
                  "(dimensionless)")
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Counterflow, swept over the capacity ratio Cr = Cmin / Cmax",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def arrangement_panel(ax, ntu):
    for name, fn, col, ls in ARRANGEMENTS:
        ax.plot(ntu, fn(ntu, CR_DESIGN), color=col, lw=1.6, ls=ls, zorder=6)
    ax.axhline(EPS_TARGET, color=RED, lw=0.9, ls=(0, (2.2, 1.8)), zorder=4)
    ax.text(0.08, EPS_TARGET + 0.018, "target %.2f" % EPS_TARGET,
            fontsize=6.1, color=RED, ha="left", va="bottom", zorder=8)
    labels = [("counterflow", DEEP, 0.960), ("crossflow", GREEN, 0.855),
              ("shell and tube", GOLD, 0.615), ("parallel", EMBER, 0.505)]
    for name, col, y in labels:
        ax.text(5.92, y, name, fontsize=6.2, color=col, ha="right",
                va="center", fontweight="bold", zorder=8)
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("NTU (dimensionless)", fontsize=6.8)
    ax.set_ylabel(r"$\varepsilon$" + " (dimensionless)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.40)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Four arrangements at the design Cr = %.2f" % CR_DESIGN,
                 fontsize=8.0, fontweight="bold", color=INK)


def ceiling_panel(ax):
    names, vals, cols = [], [], []
    for name, fn, col, _ in ARRANGEMENTS:
        names.append(name)
        vals.append(float(fn(NTU_BIG, CR_DESIGN)))
        cols.append(col)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.56, color=cols, edgecolor="white", linewidth=0.7,
            zorder=5)
    ax.axvline(EPS_TARGET, color=RED, lw=1.0, ls=(0, (3.0, 2.0)), zorder=7)
    ax.text(EPS_TARGET - 0.015, 3.62, "design target %.2f" % EPS_TARGET,
            fontsize=6.1, color=RED, ha="right", va="top", zorder=8)
    for yy, v, name in zip(y, vals, names):
        ok = v >= EPS_TARGET
        ax.text(0.022, yy + 0.38,
                name if ok else name + "  (cannot reach the target)",
                fontsize=6.2, color=INK if ok else EMBER, ha="left",
                va="bottom", zorder=8)
        ax.text(v - 0.018, yy, "%.3f" % v, fontsize=6.3, color="white",
                ha="right", va="center", fontweight="bold", zorder=8)
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(-0.55, 3.95)
    ax.set_yticks([])
    ax.set_xlabel("effectiveness ceiling as NTU grows without limit "
                  "(dimensionless)", fontsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.40)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Surface cannot buy what the arrangement forbids",
                 fontsize=8.0, fontweight="bold", color=INK)
    return dict(zip(names, vals))


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    np.random.default_rng(SEED)

    ntu = np.linspace(0.0, 6.0, 601)
    e_design = eps_counter(ntu, CR_DESIGN)
    ntu_design = float(np.interp(EPS_TARGET, e_design, ntu))

    assert np.all(np.diff(e_design) > 0), "effectiveness must rise with NTU"
    assert abs(float(eps_counter(ntu_design, CR_DESIGN)) - EPS_TARGET) < 1e-4, \
        "the design NTU has to be the inverse of the plotted curve"
    for cr in CR_FAMILY[1:]:
        assert np.all(eps_counter(ntu[1:], cr) <=
                      eps_counter(ntu[1:], 0.0) + 1e-9), \
            "a condensing stream must dominate every finite capacity ratio"
    assert np.all(eps_counter(ntu[1:], CR_DESIGN) >
                  eps_parallel(ntu[1:], CR_DESIGN)), \
        "counterflow must beat parallel flow at every NTU"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_arr = axes_inches(fig, AUX_T)
    ax_cap = axes_inches(fig, AUX_B)

    family_panel(ax_main, ntu, ntu_design)
    arrangement_panel(ax_arr, ntu)
    ceil = ceiling_panel(ax_cap)

    ntu_cross = float(np.interp(EPS_TARGET, eps_crossflow(ntu, CR_DESIGN),
                                ntu))
    duty = EPS_TARGET * C_MIN * DT_INLET



    panel_labels_inches(fig, [ax_main, ax_arr, ax_cap], dx=-0.62, dy=0.14)
    savefig(fig, "1039_ht_effectiveness_ntu.png")


if __name__ == "__main__":
    main()
