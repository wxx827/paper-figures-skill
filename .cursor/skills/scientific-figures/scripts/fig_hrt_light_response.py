"""Leaf light-response curves: every quoted threshold is solved, not typed.

Four ambient CO2 concentrations drive one non-rectangular hyperbola each, so
the whole figure comes out of the same four parameter sets.  The light
compensation point is a root of the plotted curve found by bisection, the
saturation point is the root of "net assimilation has reached 90 % of its
asymptote", the apparent quantum yield is the ordinary least-squares slope of
the low-light limb of the same curve, and dark respiration is the curve
evaluated at zero light.  The upper side panel is the low-light limb enlarged
with those regressions drawn over it; the lower one collects the derived
parameters against CO2 so the reader can see that enrichment lifts the plateau
far more than it lifts the initial slope.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
GOLD = "#c9a227"
BLUE = "#3b5b92"

UNIT_A = r"$\mu mol\ m^{-2} s^{-1}$"

SEED = 1032
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

I_MAX = 2000.0                     # widest PPFD on the main axis
LOW_LIGHT = 100.0                  # limb used for the quantum-yield fit
SAT_FRAC = 0.90                    # definition of the light saturation point

# CO2 (ppm), apparent quantum yield seed, asymptote, convexity, dark
# respiration, colour.  The greens run light to dark with enrichment.
TREAT = [
    (400.0, 0.052, 22.0, 0.72, 1.40, "#9ccfae"),
    (600.0, 0.058, 28.5, 0.74, 1.42, "#6bba8e"),
    (800.0, 0.062, 33.0, 0.76, 1.44, "#2e7d5b"),
    (1000.0, 0.064, 35.5, 0.78, 1.46, "#0b3d2e"),
]

MEAS = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0,
                 700.0, 1000.0, 1400.0, 1800.0, 2000.0])


def a_net(ppfd, phi, amax, theta, rd):
    """Non-rectangular hyperbola, net of dark respiration."""
    ppfd = np.asarray(ppfd, dtype=float)
    b = phi * ppfd + amax
    disc = np.maximum(b * b - 4.0 * theta * phi * ppfd * amax, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * theta) - rd


def derive(phi, amax, theta, rd):
    """Compensation point, saturation point, fitted quantum yield, plateau."""
    f = lambda x: float(a_net(x, phi, amax, theta, rd))
    lcp = brentq(f, 1e-9, 400.0, xtol=1e-12, rtol=1e-14)
    a_top = f(I_MAX)
    lsp = brentq(lambda x: f(x) - SAT_FRAC * a_top, lcp + 1.0, I_MAX,
                 xtol=1e-12, rtol=1e-14)
    grid = np.linspace(0.0, LOW_LIGHT, 60)
    slope, intercept = np.polyfit(grid, a_net(grid, phi, amax, theta, rd), 1)
    return lcp, lsp, slope, intercept, a_top


def measurements(rng, phi, amax, theta, rd):
    """Gas-exchange points scattered about the same curve."""
    y = a_net(MEAS, phi, amax, theta, rd)
    return y + rng.normal(0.0, 0.42, MEAS.size)


def main_panel(ax, rows):
    ax.axhline(0.0, color="#9a9a9a", lw=0.8, zorder=2)
    ppfd = np.linspace(0.0, I_MAX, 900)

    for (co2, phi, amax, theta, rd, col), d in rows:
        curve = a_net(ppfd, phi, amax, theta, rd)
        ax.plot(ppfd, curve, color=col, lw=1.7, zorder=6)
        ax.plot(MEAS, d["pts"], ls="none", marker="o", ms=3.1, mfc="white",
                mec=col, mew=0.9, zorder=7)
        ax.plot([d["lcp"]], [0.0], marker="o", ms=5.0, mfc="white", mec=RED,
                mew=1.2, zorder=9)
        ax.plot([d["lsp"]], [d["a_lsp"]], marker="^", ms=4.6, mfc=GOLD,
                mec="white", mew=0.8, zorder=9)
        ax.plot([d["lsp"], d["lsp"]], [-4.0, d["a_lsp"]], color=GOLD, lw=0.7,
                ls=(0, (3.0, 2.2)), alpha=0.75, zorder=4)
        ax.text(I_MAX - 30.0, d["a_top"] + 0.35,
                "{:.0f} ppm  $A_{{max}}$ {:.1f}".format(co2, d["a_top"]),
                ha="right", va="bottom", fontsize=6.4, color=col,
                fontweight="bold", zorder=8)

    ref = rows[-1]
    (co2, phi, amax, theta, rd, col), d = ref
    tang = np.linspace(0.0, 430.0, 40)
    ax.plot(tang, d["slope"] * tang + d["intercept"], color=BLUE, lw=1.0,
            ls=(0, (5.0, 2.6)), zorder=5)
    ax.annotate("initial slope {:.3f} mol/mol\n(fitted over 0-{:.0f} "
                "{:s})".format(d["slope"], LOW_LIGHT, UNIT_A),
                xy=(410.0, d["slope"] * 410.0 + d["intercept"]),
                xytext=(560.0, 12.0), fontsize=6.4, color=BLUE, ha="left",
                va="top", linespacing=1.34, zorder=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.75,
                                shrinkA=2, shrinkB=3))

    lo, hi = rows[0][1], rows[-1][1]
    ax.annotate("light compensation point\n{:.0f} to {:.0f} {:s}, net "
                "exchange exactly zero".format(lo["lcp"], hi["lcp"], UNIT_A),
                xy=(hi["lcp"], 0.0), xytext=(150.0, -6.4), fontsize=6.5,
                color=RED, ha="left", va="center", linespacing=1.34,
                zorder=10, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                           shrinkA=2, shrinkB=4))
    ax.text(I_MAX + 20.0, -6.4,
            "light saturation point\n{:.0f} % of the plateau"
            .format(100.0 * SAT_FRAC), ha="right", va="center", fontsize=6.4,
            color="#8a6f1f", linespacing=1.34, zorder=10)

    handles = [Line2D([], [], color=c, lw=1.7,
                      label="{:.0f} ppm CO$_2$".format(t))
               for (t, _p, _a, _th, _r, c) in TREAT]
    handles += [
        Line2D([], [], color="#9a9a9a", ls="none", marker="o", ms=3.1,
               mfc="white", mec="#6b6b6b", label="gas-exchange points"),
        Line2D([], [], color=RED, ls="none", marker="o", ms=5.0, mfc="white",
               mec=RED, label="compensation point"),
        Line2D([], [], color=GOLD, ls="none", marker="^", ms=4.6, mfc=GOLD,
               mec="white", label="saturation point"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.6, ncol=2, columnspacing=0.9,
              borderaxespad=0.35, labelspacing=0.26)

    ax.set_xlim(-40.0, I_MAX + 40.0)
    ax.set_ylim(-8.0, 44.0)
    ax.set_xlabel("photosynthetic photon flux density, PPFD (" + UNIT_A + ")")
    ax.set_ylabel("net CO$_2$ assimilation, $A$ (" + UNIT_A + ")")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cucumber leaf light response at four CO$_2$ levels",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def limb_panel(ax, rows):
    ax.axhline(0.0, color="#9a9a9a", lw=0.8, zorder=2)
    ppfd = np.linspace(0.0, 220.0, 300)
    for (co2, phi, amax, theta, rd, col), d in rows:
        ax.plot(ppfd, a_net(ppfd, phi, amax, theta, rd), color=col, lw=1.5,
                zorder=6)
        ax.plot(ppfd, d["slope"] * ppfd + d["intercept"], color=col, lw=0.75,
                ls=(0, (4.0, 2.4)), alpha=0.85, zorder=5)
        ax.plot([d["lcp"]], [0.0], marker="o", ms=4.0, mfc="white", mec=RED,
                mew=1.0, zorder=8)
    ax.axvspan(0.0, LOW_LIGHT, color="#eef4ef", zorder=1)
    ax.text(LOW_LIGHT - 4.0, -1.05, "fit window", ha="right", va="bottom",
            fontsize=6.0, color=SUB, zorder=7)

    rd_lo = rows[0][1]["rd"]
    rd_hi = rows[-1][1]["rd"]
    ax.text(214.0, -2.55, "dark respiration {:.2f} to {:.2f}\n{:s}"
            .format(rd_lo, rd_hi, UNIT_A), ha="right", va="bottom",
            fontsize=6.0, color=SUB, linespacing=1.32, zorder=7)

    ax.set_xlim(-6.0, 220.0)
    ax.set_ylim(-2.8, 9.0)
    ax.set_xlabel("PPFD (" + UNIT_A + ")", fontsize=6.8)
    ax.set_ylabel("$A$ (" + UNIT_A + ")", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Low-light limb the quantum yield is fitted on",
                 fontsize=8.0, fontweight="bold", color=INK)


def parameter_panel(ax, rows):
    co2 = np.array([t[0] for (t, _d) in rows])
    lcp = np.array([d["lcp"] for (_t, d) in rows])
    lsp = np.array([d["lsp"] for (_t, d) in rows])
    aqy = np.array([d["slope"] for (_t, d) in rows])

    x = np.arange(len(rows), dtype=float)
    ax.bar(x - 0.19, lcp, width=0.36, color="#c8dfd0", edgecolor="white",
           linewidth=0.7, zorder=4, label="compensation point")
    ax.bar(x + 0.19, lsp / 10.0, width=0.36, color="#2e7d5b",
           edgecolor="white", linewidth=0.7, zorder=4,
           label="saturation point / 10")
    for xi, v in zip(x - 0.19, lcp):
        ax.text(xi, v + 1.4, "{:.0f}".format(v), ha="center", va="bottom",
                fontsize=6, color=SUB, zorder=6)
    for xi, v in zip(x + 0.19, lsp / 10.0):
        ax.text(xi, v + 1.4, "{:.0f}".format(v * 10.0), ha="center",
                va="bottom", fontsize=6, color="#1f5d43", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(["{:.0f}".format(c) for c in co2], fontsize=6.4)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylim(0.0, max(lsp.max() / 10.0, lcp.max()) * 1.42)
    ax.set_xlabel("ambient CO$_2$ (ppm)", fontsize=6.8)
    ax.set_ylabel("PPFD (" + UNIT_A + ")", fontsize=6.6)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    axr = ax.twinx()
    axr.plot(x, aqy, color=GOLD, lw=1.5, marker="s", ms=4.0, mfc="white",
             mec=GOLD, mew=1.1, zorder=8)
    axr.set_ylim(0.040, 0.074)
    axr.set_ylabel("apparent quantum yield (mol/mol)", fontsize=6.6,
                   color="#8a6f1f")
    axr.tick_params(axis="y", labelsize=6.2, colors="#8a6f1f")
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color("#c9a227")
    axr.text(x[-1], aqy[-1] + 0.0026, "{:.3f}".format(aqy[-1]), ha="right",
             va="bottom", fontsize=6.0, color="#8a6f1f", zorder=9)

    ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.94, handlelength=1.2,
              borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Thresholds and quantum yield against enrichment",
                 fontsize=8.0, fontweight="bold", color=INK)
    return aqy


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
    rng = np.random.default_rng(SEED)

    rows = []
    for t in TREAT:
        co2, phi, amax, theta, rd, col = t
        lcp, lsp, slope, intercept, a_top = derive(phi, amax, theta, rd)
        d = dict(lcp=lcp, lsp=lsp, slope=slope, intercept=intercept,
                 a_top=a_top, rd=rd, a_lsp=float(a_net(lsp, phi, amax, theta,
                                                       rd)),
                 pts=measurements(rng, phi, amax, theta, rd))
        assert abs(float(a_net(lcp, phi, amax, theta, rd))) < 1e-9, \
            "net assimilation must be exactly zero at the compensation point"
        assert abs(float(a_net(0.0, phi, amax, theta, rd)) + rd) < 1e-12, \
            "the curve at zero light must be minus dark respiration"
        rows.append((t, d))

    lcps = np.array([d["lcp"] for (_t, d) in rows])
    assert np.all(np.diff(lcps) < 0.0), "enrichment must lower the LCP"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_limb = axes_inches(fig, AUX_T)
    ax_par = axes_inches(fig, AUX_B)

    main_panel(ax_main, rows)
    limb_panel(ax_limb, rows)
    aqy = parameter_panel(ax_par, rows)

    lo, hi = rows[0][1], rows[-1][1]
    gain_plateau = 100.0 * (hi["a_top"] / lo["a_top"] - 1.0)
    gain_slope = 100.0 * (aqy[-1] / aqy[0] - 1.0)


    panel_labels_inches(fig, [ax_main, ax_limb, ax_par], dx=-0.62, dy=0.14)
    savefig(fig, "1032_hrt_light_response.png")


if __name__ == "__main__":
    main()
