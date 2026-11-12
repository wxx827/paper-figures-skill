"""Grade-recovery performance of three flotation reagent schemes.

The main panel is the chart every concentrator argues over: concentrate
grade against metal recovery, upper right being better, one curve per
reagent scheme.  Each point on a curve is a real two-product mass balance -
pick a concentrate yield, and the grade and the recovery follow from the
feed assay, so the theoretical maximum recovery line and the separation
efficiency contours behind the curves are the same arithmetic rather than
decoration.  The two smaller panels are the kinetic evidence underneath
that trade-off: first-order recovery against flotation time with its
limiting recovery, and the floatability spread of the individual minerals.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FEED_GRADE = 0.85          # % Cu in the flotation feed
MINERAL_GRADE = 34.6       # % Cu in pure chalcopyrite, the ceiling
SEED = 4471

# scheme name, colour, selectivity (higher is a steeper curve), max recovery
SCHEMES = [
    ("xanthate only", "xanthate", BLUE, 2.30, 0.925),
    ("xanthate + dithiophosphate", "X + DTP", GOLD, 3.05, 0.945),
    ("staged collector + depressant", "staged", RED, 4.40, 0.955),
]
# mineral, colour, rate constant (1/min), limiting recovery
MINERALS = [
    ("chalcopyrite", RED, 0.86, 0.96),
    ("bornite", GOLD, 0.61, 0.91),
    ("chalcocite", PURPLE, 0.44, 0.84),
    ("pyrite", GREEN, 0.17, 0.42),
    ("silicate gangue", GREY, 0.05, 0.13),
]
T_FLOAT = np.linspace(0.0, 16.0, 220)     # flotation time, minutes


def balance(yield_frac, recovery):
    """Two-product formula: concentrate grade (% Cu) from yield and recovery."""
    y = np.clip(np.asarray(yield_frac, float), 1e-4, 1.0)
    return FEED_GRADE * np.asarray(recovery, float) / y


def scheme_curve(selectivity, r_max, n=180):
    """Recovery against concentrate yield for one reagent scheme."""
    y = np.linspace(0.004, 0.30, n)
    y_rel = y / 0.30
    rec = r_max * (1.0 - np.exp(-selectivity * 6.0 * y_rel)) \
        / (1.0 - np.exp(-selectivity * 6.0))
    return y, rec


def separation_efficiency(grade, recovery):
    """Schulz efficiency: recovery of metal minus recovery of gangue."""
    grade = np.asarray(grade, float)
    recovery = np.asarray(recovery, float)
    yield_frac = FEED_GRADE * recovery / np.maximum(grade, 1e-9)
    gangue = ((yield_frac * (100.0 - grade))
              / max(100.0 - FEED_GRADE, 1e-9))
    return (recovery - gangue) * 100.0


def kinetics(k, r_inf, t=T_FLOAT, noise=0.0, rng=None):
    rec = r_inf * (1.0 - np.exp(-k * t))
    if noise and rng is not None:
        rec = rec + rng.normal(0.0, noise, t.size)
    return rec


def main_panel(ax):
    g_grid = np.linspace(1.0, 32.0, 240)
    r_grid = np.linspace(0.40, 1.00, 240)
    G, R = np.meshgrid(g_grid, r_grid)
    eff = separation_efficiency(G, R)
    cs = ax.contour(g_grid, r_grid * 100.0, eff,
                    levels=[50.0, 60.0, 70.0, 80.0, 88.0], colors=[HAIR],
                    linewidths=0.8, zorder=1)
    ax.clabel(cs, fmt="%.0f %%", fontsize=6.0, colors=[MUTED], inline=True)

    # Perfect liberation: all the metal can be recovered at any grade up to
    # the mineral's own grade, so the limit is an L, not a curve.
    ax.plot([FEED_GRADE, MINERAL_GRADE, MINERAL_GRADE],
            [100.0, 100.0, 40.0], color=GREY, lw=1.3, ls=(0, (5, 2.4)),
            zorder=2)
    ax.text(MINERAL_GRADE - 0.7, 44.0,
            f"theoretical limit, {MINERAL_GRADE:.1f} % Cu mineral",
            ha="left", va="bottom", fontsize=6.5, color=GREY,
            rotation=90.0, rotation_mode="anchor")

    best = None
    for name, short, color, sel, r_max in SCHEMES:
        y, rec = scheme_curve(sel, r_max)
        grade = balance(y, rec)
        keep = grade <= MINERAL_GRADE
        ax.plot(grade[keep], rec[keep] * 100.0, color=color, lw=1.9, zorder=5,
                label=name)
        eff_c = separation_efficiency(grade[keep], rec[keep])
        i_best = int(np.argmax(eff_c))
        ax.plot([grade[keep][i_best]], [rec[keep][i_best] * 100.0],
                marker="o", ms=5.2, color=color, mfc="white", mew=1.5,
                zorder=8)
        if best is None or eff_c[i_best] > best[0]:
            best = (eff_c[i_best], name, grade[keep][i_best],
                    rec[keep][i_best] * 100.0, y[keep][i_best] * 100.0)

        for target in (0.06, 0.14):
            j = int(np.argmin(np.abs(y[keep] - target)))
            ax.plot([grade[keep][j]], [rec[keep][j] * 100.0], marker="o",
                    ms=2.8, color=color, zorder=7)
            ax.text(grade[keep][j], rec[keep][j] * 100.0 - 1.8,
                    f"{y[keep][j] * 100:.0f} %", ha="center", va="top",
                    fontsize=6, color=color, zorder=7)

    ax.annotate(f"best separation efficiency {best[0]:.0f} %\n"
                f"{best[2]:.1f} % Cu at {best[3]:.1f} % recovery,  "
                f"yield {best[4]:.0f} %",
                xy=(best[2], best[3]), xytext=(21.5, 81.0), ha="center",
                va="center", fontsize=6.5, color=INK, zorder=9,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    ax.text(3.0, 44.0, "grey contours: separation efficiency\n"
            "small labels on the curves: concentrate yield",
            ha="left", va="bottom", fontsize=6.3, color=SUB, zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.94))

    ax.set_xlim(0.0, 36.5)
    ax.set_ylim(40.0, 101.0)
    ax.set_xlabel("concentrate grade (% Cu)")
    ax.set_ylabel("copper recovery (%)")
    ax.tick_params(labelsize=7.2)
    ax.grid(lw=0.45, alpha=0.5)
    ax.legend(fontsize=7.0, frameon=True, framealpha=0.93, loc="upper right",
              handlelength=1.8, borderaxespad=0.4, edgecolor="#cccccc",
              title=f"feed {FEED_GRADE:.2f} % Cu")
    ax.get_legend().get_title().set_fontsize(6.4)
    ax.get_legend().get_title().set_color(SUB)
    ax.set_title("Grade against recovery: up and to the right is better",
                 fontsize=9.0, fontweight="bold", color=INK)
    return best


def kinetic_panel(ax):
    rng = np.random.default_rng(SEED)
    for name, short, color, sel, r_max in SCHEMES:
        k = 0.16 * sel
        ax.plot(T_FLOAT, kinetics(k, r_max) * 100.0, color=color, lw=1.7,
                zorder=5,
                label=f"{short}:  " + r"$R_{\infty}$ = "
                      + f"{r_max * 100:.1f} %,  k = {k:.2f}"
                      + r" min$^{-1}$")
        t_obs = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0])
        ax.plot(t_obs, (kinetics(k, r_max, t_obs)
                        + rng.normal(0.0, 0.011, t_obs.size)) * 100.0,
                marker="o", ms=2.8, ls="none", color=color, alpha=0.7,
                zorder=6)
        ax.axhline(r_max * 100.0, color=color, lw=0.7, ls=(0, (2, 2.2)),
                   alpha=0.6, zorder=3)
        t_half = np.log(2.0) / k
        ax.plot([t_half], [kinetics(k, r_max, t_half) * 100.0], marker="s",
                ms=3.6, color=color, mfc="white", mew=1.1, zorder=7)

    ax.legend(fontsize=6.0, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.30)
    ax.set_xlim(0.0, T_FLOAT[-1])
    ax.set_ylim(0.0, 104.0)
    ax.set_xlabel("flotation time (min)", fontsize=7.4)
    ax.set_ylabel("copper recovery (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.55)
    ax.set_title(r"First-order kinetics, squares at $t_{1/2}$",
                 fontsize=8.2, fontweight="bold", color=INK)


def floatability_panel(ax):
    t_ref = 6.0
    names = [m[0] for m in MINERALS]
    rec = np.array([kinetics(m[2], m[3], np.array([t_ref]))[0]
                    for m in MINERALS]) * 100.0
    cols = [lighten(m[1], 0.30) for m in MINERALS]
    order = np.argsort(rec)[::-1]

    y = np.arange(len(names))[::-1]
    ax.barh(y, rec[order], height=0.60, color=[cols[i] for i in order],
            edgecolor="white", linewidth=0.7, zorder=3)
    for yi, i in zip(y, order):
        ax.text(rec[i] + 1.6, yi, f"{rec[i]:.0f} %  (k = {MINERALS[i][2]:.2f}"
                + r" min$^{-1}$)", ha="left", va="center", fontsize=6.1,
                color=INK, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.5)
    ax.set_xlim(0.0, 128.0)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xlabel(f"recovery after {t_ref:.0f} min (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Floatability spread drives the selectivity",
                 fontsize=8.2, fontweight="bold", color=INK)


def main():
    set_pub_style()

    fig = plt.figure(figsize=(10.6, 6.6))
    ax_main = axes_inches(fig, (0.88, 0.94, 5.36, 4.66))
    ax_kin = axes_inches(fig, (7.36, 3.64, 2.80, 1.96))
    ax_flo = axes_inches(fig, (7.36, 0.94, 2.80, 1.96))

    best = main_panel(ax_main)
    kinetic_panel(ax_kin)
    floatability_panel(ax_flo)

    panel_labels_inches(fig, [ax_main, ax_kin, ax_flo], dx=-0.54, dy=0.10)
    savefig(fig, "832_mnp_flotation_recovery.png")


if __name__ == "__main__":
    main()
