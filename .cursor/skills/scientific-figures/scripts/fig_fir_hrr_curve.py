"""Design fire heat release rate curves for four t-squared growth rates.

The main panel carries the shape every performance-based fire design starts
from: a t-squared growth phase, a ventilation-limited steady burning plateau,
and a linear decay once most of the fuel is gone.  Four standard growth
coefficients are drawn on one pair of axes so the only thing that changes
between them is how fast the fire gets to the two decisions that matter -
sprinkler activation and flashover.  Both thresholds are horizontal reference
lines, and the instant each curve crosses them is marked on the curve itself.

Every number is derived, not typed.  The peak is the ventilation limit
1500 A_v sqrt(h_v); flashover uses Thomas' correlation 7.8 A_t + 378 A_v
sqrt(h_v); sprinkler activation inverts Alpert's ceiling-jet correlation for a
68 C link; the time to 1 MW is sqrt(1000 / alpha); the plateau length and the
burnout time follow from integrating the curve against the available fire
load.  Illustrative design fires, not measurements.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
BASE = "#808080"

# Warm house ramp, light to dark: slow fires read light, ultra-fast reads dark.
GROWTH = [
    dict(name="slow", alpha=0.00293, color="#e69a54"),
    dict(name="medium", alpha=0.01172, color="#cc722f"),
    dict(name="fast", alpha=0.04690, color="#a04c18"),
    dict(name="ultra-fast", alpha=0.18760, color="#66300c"),
]

# Compartment geometry (m) - one office-sized room with a single doorway.
ROOM_L, ROOM_W, ROOM_H = 5.0, 4.0, 3.0
OPEN_W, OPEN_H = 2.0, 2.0
FIRE_LOAD = 780.0            # MJ/m2, office occupancy
FLOOR_AREA = ROOM_L * ROOM_W
# The window runs past the slowest burnout so the right margin stays clear for
# the threshold tags; every fire is out well before it.
T_END = 4200.0               # s
Y_TOP = 11.6                 # MW; the band above the plateau holds the legend

T_AMB = 20.0                 # deg C
LINK_T = 68.0                # deg C, standard sprinkler link
DET_RADIUS = 2.5             # m, radial distance from plume axis


def ventilation_limit():
    """Ventilation-controlled peak HRR, 1500 A_v sqrt(h_v) in kW."""
    return 1500.0 * (OPEN_W * OPEN_H) * np.sqrt(OPEN_H)


def flashover_hrr():
    """Thomas' flashover correlation, kW."""
    a_v = OPEN_W * OPEN_H
    a_t = (2.0 * ROOM_L * ROOM_W + 2.0 * ROOM_L * ROOM_H
           + 2.0 * ROOM_W * ROOM_H - a_v)
    return 7.8 * a_t + 378.0 * a_v * np.sqrt(OPEN_H)


def sprinkler_hrr():
    """Invert Alpert's ceiling-jet correlation for a 68 C link, kW."""
    d_t = LINK_T - T_AMB
    return ((d_t * ROOM_H) / 5.38) ** 1.5 * DET_RADIUS


def hrr_profile(t, alpha, q_peak, e_total, decay_frac=0.70):
    """Growth / plateau / decay HRR in kW, with the phase breakpoints."""
    t_grow = np.sqrt(q_peak / alpha)
    e_grow = alpha * t_grow ** 3 / 3.0 / 1000.0          # MJ
    t_plateau = max((decay_frac * e_total - e_grow) * 1000.0 / q_peak, 0.0)
    t_2 = t_grow + t_plateau
    t_dec = 2.0 * (1.0 - decay_frac) * e_total * 1000.0 / q_peak
    q = alpha * np.square(np.minimum(t, t_grow))
    q = np.where(t > t_grow, q_peak, q)
    q = np.where(t > t_2, q_peak * np.clip(1.0 - (t - t_2) / t_dec, 0.0, 1.0),
                 q)
    return q, t_grow, t_2, t_2 + t_dec


def cumulative(t, q):
    """Trapezoidal integral of an HRR curve in kW, returned in GJ."""
    step = np.diff(t) * 0.5 * (q[1:] + q[:-1])
    return np.concatenate([[0.0], np.cumsum(step)]) / 1.0e6


def right_tag(ax, y, text, color):
    """Short tag parked in the clear margin to the right of every curve."""
    ax.annotate(text, xy=(T_END - 60.0, y), xytext=(0, 2.5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.5, color=color)


def main_panel(ax, t, curves, q_peak, q_fo, q_spr):
    ax.axhspan(q_fo / 1000.0, q_peak / 1000.0, color="#f5f5f5", zorder=0)
    l_fo = ax.axhline(q_fo / 1000.0, color=RED, lw=1.1, ls=(0, (4.5, 2.2)),
                      zorder=4)
    l_sp = ax.axhline(q_spr / 1000.0, color=BLUE, lw=1.1, ls=(0, (2.4, 1.8)),
                      zorder=4)
    ax.axhline(q_peak / 1000.0, color=BASE, lw=0.8, ls=(0, (1.2, 1.8)),
               zorder=3)
    ax.axhline(1.0, color="#cccccc", lw=0.7, zorder=2)

    handles, labels = [], []
    for g in curves:
        ln, = ax.plot(t, g["q"] / 1000.0, color=g["color"], lw=1.9, zorder=6)
        for thr, col, mk in ((q_spr, BLUE, "s"), (q_fo, RED, "o")):
            t_x = np.sqrt(thr / g["alpha"])
            ax.plot([t_x], [thr / 1000.0], marker=mk, ms=4.2, mfc="white",
                    mec=col, mew=1.1, zorder=8)
        t_1mw = np.sqrt(1000.0 / g["alpha"])
        ax.plot([t_1mw], [1.0], marker="D", ms=3.6, mfc=g["color"],
                mec="white", mew=0.7, zorder=8)
        handles.append(ln)
        labels.append(f"{g['name']}: 1 MW at {t_1mw:.0f} s, "
                      f"burnout {g['t_out'] / 60.0:.0f} min")

    right_tag(ax, q_fo / 1000.0, f"flashover, {q_fo / 1000.0:.2f} MW", RED)
    right_tag(ax, q_spr / 1000.0, f"sprinkler, {q_spr / 1000.0:.2f} MW", BLUE)
    right_tag(ax, 1.0, "1 MW gauge", MUTED)
    ax.annotate(f"ventilation limit {q_peak / 1000.0:.2f} MW",
                xy=(T_END - 60.0, q_peak / 1000.0), xytext=(0, -9.0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.5, color=SUB)
    # Phase captions live in the empty band above the plateau, clear of curves.
    for x_c, cap in ((430.0, "growth ($\\alpha t^{2}$)"),
                     (2000.0, "steady burning (ventilation limited)"),
                     (3450.0, "decay")):
        ax.annotate(cap, xy=(x_c, 9.25), ha="center", va="center",
                    fontsize=6.6, color=SUB)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_xlabel("time from ignition (s)", fontsize=8.0)
    ax.set_ylabel("heat release rate (MW)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    ax.set_xticks(np.arange(0.0, T_END + 1.0, 600.0))
    ax.set_yticks(np.arange(0.0, 9.0, 2.0))
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Growth coefficient decides whether sprinklers act before "
                 "flashover", fontsize=9.0, fontweight="bold", color=INK)

    handles += [l_sp, l_fo,
                plt.Line2D([], [], marker="s", ls="none", ms=3.6, mfc="white",
                           mec=BLUE, mew=1.0),
                plt.Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white",
                           mec=RED, mew=1.0)]
    labels += ["sprinkler threshold", "flashover threshold",
               "sprinkler crossing", "flashover crossing"]
    # The plateau is the ventilation limit, so everything above it is empty:
    # anchoring the legend there cannot collide with any curve.
    leg = ax.legend(handles, labels, fontsize=6.7, frameon=False, ncol=2,
                    loc="upper left", bbox_to_anchor=(0.0, 1.0),
                    handlelength=1.9, columnspacing=1.5, labelspacing=0.34)
    leg.set_zorder(10)


def energy_panel(ax, t, curves, e_total):
    for g in curves:
        ax.plot(t, g["cum"], color=g["color"], lw=1.6, zorder=5)
        ax.plot([g["t_out"]], [g["cum"][-1]], marker="o", ms=3.4,
                mfc="white", mec=g["color"], mew=1.0, zorder=6)
    ax.axhline(e_total / 1000.0, color=BASE, lw=1.0, ls=(0, (4, 2.4)),
               zorder=4)
    ax.text(0.03, e_total / 1000.0 * 1.02, f"available fire load "
                                           f"{e_total / 1000.0:.1f} GJ",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6.3, color=SUB)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, e_total / 1000.0 * 1.28)
    ax.set_xticks(np.arange(0.0, T_END + 1.0, 1400.0))
    ax.set_xlabel("time from ignition (s)", fontsize=7.6)
    ax.set_ylabel("cumulative heat released (GJ)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("All four release the same total energy", fontsize=8.4,
                 fontweight="bold", color=INK)


def duration_panel(ax, t, q_peak, densities):
    alpha = GROWTH[2]["alpha"]
    cols = ["#fbe0bf", "#f4c088", "#e69a54", "#cc722f", "#a04c18"]
    durs = []
    for d in densities:
        _, _, _, t_out = hrr_profile(t, alpha, q_peak, d * FLOOR_AREA)
        durs.append(t_out / 60.0)
    durs = np.array(durs)
    x = np.arange(durs.size)
    ax.bar(x, durs, width=0.62, color=cols[:durs.size], edgecolor=RULE,
           lw=0.55, zorder=3)
    for xi, v in zip(x, durs):
        ax.text(xi, v + 0.02 * durs.max(), f"{v:.0f}", ha="center",
                va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}" for d in densities], fontsize=6.8)
    ax.set_ylim(0.0, durs.max() * 1.22)
    ax.set_xlabel("fire load density (MJ/m$^{2}$)", fontsize=7.6)
    ax.set_ylabel("burnout duration (min)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    slope = (durs[-1] - durs[0]) / (densities[-1] - densities[0])
    ax.text(0.5, 0.965, f"fast fire, {FLOOR_AREA:.0f} m$^{{2}}$ floor: "
                        f"{slope * 100.0:.1f} min per 100 MJ/m$^{{2}}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.4,
            color=SUB)
    ax.set_title("Duration scales with the fuel, not the growth rate",
                 fontsize=8.4, fontweight="bold", color=INK)
    return durs


def main():
    set_pub_style()
    rng = np.random.default_rng(85201)
    t = np.linspace(0.0, T_END, 2401)

    q_peak = ventilation_limit()
    q_fo = flashover_hrr()
    q_spr = sprinkler_hrr()
    e_total = FIRE_LOAD * FLOOR_AREA * (1.0 + 0.02 * rng.normal())

    curves = []
    for g in GROWTH:
        q, t_g, t_2, t_out = hrr_profile(t, g["alpha"], q_peak, e_total)
        curves.append(dict(g, q=q, cum=cumulative(t, q), t_grow=t_g,
                           t_steady=t_2, t_out=t_out))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.86, 5.00, 4.52))
    ax_cum = axes_inches(fig, (7.22, 3.40, 2.76, 1.98))
    ax_dur = axes_inches(fig, (7.22, 0.86, 2.76, 1.98))

    main_panel(ax_main, t, curves, q_peak, q_fo, q_spr)
    energy_panel(ax_cum, t, curves, e_total)
    densities = np.array([400.0, 600.0, 800.0, 1000.0, 1200.0])
    durs = duration_panel(ax_dur, t, q_peak, densities)

    t_fo = [np.sqrt(q_fo / g["alpha"]) for g in curves]
    t_sp = [np.sqrt(q_spr / g["alpha"]) for g in curves]
    alpha_txt = ", ".join("%.4f" % g["alpha"] for g in GROWTH)
    panel_labels_inches(fig, [ax_main, ax_cum, ax_dur], dx=-0.52, dy=0.14)
    savefig(fig, "852_fir_hrr_curve.png")


if __name__ == "__main__":
    main()
