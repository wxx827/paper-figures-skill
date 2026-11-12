"""Bathtub hazard curve: three failure regimes, their sum, and the
maintenance strategy that belongs to each phase."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
GREY = "#8b949e"
ALERT = "#b23a48"
AMBER = "#c9922f"

T0, T1, T2, T3 = 0.0, 2.3, 6.8, 10.0          # phase boundaries, kilohours
PHASES = [
    (T0, T1, "INFANT MORTALITY", AMBER, "#fdf5e8"),
    (T1, T2, "USEFUL LIFE", BLUE, "#eef3f8"),
    (T2, T3, "WEAR-OUT", ALERT, "#fbf1f2"),
]
STRATEGY = [
    ("Burn-in & screening",
     "100% burn-in, 48 h\nTightened incoming AQL\nSupplier 8D on returns"),
    ("Condition monitoring",
     "Vibration and oil trending\nRun-to-failure on low-cost items\n"
     "Min\u2013max spares policy"),
    ("Preventive replacement",
     "Age-based swap at 7.5 kh\nOverhaul kit at each stop\nRetire asset at 10 kh"),
]

A_EARLY, TAU = 1.90, 0.78
H_CONST = 0.42
T_WEAR, M_WEAR = 6.20, 3.40


def h_early(t):
    return A_EARLY * np.exp(-t / TAU)


def h_wear(t):
    return H_CONST * (t / T_WEAR) ** M_WEAR


def main():
    set_pub_style()
    t = np.linspace(0.0, T3, 800)
    total = h_early(t) + H_CONST + h_wear(t)

    fig = plt.figure(figsize=(7.6, 4.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.34], hspace=0.55,
                          left=0.078, right=0.982, top=0.815, bottom=0.065)
    ax = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])

    ymax = 3.05
    for x0, x1, name, col, bg in PHASES:
        ax.axvspan(x0, x1, color=bg, zorder=0)
        ax.text((x0 + x1) / 2, ymax * 0.965, name, ha="center", va="top",
                fontsize=7.4, fontweight="bold", color=col, zorder=6)
    for xb in (T1, T2):
        ax.axvline(xb, color="#c8ced4", lw=0.8, ls=(0, (3, 2.5)), zorder=1)

    ax.plot(t, h_early(t), color=AMBER, lw=1.2, ls=(0, (5, 2.5)), zorder=3)
    ax.plot(t, np.full_like(t, H_CONST), color=GREY, lw=1.2, ls=(0, (5, 2.5)),
            zorder=3)
    ax.plot(t, h_wear(t), color=ALERT, lw=1.2, ls=(0, (5, 2.5)), zorder=3)
    ax.plot(t, total, color=NAVY, lw=2.4, zorder=5)

    i = int(np.argmin(total))
    ax.plot([t[i]], [total[i]], "o", ms=5.0, mfc="white", mec=NAVY, mew=1.4,
            zorder=6)
    ax.annotate(f"minimum hazard {total[i]:.2f} fails / 1000 h\n"
                f"at {t[i]:.1f} kh of service",
                (t[i], total[i]), xytext=(14, 34), textcoords="offset points",
                ha="left", va="center", fontsize=6.5, color=NAVY,
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=NAVY,
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(T0, T3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Operating age (thousand hours)", fontsize=8)
    ax.set_ylabel("Hazard rate  $h(t)$  (failures / 1000 h)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(0, 11, 1))
    style_axes(ax)

    # ---- strategy strip ----------------------------------------------------
    ax_s.set_xlim(T0, T3)
    ax_s.set_ylim(0, 1)
    ax_s.axis("off")
    pad = 0.07
    for (x0, x1, _, col, bg), (title, body) in zip(PHASES, STRATEGY):
        ax_s.add_patch(FancyBboxPatch(
            (x0 + pad, 0.02), (x1 - x0) - 2 * pad, 0.96,
            boxstyle="round,pad=0,rounding_size=0.10", facecolor=bg,
            edgecolor=col, linewidth=0.8, zorder=2))
        ax_s.text(x0 + pad + 0.18, 0.84, title.upper(), ha="left", va="top",
                  fontsize=6.7, fontweight="bold", color=col, zorder=3)
        ax_s.text(x0 + pad + 0.18, 0.58, body, ha="left", va="top",
                  fontsize=6.1, color=INK, linespacing=1.5, zorder=3)
    ax_s.text(T0, 1.16, "MAINTENANCE STRATEGY BY PHASE", ha="left",
              va="bottom", fontsize=6.8, fontweight="bold", color=GREY)

    handles = [
        Line2D([], [], color=NAVY, lw=2.4, label="Total hazard rate"),
        Line2D([], [], color=AMBER, lw=1.2, ls=(0, (5, 2.5)),
               label="Early failures (decreasing)"),
        Line2D([], [], color=GREY, lw=1.2, ls=(0, (5, 2.5)),
               label="Random failures (constant)"),
        Line2D([], [], color=ALERT, lw=1.2, ls=(0, (5, 2.5)),
               label="Wear-out failures (increasing)"),
    ]


    savefig(fig, "248_bathtub_curve.png")


if __name__ == "__main__":
    main()
