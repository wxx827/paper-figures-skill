"""Taguchi main-effects plot for an L27 array: eight control factors on one
common signal-to-noise axis so that effect sizes stay comparable."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
GREY = "#8b949e"
AMBER = "#c9922f"
TINT = "#f4f8fb"

GRAND = 24.00

# code, name, unit, level settings, effect shape (mean 0, range 1), delta dB
FACTORS = [
    ("A", "Injection pressure", "bar", ("60", "80", "100"),
     (-0.55, 0.10, 0.45), 3.1),
    ("B", "Melt temperature", "\u00b0C", ("210", "230", "250"),
     (0.45, 0.05, -0.50), 2.4),
    ("C", "Mould temperature", "\u00b0C", ("40", "55", "70"),
     (-0.30, 0.55, -0.25), 0.6),
    ("D", "Holding time", "s", ("2", "4", "6"),
     (-0.50, 0.42, 0.08), 1.8),
    ("E", "Cooling time", "s", ("10", "15", "20"),
     (0.40, -0.50, 0.10), 0.9),
    ("F", "Screw speed", "rpm", ("60", "90", "120"),
     (0.10, -0.45, 0.35), 0.4),
    ("G", "Back pressure", "bar", ("5", "10", "15"),
     (-0.45, 0.05, 0.40), 1.3),
    ("H", "Gate size", "mm", ("0.8", "1.2", "1.6"),
     (0.50, -0.20, -0.30), 2.9),
]


def stamp_labels(fig, axes, labels, dx=0.030, dy=0.008, size=9):
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, lab, fontsize=size,
                 fontweight="bold", ha="left", va="bottom")


def main():
    set_pub_style()
    rng = np.random.default_rng(1123)

    means, deltas = [], []
    for _, _, _, _, shape, delta in FACTORS:
        m = GRAND + delta * np.array(shape) + rng.normal(0, 0.035, 3)
        m -= m.mean() - GRAND                      # keep the grand mean exact
        means.append(m)
        deltas.append(m.max() - m.min())
    ranks = {i: r + 1 for r, i in enumerate(np.argsort(deltas)[::-1])}

    allv = np.concatenate(means)
    pad = 0.22 * (allv.max() - allv.min())
    ylim = (allv.min() - pad, allv.max() + pad)    # one range for all panels

    fig = plt.figure(figsize=(7.9, 4.6))
    gs = fig.add_gridspec(2, 4, wspace=0.20, hspace=0.66, left=0.072,
                          right=0.986, top=0.775, bottom=0.135)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    best_levels = []
    for k, (ax, (code, name, unit, levels, _, _)) in enumerate(
            zip(axes, FACTORS)):
        m = means[k]
        rank = ranks[k]
        if rank <= 3:
            ax.set_facecolor(TINT)
        ax.axhline(GRAND, color=GREY, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
        x = np.arange(1, 4)
        ax.plot(x, m, "-", color=NAVY, lw=1.5, zorder=4)
        ax.plot(x, m, "o", ms=4.0, mfc="white", mec=NAVY, mew=1.1, zorder=5)
        b = int(np.argmax(m))
        best_levels.append(f"{code}{b + 1}")
        ax.plot([x[b]], [m[b]], "o", ms=5.4, color=AMBER, zorder=6)
        ax.annotate(f"{code}{b + 1}", (x[b], m[b]), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.4, fontweight="bold", color=AMBER, zorder=7)

        ax.set_xlim(0.55, 3.45)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=6.4)
        ax.set_title(f"{name} ({unit})", fontsize=7.2, fontweight="bold",
                     color=INK, pad=4)
        ax.tick_params(labelsize=6.6)
        ax.yaxis.grid(True, color="#e8ecf0", lw=0.5)
        ax.set_axisbelow(True)
        style_axes(ax)
        if k % 4 != 0:
            ax.tick_params(labelleft=False)
        ax.text(0.035, 0.045, f"$\\Delta$ {deltas[k]:.1f} dB \u00b7 rank "
                f"{rank}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.0, fontweight="bold" if rank <= 3 else "normal",
                color=NAVY if rank <= 3 else GREY, zorder=7)

    stamp_labels(fig, axes, [f[0] for f in FACTORS])
    for r in (0, 1):
        axes[4 * r].set_ylabel("S/N ratio (dB)", fontsize=8)

    predicted = GRAND + sum(means[k].max() - GRAND for k in range(len(FACTORS)))
    handles = [
        Line2D([], [], color=NAVY, marker="o", ms=4.0, mfc="white", mec=NAVY,
               lw=1.5, label="Mean S/N at level"),
        Line2D([], [], color=AMBER, marker="o", ms=5.4, ls="none",
               label="Selected (best) level"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (4, 2.5)),
               label=f"Grand mean {GRAND:.1f} dB"),
    ]


    savefig(fig, "249_taguchi_effects.png")


if __name__ == "__main__":
    main()
