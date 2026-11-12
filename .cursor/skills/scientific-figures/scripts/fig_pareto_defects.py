"""Defect Pareto chart: ranked counts, cumulative share and the vital-few cut."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
PALE = "#c3d3e2"
GREY = "#8b949e"
ALERT = "#b23a48"
AMBER = "#c9922f"
VITAL_BG = "#f2f6fa"

DEFECTS = [
    ("Solder\nvoid", 268),
    ("Component\nmisalign.", 191),
    ("Surface\nscratch", 142),
    ("Dimensional\noversize", 96),
    ("Contami-\nnation", 71),
    ("Cold\njoint", 54),
    ("Burr /\nflash", 38),
    ("Discolour-\nation", 27),
    ("Cracked\nhousing", 18),
    ("Other", 11),
]
UNITS_INSPECTED = 12480
OPPORTUNITIES = 6


def main():
    set_pub_style()
    names = [d[0] for d in DEFECTS]
    counts = np.array([d[1] for d in DEFECTS], dtype=float)
    total = counts.sum()
    cum = np.cumsum(counts) / total * 100.0
    x = np.arange(len(counts))
    k = int(np.argmax(cum >= 80.0))          # last member of the vital few
    cut = k + 0.5

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.subplots_adjust(left=0.082, right=0.905, top=0.815, bottom=0.155)
    ax2 = ax.twinx()

    ax.axvspan(-0.62, cut, color=VITAL_BG, zorder=0)
    cols = [BLUE if i <= k else PALE for i in range(len(counts))]
    ax.bar(x, counts, width=0.72, color=cols, edgecolor="white", linewidth=0.6,
           zorder=3)
    for xi, c in zip(x, counts):
        ax.text(xi, c + total * 0.011, f"{int(c)}", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold",
                color=NAVY if xi <= k else GREY, zorder=4)

    ax2.plot(x, cum, color=AMBER, lw=1.5, zorder=5)
    ax2.plot(x, cum, "o", ms=3.6, mfc="white", mec=AMBER, mew=1.2, zorder=6)
    for xi, cv in zip(x, cum):
        if xi in (k, len(x) - 1):
            ax2.annotate(f"{cv:.1f}%", (xi, cv), xytext=(0, 9),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=6.5, fontweight="bold", color=AMBER,
                         zorder=7)

    ax2.axhline(80, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=4)
    ax2.text(k + 0.85, 78.5, "80% threshold", ha="left", va="top",
             fontsize=6.5, fontweight="bold", color=ALERT, zorder=7)
    ax.axvline(cut, color=ALERT, lw=0.9, ls=(0, (4, 2)), zorder=4)

    ax.set_xlim(-0.62, len(x) - 0.38)
    ax.set_ylim(0, counts.max() * 1.28)
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.4, linespacing=1.35)
    ax.set_ylabel("Defect occurrences (units)", fontsize=8)
    ax2.set_ylabel("Cumulative share (%)", fontsize=8, color=AMBER)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7, colors=AMBER)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(AMBER)
    ax2.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e8ecf0", lw=0.6, zorder=1)

    span = k + 1
    ax.text((cut - 0.62) / 2, counts.max() * 1.235,
            f"VITAL FEW \u2014 {span} of {len(x)} defect types cause "
            f"{cum[k]:.0f}% of all rejects", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=NAVY, zorder=6)
    ax.text((cut + len(x) - 0.38) / 2, counts.max() * 1.235,
            "TRIVIAL MANY", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=GREY, zorder=6)

    dpmo = total / (UNITS_INSPECTED * OPPORTUNITIES) * 1e6
    ax.text(0.995, 0.62,
            f"Total defects   {int(total):,}\n"
            f"Units inspected  {UNITS_INSPECTED:,}\n"
            f"Opportunities/unit  {OPPORTUNITIES}\n"
            f"DPMO   {dpmo:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=INK, linespacing=1.6, zorder=7,
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="#dfe4ea",
                      lw=0.7))

    handles = [
        Patch(fc=BLUE, ec="white", label="Vital few (cumulative $\\leq$ 80%)"),
        Patch(fc=PALE, ec="white", label="Trivial many"),
        Line2D([], [], color=AMBER, marker="o", ms=3.6, mfc="white",
               mec=AMBER, lw=1.5, label="Cumulative share"),
        Line2D([], [], color=ALERT, lw=0.9, ls=(0, (5, 2.5)),
               label="80% Pareto threshold"),
    ]


    savefig(fig, "245_pareto_defects.png")


if __name__ == "__main__":
    main()
