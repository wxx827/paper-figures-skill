"""LMDI decomposition of CO2 emission change, six regions (2010-2024).

Each panel is a waterfall: the population, affluence, energy-intensity and
structural effects accumulate from zero to the net change, with connectors
between consecutive bars.  Increases and decreases are colour-separated and
the net bar is drawn in dark navy.  A common y axis makes the panels
directly comparable across regions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

POS, NEG, NET = "#b5654f", "#3f8f7a", "#2f4a6b"
FACTORS = ["Population", "Affluence (GDP p.c.)", "Energy intensity",
           "Economic structure"]

REGIONS = [
    ("North region", 612.0, [38.0, 196.0, -142.0, -46.0]),
    ("East region", 884.0, [52.0, 268.0, -205.0, -78.0]),
    ("South region", 497.0, [44.0, 171.0, -119.0, -35.0]),
    ("Central region", 356.0, [19.0, 121.0, -96.0, -52.0]),
    ("West region", 402.0, [31.0, 158.0, -84.0, -21.0]),
    ("National total", 2751.0, [184.0, 914.0, -646.0, -232.0]),
]


def region_extent(effects):
    """Cumulative low/high reached by one waterfall, including the net bar."""
    cum = np.concatenate([[0.0], np.cumsum(effects)])
    return min(cum.min(), cum[-1], 0.0), max(cum.max(), cum[-1], 0.0)


def waterfall(ax, base, effects, title, panel_is_total=False, ylim=None):
    cum = np.concatenate([[0.0], np.cumsum(effects)])
    net = cum[-1]
    xs = np.arange(len(effects) + 1)

    span = (ylim[1] - ylim[0]) if ylim else \
        max(cum.max(), net, 0) - min(cum.min(), net, 0)
    off = 0.030 * span

    for i, v in enumerate(effects):
        col = POS if v >= 0 else NEG
        ax.bar(i, v, bottom=cum[i], width=0.62, color=col,
               edgecolor="white", lw=0.6, zorder=3)
        top = cum[i] + v
        ax.text(i, top + (off if v >= 0 else -off), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.6,
                color="#1a1a1a", zorder=4)
        ax.plot([i + 0.31, i + 1 - 0.31], [top, top], color="#9a9a9a",
                lw=0.7, ls=(0, (2.5, 2)), zorder=2)

    ax.bar(len(effects), net, width=0.62, color=NET, edgecolor="white",
           lw=0.6, zorder=3)
    ax.text(len(effects), net + (off if net >= 0 else -off), f"{net:+.0f}",
            ha="center", va="bottom" if net >= 0 else "top", fontsize=7,
            fontweight="bold", color=NET, zorder=4)

    ax.axhline(0, color="#333333", lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(FACTORS + ["Net change"], fontsize=6.3, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.62, len(effects) + 0.62)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        lo, hi = region_extent(effects)
        own = hi - lo
        ax.set_ylim(lo - 0.20 * own, hi + 0.20 * own)
    ax.set_title(title, fontsize=8.4, loc="left", pad=13,
                 weight="bold" if panel_is_total else "normal")
    ax.text(0.0, 1.015, f"net {100 * net / base:+.1f}% vs 2010 baseline "
            f"({base:.0f} Mt)", transform=ax.transAxes, fontsize=6.4,
            color="#6f6f6f", ha="left", va="bottom")
    ax.grid(axis="y", lw=0.5, color="#ededed")
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.1))
    flat = axes.ravel()

    # The five regional panels share one vertical scale so their bars can be
    # read against each other; only the national total, an order of magnitude
    # larger, keeps its own scale and says so.
    extents = [region_extent(eff) for _, _, eff in REGIONS[:-1]]
    lo = min(e[0] for e in extents)
    hi = max(e[1] for e in extents)
    shared = (lo - 0.22 * (hi - lo), hi + 0.16 * (hi - lo))

    for k, (ax, (name, base, eff)) in enumerate(zip(flat, REGIONS)):
        is_total = k == len(REGIONS) - 1
        waterfall(ax, base, eff, name, panel_is_total=is_total,
                  ylim=None if is_total else shared)
        add_panel_label(ax, f"({'abcdef'[k]})",
                        x=-0.22 if k % 3 == 0 else -0.19, y=1.13, size=9.5)
        if is_total:
            ax.text(0.985, 0.965, "own vertical scale", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.2, color="#8a8a8a",
                    style="italic")

    for ax in axes[:, 0]:
        ax.set_ylabel("Change in CO$_2$ emissions\n(Mt CO$_2$, 2010-2024)",
                      fontsize=7.4, linespacing=1.35, labelpad=3)

    handles = [
        Patch(fc=POS, ec="white", label="Effect raises emissions"),
        Patch(fc=NEG, ec="white", label="Effect lowers emissions"),
        Patch(fc=NET, ec="white", label="Net change"),
        Line2D([], [], color="#9a9a9a", lw=0.8, ls=(0, (2.5, 2)),
               label="Cumulative connector"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.2,
               frameon=False, bbox_to_anchor=(0.5, 0.038), handlelength=1.5,
               columnspacing=2.0)


    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.955), h_pad=2.2, w_pad=2.4)
    savefig(fig, "108_decomposition_lmdi.png")


if __name__ == "__main__":
    main()
