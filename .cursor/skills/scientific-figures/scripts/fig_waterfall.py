"""Waterfall chart: cumulative contribution decomposition.

Opening stock and closing stock are drawn as full-height total bars in a
neutral blue; the drivers in between are teal (gain) or rust (loss), a
colour-blind-safe substitute for the usual green/red.  Connectors run between
consecutive bar tops and every bar is labelled with its signed contribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

GAIN, LOSS = PALETTES["teal_rust"][0], PALETTES["teal_rust"][1]
TOTAL = "#3b5b92"

DRIVERS = ["Afforestation", "Urban\nexpansion", "Cropland\nconversion",
           "Wetland\nrestoration", "Policy\nprogramme"]


def waterfall(ax, opening, values, ylabel, unit):
    labels = ["Opening\nstock"] + DRIVERS + ["Closing\nstock"]
    cum = opening
    ax.bar(0, opening, color=TOTAL, edgecolor="white", width=0.62, zorder=3)
    ax.text(0, opening * 1.015, f"{opening:.0f}", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1a1a1a", zorder=5)

    for i, v in enumerate(values, start=1):
        col = GAIN if v >= 0 else LOSS
        ax.bar(i, v, bottom=cum, color=col, edgecolor="white", width=0.62,
               zorder=3)
        top = cum + v
        ax.text(i, max(cum, top) + 0.012 * opening, f"{v:+.0f}",
                ha="center", va="bottom", fontsize=7.5, color="#1a1a1a",
                zorder=5)
        ax.plot([i - 0.31, i + 0.31 + 0.38], [top, top], color="#9a9a9a",
                lw=0.8, ls=(0, (3, 2)), zorder=2)
        cum = top

    k = len(values) + 1
    ax.bar(k, cum, color=TOTAL, edgecolor="white", width=0.62, zorder=3)
    ax.text(k, cum * 1.015, f"{cum:.0f}", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1a1a1a", zorder=5)

    net = cum - opening
    ax.text(0.985, 0.965,
            f"net change {net:+.0f} {unit}\n({net / opening * 100:+.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            fontweight="bold", linespacing=1.4, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.94))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.2, linespacing=1.3)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.axhline(0, color="#444444", lw=0.8, zorder=4)
    ax.set_ylim(0, max(opening, cum) * 1.22)
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, length=2.5)
    style_axes(ax)


def main():
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.95))

    waterfall(axes[0], 240.0, [42, -58, -24, 30, 16],
              "Carbon stock (Tg C)", "Tg C")
    axes[0].set_title("Carbon stock decomposition, 1985-2020", fontsize=9,
                      fontweight="bold", pad=6)

    waterfall(axes[1], 62.0, [-9, 14, -6, 7, -3],
              "Habitat quality index (0-100)", "points")
    axes[1].set_title("Habitat quality decomposition, 1985-2020", fontsize=9,
                      fontweight="bold", pad=6)

    for ax, lab in zip(axes, "ab"):
        add_panel_label(ax, f"({lab})", x=-0.11, y=1.02, size=10)

    handles = [Patch(facecolor=TOTAL, label="Opening / closing stock"),
               Patch(facecolor=GAIN, label="Positive contribution"),
               Patch(facecolor=LOSS, label="Negative contribution")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0), columnspacing=2.0,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=3.0)
    savefig(fig, "20_waterfall.png")


if __name__ == "__main__":
    main()
