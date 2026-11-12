"""Polar bar / nightingale rose chart: monthly climatology by season.

Circular bar chart of the monthly NDVI climatology. Bars are coloured by season
rather than by month index, so the colour carries information the angle does
not; the radial axis is labelled (the usual failure of rose charts), whiskers
give the interannual spread, and a dashed ring marks the annual mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_OF = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
COLORS = [PALETTES["cb_safe"][4], PALETTES["cb_safe"][2],
          PALETTES["cb_safe"][1], PALETTES["cb_safe"][5]]
N_YEARS = 12


def main():
    set_pub_style()
    rng = np.random.default_rng(511)
    n = len(MONTHS)
    phase = np.linspace(0, 2 * np.pi, n, endpoint=False)
    climatology = 0.30 + 0.26 * np.sin(phase - 1.9)
    years = climatology[None, :] + rng.normal(0, 0.035, (N_YEARS, n))
    vals, sd = years.mean(0), years.std(0, ddof=1)

    theta = phase
    width = 2 * np.pi / n * 0.80
    fig, ax = plt.subplots(figsize=(7.2, 7.0),
                           subplot_kw=dict(projection="polar"))
    cols = [COLORS[SEASON_OF[i]] for i in range(n)]
    ax.bar(theta, vals, width=width, color=cols, edgecolor="white",
           linewidth=1.0, alpha=0.92, zorder=3)
    ax.errorbar(theta, vals, yerr=sd, fmt="none", ecolor="#444444",
                elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)

    mean = vals.mean()
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ring, np.full_like(ring, mean), color="#555555", ls="--", lw=0.9,
            zorder=6)
    ax.text(2.2 * 2 * np.pi / n, mean, f"annual mean {mean:.2f}", fontsize=7,
            color="#555555", ha="center", va="bottom", zorder=7)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rorigin(-0.16)
    ax.set_ylim(0, 0.72)
    ax.set_xticks(theta)
    ax.set_xticklabels(MONTHS, fontsize=8.5)
    ax.set_yticks([0.15, 0.30, 0.45, 0.60])
    ax.set_yticklabels(["0.15", "0.30", "0.45", "0.60"], fontsize=7,
                       color="#666666")
    ax.set_rlabel_position(8)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#dcdcdc", lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0, ax.get_rorigin(), "NDVI\nclimatology", ha="center",
            va="center", fontsize=8.5, fontweight="bold", color="#333333")
    ax.set_title("Monthly NDVI climatology, 2009-2020", fontsize=11,
                 fontweight="bold", pad=24)

    handles = [Line2D([], [], marker="s", color="none", markersize=7,
                      markerfacecolor=COLORS[s], markeredgecolor="white",
                      label=SEASONS[s]) for s in range(4)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=8.5, frameon=False, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, "52_polar_rose.png")


if __name__ == "__main__":
    main()
