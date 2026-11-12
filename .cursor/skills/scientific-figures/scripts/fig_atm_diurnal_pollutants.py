"""Diurnal cycles of six criteria pollutants at one urban monitoring station.

Each panel is an hourly mean with its interquartile band over a season of
observations, on a night-shaded background so the rush-hour and photochemical
signatures are readable at a glance.  The six species differ in unit and in
magnitude by three orders, so a shared axis would be a lie of a different
kind; every panel therefore carries its own scale and states its unit in the
title, and the one comparison the figure does make - the anti-phase between
ozone and nitrogen dioxide - is quantified with a correlation computed from
the two mean curves that are drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from pubstyle import set_pub_style, savefig, COLORS
from figkit import panel_labels_inches

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
NIGHT = "#f0f0f0"

# One colour per pollutant, reused across the whole atmospheric-chemistry set.
POLLUTANTS = [
    ("$O_3$", "ppb", "#3b5b92"),
    ("$NO_2$", "ppb", "#c0603f"),
    ("$PM_{2.5}$", "$\\mu$g m$^{-3}$", "#2e7d5b"),
    ("$PM_{10}$", "$\\mu$g m$^{-3}$", "#c9a227"),
    ("$SO_2$", "ppb", "#8e6bbf"),
    ("CO", "ppm", "#2a7f7a"),
]

N_DAY = 132                      # days contributing to each hourly statistic
SUNRISE, SUNSET = 6.0, 18.5      # local hours bounding the shaded night


def bump(h, centre, width, amp):
    return amp * np.exp(-0.5 * ((h - centre) / width) ** 2)


def shapes(h):
    """Climatological diurnal shape of each species, in its own unit."""
    o3 = 16.0 + bump(h, 15.0, 3.4, 48.0) + bump(h, 13.0, 6.0, 9.0)
    no2 = 17.0 + bump(h, 8.0, 1.7, 16.0) + bump(h, 20.5, 2.3, 13.0) \
        - bump(h, 15.0, 3.2, 8.5)
    pm25 = 44.0 + bump(h, 8.5, 2.1, 13.0) + bump(h, 22.0, 3.0, 18.0) \
        - bump(h, 15.5, 3.4, 11.0)
    pm10 = 82.0 + bump(h, 8.5, 2.2, 20.0) + bump(h, 21.5, 3.2, 22.0) \
        + bump(h, 14.0, 3.6, 9.0)
    so2 = 5.2 + bump(h, 9.5, 2.6, 4.4) + bump(h, 19.0, 2.8, 1.8)
    co = 0.66 + bump(h, 8.0, 1.8, 0.34) + bump(h, 21.0, 2.6, 0.30)
    return [o3, no2, pm25, pm10, so2, co]


def main():
    set_pub_style()
    rng = np.random.default_rng(556)

    h = np.arange(24, dtype=float)
    means, p25, p75 = [], [], []
    for base in shapes(h):
        # day-to-day variability: a whole-day multiplier plus hourly scatter
        day = rng.lognormal(0.0, 0.26, (N_DAY, 1))
        obs = base[None, :] * day * rng.lognormal(0.0, 0.13, (N_DAY, 24))
        means.append(obs.mean(axis=0))
        p25.append(np.percentile(obs, 25, axis=0))
        p75.append(np.percentile(obs, 75, axis=0))

    r_o3_no2 = float(stats.pearsonr(means[0], means[1]).statistic)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    fig.subplots_adjust(left=0.062, right=0.982, top=0.845, bottom=0.115,
                        hspace=0.46, wspace=0.26)

    for i, (ax, (name, unit, col)) in enumerate(zip(axes.ravel(), POLLUTANTS)):
        ax.axvspan(0.0, SUNRISE, color=NIGHT, zorder=0)
        ax.axvspan(SUNSET, 23.0, color=NIGHT, zorder=0)
        ax.fill_between(h, p25[i], p75[i], color=col, alpha=0.22,
                        linewidth=0, zorder=2)
        ax.plot(h, means[i], color=col, lw=1.8, zorder=4)

        hi = int(np.argmax(means[i]))
        ax.plot([h[hi]], [means[i][hi]], marker="o", ms=4.0, mfc="white",
                mec=col, mew=1.2, zorder=6)
        ax.annotate(f"max {means[i][hi]:.3g} at {hi:02d}:00",
                    xy=(h[hi], means[i][hi]), xytext=(0.5, 1.012),
                    textcoords="axes fraction", ha="center", va="bottom",
                    fontsize=6.4, color=SUB,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#b3b3b3",
                                    shrinkB=3.0))

        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["00", "06", "12", "18", "23"], fontsize=7.0)
        ax.set_ylim(0.0, float(np.max(p75[i])) * 1.16)
        ax.set_title(f"{name} ({unit})", fontsize=9.0, fontweight="bold",
                     color=INK, pad=13)
        ax.tick_params(labelsize=7.0)
        ax.grid(axis="y", color="#ececec", lw=0.55, zorder=1)
        if i >= 3:
            ax.set_xlabel("hour of day (local time, h)", fontsize=8.0)

    # The one cross-panel claim the figure makes, quantified on the spot.
    axes[0, 0].axvline(float(np.argmax(means[0])), color="#808080", lw=0.7,
                       ls=(0, (2.4, 1.8)), zorder=3)
    axes[0, 1].axvline(float(np.argmin(means[1])), color="#808080", lw=0.7,
                       ls=(0, (2.4, 1.8)), zorder=3)
    axes[0, 1].text(0.035, 0.055, f"anti-phase with $O_3$\nPearson "
                    f"$r$ = {r_o3_no2:+.2f}", transform=axes[0, 1].transAxes,
                    fontsize=6.6, color=INK, ha="left", va="bottom",
                    linespacing=1.32, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    axes[0, 1].text(float(np.argmin(means[1])) + 0.4, 0.965,
                    f"min at {int(np.argmin(means[1])):02d}:00",
                    transform=axes[0, 1].get_xaxis_transform(), fontsize=6.3,
                    color=SUB, ha="left", va="top")


    panel_labels_inches(fig, axes, dx=-0.34, dy=0.30)

    savefig(fig, "556_atm_diurnal_pollutants.png")


if __name__ == "__main__":
    main()
