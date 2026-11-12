"""Seasonal subseries plot: twelve month-groups on one shared y axis.

Each group holds the same calendar month across 11 years, drawn as connected
points so the within-month trend is legible, with a thick red group mean and a
dashed within-group regression line.  One shared y axis is the whole point -
per-group autoscaling would make every month look equally variable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

PT_C = "#3b5b92"       # house blue: yearly values
MEAN_C = "#b23a48"     # house red: the single emphasis, the month mean
GREY = "#808080"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
N_YEAR = 11
YEAR0 = 2014
SEASON = np.array([32, 38, 58, 84, 112, 148, 171, 156, 118, 76, 48, 35],
                  dtype=float)
DRIFT = np.array([0.4, 0.2, 0.6, 0.9, 1.4, 2.6, 4.1, 3.0, 1.1, 0.3,
                  -0.2, 0.1])          # mm per year, by month


def synth(rng):
    yrs = np.arange(N_YEAR, dtype=float)
    noise = rng.normal(0.0, 1.0, (12, N_YEAR))
    scale = 0.10 * SEASON[:, None]
    return SEASON[:, None] + DRIFT[:, None] * yrs[None, :] + noise * scale


def main():
    set_pub_style()
    rng = np.random.default_rng(9)
    data = synth(rng)
    yrs = np.arange(N_YEAR, dtype=float)

    means = data.mean(axis=1)
    grand = data.mean()
    slopes = np.array([np.polyfit(yrs, data[m], 1)[0] for m in range(12)])
    spread = data.max(axis=1) - data.min(axis=1)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    half = 0.36
    for m in range(12):
        if m % 2 == 0:
            ax.axvspan(m - 0.5, m + 0.5, color="#f7f7f7", lw=0, zorder=0)
        x = m - half + 2 * half * yrs / (N_YEAR - 1)
        ax.plot(x, data[m], color=PT_C, lw=0.75, alpha=0.55, zorder=3)
        ax.plot(x, data[m], ls="none", marker="o", ms=2.6, mfc="white",
                mec=PT_C, mew=0.75, zorder=4)
        b, a = np.polyfit(yrs, data[m], 1)
        ax.plot(x, a + b * yrs, color=PT_C, lw=0.9, ls=(0, (2.6, 1.8)),
                alpha=0.9, zorder=5)
        ax.plot([m - half - 0.05, m + half + 0.05], [means[m]] * 2,
                color=MEAN_C, lw=2.6, zorder=6, solid_capstyle="butt")

    ax.axhline(grand, color=GREY, lw=0.9, ls=(0, (3.2, 2.4)), zorder=2)
    ax.text(11.62, grand, f" grand mean\n {grand:.0f} mm", ha="left",
            va="center", fontsize=6.6, color="#555555", linespacing=1.3)

    peak = int(np.argmax(means))
    fast = int(np.argmax(slopes))
    ax.annotate(f"{MONTHS[peak]} is the wettest month\n"
                f"(mean {means[peak]:.0f} mm, spread {spread[peak]:.0f} mm)",
                xy=(peak - half, means[peak]),
                xytext=(peak - 4.5, means[peak] + 40),
                fontsize=6.8, color="#555555", ha="left", va="bottom",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#8a8a8a",
                                connectionstyle="arc3,rad=0.22"))
    ax.annotate(f"steepest within-month trend:\n{MONTHS[fast]}, "
                f"{slopes[fast]:+.1f} mm per year",
                xy=(fast + half, data[fast, -1]),
                xytext=(fast + 1.85, data[fast].max() + 22),
                fontsize=6.8, color=PT_C, ha="left", va="bottom",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=PT_C,
                                connectionstyle="arc3,rad=-0.28"))

    ax.set_xlim(-0.5, 12.4)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=7.5)
    ax.set_ylim(0, data.max() * 1.20)
    ax.set_xlabel(f"Calendar month (within each group: {YEAR0} "
                  f"to {YEAR0 + N_YEAR - 1}, left to right)")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.grid(axis="y", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    style_axes(ax)

    ax.legend(handles=[
        Line2D([], [], color=PT_C, lw=0.9, marker="o", ms=2.8, mfc="white",
               mew=0.75, label="Annual value"),
        Line2D([], [], color=PT_C, lw=0.9, ls=(0, (2.6, 1.8)),
               label="Within-month trend"),
        Line2D([], [], color=MEAN_C, lw=2.4, label="Month mean"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (3.2, 2.4)),
               label="Grand mean")],
        loc="lower center", bbox_to_anchor=(0.5, 1.002), ncol=4, fontsize=7,
        handlelength=1.8, columnspacing=2.0, borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0.03, 1, 0.935])
    savefig(fig, "360_ts_seasonal_subseries.png")


if __name__ == "__main__":
    main()
