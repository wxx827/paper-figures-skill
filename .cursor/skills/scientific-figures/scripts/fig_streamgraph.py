"""Streamgraph: stacked area with a centred (themeriver) baseline.

Land-cover composition over 36 years.  Because a wiggle baseline has no
meaningful y axis, the magnitude is given by an explicit scale bar instead of
tick labels, and every band is labelled in place.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig

# semantic colours: green vegetation, blue water, warm anthropogenic
CATS = [
    ("Cropland", "#c9a227"),
    ("Forest", "#1f6b4d"),
    ("Grassland", "#7cb342"),
    ("Wetland", "#4fa5a0"),
    ("Water", "#3d6b8e"),
    ("Urban", "#a4503f"),
    ("Barren", "#b9b2a6"),
]
SCALE_UNITS = 100.0


def main():
    set_pub_style()
    rng = np.random.default_rng(71)
    years = np.arange(1985, 2021)
    m = len(years)

    series = []
    for i in range(len(CATS)):
        base = rng.uniform(20, 60)
        trend = np.cumsum(rng.normal(0, 1.2, m))
        y = np.clip(base + trend + 6 * np.sin(np.linspace(0, 3, m) + i),
                    5, None)
        series.append(y)
    series = np.array(series)

    total = series.sum(axis=0)
    cum = -total / 2.0
    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    label_slots = np.linspace(0.18, 0.82, len(CATS))
    for i, (y, (name, col)) in enumerate(zip(series, CATS)):
        ax.fill_between(years, cum, cum + y, color=col, alpha=0.95, lw=1.0,
                        edgecolor="white", label=name, zorder=2)
        # stagger the in-place labels so they do not stack in one column
        wpos = int(round(label_slots[i] * (m - 1)))
        if y[wpos] > total.max() * 0.055:
            ymid = cum[wpos] + y[wpos] / 2
            r, g, b = plt.matplotlib.colors.to_rgb(col)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(years[wpos], ymid, name, ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white" if lum < 0.58 else "#2b2b2b", zorder=3)
        cum = cum + y

    # explicit magnitude reference in place of a y axis
    x0 = years[0] - 1.6
    y0 = -total.max() / 2 * 0.85
    ax.plot([x0, x0], [y0, y0 + SCALE_UNITS], color="#444444", lw=1.1,
            clip_on=False, zorder=6)
    for yy in (y0, y0 + SCALE_UNITS):
        ax.plot([x0 - 0.45, x0 + 0.45], [yy, yy], color="#444444", lw=1.1,
                clip_on=False, zorder=6)
    ax.text(x0 - 1.0, y0 + SCALE_UNITS / 2,
            f"{SCALE_UNITS:.0f} $\\times$ 10$^3$ km$^2$", rotation=90,
            ha="right", va="center", fontsize=7.5, color="#444444")

    ax.set_xlim(years[0] - 3.4, years[-1])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Year")
    ax.set_title("Land-cover composition flow, 1985-2020", fontsize=10.5,
                 fontweight="bold")
    ax.text(0.995, 0.985,
            f"total mapped area {total[0]:.0f} $\\rightarrow$ "
            f"{total[-1]:.0f} $\\times$ 10$^3$ km$^2$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#444444", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.9))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=7,
              fontsize=7.5, frameon=False, columnspacing=1.4,
              handlelength=1.3)
    fig.tight_layout()
    savefig(fig, "16_streamgraph.png")


if __name__ == "__main__":
    main()
