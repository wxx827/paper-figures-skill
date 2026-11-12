"""Lollipop chart: ranked permutation importance with cross-validation spread.

Clean ranking view: one stem plus a circular head per predictor, ordered by
importance, coloured by predictor family, with +/-1 s.d. whiskers across the
20 cross-validation folds and value labels outside the whiskers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig, style_axes

# (feature, family, true importance)
FEATURES = [
    ("NDVI", "Vegetation", 24.0), ("LAI", "Vegetation", 8.4),
    ("Albedo", "Vegetation", 4.1), ("Soil moisture", "Vegetation", 9.6),
    ("Temperature", "Climate", 18.7), ("Precipitation", "Climate", 15.2),
    ("Elevation", "Topography", 11.3), ("Slope", "Topography", 5.5),
    ("Population", "Human pressure", 13.9),
    ("Night-time light", "Human pressure", 7.2),
    ("Road density", "Human pressure", 6.3),
    ("Distance to water", "Topography", 3.2),
]
FAMILIES = ["Vegetation", "Climate", "Topography", "Human pressure"]
FAM_COLORS = dict(zip(FAMILIES, PALETTES["cb_safe"][:4]))
N_FOLDS = 20


def main():
    set_pub_style()
    rng = np.random.default_rng(361)
    names = [f[0] for f in FEATURES]
    fams = [f[1] for f in FEATURES]
    truth = np.array([f[2] for f in FEATURES])

    folds = np.maximum(truth[:, None] * rng.normal(1.0, 0.18, (len(truth), N_FOLDS)),
                       0.2)
    mean, sd = folds.mean(1), folds.std(1, ddof=1)
    order = np.argsort(-mean)
    mean, sd = mean[order], sd[order]
    names = [names[i] for i in order]
    cols = [FAM_COLORS[fams[i]] for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    y = np.arange(len(names))
    ax.hlines(y, 0, mean, color=cols, lw=1.6, alpha=0.55, zorder=2)
    ax.errorbar(mean, y, xerr=sd, fmt="none", ecolor="#666666", elinewidth=0.9,
                capsize=2.2, capthick=0.9, zorder=3)
    ax.scatter(mean, y, color=cols, s=76, zorder=4, edgecolors="white",
               linewidths=1.0)
    for yi, (m, s) in enumerate(zip(mean, sd)):
        ax.text(m + s + mean.max() * 0.018, yi, f"{m:.1f}", va="center",
                fontsize=7.5, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(0, (mean + sd).max() * 1.14)
    ax.set_xlabel("Permutation importance (% increase in RMSE)", fontsize=9)
    ax.set_title("Predictor importance in the random-forest model",
                 fontsize=10.5, fontweight="bold", loc="left", x=-0.19, pad=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    style_axes(ax)

    handles = [Line2D([], [], marker="o", color="none", markersize=7,
                      markerfacecolor=FAM_COLORS[f], markeredgecolor="white",
                      label=f) for f in FAMILIES]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False,
              handletextpad=0.4, labelspacing=0.5)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    savefig(fig, "40_lollipop.png")


if __name__ == "__main__":
    main()
