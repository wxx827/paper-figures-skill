"""Prediction fit + permutation-importance ranking for two periods.

Left column: predicted-vs-observed scatter with the OLS fit, the 1:1 standard
curve and a metrics box.  Right column: permutation importance (%IncMSE) as a
sorted horizontal bar ranking with bootstrap error bars.  One row per time
window so the two periods can be compared directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

FEATURES = ["Precipitation", "Terrain relief", "Temperature",
            "Distance to city", "Population density", "Elevation",
            "Evapotranspiration", "Distance to road", "Forest loss", "GDP"]
BAR = "#2f6d63"
C_FIT = "#3b4fb0"
C_ONE2ONE = "#8a8a8a"
POINT = PALETTES["teal_rust"][0]


def fit_panel(ax, rng, shrink, sigma):
    n = 300
    obs = rng.uniform(115, 255, n)
    pred = 185 + shrink * (obs - 185) + rng.normal(0, sigma, n)

    ax.scatter(obs, pred, s=17, facecolors=POINT, edgecolors="white",
               linewidths=0.35, alpha=0.62, zorder=3)
    xr = np.array([105.0, 265.0])
    b = np.polyfit(obs, pred, 1)
    ax.plot(xr, np.polyval(b, xr), color=C_FIT, lw=1.6, zorder=5,
            label="OLS fit")
    ax.plot(xr, xr, "--", color=C_ONE2ONE, lw=1.2, zorder=4,
            label="1 : 1 line")

    resid = pred - obs
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((obs - obs.mean()) ** 2)
    rmse = np.sqrt(np.mean(resid ** 2))
    ax.text(0.965, 0.05,
            f"R$^2$ = {r2:.3f}\nRMSE = {rmse:.1f} mm\nslope = {b[0]:.2f}"
            f"\nn = {n}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.92))

    ax.set_xlabel("Observed water yield (mm)", fontsize=8)
    ax.set_ylabel("Predicted water yield (mm)", fontsize=8)
    ax.set_xlim(105, 265)
    ax.set_ylim(105, 265)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper left", frameon=False,
              handlelength=1.8, borderaxespad=0.3)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def imp_panel(ax, rng, color):
    vals = np.sort(rng.uniform(2.5, 41.0, len(FEATURES)))
    err = vals * rng.uniform(0.06, 0.16, len(FEATURES))
    ypos = np.arange(len(FEATURES))
    names = [FEATURES[i] for i in np.argsort(rng.permutation(len(FEATURES)))]

    ax.barh(ypos, vals, height=0.68, color=color, edgecolor="white",
            linewidth=0.6, zorder=3)
    ax.errorbar(vals, ypos, xerr=err, fmt="none", ecolor="#3a3a3a",
                elinewidth=0.7, capsize=2, zorder=4)
    for yy, vv, ee in zip(ypos, vals, err):
        ax.text(vv + ee + 1.0, yy, f"{vv:.1f}", va="center", ha="left",
                fontsize=6.8, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, vals.max() * 1.28)
    ax.set_xlabel("Permutation importance, %IncMSE", fontsize=8)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, length=2.5)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    fit_panel(axes[0, 0], rng, shrink=0.93, sigma=11.0)
    imp_panel(axes[0, 1], rng, BAR)
    fit_panel(axes[1, 0], rng, shrink=0.97, sigma=7.5)
    imp_panel(axes[1, 1], rng, BAR)

    titles = ["Model fit, 1985-2005", "Driver ranking, 1985-2005",
              "Model fit, 2005-2020", "Driver ranking, 2005-2020"]
    for ax, lab, ttl in zip(axes.ravel(), "abcd", titles):
        ax.set_title(ttl, fontsize=8.5, fontweight="bold", pad=4)
        add_panel_label(ax, f"({lab})", x=-0.22 if lab in "bd" else -0.17,
                        y=1.03, size=10)

    fig.tight_layout(h_pad=2.2, w_pad=2.6)
    savefig(fig, "04_pred_importance.png")


if __name__ == "__main__":
    main()
