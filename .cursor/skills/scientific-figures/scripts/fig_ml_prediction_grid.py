"""ML model comparison grid: predicted vs observed with marginal density.

One small panel per model, ranked best-to-worst.  Each panel carries a
density-coloured predicted-vs-observed scatter, the 1:1 reference line, the
OLS fit, a right-hand kernel-density strip of the predictions and a metrics
box.  A shared density colorbar sits at the bottom of the figure; axis labels
are shared so the individual panels stay uncluttered.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, PALETTES, savefig, style_axes

MODELS = ["MLR", "ElasticNet", "Ridge", "Lasso", "SVR", "KNN", "MLP",
          "RandomForest", "ExtraTrees", "GBDT", "AdaBoost", "XGBoost",
          "LightGBM", "CatBoost"]

CMAP = LinearSegmentedColormap.from_list("dens", PALETTES["greens"][::-1])
C_ONE2ONE = "#9a9a9a"
C_FIT = "#2c3e70"
LIM = (17.5, 35.5)


def simulate(rng, quality, n=260):
    """Draw one model's predictions; weaker models shrink and scatter more."""
    obs = rng.uniform(20, 33, n)
    shrink = 0.72 + 0.26 * quality
    sigma = (1.0 - quality) * 6.0 + 0.35
    pred = 26.5 + shrink * (obs - 26.5) + rng.normal(0, sigma, n)
    resid = pred - obs
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((obs - obs.mean()) ** 2)
    return obs, pred, r2, np.sqrt(np.mean(resid ** 2)), np.mean(np.abs(resid))


def panel(fig, subspec, name, data, show_x, show_y):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec,
                                 width_ratios=[4.2, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    axk = fig.add_subplot(gs[1], sharey=ax)

    obs, pred, r2, rmse, mae = data

    xy = np.vstack([obs, pred])
    dens = gaussian_kde(xy)(xy)
    dens = (dens - dens.min()) / (np.ptp(dens) or 1.0)
    order = np.argsort(dens)
    ax.scatter(obs[order], pred[order], s=11, c=dens[order], cmap=CMAP,
               vmin=0, vmax=1, alpha=0.85, linewidths=0.2,
               edgecolors="white", zorder=3)

    ax.plot(LIM, LIM, "--", color=C_ONE2ONE, lw=1.0, zorder=2)
    b = np.polyfit(obs, pred, 1)
    xr = np.linspace(20, 33, 20)
    ax.plot(xr, np.polyval(b, xr), color=C_FIT, lw=1.4, zorder=4)

    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=6.5)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.set_title(name, fontsize=8, fontweight="bold", pad=3)
    style_axes(ax)

    ax.text(0.965, 0.045,
            f"R$^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#d5d5d5",
                      lw=0.5, alpha=0.92))

    ys = np.linspace(*LIM, 120)
    kde = gaussian_kde(pred)(ys)
    axk.fill_betweenx(ys, 0, kde, color=PALETTES["greens"][2], alpha=0.45,
                      linewidth=0)
    axk.plot(kde, ys, color=PALETTES["greens"][1], lw=0.8)
    axk.set_xlim(0, kde.max() * 1.25)
    axk.axis("off")


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    qualities = np.linspace(0.62, 0.95, len(MODELS))
    rng.shuffle(qualities)
    runs = [simulate(rng, q) for q in qualities]
    order = np.argsort([r[2] for r in runs])[::-1]      # rank on realised R2
    models = [MODELS[i] for i in order]
    runs = [runs[i] for i in order]

    fig = plt.figure(figsize=(9.4, 9.8))
    outer = fig.add_gridspec(4, 8, wspace=0.55, hspace=0.42,
                             left=0.075, right=0.985, top=0.925, bottom=0.105)
    for i, (name, data) in enumerate(zip(models, runs)):
        row = i // 4
        if row < 3:
            col = (i % 4) * 2
        else:
            col = 2 + (i % 4) * 2        # centre the short last row
        panel(fig, outer[row, col:col + 2], name, data,
              show_x=(row >= 2), show_y=(i % 4 == 0) or i == 12)

    fig.supxlabel("Observed yield (t ha$^{-1}$)", fontsize=9, y=0.072)
    fig.supylabel("Predicted yield (t ha$^{-1}$)", fontsize=9, x=0.022)

    cax = fig.add_axes([0.63, 0.045, 0.20, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("Relative point density", fontsize=7.5, labelpad=2)
    cb.ax.set_xticklabels(["low", "", "high"])
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    handles = [
        Line2D([], [], color=C_ONE2ONE, ls="--", lw=1.0, label="1 : 1 line"),
        Line2D([], [], color=C_FIT, lw=1.4, label="OLS fit"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.10, 0.032),
               ncol=2, frameon=False, fontsize=8, columnspacing=1.6)

    savefig(fig, "03_ml_prediction_grid.png")


if __name__ == "__main__":
    main()
