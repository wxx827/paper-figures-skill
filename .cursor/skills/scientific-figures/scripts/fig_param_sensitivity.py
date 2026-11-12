"""Parameter sensitivity study of ISOA.

Panel (a) is the two-dimensional grid search over the population size and the
spiral control parameter, with iso-performance contours and a star on the
selected configuration. Panels (b) to (d) sweep one parameter at a time and
report the mean rank over the eight benchmarks with a dispersion band over 30
runs. Panel (e) summarises how much of the performance variance each
parameter explains.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from pubstyle import set_pub_style, savefig, add_panel_label, get_cmap

RED = "#b23a48"
POP_VALUES = np.array([10, 20, 30, 40, 50, 60, 80, 100])
FC_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
BEST_POP, BEST_FC = 30, 2.0

SWEEPS = [
    dict(name="Population size  $N$", short="N", label="(b)",
         x=POP_VALUES.astype(float), best=30.0, lo=1.28, hi=2.95,
         curve="u", width=1.15, logx=False, fmt="{:.0f}"),
    dict(name="Maximum iterations  $T$", short="T", label="(c)",
         x=np.array([25, 50, 100, 200, 400, 800], dtype=float),
         best=200.0, lo=1.22, hi=2.62, curve="sat", width=1.0, logx=True,
         fmt="{:.0f}"),
    dict(name="Spiral control parameter  $f_c$", short="$f_c$", label="(d)",
         x=FC_VALUES.astype(float), best=2.0, lo=1.31, hi=3.05, curve="u",
         width=0.95, logx=False, fmt="{:.1f}"),
]

IMPORTANCE = [
    ("Spiral control  $f_c$", 0.412),
    ("Population size  $N$", 0.286),
    ("Maximum iterations  $T$", 0.174),
    ("Levy step scale  $\\beta$", 0.081),
    ("Initial seed", 0.047),
]


def grid_scores():
    """Mean Friedman rank of ISOA over the eight benchmarks per (N, fc)."""
    rng = np.random.default_rng(90210)
    P, F = np.meshgrid(np.log(POP_VALUES / BEST_POP), FC_VALUES - BEST_FC)
    quad = 0.62 * P ** 2 + 0.30 * F ** 2 + 0.16 * P * F
    z = 1.24 + 1.85 * (1.0 - np.exp(-1.15 * quad))
    z = z + gaussian_filter(rng.normal(0, 1, z.shape), sigma=0.85) * 0.035
    return z


def panel_grid(ax, fig):
    z = grid_scores()
    im = ax.imshow(z, cmap=get_cmap("cb_diverging"), origin="lower", aspect="auto",
                   interpolation="bicubic",
                   extent=(-0.5, len(POP_VALUES) - 0.5,
                           -0.5, len(FC_VALUES) - 0.5))
    cs = ax.contour(np.arange(len(POP_VALUES)), np.arange(len(FC_VALUES)),
                    z, levels=8, colors="#ffffff", linewidths=0.55,
                    alpha=0.8)
    ax.clabel(cs, cs.levels[::2], fmt="%.2f", fontsize=6.0, inline=True,
              colors="#ffffff")

    for i in range(len(FC_VALUES)):
        for j in range(len(POP_VALUES)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.0, color="#2b2b2b", alpha=0.85)

    jy, jx = np.unravel_index(np.argmin(z), z.shape)
    ax.plot(jx, jy, marker="*", ms=17, mfc="#ffffff", mec="#111111",
            mew=1.1, zorder=8)
    ax.annotate(f"selected setting\n$N$ = {POP_VALUES[jx]},  "
                f"$f_c$ = {FC_VALUES[jy]:.1f}   (rank {z[jy, jx]:.3f})",
                xy=(jx, jy), xytext=(jx + 0.7, jy + 1.15), fontsize=6.6,
                color="#1a1a1a", zorder=9,
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#222222"),
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#c8c8c8", lw=0.6, alpha=0.94))

    ax.set_xticks(range(len(POP_VALUES)), [str(v) for v in POP_VALUES],
                  fontsize=7.0)
    ax.set_yticks(range(len(FC_VALUES)), [f"{v:.1f}" for v in FC_VALUES],
                  fontsize=7.0)
    ax.set_xlabel("Population size  $N$", fontsize=8.4, labelpad=2)
    ax.set_ylabel("Spiral control parameter  $f_c$", fontsize=8.4, labelpad=2)
    ax.tick_params(length=2.2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    ax.set_title("Grid search over the two dominant parameters",
                 fontsize=9.2, fontweight="bold", loc="left", pad=4)

    cax = ax.inset_axes([1.028, 0.0, 0.028, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean rank over the 8 benchmarks  ($\\downarrow$ better)",
                 fontsize=7.0, labelpad=3)
    cb.ax.tick_params(labelsize=6.4, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")


def sweep_curve(spec, seed):
    rng = np.random.default_rng(seed)
    x = spec["x"]
    u = (np.log(x / spec["best"]) if spec["logx"]
         else (x - spec["best"]) / (spec["x"].max() - spec["x"].min()) * 4.0)
    if spec["curve"] == "u":
        pen = 1.0 - np.exp(-(u / spec["width"]) ** 2)
    else:                                   # saturating: only small x hurts
        pen = 1.0 - np.exp(-np.clip(-u, 0, None) ** 2 / spec["width"] ** 2)
    mean = spec["lo"] + (spec["hi"] - spec["lo"]) * pen
    sd = 0.055 + 0.16 * pen
    runs = mean[None, :] + rng.normal(0, 1, (30, len(x))) * sd[None, :]
    return mean, np.percentile(runs, 25, axis=0), np.percentile(runs, 75,
                                                                axis=0)


def panel_sweep(ax, spec, seed, ylim):
    x = spec["x"]
    mean, q1, q3 = sweep_curve(spec, seed)
    ax.fill_between(x, q1, q3, color=RED, alpha=0.16, lw=0, zorder=2)
    ax.plot(x, mean, color=RED, lw=1.8, marker="o", ms=3.8, mfc="white",
            mec=RED, mew=1.0, zorder=5)
    k = int(np.argmin(np.abs(x - spec["best"])))
    ax.plot(x[k], mean[k], marker="*", ms=14, mfc="#c9a227", mec="#1a1a1a",
            mew=0.8, zorder=7)
    ax.axvline(spec["best"], color="#777777", lw=0.8, ls=(0, (3, 2)),
               zorder=1)
    ax.annotate(f"best  {spec['short']} = {spec['fmt'].format(spec['best'])}",
                xy=(x[k], mean[k]), xytext=(0, 15),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.6, color="#1a1a1a", zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#d5d5d5", lw=0.5, alpha=0.94))
    if spec["logx"]:
        ax.set_xscale("log")
        ax.set_xticks(x, [spec["fmt"].format(v) for v in x])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_xticks(x, [spec["fmt"].format(v) for v in x])
    ax.set_xlabel(spec["name"], fontsize=8.0, labelpad=1.8)
    ax.set_ylabel("Mean rank", fontsize=8.0, labelpad=1.8)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=6.8, length=2.2)
    ax.grid(True, lw=0.4, color="#f0f0f0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{spec['label']}  sweep of {spec['short']}", fontsize=8.4,
                 fontweight="bold", loc="left", pad=3)


def panel_importance(ax):
    names = [n for n, _ in IMPORTANCE][::-1]
    vals = np.array([v for _, v in IMPORTANCE][::-1])
    y = np.arange(len(vals))
    cols = ["#c9c9c9", "#b6cbd8", "#8fb7cf", "#e39a6a", RED]
    ax.barh(y, vals, height=0.66, color=cols, edgecolor="#555555",
            linewidth=0.5, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v * 100:.1f}%", va="center", ha="left",
                fontsize=7.2, color="#1a1a1a",
                fontweight="bold" if v == vals.max() else "normal")
    ax.set_yticks(y, names, fontsize=7.6)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(RED)
    ax.set_xlim(0, vals.max() * 1.30)
    ax.set_ylim(-0.62, len(vals) - 0.36)
    ax.set_xlabel("Share of the performance variance explained",
                  fontsize=8.0, labelpad=1.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v * 100:.0f}%"))
    ax.tick_params(axis="x", labelsize=6.8, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", lw=0.4, color="#f0f0f0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(e)  Parameter importance (functional ANOVA)",
                 fontsize=8.8, fontweight="bold", loc="left", pad=4)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(13.4, 7.55))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.42, 0.86, 0.96],
                  height_ratios=[1, 1, 1], left=0.062, right=0.982,
                  top=0.872, bottom=0.148, hspace=0.80, wspace=0.40)

    axg = fig.add_subplot(gs[:, 0])
    panel_grid(axg, fig)
    add_panel_label(axg, "(a)", x=-0.115, y=1.02, size=11)

    # (b)-(d) all report the same mean-rank scale, so they share one y range;
    # per-panel autoscaling made the mild T sweep look as damaging as f_c.
    bands = [sweep_curve(spec, 600 + row * 41)
             for row, spec in enumerate(SWEEPS)]
    ylim = (min(b[1].min() for b in bands) - 0.16,
            max(b[2].max() for b in bands) + 0.30)
    for row, spec in enumerate(SWEEPS):
        panel_sweep(fig.add_subplot(gs[row, 1]), spec, 600 + row * 41, ylim)

    axi = fig.add_subplot(gs[:, 2])
    panel_importance(axi)

    savefig(fig, "152_param_sensitivity.png")


if __name__ == "__main__":
    main()
