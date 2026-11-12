"""Nonparametric bootstrap sampling distributions for six statistics.

A single paired sample (n = 600) is resampled 4,000 times with replacement.
Each panel shows the bootstrap histogram with its KDE, a normal reference
curve matched in mean and SD, the estimate from the original sample, and the
percentile 95% interval with labelled endpoints.  Bias and bootstrap standard
error are reported per panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, trim_mean
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

HIST_C = "#cfe1ee"
EDGE_C = "#7ba7c7"
KDE_C = "#1f5f8b"
EST_C = "#b23a48"
CI_C = "#2a7f7a"
REF_C = "#9a9a9a"


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(90210)

    n, B = 600, 4000
    x = rng.gamma(6.0, 0.55, n)
    y = 1.85 + 0.72 * x + rng.normal(0, 0.85, n)

    idx = rng.integers(0, n, size=(B, n))
    xb, yb = x[idx], y[idx]

    xm = xb.mean(axis=1, keepdims=True)
    ym = yb.mean(axis=1, keepdims=True)
    sxy = ((xb - xm) * (yb - ym)).mean(axis=1)
    sxx = ((xb - xm) ** 2).mean(axis=1)
    syy = ((yb - ym) ** 2).mean(axis=1)
    slope_b = sxy / sxx
    r_b = sxy / np.sqrt(sxx * syy)

    def orig(fn):
        return float(fn(x, y))

    panels = [
        (r"mean  $\bar{y}$", yb.mean(axis=1),
         orig(lambda a, b: b.mean())),
        # a trimmed mean rather than the median: bootstrapping a single order
        # statistic produces a lattice, not a smooth sampling distribution
        (r"20% trimmed mean  $\bar{y}_{0.2}$", trim_mean(yb, 0.2, axis=1),
         orig(lambda a, b: trim_mean(b, 0.2))),
        (r"std. deviation  $s_y$", yb.std(axis=1, ddof=1),
         orig(lambda a, b: b.std(ddof=1))),
        (r"Pearson  $r_{xy}$", r_b,
         orig(lambda a, b: np.corrcoef(a, b)[0, 1])),
        (r"OLS slope  $\hat{\beta}_1$", slope_b,
         orig(lambda a, b: np.polyfit(a, b, 1)[0])),
        (r"coefficient of determination  $R^2$", r_b ** 2,
         orig(lambda a, b: np.corrcoef(a, b)[0, 1] ** 2)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    fig.subplots_adjust(left=0.045, right=0.988, top=0.845, bottom=0.115,
                        hspace=0.58, wspace=0.13)

    for k, (title, boot, est) in enumerate(panels):
        ax = axes.ravel()[k]
        style_axes(ax)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        se = boot.std(ddof=1)
        bias = boot.mean() - est

        cnt, edges = np.histogram(boot, bins=44, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, cnt, width=edges[1] - edges[0], color=HIST_C,
               edgecolor=EDGE_C, lw=0.3, zorder=2)

        gx = np.linspace(boot.min(), boot.max(), 400)
        ax.plot(gx, gaussian_kde(boot)(gx), color=KDE_C, lw=1.4, zorder=5)
        ax.plot(gx, norm(boot.mean(), se).pdf(gx), color=REF_C, lw=0.9,
                ls=(0, (4, 2)), zorder=4)

        inside = (gx >= lo) & (gx <= hi)
        ax.fill_between(gx[inside], 0, gaussian_kde(boot)(gx[inside]),
                        color=CI_C, alpha=0.16, lw=0, zorder=3)
        ymax = cnt.max()
        for v in (lo, hi):
            ax.plot([v, v], [0, ymax * 0.92], color=CI_C, lw=0.9,
                    ls=(0, (2, 2)), zorder=6)
        ax.plot([est, est], [0, ymax * 1.06], color=EST_C, lw=1.5, zorder=7)

        ax.set_ylim(0, ymax * 1.42)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=6.8)
        ax.set_xlabel(title, fontsize=8.2, labelpad=2)

        ax.text(lo, ymax * 0.96, f"{lo:.3g}", ha="center", va="bottom",
                fontsize=6.0, color=CI_C)
        ax.text(hi, ymax * 0.96, f"{hi:.3g}", ha="center", va="bottom",
                fontsize=6.0, color=CI_C)
        ax.text(0.015, 0.995,
                r"$\hat{\theta}$ = " + f"{est:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.6, color=EST_C,
                fontweight="bold")
        ax.text(0.985, 0.995, f"SE = {se:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.6, color="#444444")
        ax.text(0.985, 0.885, f"bias = {bias:+.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.6, color="#777777")
        add_panel_label(ax, f"({'abcdef'[k]})", x=-0.02, y=1.03, size=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=HIST_C, ec=EDGE_C, lw=0.4,
                      label="bootstrap replicates"),
        plt.Line2D([], [], color=KDE_C, lw=1.5, label="bootstrap KDE"),
        plt.Line2D([], [], color=REF_C, lw=1.0, ls=(0, (4, 2)),
                   label="normal reference"),
        plt.Line2D([], [], color=EST_C, lw=1.6,
                   label=r"estimate from original sample"),
        plt.Line2D([], [], color=CI_C, lw=1.0, ls=(0, (2, 2)),
                   label="percentile 95% CI"),
    ]

    savefig(fig, "132_bootstrap_dist.png")


if __name__ == "__main__":
    main()
