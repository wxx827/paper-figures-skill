"""Statistical power analysis for a two-sample t-test.

(a) Power curves against sample size per group for six effect sizes, with
    the 80% target and the sample size at which each curve crosses it.
(b) Required n per group as a function of Cohen's d, showing the 1/d^2 law.
(c-e) Sensitivity of the same curve family to the significance level.

Power is exact: it uses the noncentral t distribution rather than a normal
approximation, so the crossing points can be quoted directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.optimize import brentq
from scipy.stats import nct, t as tdist
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, \
    add_panel_label

RAMP = LinearSegmentedColormap.from_list(
    "pw", PALETTES["cb_sequential_blue"][1:])
TARGET_C = "#b23a48"
EFFECTS = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]


def power_ttest(n, d, alpha=0.05):
    """Two-sided two-sample t-test power, equal group sizes."""
    n = np.asarray(n, dtype=float)
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2.0)
    crit = tdist.ppf(1 - alpha / 2.0, df)
    val = (1 - nct.cdf(crit, df, nc)) + nct.cdf(-crit, df, nc)
    # nct underflows for very large noncentrality: fill in the limit
    return np.where(np.isnan(val), np.where(nc > crit, 1.0, 0.0), val)


def n_for_power(d, alpha=0.05, target=0.80):
    """Smallest integer group size reaching the target power."""
    grid = np.arange(4, 8000)
    pw = power_ttest(grid, d, alpha)
    hit = np.argmax(pw >= target)
    return int(grid[hit]) if pw[hit] >= target else np.nan


def n_exact(d, alpha=0.05, target=0.80):
    """Continuous solution of power(n) = target, for a smooth curve."""
    f = lambda v: float(power_ttest(v, d, alpha)) - target
    if f(4.0) > 0:
        return 4.0
    return brentq(f, 4.0, 8000.0, xtol=1e-3)


def main():
    set_pub_style(base_font=8)

    colors = [RAMP(v) for v in np.linspace(0.30, 1.0, len(EFFECTS))]
    n_grid = np.logspace(np.log10(5), np.log10(600), 260)
    ticks = [5, 10, 20, 50, 100, 200, 500]

    fig = plt.figure(figsize=(7.4, 5.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.62, 1.0],
                  width_ratios=[1.0, 1.0, 1.06], hspace=0.50, wspace=0.30,
                  left=0.068, right=0.982, top=0.855, bottom=0.108)

    # ---- (a) power curves --------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    style_axes(ax)
    ax.axhspan(0.80, 1.02, color="#f3f6f4", lw=0, zorder=0)
    ax.axhline(0.80, color=TARGET_C, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(580, 0.815, "80% power target", fontsize=7.0, color=TARGET_C,
            va="bottom", ha="right")

    for d, col in zip(EFFECTS, colors):
        pw = power_ttest(n_grid, d)
        ax.plot(n_grid, pw, color=col, lw=1.7, zorder=4,
                label=f"$d$ = {d:.2f}")
        n80 = n_for_power(d)
        if np.isfinite(n80) and n80 <= n_grid[-1]:
            ax.plot([n80, n80], [0, 0.80], color=col, lw=0.7, ls=(0, (2, 2)),
                    alpha=0.55, zorder=2)
            ax.plot([n80], [0.80], marker="o", ms=4.2, mfc="white", mec=col,
                    mew=1.3, zorder=6)
            ax.annotate(f"{n80}", xy=(n80, 0.80), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6.6, color=col, fontweight="bold")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(5, 600)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("sample size per group  $n$   (log scale)", labelpad=2)
    ax.set_ylabel("power  $1 - \\beta$")
    ax.text(0.985, 0.045, "markers: $n$ reaching 80% power",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#555555", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.88))
    add_panel_label(ax, "(a)", x=-0.055, y=1.02, size=10)

    # ---- (b) required n vs effect size ------------------------------------
    axr = fig.add_subplot(gs[0, 2])
    style_axes(axr)
    d_grid = np.linspace(0.12, 1.25, 200)
    need = np.array([n_exact(d) for d in d_grid], dtype=float)
    axr.plot(d_grid, need, color="#1f5f8b", lw=1.7, zorder=4)
    axr.fill_between(d_grid, need, 4, color="#e4ecf3", lw=0, zorder=1)
    for d, col in zip(EFFECTS, colors):
        axr.plot([d], [n_for_power(d)], marker="o", ms=4.2, mfc="white",
                 mec=col, mew=1.3, zorder=6)
    axr.set_yscale("log")
    axr.set_ylim(4, 3000)
    axr.set_xlim(0.1, 1.28)
    yt = [10, 30, 100, 300, 1000, 3000]
    axr.yaxis.set_major_locator(FixedLocator(yt))
    axr.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
    axr.yaxis.set_minor_locator(FixedLocator([]))
    axr.set_xlabel("effect size  $d$", labelpad=2)
    axr.set_ylabel("required $n$ per group", labelpad=2)
    axr.text(0.97, 0.95, r"$n \propto d^{-2}$", transform=axr.transAxes,
             ha="right", va="top", fontsize=8.5, color="#1f5f8b")
    axr.text(0.97, 0.83, r"at 80% power, $\alpha$ = 0.05",
             transform=axr.transAxes, ha="right", va="top", fontsize=6.6,
             color="#666666")
    add_panel_label(axr, "(b)", x=-0.20, y=1.02, size=10)

    # ---- (c-e) sensitivity to alpha ---------------------------------------
    for k, alpha in enumerate([0.05, 0.01, 0.001]):
        axa = fig.add_subplot(gs[1, k])
        style_axes(axa)
        axa.axhline(0.80, color=TARGET_C, lw=0.8, ls=(0, (4, 3)), zorder=3)
        for d, col in zip(EFFECTS, colors):
            axa.plot(n_grid, power_ttest(n_grid, d, alpha), color=col,
                     lw=1.2, zorder=4)
        axa.set_xscale("log")
        axa.xaxis.set_major_locator(FixedLocator(ticks))
        axa.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        axa.xaxis.set_minor_locator(FixedLocator([]))
        axa.set_xlim(5, 600)
        axa.set_ylim(0, 1.02)
        axa.set_yticks(np.arange(0, 1.01, 0.25))
        axa.tick_params(labelsize=7)
        axa.set_xlabel("$n$ per group", labelpad=1)
        if k == 0:
            axa.set_ylabel("power")
        else:
            axa.tick_params(labelleft=False)
        n_med = n_for_power(0.50, alpha)
        axa.set_title(rf"$\alpha$ = {alpha:g}", fontsize=8.4,
                      fontweight="bold", color="#333333")
        axa.text(0.97, 0.06, f"$d$ = 0.50 needs $n$ = {n_med}",
                 transform=axa.transAxes, ha="right", va="bottom",
                 fontsize=6.5, color="#555555", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="none", alpha=0.88))
        add_panel_label(axa, f"({'cde'[k]})", x=-0.09 if k == 0 else -0.05,
                        y=1.06, size=10)

    handles = [plt.Line2D([], [], color=c, lw=1.8, label=f"$d$ = {d:.2f}")
               for d, c in zip(EFFECTS, colors)]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_title().set_fontsize(7.2)
    leg.get_title().set_color("#666666")

    savefig(fig, "133_power_analysis.png")


if __name__ == "__main__":
    main()
