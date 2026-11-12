"""Scatter with marginal densities + threshold split + violin inset.

2x2 grid of joint plots.  Each cell carries a top marginal histogram split by
group, a right marginal histogram, a piecewise scatter separated by a fitted
threshold, the two segment fits with confidence bands, and an inset violin
comparing the response either side of the break (with a permutation p-value).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

C_BEFORE, C_AFTER = PALETTES["teal_rust"]
BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d5d5d5",
           linewidth=0.5, alpha=0.92)

CFG = [
    ("Nitrogen deposition (kg ha$^{-1}$ yr$^{-1}$)", -1.71, 0.42, 0.11),
    ("Growing-season length (z-score)", 0.44, 0.10, 0.34),
    ("Soil moisture (z-score)", -0.13, 0.34, 0.13),
    ("Grazing intensity (z-score)", -1.09, 0.38, 0.12),
]


def perm_pvalue(a, b, rng, n_perm=2000):
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    k = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:k].mean() - pool[k:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def fit_band(ax, x, y, color):
    """Least-squares fit plus a 95% mean-prediction band."""
    if x.size < 6:
        return
    n = x.size
    b = np.polyfit(x, y, 1)
    resid = y - np.polyval(b, x)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    xr = np.linspace(x.min(), x.max(), 60)
    yr = np.polyval(b, xr)
    sxx = np.sum((x - x.mean()) ** 2)
    se = s * np.sqrt(1.0 / n + (xr - x.mean()) ** 2 / sxx)
    ax.fill_between(xr, yr - 1.96 * se, yr + 1.96 * se, color=color,
                    alpha=0.18, linewidth=0, zorder=3)
    ax.plot(xr, yr, color=color, lw=1.5, zorder=4)


def one_joint(fig, subspec, rng, xlabel, thr, slope_lo, slope_hi, label):
    gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subspec, width_ratios=[4.4, 1.0],
        height_ratios=[1.0, 4.4], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    n = 240
    x = rng.normal(thr, 0.75, n)
    y = (19.40 + np.where(x < thr, slope_lo * (x - thr), slope_hi * (x - thr))
         + rng.normal(0, 0.085, n))
    before, after = x < thr, x >= thr

    ax.scatter(x[before], y[before], s=15, c=C_BEFORE, alpha=0.7,
               edgecolors="white", linewidths=0.3, zorder=2)
    ax.scatter(x[after], y[after], s=15, c=C_AFTER, alpha=0.7,
               edgecolors="white", linewidths=0.3, zorder=2)
    fit_band(ax, x[before], y[before], C_BEFORE)
    fit_band(ax, x[after], y[after], C_AFTER)
    ax.axvline(thr, color="#555555", ls=(0, (2, 2)), lw=1.0, zorder=1)

    span = y.max() - y.min()
    ax.set_ylim(y.min() - 0.30 * span, y.max() + 0.10 * span)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Ecosystem multifunctionality index", fontsize=8)
    ax.tick_params(labelsize=7)
    style_axes(ax)
    ax.text(0.03, 0.97, f"threshold = {thr:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="#444444", bbox=BOX,
            zorder=6)

    bins = np.linspace(x.min(), x.max(), 26)
    ax_top.hist([x[before], x[after]], bins=bins, stacked=True,
                color=[C_BEFORE, C_AFTER], edgecolor="white", linewidth=0.3,
                alpha=0.85)
    ax_top.axis("off")
    ybins = np.linspace(y.min(), y.max(), 26)
    ax_right.hist([y[before], y[after]], bins=ybins, stacked=True,
                  orientation="horizontal", color=[C_BEFORE, C_AFTER],
                  edgecolor="white", linewidth=0.3, alpha=0.85)
    ax_right.axis("off")

    # inset violin sits in the empty lower-right corner of the joint panel
    axins = ax.inset_axes([0.60, 0.055, 0.37, 0.30])
    parts = axins.violinplot([y[before], y[after]], showmeans=False,
                             showextrema=False, widths=0.8)
    for pc, col in zip(parts["bodies"], [C_BEFORE, C_AFTER]):
        pc.set_facecolor(col)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.4)
        pc.set_alpha(0.8)
    for i, arr in enumerate([y[before], y[after]], start=1):
        axins.plot([i], [arr.mean()], marker="o", ms=2.6, color="white",
                   mec="#333333", mew=0.5, zorder=5)
    p = perm_pvalue(y[before].copy(), y[after].copy(), rng)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    axins.set_title(f"permutation {ptxt}", fontsize=6, pad=2, color="#333333")
    axins.set_xticks([1, 2])
    axins.set_xticklabels(["below", "above"], fontsize=6)
    axins.set_yticks([])
    axins.tick_params(length=1.5, pad=1)
    axins.patch.set_alpha(0.0)
    for side in ("top", "right", "left"):
        axins.spines[side].set_visible(False)
    axins.spines["bottom"].set_linewidth(0.5)

    add_panel_label(ax_top, label, x=-0.11, y=0.75, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(9.2, 7.25))
    outer = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.30,
                             left=0.07, right=0.985, top=0.90, bottom=0.09)
    for i, (xl, thr, s_lo, s_hi) in enumerate(CFG):
        one_joint(fig, outer[i // 2, i % 2], rng, xl, thr, s_lo, s_hi,
                  f"({chr(97 + i)})")

    handles = [
        Line2D([], [], marker="o", ls="none", color=C_BEFORE, ms=4.5,
               mec="white", mew=0.4, label="Below threshold"),
        Line2D([], [], marker="o", ls="none", color=C_AFTER, ms=4.5,
               mec="white", mew=0.4, label="Above threshold"),
        Line2D([], [], color="#555555", ls=(0, (2, 2)), lw=1.0,
               label="Fitted threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005), columnspacing=1.6)
    savefig(fig, "02_marginal_violin.png")


if __name__ == "__main__":
    main()
