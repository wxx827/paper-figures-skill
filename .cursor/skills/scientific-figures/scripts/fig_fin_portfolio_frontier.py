"""Mean-variance frontier for a six-asset synthetic allocation problem.

(a) 4000 randomly weighted long-only portfolios coloured by Sharpe ratio, the
    unconstrained efficient frontier that envelops them, the capital market
    line through the tangency portfolio, and the single assets for scale;
    (b) the tangency weights, negative bars being short positions;
    (c) how much of total portfolio risk each holding actually contributes,
    which is not the same ranking as the weights.

Frontier, tangency point and both bar panels come from one covariance matrix,
so the annotated Sharpe ratio is the one plotted.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import (set_pub_style, savefig, style_axes, get_cmap,
                      add_colorbar, annotate_box)
from figkit import panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

ASSETS = ["US eq", "Intl eq", "EM eq", "IG credit", "Govt bond", "Commod."]
MU = np.array([7.5, 6.6, 9.2, 4.3, 2.7, 5.6])          # % p.a.
VOL = np.array([15.5, 17.0, 22.5, 6.4, 4.3, 19.0])     # % p.a.
RF = 2.0                                               # % p.a.
N_CLOUD = 4000

CORR = np.array([
    [1.00, 0.82, 0.72, 0.28, -0.22, 0.34],
    [0.82, 1.00, 0.75, 0.30, -0.18, 0.36],
    [0.72, 0.75, 1.00, 0.34, -0.12, 0.44],
    [0.28, 0.30, 0.34, 1.00, 0.42, 0.16],
    [-0.22, -0.18, -0.12, 0.42, 1.00, -0.08],
    [0.34, 0.36, 0.44, 0.16, -0.08, 1.00],
])


def covariance():
    return CORR * np.outer(VOL, VOL)


def frontier(cov, mus):
    inv = np.linalg.inv(cov)
    one = np.ones(MU.size)
    a = one @ inv @ one
    b = one @ inv @ MU
    c = MU @ inv @ MU
    d = a * c - b ** 2
    var = (a * mus ** 2 - 2 * b * mus + c) / d
    return np.sqrt(var), (b / a, np.sqrt(1.0 / a))


def tangency(cov):
    inv = np.linalg.inv(cov)
    w = inv @ (MU - RF)
    w = w / w.sum()
    mu = float(w @ MU)
    sig = float(np.sqrt(w @ cov @ w))
    return w, mu, sig


def panel_frontier(fig, ax, cloud, cov, w_t, mu_t, sig_t, mv):
    sig_c, mu_c, sharpe_c = cloud
    cmap = get_cmap("cb_sequential_blue")
    sc = ax.scatter(sig_c, mu_c, c=sharpe_c, cmap=cmap, s=4.5, lw=0,
                    alpha=0.72, zorder=2, rasterized=True)
    add_colorbar(fig, sc, ax, label="Sharpe ratio (-)", fraction=0.040,
                 pad=0.02, shrink=0.86)

    mus = np.linspace(min(mu_c.min(), MU.min()) - 0.6, MU.max() + 1.4, 220)
    sig_f, _ = frontier(cov, mus)
    eff = mus >= mv[0]
    ax.plot(sig_f[eff], mus[eff], color=INK, lw=1.6, zorder=6,
            label="efficient frontier")
    ax.plot(sig_f[~eff], mus[~eff], color=INK, lw=1.0, ls=(0, (3, 2)),
            alpha=0.55, zorder=6, label="inefficient branch")

    x_cml = np.array([0.0, sig_c.max() * 1.12])
    ax.plot(x_cml, RF + (mu_t - RF) / sig_t * x_cml, color=RED, lw=1.3,
            ls=(0, (5, 2.5)), zorder=6,
            label="capital market line, $r_f$ = %.1f%%" % RF)

    ax.plot([sig_t], [mu_t], marker="*", ms=13, mfc=RED, mec="white",
            mew=0.8, lw=0, zorder=8, label="tangency portfolio")
    ax.plot([mv[1]], [mv[0]], marker="*", ms=12, mfc=GREEN, mec="white",
            mew=0.8, lw=0, zorder=8, label="minimum variance portfolio")
    ax.plot(VOL, MU, marker="D", ms=4.2, mfc=GOLD, mec="#7a5f14", mew=0.6,
            lw=0, zorder=7, label="single assets")
    for name, s, m in zip(ASSETS, VOL, MU):
        ax.annotate(name, xy=(s, m), xytext=(s + 0.55, m - 0.32),
                    fontsize=6.6, color="#7a5f14", ha="left", va="top")

    sharpe_t = (mu_t - RF) / sig_t
    annotate_box(ax, "tangency: $\\mu$ = %.1f%%, $\\sigma$ = %.1f%%,\n"
                     "Sharpe = %.2f\nmin-var: $\\sigma$ = %.1f%% at "
                     "$\\mu$ = %.1f%%"
                 % (mu_t, sig_t, sharpe_t, mv[1], mv[0]),
                 loc="lower right", fontsize=6.8, margin=0.025)

    ax.set_xlabel("volatility $\\sigma$ (% p.a.)")
    ax.set_ylabel("expected return $\\mu$ (% p.a.)")
    ax.set_xlim(0, max(sig_c.max(), VOL.max()) * 1.10)
    ax.set_ylim(min(mu_c.min(), MU.min()) - 1.0, MU.max() + 1.6)
    ax.grid(lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.9, handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.32)
    style_axes(ax)
    return sharpe_t


def panel_weights(ax, w):
    x = np.arange(w.size)
    cols = [BLUE if v >= 0 else RED for v in w]
    ax.bar(x, 100.0 * w, width=0.62, color=cols, alpha=0.9, lw=0)
    ax.axhline(0.0, color="#808080", lw=0.7)
    for xi, v in zip(x, 100.0 * w):
        ax.text(xi, v + (1.6 if v >= 0 else -1.6), "%.0f" % v, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.5,
                color=INK if v >= 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS, rotation=28, ha="right", fontsize=6.6)
    ax.set_ylabel("weight (%)")
    lo, hi = (100.0 * w).min(), (100.0 * w).max()
    ax.set_ylim(min(lo * 1.45, -6.0), hi * 1.28)
    ax.set_title("Tangency portfolio weights (negative = short)",
                 fontsize=8.6, fontweight="bold", pad=4)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def panel_risk(ax, w, cov):
    sig = float(np.sqrt(w @ cov @ w))
    rc = w * (cov @ w) / sig                      # marginal risk x weight
    share = 100.0 * rc / rc.sum()
    x = np.arange(w.size)
    cols = [RED if v < 0 else BLUE for v in share]
    ax.bar(x, share, width=0.62, color=cols, alpha=0.9, lw=0)
    ax.axhline(0.0, color="#808080", lw=0.7)
    for xi, v in zip(x, share):
        ax.text(xi, v + (1.8 if v >= 0 else -1.8), "%.0f%%" % v, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.5,
                color=INK if v >= 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS, rotation=28, ha="right", fontsize=6.6)
    ax.set_ylabel("risk contribution (%)")
    ax.set_ylim(min(share.min() * 1.5, -8.0), share.max() * 1.30)
    ax.set_title("Risk contribution to $\\sigma$ = %.1f%% p.a." % sig,
                 fontsize=8.6, fontweight="bold", pad=4)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(575)
    cov = covariance()

    w_cloud = rng.dirichlet(np.full(MU.size, 0.85), N_CLOUD)
    mu_c = w_cloud @ MU
    sig_c = np.sqrt(np.einsum("ij,jk,ik->i", w_cloud, cov, w_cloud))
    sharpe_c = (mu_c - RF) / sig_c

    w_t, mu_t, sig_t = tangency(cov)
    _, mv = frontier(cov, np.array([0.0]))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], hspace=0.62,
                          wspace=0.34, left=0.062, right=0.975, top=0.900,
                          bottom=0.115)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    sharpe_t = panel_frontier(fig, ax_main, (sig_c, mu_c, sharpe_c), cov,
                              w_t, mu_t, sig_t, mv)
    panel_weights(ax_w, w_t)
    panel_risk(ax_r, w_t, cov)

    best = sharpe_c.max()

    panel_labels_inches(fig, [ax_main, ax_w, ax_r], dx=-0.42, dy=0.06)
    savefig(fig, "575_fin_portfolio_frontier.png")


if __name__ == "__main__":
    main()
