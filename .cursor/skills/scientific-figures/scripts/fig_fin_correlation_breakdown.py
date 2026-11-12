"""Correlation breakdown: diversification disappears when it is needed most.

(a) and (b) are the asset correlation matrices of a calm and a crisis window,
drawn on **one shared diverging colour scale** centred on zero - without that
the two panels would autoscale and the across-the-board rise, which is the
entire point of the figure, would be invisible.  (c) tracks the mean pairwise
correlation through a rolling window with the crisis period shaded; (d) shows
how much of the total variance the leading principal components absorb in each
regime.

All four panels are computed from one simulated single-factor return series,
so the matrices, the rolling curve and the eigenvalues cannot disagree.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

ASSETS = ["US eq", "EU eq", "EM eq", "HY credit", "IG credit",
          "Govt bond", "Commod.", "REIT"]
N_DAYS = 900
CALM = (40, 400)
CRISIS = (430, 540)
WIN = 60


def simulate(rng):
    """Single-factor returns whose factor loadings spike during the crisis."""
    n_a = len(ASSETS)
    base = np.array([0.62, 0.58, 0.64, 0.54, 0.42, -0.28, 0.38, 0.56])
    peak = np.array([0.88, 0.86, 0.90, 0.84, 0.72, 0.30, 0.70, 0.82])
    t = np.arange(N_DAYS)
    ramp = np.zeros(N_DAYS)
    c0, c1 = CRISIS
    ramp[c0 - 25:c0] = np.linspace(0.0, 1.0, 25)
    ramp[c0:c1] = 1.0
    ramp[c1:c1 + 120] = np.linspace(1.0, 0.0, 120) ** 0.8
    load = base[None, :] + (peak - base)[None, :] * ramp[:, None]

    factor = rng.normal(0, 1.0, N_DAYS)
    factor[c0 - 25:c1 + 60] *= 1.9                      # volatility spike
    idio = rng.normal(0, 1.0, (N_DAYS, n_a))
    ret = load * factor[:, None] + np.sqrt(
        np.clip(1.0 - load ** 2, 0.05, None)) * idio
    return ret * 0.011


def off_diag_mean(c):
    iu = np.triu_indices_from(c, k=1)
    return float(c[iu].mean())


def rolling_mean_corr(ret, win=WIN):
    out = np.full(ret.shape[0], np.nan)
    for i in range(win, ret.shape[0]):
        out[i] = off_diag_mean(np.corrcoef(ret[i - win:i].T))
    return out


def explained_variance(corr):
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    return 100.0 * ev / ev.sum()


def panel_matrix(ax, corr, title, cmap, norm):
    im = ax.imshow(corr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_aspect("equal")
    n = len(ASSETS)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(ASSETS, rotation=42, ha="right", fontsize=6.4)
    ax.set_yticklabels(ASSETS, fontsize=6.4)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, ("%.2f" % v).replace("0.", "."), ha="center",
                    va="center", fontsize=6.0,
                    color="white" if abs(v) > 0.62 else INK)
    ax.set_title("%s   mean off-diagonal $\\rho$ = %.2f"
                 % (title, off_diag_mean(corr)), fontsize=8.8,
                 fontweight="bold", pad=5)
    return im


def panel_rolling(ax, roll, calm_mu, crisis_mu):
    day = np.arange(roll.size)
    c0, c1 = CRISIS
    ax.axvspan(c0, c1, color=RED, alpha=0.10, lw=0, zorder=1)
    ax.text((c0 + c1) / 2.0, 0.42, "crisis window",
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            rotation=90, fontsize=6.5, color=RED)
    ax.plot(day, roll, color=BLUE, lw=1.25, zorder=4,
            label="%d-day rolling mean pairwise $\\rho$" % WIN)
    ax.axhline(calm_mu, color=SUB, lw=0.8, ls=(0, (3, 2)), zorder=3,
               label="calm regime mean %.2f" % calm_mu)
    ax.axhline(crisis_mu, color=RED, lw=0.9, ls=(0, (3, 2)), zorder=3,
               label="crisis regime mean %.2f" % crisis_mu)
    ax.set_xlabel("trading day (days)")
    ax.set_ylabel("mean pairwise $\\rho$ (-)")
    ax.set_xlim(0, roll.size)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=1.5,
              borderaxespad=0.25)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def panel_pca(ax, ev_calm, ev_crisis, k=5):
    x = np.arange(k)
    ax.bar(x - 0.19, ev_calm[:k], width=0.36, color=BLUE, alpha=0.88, lw=0,
           label="calm regime")
    ax.bar(x + 0.19, ev_crisis[:k], width=0.36, color=RED, alpha=0.88, lw=0,
           label="crisis regime")
    for xi, (a, b) in enumerate(zip(ev_calm[:k], ev_crisis[:k])):
        ax.text(xi - 0.19, a + 1.4, "%.0f" % a, ha="center", va="bottom",
                fontsize=6.4, color=BLUE)
        ax.text(xi + 0.19, b + 1.4, "%.0f" % b, ha="center", va="bottom",
                fontsize=6.4, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(["PC%d" % (i + 1) for i in range(k)], fontsize=6.8)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_ylim(0, max(ev_crisis[0], ev_calm[0]) * 1.28)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.6, handlelength=1.2,
              borderaxespad=0.25)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(572)
    ret = simulate(rng)

    corr_calm = np.corrcoef(ret[CALM[0]:CALM[1]].T)
    corr_crisis = np.corrcoef(ret[CRISIS[0]:CRISIS[1]].T)

    cmap = get_cmap("cb_diverging")
    # One symmetric Normalize for both panels: the whole claim of the figure
    # is that the crisis matrix is uniformly darker than the calm one.
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig = plt.figure(figsize=(9.8, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.0, 0.56], hspace=0.42,
                          wspace=0.30, left=0.085, right=0.885, top=0.895,
                          bottom=0.095)
    ax_calm = fig.add_subplot(gs[0, 0])
    ax_cri = fig.add_subplot(gs[0, 1])
    ax_roll = fig.add_subplot(gs[1, 0])
    ax_pca = fig.add_subplot(gs[1, 1])

    panel_matrix(ax_calm, corr_calm, "Calm regime", cmap, norm)
    im = panel_matrix(ax_cri, corr_crisis, "Crisis regime", cmap, norm)
    cb = safe_colorbar(fig, im, ax_cri, label="correlation $\\rho$ (-)",
                       width=0.12, pad=0.10,
                       ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.ax.text(0.5, -0.10, "shared\nscale", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=6.4, color=MUTED,
               linespacing=1.25)

    roll = rolling_mean_corr(ret)
    calm_mu = off_diag_mean(corr_calm)
    crisis_mu = off_diag_mean(corr_crisis)
    panel_rolling(ax_roll, roll, calm_mu, crisis_mu)

    ev_calm = explained_variance(corr_calm)
    ev_crisis = explained_variance(corr_crisis)
    panel_pca(ax_pca, ev_calm, ev_crisis)


    panel_labels_inches(fig, [ax_calm, ax_cri, ax_roll, ax_pca],
                        dx=-0.58, dy=0.08)
    savefig(fig, "572_fin_correlation_breakdown.png")


if __name__ == "__main__":
    main()
