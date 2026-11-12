"""Lorenz curves with Gini coefficients: income-inequality diagnostic panel.

(a) Lorenz curves for four economies against the 45-degree equality line,
    with the concentration area shaded and each Gini annotated;
(b) Gini trajectories 1990-2024;
(c) quintile income shares, derived analytically from the same curves.

Incomes are lognormal, so the Lorenz curve L(p) = Phi(Phi^-1(p) - sigma) is
exactly convex and monotone and G = 2*Phi(sigma/sqrt(2)) - 1 holds exactly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

ECONOMIES = [
    ("Nordic EU", 0.272),
    ("East Asia", 0.338),
    ("OECD average", 0.402),
    ("Latin America", 0.496),
]
COLORS = ["#2a7f7a", "#3b5b92", "#c9a227", "#b23a48"]
QUINT_COLORS = PALETTES["cb_sequential_blue"][1:6]


def sigma_from_gini(g):
    """Lognormal shape parameter reproducing a target Gini exactly."""
    return np.sqrt(2.0) * norm.ppf(0.5 * (g + 1.0))


def lorenz(p, g):
    s = sigma_from_gini(g)
    out = np.empty_like(p)
    inner = (p > 0) & (p < 1)
    out[~inner] = np.clip(p[~inner], 0.0, 1.0)
    out[inner] = norm.cdf(norm.ppf(p[inner]) - s)
    return out


def gini_paths(rng, years):
    """Plausible 1990-2024 Gini trajectories that end at the panel-(a) values."""
    tn = (years - years[0]) / (years[-1] - years[0])
    drifts = [0.048, -0.062, -0.018, 0.085]      # start-minus-end offsets
    curves = []
    for (_, g_end), d in zip(ECONOMIES, drifts):
        shape = d * (1.0 - tn) ** 1.25
        wig = np.cumsum(rng.normal(0, 0.0035, years.size))
        wig -= np.linspace(0, wig[-1], years.size)
        path = g_end + shape + wig
        curves.append(path)
    return curves


def panel_lorenz(ax):
    p = np.linspace(0, 1, 400)
    ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.545, 0.558, "line of perfect equality", rotation=39.5,
            fontsize=7, color="#6f6f6f", ha="left", va="bottom",
            rotation_mode="anchor")

    for (name, g), col in zip(ECONOMIES, COLORS):
        L = lorenz(p, g)
        ax.fill_between(p, L, p, color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(p, L, color=col, lw=1.6, zorder=3,
                label=f"{name}  (G = {g:.3f})")

    # concentration-area call-out on the most unequal curve
    g_hi = ECONOMIES[-1][1]
    ax.annotate("area A", xy=(0.62, 0.5 * (0.62 + lorenz(np.array([0.62]), g_hi)[0])),
                xytext=(0.30, 0.60), fontsize=7.5, color="#8c2f2f",
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#8c2f2f",
                                shrinkA=0, shrinkB=2))
    ax.text(0.985, 0.045, "G = A / (A + B)", fontsize=7.5, color="#333",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))
    ax.text(0.80, 0.16, "B", fontsize=8, color="#7a7a7a", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{v:.0f}" for v in np.arange(0, 101, 20)])
    ax.set_yticklabels([f"{v:.0f}" for v in np.arange(0, 101, 20)])
    ax.set_xlabel("Cumulative share of population (%)")
    ax.set_ylabel("Cumulative share of income (%)")
    ax.set_title("Lorenz curves, disposable income, 2024", fontsize=9,
                 loc="left", pad=6)
    ax.legend(loc="upper left", fontsize=7.2, handlelength=1.4,
              labelspacing=0.3, borderpad=0.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax)


def panel_gini_time(ax, rng):
    years = np.arange(1990, 2025)
    curves = gini_paths(rng, years)
    for (name, _), col, path in zip(ECONOMIES, COLORS, curves):
        ax.plot(years, path, color=col, lw=1.5)
        ax.plot(years[-1], path[-1], "o", color=col, ms=3.4, zorder=4)
    ax.axhspan(0.30, 0.40, color="#f0f0f0", zorder=0)
    ax.text(1991, 0.352, "moderate-inequality band", fontsize=6.6,
            color="#8a8a8a", va="center")
    ax.set_xlim(1988, 2032)
    ax.set_ylim(0.20, 0.60)
    ax.set_yticks(np.arange(0.20, 0.61, 0.10))
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylabel("Gini coefficient (0-1)")
    ax.set_xlabel("Year")
    ax.set_title("Trajectories, 1990-2024", fontsize=9, loc="left", pad=6)
    for (name, _), col, path in zip(ECONOMIES, COLORS, curves):
        ax.annotate(name, xy=(years[-1] + 0.9, path[-1]), fontsize=6.8,
                    color=col, va="center", ha="left", fontweight="bold")
    style_axes(ax)


def panel_shares(ax):
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    names = [n for n, _ in ECONOMIES]
    ypos = np.arange(len(names))[::-1]
    labels = ["Bottom 20%", "2nd quintile", "3rd quintile",
              "4th quintile", "Top 20%"]

    for y, (name, g) in zip(ypos, ECONOMIES):
        L = lorenz(edges, g)
        shares = np.diff(L) * 100.0
        left = 0.0
        for sh, col in zip(shares, QUINT_COLORS):
            ax.barh(y, sh, left=left, height=0.62, color=col,
                    edgecolor="white", lw=0.7)
            lum = np.mean(plt.matplotlib.colors.to_rgb(col))
            ax.text(left + sh / 2, y, f"{sh:.1f}", ha="center",
                    va="center", fontsize=7 if sh > 5 else 6.2,
                    color="white" if lum < 0.55 else "#1a1a1a")
            left += sh
        ratio = shares[-1] / shares[0]
        ax.text(101.5, y, f"S80/S20 = {ratio:.1f}", fontsize=7.2,
                va="center", ha="left", fontweight="bold", color="#333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Share of total disposable income (%)")
    ax.set_title("Quintile income shares implied by the Lorenz curves",
                 fontsize=9, loc="left", pad=6)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white", lw=0.6)
               for c in QUINT_COLORS]
    ax.legend(handles, labels, ncol=5, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.46, 1.16), frameon=False, handlelength=1.1,
              columnspacing=1.1, handletextpad=0.45)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.92],
                  height_ratios=[1.0, 0.72], wspace=0.34, hspace=0.50,
                  left=0.085, right=0.86, top=0.93, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_lorenz(ax_a)
    panel_gini_time(ax_b, rng)
    panel_shares(ax_c)

    add_panel_label(ax_a, "(a)", x=-0.16, y=1.04)
    add_panel_label(ax_b, "(b)", x=-0.18, y=1.04)
    # ax_c spans both columns, so the axes-fraction offset must be rescaled to
    # keep the label in the same figure-x column as (a) and (b).
    add_panel_label(ax_c, "(c)", x=-0.071, y=1.06)

    savefig(fig, "103_lorenz_gini.png")


if __name__ == "__main__":
    main()
