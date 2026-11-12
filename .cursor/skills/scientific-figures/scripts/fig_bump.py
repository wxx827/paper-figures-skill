"""Bump chart: ranking changes over time with smooth connectors.

Ranks come from an autocorrelated latent quality score, so counties drift up
and down instead of being reshuffled at random.  Smooth connectors, endpoint
labels on both sides and the rank printed inside each marker.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig

ENTITIES = ["Antu", "Fusong", "Changbai", "Baishan", "Jingyu", "Linjiang"]
TIMES = ["2000", "2005", "2010", "2015", "2020"]
COLORS = ["#2e7d5b", "#3d6b8e", "#c9a227", "#8c2f2f", "#d98c5f", "#6b5b95"]


def smoothstep(x0, y0, x1, y1, n=40):
    t = np.linspace(0, 1, n)
    s = 3 * t ** 2 - 2 * t ** 3
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * s


def main():
    set_pub_style()
    rng = np.random.default_rng(101)
    n_e, n_t = len(ENTITIES), len(TIMES)

    # autocorrelated latent score -> plausible, gradual rank changes
    score = rng.normal(0, 1, n_e)
    scores = np.zeros((n_e, n_t))
    for j in range(n_t):
        scores[:, j] = score
        score = 0.72 * score + rng.normal(0, 0.72, n_e)
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0) + 1

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for r in range(1, n_e + 1):
        ax.axhline(r, color="#f0f0f0", lw=0.8, zorder=0)

    for i in range(n_e):
        for j in range(n_t - 1):
            xs, ys = smoothstep(j, ranks[i, j], j + 1, ranks[i, j + 1])
            ax.plot(xs, ys, color=COLORS[i], lw=2.2, alpha=0.9, zorder=2,
                    solid_capstyle="round")
        ax.scatter(range(n_t), ranks[i], color=COLORS[i], s=110, zorder=3,
                   edgecolors="white", linewidths=1.1)
        for j in range(n_t):
            ax.text(j, ranks[i, j], str(ranks[i, j]), ha="center",
                    va="center", fontsize=6.2, color="white",
                    fontweight="bold", zorder=4)
        ax.text(-0.16, ranks[i, 0], ENTITIES[i], ha="right", va="center",
                fontsize=8.5, color=COLORS[i], fontweight="bold")
        ax.text(n_t - 1 + 0.16, ranks[i, -1], ENTITIES[i], ha="left",
                va="center", fontsize=8.5, color=COLORS[i], fontweight="bold")

    ax.set_xticks(range(n_t))
    ax.set_xticklabels(TIMES, fontsize=9)
    ax.set_yticks(range(1, n_e + 1))
    ax.set_yticklabels(range(1, n_e + 1), fontsize=8)
    ax.set_ylabel("Rank of ecosystem-quality index (1 = best)", fontsize=8.5)
    ax.set_ylim(n_e + 0.6, 0.4)
    ax.set_xlim(-1.05, n_t - 1 + 1.05)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Ranking dynamics of ecosystem quality, 2000-2020",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "19_bump.png")


if __name__ == "__main__":
    main()
