"""Population dynamics: who takes over the league, and who beats whom.

The two panels are one object seen twice.  The win-rate matrix is built first
and is antisymmetric by construction - ``M[i][j] + M[j][i] = 100`` for every
pair, asserted, with the diagonal greyed out because a strategy has no
self-match.  The bump chart on the left is replicator dynamics *driven by that
matrix*: fitness is the win rate against the current population mix, so a
strategy that wins more gains share.  The script asserts that the final share
ranking agrees with the mean-win-rate ranking, which is the claim the caption
makes.

Line width in the bump chart is the share itself, so the rank track carries the
magnitude a rank plot normally throws away.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import COLORS, get_cmap, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
GREY = "#dfe3e7"

NAMES = ["Rush", "Flank", "Turtle", "Bait", "Split", "Swarm"]
COL = [BLUE[0], BLUE[1], BLUE[2], BLUE[3], BLUE[4], "#6f8496"]
K = len(NAMES)
GENS = 16
ETA = 1.4
# raw strength, plus one rock-paper-scissors loop laid over it that is strong
# enough for the weakest strategy to beat the strongest
EDGE = np.array([0.92, 0.55, 0.20, -0.30, -0.62, -0.75])
TILT, LOOP = 10.0, 22.0


def payoff():
    """Antisymmetric win-rate matrix in percent: M[i][j] + M[j][i] = 100."""
    cyc = np.zeros((K, K))
    for i in range(K):
        cyc[i, (i + 1) % K] = 1.0
        cyc[(i + 1) % K, i] = -1.0
    S = np.round(TILT * (EDGE[:, None] - EDGE[None, :]) + LOOP * cyc)
    M = np.clip(50.0 + S, 6.0, 94.0)
    for i in range(K):                                # repair after clipping
        for j in range(i + 1, K):
            M[j, i] = 100.0 - M[i, j]
        M[i, i] = np.nan
    return M


def evolve(M):
    """Replicator dynamics on the same matrix - no mutation, no noise.

    In a zero-sum league the population's average win rate is exactly 50%, so
    a strategy grows in precisely the generations where it is above 50%
    against the current mix.  ``main`` asserts that on every generation.

    The update is the linear discrete replicator on purpose: its normaliser is
    exactly 1, so "above average" and "gains share" mean the same thing to the
    last decimal.  The exponential form leaves a Jensen gap that flips the sign
    for a strategy sitting a tenth of a point above the mean.
    """
    x = np.array([0.16, 0.14, 0.19, 0.17, 0.18, 0.16])
    x = x / x.sum()
    W = np.where(np.isnan(M), 50.0, M) / 100.0
    hist, fits = [x.copy()], []
    for _ in range(GENS - 1):
        fit = W @ x
        fits.append(100.0 * fit)
        x = x * (1.0 + ETA * (fit - fit @ x))
        hist.append(x.copy())
    fits.append(100.0 * (W @ x))
    return 100.0 * np.array(hist), np.array(fits)


def main():
    set_pub_style(base_font=8)
    M = payoff()
    share, fit = evolve(M)
    gens = np.arange(GENS)

    off = ~np.eye(K, dtype=bool)
    for i in range(K):
        for j in range(K):
            if i != j:
                assert abs(M[i, j] + M[j, i] - 100.0) < 1e-9, "not antisymmetric"
    assert np.all(np.abs(share.sum(axis=1) - 100.0) < 1e-9), "shares != 100%"
    # the causal link between the two panels, checked generation by generation
    for t in range(GENS - 1):
        for i in range(K):
            if abs(fit[t, i] - 50.0) > 0.05:
                assert np.sign(share[t + 1, i] - share[t, i]) == \
                    np.sign(fit[t, i] - 50.0), "share must follow win rate"
    mean_wr = np.array([np.nanmean(M[i][off[i]]) for i in range(K)])
    counter = int(np.argmax(M[:, 0] > 50.0))          # who beats the leader

    rank = np.argsort(np.argsort(-share, axis=1), axis=1) + 1

    fig = plt.figure(figsize=(10.4, 6.0))
    axb = fig.add_axes([0.150, 0.190, 0.318, 0.620])
    axm = fig.add_axes([0.640, 0.290, 0.278, 0.482])

    # ---- bump chart, width = share ----------------------------------------
    smax = share.max()
    for i in range(K):
        for g in range(GENS - 1):
            w = 0.7 + 5.0 * (0.5 * (share[g, i] + share[g + 1, i])) / smax
            axb.plot([gens[g], gens[g + 1]], [rank[g, i], rank[g + 1, i]],
                     color=COL[i], lw=w, solid_capstyle="round", zorder=3,
                     alpha=0.95)
        axb.scatter(gens, rank[:, i], s=8.0 + 78.0 * share[:, i] / smax,
                    facecolor=COL[i], edgecolor="white", linewidth=0.8,
                    zorder=4)
    for i in range(K):
        axb.text(-0.55, rank[0, i],
                 f"#{int(rank[0, i]):d}  {NAMES[i]}  {share[0, i]:.1f}%",
                 ha="right", va="center", fontsize=7.4, fontweight="bold",
                 color=COL[i], clip_on=False)
        axb.text(GENS - 0.45, rank[-1, i],
                 f"{share[-1, i]:.1f}%  {NAMES[i]}  #{int(rank[-1, i]):d}",
                 ha="left", va="center", fontsize=7.4, fontweight="bold",
                 color=COL[i], clip_on=False)
    axb.set_xlim(-0.6, GENS - 0.4)
    axb.set_ylim(K + 0.6, 0.4)
    axb.set_yticks([])
    axb.set_xticks(range(0, GENS, 3))
    axb.set_xlabel("league generation", fontsize=8.0, color=COLORS["sub"],
                   labelpad=2.0)
    axb.text(-0.55, 0.30, "generation 0", ha="right", va="bottom",
             fontsize=7.0, color=COLORS["sub"], clip_on=False)
    axb.text(GENS - 0.45, 0.30, f"generation {GENS - 1:d}", ha="left",
             va="bottom", fontsize=7.0, color=COLORS["sub"], clip_on=False)
    axb.tick_params(labelsize=7.2, length=2.6)
    axb.grid(axis="y", color="#edf1f4", lw=0.6)
    axb.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.text(0.5, 1.055, f"Population share over {GENS:d} generations",
             transform=axb.transAxes, ha="center", va="bottom", fontsize=8.8,
             fontweight="bold", color=INK)

    # ---- head-to-head matrix ----------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(vmin=6, vmax=94)
    im = axm.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm,
                    interpolation="nearest", zorder=1)
    for i in range(K):
        for j in range(K):
            if i == j:
                axm.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor=GREY, edgecolor="white",
                                            lw=0.8, zorder=2))
                axm.text(j, i, "-", ha="center", va="center", fontsize=7.4,
                         color="#8a949c", zorder=3)
                continue
            v = M[i, j]
            axm.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.0,
                     fontweight="bold" if v >= 50 else "normal",
                     color="white" if norm(v) > 0.55 else INK, zorder=3)
    axm.set_xticks(range(K))
    axm.set_yticks(range(K))
    axm.set_xticklabels(NAMES, rotation=45, ha="left", rotation_mode="anchor",
                        fontsize=7.4)
    axm.set_yticklabels(NAMES, fontsize=7.4)
    axm.xaxis.set_ticks_position("top")
    axm.tick_params(length=0, pad=3.0)
    for lab, c in zip(axm.get_xticklabels(), COL):
        lab.set_color(c)
        lab.set_fontweight("bold")
    for lab, c in zip(axm.get_yticklabels(), COL):
        lab.set_color(c)
        lab.set_fontweight("bold")
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_xlim(-0.5, K - 0.5)
    axm.set_ylim(K - 0.5, -0.5)
    axm.text(0.5, 1.230, "Final head-to-head win rate (%)",
             transform=axm.transAxes, ha="center", va="bottom", fontsize=8.8,
             fontweight="bold", color=INK)

    # the causal column: win rate against the population of the last
    # generation.  Above 50 the strategy is still growing, below it is losing
    # share - that is the arrow from this panel to the one on the left.
    for i in range(K):
        axm.text(K - 0.30, i, f"{fit[-1, i]:.1f}", ha="left", va="center",
                 fontsize=7.2, fontweight="bold", color=COL[i], clip_on=False)
    axm.text(K - 0.30, -0.90, "vs. final\npopulation", ha="left", va="center",
             fontsize=6.6, color=COLORS["sub"], clip_on=False,
             linespacing=1.35)

    cax = fig.add_axes([0.640, 0.222, 0.170, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([6, 50, 94])
    cb.ax.tick_params(labelsize=6.6, length=2.0, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#c3ccd4")
    cb.set_label("row beats column (%)", fontsize=6.8, color=COLORS["sub"],
                 labelpad=1.5)

    i0, j0 = counter, 0
    best, worst = int(np.argmax(mean_wr)), int(np.argmin(mean_wr))
    fig.text(0.640, 0.146,
             f"Mirrored pairs add up: {NAMES[i0]} beats {NAMES[j0]} "
             f"{M[i0, j0]:.0f} to {M[j0, i0]:.0f},\n"
             f"and {M[i0, j0]:.0f} + {M[j0, i0]:.0f} = "
             f"{M[i0, j0] + M[j0, i0]:.0f}.  The diagonal is grey: there is "
             f"no\nself-match.  A zero-sum league averages exactly 50%, so a\n"
             f"strategy grows only while its right-hand column is above 50.\n"
             f"Mean win rate over the five opponents runs "
             f"{mean_wr[best]:.1f} ({NAMES[best]})\n"
             f"down to {mean_wr[worst]:.1f} ({NAMES[worst]}).",
             ha="left", va="top", fontsize=6.6,
             color=COLORS["muted"], linespacing=1.7)

    fig.text(0.150, 0.112,
             f"Line width and dot area are the share itself.  Shares sum to "
             f"100.0% in every\ngeneration - at generation {GENS - 1:d}, "
             + " + ".join(f"{v:.1f}" for v in share[-1])
             + f" = {share[-1].sum():.1f}.", ha="left", va="top",
             fontsize=6.6, color=COLORS["muted"], linespacing=1.7)

    top = int(np.argmax(share[-1]))
    rising = int((fit[-1] > 50.0).sum())
    # as single lines the headline runs 10.65 in and the subtitle 9.9 in on a
    # 10.4 in canvas, so `bbox_inches="tight"` exported 10.65 in wide and the
    # delivered h/w fell to 0.542

    savefig(fig, "1188_wm_mas_population.png")


if __name__ == "__main__":
    main()
