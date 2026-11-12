"""Human evaluation of generated rollouts: preference bars and a win-rate matrix.

Top: five head-to-head studies against our model, each shown as a 100 % stacked
Win / Tie / Lose bar with the percentages printed inside the segments.  Bottom:
the full round-robin, cell (i, j) being how often raters preferred method i over
method j.  The diagonal is greyed out and stars mark cells where a two-sided
binomial test rejects the 50 % null.

Synthetic data lives in ``gen_votes`` only -- swap that one function for a real
loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from pubstyle import (set_pub_style, savefig, style_axes, get_cmap,
                      add_colorbar, sig_stars)

METHODS = ["IRIS", "DreamerV3", "Genie-style", "OpenVLA-WM", "Ours (WM-8)"]
SHORT = ["IRIS", "DreamerV3", "Genie", "OpenVLA-WM", "Ours"]
# latent quality on a Bradley-Terry style scale; drives every number below
QUALITY = np.array([0.30, 0.62, 0.70, 0.84, 1.32])
HUMAN = "Human demo"
HUMAN_QUALITY = 1.25  # the reference ceiling; rated, but not in the matrix
# Rated pairs per comparison.  Keep this a divisor of 1000: one vote is then
# worth exactly 0.4 %, the bar labels are exact at one decimal, and a reader
# dividing the printed win by (win + lose) lands on the printed matrix cell
# rather than 0.1 away from it.
N_PAIR = 250
TIE_RATE = 0.18

WIN, TIE, LOSE = "#3b5b92", "#b3b3b3", "#b23a48"
GRID = "#ededed"


def _rate_pair(rng, q_row, q_col):
    """One head-to-head study: integer Win / Tie / Lose counts over N_PAIR."""
    p_tie = float(np.clip(TIE_RATE * rng.normal(1.0, 0.05), 0.05, 0.35))
    bt = q_row / (q_row + q_col)
    w, t, l = rng.multinomial(N_PAIR, [(1.0 - p_tie) * bt, p_tie,
                                       (1.0 - p_tie) * (1.0 - bt)])
    return int(w), int(t), int(l)


def gen_votes(seed=1502):
    """(win-rate %, win counts, decisive counts, {opponent: (win, tie, lose)}).

    Each unordered pair of methods is rated exactly once, and both panels then
    read that single draw: the bars show Win / Tie / Lose as shares of all
    ``N_PAIR`` pairs, the matrix shows wins as a share of the decisive ones.
    So the Ours row of the matrix is ``win / (win + lose)`` of the bars above
    it by construction, and the matrix is antisymmetric to the last digit.
    """
    rng = np.random.default_rng(seed)
    n = len(METHODS)
    ours = n - 1
    wins = np.zeros((n, n), dtype=int)
    ties = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            w, t, l = _rate_pair(rng, QUALITY[i], QUALITY[j])
            wins[i, j], wins[j, i] = w, l
            ties[i, j] = ties[j, i] = t

    decisive = wins + wins.T
    off = ~np.eye(n, dtype=bool)
    wr = np.full((n, n), np.nan)
    wr[off] = 100.0 * wins[off] / decisive[off]

    head = {}
    for j in range(ours):
        head[METHODS[j]] = (100.0 * wins[ours, j] / N_PAIR,
                            100.0 * ties[ours, j] / N_PAIR,
                            100.0 * wins[j, ours] / N_PAIR)
    w, t, l = _rate_pair(rng, QUALITY[ours], HUMAN_QUALITY)
    head[HUMAN] = (100.0 * w / N_PAIR, 100.0 * t / N_PAIR,
                   100.0 * l / N_PAIR)
    return wr, wins, decisive, head


def stacked_bars(ax, head):
    labels = list(head.keys())
    y = np.arange(len(labels))[::-1]
    for k, name in enumerate(labels):
        w, t, l = head[name]
        left = 0.0
        for v, col in ((w, WIN), (t, TIE), (l, LOSE)):
            ax.barh(y[k], v, left=left, height=0.62, color=col, lw=0.0,
                    zorder=3)
            if v >= 6.0:
                ax.text(left + v / 2.0, y[k], f"{v:.1f}%", ha="center",
                        va="center", fontsize=7.2, zorder=4,
                        color="#1a1a1a" if col == TIE else "white",
                        fontweight="bold" if col == WIN else "normal")
            left += v

    ax.axvline(50.0, color="#555555", lw=0.9, ls=(0, (3.0, 2.2)), zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Ours  vs  {n}" for n in labels], fontsize=8.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(y.min() - 0.62, y.max() + 0.62)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of rater preferences", fontsize=8.4)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    style_axes(ax)
    for lab in ax.get_yticklabels():
        lab.set_color("#333333")


def matrix(fig, ax, wr, wins, decisive, norm, cmap):
    n = len(METHODS)
    masked = np.ma.masked_invalid(wr)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", zorder=2)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#e6e6e6", lw=0, zorder=3))
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=8.0,
                        color="#8a8a8a", zorder=4)
                continue
            v = wr[i, j]
            # Tested on the same decisive pairs the rate is computed from, so
            # the stars can never disagree with the number above them.
            p = stats.binomtest(int(wins[i, j]), int(decisive[i, j]),
                                0.5).pvalue
            s = sig_stars(p, ns="")
            dark = norm(v) > 0.58
            ax.text(j, i - (0.10 if s else 0.0), f"{v:.1f}", ha="center",
                    va="center", fontsize=8.0, zorder=4,
                    color="white" if dark else "#1a1a1a",
                    fontweight="bold" if v > 50 else "normal")
            if s:
                ax.text(j, i + 0.26, s, ha="center", va="center", fontsize=7.6,
                        color="white" if dark else "#1a1a1a",
                        fontweight="bold", zorder=4)

    ax.set_xticks(range(n))
    ax.set_xticklabels(SHORT, fontsize=7.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SHORT, fontsize=7.8)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("... preferred over this method (column)", fontsize=8.0)
    ax.set_ylabel("This method (row) ...", fontsize=8.0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.3, zorder=3)
    ax.tick_params(length=0, pad=3.0)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main():
    set_pub_style(base_font=8)
    wr, wins, decisive, head = gen_votes()
    # symmetric colour range about the 50 % null, shared by every cell
    norm = Normalize(vmin=0.0, vmax=100.0)
    # blue sequential, so "deeper blue = the row method won" matches the bars
    cmap = get_cmap("cb_sequential_blue")

    fig = plt.figure(figsize=(7.2, 4.5))
    ax_b = fig.add_axes([0.175, 0.665, 0.795, 0.235])
    ax_m = fig.add_axes([0.175, 0.105, 0.455, 0.395])

    stacked_bars(ax_b, head)
    im = matrix(fig, ax_m, wr, wins, decisive, norm, cmap)

    ax_b.set_title(f"Pairwise human preference on {N_PAIR} rated clip pairs",
                   fontsize=9.4, fontweight="bold", pad=5.0)
    ax_m.set_title("Round-robin win rate (%)", fontsize=9.4,
                   fontweight="bold", pad=5.0)

    handles = [Patch(facecolor=WIN, label="Ours wins"),
               Patch(facecolor=TIE, label="Tie"),
               Patch(facecolor=LOSE, label="Ours loses")]

    cb = add_colorbar(fig, im, ax_m, label="win rate (%)",
                      orientation="vertical", fraction=0.055, pad=0.045,
                      shrink=0.95, ticks=[0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=6.8)
    cb.ax.axhline(50.0, color="#b23a48", lw=1.1, ls=(0, (2.4, 1.6)),
                  zorder=5)

    off = ~np.eye(len(METHODS), dtype=bool)
    n_lo, n_hi = int(decisive[off].min()), int(decisive[off].max())
    note = ("Reading the matrix\n"
            "cell $(i, j)$ = share of the decisive\n"
            "pairs (ties dropped) in which raters\n"
            "preferred row $i$ over column $j$\n"
            "The Ours row is win / (win + lose)\n"
            "of the stacked bars above\n"
            "Dashed rule: 50 % of all rated pairs\n\n"
            "Two-sided binomial test\n"
            "*** $p<0.001$   ** $p<0.01$   * $p<0.05$\n"
            f"$n = {n_lo}$ to ${n_hi}$ decisive pairs per cell")
    fig.text(0.735, 0.495, note, ha="left", va="top", fontsize=7.0,
             color="#555555", linespacing=1.38,
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#f5f5f5",
                       edgecolor="#e0e0e0", lw=0.7))

    savefig(fig, "1097_wm_quant_human_eval.png")


if __name__ == "__main__":
    main()
