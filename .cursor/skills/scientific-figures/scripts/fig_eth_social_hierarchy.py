"""Dominance structure of a ten-animal group from one contest record.

Every panel is computed from the same simulated sequence of dyadic contests,
so the four views cannot disagree.  The main panel is the claim: once the
animals are sorted by final Elo, the wins collapse into the upper triangle and
Landau's linearity index is close to one, which is what "linear hierarchy"
means operationally.  The inset redraws the same matrix as a directed network
so the direction of dominance is visible without reading indices.  The
subordinate panels show that the order is not static - one rank takeover
happens mid-season - and that observation effort is far from uniform across
dyads, which is exactly why the index needs its unknown-dyad correction.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREY = "#b3b3b3"

N_IND = 10
N_BOUT = 760
K_ELO = 22.0
SWAP_AT = 400            # contest index of the rank takeover
SWAP_PAIR = (1, 2)       # the two animals that exchange true strength


def contests(seed=17):
    """Simulate dyadic contests and score them with an Elo update rule."""
    rng = np.random.default_rng(seed)
    theta = 1000.0 + 340.0 * np.arange(N_IND - 1, -1, -1) \
        + rng.normal(0.0, 22.0, N_IND)
    # Encounter propensity: neighbours in rank meet far more often.
    r = np.arange(N_IND)
    prop = np.exp(-np.abs(r[:, None] - r[None, :]) / 1.9)
    np.fill_diagonal(prop, 0.0)
    iu = np.triu_indices(N_IND, 1)
    pw = prop[iu] / prop[iu].sum()

    elo = np.full(N_IND, 1000.0)
    hist = np.empty((N_BOUT + 1, N_IND))
    hist[0] = elo
    wins = np.zeros((N_IND, N_IND))
    for b in range(N_BOUT):
        if b == SWAP_AT:
            a, c = SWAP_PAIR
            theta[a], theta[c] = theta[c], theta[a]
        k = rng.choice(pw.size, p=pw)
        i, j = int(iu[0][k]), int(iu[1][k])
        p = 1.0 / (1.0 + 10.0 ** ((theta[j] - theta[i]) / 400.0))
        i_wins = rng.random() < p
        w, l = (i, j) if i_wins else (j, i)
        wins[w, l] += 1.0
        exp_w = 1.0 / (1.0 + 10.0 ** ((elo[l] - elo[w]) / 400.0))
        elo[w] += K_ELO * (1.0 - exp_w)
        elo[l] -= K_ELO * (1.0 - exp_w)
        hist[b + 1] = elo
    return wins, hist


def davids_score(wins):
    """David's score - the standard ranking criterion for a win/loss matrix."""
    tot = wins + wins.T
    p = np.divide(wins, tot, out=np.zeros_like(wins), where=tot > 0)
    w1, l1 = p.sum(axis=1), p.sum(axis=0)
    return w1 + p @ w1 - l1 - p.T @ l1


def rank_order(wins):
    return np.argsort(-davids_score(wins))


def landau(wins):
    """Landau's h with de Vries' correction for unknown dyads."""
    n = wins.shape[0]
    v = np.zeros(n)
    unknown = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if wins[i, j] == 0 and wins[j, i] == 0:
                v[i] += 0.5
                if i < j:
                    unknown += 1
            elif wins[i, j] > wins[j, i]:
                v[i] += 1.0
            elif wins[i, j] == wins[j, i]:
                v[i] += 0.5
    h = 12.0 / (n ** 3 - n) * float(np.sum((v - (n - 1) / 2.0) ** 2))
    return h, h + 6.0 * unknown / (n ** 3 - n), unknown


def network_inset(ax, wins, names, top=16):
    axi = ax.inset_axes([0.025, 0.030, 0.345, 0.395])
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.patch.set_facecolor("white")
    axi.patch.set_alpha(0.94)
    axi.set_xlim(0.0, 1.0)
    axi.set_ylim(0.0, 1.0)
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#cccccc")

    n = wins.shape[0]
    ys = np.linspace(0.93, 0.07, n)
    xs = 0.42 + 0.13 * (np.arange(n) % 2)

    pairs = [(i, j, wins[i, j]) for i in range(n) for j in range(n) if i != j]
    pairs.sort(key=lambda p: -p[2])
    wmax = max(p[2] for p in pairs) or 1.0
    for i, j, w in pairs[:top]:
        if w <= 0:
            continue
        rad = 0.42 if i < j else -0.42
        axi.add_patch(FancyArrowPatch(
            (xs[i], ys[i]), (xs[j], ys[j]), arrowstyle="-|>",
            mutation_scale=6.0, lw=0.4 + 1.5 * w / wmax, color="#8a8a8a",
            alpha=0.85, connectionstyle=f"arc3,rad={rad}",
            shrinkA=4.2, shrinkB=4.6, zorder=3))
    for i in range(n):
        axi.plot([xs[i]], [ys[i]], marker="o", ms=5.4, mfc="white", mec=RULE,
                 mew=0.9, zorder=5)
        axi.text(xs[i] - 0.085, ys[i], names[i], ha="right", va="center",
                 fontsize=6.0, color=INK, zorder=6)
    axi.text(0.5, 1.0, "dominance network", ha="center", va="bottom",
             fontsize=6.6, color=SUB, transform=axi.transAxes)
    axi.text(0.97, 0.985, "arrow to\nsubordinate,\nwidth = wins",
             ha="right", va="top", fontsize=6.0, color=MUTED)
    return axi


def matrix_panel(fig, ax, wins, names):
    n = wins.shape[0]
    im = ax.imshow(wins, cmap=get_cmap("cb_sequential_blue"),
                   vmin=0.0, vmax=float(wins.max()))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if wins[i, j] <= 0:
                continue
            ax.text(j, i, f"{wins[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.0,
                    color="white" if wins[i, j] > 0.55 * wins.max() else INK)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1.0, 1.0,
                                   facecolor="#f0f0f0", edgecolor="white",
                                   linewidth=0.6, zorder=2))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlabel("loser, ordered by David's score (-)")
    ax.set_ylabel("winner, ordered by David's score (-)")
    ax.set_aspect("equal")
    ax.tick_params(length=2.0)
    ax.set_title("Wins collapse into the upper triangle", fontsize=9.0,
                 fontweight="bold", color=INK)

    h, hp, unknown = landau(wins)
    same = float(np.triu(wins, 1).sum())
    dc = (same - float(np.tril(wins, -1).sum())) / float(wins.sum())
    ax.text(0.415, 0.045, f"Landau h = {h:.2f},  h' = {hp:.2f}\n"
                          f"directional consistency = {dc:.2f}\n"
                          f"unknown dyads = {unknown} of {n * (n - 1) // 2}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6,
            color=SUB, zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    safe_colorbar(fig, im, ax, label="wins (count)", width=0.10, pad=0.07,
                  labelsize=6.6)
    network_inset(ax, wins, names)
    return h, hp, dc, unknown


def elo_panel(ax, hist, order, names):
    x = np.arange(hist.shape[0])
    focus = [int(np.where(order == SWAP_PAIR[0])[0][0]),
             int(np.where(order == SWAP_PAIR[1])[0][0])]
    for k, idx in enumerate(order):
        if k in focus:
            continue
        ax.plot(x, hist[:, idx], color=GREY, lw=0.8, zorder=2)
    for k, col in zip(focus, [BLUE, RED]):
        ax.plot(x, hist[:, order[k]], color=col, lw=1.7, zorder=4,
                label=names[k])
    ax.axvline(SWAP_AT, color=MUTED, lw=0.9, ls=(0, (2.6, 2.0)), zorder=3)
    ax.text(SWAP_AT + 8, 0.05, "rank takeover", fontsize=6.3, color=MUTED,
            ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("contest index (-)", fontsize=7.4)
    ax.set_ylabel("Elo rating (-)", fontsize=7.4)
    ax.set_xlim(0.0, hist.shape[0] - 1)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left", ncol=2,
              handlelength=1.5, borderaxespad=0.25,
              columnspacing=1.0)
    ax.set_title(f"One reversal at contest {SWAP_AT}", fontsize=8.2,
                 fontweight="bold", color=INK)


def effort_panel(ax, wins):
    n = wins.shape[0]
    iu = np.triu_indices(n, 1)
    dyad = (wins + wins.T)[iu]
    bins = np.arange(-0.5, dyad.max() + 2.5, 3.0)
    ax.hist(dyad, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6,
            zorder=3)
    med = float(np.median(dyad))
    ax.axvline(med, color=RED, lw=1.2, ls=(0, (3.2, 2.0)), zorder=4)
    ax.text(med + 1.2, 0.95, f"median {med:.0f}", fontsize=6.3, color=RED,
            ha="left", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("interactions per dyad (count)", fontsize=7.4)
    ax.set_ylabel(f"dyads of n={dyad.size} (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Observation effort is uneven", fontsize=8.2,
                 fontweight="bold", color=INK)
    return dyad


def main():
    set_pub_style()
    wins_raw, hist = contests()
    order = rank_order(wins_raw)
    wins = wins_raw[np.ix_(order, order)]
    names = [f"R{k + 1:02d}" for k in range(N_IND)]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_mat = axes_inches(fig, (0.86, 0.80, 5.00, 4.62))
    ax_elo = axes_inches(fig, (7.02, 3.40, 2.84, 2.02))
    ax_eff = axes_inches(fig, (7.02, 0.80, 2.84, 2.02))

    h, hp, dc, unknown = matrix_panel(fig, ax_mat, wins, names)
    elo_panel(ax_elo, hist, order, names)
    dyad = effort_panel(ax_eff, wins)

    panel_labels_inches(fig, [ax_mat, ax_elo, ax_eff], dx=-0.52, dy=0.14)
    savefig(fig, "777_eth_social_hierarchy.png")


if __name__ == "__main__":
    main()
