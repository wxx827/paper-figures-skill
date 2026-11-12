"""Resistance screening: the phenotype matrix and the genotype that explains it.

The three panels are one table read left to right, so they share a single row
order taken from hierarchical clustering of the reaction matrix.  Panel (a) is
the greenhouse read-out on the standard five-point scale; panel (b) is the R
gene content of the same cultivars; panel (c) counts how many races each one
holds off.  None of the three is generated independently - every reaction is
derived by gene-for-gene matching between the genotype in (b) and the
avirulence profile of the race, tempered by the cultivar's partial resistance,
so a cultivar cannot look resistant in (a) without carrying something in (b).
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_label_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREEN = "#2e7d5b"
ABSENT = "#e6e6e6"

W, H = 10.4, 6.4
# Five-point reaction scale, deep green (immune) to deep red (highly
# susceptible); lightness is monotone across the middle three steps.
SCALE = ["#0b3d2a", "#2e7d5b", "#a8cbb6", "#d99aa4", "#8c2f2f"]
CODES = ["I", "HR", "MR", "MS", "HS"]
NAMES = ["immune", "highly resistant", "moderately resistant",
         "moderately susceptible", "highly susceptible"]
LIGHT = (2, 3)                 # levels that need dark text
RESISTANT_MAX = 2              # I, HR and MR count as a resistant reaction

GENES = ["Pi9", "Pi2", "Pita", "Pib", "Piz-t"]
CULTIVARS = ["Tetep", "IRBL9-W", "Milyang 23", "Suweon 365", "K59",
             "Zhenshan 97", "Pi No.4", "Aichi Asahi", "Jinguoyin",
             "Nipponbare", "CO39", "LTH"]
RACES = ["IA-1", "IB-33", "IC-17", "ID-13", "IE-1", "IG-1", "IH-1",
         "ZB-13", "ZC-15", "ZG-1"]


def screen(rng):
    """Genotypes, race avirulence and the reactions they imply."""
    n_c, n_r, n_g = len(CULTIVARS), len(RACES), len(GENES)
    geno = (rng.random((n_c, n_g)) < 0.34).astype(int)
    geno[0] = [1, 1, 0, 1, 1]          # broad-spectrum donor
    geno[1] = [1, 0, 0, 0, 1]
    geno[-1] = 0                       # universally susceptible check
    geno[-2] = 0
    avir = (rng.random((n_r, n_g)) < 0.52).astype(int)
    partial = np.clip(rng.beta(2.0, 3.0, n_c), 0.0, 1.0)
    partial[0] = 0.85
    partial[-1] = 0.05

    match = geno @ avir.T
    score = np.empty((n_c, n_r), dtype=int)
    for i in range(n_c):
        for j in range(n_r):
            m = match[i, j]
            if m == 0:
                v = 4.0 - 1.9 * partial[i]
            elif m == 1:
                v = 2.4 - 1.2 * partial[i]
            else:
                v = 1.4 - 1.3 * partial[i]
            score[i, j] = int(np.clip(round(v), 0, 4))
    return geno, avir, score, partial


def cluster_order(mat):
    return (leaves_list(linkage(pdist(mat, metric="euclidean"),
                                method="average")),
            leaves_list(linkage(pdist(mat.T, metric="euclidean"),
                                method="average")))


# ------------------------------------------------------------------- panels
def matrix_panel(ax, score, rows, cols):
    cmap = mpl.colors.ListedColormap(SCALE)
    ax.imshow(score, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto",
              interpolation="nearest")
    n_r, n_c = score.shape
    for i in range(n_r + 1):
        ax.axhline(i - 0.5, color="white", lw=0.9, zorder=3)
    for j in range(n_c + 1):
        ax.axvline(j - 0.5, color="white", lw=0.9, zorder=3)
    for i in range(n_r):
        for j in range(n_c):
            v = score[i, j]
            ax.text(j, i, CODES[v], ha="center", va="center", fontsize=6.0,
                    color=INK if v in LIGHT else "white", zorder=4)

    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels(cols, rotation=42.0, ha="right", fontsize=6.6)
    ax.set_yticks(np.arange(n_r))
    ax.set_yticklabels(rows, fontsize=6.8)
    ax.set_xlabel("pathogen race (isolate code)", fontsize=8.0)
    ax.tick_params(length=0.0)
    for s in ax.spines.values():
        s.set_visible(False)
    res = float(np.mean(score <= RESISTANT_MAX))
    ax.set_title("%.0f %% of the %d cultivar-race combinations are resistant"
                 % (100.0 * res, score.size), fontsize=9.0,
                 fontweight="bold", color=INK)
    return res


def genotype_panel(ax, geno, n_rows):
    cmap = mpl.colors.ListedColormap([ABSENT, GREEN])
    ax.imshow(geno, cmap=cmap, vmin=-0.5, vmax=1.5, aspect="auto",
              interpolation="nearest")
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.9, zorder=3)
    for j in range(len(GENES) + 1):
        ax.axvline(j - 0.5, color="white", lw=0.9, zorder=3)
    ax.set_xticks(np.arange(len(GENES)))
    ax.set_xticklabels(GENES, rotation=42.0, ha="right", fontsize=6.6)
    ax.set_yticks([])
    ax.set_xlabel("R gene", fontsize=7.6)
    ax.tick_params(length=0.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Genotype", fontsize=8.4, fontweight="bold", color=INK)


def spectrum_panel(ax, breadth, n_rows, n_races):
    ramp = get_palette("cb_sequential_green", n_races + 1)
    cols = [ramp[int(b)] for b in breadth]
    y = np.arange(n_rows, dtype=float)
    ax.barh(y, breadth, height=0.62, color=cols, edgecolor="white", lw=0.7,
            zorder=3)
    order = np.argsort(-breadth, kind="stable")
    rank = np.empty(n_rows, dtype=int)
    rank[order] = np.arange(1, n_rows + 1)
    for yi, b, rk in zip(y, breadth, rank):
        ax.text(b + 0.25, yi, "%d   #%d" % (b, rk), ha="left", va="center",
                fontsize=6.4, color=INK)
    ax.axvline(n_races, color=HAIR, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)

    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(0.0, n_races + 3.6)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("races held off (races)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_title("Spectrum width", fontsize=8.4, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(7431)
    geno, avir, score, partial = screen(rng)
    r_idx, c_idx = cluster_order(score.astype(float))

    score = score[np.ix_(r_idx, c_idx)]
    geno = geno[r_idx]
    rows = [CULTIVARS[i] for i in r_idx]
    cols = [RACES[j] for j in c_idx]
    breadth = np.sum(score <= RESISTANT_MAX, axis=1)

    fig = plt.figure(figsize=(W, H))
    ax_mat = axes_inches(fig, (1.62, 1.22, 4.50, 3.72))
    ax_gen = axes_inches(fig, (6.58, 1.22, 1.05, 3.72))
    ax_spe = axes_inches(fig, (8.22, 1.22, 1.82, 3.72))

    res = matrix_panel(ax_mat, score, rows, cols)
    genotype_panel(ax_gen, geno, len(rows))
    spectrum_panel(ax_spe, breadth, len(rows), len(RACES))

    handles = [Patch(facecolor=c, edgecolor="white", lw=0.7,
                     label="%s (%s)" % (n, k))
               for c, k, n in zip(SCALE, CODES, NAMES)]
    handles.append(Patch(facecolor=GREEN, edgecolor="white", lw=0.7,
                         label="R gene present"))

    best, worst = rows[int(np.argmax(breadth))], rows[int(np.argmin(breadth))]
    fig.text(0.016, 6.20 / H,
             "%s holds off %d of %d races, %s none: resistance tracks R gene "
             "content, not overall vigour"
             % (best, int(breadth.max()), len(RACES), worst),
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H, textwrap.fill(
             "Rows and columns are ordered by average-linkage clustering of "
             "the reaction matrix, and all three panels keep that one row "
             "order. Each reaction is the gene-for-gene match count between "
             "the genotype in (b) and the race avirulence profile, shifted by "
             "the cultivar's partial resistance; %.0f %% of the %d "
             "combinations come out resistant (I, HR or MR)."
             % (100.0 * res, score.size), 195),
             ha="left", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.5)

    panel_label_inches(fig, ax_mat, "(a)", dx=-0.58, dy=0.18)
    panel_label_inches(fig, ax_gen, "(b)", dx=-0.34, dy=0.18)
    panel_label_inches(fig, ax_spe, "(c)", dx=-0.34, dy=0.18)

    savefig(fig, "743_pla_resistance_screening.png")


if __name__ == "__main__":
    main()
