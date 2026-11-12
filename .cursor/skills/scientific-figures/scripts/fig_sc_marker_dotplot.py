"""Marker gene dot plot: the signature panel of a single-cell paper.

Cell types on y, marker genes on x, one dot per cell: area encodes the
fraction of cells with detectable counts, colour encodes mean log-normalised
expression.  Genes are grouped three-per-cell-type behind a coloured band, and
the two visual channels get two separate keys on the right so neither has to
be guessed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_label_inches, shared_norm

CELL_TYPES = ["T cell", "B cell", "NK cell", "Monocyte",
              "Macrophage", "Dendritic", "Fibroblast", "Endothelial"]
CELL_SHORT = ["T", "B", "NK", "Mono", "Mac", "DC", "Fibro", "Endo"]
CELL_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227",
               "#8e6bbf", "#2a7f7a", "#c0603f", "#8fa6c2"]

GENE_BLOCKS = [
    ["CD3D", "CD3E", "IL7R"],
    ["MS4A1", "CD79A", "BANK1"],
    ["NKG7", "GNLY", "KLRD1"],
    ["LYZ", "S100A8", "VCAN"],
    ["C1QA", "C1QB", "MRC1"],
    ["FCER1A", "CST3", "CLEC9A"],
    ["COL1A1", "DCN", "LUM"],
    ["PECAM1", "VWF", "CLDN5"],
]

# Lineage affinity: how much a marker block bleeds into a related cell type.
AFFINITY = np.eye(8)
for a, b, v in [(0, 2, 0.34), (3, 4, 0.46), (3, 5, 0.40), (4, 5, 0.28),
                (1, 5, 0.16), (6, 7, 0.20), (0, 1, 0.10)]:
    AFFINITY[a, b] = AFFINITY[b, a] = v

FW, FH = 10.4, 5.35
LEFT, MAIN_W, MAIN_H, BOTTOM = 1.02, 7.00, 3.50, 0.62
GAP, LEG_W = 0.38, 1.50


def size_of(frac):
    """Marker area in pt^2; one formula for the plot and for its legend."""
    return 6.0 + 120.0 * np.asarray(frac, dtype=float)


def synth_dotplot(rng):
    n_t, n_g = len(CELL_TYPES), sum(len(b) for b in GENE_BLOCKS)
    mean = np.zeros((n_t, n_g))
    frac = np.zeros((n_t, n_g))
    j = 0
    for g, block in enumerate(GENE_BLOCKS):
        for _ in block:
            amp = rng.uniform(0.82, 1.16)
            mean[:, j] = 2.75 * amp * AFFINITY[:, g] + rng.uniform(0.02, 0.26, n_t)
            frac[:, j] = np.clip(AFFINITY[:, g] * rng.uniform(0.74, 0.94)
                                 + rng.uniform(0.01, 0.11, n_t), 0.01, 1.0)
            j += 1
    return mean, frac


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(529)
    mean, frac = synth_dotplot(rng)
    genes = [g for block in GENE_BLOCKS for g in block]
    n_t, n_g = mean.shape

    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (LEFT, BOTTOM, MAIN_W, MAIN_H))

    gx, gy = np.meshgrid(np.arange(n_g), np.arange(n_t))
    norm = shared_norm([mean])
    cmap = get_cmap("cb_sequential_blue")

    ax.set_axisbelow(True)
    for i in range(n_t):
        ax.axhline(i, color="#d5d5d5", lw=0.6, zorder=0)
    for b in range(1, len(GENE_BLOCKS)):
        ax.axvline(3 * b - 0.5, color="#cccccc", lw=0.7, zorder=0)

    sc = ax.scatter(gx.ravel(), gy.ravel(), s=size_of(frac).ravel(),
                    c=mean.ravel(), cmap=cmap, norm=norm,
                    edgecolors="#3a3a3a", linewidths=0.25, zorder=3)

    ax.set_xlim(-0.75, n_g - 0.25)
    ax.set_ylim(n_t - 0.5, -0.5)                 # first cell type on top
    ax.set_xticks(np.arange(n_g))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(n_t))
    ax.set_yticklabels(CELL_TYPES, fontsize=7.5)
    for tick, col in zip(ax.get_yticklabels(), CELL_COLORS):
        tick.set_color(col)
    ax.tick_params(length=2.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("marker genes, grouped by the cell type they define",
                  fontsize=8.0, labelpad=3.0)

    # ---- gene-group band above the grid ------------------------------------
    for g, short in enumerate(CELL_SHORT):
        ax.add_patch(Rectangle((3 * g - 0.42, -1.02), 2.84, 0.30,
                               facecolor=CELL_COLORS[g], edgecolor="none",
                               clip_on=False, zorder=4))
        ax.text(3 * g + 1.0, -1.16, short, ha="center", va="bottom",
                fontsize=6.8, color=CELL_COLORS[g], fontweight="bold",
                clip_on=False)

    # ---- key 1: dot area = fraction of cells expressing ---------------------
    lx = LEFT + MAIN_W + GAP
    axk = axes_inches(fig, (lx, BOTTOM + 1.86, LEG_W, MAIN_H - 1.86))
    axk.set_axis_off()
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    axk.text(0.0, 0.98, "Fraction of cells\nexpressing (%)", ha="left",
             va="top", fontsize=7.2, color="#1a1a1a", fontweight="bold",
             linespacing=1.25)
    keys = [0.2, 0.4, 0.6, 0.8, 1.0]
    ys = np.linspace(0.60, 0.06, len(keys))
    axk.scatter(np.full(len(keys), 0.10), ys, s=size_of(keys),
                facecolor="#d5d5d5", edgecolors="#3a3a3a", linewidths=0.25)
    for k, y in zip(keys, ys):
        axk.text(0.26, y, f"{100 * k:.0f}", ha="left", va="center",
                 fontsize=6.8, color="#333333")

    # ---- key 2: colour = mean expression ------------------------------------
    cax = axes_inches(fig, (lx + 0.06, BOTTOM + 0.28, 0.135, 1.28))
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.5, color="#333333")
    cb.set_label("mean expression (log-norm)", fontsize=6.8, color="#1a1a1a")
    cb.ax.set_title("Expression", fontsize=7.2, fontweight="bold", pad=4.0,
                    loc="left")

    panel_label_inches(fig, ax, "(a)", dx=-0.92, dy=0.30)
    on_target = np.mean([mean[g, 3 * g:3 * g + 3].mean()
                         for g in range(n_t)])
    off = mean[~np.eye(n_t, dtype=bool).repeat(3, axis=1)].mean()
    fig.text(LEFT / FW, 1.0 - 0.28 / FH,
             "Each marker block is expressed almost exclusively in the cell "
             "type it annotates",
             ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(LEFT / FW, 1.0 - 0.53 / FH,
             f"{n_g} markers x {n_t} annotated cell types; on-block mean "
             f"{on_target:.2f} vs off-block mean {off:.2f} log-normalised "
             f"units, residual signal follows the myeloid lineage",
             ha="left", va="center", fontsize=7.5, color="#555555")
    savefig(fig, "529_sc_marker_dotplot.png")


if __name__ == "__main__":
    main()
