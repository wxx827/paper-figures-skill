"""Single-cell UMAP atlas: annotated clusters plus marker-gene overlays.

The house layout for the opening figure of a single-cell paper: one large
UMAP embedding with eight annotated cell types labelled on the cloud itself,
and a 2x3 block of small marker-gene overlays that all share one expression
colour scale so the panels can be compared at a glance.

Aspect note: every panel is ``set_aspect("equal")`` and all seven share one
square data window, so the canvas is wide by construction (DESIGN_SYSTEM
section 1.2, exception E1); panels are placed in inches to keep the gutters
physically equal and leave no dead space between the equal-aspect boxes.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import (axes_inches, halo_text, image_axes, panel_label_inches,
                    safe_colorbar, shared_norm)

# Cell-type order and colours are shared by every figure in the single-cell
# set (528-533), so a cell type keeps its colour across the whole story.
CELL_TYPES = ["T cell", "B cell", "NK cell", "Monocyte",
              "Macrophage", "Dendritic", "Fibroblast", "Endothelial"]
CELL_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227",
               "#8e6bbf", "#2a7f7a", "#c0603f", "#8fa6c2"]

# gene -> cell types in which it is enriched
MARKERS = [("CD3D", [0]), ("MS4A1", [1]), ("NKG7", [2, 0]),
           ("LYZ", [3, 5]), ("C1QA", [4]), ("COL1A1", [6])]

CENTERS = np.array([[-6.6, 2.2], [-4.6, -4.8], [-2.2, 6.1], [2.4, 3.4],
                    [5.9, 0.2], [1.1, -3.0], [-8.0, -1.6], [5.2, -5.2]])
SIZES = [860, 430, 285, 690, 545, 240, 330, 395]

FW, FH = 11.8, 5.6            # canvas
MAIN, TILE = 3.95, 1.80       # panel side lengths, inches
GAP_C, GAP_R, GAP_M = 0.24, 0.46, 0.62


def synth_embedding(rng):
    """Anisotropic, slightly heavy-tailed blobs - UMAP clusters are not round."""
    pts, lab = [], []
    for i, (c, n) in enumerate(zip(CENTERS, SIZES)):
        ang = rng.uniform(0.0, np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang), np.cos(ang)]])
        z = rng.normal(size=(n, 2)) * (rng.uniform(0.75, 1.55),
                                       rng.uniform(0.50, 0.95))
        z += 0.16 * rng.standard_t(3, size=(n, 2))
        pts.append(z @ rot.T + c)
        lab.append(np.full(n, i))
    return np.vstack(pts), np.concatenate(lab)


def synth_expression(rng, lab):
    """Log-normalised counts: low background, enriched in the marker's types."""
    out = []
    for _, hi in MARKERS:
        e = rng.gamma(1.15, 0.26, lab.size)
        m = np.isin(lab, hi)
        e[m] += rng.gamma(4.2, 0.52, int(m.sum()))
        e *= rng.random(lab.size) > 0.17          # dropout
        out.append(e)
    return out


def square_window(X, pad=0.06):
    """One square data window shared by every panel - same view everywhere."""
    cx, cy = X[:, 0].mean(), X[:, 1].mean()
    half = 0.5 * (1.0 + 2 * pad) * max(np.ptp(X[:, 0]), np.ptp(X[:, 1]))
    return (cx - half, cx + half), (cy - half, cy + half)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(528)
    X, lab = synth_embedding(rng)
    expr = synth_expression(rng, lab)
    xlim, ylim = square_window(X)

    fig = plt.figure(figsize=(FW, FH))
    block_h = 2 * TILE + GAP_R
    top = FH - 1.05                       # below the title band
    bx0 = 0.52 + MAIN + GAP_M             # left edge of the 2x3 block
    my0 = top - block_h + (block_h - MAIN) / 2.0

    # ---- main embedding ----------------------------------------------------
    ax = axes_inches(fig, (0.52, my0, MAIN, MAIN))
    for i, col in enumerate(CELL_COLORS):
        m = lab == i
        ax.scatter(X[m, 0], X[m, 1], s=2.6, c=col, linewidths=0.0,
                   alpha=0.85, rasterized=True)
    image_axes(ax, color="#8a8a8a", lw=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for i, name in enumerate(CELL_TYPES):
        cx, cy = np.median(X[lab == i], axis=0)
        halo_text(ax, cx, cy, name, ha="center", va="center", fontsize=7.6,
                  halo_lw=2.0)
    ax.text(0.018, 0.018, f"{X.shape[0]:,} cells  |  "
                          f"{len(CELL_TYPES)} clusters  |  UMAP on 30 PCs",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            color="#555555")
    ax.set_xlabel("UMAP 1 (a.u.)", fontsize=8.0)
    ax.set_ylabel("UMAP 2 (a.u.)", fontsize=8.0)

    # ---- marker-gene overlays: one shared expression scale ------------------
    norm = shared_norm(expr, robust=(0, 99.5))
    cmap = get_cmap("cb_sequential_blue")
    smalls = []
    for k, ((gene, _), e) in enumerate(zip(MARKERS, expr)):
        x0 = bx0 + (k % 3) * (TILE + GAP_C)
        y0 = top - TILE - (k // 3) * (TILE + GAP_R)
        axs = axes_inches(fig, (x0, y0, TILE, TILE))
        order = np.argsort(e)                     # high expression on top
        sc = axs.scatter(X[order, 0], X[order, 1], s=1.4, c=e[order],
                         cmap=cmap, norm=norm, linewidths=0.0,
                         rasterized=True)
        image_axes(axs, color="#cccccc", lw=0.5)
        axs.set_xlim(*xlim)
        axs.set_ylim(*ylim)
        axs.set_title(gene, fontsize=8.5, fontweight="bold", pad=2.5)
        axs.text(0.025, 0.025, f"{100.0 * np.mean(e > 0.5):.0f}% cells > 0.5",
                 transform=axs.transAxes, ha="left", va="bottom",
                 fontsize=6.2, color="#777777")
        smalls.append(axs)

    # shared colourbar pinned to the right edge of the whole 2x3 block
    anchor = axes_inches(fig, (bx0 + 2 * (TILE + GAP_C), top - block_h,
                               TILE, block_h), frameon=False)
    anchor.set_axis_off()
    safe_colorbar(fig, sc, anchor, label="log-normalised expression (-)",
                  width=0.11, pad=0.11, shrink=0.62, labelsize=6.8)

    panel_label_inches(fig, ax, "(a)", dx=-0.34, dy=0.06)
    panel_label_inches(fig, smalls[0], "(b)", dx=-0.22, dy=0.04)
    fig.text((bx0 + (3 * TILE + 2 * GAP_C) / 2) / FW, (top + 0.34) / FH,
             "Marker gene expression", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", color="#1a1a1a")

    fig.text(0.52 / FW, 1.0 - 0.30 / FH,
             "Eight transcriptional states resolve as separate neighbourhoods "
             "in the UMAP embedding",
             ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(0.52 / FW, 1.0 - 0.56 / FH,
             f"{X.shape[0]:,} quality-filtered cells, Leiden resolution 0.8; "
             f"{len(MARKERS)} canonical markers on one shared scale",
             ha="left", va="center", fontsize=7.5, color="#555555")
    savefig(fig, "528_sc_umap_clusters.png")


if __name__ == "__main__":
    main()
