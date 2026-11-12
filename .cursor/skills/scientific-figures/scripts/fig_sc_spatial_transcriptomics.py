"""Spatial transcriptomics: hexagonal spots on a tissue section.

The main panel keeps the geometry of the slide - a hexagonal capture grid
clipped to the tissue outline, coloured by spatial domain.  The 2x2 companion
block maps four spatially variable genes on one shared expression scale, each
stamped with its Moran's I computed from the very spots that are drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from scipy.spatial import cKDTree

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import (axes_inches, halo_text, image_axes, panel_label_inches,
                    safe_colorbar, shared_norm)

DOMAINS = ["Tumour core", "Invasive front", "Stroma",
           "Immune infiltrate", "Normal epithelium"]
DOMAIN_COLORS = ["#b23a48", "#c9a227", "#c0603f", "#3b5b92", "#2e7d5b"]
SEEDS = np.array([[-3.4, 3.0], [2.6, 5.2], [5.6, -2.4],
                  [-5.2, -4.6], [1.0, -6.4]])
# gene, domain it marks, technical noise (drives how coherent the map looks)
SVGS = [("SPP1", 0, 0.22), ("MMP9", 1, 0.70),
        ("COL1A1", 2, 0.38), ("CXCL9", 3, 1.05)]

FW, FH = 10.4, 5.9
RADIUS, PITCH = 13.0, 0.86


def tissue_radius(theta):
    """Irregular but smooth section outline."""
    return RADIUS * (1.0 + 0.15 * np.sin(3 * theta + 0.7)
                     + 0.08 * np.cos(5 * theta - 1.1)
                     + 0.05 * np.sin(2 * theta + 2.4))


def hex_spots():
    dy = PITCH * np.sqrt(3) / 2.0
    rows = int(2.2 * RADIUS / dy)
    cols = int(2.2 * RADIUS / PITCH)
    pts = []
    for r in range(-rows // 2, rows // 2 + 1):
        y = r * dy
        for c in range(-cols // 2, cols // 2 + 1):
            pts.append((c * PITCH + (r % 2) * PITCH / 2.0, y))
    p = np.array(pts)
    keep = np.hypot(p[:, 0], p[:, 1]) < 0.985 * tissue_radius(
        np.arctan2(p[:, 1], p[:, 0]))
    return p[keep]


def assign_domains(rng, p):
    """Nearest seed plus a smooth field, so domains are contiguous but ragged."""
    score = np.empty((p.shape[0], len(SEEDS)))
    for k, s in enumerate(SEEDS):
        ph = rng.uniform(0, 2 * np.pi, 3)
        wob = (1.6 * np.sin(0.32 * p[:, 0] + ph[0])
               + 1.4 * np.cos(0.27 * p[:, 1] + ph[1])
               + 1.1 * np.sin(0.19 * (p[:, 0] + p[:, 1]) + ph[2]))
        score[:, k] = -np.hypot(*(p - s).T) + wob
    return np.argmax(score, axis=1)


def synth_svg(rng, p, dom, target, noise):
    d = np.hypot(*(p - SEEDS[target]).T)
    e = (2.5 * (dom == target) + 1.5 * np.exp(-d / 6.5)
         + 0.35 * np.exp(-((p[:, 1] - 4.0) / 9.0) ** 2))
    return np.clip(e + rng.normal(0, noise, p.shape[0]), 0.0, None)


def morans_i(coords, x, k=6):
    """Global Moran's I on a k-nearest-neighbour weight graph."""
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    idx = idx[:, 1:]
    z = x - x.mean()
    return (z.size / idx.size) * (z[:, None] * z[idx]).sum() / (z ** 2).sum()


def hex_area(inches, span, fill=0.92):
    """Marker area in pt^2 so hexes tile without overlapping."""
    return (fill * PITCH * 72.0 * inches / span) ** 2


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(532)
    p = hex_spots()
    dom = assign_domains(rng, p)
    expr = [synth_svg(rng, p, dom, t, nz) for _, t, nz in SVGS]
    moran = [morans_i(p, e) for e in expr]

    th = np.linspace(0, 2 * np.pi, 361)
    outline = np.column_stack([tissue_radius(th) * np.cos(th),
                               tissue_radius(th) * np.sin(th)])
    lim = 1.06 * np.abs(outline).max()

    fig = plt.figure(figsize=(FW, FH))
    main_s, tile, gap_c, gap_r = 4.10, 1.85, 0.26, 0.46
    top = FH - 1.00
    block_h = 2 * tile + gap_r
    bx0 = 0.55 + main_s + 0.65

    # ---- main: spatial domains ---------------------------------------------
    ax = axes_inches(fig, (0.55, top - block_h + (block_h - main_s) / 2,
                           main_s, main_s))
    ax.add_patch(Polygon(outline, closed=True, facecolor="#f0f0f0",
                         edgecolor="#cccccc", lw=0.7, zorder=0))
    s_main = hex_area(main_s, 2 * lim)
    for k, col in enumerate(DOMAIN_COLORS):
        m = dom == k
        ax.scatter(p[m, 0], p[m, 1], s=s_main, marker="h", c=col,
                   linewidths=0.0, zorder=2)
    image_axes(ax, color="#8a8a8a", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("slide x (spot units)", fontsize=8.0)
    ax.set_ylabel("slide y (spot units)", fontsize=8.0)
    ax.text(0.018, 0.018, f"{p.shape[0]:,} spots  |  {len(DOMAINS)} spatial "
                          f"domains  |  55 um capture area",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            color="#555555")
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=d)
                       for d, c in zip(DOMAINS, DOMAIN_COLORS)],
              loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3,
              frameon=False, fontsize=7.0, handlelength=1.1,
              handleheight=0.9, columnspacing=1.1)

    # ---- 2x2 spatially variable genes, one shared scale ---------------------
    norm = shared_norm(expr, robust=(0, 99.5))
    cmap = get_cmap("cb_sequential_green")
    s_tile = hex_area(tile, 2 * lim)
    tiles = []
    for k, ((gene, _, _), e) in enumerate(zip(SVGS, expr)):
        axt = axes_inches(fig, (bx0 + (k % 2) * (tile + gap_c),
                                top - tile - (k // 2) * (tile + gap_r),
                                tile, tile))
        axt.add_patch(Polygon(outline, closed=True, facecolor="#f5f5f5",
                              edgecolor="#d5d5d5", lw=0.5, zorder=0))
        sc = axt.scatter(p[:, 0], p[:, 1], s=s_tile, marker="h", c=e,
                         cmap=cmap, norm=norm, linewidths=0.0, zorder=2)
        image_axes(axt, color="#cccccc", lw=0.5)
        axt.set_xlim(-lim, lim)
        axt.set_ylim(-lim, lim)
        axt.set_title(gene, fontsize=8.5, fontweight="bold", pad=2.5)
        halo_text(axt, 0.035, 0.965, f"Moran's I = {moran[k]:.2f}",
                  transform=axt.transAxes, ha="left", va="top", fontsize=6.4,
                  color="#1a1a1a", halo="white", halo_lw=2.0)
        tiles.append(axt)

    anchor = axes_inches(fig, (bx0 + tile + gap_c, top - block_h, tile,
                               block_h))
    anchor.set_axis_off()
    safe_colorbar(fig, sc, anchor, label="log-normalised expression (-)",
                  width=0.11, pad=0.11, shrink=0.62, labelsize=6.8)

    panel_label_inches(fig, ax, "(a)", dx=-0.36, dy=0.06)
    panel_label_inches(fig, tiles[0], "(b)", dx=-0.20, dy=0.04)
    fig.text((bx0 + (2 * tile + gap_c) / 2) / FW, (top + 0.34) / FH,
             "Spatially variable genes", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color="#1a1a1a")
    fig.text(0.55 / FW, 1.0 - 0.30 / FH,
             "Five spatial domains partition the section along the "
             "tumour-stroma axis",
             ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(0.55 / FW, 1.0 - 0.56 / FH,
             f"{p.shape[0]:,} hexagonal spots; Moran's I "
             f"{min(moran):.2f} to {max(moran):.2f} (6-NN graph)",
             ha="left", va="center", fontsize=7.5, color="#555555")
    savefig(fig, "532_sc_spatial_transcriptomics.png")


if __name__ == "__main__":
    main()
