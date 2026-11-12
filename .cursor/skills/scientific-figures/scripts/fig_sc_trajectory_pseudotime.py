"""Trajectory inference: a branching lineage ordered by pseudotime.

Left, the embedding coloured by pseudotime with the fitted principal curve,
its direction arrows and the bifurcation drawn on top.  Right, gene programmes
smoothed along pseudotime (colour = gene, line style = branch) over a smaller
companion panel showing where the cells actually sit along the ordering, so a
confident-looking curve cannot hide a region with almost no cells.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import (arrow, axes_inches, halo_text, image_axes, lowess,
                    panel_labels_inches, safe_colorbar, shared_limits,
                    unclip_legend)

BRANCH_T = 0.45                      # pseudotime of the bifurcation
BRANCH_NAMES = ["Branch 1 - secretory", "Branch 2 - ciliated"]
BRANCH_LS = ["-", "--"]
GENES = ["SOX9", "MKI67", "MUC5AC", "FOXJ1"]
GENE_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"]

FW, FH = 10.4, 6.0


def backbone(t, branch):
    """Principal curve: one trunk that splits at ``BRANCH_T``."""
    x = -7.5 + 13.5 * t
    y = -2.0 + 4.0 * t + 0.9 * np.sin(3.4 * t)
    d = np.clip(t - BRANCH_T, 0.0, None)
    y = y + np.where(branch == 0, 8.6, -7.2) * d ** 1.20
    return x, y


def synth_cells(rng, n=2600):
    t = np.clip(rng.beta(1.8, 1.9, n) * 1.06, 0.0, 1.0)
    branch = (rng.random(n) > 0.46).astype(int)
    branch[t < BRANCH_T] = 0                       # trunk is shared
    x, y = backbone(t, branch)
    spread = 0.38 + 0.30 * t
    x = x + rng.normal(0, spread) * 0.9
    y = y + rng.normal(0, spread)
    return t, branch, np.column_stack([x, y])


def synth_programmes(rng, t, branch):
    """Four gene programmes: early, transient, and one per branch."""
    late = np.clip(t - BRANCH_T, 0.0, None) / (1.0 - BRANCH_T)
    on_1 = (branch == 0) | (t < BRANCH_T)
    on_2 = (branch == 1) | (t < BRANCH_T)
    prog = np.column_stack([
        2.6 * np.exp(-3.6 * t),                                   # SOX9
        2.4 * np.exp(-0.5 * ((t - BRANCH_T) / 0.16) ** 2),        # MKI67
        2.8 * late ** 1.5 * on_1,                                 # MUC5AC
        2.9 * late ** 1.2 * on_2,                                 # FOXJ1
    ])
    prog = np.clip(prog + rng.normal(0, 0.38, prog.shape), 0.0, None)
    return prog


def window(X, ratio, pad=0.06):
    """Data window whose height/width equals the panel's, so nothing is cropped."""
    cx, cy = X[:, 0].mean(), X[:, 1].mean()
    w = (1.0 + 2 * pad) * max(np.ptp(X[:, 0]), np.ptp(X[:, 1]) / ratio)
    return (cx - w / 2, cx + w / 2), (cy - ratio * w / 2, cy + ratio * w / 2)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(530)
    t, branch, X = synth_cells(rng)
    prog = synth_programmes(rng, t, branch)

    fig = plt.figure(figsize=(FW, FH))
    top = FH - 0.98
    mw, mh = 4.25, 4.05
    ax = axes_inches(fig, (0.62, 0.62 + (top - 0.62 - mh) / 2, mw, mh))
    rx, rw = 6.10, 4.00
    ax_g = axes_inches(fig, (rx, top - 2.45, rw, 2.45))
    ax_d = axes_inches(fig, (rx, 0.62, rw, 1.28))

    # ---- embedding coloured by pseudotime -----------------------------------
    xlim, ylim = window(X, mh / mw)
    sc = ax.scatter(X[:, 0], X[:, 1], c=t, s=3.4, cmap=get_cmap(
        "cb_sequential_purple"), vmin=0.0, vmax=1.0, linewidths=0.0,
        alpha=0.9, rasterized=True)
    image_axes(ax, color="#8a8a8a", lw=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1 (a.u.)", fontsize=8.0)
    ax.set_ylabel("UMAP 2 (a.u.)", fontsize=8.0)

    for b in (0, 1):
        tt = np.linspace(0.0, 1.0, 200)
        bx, by = backbone(tt, np.full_like(tt, b, dtype=int))
        keep = (tt <= BRANCH_T) if b == 0 else (tt >= BRANCH_T)
        seg = keep if b else np.ones_like(tt, dtype=bool)
        ax.plot(bx[seg], by[seg], color="white", lw=3.4, zorder=5,
                solid_capstyle="round")
        ax.plot(bx[seg], by[seg], color="#333333", lw=1.5, zorder=6,
                solid_capstyle="round")
    for b, frac in ((0, 0.24), (0, 0.88), (1, 0.88)):
        t0, t1 = frac, frac + 0.035
        p0 = backbone(np.array([t0]), np.array([b]))
        p1 = backbone(np.array([t1]), np.array([b]))
        arrow(ax, (p0[0][0], p0[1][0]), (p1[0][0], p1[1][0]),
              color="#333333", lw=1.5, ms=9, zorder=7)

    bpx, bpy = backbone(np.array([BRANCH_T]), np.array([0]))
    ax.plot(bpx, bpy, marker="o", ms=6.5, mfc="white", mec="#b23a48",
            mew=1.4, zorder=8, linestyle="none")
    halo_text(ax, bpx[0] - 0.55, bpy[0] + 1.35,
              f"bifurcation\npseudotime {BRANCH_T:.2f}", ha="right",
              va="bottom", fontsize=6.8, color="#b23a48", halo="white",
              halo_lw=2.2, linespacing=1.2)
    for b, name in enumerate(BRANCH_NAMES):
        ex, ey = backbone(np.array([1.0]), np.array([b]))
        halo_text(ax, ex[0], ey[0] + (0.85 if b == 0 else -0.95),
                  name.split(" - ")[0], ha="right",
                  va="bottom" if b == 0 else "top", fontsize=7.2,
                  color="#1a1a1a", halo="white", halo_lw=2.2)
    ax.text(0.018, 0.018, f"{t.size:,} cells  |  2 terminal states  |  "
                          f"principal-curve fit",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            color="#555555")
    safe_colorbar(fig, sc, ax, label="pseudotime (-)", width=0.11, pad=0.10,
                  shrink=0.70, labelsize=7.0)

    # ---- gene programmes along pseudotime -----------------------------------
    curves = []
    for gi in range(len(GENES)):
        for b in (0, 1):
            m = (branch == b) | (t < BRANCH_T)
            xs, ys = lowess(t[m], prog[m, gi], frac=0.34, n_out=160)
            curves.append((gi, b, xs, ys))
    peak = max(c[3].max() for c in curves)
    for gi, b, xs, ys in curves:
        ax_g.plot(xs, ys / peak, color=GENE_COLORS[gi], lw=1.5,
                  ls=BRANCH_LS[b], alpha=0.95)
    ax_g.axvline(BRANCH_T, color="#808080", lw=0.8, ls=":", zorder=0)
    ax_g.set_xlim(0.0, 1.0)
    ax_g.set_ylim(*shared_limits([c[3] / peak for c in curves], pad=0.08,
                                 lo=0.0))
    ax_g.set_ylabel("scaled expression (-)", fontsize=8.0)
    ax_g.set_title("Gene programmes along the ordering", fontsize=8.5,
                   fontweight="bold")
    ax_g.tick_params(labelsize=7.0)

    h_gene = [Line2D([], [], color=c, lw=1.5, label=g)
              for g, c in zip(GENES, GENE_COLORS)]
    h_br = [Line2D([], [], color="#333333", lw=1.5, ls=ls, label=n)
            for ls, n in zip(BRANCH_LS, BRANCH_NAMES)]
    leg1 = ax_g.legend(handles=h_gene, loc="upper left", fontsize=7.0, ncol=2,
                       frameon=False, handlelength=1.5, columnspacing=1.0)
    leg2 = ax_g.legend(handles=h_br, loc="upper right", fontsize=7.0,
                       frameon=False, handlelength=1.8)
    ax_g.add_artist(leg1)          # the second legend() call evicts the first
    unclip_legend(leg1)

    # ---- companion: where the cells actually are ----------------------------
    edges = np.linspace(0.0, 1.0, 41)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dt = edges[1] - edges[0]
    total, _ = np.histogram(t, bins=edges)
    ax_d.fill_between(ctr, total / (t.size * dt), color="#b3b3b3", alpha=0.35,
                      lw=0.0, label="all cells")
    for b in (0, 1):                     # branch shares, they sum to the total
        cnt, _ = np.histogram(t[(branch == b) & (t >= BRANCH_T)], bins=edges)
        keep = ctr >= BRANCH_T
        ax_d.plot(ctr[keep], (cnt / (t.size * dt))[keep], color="#333333",
                  lw=1.2, ls=BRANCH_LS[b])
    ax_d.axvline(BRANCH_T, color="#808080", lw=0.8, ls=":", zorder=0)
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(bottom=0.0)
    ax_d.set_xlabel("pseudotime (-)", fontsize=8.0)
    ax_d.set_ylabel("cell density (-)", fontsize=7.5)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_title("Cell density along pseudotime", fontsize=8.0,
                   fontweight="bold")
    n_late = int(np.sum(t > BRANCH_T))
    ax_d.text(0.985, 0.90, f"{100 * n_late / t.size:.0f}% of cells past the "
                           f"bifurcation", transform=ax_d.transAxes,
              ha="right", va="top", fontsize=6.5, color="#777777")

    panel_labels_inches(fig, [ax, ax_g, ax_d], dx=-0.40, dy=0.08)
    fig.text(0.62 / FW, 1.0 - 0.30 / FH,
             "A single bifurcation splits the lineage into two terminal "
             "programmes",
             ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(0.62 / FW, 1.0 - 0.56 / FH,
             f"{t.size:,} cells ordered by diffusion pseudotime; curves are "
             f"local-linear fits per branch, each gene scaled by its own "
             f"maximum across both branches",
             ha="left", va="center", fontsize=7.5, color="#555555")
    savefig(fig, "530_sc_trajectory_pseudotime.png")


if __name__ == "__main__":
    main()
