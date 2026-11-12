"""Annotated expression heatmap: genes x samples with multi-track metadata.

Nature-style omics main figure. Row-clustered z-score heatmap flanked by a
left dendrogram and right co-expression module blocks, topped by three
categorical annotation tracks (group / sex / stage) that each get their own
compact legend, plus a shared diverging colourbar.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from pubstyle import set_pub_style, PALETTES, savefig

EXPR_CMAP = LinearSegmentedColormap.from_list("expr", PALETTES["div_rdbu"][::-1])

GROUPS = ["Control", "Treated", "Relapse"]
GROUP_COLORS = ["#4c72a8", "#c0603f", "#7a9a53"]
SEX_COLORS = ["#8e6bbf", "#d8b26e"]
STAGE_COLORS = ["#cfd8e3", "#8fa6c2", "#3f5a80"]
MODULE_COLORS = ["#2a7f7a", "#c9a227", "#8c2f2f", "#5b6e9e"]

GENES = [
    "IFIT1", "ISG15", "MX1", "OAS1", "STAT1", "IRF7", "CXCL10", "RSAD2",
    "COL1A1", "COL3A1", "FN1", "POSTN", "TGFB1", "SPARC", "LUM", "TIMP1",
    "MKI67", "CCNB1", "TOP2A", "AURKA", "BUB1", "PLK1", "CDK1", "RRM2",
    "ALB", "APOA1", "CYP3A4", "HNF4A", "TTR", "SERPINA1", "G6PC", "PCK1",
    "SOD2", "NQO1",
]


def synth_expression(rng):
    """Genes fall into 4 co-expression modules with group-specific shifts."""
    n_genes = len(GENES)
    n_per = 13
    n_samp = n_per * len(GROUPS)
    module_of = np.repeat(np.arange(4), 8)[:n_genes]
    module_of = np.concatenate([module_of, np.full(n_genes - module_of.size, 3)])

    # module x group mean effect (log2 fold-change like)
    effect = np.array([
        [-0.9, 1.5, 0.6],    # interferon: up on treatment
        [0.4, -1.1, 1.2],    # matrix: down then rebound
        [-1.2, 0.3, 1.6],    # proliferation: highest at relapse
        [1.3, -0.4, -1.0],   # hepatic/metabolic: lost with disease
    ])
    group_of = np.repeat(np.arange(len(GROUPS)), n_per)
    X = np.zeros((n_genes, n_samp))
    for g in range(n_genes):
        m = module_of[g]
        X[g] = effect[m, group_of] * rng.uniform(0.7, 1.3)
    # per-sample library-size / batch drift + gene noise
    X += rng.normal(0, 0.45, X.shape)
    X += np.linspace(-0.35, 0.35, n_samp)[None, :] * rng.uniform(0.4, 1.0, (n_genes, 1))
    X += rng.normal(0, 0.25, (n_genes, 1))
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return Z, module_of, group_of, n_samp


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(2024)
    Z, module_of, group_of, n_samp = synth_expression(rng)
    n_genes = Z.shape[0]

    sex = rng.integers(0, 2, n_samp)
    stage = np.clip(group_of + rng.integers(0, 2, n_samp), 0, 2)

    # ---- row clustering ---------------------------------------------------
    link = linkage(pdist(Z, metric="correlation"), method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    Zo = Z[order]
    mod_o = module_of[order]
    genes_o = [GENES[i] for i in order]

    # ---- sort samples inside each group by module-1 score -----------------
    score = Zo[mod_o == 0].mean(0)
    col_order = np.concatenate([
        np.where(group_of == g)[0][np.argsort(score[group_of == g])]
        for g in range(len(GROUPS))
    ])
    Zo = Zo[:, col_order]
    g_o, sex_o, stage_o = group_of[col_order], sex[col_order], stage[col_order]

    fig = plt.figure(figsize=(9.2, 7.3))
    L, R, T, B = 0.055, 0.695, 0.885, 0.085
    gs = fig.add_gridspec(
        4, 3, left=L, right=R, top=T, bottom=B,
        width_ratios=[0.13, 1.0, 0.028], height_ratios=[0.032, 0.032, 0.032, 1.0],
        wspace=0.018, hspace=0.16,
    )

    # ---- annotation tracks ------------------------------------------------
    tracks = [("Group", g_o, GROUP_COLORS, GROUPS),
              ("Sex", sex_o, SEX_COLORS, ["Female", "Male"]),
              ("Stage", stage_o, STAGE_COLORS, ["I-II", "III", "IV"])]
    for r, (name, vals, colors, _) in enumerate(tracks):
        axt = fig.add_subplot(gs[r, 1])
        axt.imshow(vals[None, :], aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors)))
        axt.set_xticks([]); axt.set_yticks([])
        for s in axt.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#333333")
        axt.text(-0.012, 0.5, name, transform=axt.transAxes, ha="right",
                 va="center", fontsize=7.5, color="#1a1a1a")

    # ---- row dendrogram ---------------------------------------------------
    axd = fig.add_subplot(gs[3, 0])
    dendrogram(link, ax=axd, orientation="left", color_threshold=0,
               above_threshold_color="#666666", no_labels=True)
    axd.invert_yaxis()
    axd.axis("off")

    # ---- main heatmap -----------------------------------------------------
    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(Zo, aspect="auto", cmap=EXPR_CMAP, vmin=-2.2, vmax=2.2,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_edgecolor("#333333")
    # group separators + bottom group captions
    bounds = np.cumsum([np.sum(g_o == g) for g in range(len(GROUPS))])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.4)
    start = 0
    for gi, b in enumerate(bounds):
        ax.text((start + b - 1) / 2, n_genes + 0.4, f"{GROUPS[gi]}\n(n={b - start})",
                ha="center", va="top", fontsize=7.5, color=GROUP_COLORS[gi],
                fontweight="bold")
        start = b

    # ---- module blocks + gene labels --------------------------------------
    axm = fig.add_subplot(gs[3, 2])
    axm.imshow(mod_o[:, None], aspect="auto", interpolation="nearest",
               cmap=ListedColormap(MODULE_COLORS),
               norm=BoundaryNorm(np.arange(5) - 0.5, 4))
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#333333")
    for i, gname in enumerate(genes_o):
        axm.text(0.85, i, gname, ha="left", va="center", fontsize=6.2,
                 color="#1a1a1a", clip_on=False)

    # ---- shared colourbar --------------------------------------------------
    cax = fig.add_axes([0.800, 0.605, 0.016, 0.155])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, -1, 0, 1, 2])
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(length=2.5, labelsize=7)
    cb.ax.set_title("Row\nz-score", fontsize=7.5, fontweight="bold", pad=5,
                    loc="left")

    # ---- stacked legends ---------------------------------------------------
    axl = fig.add_axes([0.795, B, 0.20, 0.455]); axl.axis("off")
    blocks = [("Group", GROUPS, GROUP_COLORS),
              ("Sex", ["Female", "Male"], SEX_COLORS),
              ("Stage", ["I-II", "III", "IV"], STAGE_COLORS),
              ("Module", ["M1 interferon", "M2 matrix", "M3 cell cycle",
                          "M4 metabolic"], MODULE_COLORS)]
    y = 1.0
    for title, labels, colors in blocks:
        handles = [Patch(facecolor=c, edgecolor="#333333", lw=0.4, label=l)
                   for l, c in zip(labels, colors)]
        leg = axl.legend(handles=handles, title=title, loc="upper left",
                         bbox_to_anchor=(0.0, y), fontsize=7, handlelength=1.0,
                         handleheight=0.9, borderpad=0.0, labelspacing=0.3,
                         handletextpad=0.5)
        leg.get_title().set_fontsize(7.5)
        leg.get_title().set_fontweight("bold")
        leg._legend_box.align = "left"
        axl.add_artist(leg)
        y -= 0.115 + 0.045 * len(labels)

    savefig(fig, "71_heatmap_annotated.png")


if __name__ == "__main__":
    main()
