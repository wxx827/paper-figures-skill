"""Multi-omics integration panel: one figure that tells the whole story.

(a) consensus sample-similarity heatmap with molecular-subtype strips,
(b) circle matrix of cross-omics factor correlations,
(c) variance decomposition of each latent factor across omics layers,
(d) cross-omics z-score trajectories for the signature locus.
All four share the same synthetic cohort of 60 tumours profiled on four
omics layers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

SIM_CMAP = LinearSegmentedColormap.from_list(
    "sim", ["#f7f7f5", "#cfe0d8", "#8dbfae", "#3f8d78", "#14523f"])
COR_CMAP = LinearSegmentedColormap.from_list("cor", PALETTES["div_rdbu"][::-1])

LAYERS = ["Transcriptome", "Proteome", "Methylome", "Metabolome"]
LAYER_C = ["#3f5d80", "#c0603f", "#7a9a53", "#8e6bbf"]
SUBTYPES = ["MS1 proliferative", "MS2 immune", "MS3 mesenchymal"]
SUB_C = ["#2a7f7a", "#c9a227", "#b23a48"]
STAGES = ["Stage I", "Stage II", "Stage III", "Stage IV"]
N_PER = 20


def synth(rng):
    """Three subtypes driven by two shared latent factors, expressed with
    layer-specific loadings and layer-specific noise levels."""
    f1 = np.concatenate([rng.normal(m, 0.45, N_PER) for m in (1.4, -0.3, -1.1)])
    f2 = np.concatenate([rng.normal(m, 0.45, N_PER) for m in (-0.5, 1.3, -0.8)])
    blocks, noise = [], [0.75, 0.95, 1.15, 1.35]
    for li, sd in enumerate(noise):
        p = [220, 140, 180, 90][li]
        a = rng.normal(0, 1, p) * (1.0 - 0.13 * li)
        b = rng.normal(0, 1, p) * (0.8 - 0.10 * li)
        X = np.outer(f1, a) + np.outer(f2, b) + rng.normal(0, sd, (60, p))
        blocks.append((X - X.mean(0)) / X.std(0))
    labels = np.repeat(np.arange(3), N_PER)
    return blocks, labels


def top_pcs(X, k=2):
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (U[:, :k] * S[:k])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(2718)
    blocks, labels = synth(rng)
    n = labels.size

    fig = plt.figure(figsize=(10.0, 7.9))
    gs = fig.add_gridspec(2, 2, left=0.070, right=0.955, top=0.845,
                          bottom=0.078, wspace=0.40, hspace=0.46,
                          height_ratios=[1.0, 0.82])

    # ---- (a) consensus similarity -----------------------------------------
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                  height_ratios=[0.05, 1], hspace=0.05)
    Z = np.hstack(blocks)
    sim = np.corrcoef(Z)
    order = np.argsort(labels + rng.uniform(0, 0.9, n))
    sim, lab_o = sim[np.ix_(order, order)], labels[order]

    axs = fig.add_subplot(sub[0])
    axs.imshow(lab_o[None, :], aspect="auto", interpolation="nearest",
               cmap=ListedColormap(SUB_C),
               norm=BoundaryNorm(np.arange(4) - 0.5, 3))
    axs.set_xticks([]); axs.set_yticks([])
    for s in axs.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#333333")
    axs.text(-0.012, 0.5, "Subtype", transform=axs.transAxes, ha="right",
             va="center", fontsize=7)

    axa = fig.add_subplot(sub[1])
    im = axa.imshow(sim, cmap=SIM_CMAP, vmin=-0.15, vmax=1.0, aspect="auto",
                    interpolation="nearest")
    bounds = np.cumsum([np.sum(lab_o == g) for g in range(3)])
    for b in bounds[:-1]:
        axa.axhline(b - 0.5, color="white", lw=1.1)
        axa.axvline(b - 0.5, color="white", lw=1.1)
    axa.set_xticks([]); axa.set_yticks([])
    for s in axa.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_edgecolor("#333333")
    axa.set_xlabel(f"{n} tumours (ordered by consensus subtype)", labelpad=3)
    cb = fig.colorbar(im, ax=axa, fraction=0.043, pad=0.025,
                      ticks=[0, 0.5, 1.0])
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(length=2.5, labelsize=7)
    cb.set_label("Pairwise similarity", fontsize=7.2, labelpad=3)
    add_panel_label(axa, "(a)", x=-0.095, y=1.14)
    axa.set_title("Integrated sample similarity", fontsize=9, loc="left",
                  fontweight="bold", pad=16)
    axa.legend(handles=[Patch(fc=c, ec="#333333", lw=0.4, label=s)
                        for s, c in zip(SUBTYPES, SUB_C)],
               loc="upper left", bbox_to_anchor=(0.0, -0.115), ncol=3,
               fontsize=6.8, handlelength=0.9, handletextpad=0.4,
               columnspacing=0.9, borderpad=0.0)
    # the colourbar shrinks the heatmap, so re-align the annotation strip to it
    pa, ps = axa.get_position(), axs.get_position()
    axs.set_position([pa.x0, ps.y0, pa.width, ps.height])

    # ---- (b) cross-omics factor correlations -------------------------------
    axb = fig.add_subplot(gs[0, 1])
    pcs = np.hstack([top_pcs(X) for X in blocks])
    names = [f"{l[:4]} F{k + 1}" for l in LAYERS for k in (0, 1)]
    C = np.corrcoef(pcs.T)
    m = len(names)
    xx, yy = np.meshgrid(np.arange(m), np.arange(m))
    sc = axb.scatter(xx.ravel(), yy.ravel(), s=np.abs(C).ravel() * 105 + 4,
                     c=C.ravel(), cmap=COR_CMAP, vmin=-1, vmax=1,
                     edgecolor="#3a3a3a", linewidth=0.35)
    for i in range(0, m, 2):
        axb.axhline(i - 0.5, color="#dddddd", lw=0.6)
        axb.axvline(i - 0.5, color="#dddddd", lw=0.6)
    axb.set_xticks(range(m)); axb.set_yticks(range(m))
    axb.set_xticklabels(names, rotation=45, ha="right", fontsize=6.6)
    axb.set_yticklabels(names, fontsize=6.6)
    for tl, li in zip(axb.get_xticklabels(), np.repeat(np.arange(4), 2)):
        tl.set_color(LAYER_C[li])
    for tl, li in zip(axb.get_yticklabels(), np.repeat(np.arange(4), 2)):
        tl.set_color(LAYER_C[li])
    axb.set_xlim(-0.6, m - 0.4); axb.set_ylim(m - 0.4, -0.6)
    axb.tick_params(length=0)
    axb.set_aspect("equal")
    for s in axb.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_edgecolor("#333333")
    cb2 = fig.colorbar(sc, ax=axb, fraction=0.043, pad=0.025,
                       ticks=[-1, 0, 1])
    cb2.outline.set_linewidth(0.6); cb2.outline.set_edgecolor("#333333")
    cb2.ax.tick_params(length=2.5, labelsize=7)
    cb2.set_label("Pearson $r$", fontsize=7.2, labelpad=3)
    add_panel_label(axb, "(b)", x=-0.24, y=1.14)
    axb.set_title("Cross-omics factor coupling", fontsize=9, loc="left",
                  fontweight="bold", pad=16)

    # ---- (c) variance decomposition ----------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    k = 6
    base = np.array([28.0, 19.5, 12.0, 7.5, 4.5, 2.6])
    frac = rng.dirichlet([2.4, 2.0, 1.6, 1.2], k)
    var = frac * base[:, None]
    x = np.arange(1, k + 1)
    bottom = np.zeros(k)
    for li, (name, c) in enumerate(zip(LAYERS, LAYER_C)):
        axc.bar(x, var[:, li], bottom=bottom, width=0.66, color=c,
                edgecolor="white", lw=0.5, label=name)
        bottom += var[:, li]
    for xi, tot in zip(x, bottom):
        axc.text(xi, tot + 0.8, f"{tot:.0f}%", ha="center", va="bottom",
                 fontsize=6.6, color="#444444")
    axc.set_xticks(x)
    axc.set_xticklabels([f"F{i}" for i in x])
    axc.set_ylim(0, bottom.max() * 1.22)
    axc.set_xlabel("Latent factor", labelpad=2)
    axc.set_ylabel("Variance explained (%)")
    axc.legend(loc="upper right", fontsize=7, handlelength=1.0,
               handleheight=0.9, labelspacing=0.3, borderpad=0.2,
               title="Omics layer", title_fontsize=7.2)
    style_axes(axc)
    add_panel_label(axc, "(c)", x=-0.095, y=1.04)
    axc.set_title("Variance decomposition by layer", fontsize=9, loc="left",
                  fontweight="bold", pad=4)

    # ---- (d) cross-omics trajectories --------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    trends = np.array([[-0.9, -0.1, 0.75, 1.35],     # mRNA
                       [-0.75, -0.2, 0.55, 1.15],    # protein
                       [0.85, 0.35, -0.45, -1.05],   # promoter methylation
                       [-0.45, 0.15, 0.65, 0.95]])   # metabolite
    xs = np.arange(len(STAGES))
    for li, (name, c) in enumerate(zip(LAYERS, LAYER_C)):
        mu = trends[li] + rng.normal(0, 0.06, len(STAGES))
        sem = rng.uniform(0.10, 0.20, len(STAGES))
        axd.fill_between(xs, mu - sem, mu + sem, color=c, alpha=0.16, lw=0)
        axd.plot(xs, mu, color=c, lw=1.7, marker="o", ms=4.2, mfc="white",
                 mew=1.3, label=name)
    axd.axhline(0, color="#bbbbbb", lw=0.7, zorder=0)
    axd.set_xticks(xs); axd.set_xticklabels(STAGES)
    axd.set_xlim(-0.25, len(STAGES) - 0.75)
    axd.set_ylim(-1.95, 1.75)
    axd.set_ylabel("Signature z-score")
    axd.set_xlabel("Clinical stage", labelpad=2)
    # parked in the headroom below the curves so no series is obscured
    axd.legend(loc="lower center", fontsize=6.6, handlelength=1.3,
               labelspacing=0.3, borderpad=0.2, ncol=4, columnspacing=0.7,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.02))
    axd.text(0.985, 0.965, "MYC locus, mean $\\pm$ s.e.m.",
             transform=axd.transAxes, ha="right", va="top", fontsize=6.8,
             color="#666666")
    style_axes(axd)
    add_panel_label(axd, "(d)", x=-0.115, y=1.04)
    axd.set_title("Coordinated multi-layer response", fontsize=9, loc="left",
                  fontweight="bold", pad=4)

    savefig(fig, "78_multiomics_panel.png")


if __name__ == "__main__":
    main()
