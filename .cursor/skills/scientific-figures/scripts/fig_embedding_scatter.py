"""t-SNE style embedding of a single-cell dataset with cluster annotation.

(a) Main embedding: every cell is a point coloured by its annotated type, each
    type wrapped in a core convex hull with a soft fill and a haloed centroid
    label. Three inset thumbnails in the free lower-left corner compare the
    same data embedded at three perplexity settings.
(b) Combined legend / composition panel: horizontal bars of cells per type,
    labelled with count and share, doubling as the colour key.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_score
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

# name, centre, spread (sx, sy), rotation, n cells, colour
CLUSTERS = [
    ("CD4+ T cells", (-8.4, 5.4), (2.5, 1.6), 0.55, 3120, "#3f6f9c"),
    ("CD8+ T cells", (-2.4, 11.4), (1.9, 1.4), -0.35, 1840, "#7ba7cc"),
    ("B cells", (6.2, 11.8), (1.7, 1.5), 0.20, 1260, "#8a5f9c"),
    ("NK cells", (13.4, 6.6), (1.4, 1.7), 0.90, 720, "#4a7c59"),
    ("Monocytes", (13.8, -2.0), (2.2, 1.9), -0.60, 2450, "#b0653a"),
    ("Dendritic cells", (6.4, -7.2), (1.3, 1.1), 0.30, 410, "#c39b3f"),
    ("Platelets", (15.2, -8.4), (0.9, 0.8), 0.0, 180, "#a8493f"),
]


def sample_cluster(rng, centre, spread, rot, n, tail=0.04):
    """Anisotropic, slightly heavy-tailed Gaussian blob."""
    z = rng.normal(size=(n, 2)) * np.array(spread)
    heavy = rng.random(n) < tail
    z[heavy] *= rng.uniform(1.6, 2.4, (heavy.sum(), 1))
    c, s = np.cos(rot), np.sin(rot)
    z = z @ np.array([[c, -s], [s, c]])
    return z + np.array(centre)


def core_hull(pts, keep=0.88):
    """Convex hull of the central `keep` fraction, so outliers do not bloat it."""
    d = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    core = pts[d <= np.quantile(d, keep)]
    return core[ConvexHull(core).vertices]


def draw_embedding(ax, xy, lab, ms=3.6, hulls=True, labels=True):
    for i, (name, *_rest) in enumerate(CLUSTERS):
        col = CLUSTERS[i][5]
        p = xy[lab == i]
        ax.scatter(p[:, 0], p[:, 1], s=ms, color=col, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=2)
        if hulls:
            v = core_hull(p)
            ax.add_patch(Polygon(v, closed=True, facecolor=col, alpha=0.09,
                                 edgecolor="none", zorder=3))
            ax.add_patch(Polygon(v, closed=True, fill=False, edgecolor=col,
                                 lw=1.15, alpha=0.95, zorder=4,
                                 joinstyle="round"))
        if labels:
            c = np.median(p, axis=0)
            ax.text(c[0], c[1], name, fontsize=7.4, fontweight="bold",
                    ha="center", va="center", color="#1a1a1a", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="white")])


def build(rng, scale=1.0, pull=0.0, frag=0):
    """`scale` widens blobs, `pull` contracts centres, `frag` splits clusters."""
    xs, ls = [], []
    for i, (_n, centre, spread, rot, n, _c) in enumerate(CLUSTERS):
        c = np.array(centre) * (1 - pull)
        p = sample_cluster(rng, c, np.array(spread) * scale, rot, n)
        if frag:
            offs = rng.normal(0, 2.3, (frag, 2))
            p = p + offs[rng.integers(0, frag, n)]
        xs.append(p)
        ls.append(np.full(n, i))
    return np.vstack(xs), np.concatenate(ls)


def main():
    set_pub_style()
    rng = np.random.default_rng(606)
    xy, lab = build(rng)

    sub = rng.choice(xy.shape[0], 2500, replace=False)
    sil = silhouette_score(xy[sub], lab[sub])

    fig = plt.figure(figsize=(10.6, 6.9), layout="constrained")
    fig.get_layout_engine().set(rect=(0.004, 0.030, 0.992, 0.935),
                                h_pad=0.06, w_pad=0.10)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.95, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    draw_embedding(ax, xy, lab)
    xlo, xhi = np.percentile(xy[:, 0], [0.3, 99.7])
    ylo, yhi = np.percentile(xy[:, 1], [0.3, 99.7])
    xpad, ypad = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad - 0.34 * (yhi - ylo), yhi + ypad)  # band for insets
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE dimension 1", fontsize=8.2)
    ax.set_ylabel("t-SNE dimension 2", fontsize=8.2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    ax.text(0.985, 0.975,
            f"n = {xy.shape[0]:,} cells\n{len(CLUSTERS)} annotated types\n"
            f"silhouette = {sil:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            linespacing=1.6, color="#2b2b2b",
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.92))
    add_panel_label(ax, "(a)", x=-0.045, y=1.005, size=10)
    ax.set_title("t-SNE embedding of PBMC transcriptomes "
                 "(perplexity 30, 2,000 HVGs)",
                 fontsize=9.0, fontweight="bold", pad=4)

    # ---- inset thumbnails: perplexity sweep ------------------------------
    ax.text(0.016, 0.250, "Perplexity sweep (same cells, same seed)",
            transform=ax.transAxes, fontsize=6.8, fontweight="bold",
            color="#444444", ha="left", va="bottom")
    settings = [("perplexity 5", 0.42, 0.10, 3), ("perplexity 30", 1.0, 0.0, 0),
                ("perplexity 100", 1.5, 0.58, 0)]
    for k, (tag, sc, pull, frag) in enumerate(settings):
        sub_ax = ax.inset_axes([0.016 + k * 0.132, 0.032, 0.120, 0.200])
        sxy, slab = build(np.random.default_rng(606 + k), scale=sc, pull=pull,
                          frag=frag)
        draw_embedding(sub_ax, sxy, slab, ms=0.9, hulls=False, labels=False)
        sub_ax.set_xlim(*np.percentile(sxy[:, 0], [0.5, 99.5]))
        sub_ax.set_ylim(*np.percentile(sxy[:, 1], [0.5, 99.5]))
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(tag, fontsize=6.2, pad=1.6,
                         fontweight="bold" if k == 1 else "normal",
                         color="#1a1a1a" if k == 1 else "#666666")
        for s in sub_ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0 if k == 1 else 0.6)
            s.set_edgecolor("#333333" if k == 1 else "#aaaaaa")

    # ---- (b) composition bars = colour key -------------------------------
    counts = np.array([c[4] for c in CLUSTERS])
    names = [c[0] for c in CLUSTERS]
    cols = [c[5] for c in CLUSTERS]
    o = np.argsort(-counts)
    ypos = np.arange(len(counts))
    axb.barh(ypos, counts[o], height=0.62, color=[cols[i] for i in o],
             alpha=0.90, edgecolor="white", linewidth=0.6)
    total = counts.sum()
    for k, i in enumerate(o):
        axb.text(counts[i] + total * 0.012, k,
                 f"{counts[i]:,}   ({100 * counts[i] / total:.1f}%)",
                 ha="left", va="center", fontsize=7.0, color="#2b2b2b")
    axb.set_yticks(ypos, [names[i] for i in o], fontsize=7.6)
    for tick, i in zip(axb.get_yticklabels(), o):
        tick.set_color(cols[i])
        tick.set_fontweight("bold")
    axb.set_xlim(0, counts.max() * 1.52)
    axb.set_ylim(len(counts) - 0.35, -0.75)
    axb.set_xlabel("Cells per annotated type", fontsize=8.0)
    axb.tick_params(labelsize=7.0, length=2.6)
    axb.xaxis.set_major_locator(plt.MaxNLocator(4))
    style_axes(axb)
    add_panel_label(axb, "(b)", x=-0.42, y=1.005, size=10)
    axb.set_title("Cluster composition", fontsize=9.0, fontweight="bold",
                  loc="left", pad=4)

    savefig(fig, "69_embedding_scatter.png")


if __name__ == "__main__":
    main()
