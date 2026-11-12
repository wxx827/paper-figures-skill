"""Figure 276 - Embedding quality before and after representation learning.

Three t-SNE views of the same ten-class test split: the raw backbone
features, a contrastive baseline and the proposed objective.  Silhouette
scores are computed from the plotted coordinates, and a subordinate strip of
inter-class distance matrices underneath quantifies the visual separation on
one shared colour scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score

from pubstyle import set_pub_style, PALETTES, savefig

DIST = "magma"
CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
COLORS = PALETTES["cb_safe"] + ["#8e6bbf", "#2a7f7a"]
METHODS = [("Raw features", 0.30, 1.05, 0.55),
           ("Baseline (SupCon)", 0.72, 0.62, 0.28),
           ("Ours", 1.00, 0.30, 0.10)]
PER_CLASS = 90


def embed(seed, sep, spread, drift):
    """Ten Gaussian clusters on a ring; ``sep`` sets how far apart they sit."""
    rng = np.random.default_rng(seed)
    k = len(CLASSES)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + 0.22
    radius = 4.6 * sep
    cx = radius * np.cos(ang) + 0.9 * sep * rng.standard_normal(k)
    cy = radius * np.sin(ang) + 0.9 * sep * rng.standard_normal(k)
    # A few semantically close pairs are pulled together in every embedding.
    for a, b, pull in ((3, 5, 0.42), (2, 6, 0.30), (1, 9, 0.38), (0, 8, 0.26)):
        mx, my = (cx[a] + cx[b]) / 2, (cy[a] + cy[b]) / 2
        for i in (a, b):
            cx[i] += pull * (mx - cx[i])
            cy[i] += pull * (my - cy[i])

    xs, ys, lab = [], [], []
    for i in range(k):
        n = PER_CLASS
        s = spread * (0.85 + 0.35 * rng.random())
        ang_i = rng.uniform(0, np.pi)
        a = rng.standard_normal(n) * s * 1.35
        b = rng.standard_normal(n) * s * 0.75
        px = cx[i] + a * np.cos(ang_i) - b * np.sin(ang_i)
        py = cy[i] + a * np.sin(ang_i) + b * np.cos(ang_i)
        # A handful of outliers wander toward the global centre.
        m = rng.random(n) < drift * 0.30
        px[m] += drift * 2.0 * (0 - px[m]) * rng.random(m.sum())
        py[m] += drift * 2.0 * (0 - py[m]) * rng.random(m.sum())
        xs.append(px)
        ys.append(py)
        lab.append(np.full(n, i))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    X = X - X.mean(axis=0)
    X = X / X.std()
    return X, np.concatenate(lab)


def class_distances(X, y):
    """Centroid distance divided by the two classes' own scatter.

    A scale-free separation index: it grows only when clusters move apart
    *relative to their own size*, which is exactly what the three panels are
    supposed to differ in.
    """
    k = len(CLASSES)
    cent = np.array([X[y == i].mean(axis=0) for i in range(k)])
    rms = np.array([np.sqrt(((X[y == i] - cent[i]) ** 2).sum(axis=1).mean())
                    for i in range(k)])
    d = np.hypot(cent[:, None, 0] - cent[None, :, 0],
                 cent[:, None, 1] - cent[None, :, 1])
    return d / (0.5 * (rms[:, None] + rms[None, :]))


def panel_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#bcbcbc")


def main():
    set_pub_style(base_font=8)
    runs = []
    for i, (name, sep, spread, drift) in enumerate(METHODS):
        X, y = embed(101 + 17 * i, sep, spread, drift)
        runs.append((name, X, y, silhouette_score(X, y),
                     class_distances(X, y)))

    dnorm = Normalize(vmin=0.0, vmax=14.0)
    # one span for all three embeddings: per-panel autoscaling would rescale a
    # loosely spread method down to look as compact as a tight one
    lim = 1.07 * max(np.abs(X).max() for _, X, _, _, _ in runs)

    fig = plt.figure(figsize=(7.2, 4.75))
    w_frac, gap = 1.853 / 7.2, 0.035 * 1.853 / 7.2
    lefts = [0.055 + i * (w_frac + gap) for i in range(3)]
    top, hgt = 0.895, 1.853 / 4.75
    mw, mh = 1.019 / 7.2, 1.019 / 4.75
    mbottom = 0.145

    band = plt.Rectangle((0.040, 0.088), 0.905, 0.392, transform=fig.transFigure,
                         facecolor="#f4f4f4", edgecolor="none", zorder=-5)
    fig.add_artist(band)

    for i, (name, X, y, sil, dmat) in enumerate(runs):
        ax = fig.add_axes([lefts[i], top - hgt, w_frac, hgt])
        panel_axes(ax)
        for c in range(len(CLASSES)):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], s=5.0, c=COLORS[c], alpha=0.78,
                       linewidths=0.0, rasterized=True)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        best = name == "Ours"
        ax.set_title(name, fontsize=8.2, pad=3.6,
                     fontweight="bold" if best else "normal",
                     color="#12503a" if best else "#1a1a1a")
        ax.text(0.035, 0.035, f"silhouette {sil:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.6,
                fontweight="bold" if best else "normal",
                color="#1a5c38" if best else "#1a1a1a", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#b8b8b8", linewidth=0.4, alpha=0.92))

        axm = fig.add_axes([lefts[i] + (w_frac - mw) / 2, mbottom, mw, mh])
        axm.set_aspect("equal")
        axm.imshow(dmat, cmap=DIST, norm=dnorm, interpolation="nearest")
        axm.set_xticks(range(len(CLASSES)))
        axm.set_yticks(range(len(CLASSES)))
        axm.set_xticklabels([])
        axm.set_yticklabels([])
        axm.tick_params(length=1.2, width=0.4, color="#999999")
        for s in axm.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#bcbcbc")
        axm.set_xlabel(f"mean ratio {dmat[np.triu_indices(10, 1)].mean():.1f}",
                       fontsize=6.3, color="#4a4a4a", labelpad=2.0)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.0,
                      markerfacecolor=COLORS[c], markeredgecolor="none",
                      label=CLASSES[c]) for c in range(len(CLASSES))]
    leg = fig.legend(handles=handles, loc="center left",
                     bbox_to_anchor=(0.862, top - hgt / 2), frameon=False,
                     fontsize=7.0, handletextpad=0.35, labelspacing=0.42,
                     borderpad=0.0)
    leg.set_clip_on(False)
    fig.text(0.862, top - 0.012, "class", fontsize=7.0, fontweight="bold",
             ha="left", va="top")

    cax = fig.add_axes([lefts[2] + (w_frac - mw) / 2 + mw + 0.016,
                        mbottom, 0.012, mh])
    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap=DIST), cax=cax,
                      extend="max", ticks=[0, 4, 8, 12])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("separation ratio (-)", fontsize=7.0, color="#1a1a1a",
                 labelpad=3.0)

    fig.text(0.055, top - hgt - 0.055,
             "Inter-class separation matrices  (10 x 10, one shared scale)",
             fontsize=7.2, fontweight="bold", ha="left", va="top",
             color="#3a3a3a")
    fig.text(0.055, top - hgt - 0.105,
             "centroid distance divided by within-class scatter, so the three "
             "matrices are directly comparable; class order follows the legend",
             fontsize=6.4, ha="left", va="top", color="#7a7a7a")


    savefig(fig, "276_tsne_embedding.png")


if __name__ == "__main__":
    main()
