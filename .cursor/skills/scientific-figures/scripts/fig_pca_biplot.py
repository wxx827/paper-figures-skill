"""PCA biplot with scree and cumulative-variance companions.

Left: sample scores coloured by clinical group with 95% confidence ellipses,
overlaid by scaled variable loadings drawn as labelled arrows. Right: the
eigenvalue scree bars and the cumulative explained-variance curve. PCA is a
plain SVD of the standardised matrix, so no sklearn dependency is needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

GROUPS = ["Healthy", "Mild", "Severe", "Recovered"]
GROUP_C = ["#3f7f93", "#d4a13a", "#b23a48", "#6a8f4f"]
VARS = ["IL6", "TNFA", "CRP", "ALT", "AST", "HDL", "LDL", "Glucose",
        "Insulin", "Lactate", "Ferritin", "Albumin"]


def synth(rng, n_per=26):
    """Two latent axes (inflammation, metabolic tone) drive all readouts."""
    n = n_per * len(GROUPS)
    infl = np.concatenate([rng.normal(m, 0.55, n_per)
                           for m in (-1.1, 0.1, 1.5, -0.3)])
    metab = np.concatenate([rng.normal(m, 0.55, n_per)
                            for m in (0.9, 0.2, -1.2, 0.6)])
    load_i = np.array([1.5, 1.4, 1.6, 0.7, 0.6, -0.9, 0.4, 0.5, 0.3, 1.0,
                       1.2, -1.3])
    load_m = np.array([0.2, 0.1, -0.2, 0.8, 0.9, 1.1, 1.3, 1.4, 1.5, -0.4,
                       -0.1, 0.6])
    X = np.outer(infl, load_i) + np.outer(metab, load_m)
    X += rng.normal(0, 0.55, X.shape)
    X += rng.normal(0, 0.3, (1, X.shape[1]))          # assay offsets
    labels = np.repeat(np.arange(len(GROUPS)), n_per)
    return X, labels


def pca(X):
    Z = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U * S
    evr = S ** 2 / np.sum(S ** 2)
    return scores, Vt.T, evr


def repel2d(pos, w, h, rounds=300):
    """Separate label boxes that overlap, preferring vertical moves."""
    pos = pos.copy()
    for _ in range(rounds):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx, dy = pos[j] - pos[i]
                ox = (w[i] + w[j]) / 2 - abs(dx)
                oy = (h[i] + h[j]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    step = oy / 2 + h[i] * 0.06
                    s = 1.0 if dy >= 0 else -1.0
                    pos[i, 1] -= s * step
                    pos[j, 1] += s * step
                    moved = True
        if not moved:
            break
    return pos


def conf_ellipse(ax, x, y, color, nsig=0.95):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(stats.chi2.ppf(nsig, 2))
    w, h = 2 * scale * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, fc=color,
                         alpha=0.10, ec=color, lw=1.0, ls=(0, (5, 2)),
                         zorder=1))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(31)
    X, labels = synth(rng)
    scores, loadings, evr = pca(X)
    pc1, pc2 = scores[:, 0], scores[:, 1]

    fig = plt.figure(figsize=(9.4, 5.5))
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, top=0.845,
                          bottom=0.10, width_ratios=[1.0, 0.46],
                          hspace=0.55, wspace=0.20)

    # ---- biplot -----------------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.axhline(0, color="#cccccc", lw=0.7, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.7, zorder=0)
    for g, (name, c) in enumerate(zip(GROUPS, GROUP_C)):
        m = labels == g
        ax.scatter(pc1[m], pc2[m], s=22, c=c, alpha=0.85, lw=0.4,
                   edgecolor="white", label=name, zorder=3)
        conf_ellipse(ax, pc1[m], pc2[m], c)

    span = max(np.abs(pc1).max(), np.abs(pc2).max()) * 1.22
    # normalise so the longest loading reaches most of the plotting radius
    arrow_scale = span * 0.78 / np.hypot(loadings[:, 0], loadings[:, 1]).max()
    tips = loadings[:, :2] * arrow_scale
    for dx, dy in tips:
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#3a3a3a",
                                    mutation_scale=7, shrinkA=0, shrinkB=0),
                    zorder=4)
    # co-linear loadings would stack their labels, so de-collide them in data
    # space and keep a leader line back to each arrow tip
    labels = repel2d(tips * 1.13,
                     np.array([0.022 * span * len(v) + 0.05 * span
                               for v in VARS]),
                     np.full(len(VARS), 0.062 * span))
    for k, v in enumerate(VARS):
        lx, ly = labels[k]
        if np.hypot(lx - tips[k, 0], ly - tips[k, 1]) > 0.09 * span:
            ax.plot([tips[k, 0], lx], [tips[k, 1], ly], color="#9aa0a6",
                    lw=0.45, zorder=4)
        ax.text(lx, ly, v, fontsize=6.6, fontweight="bold", color="#1a1a1a",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                          alpha=0.78))
    ax.set_xlim(-span, span)
    ax.set_ylim(-span * 0.95, span * 0.95)
    ax.set_xlabel(f"PC1  ({evr[0] * 100:.1f}% explained variance)")
    ax.set_ylabel(f"PC2  ({evr[1] * 100:.1f}% explained variance)")
    ax.legend(loc="lower left", fontsize=7.4, ncol=2, columnspacing=1.0,
              handletextpad=0.35, borderpad=0.25,
              title="Clinical group", title_fontsize=7.6)
    ax.text(0.985, 0.015, "arrows: variable loadings\n(scaled for display)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#666666", linespacing=1.4)
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.085, y=1.01)

    # ---- scree ------------------------------------------------------------
    k = 8
    axs = fig.add_subplot(gs[0, 1])
    axs.bar(np.arange(1, k + 1), evr[:k] * 100, width=0.62, color="#8fa6c2",
            edgecolor="#33475e", lw=0.5)
    for i, v in enumerate(evr[:k] * 100):
        axs.text(i + 1, v + 1.4, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=6.2, color="#33475e")
    axs.set_xticks(np.arange(1, k + 1))
    axs.set_ylim(0, evr[0] * 100 * 1.24)
    axs.set_xlabel("Principal component", labelpad=2)
    axs.set_ylabel("Variance (%)")
    axs.set_title("Scree", fontsize=8.5, fontweight="bold", loc="left", pad=3)
    style_axes(axs)
    add_panel_label(axs, "(b)", x=-0.20, y=1.01)

    # ---- cumulative variance ----------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    cum = np.cumsum(evr) * 100
    axc.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=3.4,
             color="#b23a48", lw=1.5, mfc="white", mew=1.0)
    axc.axhline(90, color="#666666", ls=(0, (4, 3)), lw=0.7)
    n90 = int(np.argmax(cum >= 90) + 1)
    axc.text(len(cum) * 0.98, 91.5, f"90% at PC{n90}", ha="right",
             va="bottom", fontsize=6.8, color="#444444")
    axc.set_xticks(np.arange(1, len(cum) + 1, 2))
    axc.set_ylim(0, 105)
    axc.set_xlabel("Number of components", labelpad=2)
    axc.set_ylabel("Cumulative (%)")
    axc.set_title("Cumulative variance", fontsize=8.5, fontweight="bold",
                  loc="left", pad=3)
    style_axes(axc)
    add_panel_label(axc, "(c)", x=-0.20, y=1.01)

    savefig(fig, "75_pca_biplot.png")


if __name__ == "__main__":
    main()
