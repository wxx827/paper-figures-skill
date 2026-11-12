"""Link prediction on a graph with a fifth of its edges hidden.

The graph is a latent-geometry model: three clusters of points, wired with a
probability that decays with distance.  The drawing uses those latent
positions directly, so a reader can see *why* a prediction is plausible - the
model is trying to recover proximity it was never shown.

The main panel carries three layers.  Grey is the observed graph the encoder
was fitted on, dark dashed is the answer key (edges deleted before fitting),
and the thick coloured edges are the decoder's top-ranked candidates, shaded
by score.  A coloured edge lying on a dashed one is a recovered link; one
lying on nothing is a plausible mistake.

Scoring follows the usual protocol: a spectral encoder of the observed graph
with an inner-product decoder, evaluated against an equal number of sampled
non-edges.  AUC, average precision and Hits@K in the auxiliary panels all come
from that one score vector.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from pubstyle import (set_pub_style, savefig, get_palette, add_panel_label,
                      annotate_box)
from figkit import safe_colorbar

CLASS_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]     # house序位 1, 2, 3
TRUTH = "#333333"
INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#777777", "#333333"
CONF_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "conf", get_palette("cb_sequential_purple")[2:])

N_PER, N_CLASS = 20, 3
SPREAD, SIGMA, PMAX = 0.30, 0.36, 0.95
HIDE_FRAC = 0.20
EMB_DIM = 8
TOP_K = 14
HITS_AT = (1, 3, 5, 10)
SEED = 55


def geometric_graph(rng):
    """Latent positions in three clusters; edge probability decays with range."""
    labels = np.repeat(np.arange(N_CLASS), N_PER)
    n = labels.size
    centres = np.stack([[np.cos(a), np.sin(a)]
                        for a in np.radians([90, 210, 330])]) * 0.62
    pos = centres[labels] + rng.normal(0, SPREAD, (n, 2))
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    p = PMAX * np.exp(-(d / SIGMA) ** 2)
    np.fill_diagonal(p, 0.0)
    adj = np.triu((rng.random((n, n)) < p).astype(float), 1)
    return adj + adj.T, labels, pos / np.abs(pos).max()


def hide_edges(edges, adj, rng, frac):
    """Remove a fraction of the edges without stranding either endpoint."""
    deg = adj.sum(1).copy()
    want = int(round(frac * len(edges)))
    hidden = []
    for i in rng.permutation(len(edges)):
        if len(hidden) >= want:
            break
        a, b = edges[i]
        if deg[a] > 2 and deg[b] > 2:
            hidden.append(edges[i])
            deg[a] -= 1
            deg[b] -= 1
    return np.array(hidden)


def encode(adj, dim=EMB_DIM):
    """Spectral encoder over the observed graph, then row-normalised.

    The leading non-trivial eigenvectors of the normalised adjacency are what
    a converged GCN encoder settles on; using them keeps the figure honest
    without pulling in a training loop.
    """
    n = adj.shape[0]
    a = adj + np.eye(n)
    inv = 1.0 / np.sqrt(a.sum(1))
    a = a * inv[:, None] * inv[None, :]
    vals, vecs = np.linalg.eigh(a)
    idx = np.argsort(-vals)[1:dim + 1]               # drop the trivial mode
    z = vecs[:, idx] * vals[idx]
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    adj_true, labels, pos = geometric_graph(rng)
    n = labels.size

    true_edges = np.stack(np.nonzero(np.triu(adj_true, 1)), axis=1)
    hidden = hide_edges(true_edges, adj_true, rng, HIDE_FRAC)
    n_hide = len(hidden)

    adj_obs = adj_true.copy()
    for a, b in hidden:
        adj_obs[a, b] = adj_obs[b, a] = 0.0
    obs_edges = np.stack(np.nonzero(np.triu(adj_obs, 1)), axis=1)

    z = encode(adj_obs)
    cu, cv = np.triu_indices(n, 1)
    free = adj_obs[cu, cv] == 0
    cu, cv = cu[free], cv[free]
    score_all = (z[cu] * z[cv]).sum(1)
    hidden_set = {(int(a), int(b)) for a, b in hidden}
    y_all = np.array([(int(a), int(b)) in hidden_set for a, b in zip(cu, cv)],
                     dtype=int)

    # balanced test set: every hidden edge against one sampled non-edge
    pos_i = np.flatnonzero(y_all == 1)
    neg_i = rng.choice(np.flatnonzero(y_all == 0), n_hide, replace=False)
    test = np.concatenate([pos_i, neg_i])
    y, score = y_all[test], score_all[test]

    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    neg_sorted = np.sort(score[y == 0])[::-1]
    hits = [float((score[y == 1] > neg_sorted[k - 1]).mean()) for k in HITS_AT]

    top = test[np.argsort(-score)[:TOP_K]]
    n_correct = int(y_all[top].sum())

    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58], wspace=0.24,
                          hspace=0.62, left=0.045, right=0.955, top=0.865,
                          bottom=0.145)
    ax_g = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[2, 1])

    # ---- main: graph, answer key, predictions ----------------------------
    for a, b in obs_edges:
        ax_g.plot(pos[[a, b], 0], pos[[a, b], 1], color="#8a8a8a", lw=0.7,
                  alpha=0.30, zorder=1, solid_capstyle="round")
    # the answer key rides *above* the predictions, so a recovered link reads
    # as a dashed dark line on a coloured one and a mistake as bare colour
    for a, b in hidden:
        ax_g.plot(pos[[a, b], 0], pos[[a, b], 1], color=TRUTH, lw=1.25,
                  ls=(0, (3.2, 2.0)), alpha=0.9, zorder=4)
    norm = mpl.colors.Normalize(score_all[top].min(), score_all[top].max())
    for i in top:
        ax_g.plot(pos[[cu[i], cv[i]], 0], pos[[cu[i], cv[i]], 1],
                  color=CONF_CMAP(norm(score_all[i])), lw=2.2, alpha=0.95,
                  zorder=3, solid_capstyle="round")
    for c in range(N_CLASS):
        m = labels == c
        ax_g.scatter(pos[m, 0], pos[m, 1], s=34, c=CLASS_COLORS[c],
                     edgecolors="white", linewidths=0.6, zorder=5)
    ax_g.set_aspect("equal")
    ax_g.axis("off")
    ax_g.set_xlim(-1.12, 1.12)
    ax_g.set_ylim(-1.12, 1.12)
    ax_g.set_title("Observed graph, hidden edges and the top-ranked "
                   "candidates", fontsize=9.0, fontweight="bold", color=INK,
                   pad=6)
    annotate_box(ax_g, f"|V| = {n},  observed |E| = {len(obs_edges)}\n"
                       f"{n_hide} edges hidden ({100 * HIDE_FRAC:.0f}%)\n"
                       f"top {TOP_K}: {n_correct} recovered, "
                       f"{TOP_K - n_correct} plausible misses",
                 loc="upper left", fontsize=6.8)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CONF_CMAP)
    safe_colorbar(fig, sm, ax_g, label="decoder score, top-K range (-)",
                  side="bottom", width=0.10, pad=0.10)

    # ---- aux 1: ROC --------------------------------------------------------
    ax_r.plot(fpr, tpr, color="#3b5b92", lw=1.6, zorder=3)
    ax_r.fill_between(fpr, 0, tpr, color="#3b5b92", alpha=0.10, zorder=2)
    ax_r.plot([0, 1], [0, 1], color="#808080", lw=0.8, ls="--", zorder=1)
    ax_r.set_xlabel("false-positive rate (-)", fontsize=8.0)
    ax_r.set_ylabel("true-positive rate (-)", fontsize=8.0)
    ax_r.set_title(f"ROC  ·  AUC = {roc_auc:.3f}", fontsize=8.5,
                   fontweight="bold", color=INK, pad=4)

    # ---- aux 2: precision-recall -------------------------------------------
    ax_p.plot(rec, prec, color="#2e7d5b", lw=1.6, zorder=3)
    ax_p.axhline(0.5, color="#808080", lw=0.8, ls="--", zorder=1)
    ax_p.text(0.02, 0.53, "chance 0.50 (balanced)", ha="left", va="bottom",
              fontsize=6.4, color=MUTED)
    ax_p.set_xlabel("recall (-)", fontsize=8.0)
    ax_p.set_ylabel("precision (-)", fontsize=8.0)
    ax_p.set_title(f"Precision-recall  ·  AP = {ap:.3f}", fontsize=8.5,
                   fontweight="bold", color=INK, pad=4)

    for ax in (ax_r, ax_p):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.04)
        ax.tick_params(labelsize=6.8, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- aux 3: Hits@K -----------------------------------------------------
    bars = ax_h.bar(np.arange(len(HITS_AT)), [100 * h for h in hits],
                    width=0.58, color="#8a8a8a", edgecolor="white",
                    linewidth=0.6)
    bars[-1].set_color("#b23a48")
    for i, h in enumerate(hits):
        ax_h.text(i, 100 * h + 2.5, f"{100 * h:.0f}%", ha="center",
                  va="bottom", fontsize=6.8, color=SUB)
    ax_h.set_xticks(np.arange(len(HITS_AT)))
    ax_h.set_xticklabels([f"@{k}" for k in HITS_AT], fontsize=7.0)
    ax_h.set_ylim(0, 112)
    ax_h.set_yticks([0, 25, 50, 75, 100])
    ax_h.set_xlabel("ranked above the K-th best non-edge (-)", fontsize=8.0)
    ax_h.set_ylabel("Hits@K (%)", fontsize=8.0)
    ax_h.tick_params(labelsize=6.8, length=2.5)
    for s in ("top", "right"):
        ax_h.spines[s].set_visible(False)
    ax_h.set_title(f"Hits@K over {n_hide} hidden edges", fontsize=8.5,
                   fontweight="bold", color=INK, pad=4)

    for ax, lab in zip((ax_g, ax_r, ax_p, ax_h), ("(a)", "(b)", "(c)", "(d)")):
        add_panel_label(ax, lab, x=-0.05 if ax is ax_g else -0.20)

    handles = [
        Line2D([], [], color="#8a8a8a", lw=1.1, alpha=0.6,
               label="observed edge"),
        Line2D([], [], color=TRUTH, lw=1.3, ls=(0, (3.2, 2.0)),
               label="hidden true edge"),
        Line2D([], [], color=CONF_CMAP(0.85), lw=2.2,
               label=f"top-{TOP_K} prediction"),
    ] + [Line2D([], [], marker="o", ls="", ms=4.4, mfc=c, mec="white",
                mew=0.6, label=f"cluster {t}")
         for c, t in zip(CLASS_COLORS, "ABC")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=6, frameon=False,
               fontsize=7.2)
    savefig(fig, "450_gnn_link_prediction.png")


if __name__ == "__main__":
    main()
