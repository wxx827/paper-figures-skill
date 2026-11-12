"""Graph attention weights: edge thickness on the graph, heads compared aside.

The main panel is a force-directed layout of a 34-node three-community graph.
Every edge is drawn with a width and an opacity proportional to the
head-averaged attention coefficient it carries, so the subgraph the model
actually listens to stands out from the wiring diagram.  The highest-degree
node v is promoted to the house accent colour and an inset bar chart opens up
its own attention distribution over N(v).

The two auxiliary panels on the right show why "attention" is plural: the
same neighbourhood receives four visibly different distributions from the
four heads, and the entropy bar quantifies how peaked each one is.  All
coefficients come from a real masked-softmax GAT scoring pass.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import (set_pub_style, savefig, get_cmap, add_colorbar,
                      add_panel_label, annotate_box)
from figkit import halo_text

CLASS_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]     # house序位 1, 2, 3
ACCENT = "#b23a48"
INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#777777", "#333333"
BLUES = get_cmap("cb_sequential_blue")

SIZES = (11, 11, 12)
P_IN, P_OUT = 0.30, 0.030
N_HEADS, DIM = 4, 8
SEED = 5
LW_LO, LW_HI = 0.30, 2.50        # attention -> line width, per the spec
A_LO, A_HI = 0.15, 0.40          # attention -> edge opacity


def block_model(rng):
    labels = np.repeat(np.arange(len(SIZES)), SIZES)
    n = labels.size
    prob = np.where(labels[:, None] == labels[None, :], P_IN, P_OUT)
    adj = np.triu((rng.random((n, n)) < prob).astype(float), 1)
    adj = adj + adj.T
    for i in np.flatnonzero(adj.sum(1) == 0):
        peers = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
        j = rng.choice(peers)
        adj[i, j] = adj[j, i] = 1.0
    return adj, labels


def spring_layout(adj, rng, iters=520):
    """Fruchterman-Reingold in plain numpy (no networkx on this machine)."""
    n = adj.shape[0]
    pos = rng.normal(0, 0.6, (n, 2))
    k = 1.35 / np.sqrt(n)
    temp = 0.22
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        np.fill_diagonal(dist, 1e9)
        rep = (k ** 2 / dist ** 2)[:, :, None] * delta
        att = (dist / k * adj)[:, :, None] * delta
        disp = (rep - att).sum(axis=1) - 0.05 * pos          # mild centring
        step = np.linalg.norm(disp, axis=1, keepdims=True)
        pos += disp / np.maximum(step, 1e-9) * np.minimum(step, temp)
        temp *= 0.994
    pos -= pos.mean(0)
    return pos / np.abs(pos).max()


def gat_attention(adj, labels, rng):
    """Masked-softmax GAT scores; returns alpha[head, i, j] (row-stochastic)."""
    n = labels.size
    feats = np.eye(len(SIZES))[labels] @ rng.normal(0, 1.0, (len(SIZES), DIM))
    feats += rng.normal(0, 0.55, (n, DIM))
    alpha = np.zeros((N_HEADS, n, n))
    for h in range(N_HEADS):
        w = rng.normal(0, 0.7, (DIM, DIM))
        z = feats @ w
        a_src, a_dst = rng.normal(0, 0.9, DIM), rng.normal(0, 0.9, DIM)
        e = (z @ a_src)[:, None] + (z @ a_dst)[None, :]
        e = np.where(e > 0, e, 0.2 * e)                      # LeakyReLU
        # no self-loop, so every row is a distribution over N(i) alone and
        # the auxiliary heat map rows are directly comparable
        e = np.where(adj > 0, e, -np.inf)
        e = np.exp(e - e.max(axis=1, keepdims=True))
        alpha[h] = e / e.sum(axis=1, keepdims=True)
    return alpha


def draw_graph(ax, pos, edges, weight, labels, v, nbrs, node_size=44):
    lo, hi = weight.min(), weight.max()
    t = (weight - lo) / (hi - lo + 1e-12)
    order = np.argsort(t)
    for idx in order:
        i, j = edges[idx]
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=RULE, lw=LW_LO + (LW_HI - LW_LO) * t[idx],
                alpha=A_LO + (A_HI - A_LO) * t[idx], solid_capstyle="round",
                zorder=2)
    for k in range(len(CLASS_COLORS)):
        m = labels == k
        ax.scatter(pos[m, 0], pos[m, 1], s=node_size, c=CLASS_COLORS[k],
                   edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(pos[nbrs, 0], pos[nbrs, 1], s=node_size * 3.1, facecolors="none",
               edgecolors="#c9a227", linewidths=1.1, zorder=5)
    ax.scatter(pos[v, 0], pos[v, 1], s=node_size * 3.4, c=ACCENT,
               edgecolors="white", linewidths=1.0, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    adj, labels = block_model(rng)
    n = labels.size
    pos = spring_layout(adj, np.random.default_rng(SEED + 1))
    alpha = gat_attention(adj, labels, rng)

    eu, ev = np.triu_indices(n, 1)
    keep = adj[eu, ev] > 0
    edges = np.stack([eu[keep], ev[keep]], axis=1)
    mean_a = alpha.mean(0)
    w_edge = 0.5 * (mean_a[edges[:, 0], edges[:, 1]] +
                    mean_a[edges[:, 1], edges[:, 0]])

    v = int(np.argmax(adj.sum(1)))
    nbrs = np.flatnonzero(adj[v] > 0)
    order = np.argsort(-mean_a[v, nbrs])
    nbrs = nbrs[order]
    k = nbrs.size

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.20, hspace=0.42,
                          left=0.035, right=0.955, top=0.875, bottom=0.155)
    ax_g = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])

    # ---- main: the attended graph ----------------------------------------
    draw_graph(ax_g, pos, edges, w_edge, labels, v, nbrs)
    ax_g.set_title("Attention-weighted graph: width and opacity carry "
                   r"$\alpha_{ij}$", fontsize=9.0, fontweight="bold",
                   color=INK, pad=6)
    ax_g.text(pos[v, 0], pos[v, 1] - 0.075, "$v$", ha="center", va="top",
              fontsize=8.6, color=ACCENT, fontweight="bold", zorder=8)
    for rank, j in enumerate(nbrs):
        d = pos[j] - pos[v]
        d = d / (np.linalg.norm(d) + 1e-9)
        halo_text(ax_g, pos[j, 0] + 0.125 * d[0], pos[j, 1] + 0.125 * d[1],
                  f"$u_{rank + 1}$", ha="center", va="center", fontsize=6.8,
                  color="#8a6d16", halo="white", halo_lw=1.9, zorder=8)
    ax_g.set_xlim(-1.20, 1.20)
    ax_g.set_ylim(-1.16, 1.24)
    annotate_box(ax_g, f"|V| = {n},  |E| = {len(edges)}\n"
                       f"deg(v) = {k},  {N_HEADS} heads",
                 loc="upper left", fontsize=6.8)

    ax_g.add_patch(Rectangle((0.575, -0.015), 0.435, 0.415,
                             transform=ax_g.transAxes, facecolor="white",
                             edgecolor="none", zorder=6.5))
    axi = ax_g.inset_axes([0.635, 0.02, 0.345, 0.315])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    axi.barh(np.arange(k), mean_a[v, nbrs][::-1], height=0.68,
             color="#c9a227", edgecolor="white", linewidth=0.5)
    axi.set_yticks(np.arange(k))
    axi.set_yticklabels([f"$u_{i + 1}$" for i in range(k)][::-1], fontsize=6.0)
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_xlabel(r"$\alpha_{vu}$ (-)", fontsize=6.5, labelpad=1.5)
    axi.set_title(r"head-averaged $\mathcal{N}(v)$", fontsize=6.5,
                  color=SUB, pad=2.5)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axi.spines[s].set_linewidth(0.8)
        axi.spines[s].set_color(RULE)

    # ---- aux 1: per-head attention over N(v) -----------------------------
    mat = alpha[:, v, :][:, nbrs]
    im = ax_h.imshow(mat, cmap=BLUES, aspect="auto", vmin=0.0,
                     vmax=mat.max())
    ax_h.set_xticks(np.arange(k))
    ax_h.set_xticklabels([f"$u_{i + 1}$" for i in range(k)], fontsize=6.5)
    ax_h.set_yticks(np.arange(N_HEADS))
    ax_h.set_yticklabels([f"head {i + 1}" for i in range(N_HEADS)],
                         fontsize=6.5)
    ax_h.tick_params(length=2.0)
    ax_h.set_title("Heads disagree about the same neighbourhood",
                   fontsize=8.5, fontweight="bold", color=INK, pad=5)
    ax_h.set_xlabel(r"neighbour of $v$ (-)", fontsize=8.0, labelpad=2)
    for i in range(N_HEADS):
        for j in range(k):
            ax_h.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                      fontsize=6.0,
                      color="white" if mat[i, j] > 0.62 * mat.max() else INK)
    add_colorbar(fig, im, ax_h, label=r"$\alpha_{vu}$ (-)", aspect=13)

    # ---- aux 2: how peaked each head is ----------------------------------
    ent = -(mat * np.log(mat + 1e-12)).sum(1) / np.log(k)
    bars = ax_e.bar(np.arange(N_HEADS), ent, width=0.62, color="#3b5b92",
                    edgecolor="white", linewidth=0.6)
    sharp = int(np.argmin(ent))
    bars[sharp].set_color(ACCENT)
    ax_e.set_xticks(np.arange(N_HEADS))
    ax_e.set_xticklabels([f"h{i + 1}" for i in range(N_HEADS)], fontsize=7.0)
    ax_e.set_ylim(0, 1.06)
    ax_e.set_ylabel("normalised entropy (-)", fontsize=8.0)
    ax_e.axhline(1.0, color="#808080", lw=0.8, ls="--", zorder=1)
    ax_e.text(N_HEADS - 0.55, 1.0, "uniform", ha="right", va="bottom",
              fontsize=6.5, color=MUTED)
    ax_e.tick_params(labelsize=7.0, length=2.5)
    for s in ("top", "right"):
        ax_e.spines[s].set_visible(False)
    for i, e in enumerate(ent):
        ax_e.text(i, e + 0.03, f"{e:.2f}", ha="center", va="bottom",
                  fontsize=6.5, color=SUB)
    ax_e.set_title(f"Head {sharp + 1} is the most selective", fontsize=8.5,
                   fontweight="bold", color=INK, pad=5)

    for ax, lab in zip((ax_g, ax_h, ax_e), ("(a)", "(b)", "(c)")):
        add_panel_label(ax, lab, x=-0.06 if ax is ax_g else -0.20)

    handles = [Line2D([], [], marker="o", ls="", ms=4.6, mfc=c, mec="white",
                      mew=0.6, label=f"community {t}")
               for c, t in zip(CLASS_COLORS, "ABC")]
    handles += [Line2D([], [], marker="o", ls="", ms=6.4, mfc="none",
                       mec="#c9a227", mew=1.1, label=r"$\mathcal{N}(v)$"),
                Line2D([], [], color=RULE, lw=2.2, alpha=0.55,
                       label=r"high $\alpha_{ij}$"),
                Line2D([], [], color=RULE, lw=0.4, alpha=0.30,
                       label=r"low $\alpha_{ij}$")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=6, frameon=False,
               fontsize=7.2)
    savefig(fig, "446_gnn_attention_graph.png")


if __name__ == "__main__":
    main()
