"""Explanation subgraphs from three methods, on three graphs with a planted motif.

Every row is one sample and every column keeps that row's layout, so the four
tiles differ only in which edges are called important - the comparison is
meaningless otherwise.  The leftmost column is the input with the planted
house motif marked in dashed dark as the answer key; the three method columns
fade the graph back and pick out their own top-6 edges in gold.

Fidelity is the drop in the detector's output when the explanation subgraph is
deleted, computed per tile from that tile's own mask, so a method that selects
the wrong edges scores low no matter how confident its mask looks.  The
proposed column is the only emphasised object in the figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import background_card

NODE_C = "#3b5b92"          # house序位 2
PICK_C = "#c9a227"          # house序位 4, the selected subgraph
TRUTH_C = "#333333"
ACCENT = "#b23a48"          # reserved for the proposed column
INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"

METHODS = ["GNNExplainer", "PGExplainer", "Ours"]
MASK_NOISE = [0.95, 0.62, 0.27]      # how ragged each method's mask is
N_BASE, N_ROWS = 14, 3
MOTIF_E = 6                          # house motif: 4-cycle plus a roof
DETECT_W, DETECT_T = 1.7, 4.6
SEED = 84


def make_sample(rng):
    """A random base graph with a house motif welded on; returns motif edges."""
    n = N_BASE + 5
    edges = {(int(rng.integers(0, i)), i) for i in range(1, N_BASE)}
    for _ in range(4):                                  # a little extra mesh
        a, b = sorted(rng.choice(N_BASE, 2, replace=False))
        if a != b:
            edges.add((int(a), int(b)))
    a, b, c, d, e = range(N_BASE, N_BASE + 5)
    motif = {(a, b), (b, c), (c, d), (a, d), (a, e), (b, e)}
    motif = {tuple(sorted(p)) for p in motif}
    edges |= motif
    for anchor in rng.choice(N_BASE, 2, replace=False):  # tether to the base
        edges.add(tuple(sorted((int(anchor), int(rng.choice([a, c, d]))))))
    adj = np.zeros((n, n))
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1.0
    return adj, sorted(edges), motif


def spring_layout(adj, rng, iters=420):
    """Fruchterman-Reingold in plain numpy (no networkx on this machine)."""
    n = adj.shape[0]
    pos = rng.normal(0, 0.5, (n, 2))
    k, temp = 1.8 / np.sqrt(n), 0.28
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        np.fill_diagonal(dist, 1e9)
        disp = ((k ** 2 / dist ** 2)[:, :, None] * delta
                - (dist / k * adj)[:, :, None] * delta).sum(1) - 0.05 * pos
        step = np.linalg.norm(disp, axis=1, keepdims=True)
        pos += disp / np.maximum(step, 1e-9) * np.minimum(step, temp)
        temp *= 0.993
    pos -= pos.mean(0)
    return pos / np.abs(pos).max()


def detector(n_motif_present):
    """Surrogate motif detector; the thing fidelity is measured against."""
    return 1.0 / (1.0 + np.exp(-(DETECT_W * n_motif_present - DETECT_T)))


def explain(edges, motif, noise, rng):
    """A noisy edge mask, then its top-|motif| edges as the explanation."""
    mask = np.array([1.0 if e in motif else 0.0 for e in edges])
    mask = mask + rng.normal(0, noise, mask.size)
    pick = set(np.argsort(-mask)[:MOTIF_E].tolist())
    chosen = {edges[i] for i in pick}
    kept = len(chosen & motif)
    fidelity = detector(MOTIF_E) - detector(MOTIF_E - kept)
    return chosen, fidelity, kept


def draw(ax, pos, edges, motif, chosen=None, tint=False):
    if tint:
        ax.set_facecolor("#fdf6f7")
    for u, v in edges:
        on = chosen is not None and (u, v) in chosen
        key = chosen is None and (u, v) in motif
        if on:
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=PICK_C, lw=2.4,
                    alpha=0.95, zorder=3, solid_capstyle="round")
        elif key:
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=TRUTH_C, lw=1.4,
                    ls=(0, (3.0, 1.9)), alpha=0.9, zorder=3)
        else:
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color="#8a8a8a", lw=0.7,
                    alpha=0.35 if chosen is None else 0.14, zorder=1,
                    solid_capstyle="round")
    ax.scatter(pos[:, 0], pos[:, 1], s=22, c=NODE_C, edgecolors="white",
               linewidths=0.6, zorder=5,
               alpha=1.0 if chosen is None else 0.75)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(HAIR)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(9.6, 6.8))
    left, right, top, bottom = 0.075, 0.975, 0.845, 0.115
    gs = fig.add_gridspec(N_ROWS, 4, wspace=0.10, hspace=0.30, left=left,
                          right=right, top=top, bottom=bottom)
    col_w = (right - left) / 4.0
    background_card(fig, (left + 3 * col_w - 0.012, bottom - 0.040,
                          col_w + 0.014, top - bottom + 0.098),
                    facecolor="#fbeff1", zorder=-5, radius=0.010)

    # everything is scored first, so the column headers can carry the means
    samples, fid = [], np.zeros((N_ROWS, len(METHODS)))
    for r in range(N_ROWS):
        adj, edges, motif = make_sample(rng)
        pos = spring_layout(adj, np.random.default_rng(SEED + 40 + r))
        picks = [explain(edges, motif, noise,
                         np.random.default_rng(SEED + 400 + 10 * r + c))
                 for c, noise in enumerate(MASK_NOISE)]
        fid[r] = [p[1] for p in picks]
        samples.append((adj, edges, motif, pos, picks))
    means = fid.mean(0)

    for r, (adj, edges, motif, pos, picks) in enumerate(samples):
        ax0 = fig.add_subplot(gs[r, 0])
        draw(ax0, pos, edges, motif)
        ax0.set_ylabel(f"sample {r + 1}", fontsize=8.0, color=INK,
                       labelpad=4)
        ax0.text(0.5, -0.075, f"|V| = {adj.shape[0]},  |E| = {len(edges)},  "
                              f"motif = {len(motif)} edges",
                 transform=ax0.transAxes, ha="center", va="top",
                 fontsize=6.4, color=MUTED)
        for c, name in enumerate(METHODS):
            ax = fig.add_subplot(gs[r, c + 1])
            chosen, f, kept = picks[c]
            draw(ax, pos, edges, motif, chosen=chosen, tint=(c == 2))
            ax.text(0.5, -0.075,
                    f"fidelity {f:.2f}   ·   {kept}/{MOTIF_E} motif edges",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.4, color=ACCENT if c == 2 else SUB,
                    fontweight="bold" if c == 2 else "normal")
            if r == 0:
                ax.set_title(name, fontsize=8.8, fontweight="bold",
                             color=ACCENT if c == 2 else INK, pad=5)
        if r == 0:
            ax0.set_title("Input graph", fontsize=8.8, fontweight="bold",
                          color=INK, pad=5)

    means = fid.mean(0)
    fig.text(left + col_w * 0.5, bottom - 0.082,
             "dashed = planted motif", ha="center", va="top", fontsize=6.8,
             color=TRUTH_C)
    for c, m in enumerate(means):
        fig.text(left + col_w * (1.5 + c), bottom - 0.082,
                 f"mean fidelity {m:.2f}", ha="center", va="top",
                 fontsize=7.2, color=ACCENT if c == 2 else SUB,
                 fontweight="bold" if c == 2 else "normal")

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.2, mfc=NODE_C, mec="white",
               mew=0.6, label="node"),
        Line2D([], [], color="#8a8a8a", lw=1.0, alpha=0.5, label="other edge"),
        Line2D([], [], color=TRUTH_C, lw=1.4, ls=(0, (3.0, 1.9)),
               label="planted motif (ground truth)"),
        Line2D([], [], color=PICK_C, lw=2.4,
               label=f"top-{MOTIF_E} explanation edges"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.004), ncol=4, frameon=False,
               fontsize=7.2)
    savefig(fig, "452_gnn_explainability.png")


if __name__ == "__main__":
    main()
