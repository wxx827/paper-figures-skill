"""A heterogeneous academic graph: three node types, three relation types.

Node type is carried twice over - by shape (circle / square / triangle) and by
colour - so the distinction survives being shrunk to a thumbnail, where a
colour-only encoding would not.  Relation type is carried by line style, which
is the only channel left once shape and colour are spent.

The layout is force-directed but with a per-type vertical bias, so authors
settle at the bottom, papers in the middle and venues on top: the bipartite
structure of the schema becomes visible without forcing a rigid layered
drawing.  The right panel walks the two metapaths a heterogeneous GNN would
sample along, and the bottom row counts what is actually in the graph.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label, annotate_box

TYPES = ["Author", "Paper", "Venue"]
TYPE_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]      # house序位 1, 2, 3
TYPE_MARKERS = ["o", "s", "^"]
TYPE_SIZES = [52, 46, 70]
TYPE_Y = [-0.62, 0.05, 0.72]
X_STRETCH = 1.75          # the slot is much wider than tall; fill it

RELATIONS = ["writes", "cites", "published in"]
REL_STYLES = ["-", (0, (3.6, 2.2)), (0, (5.0, 1.6, 1.0, 1.6))]
REL_PAIRS = [(0, 1), (1, 1), (1, 2)]

INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#777777", "#333333"
N_TYPE = (18, 14, 4)
SEED = 41


def build(rng):
    types = np.repeat(np.arange(3), N_TYPE)
    n = types.size
    author = np.flatnonzero(types == 0)
    paper = np.flatnonzero(types == 1)
    venue = np.flatnonzero(types == 2)

    edges = {r: set() for r in range(3)}
    for p in paper:                                   # 1-3 authors per paper
        for a in rng.choice(author, rng.integers(1, 4), replace=False):
            edges[0].add((int(a), int(p)))
        edges[2].add((int(p), int(rng.choice(venue))))
    written = {a for a, _ in edges[0]}
    for a in author:                                  # nobody floats free
        if int(a) not in written:
            edges[0].add((int(a), int(rng.choice(paper))))
    for p in paper:                                   # a few citations
        if rng.random() < 0.75:
            for q in rng.choice(paper, rng.integers(1, 3), replace=False):
                if q != p:
                    edges[1].add((int(p), int(q)))
    return types, edges


def layout(types, edges, rng, iters=700):
    n = types.size
    adj = np.zeros((n, n))
    for es in edges.values():
        for a, b in es:
            adj[a, b] = adj[b, a] = 1.0
    pos = rng.normal(0, 0.4, (n, 2))
    pos[:, 1] += np.array(TYPE_Y)[types]
    k, temp = 1.7 / np.sqrt(n), 0.16
    target = np.array(TYPE_Y)[types]
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        np.fill_diagonal(dist, 1e9)
        disp = ((k ** 2 / dist ** 2)[:, :, None] * delta
                - (dist / k * adj)[:, :, None] * delta)
        disp = disp.sum(1)
        disp[:, 0] -= 0.05 * pos[:, 0]
        disp[:, 1] += 4.2 * (target - pos[:, 1])       # keep the type bands
        step = np.linalg.norm(disp, axis=1, keepdims=True)
        pos += disp / np.maximum(step, 1e-9) * np.minimum(step, temp)
        temp *= 0.996
    pos[:, 0] -= pos[:, 0].mean()
    pos[:, 0] *= X_STRETCH / np.abs(pos[:, 0]).max()
    pos[:, 1] /= np.abs(pos[:, 1]).max()
    return pos


def scatter_types(ax, pos, types, scale=1.0, lw=0.6, zorder=5):
    for t in range(3):
        m = types == t
        ax.scatter(pos[m, 0], pos[m, 1], marker=TYPE_MARKERS[t],
                   s=TYPE_SIZES[t] * scale, c=TYPE_COLORS[t],
                   edgecolors="white", linewidths=lw, zorder=zorder)


def glyph(ax, x, y, t, size=1.0):
    ax.scatter([x], [y], marker=TYPE_MARKERS[t], s=TYPE_SIZES[t] * size,
               c=TYPE_COLORS[t], edgecolors="white", linewidths=0.7, zorder=4)


def metapath_panel(ax, edges):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    paths = [([0, 1, 2], [0, 2], "where an author publishes"),
             ([0, 1, 0], [0, 0], "co-authorship in two hops")]
    for row, (ts, rels, name) in enumerate(paths):
        y = 0.70 - 0.40 * row
        xs = [0.17, 0.50, 0.83]
        for x, t in zip(xs, ts):
            glyph(ax, x, y, t, size=2.8)
            ax.text(x, y - 0.075, TYPES[t], ha="center", va="top",
                    fontsize=7.0, color=TYPE_COLORS[t], fontweight="bold")
        for i, r in enumerate(rels):
            ax.annotate("", xy=(xs[i + 1] - 0.085, y),
                        xytext=(xs[i] + 0.085, y),
                        arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0,
                                        linestyle=REL_STYLES[r],
                                        shrinkA=0, shrinkB=0))
            ax.text((xs[i] + xs[i + 1]) / 2, y + 0.035, RELATIONS[r],
                    ha="center", va="bottom", fontsize=6.4, color=MUTED)
        ax.text(0.5, y + 0.175, name, ha="center", va="bottom", fontsize=7.4,
                color=INK, fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    types, edges = build(rng)
    pos = layout(types, edges, np.random.default_rng(SEED + 1))
    n = types.size
    n_e = sum(len(v) for v in edges.values())

    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.55], wspace=0.24, hspace=0.40,
                          left=0.055, right=0.965, top=0.855, bottom=0.135)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    # ---- main: the heterogeneous graph ------------------------------------
    for r in (1, 0, 2):
        for a, b in edges[r]:
            ax_g.plot(pos[[a, b], 0], pos[[a, b], 1], color=RULE, lw=0.75,
                      ls=REL_STYLES[r], alpha=0.32, zorder=2,
                      solid_capstyle="round")
    scatter_types(ax_g, pos, types)
    ax_g.set_aspect("equal")
    ax_g.axis("off")
    ax_g.set_xlim(-X_STRETCH - 0.30, X_STRETCH + 0.30)
    ax_g.set_ylim(-1.22, 1.30)
    for t in range(3):
        ax_g.text(-X_STRETCH - 0.26, TYPE_Y[t] / max(np.abs(TYPE_Y)),
                  TYPES[t] + "s", ha="left", va="center", fontsize=7.2,
                  color=TYPE_COLORS[t], fontweight="bold", alpha=0.75)
    ax_g.set_title("Three node types, three relation types, one graph",
                   fontsize=9.0, fontweight="bold", color=INK, pad=6)
    annotate_box(ax_g, f"|V| = {n},  |E| = {n_e}\n3 node types, "
                       f"{len(RELATIONS)} relations",
                 loc="upper right", fontsize=6.8)

    # ---- aux 1: metapaths --------------------------------------------------
    metapath_panel(ax_m, edges)
    ax_m.set_title("Metapaths a hetero-GNN samples along", fontsize=8.5,
                   fontweight="bold", color=INK, pad=6)

    # ---- aux 2 / 3: what the graph is made of -----------------------------
    counts = [int((types == t).sum()) for t in range(3)]
    ax_n.bar(np.arange(3), counts, width=0.58, color=TYPE_COLORS,
             edgecolor="white", linewidth=0.6)
    for i, c in enumerate(counts):
        ax_n.text(i, c + 0.4, str(c), ha="center", va="bottom", fontsize=7.0,
                  color=SUB)
    ax_n.set_xticks(np.arange(3))
    ax_n.set_xticklabels(TYPES, fontsize=7.5)
    ax_n.set_ylabel("nodes (count)", fontsize=8.0)
    ax_n.set_ylim(0, max(counts) * 1.30)
    ax_n.set_title(f"Node inventory  ·  {n} nodes total", fontsize=8.5,
                   fontweight="bold", color=INK, pad=5)

    e_counts = [len(edges[r]) for r in range(3)]
    ax_e.bar(np.arange(3), e_counts, width=0.58, color="#8a8a8a",
             edgecolor="white", linewidth=0.6)
    for i, c in enumerate(e_counts):
        ax_e.plot([i - 0.20, i + 0.20], [c + max(e_counts) * 0.055] * 2,
                  ls=REL_STYLES[i], color=RULE, lw=1.2,
                  solid_capstyle="butt", clip_on=False)
        ax_e.text(i, c + max(e_counts) * 0.115, str(c), ha="center",
                  va="bottom", fontsize=7.0, color=SUB)
    ax_e.set_xticks(np.arange(3))
    ax_e.set_xticklabels(RELATIONS, fontsize=7.5)
    ax_e.set_ylabel("edges (count)", fontsize=8.0)
    ax_e.set_ylim(0, max(e_counts) * 1.34)
    ax_e.set_title(f"Relation inventory  ·  {n_e} edges total", fontsize=8.5,
                   fontweight="bold", color=INK, pad=5)

    for ax in (ax_n, ax_e):
        ax.tick_params(labelsize=7.0, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax, lab in zip((ax_g, ax_m, ax_n, ax_e), ("(a)", "(b)", "(c)", "(d)")):
        add_panel_label(ax, lab, x=-0.05 if ax is ax_g else -0.14)

    handles = [Line2D([], [], marker=TYPE_MARKERS[t], ls="", ms=5.4,
                      mfc=TYPE_COLORS[t], mec="white", mew=0.6, label=TYPES[t])
               for t in range(3)]
    handles += [Line2D([], [], color=RULE, lw=1.1, ls=REL_STYLES[r],
                       label=f"{RELATIONS[r]}  "
                             f"({TYPES[REL_PAIRS[r][0]][0]}→"
                             f"{TYPES[REL_PAIRS[r][1]][0]})")
                for r in range(3)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=6, frameon=False,
               fontsize=7.2)
    savefig(fig, "449_gnn_heterogeneous.png")


if __name__ == "__main__":
    main()
