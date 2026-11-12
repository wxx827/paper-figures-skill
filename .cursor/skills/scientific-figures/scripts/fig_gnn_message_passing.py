"""Message passing in a GNN layer, drawn as three stages on one small graph.

The same nine-node graph is redrawn three times.  Left, the neighbourhood
N(v) lights up and curved messages converge on the centre node.  Middle,
only v stays lit while its feature vector is pushed through the update MLP
together with the aggregated message.  Right, every node feeds a graph-level
readout.  A strip of feature-vector chips under each stage shows the tensors
that actually move, and the bottom card carries the three equations aligned
under the stage they belong to.

All chip values come from one real forward pass (mean aggregation, tanh
update, mean readout) so the colours and the equations agree.
"""
import numpy as np
from matplotlib.patches import Circle, Rectangle
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import inch_canvas, block, arrow, panel_card, op_circle

# ---- house palette ---------------------------------------------------------
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

CENTER = "#b23a48"      # the node being updated  (house red, sole accent)
NEIGH = "#c9a227"       # N(v)                    (house gold)
FAR = "#3b5b92"         # everything else         (house blue)
DIM = "#c3c8d0"         # faded-out nodes
FEAT_CMAP = get_cmap("cb_sequential_blue")

W, H = 10.4, 6.4
NR = 0.105              # node radius, inches
CX = [1.82, 5.20, 8.58]  # stage centres
CARD_Y, CARD_H, CARD_W = 3.90, 3.44, 2.94
GY = 4.12               # graph centre height
STRIP_Y = 2.72          # tensor strip height
R1, R2 = 0.52, 0.80     # neighbour / second-hop radii


# ---- the toy graph ---------------------------------------------------------
def build_graph():
    """Nine nodes: v, its four neighbours, four second-hop nodes."""
    pos = [(0.0, 0.0)]                                       # 0 = v
    for a in (62, 134, 206, 318):                            # 1..4 = N(v)
        pos.append((R1 * np.cos(np.radians(a)), R1 * np.sin(np.radians(a))))
    for a in (20, 92, 170, 250):                             # 5..8 = 2-hop
        pos.append((R2 * np.cos(np.radians(a)), R2 * np.sin(np.radians(a))))
    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (3, 4),
             (1, 5), (2, 6), (3, 7), (4, 8), (5, 6), (7, 8)]
    return np.array(pos), edges


def forward_pass(n_nodes, d=5, seed=7):
    """One honest message-passing step; every chip in the figure reads this."""
    rng = np.random.default_rng(seed)
    h_prev = rng.random((n_nodes, d))
    m_v = h_prev[[1, 2, 3, 4]].mean(axis=0)                  # mean over N(v)
    w = rng.normal(0, 0.9, (d, 2 * d))
    h_new = 0.5 * (1.0 + np.tanh(w @ np.concatenate([h_prev[0], m_v])))
    h_all = h_prev.copy()
    h_all[0] = h_new
    return h_prev, m_v, h_new, h_all.mean(axis=0)


# ---- primitives ------------------------------------------------------------
def node(ax, x, y, color, r=NR, glow=None, glow_a=0.22, z=6):
    if glow:
        ax.add_patch(Circle((x, y), r * 1.82, facecolor=glow, alpha=glow_a * 0.55,
                            edgecolor="none", zorder=z - 2))
        ax.add_patch(Circle((x, y), r * 1.36, facecolor=glow, alpha=glow_a,
                            edgecolor="none", zorder=z - 1))
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                        linewidth=0.6, zorder=z))


def chips(ax, x, y, vals, cell=0.098, h=0.16, z=5):
    """A feature vector as a row of coloured cells, centred on (x, y)."""
    n = len(vals)
    x0 = x - n * cell / 2.0
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + i * cell, y - h / 2), cell, h,
                               facecolor=FEAT_CMAP(float(v)), edgecolor="white",
                               linewidth=0.45, zorder=z))
    ax.add_patch(Rectangle((x0, y - h / 2), n * cell, h, facecolor="none",
                           edgecolor="#8a8a8a", linewidth=0.6, zorder=z + 1))
    return dict(l=(x0, y), r=(x0 + n * cell, y), t=(x, y + h / 2),
                b=(x, y - h / 2))


def draw_edges(ax, pos, edges, cx, keep, color="#8a8a8a", lw=0.9, alpha=0.32,
               dim_alpha=0.12):
    for u, v in edges:
        on = (u, v) in keep or (v, u) in keep
        ax.plot([cx + pos[u, 0], cx + pos[v, 0]],
                [GY + pos[u, 1], GY + pos[v, 1]],
                color=color if on else "#b9bec6", zorder=4 if on else 3,
                lw=lw if on else 0.7, alpha=alpha if on else dim_alpha,
                solid_capstyle="round")


def stage_frame(ax, cx, title, note):
    panel_card(ax, cx, CARD_Y, CARD_W, CARD_H, fc="white", ec="#dcdfe4",
               lw=0.9, rounding=0.09, zorder=1)
    ax.text(cx, CARD_Y + CARD_H / 2 - 0.20, title, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=INK, zorder=4)
    ax.text(cx, CARD_Y + CARD_H / 2 - 0.42, note, ha="center", va="center",
            fontsize=6.8, color=MUTED, zorder=4)


def corner_note(ax, cx, text, color):
    ax.text(cx - CARD_W / 2 + 0.14, GY + 0.86, text, ha="left", va="center",
            fontsize=8.0, color=color, fontweight="bold", zorder=9)


# ---- stage 1: aggregate ----------------------------------------------------
def stage_aggregate(ax, pos, edges, h_prev, m_v):
    cx = CX[0]
    stage_frame(ax, cx, "1 · Aggregate",
                r"pool messages over $\mathcal{N}(v)$")
    draw_edges(ax, pos, edges, cx, {(0, 1), (0, 2), (0, 3), (0, 4)},
               color=NEIGH, lw=1.0, alpha=0.50)

    for i in range(5, 9):
        node(ax, cx + pos[i, 0], GY + pos[i, 1], DIM, r=NR * 0.85)
    for i in (1, 2, 3, 4):
        node(ax, cx + pos[i, 0], GY + pos[i, 1], NEIGH, glow=NEIGH,
             glow_a=0.30)
    node(ax, cx + pos[0, 0], GY + pos[0, 1], CENTER, r=NR * 1.20)

    for i, rad in zip((1, 2, 3, 4), (0.26, -0.26, 0.26, -0.26)):
        arrow(ax, (cx + pos[i, 0], GY + pos[i, 1]), (cx, GY), color=NEIGH,
              rad=rad, lw=1.15, ms=8.5, gap_a=0.17, gap_b=0.17, zorder=8)

    corner_note(ax, cx, r"$\mathcal{N}(v)$", NEIGH)
    ax.text(cx + 0.17, GY - 0.15, r"$v$", ha="left", va="top", fontsize=8.2,
            color=CENTER, fontweight="bold", zorder=9)

    for k, i in enumerate((1, 2, 3, 4)):
        row = chips(ax, cx - 0.94, STRIP_Y + 0.27 - 0.165 * k, h_prev[i],
                    cell=0.066, h=0.11)
        arrow(ax, row["r"], (cx - 0.31, STRIP_Y - 0.02), color="#9aa3ae",
              lw=0.7, ms=6.0, gap_a=0.04, gap_b=0.02)
    ax.text(cx - 0.94, STRIP_Y - 0.40, r"$h_u^{(l-1)},\ u\in\mathcal{N}(v)$",
            ha="center", va="center", fontsize=6.6, color=SUB)
    op_circle(ax, cx - 0.16, STRIP_Y - 0.02, r"$\Sigma$", r=0.135,
              fontsize=8.0)
    m = chips(ax, cx + 0.76, STRIP_Y - 0.02, m_v)
    arrow(ax, (cx - 0.01, STRIP_Y - 0.02), (m["l"][0], STRIP_Y - 0.02),
          color=RULE, lw=0.9, ms=8, gap_a=0.03, gap_b=0.03)
    ax.text(cx + 0.76, STRIP_Y - 0.40, r"message  $m_v^{(l)}$", ha="center",
            va="center", fontsize=6.6, color=SUB)


# ---- stage 2: update -------------------------------------------------------
def stage_update(ax, pos, edges, h_prev, m_v, h_new):
    cx = CX[1]
    stage_frame(ax, cx, "2 · Update",
                r"combine $h_v^{(l-1)}$ with $m_v^{(l)}$")
    draw_edges(ax, pos, edges, cx, set(), alpha=0.10)

    for i in range(1, 9):
        node(ax, cx + pos[i, 0], GY + pos[i, 1], DIM, r=NR * 0.85)
    node(ax, cx + pos[0, 0], GY + pos[0, 1], CENTER, r=NR * 1.42, glow=CENTER,
         glow_a=0.26)
    ax.text(cx + 0.24, GY - 0.14, r"$h_v^{(l-1)}\!\rightarrow\!h_v^{(l)}$",
            ha="left", va="top", fontsize=7.4, color=CENTER, zorder=9)
    corner_note(ax, cx, r"only $v$ changes", CENTER)

    a = chips(ax, cx - 1.00, STRIP_Y + 0.17, h_prev[0], cell=0.080, h=0.13)
    b = chips(ax, cx - 1.00, STRIP_Y - 0.19, m_v, cell=0.080, h=0.13)
    ax.text(cx - 1.00, STRIP_Y + 0.31, r"$h_v^{(l-1)}$", ha="center",
            va="bottom", fontsize=6.6, color=SUB)
    ax.text(cx - 1.00, STRIP_Y - 0.33, r"$m_v^{(l)}$", ha="center", va="top",
            fontsize=6.6, color=SUB)
    mlp = block(ax, cx + 0.06, STRIP_Y - 0.01, 0.46, 0.60,
                "$W^{(l)}$\n$\\sigma$", color=FAR, fontsize=7.2)
    for anc in (a, b):
        arrow(ax, anc["r"], (mlp["l"][0], STRIP_Y - 0.01), color=RULE,
              lw=0.9, ms=8, gap_a=0.05, gap_b=0.03)
    c = chips(ax, cx + 0.94, STRIP_Y - 0.01, h_new)
    arrow(ax, mlp["r"], (c["l"][0], STRIP_Y - 0.01), color=RULE, lw=0.9,
          ms=8, gap_a=0.03, gap_b=0.03)
    ax.text(cx + 0.94, STRIP_Y - 0.40, r"updated  $h_v^{(l)}$", ha="center",
            va="center", fontsize=6.6, color=SUB)


# ---- stage 3: readout ------------------------------------------------------
def stage_readout(ax, pos, edges, h_graph):
    cx = CX[2]
    stage_frame(ax, cx, "3 · Readout", r"pool all nodes into $h_G$")
    draw_edges(ax, pos, edges, cx, {tuple(e) for e in edges}, alpha=0.34)

    for i in range(1, 9):
        node(ax, cx + pos[i, 0], GY + pos[i, 1], FAR, glow=FAR, glow_a=0.16)
    node(ax, cx + pos[0, 0], GY + pos[0, 1], CENTER, r=NR * 1.20, glow=CENTER,
         glow_a=0.20)
    corner_note(ax, cx, r"every $v \in \mathcal{V}$", FAR)

    pool = block(ax, cx - 0.72, STRIP_Y - 0.01, 0.80, 0.44, "mean\npool",
                 color=FAR, fontsize=7.0)
    for i in range(9):
        arrow(ax, (cx + pos[i, 0], GY + pos[i, 1]),
              (pool["x"], pool["t"][1]), color="#9aa3ae", lw=0.6, ms=5.5,
              gap_a=0.15, gap_b=0.02, zorder=2, alpha=0.60)
    g = chips(ax, cx + 0.74, STRIP_Y - 0.01, h_graph)
    arrow(ax, pool["r"], (g["l"][0], STRIP_Y - 0.01), color=RULE, lw=0.9,
          ms=8, gap_a=0.03, gap_b=0.03)
    ax.text(cx + 0.74, STRIP_Y - 0.40, r"graph vector  $h_G$", ha="center",
            va="center", fontsize=6.6, color=SUB)


# ---- bottom equation card --------------------------------------------------
def equations(ax, n_nodes, n_edges, d):
    panel_card(ax, W / 2.0, 1.14, 9.70, 1.32, fc=SOFT, ec=HAIR, lw=0.8,
               rounding=0.08, zorder=1)
    ax.text(0.62, 1.68, "One layer, three equations", ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=INK, zorder=4)
    ax.text(9.78, 1.68, f"|V| = {n_nodes},   |E| = {n_edges},   d = {d}",
            ha="right", va="center", fontsize=7.0, color=MUTED, zorder=4)
    forms = [
        r"$m_v^{(l)}=\frac{1}{|\mathcal{N}(v)|}\sum_{u\in\mathcal{N}(v)}"
        r"\phi\left(h_u^{(l-1)}\right)$",
        r"$h_v^{(l)}=\sigma\!\left(W^{(l)}\left[\,h_v^{(l-1)}\;\|\;"
        r"m_v^{(l)}\,\right]\right)$",
        r"$h_G=\mathrm{READOUT}\left(\{h_v^{(L)}\}_{v\in\mathcal{V}}\right)$",
    ]
    caps = ["aggregate over the neighbourhood",
            "update the centre node",
            "pool to a graph embedding"]
    for cx, f, c in zip(CX, forms, caps):
        ax.text(cx, 1.24, f, ha="center", va="center", fontsize=8.6,
                color=INK, zorder=4)
        ax.text(cx, 0.80, c, ha="center", va="center", fontsize=6.8,
                color=SUB, zorder=4)
    for x in (3.51, 6.89):
        ax.plot([x, x], [0.64, 1.62], color="#dcdfe4", lw=0.8, zorder=3)


def main():
    set_pub_style()
    pos, edges = build_graph()
    h_prev, m_v, h_new, h_graph = forward_pass(len(pos))

    fig, ax = inch_canvas(W, H)
    ax.text(0.34, 6.10,
            "A GNN layer moves information one hop: aggregate, update, "
            "read out", fontsize=11.0, fontweight="bold", color=INK,
            ha="left", va="center")
    ax.text(0.34, 5.83,
            "The same graph at each stage; highlighted elements are the ones "
            "the stage touches, chips below are the tensors it moves",
            fontsize=7.6, color=SUB, ha="left", va="center")

    stage_aggregate(ax, pos, edges, h_prev, m_v)
    stage_update(ax, pos, edges, h_prev, m_v, h_new)
    stage_readout(ax, pos, edges, h_graph)

    for x in (3.36, 6.74):
        arrow(ax, (x, CARD_Y), (x + 0.32, CARD_Y), color=RULE, lw=1.2, ms=11)

    equations(ax, len(pos), len(edges), h_prev.shape[1])
    savefig(fig, "444_gnn_message_passing.png")


if __name__ == "__main__":
    main()
