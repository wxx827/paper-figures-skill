"""Cell-cell communication: chord diagram plus a sender x receiver bubble map.

The chord tells the reader who talks to whom and how loudly (ribbon width =
interaction strength, ribbon colour = sender).  The bubble map unfolds the
same matrix so individual pairs can be read off exactly, and the bottom panel
ranks the ligand-receptor pathways that carry the signal.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar, shared_norm

CELL_TYPES = ["T cell", "B cell", "NK cell", "Monocyte",
              "Macrophage", "Dendritic", "Fibroblast", "Endothelial"]
CELL_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227",
               "#8e6bbf", "#2a7f7a", "#c0603f", "#8fa6c2"]
PATHWAYS = ["COLLAGEN", "CXCL", "TGFb", "MIF", "VEGF", "IL1"]

FW, FH = 10.4, 6.2
R_IN, R_OUT = 1.0, 1.075          # chord rim radii


def synth_strength(rng):
    """Sender x receiver interaction strength; myeloid and stroma dominate."""
    n = len(CELL_TYPES)
    send = np.array([0.55, 0.40, 0.45, 0.95, 1.00, 0.75, 1.05, 0.70])
    recv = np.array([1.00, 0.55, 0.60, 0.80, 0.90, 0.65, 0.85, 0.75])
    S = np.outer(send, recv) * rng.uniform(0.35, 1.35, (n, n))
    S[np.arange(n), np.arange(n)] *= 0.35          # autocrine is weaker
    S[S < np.percentile(S, 67)] = 0.0              # keep the readable top set
    return S


def ribbon(ax, a0, a1, b0, b1, color, alpha=0.55, n=26):
    """Bezier band between two rim arcs, pulled through the circle centre."""
    def arc(t0, t1):
        a = np.linspace(t0, t1, n)
        return np.column_stack([R_IN * np.cos(a), R_IN * np.sin(a)])

    src, dst = arc(a0, a1), arc(b1, b0)
    verts = ([src[0]] + list(src[1:]) + [(0, 0), (0, 0), dst[0]]
             + list(dst[1:]) + [(0, 0), (0, 0), src[0]])
    codes = ([Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CURVE4] * 3
             + [Path.LINETO] * (n - 1) + [Path.CURVE4] * 3)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=alpha,
                           edgecolor="white", lw=0.35, zorder=2))


def draw_chord(ax, S):
    n = len(CELL_TYPES)
    total = S.sum(1) + S.sum(0)
    gap = np.deg2rad(2.6)
    span = (2 * np.pi - n * gap) * total / total.sum()
    start = np.deg2rad(95.0) + np.cumsum(np.r_[0.0, span[:-1] + gap])

    for i, (a0, w) in enumerate(zip(start, span)):
        ax.add_patch(Wedge((0, 0), R_OUT, np.rad2deg(a0),
                           np.rad2deg(a0 + w), width=R_OUT - R_IN,
                           facecolor=CELL_COLORS[i], edgecolor="white",
                           lw=0.5, zorder=4))
        mid = a0 + w / 2
        deg = np.rad2deg(mid) % 360
        right = deg < 90 or deg > 270
        ax.text(1.12 * np.cos(mid), 1.12 * np.sin(mid), CELL_TYPES[i],
                rotation=deg if right else deg - 180,
                rotation_mode="anchor", ha="left" if right else "right",
                va="center", fontsize=7.0, color="#1a1a1a")

    # sub-arcs: outgoing first, then incoming, both in partner order
    cursor = start.copy()
    out_at, in_at = {}, {}
    for i in range(n):
        for j in range(n):
            if S[i, j] > 0:
                w = span[i] * S[i, j] / total[i]
                out_at[(i, j)] = (cursor[i], cursor[i] + w)
                cursor[i] += w
    for j in range(n):
        for i in range(n):
            if S[i, j] > 0:
                w = span[j] * S[i, j] / total[j]
                in_at[(i, j)] = (cursor[j], cursor[j] + w)
                cursor[j] += w
    order = sorted(out_at, key=lambda k: S[k])      # weak ribbons underneath
    for i, j in order:
        a0, a1 = out_at[(i, j)]
        b0, b1 = in_at[(i, j)]
        ribbon(ax, a0, a1, b0, b1, CELL_COLORS[i],
               alpha=0.30 + 0.42 * S[i, j] / S.max())
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(531)
    S = synth_strength(rng)
    n = len(CELL_TYPES)
    pairs = np.argwhere(S > 0)
    # significance of each surviving pair, and its pathway decomposition
    fdr = 10.0 ** (-(1.3 + 3.4 * S[S > 0] / S.max() + rng.uniform(0, 0.8,
                                                                  len(pairs))))
    weights = rng.dirichlet(np.array([2.4, 1.6, 1.5, 1.2, 1.0, 0.9]),
                            len(pairs))
    path_strength = (weights * S[S > 0][:, None]).sum(0)

    fig = plt.figure(figsize=(FW, FH))
    top = FH - 0.95
    ax_c = axes_inches(fig, (0.30, 0.70 + (top - 0.70 - 4.20) / 2, 4.20, 4.20))
    ax_b = axes_inches(fig, (5.55, 2.72, 3.40, 2.16))
    ax_p = axes_inches(fig, (5.55, 0.68, 3.40, 1.12))

    # ---- (a) chord ----------------------------------------------------------
    draw_chord(ax_c, S)
    ax_c.text(0.5, -0.055, f"{len(pairs)} significant sender-receiver pairs, "
                           f"ribbon width = interaction strength",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=7.0,
              color="#555555")

    # ---- (b) sender x receiver bubbles --------------------------------------
    norm = shared_norm([S[S > 0]])
    sizes = 6.0 + 26.0 * (-np.log10(fdr))
    sc = ax_b.scatter(pairs[:, 0], pairs[:, 1], s=sizes, c=S[S > 0],
                      cmap=get_cmap("cb_sequential_warm"), norm=norm,
                      edgecolors="#3a3a3a", linewidths=0.3, zorder=3)
    ax_b.set_xticks(range(n))
    ax_b.set_yticks(range(n))
    ax_b.set_xticklabels(CELL_TYPES, rotation=45, ha="right", fontsize=6.5)
    ax_b.set_yticklabels(CELL_TYPES, fontsize=6.5)
    for tick, col in zip(ax_b.get_xticklabels(), CELL_COLORS):
        tick.set_color(col)
    for tick, col in zip(ax_b.get_yticklabels(), CELL_COLORS):
        tick.set_color(col)
    ax_b.set_xlim(-0.6, n - 0.4)
    ax_b.set_ylim(-0.6, n - 0.4)
    ax_b.set_xlabel("sender", fontsize=8.0, labelpad=1.0)
    ax_b.set_ylabel("receiver", fontsize=8.0)
    ax_b.grid(True, color="#d5d5d5", lw=0.5)
    ax_b.set_axisbelow(True)
    ax_b.tick_params(length=2.0)
    ax_b.set_title("Pairwise interaction strength", fontsize=8.5,
                   fontweight="bold", pad=26.0)
    keys = [2.0, 3.5, 5.0]
    h_sz = [Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor="#d5d5d5", markeredgecolor="#3a3a3a",
                   markeredgewidth=0.3,
                   markersize=np.sqrt(6.0 + 26.0 * k), label=f"{k:.1f}")
            for k in keys]
    ax_b.legend(handles=h_sz, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                ncol=3, frameon=False, fontsize=6.8,
                title="-log10 FDR", title_fontsize=6.8, handletextpad=0.4,
                columnspacing=1.1)
    safe_colorbar(fig, sc, ax_b, label="interaction strength (a.u.)",
                  width=0.11, pad=0.10, shrink=0.92, labelsize=6.8)

    # ---- (c) top pathways ---------------------------------------------------
    idx = np.argsort(path_strength)[::-1][:5]
    ypos = np.arange(len(idx))[::-1]
    ax_p.barh(ypos, path_strength[idx], height=0.62, color="#3b5b92",
              edgecolor="none")
    for y, k in zip(ypos, idx):
        ax_p.text(path_strength[k] * 1.02, y,
                  f"{100 * path_strength[k] / path_strength.sum():.0f}%",
                  ha="left", va="center", fontsize=6.5, color="#555555")
    ax_p.set_yticks(ypos)
    ax_p.set_yticklabels([PATHWAYS[k] for k in idx], fontsize=6.8)
    ax_p.set_xlim(0, path_strength.max() * 1.16)
    ax_p.set_xlabel("aggregate strength (a.u.)", fontsize=7.5)
    ax_p.tick_params(labelsize=6.8, length=2.0)
    ax_p.set_title("Top signalling pathways", fontsize=8.0, fontweight="bold")

    panel_labels_inches(fig, [ax_c, ax_b, ax_p], dx=-0.22, dy=0.08)
    top_pair = pairs[np.argmax(S[S > 0])]
    fig.text(0.30 / FW, 1.0 - 0.30 / FH,
             "Myeloid and stromal senders dominate the interaction network",
             ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(0.30 / FW, 1.0 - 0.56 / FH,
             f"{len(pairs)} pairs pass FDR < 0.05; strongest link "
             f"{CELL_TYPES[top_pair[0]]} to {CELL_TYPES[top_pair[1]]}, "
             f"{PATHWAYS[idx[0]]} carries "
             f"{100 * path_strength[idx[0]] / path_strength.sum():.0f}% of the "
             f"aggregate signal",
             ha="left", va="center", fontsize=7.5, color="#555555")
    savefig(fig, "531_sc_cell_communication.png")


if __name__ == "__main__":
    main()
