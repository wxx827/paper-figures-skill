"""Figure 313 - Canonical attention-head patterns in a causal language model.

Six heads of one decoder layer are probed with the same ten-token prompt.
Every panel is a query (row) x key (column) matrix under the causal mask, so
the upper triangle is structurally empty rather than merely zero.  Rows are
normalised to sum to one and all six panels share a single colour scale, which
is what makes the qualitative differences between heads comparable at a
glance: sharp positional heads, delimiter sinks, an induction head that copies
across a repeated token, and a diffuse context head.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pubstyle import set_pub_style, savefig, add_panel_label

TOKENS = ["<s>", "The", "cat", "sat", "on", "the", "mat", ".", "It", "purred"]
N = len(TOKENS)

CMAP = "viridis"
VMAX = 0.90
MASK_COLOR = "#efefef"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"
ACCENT = "#b23a48"

# canvas layout, in inches
FIG_W, FIG_H = 10.4, 6.4
LEFT, RIGHT = 0.62, 1.00
TOP, BOTTOM = 0.60, 0.78
ROW_GAP = 0.55


def _rows(mat, floor=1e-3):
    """Causal-mask a raw score matrix and row-normalise it."""
    m = np.tril(np.asarray(mat, dtype=float) + floor)
    m = m / m.sum(axis=1, keepdims=True)
    return np.where(np.tril(np.ones((N, N), bool)), m, np.nan)


def build_heads():
    """Six qualitatively distinct, hand-designed head signatures."""
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    d = i - j

    self_head = np.where(d == 0, 9.0, 0.04)
    prev_head = np.where(d == 1, 9.0, 0.03) + np.where(d == 0, 0.45, 0.0)

    # sink head: mass parked on <s> and on the sentence-final full stop
    sink = np.zeros((N, N))
    sink[:, 0] = 6.0
    sink[8:, 7] = 3.2
    sink = sink + 0.10 + 0.25 * np.exp(-((d - 1) ** 2) / 2.0)

    # induction head: "the" (5) copies the token that followed "The" (1)
    ind = 0.06 + 0.10 * np.exp(-((d - 1) ** 2) / 3.0)
    ind[5, 2] = 4.2
    ind[8, 2] = 3.4          # "It" resolves back to "cat"
    ind[9, 3] = 2.6          # "purred" attends to "sat"
    ind[:, 0] = 0.55

    # broad context head: slowly decaying, near-uniform over the prefix
    broad = np.exp(-np.clip(d, 0, None) / 6.5) + 0.30

    # syntactic head: verb -> subject, preposition -> object, sparse elsewhere
    syn = np.full((N, N), 0.05)
    syn[:, 0] = 0.30
    for q, k in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 3), (8, 2), (9, 8)):
        syn[q, k] = 5.0
    syn[6, 2] = 1.6
    syn[9, 2] = 1.4

    return [
        ("Self / diagonal", self_head),
        ("Previous token", prev_head),
        ("Delimiter sink", sink),
        ("Induction (copy)", ind),
        ("Broad context", broad),
        ("Syntactic (sparse)", syn),
    ]


def entropy_bits(mat):
    """Mean row entropy over the causal support, in bits."""
    out = []
    for r in range(1, N):
        p = mat[r, : r + 1]
        p = p[p > 0]
        out.append(-(p * np.log2(p)).sum())
    return float(np.mean(out))


def panel_grid():
    """Return the six axes rectangles (figure fractions) plus the cbar rect."""
    side = (FIG_H - TOP - BOTTOM - ROW_GAP) / 2.0
    col_gap = (FIG_W - LEFT - RIGHT - 3 * side) / 2.0
    rects = []
    for r in range(2):
        for c in range(3):
            x0 = LEFT + c * (side + col_gap)
            y0 = FIG_H - TOP - side - r * (side + ROW_GAP)
            rects.append([x0 / FIG_W, y0 / FIG_H, side / FIG_W, side / FIG_H])
    x_cb = LEFT + 2 * (side + col_gap) + side + 0.16
    cbar = [x_cb / FIG_W, BOTTOM / FIG_H, 0.11 / FIG_W,
            (2 * side + ROW_GAP) / FIG_H]
    return rects, cbar


def main():
    set_pub_style(base_font=9)
    heads = build_heads()
    mats = [_rows(m) for _, m in heads]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    rects, cbar_rect = panel_grid()
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(MASK_COLOR)
    norm = Normalize(vmin=0.0, vmax=VMAX)

    for k, ((name, _), mat) in enumerate(zip(heads, mats)):
        ax = fig.add_axes(rects[k])
        ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest",
                  extent=[-0.5, N - 0.5, N - 0.5, -0.5])
        for g in np.arange(0.5, N - 1, 1.0):
            ax.axhline(g, color="white", lw=0.35, alpha=0.55)
            ax.axvline(g, color="white", lw=0.35, alpha=0.55)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_color("#333333")

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(TOKENS, rotation=90, fontsize=6.4,
                           fontfamily="monospace")
        ax.set_yticklabels(TOKENS, fontsize=6.4, fontfamily="monospace")
        ax.tick_params(length=2.0, width=0.7, pad=1.6)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(N - 0.5, -0.5)

        ax.set_title(f"Head {k + 1}/16  ·  {name}", fontsize=8.5,
                     fontweight="bold", pad=4.0)
        add_panel_label(ax, f"({'abcdef'[k]})", x=-0.20, y=1.05)

        ax.text(0.97, 0.94, f"$H$ = {entropy_bits(mat):.2f} bits",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
                color=INK, bbox=dict(boxstyle="round,pad=0.22",
                                     facecolor="white", edgecolor="#c4c4c4",
                                     linewidth=0.5, alpha=0.92))
        if k >= 3:
            ax.set_xlabel("key token", fontsize=8.0, labelpad=2.0)
        if k % 3 == 0:
            ax.set_ylabel("query token", fontsize=8.0, labelpad=2.0)

    cax = fig.add_axes(cbar_rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      ticks=[0.0, 0.3, 0.6, 0.9])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("attention weight (-)", fontsize=8.0, color=INK, labelpad=3.0)
    cax.text(0.5, 1.012, "rows sum to 1", transform=cax.transAxes,
             ha="center", va="bottom", fontsize=6.5, color=FAINT)


    savefig(fig, "313_llm_attention_pattern.png")


if __name__ == "__main__":
    main()
