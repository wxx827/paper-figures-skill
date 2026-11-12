"""Figure 467 - Cross-attention alignment between a source and target sentence.

Main panel: the source x target attention matrix, drawn on an inch canvas so
the token chips on both margins keep the batch-wide chip geometry instead of
being squeezed into tick labels.  Gold alignment links are ringed, which makes
the near-diagonal trend and the single adverb-verb swap visible at a glance.

Subordinate column: the attention entropy each target token spends over the
source, and the alignment error rate of four aligners.  Every number - the
entropies, the error rates, the swap - is computed from the same matrices that
are drawn, so re-seeding the script moves all of them together.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, lighten, darken
from figkit import inch_axes, panel_label_inches

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
GRID = "#d9d9d9"
ACCENT = "#b23a48"
BLUE = "#3b5b92"

MONO = {"fontfamily": "monospace"}
CHAR_W = 0.6022 / 72.0        # DejaVu Sans Mono advance, inches per pt
CHIP_H = 0.205                # batch-wide chip height, inches
CHIP_PAD = 0.058              # batch-wide horizontal padding, inches
CHIP_R = 0.045                # batch-wide corner radius, inches

SRC = ["der", "kleine", "braune", "Hund", "überquerte", "schnell", "die",
       "leere", "Straße", "."]
TGT = ["the", "small", "brown", "dog", "quickly", "crossed", "the", "empty",
       "street", "."]
GOLD = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]          # target index -> source index
SYSTEMS = [("IBM Model 4", 1.95, 0.055), ("fast_align", 1.45, 0.038),
           ("Transformer", 1.05, 0.024),
           ("Transformer + prior", 0.62, 0.012)]

CELL_W, CELL_H = 0.525, 0.345
CHIP_FS = 6.8


def chip_width(text, fs=7.0, pad=CHIP_PAD):
    """Exact chip width in inches for monospaced text of ``fs`` points."""
    return 2.0 * pad + len(text) * CHAR_W * fs


def chip(ax, x, y, text, fs=7.0, h=CHIP_H, fc="#f0f0f0", ec=HAIR, color=INK,
         weight="normal", pad=CHIP_PAD, lw=0.7, zorder=6):
    """Rounded token chip with its lower-left corner at ``(x, y)`` in inches."""
    w = chip_width(text, fs, pad)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CHIP_R}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        mutation_aspect=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, zorder=zorder + 2,
            **MONO)
    return w


def attention(sigma, noise, rng):
    """Column-stochastic attention peaked on the gold link."""
    n = len(SRC)
    i = np.arange(n)[:, None]
    peak = np.array(GOLD)[None, :]
    # function words spread their attention wider than content words
    spread = sigma * np.array([1.35, 1.0, 1.0, 0.85, 0.95, 0.85, 1.35, 1.0,
                               0.85, 1.15])[None, :]
    A = np.exp(-0.5 * ((i - peak) / spread) ** 2)
    A = A + noise * np.abs(rng.standard_normal(A.shape))
    return A / A.sum(axis=0, keepdims=True)


def entropy(A):
    """Attention entropy of every target token, in nats."""
    p = np.clip(A, 1e-12, None)
    return -(p * np.log(p)).sum(axis=0)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(467)
    n = len(SRC)

    mats = {name: attention(s, q, rng) for name, s, q in SYSTEMS}
    best = SYSTEMS[-1][0]
    A = mats[best]
    aer = {k: 1.0 - float(M[GOLD, np.arange(n)].mean())
           for k, M in mats.items()}
    H = entropy(A)
    cmap = get_cmap("cb_sequential_blue")
    vmax = float(A.max())

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) attention matrix on an inch canvas ---------------------------
    ax = inch_axes(fig, (0.26, 0.42, 6.70, 5.55))
    mx0 = 1.16
    mx1 = mx0 + n * CELL_W
    my1 = 4.42
    my0 = my1 - n * CELL_H

    im = ax.imshow(A, cmap=cmap, vmin=0.0, vmax=vmax, origin="upper",
                   extent=(mx0, mx1, my0, my1), interpolation="nearest",
                   zorder=3)
    for k in range(n + 1):
        ax.plot([mx0 + k * CELL_W] * 2, [my0, my1], color="white", lw=0.5,
                zorder=4)
        ax.plot([mx0, mx1], [my1 - k * CELL_H] * 2, color="white", lw=0.5,
                zorder=4)
    ax.add_patch(Rectangle((mx0, my0), mx1 - mx0, my1 - my0, fill=False,
                           edgecolor="#333333", linewidth=0.7, zorder=6))

    for j in range(n):
        for i in range(n):
            v = A[i, j]
            if v < 0.14:
                continue
            ax.text(mx0 + (j + 0.5) * CELL_W, my1 - (i + 0.5) * CELL_H,
                    f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                    color="white" if v > 0.55 * vmax else SUB, zorder=5,
                    **MONO)
    for j, i in enumerate(GOLD):
        ax.add_patch(Rectangle((mx0 + j * CELL_W, my1 - (i + 1) * CELL_H),
                               CELL_W, CELL_H, fill=False,
                               edgecolor=ACCENT, linewidth=1.15, zorder=7))

    # margin chips
    for j, t in enumerate(TGT):
        w = chip_width(t, CHIP_FS)
        chip(ax, mx0 + (j + 0.5) * CELL_W - w / 2.0, my1 + 0.10, t,
             fs=CHIP_FS, fc="#eef2f7", ec="#c6d2e0", color=darken(BLUE, 0.10))
    for i, s in enumerate(SRC):
        w = chip_width(s, CHIP_FS)
        chip(ax, mx0 - 0.13 - w, my1 - (i + 0.5) * CELL_H - CHIP_H / 2.0, s,
             fs=CHIP_FS, fc="#f4f0f5", ec="#ddd1e2",
             color=darken("#8e6bbf", 0.18))
    ax.text(mx0 - 0.13, my1 + 0.21, "source (de)", ha="right", va="center",
            fontsize=7.0, color=MUTED)
    ax.text(mx1, my1 + 0.42, "target (en)", ha="right", va="bottom",
            fontsize=7.0, color=MUTED)

    # swap call-out: the one pair whose gold link leaves the diagonal
    off = [j for j in range(n) if GOLD[j] != j]
    if off:
        j0, j1 = min(off), max(off)
        ax.annotate("", xy=(mx0 + (j0 + 0.5) * CELL_W, my0 - 0.10),
                    xytext=(mx0 + (j1 + 0.5) * CELL_W, my0 - 0.10),
                    arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text((2 * mx0 + (j0 + j1 + 1) * CELL_W) / 2.0, my0 - 0.18,
                f"reordered pair: {TGT[j0]} / {TGT[j1]}", ha="center",
                va="top", fontsize=6.6, color=ACCENT)

    # in-canvas colour bar
    cbx0, cbx1 = mx0, mx0 + 2.60
    cby0 = my0 - 0.66
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap,
              extent=(cbx0, cbx1, cby0, cby0 + 0.13), zorder=3)
    ax.add_patch(Rectangle((cbx0, cby0), cbx1 - cbx0, 0.13, fill=False,
                           edgecolor="#333333", linewidth=0.6, zorder=4))
    for f in (0.0, 0.5, 1.0):
        ax.plot([cbx0 + f * (cbx1 - cbx0)] * 2, [cby0, cby0 - 0.045],
                color="#333333", lw=0.7, zorder=4)
        ax.text(cbx0 + f * (cbx1 - cbx0), cby0 - 0.075, f"{f * vmax:.2f}",
                ha="center", va="top", fontsize=6.4, color=SUB, **MONO)
    ax.text(cbx1 + 0.14, cby0 + 0.065, "attention weight (-)", ha="left",
            va="center", fontsize=7.0, color=INK)
    ax.add_patch(Rectangle((cbx1 + 1.70, cby0 + 0.005), 0.16, 0.12,
                           fill=False, edgecolor=ACCENT, linewidth=1.15,
                           zorder=4))
    ax.text(cbx1 + 1.94, cby0 + 0.065, "gold link", ha="left", va="center",
            fontsize=7.0, color=SUB)

    ax.text(mx0 - 0.13, my1 + 0.62, f"Cross-attention weights, {best}",
            ha="left", va="bottom", fontsize=9.0, fontweight="bold",
            color=INK)

    # ---- (b) attention entropy per target token ---------------------------
    axb = fig.add_axes([0.745, 0.585, 0.225, 0.320])
    cols = [ACCENT if GOLD[j] != j else BLUE for j in range(n)]
    axb.bar(np.arange(n), H, width=0.68, color=cols, edgecolor="white",
            linewidth=0.4, zorder=3)
    axb.axhline(H.mean(), color="#808080", lw=0.8, ls=(0, (3.2, 1.8)),
                zorder=4)
    axb.set_ylim(0.0, H.max() * 1.24)
    axb.text(-0.55, H.max() * 1.21, f"mean {H.mean():.2f} nats", ha="left",
             va="top", fontsize=6.2, color=MUTED)
    axb.set_xticks(np.arange(n))
    axb.set_xticklabels(TGT, fontsize=6.2, rotation=42, ha="right", **MONO)
    axb.set_xlim(-0.7, n - 0.3)
    axb.set_ylabel("entropy (nats)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axb.set_title("Entropy per target token", fontsize=8.5,
                  fontweight="bold", pad=3.0, loc="left")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- (c) alignment error rate -----------------------------------------
    axc = fig.add_axes([0.745, 0.115, 0.225, 0.290])
    names = [s[0] for s in SYSTEMS]
    vals = [aer[k] for k in names]
    cols = ["#9fb0c8"] * (len(names) - 1) + [ACCENT]
    axc.barh(np.arange(len(names))[::-1], vals, height=0.62, color=cols,
             edgecolor="white", linewidth=0.4, zorder=3)
    for y, v in zip(np.arange(len(names))[::-1], vals):
        axc.text(v + 0.008, y, f"{v:.3f}", va="center", ha="left",
                 fontsize=6.2, color=SUB, **MONO)
    axc.set_yticks(np.arange(len(names))[::-1])
    axc.set_yticklabels(names, fontsize=6.5)
    axc.set_xlim(0, max(vals) * 1.28)
    axc.set_xlabel("alignment error rate (-)", fontsize=7.0)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axc.set_title("1 $-$ mass on the gold link", fontsize=8.5,
                  fontweight="bold", pad=3.0, loc="left")
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    drop = 1.0 - aer[names[-1]] / aer[names[0]]
    panel_label_inches(fig, ax, "(a)", dx=-0.10, dy=-0.10)
    panel_label_inches(fig, axb, "(b)", dx=-0.52, dy=0.08)
    panel_label_inches(fig, axc, "(c)", dx=-0.52, dy=0.08)
    savefig(fig, "467_nlp_attention_alignment.png")


if __name__ == "__main__":
    main()
