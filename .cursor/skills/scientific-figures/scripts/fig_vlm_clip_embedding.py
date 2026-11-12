"""Figure 347 - Joint image-text embedding space after contrastive training.

Main panel: a t-SNE view of the shared space where image embeddings are
circles, caption embeddings are triangles, colour encodes the semantic class
and a hairline joins every ground-truth pair.  The two subordinate panels
(58 % of the main height) carry the paired-distance distribution before and
after alignment, and retrieval recall at K.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, PALETTES
from fig_vlm_image_text_align import panel_tag, C_INK, C_SUB, C_FAINT

CLASSES = ["beach", "street", "park", "kitchen", "living room"]
CLASS_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf", "#c0603f"]
CENTERS = np.array([[-5.2, 6.6], [5.0, 7.4], [-5.6, -4.4], [5.4, -5.2],
                    [0.0, 0.6]])
LABEL_OFF = [(0.0, 3.7), (0.0, 3.7), (0.0, -4.0), (0.0, -4.0), (4.9, 0.0)]
ACCENT = "#b23a48"
BASE = "#808080"
N_PER = 16


def embed(rng):
    """Paired image / text embeddings with a small residual modality gap."""
    imgs, txts = [], []
    for c, ctr in enumerate(CENTERS):
        spread = 1.55 if c < 4 else 1.9
        p = ctr + rng.normal(0, spread, size=(N_PER, 2))
        gap = np.array([0.62, 0.48])
        t = p + gap + rng.normal(0, 0.52, size=(N_PER, 2))
        imgs.append(p)
        txts.append(t)
    return imgs, txts


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(17)
    imgs, txts = embed(rng)

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(3, 3, width_ratios=[0.52, 1.25, 0.52],
                          height_ratios=[0.30, 1.0, 0.30],
                          left=0.055, right=0.975, top=0.900, bottom=0.085,
                          wspace=0.32, hspace=0.0)
    axm = fig.add_subplot(gs[:, 1])
    axl = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[1, 2])

    # ---- main: joint embedding -------------------------------------------
    for c, (p, t) in enumerate(zip(imgs, txts)):
        col = CLASS_COLORS[c]
        for (px, py), (tx, ty) in zip(p, t):
            axm.plot([px, tx], [py, ty], color=col, lw=0.45, alpha=0.42,
                     zorder=1)
        axm.scatter(p[:, 0], p[:, 1], s=26, marker="o", facecolor=col,
                    edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3)
        axm.scatter(t[:, 0], t[:, 1], s=30, marker="^", facecolor="white",
                    edgecolor=col, linewidth=1.0, alpha=0.95, zorder=3)
        ctr = np.vstack([p, t]).mean(axis=0)
        dx, dy = LABEL_OFF[c]
        axm.text(ctr[0] + dx, ctr[1] + dy, CLASSES[c],
                 ha="left" if dx else "center",
                 va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                 fontsize=7.5, fontweight="bold", color=col, zorder=5,
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                           edgecolor=col, linewidth=0.6, alpha=0.9))
    axm.set_xlim(-11, 11)
    axm.set_ylim(-12.4, 12.4)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_xlabel("t-SNE dimension 1 (-)", fontsize=8.0)
    axm.set_ylabel("t-SNE dimension 2 (-)", fontsize=8.0)
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#b8b8b8")
    axm.set_title("Shared embedding space, 5 semantic classes", fontsize=9.0,
                  fontweight="bold", pad=4.0)
    handles = [Line2D([], [], marker="o", ls="none", mfc="#5a5a5a",
                      mec="white", mew=0.5, ms=5, label="image embedding"),
               Line2D([], [], marker="^", ls="none", mfc="white",
                      mec="#5a5a5a", mew=1.0, ms=5, label="caption embedding"),
               Line2D([], [], color="#8a8a8a", lw=0.8,
                      label="ground-truth pair")]
    axm.legend(handles=handles, loc="lower left", frameon=True,
               framealpha=0.92, facecolor="white", edgecolor="#cccccc",
               fontsize=7.0, borderpad=0.4, handletextpad=0.5)

    # ---- subordinate: paired-distance distribution ------------------------
    d_before = rng.gamma(7.0, 0.30, 900) + 0.6
    d_after = rng.gamma(4.2, 0.19, 900) + 0.15
    bins = np.linspace(0, 5.0, 34)
    axl.hist(d_before, bins=bins, color=BASE, alpha=0.42, edgecolor=BASE,
             linewidth=0.6, label="before alignment")
    axl.hist(d_after, bins=bins, color=ACCENT, alpha=0.55, edgecolor=ACCENT,
             linewidth=0.6, label="after alignment")
    axl.axvline(d_before.mean(), color=BASE, lw=0.9, ls=(0, (3, 2)))
    axl.axvline(d_after.mean(), color=ACCENT, lw=0.9, ls=(0, (3, 2)))
    axl.set_xlabel("image-caption distance (-)", fontsize=7.0)
    axl.set_ylabel("pairs (count)", fontsize=7.0)
    axl.tick_params(labelsize=6.5, length=2.2, width=0.6)
    axl.legend(frameon=False, fontsize=6.5, loc="upper right",
               handlelength=1.1, borderaxespad=0.2)
    axl.set_title("Modality gap shrinks", fontsize=8.5, fontweight="bold",
                  pad=3.5)
    axl.spines["left"].set_linewidth(0.6)
    axl.spines["bottom"].set_linewidth(0.6)

    # ---- subordinate: recall at K ----------------------------------------
    ks = ["R@1", "R@5", "R@10"]
    base_v = [41.2, 68.5, 78.9]
    ours_v = [52.7, 79.4, 87.6]
    x = np.arange(3)
    axr.bar(x - 0.19, base_v, width=0.36, color=BASE, edgecolor="white",
            linewidth=0.5, label="CLIP baseline")
    axr.bar(x + 0.19, ours_v, width=0.36, color=ACCENT, edgecolor="white",
            linewidth=0.5, label="Ours")
    for xi, (b, o) in enumerate(zip(base_v, ours_v)):
        axr.text(xi - 0.19, b + 1.6, f"{b:.1f}", ha="center", va="bottom",
                 fontsize=6.2, color=C_SUB)
        axr.text(xi + 0.19, o + 1.6, f"{o:.1f}", ha="center", va="bottom",
                 fontsize=6.2, color=ACCENT, fontweight="bold")
    axr.set_xticks(x)
    axr.set_xticklabels(ks, fontsize=7.0)
    axr.set_ylim(0, 100)
    axr.set_ylabel("recall (%)", fontsize=7.0)
    axr.tick_params(labelsize=6.5, length=2.2, width=0.6)
    axr.legend(frameon=False, fontsize=6.5, loc="upper left",
               handlelength=1.1, borderaxespad=0.2)
    axr.set_title("Text $\\rightarrow$ image recall", fontsize=8.5,
                  fontweight="bold", pad=3.5)
    axr.spines["left"].set_linewidth(0.6)
    axr.spines["bottom"].set_linewidth(0.6)

    for a, lab in ((axl, "(a)"), (axm, "(b)"), (axr, "(c)")):
        panel_tag(fig, a, lab)
    savefig(fig, "347_vlm_clip_embedding.png")


if __name__ == "__main__":
    main()
