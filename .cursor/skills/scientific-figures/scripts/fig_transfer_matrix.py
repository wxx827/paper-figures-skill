"""Task-to-task transfer matrix with clustered ordering and marginals.

Cell (i, j) is the accuracy change on target task j when the encoder is
pre-trained on source task i, relative to training j from scratch.  Rows and
columns follow a single hierarchical-clustering order so related tasks sit
next to each other; the diagonal is the same-task reference and is greyed
out.  The two marginal bar strips are intentionally small - they are a
summary of the matrix, not a second result.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from pubstyle import set_pub_style, savefig, add_colorbar

OURS = "#3b5b92"

TASKS = ["Classification", "Captioning", "VQA", "Detection",
         "Segmentation", "Depth", "Surface normals", "Keypoints"]
GROUP = [0, 0, 0, 1, 1, 2, 2, 2]
VLIM = 10.0


def transfer_matrix():
    rng = np.random.default_rng(312)
    n = len(TASKS)
    src = rng.uniform(-1.0, 1.0, n)
    tgt = rng.uniform(-1.0, 1.0, n)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(GROUP[i] - GROUP[j])
            aff = {0: 7.2, 1: 1.4, 2: -3.8}[d]
            M[i, j] = aff + 1.7 * src[i] + 1.1 * tgt[j] + rng.normal(0, 0.7)
    np.fill_diagonal(M, np.nan)
    return M


def cluster_order(M):
    """One shared order for rows and columns, from the symmetrised matrix."""
    F = np.where(np.isnan(M), np.nanmean(M), M)
    S = 0.5 * (F + F.T)
    D = S.max() - S
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return list(leaves_list(linkage(squareform(D, checks=False),
                                    method="average")))


def main():
    set_pub_style(base_font=8)
    M = transfer_matrix()
    order = cluster_order(M)
    M = M[np.ix_(order, order)]
    labels = [TASKS[i] for i in order]
    print("  clustered order:", " > ".join(labels))

    n = len(labels)
    col_mean = np.nanmean(M, axis=0)     # gain when used as a target
    row_mean = np.nanmean(M, axis=1)     # gain when used as a source

    fig = plt.figure(figsize=(7.2, 4.85))
    axm = fig.add_axes([0.170, 0.170, 0.495, 0.590])
    axt = fig.add_axes([0.170, 0.782, 0.495, 0.098])
    axr = fig.add_axes([0.682, 0.170, 0.080, 0.590])
    axc = fig.add_axes([0.878, 0.245, 0.019, 0.440])

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e2e2e2")

    im = axm.imshow(M, cmap=cmap, vmin=-VLIM, vmax=VLIM, aspect="auto",
                    interpolation="nearest")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                axm.text(j, i, "\u2013", ha="center", va="center",
                         fontsize=7.0, color="#8c8c8c")
                continue
            axm.text(j, i, f"{v:+.1f}", ha="center", va="center",
                     fontsize=6.4,
                     color="white" if abs(v) > 5.6 else "#2b2b2b")
    axm.set_xticks(range(n))
    axm.set_yticks(range(n))
    axm.set_xticklabels(labels, fontsize=7.0, rotation=32, ha="right",
                        rotation_mode="anchor")
    axm.set_yticklabels(labels, fontsize=7.0)
    axm.tick_params(length=0, pad=2)
    axm.set_xlabel("Target task", fontsize=8.4, labelpad=4)
    axm.set_ylabel("Source task (pre-training)", fontsize=8.4, labelpad=4)
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    # cell separators
    for k in range(1, n):
        axm.axhline(k - 0.5, color="white", lw=0.7)
        axm.axvline(k - 0.5, color="white", lw=0.7)

    # ---- subordinate marginals --------------------------------------------
    for ax, vals, horiz in ((axt, col_mean, True), (axr, row_mean, False)):
        cols = [cmap((v + VLIM) / (2 * VLIM)) for v in vals]
        if horiz:
            ax.bar(range(n), vals, width=0.68, color=cols, edgecolor="#ffffff",
                   lw=0.5, zorder=3)
            ax.axhline(0, color="#999999", lw=0.7, zorder=4)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(min(vals.min() * 1.35, -1.2), vals.max() * 1.35)
            ax.set_xticks([])
            ax.set_yticks([0, np.floor(vals.max())])
        else:
            ax.barh(range(n), vals, height=0.68, color=cols,
                    edgecolor="#ffffff", lw=0.5, zorder=3)
            ax.axvline(0, color="#999999", lw=0.7, zorder=4)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_xlim(min(vals.min() * 1.35, -1.2), vals.max() * 1.35)
            ax.set_yticks([])
            ax.set_xticks([0, np.floor(vals.max())])
        ax.tick_params(labelsize=6.0, length=1.8, colors="#777777")
        for side in ("top", "right", "left" if horiz else "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom" if horiz else "left"].set_color("#bbbbbb")
        ax.spines["bottom" if horiz else "left"].set_linewidth(0.6)
        ax.set_facecolor("#f5f6f8")

    axt.text(-0.5, axt.get_ylim()[1] * 1.06, "mean gain as target (%)",
             fontsize=6.6, color="#666666", ha="left", va="bottom")
    axr.text(axr.get_xlim()[1] * 1.02, -0.75, "as source",
             fontsize=6.6, color="#666666", ha="right", va="bottom")

    cb = fig.colorbar(im, cax=axc, ticks=[-10, -5, 0, 5, 10])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("accuracy change vs. training from scratch (pp)",
                 fontsize=7.0, color="#1a1a1a")
    axc.text(0.5, 1.045, "positive\ntransfer", transform=axc.transAxes,
             ha="center", va="bottom", fontsize=6.2, color="#b2182b",
             linespacing=1.25)
    axc.text(0.5, -0.045, "negative\ntransfer", transform=axc.transAxes,
             ha="center", va="top", fontsize=6.2, color="#2166ac",
             linespacing=1.25)

    best = np.unravel_index(np.nanargmax(M), M.shape)
    axm.add_patch(Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                            edgecolor="#1a1a1a", lw=1.5, zorder=6))


    print(f"  best pair {labels[best[0]]} -> {labels[best[1]]}: "
          f"{M[best]:+.2f}")
    print(f"  worst cell {np.nanmin(M):+.2f}, best cell {np.nanmax(M):+.2f}")
    savefig(fig, "312_transfer_matrix.png")


if __name__ == "__main__":
    main()
