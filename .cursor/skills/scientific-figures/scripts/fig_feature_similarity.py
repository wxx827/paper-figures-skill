"""Figure 280 - Representation similarity (linear CKA).

Panel (a) is the layer-by-layer CKA of one ResNet-50 with itself: the four
residual stages show up as bright diagonal blocks.  Panel (b) compares those
layers against the twelve blocks of a ViT-B/16.  Panel (c) tracks similarity
along the depth axis.  All three read off the same colour scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes

SIM = "magma"
NR = 20                      # ResNet-50 read-out points
NV = 12                      # ViT-B/16 blocks
STAGE_EDGES = [5, 10, 15]
STAGE_NAMES = ["stage 1", "stage 2", "stage 3", "stage 4"]


def resnet_cka(seed=3):
    """Distance decay plus a stage-wise block structure, kept symmetric."""
    rng = np.random.default_rng(seed)
    i = np.arange(NR)
    d = np.abs(i[:, None] - i[None, :])
    m = 0.20 + 0.68 * np.exp(-d / 3.4)

    stage = np.digitize(i, STAGE_EDGES)
    same = stage[:, None] == stage[None, :]
    m = m + 0.22 * same * np.exp(-d / 5.0)
    # Neighbouring stages stay related; the first and last barely are.
    m = m - 0.16 * (np.abs(stage[:, None] - stage[None, :]) >= 2)
    noise = gaussian_filter(rng.standard_normal((NR, NR)), 1.1)
    m = m + 0.035 * (noise + noise.T) / 2.0
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    return np.clip(m, 0.0, 1.0)


def cross_cka(seed=9):
    """ViT blocks (rows) against ResNet layers (columns)."""
    rng = np.random.default_rng(seed)
    vi = np.arange(NV) / (NV - 1.0)
    ri = np.arange(NR) / (NR - 1.0)
    # Depth correspondence is not linear: ViT mixes globally much earlier.
    warp = ri ** 1.45
    d = np.abs(vi[:, None] - warp[None, :])
    m = 0.14 + 0.62 * np.exp(-(d / 0.22) ** 2)
    m += 0.22 * np.exp(-((vi[:, None] + warp[None, :]) / 0.34) ** 2)
    m += 0.10 * (vi[:, None] > 0.75) * (warp[None, :] > 0.75)
    m += 0.030 * gaussian_filter(rng.standard_normal((NV, NR)), 1.0)
    return np.clip(m, 0.0, 1.0)


def depth_curves(seed=15):
    rng = np.random.default_rng(seed)
    x = np.arange(1, NR + 1)
    seed_pair = 0.97 - 0.30 * ((x - 1) / (NR - 1)) ** 1.7
    seed_pair += 0.012 * rng.standard_normal(NR)
    cross = 0.82 - 0.47 * ((x - 1) / (NR - 1)) ** 0.9
    cross += 0.018 * rng.standard_normal(NR)
    return x, np.clip(seed_pair, 0, 1), np.clip(cross, 0, 1)


def matrix_axes(ax):
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#bcbcbc")
    ax.tick_params(length=1.8, width=0.5, color="#999999", labelsize=6.2)


def main():
    set_pub_style(base_font=8)
    A = resnet_cka()
    C = cross_cka()
    x, seed_pair, cross = depth_curves()
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(SIM)

    fig = plt.figure(figsize=(7.2, 4.30))
    a_left, a_w = 0.075, 2.90 / 7.2
    a_bot, a_h = 0.155, 2.90 / 4.30
    a_top = a_bot + a_h
    b_left, b_w = 0.545, 0.340
    b_h = (b_w * 7.2) * (NV / NR) / 4.30
    b_bot = a_top - b_h

    axa = fig.add_axes([a_left, a_bot, a_w, a_h])
    matrix_axes(axa)
    axa.imshow(A, cmap=SIM, norm=norm, interpolation="nearest")
    for e in STAGE_EDGES:
        axa.axhline(e - 0.5, color="white", lw=0.8)
        axa.axvline(e - 0.5, color="white", lw=0.8)
    ticks = [0, 4, 9, 14, 19]
    axa.set_xticks(ticks)
    axa.set_yticks(ticks)
    axa.set_xticklabels([f"L{t + 1}" for t in ticks])
    axa.set_yticklabels([f"L{t + 1}" for t in ticks])
    axa.set_xlabel("ResNet-50 layer", fontsize=7.2, labelpad=2.5)
    axa.set_ylabel("ResNet-50 layer", fontsize=7.2, labelpad=2.5)
    centres = [2, 7, 12, 17]
    for cen, nm in zip(centres, STAGE_NAMES):
        axa.text(cen, -1.35, nm, ha="center", va="bottom", fontsize=6.2,
                 color="#7a7a7a")

    axb = fig.add_axes([b_left, b_bot, b_w, b_h])
    matrix_axes(axb)
    axb.imshow(C, cmap=SIM, norm=norm, interpolation="nearest")
    axb.set_xticks(ticks)
    axb.set_xticklabels([f"L{t + 1}" for t in ticks])
    axb.set_yticks([0, 3, 7, 11])
    axb.set_yticklabels(["B1", "B4", "B8", "B12"])
    axb.set_xlabel("ResNet-50 layer", fontsize=7.2, labelpad=2.5)
    axb.set_ylabel("ViT-B/16 block", fontsize=7.2, labelpad=2.5)

    axc = fig.add_axes([b_left, a_bot, b_w, 0.210])
    style_axes(axc)
    axc.plot(x, seed_pair, color="#4a4a4a", lw=1.2, zorder=2,
             marker="o", markersize=3.2, markerfacecolor="#d8d8d8",
             label="same architecture, different seed")
    axc.plot(x, cross, color="#4a4a4a", lw=1.2, ls="--", zorder=2,
             marker="s", markersize=3.0, markerfacecolor="#d8d8d8",
             label="best-matching ViT-B/16 block")
    axc.scatter(x, seed_pair, s=17, c=cmap(norm(seed_pair)),
                edgecolors="#3a3a3a", linewidths=0.4, zorder=3)
    axc.scatter(x, cross, s=17, c=cmap(norm(cross)), marker="s",
                edgecolors="#3a3a3a", linewidths=0.4, zorder=3)
    axc.set_xlim(0.2, NR + 0.8)
    axc.set_ylim(0.0, 1.05)
    axc.set_yticks([0, 0.5, 1.0])
    axc.set_xticks([1, 5, 10, 15, 20])
    axc.tick_params(labelsize=6.2)
    axc.set_xlabel("ResNet-50 layer", fontsize=7.2, labelpad=2.0)
    axc.set_ylabel("CKA", fontsize=7.2, labelpad=2.5)
    leg = axc.legend(loc="lower left", fontsize=6.1, frameon=False,
                     handlelength=1.5, borderpad=0.0, labelspacing=0.22,
                     handletextpad=0.4, borderaxespad=0.25)
    leg.set_clip_on(False)

    cax = fig.add_axes([0.912, a_bot, 0.014, a_h])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SIM), cax=cax,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("linear CKA similarity (-)", fontsize=7.0, color="#1a1a1a",
                 labelpad=3.5)

    fig.text(a_left, a_top + 0.075, "(a)  ResNet-50 against itself",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")
    fig.text(b_left, a_top + 0.075, "(b)  ResNet-50 against ViT-B/16",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")
    fig.text(b_left, b_bot - 0.115, "(c)  Similarity along depth",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")


    savefig(fig, "280_feature_similarity.png")


if __name__ == "__main__":
    main()
