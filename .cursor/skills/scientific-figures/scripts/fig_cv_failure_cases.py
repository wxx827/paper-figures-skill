"""Figure 260 - Honest failure-case analysis for a cell-segmentation model.

Each row is one failure mode (touching cells, small and faint cells, uneven
illumination) shown as input, our prediction, ground truth and the signed error
map, followed by a deliberately subordinate text column that names the cause.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from scipy.ndimage import (gaussian_filter, binary_dilation, binary_erosion,
                           binary_closing, map_coordinates)

from pubstyle import set_pub_style, PALETTES, savefig

N = 224
FP_COL = "#b23a48"          # predicted but not there
FN_COL = "#3b5b92"          # there but not predicted


def image_axes(ax, color="#b8b8b8", lw=0.6):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def _sm(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _hex(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


# ---------------------------------------------------------------------------
# Fluorescence field with a controllable failure mode
# ---------------------------------------------------------------------------
def field(mode, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    cyto = np.zeros((N, N))
    edge = np.zeros((N, N))
    nuc = np.zeros((N, N))
    gt = np.zeros((N, N), bool)
    small = np.zeros((N, N), bool)

    cells = []
    if mode == "touch":                      # dense clusters that abut
        for _ in range(6):
            gx, gy = rng.uniform(40, N - 40, 2)
            for _ in range(int(rng.integers(3, 6))):
                ang = rng.uniform(0, 2 * np.pi)
                d = rng.uniform(10, 22)
                cells.append((gx + d * np.cos(ang), gy + d * np.sin(ang),
                              rng.uniform(12, 17), False))
    else:
        for _ in range(20):
            cells.append((rng.uniform(14, N - 14), rng.uniform(14, N - 14),
                          rng.uniform(12, 19), False))
        if mode == "small":
            for _ in range(16):
                cells.append((rng.uniform(10, N - 10), rng.uniform(10, N - 10),
                              rng.uniform(4.0, 6.2), True))

    for (cx, cy, r, is_small) in cells:
        ar = rng.uniform(0.70, 1.30)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        d = np.sqrt((u / (r * ar)) ** 2 + (v / (r / ar)) ** 2)
        d = d * (1 + 0.14 * np.sin(4 * np.arctan2(v, u) + rng.uniform(0, 6)))
        body = np.clip(1.0 - d, 0, 1)
        amp = 0.28 if is_small else 0.46 + 0.26 * rng.random()
        cyto = np.maximum(cyto, amp * body ** 0.55)
        edge = np.maximum(edge, (0.45 if is_small else 1.0) *
                          np.exp(-((d - 0.92) / 0.09) ** 2))
        nd = np.sqrt(((u - 0.10 * r) / (0.40 * r)) ** 2
                     + ((v + 0.08 * r) / (0.34 * r)) ** 2)
        nuc = np.maximum(nuc, (0.4 if is_small else 1.0) *
                         np.clip(1.0 - nd, 0, 1) ** 0.7)
        gt |= d <= 1.0
        if is_small:
            small |= d <= 1.0

    g = np.clip(cyto * (1 + 0.10 * _sm(rng, (N, N), 1.6)) + 0.80 * edge, 0, 1.4)
    b = np.clip(nuc * (0.95 + 0.12 * _sm(rng, (N, N), 2.0)), 0, 1.3)
    g = g + 0.035 + 0.030 * _sm(rng, (N, N), 9)

    glare = np.zeros((N, N))
    if mode == "illum":
        glare = 1.05 * np.exp(-(((xx - 0.30 * N) ** 2 + (yy - 0.34 * N) ** 2)
                                / (2 * 46.0 ** 2)))
        ramp = 0.55 * (1 - yy / N) + 0.25
        g = g * (ramp + glare) + 0.55 * glare
        b = b * (ramp + 0.5 * glare)

    img = np.dstack([0.14 * g + 0.30 * b, 0.98 * g + 0.16 * b,
                     0.20 * g + 0.96 * b])
    return np.clip(img, 0, 1), gt, small, glare


def predict(gt, small, glare, mode, rng):
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    dy, dx = 1.6 * _sm(rng, (N, N), 14), 1.6 * _sm(rng, (N, N), 14)
    pred = map_coordinates(gt.astype(np.uint8), [yy + dy, xx + dx], order=0,
                           mode="nearest").astype(bool)
    if mode == "touch":                      # neighbouring cells merge
        pred = binary_closing(binary_dilation(pred, iterations=3),
                              iterations=4)
        pred = binary_erosion(pred, iterations=1)
    if mode == "small":                      # faint cells drop out
        pred &= ~small
        pred = binary_erosion(pred, iterations=1)
    if mode == "illum":                      # glare hallucinates signal
        halo = (glare > 0.28) & (_sm(rng, (N, N), 4.0) > -0.35)
        pred |= halo
        pred &= ~((glare > 0.75) & (_sm(rng, (N, N), 3.0) > 0.6))
    return pred


def overlay(img, mask, color, alpha=0.42):
    out = np.where(mask[..., None], img * (1 - alpha) + _hex(color) * alpha,
                   img)
    ring = binary_dilation(mask, iterations=1) & ~binary_erosion(
        mask, iterations=1)
    return np.clip(np.where(ring[..., None], _hex(color), out), 0, 1)


def error_rgb(pred, gt):
    out = np.ones((N, N, 3)) * _hex("#f4f4f4")
    out = np.where((pred & gt)[..., None], _hex("#d9d9d9"), out)
    out = np.where((pred & ~gt)[..., None], _hex(FP_COL), out)
    out = np.where((~pred & gt)[..., None], _hex(FN_COL), out)
    return out


MODES = [
    ("touch", "Occlusion",
     "Touching cells share one bright membrane, so the decoder emits a "
     "single connected component. Splitting fails below a 2 px gap."),
    ("small", "Small objects",
     "Cells under 6 px radius sit barely above the background haze and are "
     "erased by the stride-16 bottleneck. Recall 0.94 -> 0.41."),
    ("illum", "Illumination",
     "Saturated glare mimics cytoplasmic texture: the model hallucinates "
     "membrane and drops the core of the brightest blob."),
]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(31)
    col_names = ["Input", "Ours", "Ground truth", "Error map"]
    ncol = len(col_names)

    fig = plt.figure(figsize=(7.9, 4.75))
    gs = fig.add_gridspec(3, ncol + 1, wspace=0.035, hspace=0.035,
                          width_ratios=[1, 1, 1, 1, 1.02],
                          left=0.075, right=0.995, top=0.895, bottom=0.115)

    axes = np.empty((3, ncol), object)
    for i, (mode, short, story) in enumerate(MODES):
        img, gt, small, glare = field(mode, seed=41 + 11 * i)
        pred = predict(gt, small, glare, mode, rng)
        iou = float((pred & gt).sum()) / float((pred | gt).sum() + 1e-9)
        tiles = [img, overlay(img, pred, "#c9a227"), overlay(img, gt, "#2e7d5b"),
                 error_rgb(pred, gt)]
        for j, im in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(im, interpolation="bilinear")
            if j == 1:
                ax.text(0.032, 0.962, f"IoU {iou:.3f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6.2, color="white",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.20",
                                  facecolor="#000000aa", edgecolor="none"))
            if j == 3:
                fp = float((pred & ~gt).sum()) / gt.size * 100
                fn = float((~pred & gt).sum()) / gt.size * 100
                ax.text(0.032, 0.962, f"FP {fp:.1f}%   FN {fn:.1f}%",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.0, color="#1a1a1a", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.20",
                                  facecolor="#ffffffdd", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(short, fontsize=7.8, labelpad=3.0)
            if i == 2:
                ax.set_xlabel(col_names[j], fontsize=8.0, labelpad=3.2)

        # ---- subordinate commentary column: grey card, no image frame
        axt = fig.add_subplot(gs[i, ncol])
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_xlim(0, 1)
        axt.set_ylim(0, 1)
        for s in axt.spines.values():
            s.set_visible(False)
        axt.add_patch(FancyBboxPatch((0.045, 0.06), 0.91, 0.88,
                                     boxstyle="round,pad=0.012,rounding_size=0.05",
                                     facecolor="#f2f2f0", edgecolor="#dcdcd8",
                                     linewidth=0.7, transform=axt.transAxes,
                                     clip_on=False, zorder=0))
        axt.text(0.095, 0.885, f"({'abc'[i]})  Why it fails", fontsize=7.0,
                 fontweight="bold", color="#3c3c3c", va="top", ha="left")
        axt.text(0.095, 0.775, textwrap.fill(story, 31), fontsize=6.1,
                 color="#5a5a5a", va="top", ha="left", linespacing=1.40,
                 transform=axt.transAxes)

    p = axes[2, 0].get_position()
    q = fig.axes[-1].get_position()
    fig.text((q.x0 + q.x1) / 2, p.y0 - 0.030, "Failure analysis", fontsize=7.6,
             ha="center", va="top", color="#5a5a5a", style="italic")

    handles = [Patch(facecolor="#d9d9d9", edgecolor="none", label="correct"),
               Patch(facecolor=FP_COL, edgecolor="none",
                     label="false positive"),
               Patch(facecolor=FN_COL, edgecolor="none",
                     label="false negative"),
               Patch(facecolor="#c9a227", edgecolor="none",
                     label="our mask"),
               Patch(facecolor="#2e7d5b", edgecolor="none", label="GT mask")]
    leg = fig.legend(handles=handles, ncol=5, loc="lower center",
                     bbox_to_anchor=(0.5, 0.004), frameon=False, fontsize=7.0,
                     handlelength=1.1, handleheight=0.9, columnspacing=1.6,
                     handletextpad=0.45)
    leg.set_zorder(10)

    savefig(fig, "260_cv_failure_cases.png")


if __name__ == "__main__":
    main()
