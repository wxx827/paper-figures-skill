"""Figure 254 - Ablation study laid out as a qualitative comparison grid.

Two fluorescence-microscopy fields of view (rows) are restored by four
progressively complete variants of the model.  Each tile carries one red-boxed
zoom inset; the crop window is shared by every column of a row.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig

ZOOM_RED = "#b23a48"
ZOOM_ORANGE = "#c9a227"
N = 256


# ---------------------------------------------------------------------------
# Shared tile / zoom helpers
# ---------------------------------------------------------------------------
def image_axes(ax, color="#b8b8b8", lw=0.6):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def add_zoom_inset(ax, img, box_xywh, inset_loc, color, zoom_border_lw=1.6,
                   connect=False, box_lw=1.15, interpolation="nearest",
                   **imshow_kw):
    """Mark ``box_xywh`` (pixels) and paste the magnified crop at ``inset_loc``."""
    x0, y0, bw, bh = (int(v) for v in box_xywh)
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), bw, bh, facecolor="none",
                           edgecolor=color, linewidth=box_lw, zorder=6))
    axi = ax.inset_axes(inset_loc)
    axi.imshow(img[y0:y0 + bh, x0:x0 + bw], interpolation=interpolation,
               aspect="auto", **imshow_kw)
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(zoom_border_lw)
        s.set_color(color)
    axi.set_zorder(7)
    if connect:
        right = inset_loc[0] > 0.4
        ax.add_artist(ConnectionPatch(
            xyA=(x0 + bw if right else x0, y0 + bh), coordsA=ax.transData,
            xyB=(1.0 if right else 0.0, 1.0), coordsB=axi.transAxes,
            color=color, lw=0.6, alpha=0.85, zorder=6))
    return axi


def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


# ---------------------------------------------------------------------------
# Synthetic fluorescence field: green cytoplasm + blue nuclei on a dark slide
# ---------------------------------------------------------------------------
def cell_field(n=N, seed=0, n_cells=26):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cyto = np.zeros((n, n))
    edge = np.zeros((n, n))
    nuc = np.zeros((n, n))

    centers = []
    tries = 0
    while len(centers) < n_cells and tries < 4000:
        tries += 1
        cx, cy = rng.uniform(8, n - 8, 2)
        rr = rng.uniform(11, 20)
        if all((cx - a) ** 2 + (cy - b) ** 2 > (0.80 * (rr + c)) ** 2
               for a, b, c in centers):
            centers.append((cx, cy, rr))

    for (cx, cy, rr) in centers:
        ar = rng.uniform(0.62, 1.35)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        d = np.sqrt((u / (rr * ar)) ** 2 + (v / (rr / ar)) ** 2)
        d = d * (1 + 0.16 * np.sin(4 * np.arctan2(v, u) + rng.uniform(0, 6)))
        body = np.clip(1.0 - d, 0, 1)
        cyto = np.maximum(cyto, (0.42 + 0.30 * rng.random()) * body ** 0.55)
        edge = np.maximum(edge, np.exp(-((d - 0.92) / 0.085) ** 2))
        nd = np.sqrt(((u - 0.12 * rr) / (0.40 * rr)) ** 2
                     + ((v + 0.10 * rr) / (0.34 * rr)) ** 2)
        nuc = np.maximum(nuc, np.clip(1.0 - nd, 0, 1) ** 0.7)

    grain = 0.10 * _sm(rng, n, 1.6) + 0.05 * _sm(rng, n, 6.0)
    g = np.clip(cyto * (1 + grain) + 0.85 * edge, 0, 1.4)
    b = np.clip(nuc * (0.95 + 0.12 * _sm(rng, n, 2.0)), 0, 1.3)
    g = g + 0.035 + 0.030 * _sm(rng, n, 9)          # faint background haze
    rgb = np.dstack([0.14 * g + 0.30 * b,
                     0.98 * g + 0.16 * b,
                     0.20 * g + 0.96 * b])
    return np.clip(rgb, 0, 1)


def degrade(rgb, rng, blur=0.0, noise=0.0, gain=1.0, bias=0.0, halo=0.0,
            sat=1.0, quant=0):
    out = rgb.astype(float).copy()
    if blur:
        out = np.dstack([gaussian_filter(out[..., k], blur) for k in range(3)])
    if halo:
        hp = out - np.dstack([gaussian_filter(out[..., k], 4.0)
                              for k in range(3)])
        out = out + halo * hp
    if sat != 1.0:
        lum = out.mean(2, keepdims=True)
        out = lum + sat * (out - lum)
    out = (out - 0.5) * gain + 0.5 + bias
    if quant:
        out = np.round(out * quant) / quant
    if noise:
        out = out + noise * rng.standard_normal(out.shape)
    return np.clip(out, 0, 1)


VARIANTS = [
    ("(a)  Input", r"low-SNR acquisition",
     dict(blur=2.4, noise=0.055, gain=0.72, bias=0.045, sat=0.80), 0.512),
    ("(b)  w/o $\\Phi$, $\\mathcal{L}_{\\mathrm{sem}}$, $\\gamma$",
     r"plain U-Net backbone",
     dict(blur=1.55, noise=0.028, gain=0.84, bias=0.020, sat=0.88), 0.671),
    ("(c)  w/o $\\mathcal{L}_{\\mathrm{sem}}$, $\\gamma$",
     r"+ prior module $\Phi$",
     dict(blur=1.00, noise=0.018, gain=0.92, halo=0.45, sat=0.93), 0.758),
    ("(d)  w/o $\\gamma$", r"+ semantic loss $\mathcal{L}_{\mathrm{sem}}$",
     dict(blur=0.55, noise=0.010, gain=0.97, halo=0.25, sat=0.97), 0.826),
    ("(e)  Ours  (full)", r"+ adaptive gate $\gamma$",
     dict(blur=0.0, noise=0.003, gain=1.03, halo=0.20, sat=1.02), 0.891),
]

ROW_IOU_SHIFT = [0.0, -0.024]
BOXES = {0: (150, 52), 1: (58, 138)}
BW = BH = 62
INSET = (0.575, 0.0, 0.425, 0.425)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4)
    rows = ["Field #1", "Field #2"]
    ncol = len(VARIANTS)

    fig = plt.figure(figsize=(7.6, 3.62))
    gs = fig.add_gridspec(2, ncol, wspace=0.035, hspace=0.035,
                          left=0.052, right=0.995, top=0.865, bottom=0.135)

    axes = np.empty((2, ncol), object)
    for i in range(2):
        clean = cell_field(seed=21 + 7 * i, n_cells=24 + 4 * i)
        bx, by = BOXES[i]
        for j, (name, _sub, kw, iou) in enumerate(VARIANTS):
            im = degrade(clean, rng, **kw)
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(im, interpolation="bilinear")
            add_zoom_inset(ax, im, (bx, by, BW, BH), INSET, ZOOM_RED,
                           connect=True)
            val = iou + ROW_IOU_SHIFT[i] * (0.4 + 0.6 * j / (ncol - 1))
            ax.text(0.035, 0.965, f"IoU {val:.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.3, color="white",
                    fontweight="bold" if j == ncol - 1 else "normal",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#00000099",
                              edgecolor="none"))
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=8.0, labelpad=3.0)
            if i == 1:
                bold = j == ncol - 1
                ax.set_xlabel(name, fontsize=7.8, labelpad=3.2,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    # component sub-captions in a second, visibly subordinate label line
    for j, (_n, sub, _kw, _iou) in enumerate(VARIANTS):
        p = axes[1, j].get_position()

    p0, p1 = axes[0, ncol - 1].get_position(), axes[1, ncol - 1].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.006, 0.028), p0.width + 0.012,
                             p0.y1 - 0.028 + 0.010,
                             transform=fig.transFigure, facecolor="#e8f2ec",
                             edgecolor="none", zorder=-5))

    savefig(fig, "254_cv_ablation_grid.png")


if __name__ == "__main__":
    main()
