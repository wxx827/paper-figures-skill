"""Figure 376 - Tumour segmentation, three slices by three networks.

Rows are three axial levels of one synthetic study, columns are the FLAIR
input, the reference contour and three segmenters.  Masks are drawn twice:
a translucent fill so the underlying anatomy stays readable, and a white
outline so the boundary is still legible where two classes touch.  Every Dice
printed on a tile is computed from the two masks in that tile.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, COLORS
from figkit import image_axes, corner_stamp

N = 200
TUMOUR_C = "#b23a48"       # house red
EDEMA_C = "#3b5b92"        # house blue


# ---------------------------------------------------------------------------
# Synthetic brain phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


# three axial levels: (head scale, ventricle scale, lesion centre, lesion size)
SLICES = [
    ("z = +18 mm", 0.90, 0.55, (0.30, -0.20), 0.150),
    ("z =  +6 mm", 1.00, 1.00, (0.32, -0.26), 0.200),
    ("z = -10 mm", 0.96, 0.80, (0.34, -0.30), 0.170),
]


def flair_slice(k, seed=17):
    """Return (flair image in [0,1], tumour mask, edema mask)."""
    _, hs, vs, (lx, ly), lr = SLICES[k]
    rng = np.random.default_rng(seed + 7 * k)
    x, y = _grid(N)
    wob = 1.0 + 0.015 * _sym(_sm(rng, N, 9))

    rr = _rad(x, y, 0.0, 0.0, 0.80 * hs, 0.92 * hs) * wob
    img = np.full((N, N), 0.02)
    img = np.where(rr <= 1.000, 0.30, img)                    # scalp
    img = np.where(rr <= 0.962, 0.08, img)                    # skull
    img = np.where(rr <= 0.900, 0.10, img)                    # subarachnoid CSF
    brain = rr <= 0.875
    img = np.where(brain, 0.56 + 0.04 * _sym(_sm(rng, N, 3)), img)   # GM
    rw = _rad(x, y, 0.0, 0.03, 0.62 * hs, 0.74 * hs) * wob
    img = np.where(brain & (rw <= 1.0), 0.37 + 0.03 * _sm(rng, N, 5), img)

    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    img = np.where(sulci, 0.09, img)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055 * vs, 0.22 * vs, -13),
                                 (-0.115, -0.02, 0.055 * vs, 0.22 * vs, 13)]:
        img = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.06, img)
    img = np.where((np.abs(x) < 0.010) & brain, 0.20, img)

    dl = _rad(x, y, lx, ly, lr * 1.30, lr * 1.14, -18)
    dl = dl * (1.0 + 0.11 * _sm(rng, N, 9))
    edema = (dl <= 1.00) & brain
    tumour = (dl <= 0.58) & brain
    img = np.where(edema, 0.76 + 0.04 * _sm(rng, N, 3), img)
    img = np.where(tumour, 0.88 + 0.04 * _sm(rng, N, 2), img)

    img = gaussian_filter(img, 0.9) + 0.014 * rng.standard_normal((N, N))
    return np.clip(img, 0, 1), tumour, edema, brain


# per-method boundary sloppiness: (smoothing px, threshold, wobble, blobs)
METHODS = [
    ("U-Net", 4.2, 0.40, 0.20, 2),
    ("nnU-Net", 3.0, 0.48, 0.13, 1),
    ("Ours", 1.6, 0.50, 0.065, 0),
]


def predict(mask, brain, rng, smooth, thr, wobble, n_blob, centre):
    """Degrade a reference mask into a plausible network output."""
    p = gaussian_filter(mask.astype(float), smooth)
    p = p / (p.max() + 1e-9)
    p = p + wobble * _sm(rng, mask.shape[0], 7.0)
    out = (p > thr) & brain
    yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(n_blob):          # false positives, plausibly peri-lesional
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(28, 62)
        bx = centre[0] + dist * np.cos(ang)
        by = centre[1] + dist * np.sin(ang)
        out |= (np.hypot(xx - bx, yy - by) < rng.uniform(3.5, 6.0)) & brain
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def overlay(ax, gray, masks, colors, alpha=0.42):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    for m, c in zip(masks, colors):
        rgba = np.zeros(m.shape + (4,), dtype=float)
        rgba[..., :3] = mpl.colors.to_rgb(c)
        rgba[..., 3] = alpha * m.astype(float)
        ax.imshow(rgba, interpolation="nearest", zorder=2)
        ax.contour(m.astype(float), levels=[0.5], colors="white",
                   linewidths=0.7, zorder=3)


def main():
    set_pub_style(base_font=8)
    cols = ["FLAIR input", "Reference"] + [m for m, *_ in METHODS]
    ncol = len(cols)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(len(SLICES), ncol, wspace=0.030, hspace=0.035,
                          left=0.055, right=0.992, top=0.845, bottom=0.075)

    axes = np.empty((len(SLICES), ncol), object)
    for i in range(len(SLICES)):
        img, gt_t, gt_e, brain = flair_slice(i)
        lx, ly = SLICES[i][3]
        centre = (lx * N / 2 + (N - 1) / 2, ly * N / 2 + (N - 1) / 2)
        rng = np.random.default_rng(101 + 3 * i)
        preds = [(predict(gt_t, brain, rng, s, t, w, nb, centre),
                  predict(gt_e, brain, rng, s * 1.25, t, w, nb, centre))
                 for _, s, t, w, nb in METHODS]

        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax, color="#d5d5d5", lw=0.6)
            if j == 0:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                          interpolation="bilinear")
            elif j == 1:
                overlay(ax, img, [gt_e, gt_t], [EDEMA_C, TUMOUR_C])
                corner_stamp(ax, "expert delineation", loc="lower left",
                             fontsize=6.4)
            else:
                pt, pe = preds[j - 2]
                overlay(ax, img, [pe, pt], [EDEMA_C, TUMOUR_C])
                bold = cols[j] == "Ours"
                corner_stamp(ax, f"Dice  T {dice(pt, gt_t):.3f}  |  "
                                 f"E {dice(pe, gt_e):.3f}",
                             loc="lower left", fontsize=6.8,
                             weight="bold" if bold else "normal")
            if j == 0:
                ax.set_ylabel(SLICES[i][0], fontsize=8.0, labelpad=3.0,
                              color=COLORS["ink"])
            if i == 0:
                bold = cols[j] == "Ours"
                ax.set_title(cols[j], fontsize=8.8, pad=4.0,
                             fontweight="bold" if bold else "normal",
                             color="#0b3d2e" if bold else COLORS["ink"])

    # subordinate highlight behind the ours column
    p0 = axes[0, ncol - 1].get_position()
    p1 = axes[len(SLICES) - 1, ncol - 1].get_position()
    pad = 0.005
    fig.add_artist(Rectangle((p0.x0 - pad, p1.y0 - pad),
                             p0.width + 2 * pad, p0.y1 - p1.y0 + 2 * pad + 0.030,
                             transform=fig.transFigure, facecolor="#e8f2ec",
                             edgecolor="none", zorder=-5))

    handles = [Patch(facecolor=TUMOUR_C, alpha=0.62, edgecolor="white",
                     linewidth=0.7, label="enhancing tumour"),
               Patch(facecolor=EDEMA_C, alpha=0.62, edgecolor="white",
                     linewidth=0.7, label="peritumoural edema")]


    savefig(fig, "376_med_segmentation_overlay.png")


if __name__ == "__main__":
    main()
