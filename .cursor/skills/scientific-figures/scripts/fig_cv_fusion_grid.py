"""Figure 253 - Multi-modal medical image fusion, qualitative comparison grid.

Rows are three anatomical planes of a synthetic brain phantom; columns are the
two source modalities followed by six fusion methods.  Every cell carries two
colour-boxed zoom insets whose crop windows are identical across a row, which
is the only way a reader can compare local detail between methods.
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
# Generic helpers: image cells and colour-boxed zoom insets
# ---------------------------------------------------------------------------
def image_axes(ax, color="#b8b8b8", lw=0.6):
    """Turn an axes into a bare image tile: square, no ticks, hairline frame."""
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
    """Mark ``box_xywh`` on ``ax`` and paste a magnified crop at ``inset_loc``.

    ``box_xywh`` is (x, y, w, h) in pixel coordinates of ``img``; ``inset_loc``
    is (x, y, w, h) in axes fractions.  Frame and box share ``color`` so the
    reader can pair them at a glance.
    """
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
        cx = x0 + bw if inset_loc[0] > 0.4 else x0
        con = ConnectionPatch(
            xyA=(cx, y0 + bh), coordsA=ax.transData,
            xyB=(1.0 if inset_loc[0] > 0.4 else 0.0, 1.0),
            coordsB=axi.transAxes, color=color, lw=0.6, alpha=0.85,
            zorder=6)
        ax.add_artist(con)
    return axi


# ---------------------------------------------------------------------------
# Synthetic brain phantom (Shepp-Logan style, layered ellipses + folds)
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
    return np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)


VIEWS = {
    "Axial": dict(
        rx=0.78, ry=0.90,
        wm=(0.0, 0.03, 0.62, 0.74),
        vent=[(0.115, -0.02, 0.055, 0.22, -13), (-0.115, -0.02, 0.055, 0.22, 13),
              (0.0, 0.10, 0.030, 0.10, 0)],
        falx=True, extra=None, cut=None,
        pet=[(0.00, -0.32, 0.21, 0.15, 0.75), (0.32, 0.05, 0.15, 0.21, 0.62),
             (-0.32, 0.05, 0.15, 0.21, 0.62), (0.00, 0.45, 0.27, 0.13, 0.55),
             (0.28, -0.36, 0.11, 0.10, 1.00)]),
    "Coronal": dict(
        rx=0.86, ry=0.90,
        wm=(0.0, 0.02, 0.68, 0.62),
        vent=[(0.15, -0.06, 0.050, 0.15, -22), (-0.15, -0.06, 0.050, 0.15, 22),
              (0.0, 0.07, 0.026, 0.09, 0)],
        falx=True, extra="temporal", cut=("flat", 0.62),
        pet=[(0.00, -0.40, 0.24, 0.13, 0.70), (0.38, 0.06, 0.14, 0.19, 0.60),
             (-0.38, 0.06, 0.14, 0.19, 0.60), (0.00, 0.24, 0.24, 0.14, 0.52),
             (-0.24, -0.16, 0.10, 0.10, 1.00)]),
    "Sagittal": dict(
        rx=0.92, ry=0.80,
        wm=(-0.06, -0.06, 0.66, 0.52),
        vent=[(0.02, -0.02, 0.19, 0.080, -6)],
        falx=False, extra="cerebellum", cut=("face", 0.0),
        pet=[(-0.42, -0.28, 0.20, 0.16, 0.72), (0.06, -0.16, 0.21, 0.16, 0.60),
             (0.42, 0.38, 0.19, 0.13, 0.58), (0.48, -0.20, 0.11, 0.10, 1.00)]),
}


def brain_slice(view, n=N, seed=0):
    """Return (mri, pet) float images in [0, 1] for one anatomical plane."""
    p = VIEWS[view]
    rng = np.random.default_rng(seed)
    x, y = _grid(n)
    wob = 1.0 + 0.014 * _sym(_sm(rng, n, 9))

    rr = _rad(x, y, 0.0, 0.0, p["rx"], p["ry"]) * wob
    if p["cut"] is not None:
        kind, lvl = p["cut"]
        if kind == "flat":                      # coronal: flat skull base
            rr = np.maximum(rr, np.where(y > lvl, 2.0, 0.0))
        elif kind == "face":                    # sagittal: slanted face profile
            rr = np.maximum(rr, np.where((y > 0.26 + 0.40 * (x + 0.65))
                                         & (x < -0.02), 2.0, 0.0))
    img = 0.012 + 0.012 * rng.random((n, n))
    img = np.where(rr <= 1.00, 0.30 + 0.05 * _sym(_sm(rng, n, 4)), img)
    img = np.where(rr <= 0.965, 0.90 + 0.05 * _sym(_sm(rng, n, 6)), img)
    img = np.where(rr <= 0.910, 0.13, img)

    brain = rr <= 0.882
    gm = 0.50 + 0.055 * _sym(_sm(rng, n, 3.0)) + 0.03 * _sm(rng, n, 1.2)
    img = np.where(brain, gm, img)

    cx, cy, wrx, wry = p["wm"]
    rw = _rad(x, y, cx, cy, wrx, wry) * wob
    wmv = 0.74 + 0.035 * _sym(_sm(rng, n, 5))
    img = np.where(brain & (rw <= 1.0), wmv, img)

    # gyral folding: a sine of a smooth symmetric field gives wavy sulcal bands
    g1 = _sym(_sm(rng, n, 7.0))
    g2 = _sym(_sm(rng, n, 15.0))
    fold = np.sin(8.0 * g1 + 4.2 * g2)
    sulci = (fold > 0.42) & brain & (rr > 0.50)
    img = np.where(sulci, 0.19 + 0.04 * _sm(rng, n, 1.5), img)

    if p["extra"] == "temporal":
        for sgn in (-1, 1):
            t = _rad(x, y, sgn * 0.52, 0.34, 0.22, 0.19, sgn * 18)
            img = np.where(t <= 1.0, 0.52 + 0.05 * _sm(rng, n, 2.5), img)
            img = np.where((t <= 1.0) & sulci, 0.20, img)
    if p["extra"] == "cerebellum":
        cb = _rad(x, y, 0.44, 0.42, 0.30, 0.22, -8)
        folia = np.sin(54 * (0.55 * x + 0.84 * y) + 2.4 * _sm(rng, n, 6))
        img = np.where(cb <= 1.0, 0.58 + 0.17 * (folia > 0.05), img)
        img = np.where((cb > 0.90) & (cb <= 1.02), 0.24, img)
        stem = _rad(x, y, 0.14, 0.50, 0.11, 0.30, 10)
        img = np.where(stem <= 1.0, 0.66, img)
        # corpus callosum: bright arc riding over the lateral ventricle
        cc = _rad(x, y, -0.04, 0.04, 0.42, 0.32)
        img = np.where((cc > 0.84) & (cc < 1.0) & (y < 0.04) & (rr < 0.88),
                       0.86, img)

    for (vx, vy, vrx, vry, va) in p["vent"]:
        v = _rad(x, y, vx, vy, vrx, vry, va) * wob
        img = np.where(v <= 1.0, 0.075, img)
    if p["falx"]:
        img = np.where((np.abs(x) < 0.010) & brain, 0.26, img)

    img = gaussian_filter(img, 0.7)
    img = np.clip(img + 0.010 * rng.standard_normal((n, n)), 0, 1)

    pet = np.zeros((n, n))
    for (bx, by, sx, sy, amp) in p["pet"]:
        pet += amp * np.exp(-0.5 * (((x - bx) / sx) ** 2 + ((y - by) / sy) ** 2))
    pet *= 0.75 + 0.45 * _sm(rng, n, 12)
    pet = gaussian_filter(pet, 5.0) * gaussian_filter(brain.astype(float), 3.0)
    pet = np.clip(pet + 0.02 * _sm(rng, n, 2.0), 0, None)
    pet /= pet.max() + 1e-9
    return img, pet


# ---------------------------------------------------------------------------
# Fusion + per-method degradation
# ---------------------------------------------------------------------------
# Perceptually uniform stand-in for the classic PET hot-metal ramp, which
# DESIGN_SYSTEM.md section 2.4 bans.
PET_CMAP = "magma"


def fuse(mri, pet, alpha=0.80, cmap=PET_CMAP):
    heat = plt.get_cmap(cmap)(np.clip(pet, 0, 1))[..., :3]
    a = (np.clip(pet, 0, 1) ** 1.25 * alpha)[..., None]
    return np.clip(np.dstack([mri] * 3) * (1 - a) + heat * a, 0, 1)


def degrade(rgb, rng, blur=0.0, sat=1.0, gain=1.0, bias=0.0, noise=0.0,
            halo=0.0, bleed=0.0):
    out = rgb.astype(float).copy()
    if bleed:
        lum = out.mean(2, keepdims=True)
        ch = out - lum
        ch = np.dstack([gaussian_filter(ch[..., k], bleed) for k in range(3)])
        out = lum + ch
    if blur:
        out = np.dstack([gaussian_filter(out[..., k], blur) for k in range(3)])
    if halo:
        hp = out - np.dstack([gaussian_filter(out[..., k], 3.0)
                              for k in range(3)])
        out = out + halo * hp
    if sat != 1.0:
        lum = out.mean(2, keepdims=True)
        out = lum + sat * (out - lum)
    out = (out - 0.5) * gain + 0.5 + bias
    if noise:
        out = out + noise * rng.standard_normal(out.shape)
    return np.clip(out, 0, 1)


METHODS = [
    ("DDFM", dict(blur=1.95, sat=0.70, gain=0.85, bias=0.035, noise=0.011,
                  bleed=3.2)),
    ("MACTFusion", dict(blur=1.35, sat=0.81, gain=0.92, noise=0.008,
                        bleed=2.3)),
    ("SAFusion", dict(blur=0.90, sat=0.87, gain=1.06, halo=1.05, noise=0.013,
                      bleed=1.7)),
    ("TextFusion", dict(blur=1.15, sat=0.69, gain=0.87, bias=0.055,
                        noise=0.006, bleed=2.7)),
    ("DiTFuse", dict(blur=0.85, sat=0.90, gain=0.95, noise=0.007, bleed=1.4)),
    ("Ours", dict(blur=0.0, sat=1.02, gain=1.02, halo=0.30, noise=0.002,
                  bleed=0.0)),
]

# identical crop windows per row -> the whole point of the layout
BOXES = {
    "Axial": [(46, 76), (148, 60)],
    "Coronal": [(44, 70), (150, 84)],
    "Sagittal": [(58, 82), (144, 46)],
}
BW, BH = 64, 64
INSETS = [(0.0, 0.0, 0.40, 0.40), (0.60, 0.0, 0.40, 0.40)]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)

    rows = ["Axial", "Coronal", "Sagittal"]
    col_names = ["MRI", "PET"] + [m for m, _ in METHODS]
    ncol = len(col_names)

    fig = plt.figure(figsize=(8.6, 4.20))
    gs = fig.add_gridspec(len(rows), ncol, wspace=0.035, hspace=0.035,
                          left=0.042, right=0.995, top=0.868, bottom=0.082)

    axes = np.empty((len(rows), ncol), object)
    for i, view in enumerate(rows):
        mri, pet = brain_slice(view, seed=11 + 5 * i)
        fused = fuse(mri, pet)
        cells = [(np.dstack([mri] * 3), {}),
                 (plt.get_cmap(PET_CMAP)(pet)[..., :3], {})]
        for _, kw in METHODS:
            cells.append((degrade(fused, rng, **kw), {}))

        boxes = [(bx, by, BW, BH) for (bx, by) in BOXES[view]]
        for j, (im, _) in enumerate(cells):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(im, interpolation="bilinear")
            for k, (box, loc) in enumerate(zip(boxes, INSETS)):
                col = ZOOM_RED if k == 0 else ZOOM_ORANGE
                add_zoom_inset(ax, im, box, loc, col, connect=True)
            if j == 0:
                ax.set_ylabel(view, fontsize=8.2, labelpad=3.0,
                              color="#1a1a1a")
            if i == len(rows) - 1:
                bold = col_names[j] == "Ours"
                ax.set_xlabel(col_names[j], fontsize=8.4, labelpad=3.5,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    # subordinate highlight behind the Ours column
    p0 = axes[0, ncol - 1].get_position()
    p1 = axes[len(rows) - 1, ncol - 1].get_position()
    pad = 0.006
    fig.add_artist(Rectangle((p0.x0 - pad, p1.y0 - 0.052),
                             p0.width + 2 * pad,
                             p0.y1 - p1.y0 + 0.052 + 0.012,
                             transform=fig.transFigure, facecolor="#e8f2ec",
                             edgecolor="none", zorder=-5))

    # group headers over the source block and the method block
    src = axes[0, 0].get_position(), axes[0, 1].get_position()
    met = axes[0, 2].get_position(), axes[0, ncol - 1].get_position()
    hy = met[0].y1 + 0.018
    for (a, b), lab in ((src, "Source modalities"),
                        (met, "Fusion methods  (state of the art  ->  ours)")):
        fig.text((a.x0 + b.x1) / 2, hy, lab, fontsize=7.0, color="#5a5a5a",
                 ha="center", va="bottom")
        fig.add_artist(plt.Line2D([a.x0, b.x1], [hy - 0.006] * 2,
                                  transform=fig.transFigure, color="#c9c9c9",
                                  lw=0.7, zorder=-4))
    xdiv = (src[1].x1 + met[0].x0) / 2
    fig.add_artist(plt.Line2D([xdiv, xdiv],
                              [axes[len(rows) - 1, 0].get_position().y0 - 0.01,
                               hy - 0.006],
                              transform=fig.transFigure, color="#c9c9c9",
                              lw=0.7, zorder=-4))

    savefig(fig, "253_cv_fusion_grid.png")


if __name__ == "__main__":
    main()
