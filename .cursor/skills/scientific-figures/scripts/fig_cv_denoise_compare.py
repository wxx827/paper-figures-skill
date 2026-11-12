"""Figure 258 - Image restoration under two degradation types.

Row 1 is additive Gaussian noise, row 2 is rain streaks.  Columns run from the
degraded observation through three baselines to ours and the clean reference;
every tile carries a red-boxed zoom inset and a corner PSNR read-out.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from pubstyle import set_pub_style, PALETTES, savefig

ZOOM_RED = "#b23a48"
H, W = 200, 256
BOXES = {0: (146, 46), 1: (34, 126)}       # one window per row, shared by cols
BW = BH = 58
INSET = (0.575, 0.0, 0.425, 0.425)


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


def _sm(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _hex(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


# ---------------------------------------------------------------------------
# Synthetic landscape: layered ridges, a lake with reflection, foreground grass
# ---------------------------------------------------------------------------
def landscape(seed=2):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    top, bot = _hex("#1f4f8c"), _hex("#e8cfa4")
    g = np.clip((yy / (0.58 * H)) ** 0.80, 0, 1)[..., None]
    img = top * (1 - g) + bot * g
    sun = np.exp(-(((xx - 200) ** 2 + (yy - 24) ** 2) / (2 * 15.0 ** 2)))
    img = np.clip(img + 0.62 * sun[..., None] * _hex("#ffeeba"), 0, 1)
    cloud = np.clip(_sm(rng, (H, W), (4.0, 18.0)) * 0.55 - 0.05, 0, 1)
    sky = (yy < 0.58 * H)[..., None]
    img = np.where(sky, img * (1 - 0.34 * cloud[..., None])
                   + 0.34 * cloud[..., None] * _hex("#f4f1ea"), img)

    water_y = int(0.62 * H)
    ridges = [(0.36, 30, _hex("#8fa3bd"), 9.0),
              (0.46, 34, _hex("#5b6f8b"), 6.0),
              (0.55, 24, _hex("#2f3f52"), 4.0)]
    for (base, amp, col, sg) in ridges:
        p = gaussian_filter1d(rng.standard_normal(W), sg, mode="wrap")
        p = p / (p.std() + 1e-9)
        prof = base * H - amp * (0.6 * p + 0.4 * np.roll(p, 37))
        m = yy > prof[None, :]
        # ridge shading plus a lit crest line so the silhouette stays crisp
        shade = 1 + 0.13 * _sm(rng, (H, W), 2.2)
        crest = m & (yy < prof[None, :] + 2.0)
        img = np.where((m & (yy < water_y))[..., None],
                       col * shade[..., None], img)
        img = np.where((crest & (yy < water_y))[..., None],
                       np.clip(col * 1.45, 0, 1), img)
        snow = m & (yy < prof[None, :] + 9) & \
            (prof[None, :] < np.percentile(prof, 28))
        img = np.where((snow & (yy < water_y))[..., None], _hex("#eef2f6"), img)

    # lake with a mirrored, rippled reflection of the nearest ridge
    lake = yy >= water_y
    refl_y = (2 * water_y - yy).astype(int)
    refl_y = np.clip(refl_y + (2.2 * np.sin(yy / 3.0 + 2 * xx / 29.0)
                               ).astype(int), 0, H - 1)
    refl = img[refl_y, xx.astype(int)]
    water = 0.40 * refl + 0.60 * _hex("#26384f")
    ripple = 1 + 0.11 * np.sin(yy / 2.2 + 0.9 * _sm(rng, (H, W), 2.0))
    water = water * ripple[..., None]
    glint = np.exp(-((xx - 196) ** 2) / (2 * 26.0 ** 2)) * \
        (np.sin(yy / 1.9) > 0.35)
    water = np.clip(water + 0.55 * glint[..., None] * _hex("#ffe9b0"), 0, 1)
    img = np.where(lake[..., None], water, img)

    shore_y = int(0.80 * H)
    grass = yy > shore_y + 7 * np.sin(xx / 23.0) + 3 * np.sin(xx / 7.0)
    gt = 1 + 0.42 * (0.65 * _sm(rng, (H, W), 0.9) + 0.35 * _sm(rng, (H, W), 3.0))
    img = np.where(grass[..., None], _hex("#42552c") * gt[..., None], img)
    pebble = grass & (_sm(rng, (H, W), 1.1) > 1.35)
    img = np.where(pebble[..., None], _hex("#9a9585"), img)

    for (tx, sc) in [(28, 1.15), (56, 0.80), (222, 1.0), (196, 0.66)]:
        th, wb = 52 * sc, 11.0 * sc
        ty = water_y + 2
        for k in range(4):
            y0 = ty - th + k * (th * 0.21)
            y1 = y0 + th * 0.40
            halfw = wb * (1 - 0.17 * k) * np.clip(1 - (yy - y0) / (y1 - y0),
                                                  0, 1)
            tri = (yy > y0) & (yy < y1) & (np.abs(xx - tx) < halfw)
            img = np.where(tri[..., None], _hex("#20301b") *
                           (1 + 0.22 * _sm(rng, (H, W), 1.1))[..., None], img)
        trunk = (np.abs(xx - tx) < 1.8 * sc) & (yy > ty - 8) & (yy < ty + 3)
        img = np.where(trunk[..., None], _hex("#3a2d22"), img)

    vign = 1 - 0.13 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5)
    img = np.clip((img * vign[..., None] - 0.5) * 1.14 + 0.5, 0, 1)
    return np.clip(np.dstack([gaussian_filter(img[..., k], 0.4)
                              for k in range(3)]), 0, 1)


def add_noise(clean, rng, sigma=0.19):
    return np.clip(clean + sigma * rng.standard_normal(clean.shape), 0, 1)


def add_rain(clean, rng, n_streaks=520, veil=0.16):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    layer = np.zeros((H, W))
    ang = np.deg2rad(16)
    for _ in range(n_streaks):
        x0 = rng.uniform(-30, W + 30)
        y0 = rng.uniform(-10, H)
        ln = rng.uniform(9, 26)
        t = np.linspace(0, ln, int(ln) + 2)
        ii = np.clip((y0 + t * np.cos(ang)).astype(int), 0, H - 1)
        jj = np.clip((x0 + t * np.sin(ang)).astype(int), 0, W - 1)
        layer[ii, jj] = np.maximum(layer[ii, jj], rng.uniform(0.45, 1.0))
    layer = gaussian_filter(layer, 0.7)
    out = clean * (1 - veil) + veil * 0.86 + 0.55 * layer[..., None]
    out = np.dstack([gaussian_filter(out[..., k], 0.35) for k in range(3)])
    return np.clip(out, 0, 1)


def restore(clean, degraded, rng, w=0.5, blur=0.0, resid=0.0, halo=0.0):
    """Blend back toward the clean image with a method-specific defect."""
    out = w * clean + (1 - w) * degraded
    if blur:
        out = np.dstack([gaussian_filter(out[..., k], blur) for k in range(3)])
    if halo:
        hp = out - np.dstack([gaussian_filter(out[..., k], 3.0)
                              for k in range(3)])
        out = out + halo * hp
    if resid:
        out = out + resid * rng.standard_normal(out.shape)
    return np.clip(out, 0, 1)


def psnr(ref, test):
    return 10.0 * np.log10(1.0 / max(float(np.mean((ref - test) ** 2)), 1e-12))


METHODS = [("BM3D", dict(w=0.72, blur=1.35, resid=0.012)),
           ("DnCNN", dict(w=0.82, blur=0.70, resid=0.022)),
           ("Restormer", dict(w=0.92, blur=0.30, resid=0.008, halo=0.25)),
           ("Ours", dict(w=0.985, blur=0.0, resid=0.002, halo=0.15))]
ROWS = [("Gaussian noise  $\\sigma$ = 50", "noise"),
        ("Rain streaks", "rain")]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(12)
    clean = landscape(seed=2)
    col_names = ["Degraded"] + [m for m, _ in METHODS] + ["Ground truth"]
    ncol = len(col_names)

    fig = plt.figure(figsize=(8.2, 3.35))
    gs = fig.add_gridspec(2, ncol, wspace=0.032, hspace=0.032,
                          left=0.062, right=0.995, top=0.885, bottom=0.115)

    axes = np.empty((2, ncol), object)
    for i, (rname, kind) in enumerate(ROWS):
        deg = add_noise(clean, rng) if kind == "noise" \
            else add_rain(clean, rng)
        tiles = [deg] + [restore(clean, deg, rng, **kw) for _, kw in METHODS] \
            + [clean]
        bx, by = BOXES[i]
        for j, im in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(im, interpolation="bilinear")
            add_zoom_inset(ax, im, (bx, by, BW, BH), INSET, ZOOM_RED,
                           connect=True)
            best = col_names[j] == "Ours"
            txt = "reference" if j == ncol - 1 else f"{psnr(clean, im):.2f} dB"
            ax.text(0.032, 0.962, txt, transform=ax.transAxes, ha="left",
                    va="top", fontsize=6.0, color="white",
                    fontweight="bold" if best else "normal", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.20",
                              facecolor="#000000aa", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(rname, fontsize=7.6, labelpad=3.0)
            if i == 1:
                ax.set_xlabel(col_names[j], fontsize=8.0, labelpad=3.2,
                              fontweight="bold" if best else "normal",
                              color="#0b3d2e" if best else "#1a1a1a")

    k = col_names.index("Ours")
    p0, p1 = axes[0, k].get_position(), axes[1, k].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.005, p1.y0 - 0.052), p0.width + 0.010,
                             p0.y1 - p1.y0 + 0.062, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    savefig(fig, "258_cv_denoise_compare.png")


if __name__ == "__main__":
    main()
