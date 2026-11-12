"""Figure 257 - x4 single-image super-resolution, classic two-row layout.

Row 1 holds the full frame of every method, row 2 the identically placed red-box
crop, column-aligned underneath.  PSNR / SSIM are measured against the HR
reference on the full frame.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, zoom

from pubstyle import set_pub_style, PALETTES, savefig

ZOOM_RED = "#b23a48"
N = 256
BOX = (92, 96, 56, 56)          # identical crop window for every column


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
# Synthetic high-frequency scene: a building facade (windows, railings, bricks)
# ---------------------------------------------------------------------------
def facade(n=N, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    wall = _hex("#b8a993") * (1 + 0.06 * _sm(rng, (n, n), 2.0))[..., None]
    mortar = ((yy % 7) < 1.0)
    off = np.where(((yy // 7) % 2) == 0, 0.0, 7.0)
    mortar |= (((xx + off) % 14) < 1.0)
    img = np.where(mortar[..., None], _hex("#9c8d78"), wall)

    # window grid + balcony railings: the fine structure the metrics live on
    for r in range(4):
        for c in range(4):
            x0, y0 = 14 + 60 * c, 10 + 60 * r
            w, h = 38, 40
            frame = (xx >= x0 - 3) & (xx < x0 + w + 3) & \
                    (yy >= y0 - 3) & (yy < y0 + h + 3)
            glass = (xx >= x0) & (xx < x0 + w) & (yy >= y0) & (yy < y0 + h)
            img = np.where(frame[..., None], _hex("#efe7d8"), img)
            sky = _hex("#3d4d63") + 0.20 * (yy - y0)[..., None] / h
            img = np.where(glass[..., None], sky, img)
            mull = glass & (((np.abs(xx - (x0 + w / 2)) < 1.0) |
                             (np.abs(yy - (y0 + h * 0.42)) < 1.0)))
            img = np.where(mull[..., None], _hex("#e6dccb"), img)
            refl = glass & (((xx - x0) * 0.9 + (yy - y0) * 1.6) % 17 < 4)
            img = np.where(refl[..., None],
                           _hex("#7d94ad") * (0.9 + 0.2 * rng.random()), img)
            # railing: 2 px period vertical bars
            rail = (xx >= x0 - 4) & (xx < x0 + w + 4) & \
                   (yy >= y0 + h + 4) & (yy < y0 + h + 13) & ((xx % 3) < 1.4)
            img = np.where(rail[..., None], _hex("#59606b"), img)
            bar = (xx >= x0 - 4) & (xx < x0 + w + 4) & \
                  (yy >= y0 + h + 3) & (yy < y0 + h + 5)
            img = np.where(bar[..., None], _hex("#4a5058"), img)

    stripe = (yy > n - 22) & (((xx + yy) % 16) < 8)
    img = np.where(((yy > n - 22))[..., None], _hex("#c8503f"), img)
    img = np.where(stripe[..., None], _hex("#f0ece2"), img)

    shade = 1 - 0.14 * (yy / n) - 0.10 * ((xx / n - 0.5) ** 2)
    img = img * shade[..., None]
    img = np.clip(img + 0.006 * rng.standard_normal(img.shape), 0, 1)
    return np.clip(np.dstack([gaussian_filter(img[..., k], 0.45)
                              for k in range(3)]), 0, 1)


def psnr(ref, test):
    mse = float(np.mean((ref - test) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def ssim(ref, test, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for k in range(ref.shape[2]):
        a, b = ref[..., k], test[..., k]
        ma, mb = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
        saa = gaussian_filter(a * a, sigma) - ma ** 2
        sbb = gaussian_filter(b * b, sigma) - mb ** 2
        sab = gaussian_filter(a * b, sigma) - ma * mb
        vals.append((((2 * ma * mb + c1) * (2 * sab + c2)) /
                     ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))).mean())
    return float(np.mean(vals))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(9)
    r = 4
    hr = facade(seed=5)

    lo = np.stack([gaussian_filter(hr[..., k], 0.45 * r)[::r, ::r]
                   for k in range(3)], -1)
    lo = np.clip(lo + 0.004 * rng.standard_normal(lo.shape), 0, 1)

    nearest = np.clip(np.repeat(np.repeat(lo, r, 0), r, 1), 0, 1)[:N, :N]
    bicubic = np.clip(np.dstack([zoom(lo[..., k], N / lo.shape[0], order=3)
                                 for k in range(3)]), 0, 1)
    hp = bicubic - np.dstack([gaussian_filter(bicubic[..., k], 1.8)
                              for k in range(3)])

    def blend(w_hr, w_hp, noise):
        out = (1 - w_hr) * bicubic + w_hr * hr + w_hp * hp
        return np.clip(out + noise * rng.standard_normal(hr.shape), 0, 1)

    srcnn = blend(0.24, 0.55, 0.006)
    edsr = blend(0.48, 0.40, 0.004)
    swinir = blend(0.68, 0.28, 0.003)
    ours = blend(0.90, 0.14, 0.0015)

    imgs = [nearest, bicubic, srcnn, edsr, swinir, ours]
    names = ["LR input  (x4)", "Bicubic", "SRCNN", "EDSR", "SwinIR", "Ours"]
    stats = [(psnr(hr, im), ssim(hr, im)) for im in imgs]
    for nm, (p, s) in zip(names, stats):
        print(f"  {nm:16s} PSNR={p:6.2f} dB  SSIM={s:.4f}")

    ncol = len(imgs)
    fig = plt.figure(figsize=(8.0, 3.95))
    gs = fig.add_gridspec(2, ncol, wspace=0.038, hspace=0.235,
                          left=0.030, right=0.994, top=0.850, bottom=0.075)

    bx, by, bw, bh = BOX
    axes_top = []
    for j, (im, nm) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(gs[0, j])
        axes_top.append(ax)
        image_axes(ax)
        ax.imshow(im, interpolation="nearest")
        best = nm == "Ours"
        ax.set_title(nm, fontsize=8.2, pad=3.2,
                     fontweight="bold" if best else "normal",
                     color="#0b3d2e" if best else "#1a1a1a")
        ax.add_patch(Rectangle((bx - 0.5, by - 0.5), bw, bh, facecolor="none",
                               edgecolor=ZOOM_RED, linewidth=1.15, zorder=5))
        p, s = stats[j]
        ax.text(0.5, -0.035, f"{p:.2f} dB / {s:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.6,
                fontweight="bold" if best else "normal",
                color="#0b3d2e" if best else "#4a4a4a")

        azx = fig.add_subplot(gs[1, j])
        image_axes(azx, color=ZOOM_RED, lw=1.5)
        azx.imshow(im[by:by + bh, bx:bx + bw], interpolation="nearest")
        if j == 0:
            azx.set_ylabel("red-box zoom", fontsize=6.8, color="#8c2f2f",
                           labelpad=3.0)

    p0, p1 = axes_top[-1].get_position(), fig.axes[-1].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.006, p1.y0 - 0.012), p0.width + 0.012,
                             p0.y1 - p1.y0 + 0.048, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    savefig(fig, "257_cv_superres_compare.png")


if __name__ == "__main__":
    main()
