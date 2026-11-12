"""Figure 275 - How feature maps change with depth.

Six stages of a convolutional backbone (columns) by four channels (rows).
Early stages answer to oriented edges at full resolution; later stages
collapse onto a handful of coarse, part-like regions.  Every tile is drawn on
a single shared ``viridis`` scale, so the growth of peak activation with
depth is a real effect rather than per-tile auto-contrast.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

FEAT = "viridis"
N = 96
STAGES = [("conv1_2", 96), ("conv2_2", 48), ("conv3_3", 24),
          ("conv4_3", 12), ("conv5_3", 6), ("head", 3)]
PEAKS = [0.62, 0.70, 0.79, 0.88, 0.95, 1.00]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(0.9, 2.5, 6.0), weights=(0.45, 0.33, 0.22)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _ellipse(yy, xx, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _soft(mask, sigma=0.8):
    return np.clip(gaussian_filter(mask.astype(float), sigma), 0, 1)


def _paint(img, mask, value, texture):
    return img * (1 - mask) + mask * np.clip(value + texture, 0.02, 0.99)


PARTS = {"head": (0.71, 0.33, 0.13, 0.12),
         "torso": (0.45, 0.55, 0.26, 0.16),
         "legs": (0.47, 0.82, 0.24, 0.12),
         "context": (0.17, 0.24, 0.30, 0.22)}


def synth_subject(seed=13, n=N):
    """A quadruped in front of a defocused meadow, in grayscale."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    v = yy / (n - 1.0)
    tex = 0.032 * _fbm(rng, n)

    img = 0.88 - 0.26 * np.clip(v / 0.58, 0, 1)
    img += 0.09 * _fbm(rng, n, (4, 11, 24))
    img = np.where(v > 0.58, 0.50 + 0.10 * _fbm(rng, n, (2, 6, 14)), img)
    img = gaussian_filter(img, 2.2)

    body = _ellipse(yy, xx, 0.45 * n, 0.555 * n, 0.245 * n, 0.140 * n, -6)
    neck = _ellipse(yy, xx, 0.635 * n, 0.440 * n, 0.075 * n, 0.100 * n, -24)
    head = _ellipse(yy, xx, 0.712 * n, 0.330 * n, 0.100 * n, 0.090 * n)
    snout = _ellipse(yy, xx, 0.822 * n, 0.362 * n, 0.070 * n, 0.042 * n, 6)
    ear = _ellipse(yy, xx, 0.662 * n, 0.258 * n, 0.038 * n, 0.072 * n, 16)
    tail = _ellipse(yy, xx, 0.208 * n, 0.442 * n, 0.092 * n, 0.024 * n, 42)
    legs = np.zeros((n, n), bool)
    for lx, lh in ((0.325, 0.150), (0.405, 0.145),
                   (0.550, 0.150), (0.625, 0.145)):
        legs |= _ellipse(yy, xx, lx * n, (0.555 + lh) * n, 0.030 * n, lh * n)
    sub = body | neck | head | snout | ear | tail | legs
    img = _paint(img, _soft(sub, 0.7), 0.31, tex * 1.3)
    img = _paint(img, _soft(head | snout | ear, 0.6), 0.39, tex * 1.0)
    img = _paint(img, _soft(_ellipse(yy, xx, 0.872 * n, 0.356 * n,
                                     0.020 * n, 0.017 * n), 0.4), 0.08, tex * 0.3)

    grass = np.zeros((n, n), bool)                      # foreground blades
    for _ in range(90):
        gx = rng.uniform(0, n)
        gh = rng.uniform(0.03, 0.10) * n
        col = np.abs(xx - (gx + 0.35 * (yy - n) * rng.uniform(-1, 1) / n)) < 0.007 * n
        grass |= col & (yy > n - gh)
    img = _paint(img, _soft(grass, 0.5), 0.36, tex * 1.6)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo), 0, 1)


def oriented_energy(img, angle_deg, sigma):
    """Response of an oriented derivative-of-Gaussian, rectified."""
    s = gaussian_filter(img, sigma)
    gy, gx = np.gradient(s)
    t = np.deg2rad(angle_deg)
    d = gx * np.cos(t) + gy * np.sin(t)
    return gaussian_filter(np.abs(d), max(0.6, sigma * 0.5))


def part_blob(n, key, spread=1.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx, cy, rx, ry = PARTS[key]
    return np.exp(-0.5 * (((xx - cx * n) / (rx * n * spread)) ** 2
                          + ((yy - cy * n) / (ry * n * spread)) ** 2))


def block_mean(a, r):
    n = a.shape[0]
    k = n // r
    return a.reshape(r, k, r, k).mean(axis=(1, 3))


def build_maps(img, seed=21):
    """maps[stage][channel] with edge selectivity fading into part selectivity."""
    rng = np.random.default_rng(seed)
    n = img.shape[0]
    angles = [0, 90, 45, 135]
    keys = ["head", "torso", "legs", "context"]
    maps = []
    for s, (name, res) in enumerate(STAGES):
        frac = s / (len(STAGES) - 1.0)
        w_edge = (1.0 - frac) ** 1.3
        w_part = frac ** 0.85
        sigma = 0.7 + 3.4 * frac
        row = []
        for c in range(4):
            e = oriented_energy(img, angles[c], sigma)
            e = (e - e.mean()) / (e.std() + 1e-9)
            p = part_blob(n, keys[c], spread=0.75 + 0.9 * frac)
            p = (p - p.mean()) / (p.std() + 1e-9)
            m = (w_edge * e + w_part * p
                 + 0.22 * (1.0 - frac) * _noise(rng, n, 1.2 + 2 * frac))
            # Shallow stages fire almost everywhere; lift their floor so the
            # texture response stays visible on the shared scale.
            m = np.clip(m + 0.32 * (1.0 - frac), 0, None)
            row.append(block_mean(m, res))
        maps.append(row)

    for s, row in enumerate(maps):                 # designed shared scaling
        hi = max(m.max() for m in row) + 1e-9
        maps[s] = [m * (PEAKS[s] / hi) for m in row]
    return maps


def image_axes(ax, color="#c9c9c9", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def main():
    set_pub_style(base_font=8)
    img = synth_subject()
    maps = build_maps(img)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 5.70))
    left, right, bottom, top = 0.085, 0.895, 0.1552, 0.8350
    gs = fig.add_gridspec(4, 6, left=left, right=right, top=top,
                          bottom=bottom, wspace=0.040, hspace=0.040)

    for r in range(4):
        for c, (name, res) in enumerate(STAGES):
            ax = fig.add_subplot(gs[r, c])
            image_axes(ax)
            ax.imshow(maps[c][r], cmap=FEAT, norm=norm,
                      interpolation="nearest")
            if c == 0:
                ax.set_ylabel(f"Channel {r + 1}", fontsize=7.0, labelpad=4)
            if r == 3:
                ax.text(0.5, -0.075, name, transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.6,
                        fontweight="bold")
                ax.text(0.5, -0.225, f"{res} x {res}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.6, color="#666666")

    ax_in = fig.add_axes([left, 0.848, 0.0861, 0.1088])
    image_axes(ax_in, color="#8a8a8a", lw=0.7)
    ax_in.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax_in.text(1.14, 0.5, "input image, 96 x 96", transform=ax_in.transAxes,
               ha="left", va="center", fontsize=6.8, color="#4a4a4a")

    arrow = FancyArrowPatch((left, 0.052), (right, 0.052),
                            transform=fig.transFigure, arrowstyle="-|>",
                            mutation_scale=9, linewidth=0.9,
                            color="#9a9a9a", shrinkA=0, shrinkB=0)
    arrow.set_zorder(-5)
    fig.add_artist(arrow)

    cax = fig.add_axes([0.908, bottom, 0.014, top - bottom])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=FEAT), cax=cax,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("normalised channel activation (-)", fontsize=7.0,
                 color="#1a1a1a", labelpad=3.5)


    savefig(fig, "275_feature_evolution.png")


if __name__ == "__main__":
    main()
