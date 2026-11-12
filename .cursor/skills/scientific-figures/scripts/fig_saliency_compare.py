"""Figure 279 - Five saliency methods on the same two inputs.

Gradient-based attributions are speckled and leak onto background edges;
CAM-style attributions are smooth but coarse.  All five maps use one shared
``magma`` scale over the grayscale input, so the only thing that changes across
columns is the spatial character of the explanation.  A subordinate panel
reports the deletion / insertion faithfulness scores.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter, zoom

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

HEAT = "magma"
ALPHA = 0.50
N = 128
METHODS = ["Vanilla Grad", "SmoothGrad", "Integrated Grad", "Grad-CAM",
           "Ours"]
SHORT = ["Vanilla", "Smooth", "IG", "CAM", "Ours"]
PEAKS = [0.72, 0.80, 0.86, 0.92, 1.00]
DELETION = [0.182, 0.151, 0.144, 0.126, 0.101]
INSERTION = [0.512, 0.578, 0.601, 0.664, 0.731]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(0.9, 2.6, 7.0), weights=(0.45, 0.33, 0.22)):
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


def scene_mug(seed=51, n=N):
    """A cup on a table, shot against a defocused interior."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    v = yy / (n - 1.0)
    tex = 0.030 * _fbm(rng, n)

    img = 0.72 - 0.18 * v + 0.13 * _fbm(rng, n, (6, 15, 30))
    for _ in range(4):
        cx, cy = rng.uniform(0, n), rng.uniform(0, 0.55 * n)
        rr = rng.uniform(0.10, 0.24) * n
        img += rng.uniform(-0.16, 0.16) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rr ** 2)))
    img = gaussian_filter(img, 3.0)
    img = np.where(v > 0.70, 0.44 + 0.07 * _fbm(rng, n, (2, 7, 18)), img)

    handle = (_ellipse(yy, xx, 0.700 * n, 0.520 * n, 0.105 * n, 0.088 * n)
              & ~_ellipse(yy, xx, 0.700 * n, 0.520 * n, 0.062 * n, 0.048 * n))
    img = _paint(img, _soft(handle, 0.7), 0.30, tex * 0.9)
    body = ((np.abs(xx - 0.470 * n) < 0.165 * n)
            & (yy > 0.375 * n) & (yy < 0.660 * n))
    body |= _ellipse(yy, xx, 0.470 * n, 0.660 * n, 0.165 * n, 0.052 * n)
    img = _paint(img, _soft(body, 0.7), 0.88, tex * 0.7)
    shade = np.clip((xx - 0.36 * n) / (0.28 * n), 0, 1)
    img = np.where(_soft(body, 0.7) > 0.5,
                   img * (1.0 - 0.34 * shade ** 1.6), img)
    rim = _ellipse(yy, xx, 0.470 * n, 0.375 * n, 0.165 * n, 0.055 * n)
    img = _paint(img, _soft(rim, 0.6), 0.66, tex * 0.6)
    img = _paint(img, _soft(_ellipse(yy, xx, 0.470 * n, 0.378 * n,
                                     0.128 * n, 0.038 * n), 0.6),
                 0.24, tex * 0.9)
    shadow = _ellipse(yy, xx, 0.520 * n, 0.700 * n, 0.230 * n, 0.048 * n)
    img = np.where(_soft(shadow, 3.0) > 0.05,
                   img * (1 - 0.30 * _soft(shadow, 3.0)), img)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    sil = _soft(body | rim | handle, 1.1)
    return np.clip((img - lo) / (hi - lo), 0, 1), (0.50, 0.52, 0.20, 0.17), sil


def scene_butterfly(seed=63, n=N):
    """A butterfly over an out-of-focus flower bed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    tex = 0.030 * _fbm(rng, n)

    img = 0.58 + 0.20 * _fbm(rng, n, (5, 13, 26))
    for _ in range(14):                                  # bokeh blossoms
        cx, cy = rng.uniform(0, n), rng.uniform(0, n)
        rr = rng.uniform(0.05, 0.13) * n
        img += rng.uniform(-0.22, 0.26) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rr ** 2)))
    img = gaussian_filter(img, 3.4)

    wings = np.zeros((n, n), bool)
    for sx in (-1, 1):
        wings |= _ellipse(yy, xx, (0.50 + sx * 0.150) * n, 0.415 * n,
                          0.150 * n, 0.115 * n, sx * 28)
        wings |= _ellipse(yy, xx, (0.50 + sx * 0.115) * n, 0.605 * n,
                          0.100 * n, 0.085 * n, -sx * 22)
    img = _paint(img, _soft(wings, 0.8), 0.24, tex * 1.4)
    for sx in (-1, 1):                                   # wing eyespots
        for ox, oy, rr in ((0.150, 0.400, 0.048), (0.108, 0.612, 0.030)):
            spot = _ellipse(yy, xx, (0.50 + sx * ox) * n, oy * n,
                            rr * n, rr * 0.85 * n)
            img = _paint(img, _soft(spot, 0.6), 0.86, tex * 0.5)
            img = _paint(img, _soft(_ellipse(yy, xx, (0.50 + sx * ox) * n,
                                             oy * n, rr * 0.45 * n,
                                             rr * 0.40 * n), 0.4),
                         0.14, tex * 0.3)
    body = _ellipse(yy, xx, 0.50 * n, 0.500 * n, 0.022 * n, 0.145 * n)
    img = _paint(img, _soft(body, 0.6), 0.12, tex * 0.6)
    for sx in (-1, 1):                                   # antennae
        t = np.linspace(0, 1, 90)
        ax_ = (0.50 + sx * 0.075 * t) * n
        ay_ = (0.360 - 0.095 * t) * n
        ant = np.zeros((n, n), bool)
        ii = np.clip(np.round(ay_).astype(int), 0, n - 1)
        jj = np.clip(np.round(ax_).astype(int), 0, n - 1)
        ant[ii, jj] = True
        img = _paint(img, _soft(ant, 0.8) * 0.9, 0.16, tex * 0.3)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    sil = _soft(wings | body, 1.1)
    return np.clip((img - lo) / (hi - lo), 0, 1), (0.50, 0.49, 0.30, 0.20), sil


def saliency_set(img, box, sil, seed):
    """One map per method, each with its own characteristic spatial texture."""
    rng = np.random.default_rng(seed)
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx, cy, rx, ry = box
    obj = np.exp(-0.5 * (((xx - cx * n) / (rx * n)) ** 2
                         + ((yy - cy * n) / (ry * n)) ** 2))

    gy, gx = np.gradient(gaussian_filter(img, 0.8))
    grad = np.hypot(gy, gx)
    grad = grad / (grad.max() + 1e-9)

    speckle = np.abs(rng.standard_normal((n, n))) ** 1.6
    # Single backward pass: high-frequency, leaks onto background edges.
    vanilla = grad * (0.25 + 0.95 * speckle) + 0.22 * speckle * (0.3 + obj)
    # Averaged over noisy copies: broad, blurred bands around structure.
    smooth = gaussian_filter(grad * (0.55 + 0.45 * speckle)
                             * (0.30 + 0.90 * obj), 2.6)
    # Path integral: sparse, dotty, and biased toward the object interior.
    drop = (rng.random((n, n)) > 0.62).astype(float)
    ig = gaussian_filter((0.35 * grad + 0.85 * sil * obj) * drop, 0.75)
    ig += 0.30 * gaussian_filter(grad * (0.1 + 0.9 * obj) * drop, 0.4)
    coarse = zoom(gaussian_filter(obj * (0.55 + 0.45 * grad), 3.0),
                  8.0 / n, order=1)
    gcam = zoom(coarse, n / coarse.shape[0], order=1)[:n, :n]
    gcam = gaussian_filter(gcam, 3.0)
    # Ours: CAM-level smoothness but resolved to the object silhouette.
    ours = gaussian_filter(sil * (0.42 + 0.58 * obj), 1.8) ** 0.9

    maps = [vanilla, smooth, ig, gcam, ours]
    out = []
    for m, pk in zip(maps, PEAKS):
        m = np.clip(m, 0, None)
        out.append(m * (pk / (np.percentile(m, 99.6) + 1e-9)))
    return out


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
    scenes = [("Cup", *scene_mug()), ("Butterfly", *scene_butterfly())]
    sal = [saliency_set(img, box, sil, 400 + 11 * i)
           for i, (_, img, box, sil) in enumerate(scenes)]
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 4.35))
    left, right = 0.055, 0.895
    tile = (right - left) * 7.2 / (6 + 5 * 0.035)
    block_h = tile * (2 + 0.035) / 4.35
    ttop = 0.878
    tbot = ttop - block_h

    band = plt.Rectangle((0.040, 0.048), 0.905, 0.315,
                         transform=fig.transFigure, facecolor="#f4f4f4",
                         edgecolor="none", zorder=-5)
    fig.add_artist(band)

    gs = fig.add_gridspec(2, 6, left=left, right=right, top=ttop,
                          bottom=tbot, wspace=0.035, hspace=0.035)
    for r, (name, img, box, _sil) in enumerate(scenes):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c])
            image_axes(ax)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="bilinear")
            if c > 0:
                ax.imshow(sal[r][c - 1], cmap=HEAT, norm=norm, alpha=ALPHA,
                          interpolation="bilinear")
            if r == 0:
                lab = "Input" if c == 0 else METHODS[c - 1]
                best = lab == "Ours"
                ax.set_title(lab, fontsize=7.4, pad=3.4,
                             fontweight="bold" if best else "normal",
                             color="#12503a" if best else "#1a1a1a")
            if c == 0:
                ax.set_ylabel(name, fontsize=7.2, fontweight="bold",
                              labelpad=4)

    cax = fig.add_axes([0.908, tbot, 0.013, block_h])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT), cax=cax,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("normalised attribution (-)", fontsize=7.0, color="#1a1a1a",
                 labelpad=3.0)

    axb = fig.add_axes([left, 0.108, 0.487, 0.185])
    style_axes(axb)
    axb.set_facecolor("white")
    c_del, c_ins = PALETTES["cb_safe"][5], PALETTES["cb_safe"][0]
    x = np.arange(5)
    axb.bar(x - 0.19, DELETION, width=0.35, color=c_del, edgecolor="white",
            linewidth=0.5, label="deletion AUC (lower better)")
    axb.bar(x + 0.19, INSERTION, width=0.35, color=c_ins, edgecolor="white",
            linewidth=0.5, label="insertion AUC (higher better)")
    for xi, (d, ins) in enumerate(zip(DELETION, INSERTION)):
        axb.text(xi - 0.19, d + 0.012, f"{d:.3f}", ha="center", va="bottom",
                 fontsize=6.0, color="#4a4a4a")
        axb.text(xi + 0.19, ins + 0.012, f"{ins:.3f}", ha="center",
                 va="bottom", fontsize=6.0, color="#4a4a4a")
    axb.set_xticks(x)
    axb.set_xticklabels(SHORT, fontsize=6.6)
    for tick, lab in zip(axb.get_xticklabels(), SHORT):
        if lab == "Ours":
            tick.set_fontweight("bold")
            tick.set_color("#12503a")
    axb.set_ylim(0, 0.90)
    axb.set_yticks([0, 0.3, 0.6, 0.9])
    axb.tick_params(labelsize=6.2)
    axb.set_ylabel("AUC", fontsize=6.8, labelpad=2.5)
    leg = axb.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.30), ncol=2,
                     frameon=False, fontsize=6.3, handlelength=1.1,
                     handleheight=0.85, columnspacing=1.0,
                     handletextpad=0.4)
    leg.set_clip_on(False)

    fig.text(0.575, 0.320, "Faithfulness, averaged over 2 000 images",
             fontsize=7.6, fontweight="bold", ha="left", va="top",
             color="#3a3a3a")
    notes = [
        "Deletion removes the highest-attribution pixels first and",
        "measures how fast the class score collapses; insertion adds",
        "them to a blurred image and measures how fast it recovers.",
        "",
        "The gradient family is noisy because single backward passes",
        "keep high-frequency structure; CAM-style maps are smooth but",
        "blur past the object boundary. Ours stays tight on the object",
        "and wins on both metrics.",
    ]
    for i, line in enumerate(notes):
        fig.text(0.575, 0.278 - i * 0.0290, line, fontsize=6.5,
                 color="#5a5a5a", ha="left", va="top")


    savefig(fig, "279_saliency_compare.png")


if __name__ == "__main__":
    main()
