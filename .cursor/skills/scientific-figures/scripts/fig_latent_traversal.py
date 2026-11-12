"""Figure 282 - Latent traversal of a disentangled generative model.

One row per latent dimension.  The first column is the input; the remaining
seven decode the same code with that single dimension swept from -3 sigma to
+3 sigma.  The z = 0 column is the plain reconstruction and is framed in blue
as the reference every row should be compared against.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

REF_C = "#3b5b92"
NP_ = 88
ZS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
COL_LABELS = ["Input", r"$-3\sigma$", r"$-2\sigma$", r"$-1\sigma$", "0",
              r"$+1\sigma$", r"$+2\sigma$", r"$+3\sigma$"]
BG_TOP, BG_BOT = (0.94, 0.945, 0.955), (0.855, 0.865, 0.885)

# name, latent index, and the default -> swept parameter
DIMS = [("Rotation", 12), ("Scale", 4), ("Colour", 27),
        ("Thickness", 7), ("Style", 19)]
DEFAULTS = dict(angle=22.0, scale=1.00, hue=0.575, thick=0.40, morph=3.0)


def sprite(angle, scale, hue, thick, morph, n=NP_, blur=0.9, seed=0,
           grain=0.006):
    """A superellipse ring on a soft background - a dSprites-style decode."""
    g = np.linspace(-1, 1, n)
    yy, xx = np.meshgrid(g, g, indexing="ij")

    bg = np.empty((n, n, 3))
    ramp = (yy + 1) / 2.0
    for c in range(3):
        bg[..., c] = BG_TOP[c] + (BG_BOT[c] - BG_TOP[c]) * ramp
    bg = bg * (1.0 - 0.06 * (xx ** 2 + yy ** 2))[..., None]

    # The sprite is deliberately oblong, otherwise the rotation dimension
    # would be invisible on a shape with four-fold symmetry.
    t = np.deg2rad(angle)
    u = (xx * np.cos(t) + yy * np.sin(t)) / (0.63 * scale)
    v = (-xx * np.sin(t) + yy * np.cos(t)) / (0.42 * scale)
    p = float(np.clip(morph, 0.7, 8.0))
    r = (np.abs(u) ** p + np.abs(v) ** p) ** (1.0 / p)

    outer = np.clip((1.0 - r) / 0.055 + 0.5, 0, 1)
    ri = max(1.0 - float(np.clip(thick, 0.05, 1.0)), 0.0)
    inner = np.clip((ri - r) / 0.055 + 0.5, 0, 1) if ri > 0.02 else 0.0
    ring = np.clip(outer - inner, 0, 1)

    shade = 0.86 + 0.26 * np.clip((0.6 - (u * 0.5 + v * 0.7)) / 1.6, 0, 1)
    face = hsv_to_rgb([hue % 1.0, 0.70, 0.86])[None, None, :] * shade[..., None]
    edge = np.clip(gaussian_filter(ring, 1.1) - ring, 0, 1)
    dark = hsv_to_rgb([hue % 1.0, 0.80, 0.42])[None, None, :]

    shadow = np.clip(gaussian_filter(np.roll(np.roll(ring, 4, 0), 3, 1), 3.2),
                     0, 1) * 0.22
    img = bg * (1 - shadow[..., None])
    img = img * (1 - ring[..., None]) + face * ring[..., None]
    img = img * (1 - 0.55 * edge[..., None]) + dark * 0.55 * edge[..., None]

    img = gaussian_filter(img, (blur, blur, 0))
    if grain:
        rng = np.random.default_rng(seed)
        img = img + grain * rng.standard_normal((n, n, 1))
    return np.clip(img, 0, 1)


def params_for(dim, z):
    """Sweep exactly one factor; everything else stays at its default."""
    kw = dict(DEFAULTS)
    f = z / 3.0
    if dim == "Rotation":
        kw["angle"] = 22.0 + 78.0 * f
    elif dim == "Scale":
        kw["scale"] = 1.00 + 0.42 * f
    elif dim == "Colour":
        kw["hue"] = (0.575 + 0.46 * f) % 1.0
    elif dim == "Thickness":
        kw["thick"] = float(np.clip(0.46 + 0.44 * f, 0.09, 1.0))
    else:                                   # Style: diamond -> disc -> square
        kw["morph"] = float(np.exp(np.log(3.0) + 1.30 * f))
    return kw


def tile_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.45)
        s.set_color("#cfcfcf")


def main():
    set_pub_style(base_font=8)
    rows, cols = len(DIMS), 8
    ws = hs = 0.035

    fig = plt.figure(figsize=(7.2, 4.95))
    left, right = 0.135, 0.968
    tile = (right - left) * 7.2 / (cols + (cols - 1) * ws)
    gh = tile * (rows + (rows - 1) * hs) / 4.95
    gtop = 0.908
    gbot = gtop - gh

    gs = fig.add_gridspec(rows, cols, left=left, right=right, top=gtop,
                          bottom=gbot, wspace=ws, hspace=hs)

    for r, (name, zi) in enumerate(DIMS):
        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            tile_axes(ax)
            if c == 0:
                ax.imshow(sprite(**DEFAULTS, blur=0.35, seed=r,
                                 grain=0.014), interpolation="bilinear")
            else:
                kw = params_for(name, ZS[c - 1])
                ax.imshow(sprite(**kw, seed=10 * r + c),
                          interpolation="bilinear")
                if ZS[c - 1] == 0.0:
                    ax.add_patch(Rectangle((0, 0), 1, 1,
                                           transform=ax.transAxes, fill=False,
                                           edgecolor=REF_C, linewidth=1.3,
                                           zorder=8, clip_on=False))
            if c == 0:
                ax.set_ylabel(f"{name}   $z_{{{zi}}}$", fontsize=7.6,
                              fontweight="bold", rotation=0, ha="right",
                              va="center", labelpad=7)
            if r == rows - 1:
                is_ref = c == 4
                ax.text(0.5, -0.085, COL_LABELS[c], transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.0,
                        fontweight="bold" if is_ref or c == 0 else "normal",
                        color=REF_C if is_ref else "#3a3a3a")

    fig.text(left + (right - left) / 2, gbot - 0.082,
             "latent value along the swept dimension",
             fontsize=6.8, color="#7a7a7a", ha="center", va="top")
    fig.text(left, gbot - 0.135,
             "Column 1 is the encoder input; the blue-framed column is its "
             "plain reconstruction (z = 0).  All other latents are held fixed,\n"
             "so any change along a row is attributable to that single "
             "dimension.",
             fontsize=6.5, color="#5a5a5a", ha="left", va="top")


    savefig(fig, "282_latent_traversal.png")


if __name__ == "__main__":
    main()
