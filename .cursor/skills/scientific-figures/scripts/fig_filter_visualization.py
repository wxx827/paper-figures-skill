"""Figure 277 - What the filters themselves look like.

Left: all 64 kernels of the first convolution, rendered as RGB tiles under a
single shared normalisation, so the relative weight energy of a filter is
readable and not an artefact of per-tile auto-contrast.  Right: twelve deep
units visualised by activation maximisation, i.e. the input that drives the
unit hardest.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

K = 11        # kernel raster size
V = 72        # activation-maximisation raster size


def gabor(n, ang, freq, phase, sigma, aspect=1.0):
    g = np.linspace(-1, 1, n)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    t = np.deg2rad(ang)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    env = np.exp(-(u ** 2 + (aspect * v) ** 2) / (2 * sigma ** 2))
    return env * np.sin(2 * np.pi * freq * u + phase)


def radial(n, freq, phase, sigma):
    g = np.linspace(-1, 1, n)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(xx, yy)
    return np.exp(-r ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * freq * r
                                                       + phase)


def build_kernels(seed=17, n=K):
    """64 first-layer kernels: oriented edges, colour opponency, blobs."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(64):
        kind = ("edge" if i < 34 else "colour" if i < 50
                else "blob" if i < 58 else "lowfreq")
        if kind == "edge":
            ang = rng.uniform(0, 180)
            f = rng.uniform(0.55, 1.75)
            k = gabor(n, ang, f, rng.uniform(0, 2 * np.pi),
                      rng.uniform(0.42, 0.66), rng.uniform(0.6, 1.6))
            tint = 0.10 * rng.standard_normal(3)
            rgb = np.stack([k * (1 + tint[c]) for c in range(3)], -1)
        elif kind == "colour":
            ang = rng.uniform(0, 180)
            f = rng.uniform(0.30, 0.95)
            base = gabor(n, ang, f, rng.uniform(0, 2 * np.pi),
                         rng.uniform(0.50, 0.72), rng.uniform(0.7, 1.3))
            pair = rng.integers(0, 3)
            w = np.array([-0.55, -0.55, -0.55])
            w[pair] = 1.0
            w[(pair + rng.integers(1, 3)) % 3] += 0.85
            rgb = base[..., None] * w[None, None, :]
        elif kind == "blob":
            k = radial(n, rng.uniform(0.45, 1.10), rng.uniform(0, np.pi),
                       rng.uniform(0.40, 0.60))
            w = 0.55 + 0.75 * rng.random(3)
            w[rng.integers(0, 3)] *= -1.0
            rgb = k[..., None] * w[None, None, :]
        else:
            g = np.linspace(-1, 1, n)
            yy, xx = np.meshgrid(g, g, indexing="ij")
            t = np.deg2rad(rng.uniform(0, 180))
            ramp = xx * np.cos(t) + yy * np.sin(t)
            w = rng.standard_normal(3)
            rgb = ramp[..., None] * w[None, None, :] * 0.6
        rgb = rgb * rng.uniform(0.55, 1.0)          # real filters differ in energy
        out.append(rgb)
    stack = np.array(out)
    scale = np.percentile(np.abs(stack), 99.5)      # one shared normalisation
    return np.clip(stack / (2 * scale) + 0.5, 0, 1)


PALETTE_SETS = [
    ((0.95, 0.55, 0.20), (0.15, 0.30, 0.62)),
    ((0.20, 0.68, 0.45), (0.62, 0.18, 0.45)),
    ((0.90, 0.82, 0.25), (0.25, 0.35, 0.72)),
    ((0.85, 0.30, 0.25), (0.20, 0.62, 0.68)),
    ((0.55, 0.35, 0.75), (0.85, 0.72, 0.35)),
    ((0.25, 0.55, 0.85), (0.92, 0.48, 0.30)),
]


def feature_viz(seed, style, n=V):
    """A DeepDream-like optimal input: warped gratings under a soft envelope."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, n)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    warp = 0.28 * gaussian_filter(rng.standard_normal((n, n)), 9.0)
    warp = warp / (warp.std() + 1e-9)

    field = np.zeros((n, n))
    if style == "stripes":
        for _ in range(3):
            t = np.deg2rad(rng.uniform(0, 180))
            u = xx * np.cos(t) + yy * np.sin(t) + 0.20 * warp
            field += np.sin(2 * np.pi * rng.uniform(2.5, 5.0) * u
                            + rng.uniform(0, 6.3))
    elif style == "rings":
        cx, cy = rng.uniform(-0.3, 0.3, 2)
        r = np.hypot(xx - cx, yy - cy) + 0.10 * warp
        field += 1.6 * np.cos(2 * np.pi * rng.uniform(2.5, 4.5) * r)
        for _ in range(2):                       # a couple of satellite eyes
            ex, ey = rng.uniform(-0.8, 0.8, 2)
            rr = np.hypot(xx - ex, yy - ey)
            field += 1.1 * np.cos(2 * np.pi * 3.4 * rr) * np.exp(-rr ** 2 / 0.10)
    elif style == "lattice":
        for t_deg in (0, 60, 120):
            t = np.deg2rad(t_deg + rng.uniform(-12, 12))
            u = xx * np.cos(t) + yy * np.sin(t) + 0.14 * warp
            field += np.sin(2 * np.pi * rng.uniform(3.0, 5.5) * u)
    elif style == "curls":
        ang = np.arctan2(yy, xx)
        r = np.hypot(xx, yy)
        field += 1.5 * np.sin(6.0 * ang + 7.0 * r + 1.6 * warp)
        field += 0.9 * np.sin(2 * np.pi * 3.0 * r + 2.0 * warp)
    else:                                         # "fur"
        base = gaussian_filter(rng.standard_normal((n, n)), 1.1)
        gy, gx = np.gradient(gaussian_filter(base, 2.2))
        flow = np.arctan2(gy, gx)
        field += 1.4 * np.sin(14.0 * (xx * np.cos(flow) + yy * np.sin(flow))
                              + 3.0 * warp)

    field += 0.35 * gaussian_filter(rng.standard_normal((n, n)), 2.0) * 3.0
    field = gaussian_filter(field, 0.7)
    field = (field - field.mean()) / (field.std() + 1e-9)
    t = np.clip(field / 3.2 + 0.5, 0, 1)

    c1, c2 = PALETTE_SETS[rng.integers(0, len(PALETTE_SETS))]
    rgb = (np.array(c2)[None, None, :] * (1 - t)[..., None]
           + np.array(c1)[None, None, :] * t[..., None])
    hi = 0.45 + 0.55 * np.clip(np.abs(field) / 2.6, 0, 1)
    rgb = rgb * hi[..., None] + 0.16 * (1 - hi)[..., None]

    env = np.exp(-(xx ** 2 + yy ** 2) ** 1.4 / 1.9)     # receptive-field falloff
    rgb = rgb * env[..., None] + 0.965 * (1 - env)[..., None]
    return np.clip(rgb, 0, 1)


def tile_axes(ax, color="#c9c9c9", lw=0.4):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def main():
    set_pub_style(base_font=8)
    kernels = build_kernels()

    styles = ["stripes", "rings", "lattice", "curls", "fur", "rings",
              "lattice", "stripes", "curls", "fur", "rings", "lattice"]
    units = [37, 112, 209, 288, 341, 405, 452, 511, 587, 640, 703, 766]
    vizzes = [feature_viz(900 + 31 * i, s) for i, s in enumerate(styles)]

    fig = plt.figure(figsize=(7.2, 3.75))
    l_left, l_w = 0.045, 2.55 / 7.2
    l_bot, l_h = 0.130, 2.55 / 3.75
    r_left, r_w = 0.4742, 3.30 / 7.2
    r_h = 2.465 / 3.75
    r_bot = l_bot + l_h - r_h

    gsk = fig.add_gridspec(8, 8, left=l_left, right=l_left + l_w,
                           bottom=l_bot, top=l_bot + l_h,
                           wspace=0.10, hspace=0.10)
    for i in range(64):
        ax = fig.add_subplot(gsk[i // 8, i % 8])
        tile_axes(ax, color="#d2d2d2", lw=0.35)
        ax.imshow(kernels[i], interpolation="nearest")

    gsv = fig.add_gridspec(3, 4, left=r_left, right=r_left + r_w,
                           bottom=r_bot, top=r_bot + r_h,
                           wspace=0.05, hspace=0.05)
    for i in range(12):
        ax = fig.add_subplot(gsv[i // 4, i % 4])
        tile_axes(ax)
        ax.imshow(vizzes[i], interpolation="bilinear")
        ax.text(0.04, 0.04, f"unit {units[i]}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.0, zorder=6,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="#b8b8b8", linewidth=0.35, alpha=0.90))

    ytit = l_bot + l_h + 0.048
    fig.text(l_left, ytit, "(a)  conv1 kernels, 11 x 11 x 3, all 64",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")
    fig.text(r_left, ytit,
             "(b)  Activation maximisation, conv5 units",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")

    fig.text(l_left, l_bot - 0.030,
             "oriented edges, colour opponency and centre-surround blobs;\n"
             "one shared weight normalisation across all 64 tiles",
             fontsize=6.4, color="#6a6a6a", ha="left", va="top")
    fig.text(r_left, l_bot - 0.030,
             "gradient ascent on the unit's response from noise, with jitter\n"
             "and total-variation regularisation; 72 x 72 crop shown",
             fontsize=6.4, color="#6a6a6a", ha="left", va="top")


    savefig(fig, "277_filter_visualization.png")


if __name__ == "__main__":
    main()
