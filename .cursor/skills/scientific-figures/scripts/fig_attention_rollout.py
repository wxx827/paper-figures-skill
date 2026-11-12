"""Figure 278 - Attention rollout through a 12-block ViT.

Class-token attention starts almost uniform, concentrates block by block, and
is finally accumulated across all twelve blocks by rollout.  The subordinate
panel underneath tracks the entropy of the same distributions, measured on the
maps that are actually drawn above.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

HEAT = "magma"
ALPHA = 0.50
P = 16
N = 128
VMAX = 5.0
SHOWN = [1, 4, 8, 12]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(0.9, 2.6, 7.0), weights=(0.45, 0.33, 0.22)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _soft(mask, sigma=0.8):
    return np.clip(gaussian_filter(mask.astype(float), sigma), 0, 1)


def _paint(img, mask, value, texture):
    return img * (1 - mask) + mask * np.clip(value + texture, 0.02, 0.99)


def _triangle(yy, xx, p0, p1, p2, n):
    pts = [np.array(p) * n for p in (p0, p1, p2)]
    inside = np.ones(yy.shape, bool)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        side = ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]))
        inside &= side >= 0
    if not inside.any():                     # winding order was reversed
        inside = np.ones(yy.shape, bool)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            side = ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]))
            inside &= side <= 0
    return inside


def sailboat(seed=29, n=N):
    """Sky, textured water, a hull and two sails - one clear salient object."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    v = yy / (n - 1.0)
    tex = 0.028 * _fbm(rng, n)

    img = 0.95 - 0.32 * np.clip(v / 0.60, 0, 1) ** 0.9
    img += 0.05 * _fbm(rng, n, (7, 18, 34))
    for _ in range(4):                                        # soft clouds
        cx, cy = rng.uniform(0, n), rng.uniform(0.05, 0.35) * n
        rr = rng.uniform(0.10, 0.22) * n
        img += 0.10 * np.exp(-(((xx - cx) / (1.9 * rr)) ** 2
                               + ((yy - cy) / rr) ** 2))
    water = v > 0.615
    wav = (0.50 + 0.11 * np.sin(yy / 2.1 + 2.0 * np.sin(xx / 13.0))
           + 0.07 * _fbm(rng, n, (1.2, 3.5)))
    img = np.where(water, wav, gaussian_filter(img, 1.8))

    hull = _triangle(yy, xx, (0.305, 0.640), (0.700, 0.640), (0.560, 0.735), n)
    hull |= (((xx - 0.50 * n) / (0.196 * n)) ** 2
             + ((yy - 0.652 * n) / (0.038 * n)) ** 2) < 1
    hull &= yy > 0.628 * n
    main_sail = _triangle(yy, xx, (0.512, 0.170), (0.512, 0.625),
                          (0.760, 0.625), n)
    jib = _triangle(yy, xx, (0.496, 0.205), (0.496, 0.625),
                    (0.300, 0.625), n)
    mast = (np.abs(xx - 0.504 * n) < 0.010 * n) & (yy > 0.150 * n) \
        & (yy < 0.645 * n)

    img = _paint(img, _soft(main_sail, 0.7), 0.93, tex * 0.8)
    img = _paint(img, _soft(jib, 0.7), 0.80, tex * 0.9)
    img = _paint(img, _soft(mast, 0.5), 0.22, tex * 0.4)
    img = _paint(img, _soft(hull, 0.7), 0.18, tex * 1.1)
    refl = ((np.abs(xx - 0.50 * n) < 0.17 * n) & (yy > 0.735 * n)
            & (yy < 0.86 * n))
    img = _paint(img, _soft(refl, 2.0) * 0.45, 0.34, tex * 1.4)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo), 0, 1), (0.512, 0.435, 0.175, 0.240)


def build_blocks(box, seed=8):
    """Twelve class-token maps that tighten onto the boat, plus the rollout."""
    rng = np.random.default_rng(seed)
    pr, pc = np.mgrid[0:P, 0:P].astype(float)
    cx, cy, rx, ry = box
    obj = np.exp(-0.5 * (((pc - cx * P) / (rx * P)) ** 2
                         + ((pr - cy * P) / (ry * P)) ** 2))

    blocks = []
    for i in range(12):
        f = i / 11.0
        w = f ** 1.05
        # The attended region also shrinks with depth, not just its weight.
        spread = 1.20 - 0.62 * f
        obj_i = np.exp(-0.5 * (((pc - cx * P) / (rx * P * spread)) ** 2
                               + ((pr - cy * P) / (ry * P * spread)) ** 2))
        clutter = 0.5 + 0.5 * _noise(rng, P, 1.6 + 1.4 * f)
        m = ((1 - w) * (0.55 + 0.45 * clutter) + 3.4 * w * obj_i
             + 0.14 * (1 - w) * np.exp(-((pr - P + 1) ** 2) / 6.0))
        m = np.clip(m, 1e-4, None)
        blocks.append(m / m.sum())

    roll = np.ones((P, P))
    for m in blocks:                       # accumulate through the depth
        roll = roll * (m * m.size + 0.35)
        roll = roll / roll.sum()
    return blocks, roll


def entropy_bits(p):
    p = p / p.sum()
    return float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())


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
    img, box = sailboat()
    blocks, roll = build_blocks(box)
    ent = [entropy_bits(m) for m in blocks]
    ent_roll = entropy_bits(roll)
    norm = Normalize(vmin=0.0, vmax=VMAX)

    fig = plt.figure(figsize=(7.2, 3.55))
    left, right = 0.055, 0.895
    tile = (right - left) * 7.2 / (6 + 5 * 0.035)
    th = tile / 3.55
    ttop, tbot = 0.860, 0.860 - th

    band = plt.Rectangle((0.040, 0.070), 0.905, 0.400,
                         transform=fig.transFigure, facecolor="#f4f4f4",
                         edgecolor="none", zorder=-5)
    fig.add_artist(band)

    gs = fig.add_gridspec(1, 6, left=left, right=right, top=ttop,
                          bottom=tbot, wspace=0.035)
    panels = [("Input", None)] + [(f"Block {b}", blocks[b - 1] * P * P)
                                  for b in SHOWN] + \
             [("Rollout", roll * P * P)]
    for c, (name, m) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        image_axes(ax)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
                  extent=[0, N, N, 0])
        if m is not None:
            ax.imshow(m, cmap=HEAT, norm=norm, alpha=ALPHA,
                      interpolation="nearest", extent=[0, N, N, 0])
            h = ent_roll if name == "Rollout" else ent[SHOWN[c - 1] - 1]
            ax.text(0.5, -0.055, f"H = {h:.2f} bits", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.3, color="#4a4a4a")
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        last = name == "Rollout"
        ax.set_title(name, fontsize=7.8, pad=3.4,
                     fontweight="bold" if last else "normal",
                     color="#12503a" if last else "#1a1a1a")

    cax = fig.add_axes([0.908, tbot, 0.013, th])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT), cax=cax,
                      ticks=[0, 1, 2, 3, 4, 5], extend="max")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("attention gain over uniform (x)", fontsize=7.0,
                 color="#1a1a1a", labelpad=3.0)

    axe = fig.add_axes([left, 0.140, 0.487, 0.240])
    style_axes(axe)
    axe.set_facecolor("white")
    blue = PALETTES["cb_safe"][0]
    axe.axhline(np.log2(P * P), color="#9a9a9a", lw=0.8, ls="--", zorder=1)
    axe.text(0.7, np.log2(P * P) + 0.08, "uniform attention", fontsize=6.2,
             color="#8a8a8a", ha="left", va="bottom")
    axe.plot(range(1, 13), ent, color=blue, lw=1.4, marker="o",
             markersize=3.0, markerfacecolor="white", markeredgewidth=1.0,
             zorder=3)
    for b in SHOWN:
        axe.plot([b], [ent[b - 1]], marker="o", markersize=4.4, color=blue,
                 zorder=4)
    axe.plot([13.0], [ent_roll], marker="*", markersize=8.5,
             color="#12503a", zorder=4, clip_on=False)
    axe.annotate("rollout", (13.0, ent_roll), textcoords="offset points",
                 xytext=(-2, 8), fontsize=6.4, color="#12503a",
                 fontweight="bold", ha="center")
    axe.set_xlim(0.4, 13.8)
    axe.set_ylim(min(ent_roll, min(ent)) - 0.55, np.log2(P * P) + 0.55)
    axe.set_xticks([1, 4, 8, 12])
    axe.set_xlabel("transformer block", fontsize=7.0, labelpad=2.0)
    axe.set_ylabel("attention entropy (bits)", fontsize=7.0, labelpad=2.5)
    axe.tick_params(labelsize=6.4)

    fig.text(0.575, 0.398, "Attention concentrates with depth",
             fontsize=7.6, fontweight="bold", ha="left", va="top",
             color="#3a3a3a")
    notes = [
        "Every map is the class token's attention over the 16 x 16",
        "patch grid, shown as gain over the uniform 1/256 baseline",
        "on one shared colour scale.",
        "",
        "Rollout multiplies the per-block matrices with a residual",
        "term, so only patches that stay attended all the way through",
        f"survive: entropy falls from {ent[0]:.2f} bits at block 1 to",
        f"{ent[-1]:.2f} at block 12, and {ent_roll:.2f} after rollout.",
    ]
    for i, line in enumerate(notes):
        fig.text(0.575, 0.352 - i * 0.0335, line, fontsize=6.5,
                 color="#5a5a5a", ha="left", va="top")


    savefig(fig, "278_attention_rollout.png")


if __name__ == "__main__":
    main()
