"""Figure 542 - Tile-level tissue classification across a whole section.

The main panel is the section rendered in greyscale with the five-class
prediction laid over it at 45 % opacity, so the reader can still read the
histology under the map and check the classifier against it.  The two
subordinate panels (58 % of the main panel) carry the evidence: how much of
the section each class occupies, and how confident the model was.

Classification runs on 120 um tiles, which is why the map is blocky - that
blockiness is the honest resolution of the prediction, not an artefact.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, style_axes
from figkit import image_axes, add_scale_bar, halo_text, panel_labels_inches

OD_H = np.array([0.650, 0.704, 0.286])
OD_E = np.array([0.092, 0.954, 0.283])
GLASS = np.array([0.988, 0.982, 0.988])

# Five tissue classes, house qualitative order (DESIGN_SYSTEM section 2.1).
CLASSES = ["Tumour", "Stroma", "Necrosis", "Normal gland", "Lymphocyte"]
CLASS_COLORS = ["#b23a48", "#3b5b92", "#c9a227", "#2e7d5b", "#8e6bbf"]
MPP = 10.0                                  # 1x overview, micrometres/pixel
PATCH = 12                                  # 120 um classification tile


def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales, weights):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _blob(shape, fy, fx, ry, rx):
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    return np.exp(-0.5 * (((yy - fy * hh) / (ry * hh)) ** 2
                          + ((xx - fx * ww) / (rx * ww)) ** 2))


def he_rgb(ch, ce, tissue, desat=0.10):
    od = (np.clip(ch, 0, None)[..., None] * OD_H
          + np.clip(ce, 0, None)[..., None] * OD_E)
    rgb = np.exp(-od)
    lum = rgb @ np.array([0.30, 0.59, 0.11])
    rgb = rgb * (1.0 - desat) + lum[..., None] * desat
    t = np.clip(tissue, 0, 1)[..., None]
    return np.clip(rgb * t + GLASS * (1.0 - t), 0, 1)


def build_section(shape=(520, 600), seed=23):
    """Tissue mask plus per-pixel membership for the five classes."""
    rng = np.random.default_rng(seed)
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    cy, cx = 0.50 * hh, 0.50 * ww
    th = np.arctan2(yy - cy, xx - cx)
    rr = (np.hypot((yy - cy) / (0.435 * hh), (xx - cx) / (0.450 * ww))
          + 0.032 * _noise(rng, shape, 6.0))
    edge = (1.0 + 0.135 * np.sin(3 * th - 1.1) + 0.100 * np.sin(5 * th + 0.6)
            + 0.055 * np.sin(9 * th + 2.0) + 0.035 * np.sin(14 * th))
    tissue = np.clip((edge - rr) / 0.045, 0, 1)

    f1 = _fbm(rng, shape, (7, 18, 40), (0.44, 0.34, 0.22))
    nests = (1.10 * _blob(shape, 0.36, 0.38, 0.150, 0.165)
             + 0.92 * _blob(shape, 0.62, 0.63, 0.120, 0.135)
             + 0.60 * _blob(shape, 0.74, 0.33, 0.080, 0.090))
    p_tum = np.clip((nests + 0.40 * f1 - 0.46) / 0.32, 0, 1) * tissue

    core = np.clip((1.10 * _blob(shape, 0.36, 0.38, 0.150, 0.165) - 0.50)
                   / 0.36, 0, 1)
    p_nec = np.clip(core * (0.82 + 0.38 * _fbm(rng, shape, (7, 16), (.6, .4)))
                    - 0.08, 0, 1) * tissue

    inner = (gaussian_filter(tissue, 9.0) > 0.94).astype(float)
    rim = np.clip(4.0 * p_tum * (1.0 - p_tum) - 0.30, 0, 1)
    spots = np.clip(_fbm(rng, shape, (4.0, 9.0), (0.55, 0.45)) - 0.28, 0, 1)
    p_lym = gaussian_filter(np.clip(4.5 * rim * spots, 0, 1) * inner, 2.4)
    p_lym /= max(p_lym.max(), 1e-9)

    gl = _blob(shape, 0.30, 0.78, 0.150, 0.130) + _blob(shape, 0.72, 0.82,
                                                        0.110, 0.095)
    p_nor = (np.clip((gl + 0.35 * _fbm(shape=shape, rng=rng, scales=(5, 13),
                                       weights=(.6, .4)) - 0.52) / 0.28, 0, 1)
             * (1.0 - p_tum) * tissue)

    p_str = np.clip(1.0 - p_tum - p_nec - p_lym - p_nor, 0, 1) * tissue
    return dict(tissue=tissue, p=np.stack([p_tum, p_str, p_nec, p_nor,
                                           p_lym]))


def render_section(sec, seed=31):
    rng = np.random.default_rng(seed)
    shape = sec["tissue"].shape
    p_t, p_s, p_n, p_g, p_l = sec["p"]
    ch = (0.13 + 0.66 * p_t + 1.30 * p_l + 0.34 * p_g) * (1.0 - 0.86 * p_n)
    ch *= 1.0 + 0.22 * _fbm(rng, shape, (0.8, 2.6, 7.0), (.45, .33, .22))
    ce = (0.58 - 0.19 * p_t - 0.25 * p_l - 0.06 * p_g) * (1.0 - 0.34 * p_n)
    ce *= 1.0 + 0.14 * _fbm(rng, shape, (1.4, 5.0, 14.0), (.42, .34, .24))
    uneven = 1.0 + 0.12 * _noise(rng, shape, 55.0)
    rgb = he_rgb(ch * uneven, ce * uneven, sec["tissue"])
    return np.dstack([gaussian_filter(rgb[..., c], 0.55) for c in range(3)])


def classify(sec, seed=57, temp=3.4, jitter=0.52):
    """Tile-level posterior over the five classes; returns map + confidence."""
    rng = np.random.default_rng(seed)
    hh, ww = sec["tissue"].shape
    ny, nx = hh // PATCH, ww // PATCH

    def pool(a):
        return a[:ny * PATCH, :nx * PATCH].reshape(
            ny, PATCH, nx, PATCH).mean(axis=(1, 3))

    cov = pool(sec["tissue"])
    logits = temp * np.stack([pool(a) for a in sec["p"]])
    logits += jitter * rng.standard_normal(logits.shape)
    logits -= logits.max(axis=0, keepdims=True)
    post = np.exp(logits)
    post /= post.sum(axis=0, keepdims=True)
    pred = post.argmax(axis=0)
    conf = post.max(axis=0)
    valid = cov >= 0.5
    return pred, conf, valid


def main():
    set_pub_style(base_font=8)
    shape = (520, 600)
    sec = build_section(shape, seed=23)
    rgb = render_section(sec, seed=31)
    pred, conf, valid = classify(sec, seed=57)

    grey = 0.18 + 0.80 * (rgb @ np.array([0.30, 0.59, 0.11]))
    base = np.dstack([grey] * 3)
    lut = np.array([to_rgb(c) for c in CLASS_COLORS])
    big = np.kron(pred, np.ones((PATCH, PATCH), int))
    ok = np.kron(valid, np.ones((PATCH, PATCH), bool))
    over = base.copy()
    hh, ww = big.shape
    sub = over[:hh, :ww]
    sub[ok] = sub[ok] * 0.55 + lut[big[ok]] * 0.45
    over[:hh, :ww] = sub

    patch_mm2 = (PATCH * MPP / 1000.0) ** 2
    counts = np.array([int(((pred == k) & valid).sum())
                       for k in range(len(CLASSES))])
    areas = counts * patch_mm2
    share = 100.0 * areas / areas.sum()
    low = 100.0 * (conf[valid] < 0.60).mean()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86, 0.11], wspace=0.16,
                          hspace=0.34, left=0.048, right=0.975, top=0.878,
                          bottom=0.055)
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_cnf = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    # ---- (a) prediction over greyscale histology --------------------------
    image_axes(ax_map, color="#bdbdbd", lw=0.6)
    ax_map.imshow(over, interpolation="nearest", zorder=1)
    add_scale_bar(ax_map, 1000.0 / MPP, "1 mm", loc=(0.032, 0.042),
                  height=0.016, fontsize=6.4)
    halo_text(ax_map, 0.978, 0.972,
              f"{int(valid.sum()):,} classified tiles of 120 µm    "
              f"{areas.sum():.1f} mm² tissue",
              transform=ax_map.transAxes, ha="right", va="top", fontsize=6.6,
              color="#333333", halo="white", halo_lw=2.2)
    ax_map.set_title("Five-class prediction over the greyscale section  "
                     "(45 % opacity)", fontsize=8.8, fontweight="bold",
                     pad=4.0)

    # ---- (b) area shares ---------------------------------------------------
    order = np.argsort(-share)
    start = 90.0
    for i in order:
        ext = 360.0 * share[i] / 100.0
        ax_pie.add_patch(Wedge((0, 0), 1.0, start - ext, start, width=0.42,
                               facecolor=CLASS_COLORS[i], edgecolor="white",
                               linewidth=0.9))
        mid = np.deg2rad(start - ext / 2.0)
        # Small wedges are labelled in the shared legend instead - leader
        # lines this short always collide with each other.
        if share[i] >= 6.0:
            ax_pie.text(0.79 * np.cos(mid), 0.79 * np.sin(mid),
                        f"{share[i]:.0f}%", ha="center", va="center",
                        fontsize=6.8, fontweight="bold", color="white")
        start -= ext
    ax_pie.text(0, 0.10, f"{areas.sum():.1f}", ha="center", va="center",
                fontsize=11.0, fontweight="bold", color="#1a1a1a")
    ax_pie.text(0, -0.16, "mm² tissue", ha="center", va="center",
                fontsize=6.5, color="#555555")
    ax_pie.set_xlim(-1.16, 1.16)
    ax_pie.set_ylim(-1.12, 1.12)
    ax_pie.set_aspect("equal")
    ax_pie.axis("off")
    ax_pie.set_title("Area share by class", fontsize=8.2, fontweight="bold",
                     pad=3.0)

    # ---- (c) confidence distribution ---------------------------------------
    grid = np.linspace(0.32, 1.0, 220)
    for k in range(len(CLASSES)):
        v = conf[valid & (pred == k)]
        if v.size < 12:
            continue
        dens = gaussian_kde(v, bw_method=0.32)(grid)
        dens = dens / dens.max() * 0.86
        y = len(CLASSES) - 1 - k
        ax_cnf.fill_between(grid, y, y + dens, color=CLASS_COLORS[k],
                            alpha=0.72, linewidth=0.0, zorder=3 + k)
        ax_cnf.plot(grid, y + dens, color=CLASS_COLORS[k], lw=0.8,
                    zorder=3 + k)
        ax_cnf.text(0.998, y + 0.58, f"median {np.median(v):.2f}",
                    fontsize=6.2, color="#555555", ha="right", va="center",
                    zorder=9)
    ax_cnf.axvline(0.60, color="#808080", lw=0.8, ls="--", zorder=2)
    ax_cnf.text(0.605, len(CLASSES) - 0.32, "review threshold 0.60",
                fontsize=6.2, color="#555555", ha="left", va="top")
    ax_cnf.set_yticks(np.arange(len(CLASSES)) + 0.30)
    ax_cnf.set_yticklabels(CLASSES[::-1], fontsize=6.5)
    ax_cnf.set_xlim(0.32, 1.005)
    ax_cnf.set_ylim(-0.12, len(CLASSES) + 0.05)
    ax_cnf.set_xlabel("softmax confidence (-)", fontsize=7.2, labelpad=1.8)
    ax_cnf.tick_params(labelsize=6.5, length=2.2, pad=1.6)
    ax_cnf.grid(axis="x", linewidth=0.4, alpha=0.6)
    style_axes(ax_cnf)
    ax_cnf.set_title(f"Tile confidence  ({low:.0f} % below threshold)",
                     fontsize=8.2, fontweight="bold", pad=3.0)

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.5,
                     label=f"{n}  ({s:.1f} %, {a:.2f} mm²)")
               for n, c, s, a in zip(CLASSES, CLASS_COLORS, share, areas)]
    leg = ax_leg.legend(handles=handles, loc="center", ncol=5, frameon=False,
                        fontsize=7.0, handlelength=1.2, handleheight=1.0,
                        columnspacing=1.4, borderaxespad=0.0)
    leg.set_clip_on(False)

    for ax in (ax_map, ax_pie, ax_cnf):
        ax.apply_aspect()
    panel_labels_inches(fig, [ax_map, ax_pie, ax_cnf], dx=-0.28, dy=0.09)

    savefig(fig, "542_path_tissue_classification.png")


if __name__ == "__main__":
    main()
