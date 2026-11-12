"""Figure 545 - Stain normalisation across three contributing centres.

Each row is one slide; the *tissue* in a row is identical across the columns,
because normalisation is allowed to change colour and nothing else.  What
differs between the three rows is the staining protocol: centre 1 runs a dark
haematoxylin, centre 2 a pale one with heavy eosin, centre 3 over-develops
both.  Each method pulls the slide's stain vectors and concentrations toward
the reference by a different amount, which is why Reinhard fixes the
brightness but keeps the hue error while the stain-vector methods do not.

The subordinate strip shows why: projecting each sub-tile onto the reference
haematoxylin and eosin directions puts the three centres in three separate
clusters before normalisation and in one cluster after.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes
from figkit import image_axes, corner_stamp, background_card, halo_text

# Reference (target) stain basis - the house H&E look used by 540-544.
REF_H = np.array([0.650, 0.704, 0.286])
REF_E = np.array([0.092, 0.954, 0.283])
BASIS = np.stack([REF_H, REF_E], axis=1)          # 3 x 2, for projection

CENTRE_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]
OURS = "#3b5b92"
ACCENT = "#b23a48"
TILE_EDGE = "#b0b0b0"
MPP = 0.5
TH, TW = 260, 374                                  # 130 x 187 micrometres

# Per-centre staining protocol: stain directions plus concentration gains.
CENTRES = [
    dict(name="Centre 1", h=[0.62, 0.70, 0.35], e=[0.14, 0.92, 0.36],
         sh=1.26, se=0.76, gain=1.12),
    dict(name="Centre 2", h=[0.70, 0.66, 0.27], e=[0.07, 0.97, 0.20],
         sh=0.70, se=1.30, gain=0.86),
    dict(name="Centre 3", h=[0.66, 0.74, 0.22], e=[0.16, 0.90, 0.42],
         sh=1.08, se=1.22, gain=1.20),
]
# How far each method moves the stain directions and the concentrations.
# Reinhard matches first-order colour statistics, so it corrects the gain
# almost completely while barely touching the stain directions.
METHODS = [("Original", 0.00, 0.00), ("Reinhard", 0.30, 0.95),
           ("Macenko", 0.72, 0.80), ("Vahadane", 0.86, 0.90),
           ("Ours", 0.97, 0.97)]


def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales, weights):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def render(ch, ce, od_h, od_e, gain, desat=0.10):
    od = (gain * np.clip(ch, 0, None)[..., None] * od_h
          + gain * np.clip(ce, 0, None)[..., None] * od_e)
    rgb = np.exp(-od)
    lum = rgb @ np.array([0.30, 0.59, 0.11])
    return np.clip(rgb * (1.0 - desat) + lum[..., None] * desat, 0, 1)


def tissue_fields(seed):
    """Stain concentration maps for one slide; colour is applied later."""
    shape = (TH, TW)
    rng = np.random.default_rng(seed)
    f = _fbm(rng, shape, (8, 20, 42), (0.44, 0.34, 0.22))
    p_t = np.clip(0.52 + 0.62 * f, 0, 1)
    p_l = np.clip(0.36 * _fbm(rng, shape, (5, 12), (.6, .4)) - 0.02, 0, 1)

    ce = (0.74 - 0.20 * p_t - 0.26 * p_l)
    ce *= 1.0 + 0.24 * _fbm(rng, shape, (1.0, 3.5, 10.0), (.42, .34, .24))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    ce += 0.10 * np.sin((0.83 * xx + 0.56 * yy) / 4.5
                        + 2.4 * _noise(rng, shape, 7.0)) * (1.0 - p_t)
    ch = 0.07 + 0.06 * p_t

    dens = np.clip(1100 + 3600 * p_t + 8600 * p_l, 60, 12000)
    dens_px = dens * (MPP / 1000.0) ** 2
    r_px = np.clip(2.25 + 1.85 * p_t - 0.55 * p_l, 1.1, None) / MPP
    peak = float(dens_px.max())
    n_try = int(np.clip(26.0 * peak * shape[0] * shape[1], 300, 90000))
    ys = rng.integers(0, shape[0], n_try)
    xs = rng.integers(0, shape[1], n_try)
    keep = rng.random(n_try) < dens_px[ys, xs] / peak
    occ = np.zeros(shape, bool)
    for y, x in zip(ys[keep], xs[keep]):
        r = r_px[y, x] * rng.uniform(0.74, 1.34)
        rr = int(max(1, round(0.92 * r)))
        y0, y1 = max(0, y - rr), min(shape[0], y + rr + 1)
        x0, x1 = max(0, x - rr), min(shape[1], x + rr + 1)
        if occ[y0:y1, x0:x1].any():
            continue
        occ[y0:y1, x0:x1] = True
        e = rng.uniform(1.0, 1.12 + 1.05 * (1 - p_t[y, x]))
        a, b = r * e, r / e
        th = rng.uniform(0, np.pi)
        rad = int(np.ceil(1.9 * max(a, b))) + 2
        y0, y1 = max(0, y - rad), min(shape[0], y + rad + 1)
        x0, x1 = max(0, x - rad), min(shape[1], x + rad + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        ct, st = np.cos(th), np.sin(th)
        u = ((gx - x) * ct + (gy - y) * st) / max(a, 1e-6)
        v = (-(gx - x) * st + (gy - y) * ct) / max(b, 1e-6)
        ang = np.arctan2(v, u)
        ph = rng.uniform(0, 6.28)
        d = np.hypot(u, v) / (1.0 + 0.14 * np.sin(3 * ang + ph)
                              + 0.08 * np.sin(5 * ang - ph))
        amp = 0.88 + 0.42 * p_t[y, x] + 0.85 * p_l[y, x]
        ch[y0:y1, x0:x1] += amp * np.clip(1.18 - d * d, 0, 1) ** 0.55
        ce[y0:y1, x0:x1] += 0.13 * np.clip(1.9 - d * d, 0, 1) ** 1.6
    ch *= 1.0 + 0.30 * _fbm(rng, shape, (0.7, 2.0), (.6, .4))
    return ch, ce


def stain_params(centre, a_vec, a_scale):
    h = _unit((1 - a_vec) * np.array(centre["h"]) + a_vec * REF_H)
    e = _unit((1 - a_vec) * np.array(centre["e"]) + a_vec * REF_E)
    sh = (1 - a_scale) * centre["sh"] + a_scale
    se = (1 - a_scale) * centre["se"] + a_scale
    gain = (1 - a_scale) * centre["gain"] + a_scale
    return h, e, sh, se, gain


def rgb_to_lab(rgb):
    """sRGB (0..1) to CIE L*a*b* under D65; enough for a Delta-E76 score."""
    m = rgb > 0.04045
    lin = np.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    mat = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]])
    xyz = lin @ mat.T / np.array([0.95047, 1.0, 1.08883])
    t = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.dstack([116.0 * t[..., 1] - 16.0,
                      500.0 * (t[..., 0] - t[..., 1]),
                      200.0 * (t[..., 1] - t[..., 2])])


def delta_e(a, b):
    return float(np.mean(np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(b),
                                        axis=-1)))


def stain_coords(rgb, nb=7):
    """Mean (haematoxylin, eosin) concentration of each sub-tile."""
    hh, ww, _ = rgb.shape
    by, bx = hh // nb, ww // nb
    od = -np.log(np.clip(rgb[:nb * by, :nb * bx], 1e-3, 1.0))
    od = od.reshape(nb, by, nb, bx, 3).mean(axis=(1, 3)).reshape(-1, 3)
    coef, *_ = np.linalg.lstsq(BASIS, od.T, rcond=None)
    return coef[0], coef[1]


def main():
    set_pub_style(base_font=8)
    fields = [tissue_fields(320 + 17 * r) for r in range(3)]
    refs = [render(ch, ce, REF_H, REF_E, 1.0) for ch, ce in fields]

    imgs, de = {}, np.zeros((3, len(METHODS)))
    for r, (ch, ce) in enumerate(fields):
        for c, (_, a_vec, a_scale) in enumerate(METHODS):
            h, e, sh, se, gain = stain_params(CENTRES[r], a_vec, a_scale)
            imgs[(r, c)] = render(ch * sh, ce * se, h, e, gain)
            de[r, c] = delta_e(imgs[(r, c)], refs[r])

    fig = plt.figure(figsize=(10.4, 6.4))
    # Row 3 is an empty spacer: the image rows want a 0.03 gutter while the
    # analysis strip needs room for its titles and axis labels.
    gs = fig.add_gridspec(5, 5, height_ratios=[1.0, 1.0, 1.0, 0.16, 0.86],
                          wspace=0.045, hspace=0.10, left=0.070, right=0.988,
                          top=0.876, bottom=0.070)
    axes = np.empty((3, 5), object)
    for r in range(3):
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            ours = c == len(METHODS) - 1
            image_axes(ax, color=OURS if ours else TILE_EDGE,
                       lw=1.1 if ours else 0.55)
            ax.imshow(imgs[(r, c)], interpolation="bilinear", zorder=1)
            if r == 0:
                ax.set_title(METHODS[c][0], fontsize=8.4, pad=3.4,
                             fontweight="bold" if ours else "normal",
                             color="#1a1a1a")
            if c == 0:
                ax.set_ylabel(CENTRES[r]["name"], fontsize=8.0, labelpad=3.4,
                              color="#1a1a1a")
            corner_stamp(ax, f"ΔE {de[r, c]:.1f}", loc="lower left",
                         fontsize=6.4, weight="bold" if ours else "normal")

    for ax in axes.ravel():
        ax.apply_aspect()
    p0, p1 = axes[0, 4].get_position(), axes[2, 4].get_position()
    background_card(fig, (p0.x0 - 0.011, p1.y0 - 0.026,
                          p0.width + 0.022, p0.y1 - p1.y0 + 0.052),
                    facecolor="#eaeff6", zorder=-5, radius=0.006)
    halo_text(axes[0, 0], 0.030, 0.955, "20x  ·  same tissue across a row",
              transform=axes[0, 0].transAxes, ha="left", va="top",
              fontsize=6.2, color="#1a1a1a", halo="white", halo_lw=2.3)

    # ---- subordinate strip -------------------------------------------------
    sub = gs[4, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.22],
                               wspace=0.32)
    ax_b = fig.add_subplot(sub[0, 0])
    ax_a = fig.add_subplot(sub[0, 1])
    ax_d = fig.add_subplot(sub[0, 2])

    pts = {}
    for r in range(3):
        for c in (0, len(METHODS) - 1):
            pts[(r, c)] = stain_coords(imgs[(r, c)])
    allh = np.concatenate([v[0] for v in pts.values()])
    alle = np.concatenate([v[1] for v in pts.values()])
    xlim = (allh.min() - 0.05, allh.max() + 0.05)
    ylim = (alle.min() - 0.05, alle.max() + 0.05)

    for ax, col, ttl in ((ax_b, 0, "Before  (original slides)"),
                         (ax_a, len(METHODS) - 1, "After  (ours)")):
        for r in range(3):
            hcoef, ecoef = pts[(r, col)]
            ax.scatter(hcoef, ecoef, s=9, color=CENTRE_COLORS[r],
                       alpha=0.72, linewidths=0.0,
                       label=CENTRES[r]["name"] if ax is ax_b else None)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("haematoxylin concentration (-)", fontsize=7.0,
                      labelpad=1.6)
        ax.tick_params(labelsize=6.4, length=2.2, pad=1.5)
        ax.grid(linewidth=0.4, alpha=0.6)
        style_axes(ax)
        spread = np.mean([np.hypot(*(np.array([pts[(r, col)][0].mean(),
                                               pts[(r, col)][1].mean()])
                                     - np.array([allh.mean(), alle.mean()])))
                          for r in range(3)])
        ax.set_title(f"{ttl}   ·   centre spread {spread:.2f}", fontsize=7.8,
                     fontweight="bold", pad=3.0)
    ax_b.set_ylabel("eosin concentration (-)", fontsize=7.0, labelpad=2.0)
    ax_a.tick_params(labelleft=False)
    ax_b.legend(frameon=True, framealpha=0.90, facecolor="white",
                edgecolor="#cccccc", borderpad=0.4, fontsize=6.4,
                loc="upper right", handletextpad=0.3, borderaxespad=0.2,
                labelspacing=0.25)

    mean_de, sd_de = de.mean(axis=0), de.std(axis=0)
    cols = ["#8a8a8a"] + ["#3b5b92"] * 3 + [ACCENT]
    xs = np.arange(len(METHODS))
    ax_d.bar(xs, mean_de, yerr=sd_de, width=0.58, color=cols, alpha=0.90,
             edgecolor="#3a3a3a", linewidth=0.5,
             error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor="#555555"))
    for x, m, s in zip(xs, mean_de, sd_de):
        ax_d.text(x, m + s + 0.035 * mean_de.max(), f"{m:.1f}", ha="center",
                  va="bottom", fontsize=6.4, color="#333333")
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels([m for m, _, _ in METHODS], fontsize=6.5)
    ax_d.set_ylabel("ΔE to reference (-)", fontsize=7.0, labelpad=2.0)
    ax_d.set_ylim(0, (mean_de + sd_de).max() * 1.25)
    ax_d.tick_params(labelsize=6.4, length=2.2, pad=1.5)
    ax_d.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax_d)
    ax_d.set_title("Colour difference to the reference stain  "
                   "(mean ± sd over the three slides)", fontsize=7.8,
                   fontweight="bold", pad=3.0)

    savefig(fig, "545_path_stain_normalization.png")


if __name__ == "__main__":
    main()
