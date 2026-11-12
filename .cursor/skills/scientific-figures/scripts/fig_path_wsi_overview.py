"""Figure 540 - Whole-slide overview with a three-level magnification chain.

The main panel is a synthetic H&E section at 1x (10 um/px): an irregular
tissue outline carrying tumour nests, desmoplastic stroma, a necrotic core
and lymphoid aggregates, overlaid with the 0.5 mm analysis tile grid and
three annotated regions.  The right column walks the same region down the
pyramid, 1x -> 4x -> 20x, each level boxed on its parent and tied to it by
leader lines, so the reader can see four orders of magnitude in one figure.

Staining is physical rather than decorative: haematoxylin and eosin
concentration fields are combined through the Beer-Lambert law with the
standard Ruifrok optical-density vectors, which is what makes the nuclei
read as blue-purple and the stroma as eosin pink.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig
from figkit import (image_axes, roi_rect, add_scale_bar, halo_text,
                    panel_labels_inches, ZOOM, ZOOM_ALT)

# Ruifrok-Johnston optical-density vectors for haematoxylin and eosin.  Every
# tile in this figure is rendered through these two, so the pink/purple
# balance is identical at every magnification.
OD_H = np.array([0.650, 0.704, 0.286])
OD_E = np.array([0.092, 0.954, 0.283])
GLASS = np.array([0.988, 0.982, 0.988])      # bare slide, not pure white

ANNOT_B = "#3b5b92"          # house blue  - annotated necrotic region
ANNOT_C = "#2e7d5b"          # house green - annotated lymphoid aggregate
TILE_EDGE = "#c9c9c9"


# ---------------------------------------------------------------------------
# Synthetic H&E machinery
# ---------------------------------------------------------------------------
def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales=(1.2, 4.0, 12.0), weights=(0.44, 0.34, 0.22)):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _blob(shape, fy, fx, ry, rx):
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    return np.exp(-0.5 * (((yy - fy * hh) / (ry * hh)) ** 2
                          + ((xx - fx * ww) / (rx * ww)) ** 2))


def he_rgb(ch, ce, tissue=None, desat=0.10):
    """Beer-Lambert compositing of two stain concentration maps.

    A slide is never a clean two-stain system - fixative, residual protein and
    the scanner's own colour management all pull the result toward neutral, so
    a small desaturation is what keeps the synthetic tile from looking like a
    saturated dye chart.
    """
    od = (np.clip(ch, 0, None)[..., None] * OD_H
          + np.clip(ce, 0, None)[..., None] * OD_E)
    rgb = np.exp(-od)
    if desat:
        lum = rgb @ np.array([0.30, 0.59, 0.11])
        rgb = rgb * (1.0 - desat) + lum[..., None] * desat
    if tissue is not None:
        t = np.clip(tissue, 0, 1)[..., None]
        rgb = rgb * t + GLASS * (1.0 - t)
    return np.clip(rgb, 0, 1)


def wsi_fields(shape=(520, 560), seed=7):
    """Tissue mask plus the tumour / necrosis / lymphoid probability fields."""
    rng = np.random.default_rng(seed)
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    cy, cx = 0.50 * hh, 0.50 * ww
    th = np.arctan2(yy - cy, xx - cx)

    rr = (np.hypot((yy - cy) / (0.435 * hh), (xx - cx) / (0.445 * ww))
          + 0.030 * _noise(rng, shape, 6.0))
    edge = (1.0 + 0.150 * np.sin(3 * th + 0.7) + 0.095 * np.sin(5 * th - 1.9)
            + 0.062 * np.sin(8 * th + 2.4) + 0.040 * np.sin(13 * th - 0.4))
    tissue = np.clip((edge - rr) / 0.045, 0, 1)
    # a processing tear at the upper right margin
    tear = np.hypot((yy - 0.20 * hh) / (0.075 * hh),
                    (xx - 0.80 * ww) / (0.115 * ww))
    tissue *= np.clip((tear - 0.95) / 0.28, 0, 1)

    f1 = _fbm(rng, shape, (7, 18, 40), (0.44, 0.34, 0.22))
    nests = (1.05 * _blob(shape, 0.34, 0.33, 0.155, 0.170)
             + 0.86 * _blob(shape, 0.62, 0.60, 0.115, 0.130)
             + 0.55 * _blob(shape, 0.76, 0.31, 0.080, 0.085))
    p_tum = np.clip((nests + 0.40 * f1 - 0.46) / 0.32, 0, 1) * tissue

    core = np.clip((1.05 * _blob(shape, 0.34, 0.33, 0.155, 0.170) - 0.60)
                   / 0.32, 0, 1)
    p_nec = np.clip(core * (0.60 + 0.70 * _fbm(rng, shape, (4, 11), (.6, .4)))
                    - 0.10, 0, 1) * tissue

    inner = (gaussian_filter(tissue, 9.0) > 0.94).astype(float)
    rim = np.clip(4.0 * p_tum * (1.0 - p_tum) - 0.55, 0, 1)
    spots = np.clip(_fbm(rng, shape, (2.5, 6.0), (0.55, 0.45)) - 0.55, 0, 1)
    p_lym = np.clip(6.0 * rim * spots, 0, 1) * inner
    p_lym = gaussian_filter(p_lym, 1.6)
    p_lym /= max(p_lym.max(), 1e-9)

    return dict(tissue=tissue, p_tum=p_tum, p_nec=p_nec, p_lym=p_lym)


def render_lowmag(fields, seed=17):
    """1x view: nuclei are sub-pixel, so paint stain concentration directly."""
    rng = np.random.default_rng(seed)
    shape = fields["tissue"].shape
    hh, ww = shape
    p_t, p_n, p_l = fields["p_tum"], fields["p_nec"], fields["p_lym"]

    ch = (0.13 + 0.66 * p_t + 1.30 * p_l) * (1.0 - 0.86 * p_n)
    ch *= 1.0 + 0.22 * _fbm(rng, shape, (0.8, 2.6, 7.0), (.45, .33, .22))
    ce = (0.58 - 0.19 * p_t - 0.25 * p_l) * (1.0 - 0.34 * p_n)
    ce *= 1.0 + 0.14 * _fbm(rng, shape, (1.4, 5.0, 14.0), (.42, .34, .24))

    # slow staining gradient across the slide, plus two section folds
    uneven = 1.0 + 0.13 * _noise(rng, shape, 55.0)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    for x0, amp, per, wid in ((0.30 * ww, 26.0, 190.0, 2.3),
                              (0.72 * ww, 18.0, 130.0, 1.7)):
        d = np.abs(xx - (x0 + amp * np.sin(yy / per * 2 * np.pi)))
        fold = np.exp(-0.5 * (d / wid) ** 2)
        ch = ch + 0.34 * fold
        ce = ce + 0.30 * fold
    ch, ce = ch * uneven, ce * uneven

    rgb = he_rgb(ch, ce, fields["tissue"])
    rgb = np.dstack([gaussian_filter(rgb[..., c], 0.55) for c in range(3)])
    return np.clip(rgb, 0, 1)


def sample_field(field, y0, x0, out_shape, factor):
    """Bilinear crop of a 1x field onto a higher-magnification pixel grid."""
    hh, ww = out_shape
    ys = y0 + np.arange(hh) / factor
    xs = x0 + np.arange(ww) / factor
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(field, [yy, xx], order=1, mode="nearest")


def place_nuclei(rng, shape, dens_px, r_px, jitter=(0.72, 1.36), gap=0.92):
    """Rejection-sampled nucleus centres honouring a spatially varying density."""
    hh, ww = shape
    peak = float(dens_px.max())
    if peak <= 0:
        return []
    n_try = int(np.clip(26.0 * peak * hh * ww, 200, 90000))
    ys = rng.integers(0, hh, n_try)
    xs = rng.integers(0, ww, n_try)
    keep = rng.random(n_try) < (dens_px[ys, xs] / peak)
    occ = np.zeros(shape, bool)
    out = []
    for y, x in zip(ys[keep], xs[keep]):
        r = r_px[y, x] * rng.uniform(*jitter)
        rr = int(max(1, round(gap * r)))
        y0, y1 = max(0, y - rr), min(hh, y + rr + 1)
        x0, x1 = max(0, x - rr), min(ww, x + rr + 1)
        if occ[y0:y1, x0:x1].any():
            continue
        occ[y0:y1, x0:x1] = True
        out.append((float(y), float(x), float(r)))
    return out


def stamp_nuclei(ch, ce, nuclei, rng, elong_map, amp_map):
    """Paint elliptical nuclei with a chromatin texture and a pale halo.

    Shape and chromatin density are read from maps at each centre, which is
    what makes the regional difference legible: spindle fibroblasts in the
    stroma, round vesicular nuclei in the tumour, tiny hyperchromatic dots in
    a lymphoid aggregate.
    """
    hh, ww = ch.shape
    for (cy, cx, r) in nuclei:
        iy, ix = int(cy), int(cx)
        amp = amp_map[iy, ix]
        e = rng.uniform(1.0, elong_map[iy, ix])
        a, b = r * e, r / e
        th = rng.uniform(0, np.pi)
        rad = int(np.ceil(2.6 * max(a, b))) + 1
        y0, y1 = max(0, int(cy) - rad), min(hh, int(cy) + rad + 1)
        x0, x1 = max(0, int(cx) - rad), min(ww, int(cx) + rad + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy, dx = yy - cy, xx - cx
        ct, st = np.cos(th), np.sin(th)
        u = (dx * ct + dy * st) / max(a, 1e-6)
        v = (-dx * st + dy * ct) / max(b, 1e-6)
        d = u * u + v * v
        core = np.clip(1.20 - d, 0, 1) ** 0.55
        halo = np.clip(2.60 - d, 0, 1) ** 1.6
        ch[y0:y1, x0:x1] += amp * rng.uniform(0.85, 1.15) * core
        ce[y0:y1, x0:x1] += 0.13 * halo
    return ch, ce


def render_tile(fields, y0, x0, out_shape, factor, mpp, seed):
    """Render one magnified field; ``factor`` = output px per 1x px."""
    rng = np.random.default_rng(seed)
    p_t = sample_field(fields["p_tum"], y0, x0, out_shape, factor)
    p_n = sample_field(fields["p_nec"], y0, x0, out_shape, factor)
    p_l = sample_field(fields["p_lym"], y0, x0, out_shape, factor)
    tis = sample_field(fields["tissue"], y0, x0, out_shape, factor)
    p_t = np.clip(p_t + 0.22 * _fbm(rng, out_shape, (2, 6), (.6, .4)), 0, 1)

    ce = (0.74 - 0.20 * p_t - 0.26 * p_l) * (1.0 - 0.26 * p_n)
    ce *= 1.0 + 0.24 * _fbm(rng, out_shape, (1.0, 3.5, 10.0), (.42, .34, .24))
    # collagen fibres in the stroma: banded, low-frequency along one axis
    yy, xx = np.mgrid[0:out_shape[0], 0:out_shape[1]].astype(float)
    fib = np.sin((0.83 * xx + 0.56 * yy) / max(2.4, 9.0 / factor)
                 + 2.4 * _noise(rng, out_shape, 7.0))
    ce += 0.10 * fib * (1.0 - p_t) * (1.0 - p_l)
    ch = 0.07 + 0.06 * p_t + 0.10 * p_n

    dens = np.clip((950 + 3700 * p_t + 8200 * p_l) * (1.0 - 0.85 * p_n),
                   60, 11000)
    dens_px = dens * (mpp / 1000.0) ** 2
    r_um = 2.25 + 1.85 * p_t - 0.55 * p_l - 0.30 * p_n
    nuclei = place_nuclei(rng, out_shape, dens_px * tis,
                          np.clip(r_um, 1.1, None) / mpp)
    elong = 1.12 + 1.05 * (1.0 - p_t) * (1.0 - p_l)
    amp = 0.88 + 0.42 * p_t + 0.85 * p_l - 0.35 * p_n
    ch, ce = stamp_nuclei(ch, ce, nuclei, rng, elong, np.clip(amp, 0.4, None))
    ch *= 1.0 + 0.30 * _fbm(rng, out_shape, (0.7, 2.0), (.6, .4))

    uneven = 1.0 + 0.10 * _noise(rng, out_shape, 0.30 * out_shape[0])
    rgb = he_rgb(ch * uneven, ce * uneven, tis)
    rgb = np.dstack([gaussian_filter(rgb[..., c], 0.45) for c in range(3)])
    rgb += 0.008 * rng.standard_normal(rgb.shape)
    return np.clip(rgb, 0, 1), nuclei


# ---------------------------------------------------------------------------
def pick_box(field, target, bw, bh, prefer, weight=0.004):
    """Locate a ``bw x bh`` window whose centre sits on ``field == target``."""
    hh, ww = field.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    cost = np.abs(field - target) + weight * np.hypot(yy - prefer[0],
                                                      xx - prefer[1])
    cost[:bh // 2 + 1, :] = np.inf
    cost[-(bh // 2 + 2):, :] = np.inf
    cost[:, :bw // 2 + 1] = np.inf
    cost[:, -(bw // 2 + 2):] = np.inf
    cy, cx = np.unravel_index(np.argmin(cost), cost.shape)
    return int(cx - bw // 2), int(cy - bh // 2)


def leader(fig, ax_a, pts_a, ax_b, pts_b, color):
    for pa, pb in zip(pts_a, pts_b):
        cp = ConnectionPatch(xyA=pa, coordsA=ax_a.transData,
                             xyB=pb, coordsB=ax_b.transAxes,
                             color=color, lw=0.7, alpha=0.85, zorder=8)
        cp.set_in_layout(False)
        fig.add_artist(cp)


def main():
    set_pub_style(base_font=8)
    MPP1, MPP4, MPP20 = 10.0, 2.5, 0.5          # um per pixel at each level
    shape = (520, 560)
    fields = wsi_fields(shape, seed=7)
    wsi = render_lowmag(fields, seed=17)

    # ---- magnification chain --------------------------------------------
    t4 = (480, 672)                              # 4x field, pixels
    t20 = (360, 504)                             # 20x field, pixels
    bw4, bh4 = int(t4[1] * MPP4 / MPP1), int(t4[0] * MPP4 / MPP1)
    x4, y4 = pick_box(fields["p_tum"], 0.55, bw4, bh4,
                      prefer=(0.62 * shape[0], 0.60 * shape[1]))
    img4, _ = render_tile(fields, y4, x4, t4, MPP1 / MPP4, MPP4, seed=41)

    bw20, bh20 = int(t20[1] * MPP20 / MPP4), int(t20[0] * MPP20 / MPP4)
    p_t4 = sample_field(fields["p_tum"], y4, x4, t4, MPP1 / MPP4)
    x20, y20 = pick_box(p_t4, 0.62, bw20, bh20,
                        prefer=(t4[0] / 2.0, t4[1] / 2.0), weight=0.0015)
    img20, nuc20 = render_tile(fields, y4 + y20 * MPP4 / MPP1,
                               x4 + x20 * MPP4 / MPP1, t20, MPP1 / MPP20,
                               MPP20, seed=73)

    # ---- quantities that get printed on the figure ------------------------
    px_mm2 = (MPP1 / 1000.0) ** 2
    tis = fields["tissue"] > 0.5
    area_mm2 = tis.sum() * px_mm2
    tum_frac = 100.0 * ((fields["p_tum"] > 0.5) & tis).sum() / max(tis.sum(), 1)
    tile_px = int(round(500.0 / MPP1))           # 0.5 mm analysis tiles
    ny, nx = shape[0] // tile_px, shape[1] // tile_px
    cov = tis[:ny * tile_px, :nx * tile_px].reshape(
        ny, tile_px, nx, tile_px).mean(axis=(1, 3))
    n_tiles = int((cov >= 0.5).sum())
    fov20_mm2 = t20[0] * t20[1] * (MPP20 / 1000.0) ** 2
    dens20 = len(nuc20) / fov20_mm2

    # ---- canvas -----------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60], hspace=0.20,
                          wspace=0.20, left=0.045, right=0.980, top=0.880,
                          bottom=0.085)
    ax_w = fig.add_subplot(gs[:, 0])
    ax_4 = fig.add_subplot(gs[0, 1])
    ax_20 = fig.add_subplot(gs[1, 1])

    # ---- (a) whole slide --------------------------------------------------
    image_axes(ax_w, color="#bdbdbd", lw=0.6)
    ax_w.imshow(wsi, interpolation="bilinear", zorder=1)
    for k in range(1, nx + 1):
        ax_w.axvline(k * tile_px - 0.5, color=TILE_EDGE, lw=0.35, alpha=0.75,
                     zorder=3)
    for k in range(1, ny + 1):
        ax_w.axhline(k * tile_px - 0.5, color=TILE_EDGE, lw=0.35, alpha=0.75,
                     zorder=3)
    roi_rect(ax_w, (x4, y4, bw4, bh4), color=ZOOM, lw=1.3, zorder=7)
    halo_text(ax_w, x4 + bw4 / 2, y4 - 6, "ROI-1  tumour front", ha="center",
              va="bottom", fontsize=6.6, color="white", halo=ZOOM)

    for fld, col, name in ((fields["p_nec"], ANNOT_B, "ROI-2  necrotic core"),
                           (fields["p_lym"], ANNOT_C, "ROI-3  lymphoid")):
        cy, cx = np.unravel_index(np.argmax(gaussian_filter(fld, 7.0)),
                                  fld.shape)
        roi_rect(ax_w, (cx - 32, cy - 24, 64, 48), color=col, lw=1.0, ls="--",
                 zorder=7)
        halo_text(ax_w, cx, cy + 27, name, ha="center", va="top",
                  fontsize=6.4, color="white", halo=col)

    add_scale_bar(ax_w, 1000.0 / MPP1, "1 mm", loc=(0.035, 0.045),
                  height=0.016, fontsize=6.4)
    halo_text(ax_w, 0.978, 0.972,
              f"section area {area_mm2:.1f} mm²    {n_tiles} analysis tiles",
              transform=ax_w.transAxes, ha="right", va="top", fontsize=6.6,
              color="#333333", halo="white", halo_lw=2.2)
    ax_w.set_title("Level 0 overview at 1x  (10 µm/px, 0.5 mm tile grid)",
                   fontsize=8.8, fontweight="bold", pad=4.0)

    # ---- (b) 4x -----------------------------------------------------------
    image_axes(ax_4, color="#bdbdbd", lw=0.6)
    ax_4.imshow(img4, interpolation="bilinear", zorder=1)
    roi_rect(ax_4, (x20, y20, bw20, bh20), color=ZOOM_ALT, lw=1.3, zorder=7)
    add_scale_bar(ax_4, 500.0 / MPP4, "500 µm", loc=(0.035, 0.055),
                  height=0.024, fontsize=6.2)
    ax_4.set_title("ROI-1 at 4x  (2.5 µm/px)", fontsize=8.5,
                   fontweight="bold", pad=3.4)

    # ---- (c) 20x ----------------------------------------------------------
    image_axes(ax_20, color="#bdbdbd", lw=0.6)
    ax_20.imshow(img20, interpolation="bilinear", zorder=1)
    add_scale_bar(ax_20, 100.0 / MPP20, "100 µm", loc=(0.035, 0.055),
                  height=0.024, fontsize=6.2)
    halo_text(ax_20, 0.972, 0.955,
              f"{len(nuc20)} nuclei in field   |   {dens20:,.0f} cells/mm²",
              transform=ax_20.transAxes, ha="right", va="top", fontsize=6.5,
              color="#1a1a1a", halo="white", halo_lw=2.4)
    ax_20.set_title("ROI-1 detail at 20x  (0.5 µm/px)", fontsize=8.5,
                    fontweight="bold", pad=3.4)

    for ax in (ax_w, ax_4, ax_20):
        ax.apply_aspect()

    leader(fig, ax_w, [(x4 + bw4, y4), (x4 + bw4, y4 + bh4)],
           ax_4, [(0.0, 1.0), (0.0, 0.0)], ZOOM)
    leader(fig, ax_4, [(x20, y20 + bh20), (x20 + bw20, y20 + bh20)],
           ax_20, [(0.0, 1.0), (1.0, 1.0)], ZOOM_ALT)

    panel_labels_inches(fig, [ax_w, ax_4, ax_20], dx=-0.26, dy=0.09)

    savefig(fig, "540_path_wsi_overview.png")


if __name__ == "__main__":
    main()
