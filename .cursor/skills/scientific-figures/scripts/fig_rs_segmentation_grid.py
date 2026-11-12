"""Figure 179 - Semantic segmentation result matrix.

Three test tiles (rows) x five views (columns): true-colour image, reference
annotation and three segmentation networks, each drawn as a translucent
class overlay. The last column also highlights its misclassified pixels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)
SEG_NAMES = ["Water", "Cropland", "Forest", "Grassland", "Built-up",
             "Bare soil"]
SEG_COLORS = ["#2f7fb8", "#8fbc4a", "#1d6b3f", "#e0d770", "#b23a48",
              "#d9a86c"]

SPECTRA = {
    WATER:  [0.055, 0.052, 0.032, 0.014, 0.008],
    CROP:   [0.038, 0.065, 0.040, 0.430, 0.210],
    FOREST: [0.028, 0.050, 0.030, 0.350, 0.145],
    GRASS:  [0.055, 0.085, 0.072, 0.265, 0.230],
    URBAN:  [0.128, 0.140, 0.152, 0.190, 0.270],
    SOIL:   [0.140, 0.180, 0.235, 0.300, 0.365],
    ROAD:   [0.098, 0.105, 0.118, 0.135, 0.165],
}
TEX_AMP = {WATER: 0.05, CROP: 0.07, FOREST: 0.24, GRASS: 0.13,
           URBAN: 0.30, SOIL: 0.13, ROAD: 0.09}


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(3, 8, 20), weights=(0.5, 0.32, 0.18)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _polyline_dist(n, pts):
    m = np.zeros((n, n), bool)
    pts = np.asarray(pts, float)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        k = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
        ii = np.clip(np.round(np.linspace(y0, y1, k)).astype(int), 0, n - 1)
        jj = np.clip(np.round(np.linspace(x0, x1, k)).astype(int), 0, n - 1)
        m[ii, jj] = True
    return distance_transform_edt(~m)


def build_labels(n=256, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    anchors = np.array([[0.20, 0.22], [0.74, 0.16],
                        [0.26, 0.76], [0.82, 0.70]]) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)

    fid = np.zeros((n, n), int)
    for r, (ang, cu, cv) in enumerate([(17, 27, 35), (-26, 31, 22),
                                       (39, 23, 29), (6, 35, 26)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 4.5 * _noise(rng, n, 16)
        v = -xx * np.sin(th) + yy * np.cos(th) + 4.5 * _noise(rng, n, 16)
        f = (np.floor(u / cu).astype(int) * 211
             + np.floor(v / cv).astype(int) * 37 + r * 1013)
        fid = np.where(reg == r, f, fid)

    lab = np.full((n, n), CROP, np.int16)
    key = np.abs(fid) % 4096
    lut_v = rng.random(4096)
    lut_k = rng.random(4096)
    vig = 0.34 + 0.64 * lut_v[key]
    kind = lut_k[key]
    lab[kind < 0.10] = SOIL
    lab[(kind >= 0.10) & (kind < 0.24)] = GRASS

    f2 = _fbm(rng, n, (5, 13, 30), (0.34, 0.34, 0.32))
    zone = _noise(rng, n, 40) > -0.35
    forest = gaussian_filter((f2 > 0.68).astype(float), 2.0) > 0.42
    lab[forest & zone] = FOREST
    vig = np.where(lab == FOREST, 0.86 + 0.09 * _noise(rng, n, 4), vig)
    vig = np.where(lab == GRASS, 0.46 + 0.10 * _noise(rng, n, 6), vig)

    wob = _noise(rng, n, 3)
    for (cx, cy, rr) in [(0.125, 0.45, 0.050), (0.235, 0.56, 0.040),
                         (0.075, 0.63, 0.034), (0.185, 0.66, 0.030)]:
        m = np.hypot(xx - cx * n, yy - cy * n) < rr * n * (1 + 0.03 * wob)
        lab[m] = CROP
        vig[m] = 0.93 + 0.05 * wob[m]

    th = np.deg2rad(-8)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    blocks = ((u % 11.0) < 9.3) & ((v % 9.0) < 7.5)
    for (cx, cy, rad, thr) in [(0.70, 0.63, 0.155, 0.60),
                               (0.36, 0.30, 0.062, 0.62)]:
        blob = (np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2)
                         / (2 * (rad * n) ** 2)))
                + 0.30 * _fbm(rng, n, (7, 17), (0.5, 0.5)))
        city = blob > thr
        lab[city & blocks] = URBAN
        lab[city & ~blocks] = ROAD

    roads = [[(0, 0.20 * n), (0.44 * n, 0.34 * n), (0.70 * n, 0.63 * n),
              (n - 1, 0.74 * n)],
             [(0.70 * n, 0.63 * n), (0.62 * n, 0.05 * n)],
             [(0.05 * n, 0.95 * n), (0.45 * n, 0.72 * n), (0.70 * n, 0.63 * n)],
             [(0.02 * n, 0.55 * n), (n - 1, 0.44 * n)]]
    for pl in roads:
        lab[_polyline_dist(n, pl) <= 1.1] = ROAD

    ty = np.linspace(0, n - 1, 240)
    tx = (0.30 * n + 0.105 * n * np.sin(2 * np.pi * ty / n * 1.05)
          + 0.045 * n * np.sin(2 * np.pi * ty / n * 2.6 + 1.3))
    dr = _polyline_dist(n, np.c_[tx, ty])
    lab[dr <= (3.3 + 1.1 * _noise(rng, n, 26))] = WATER
    sy = np.linspace(0.30 * n, 0.615 * n, 160)
    tributary_x = np.interp(sy, ty, tx)
    frac = (sy - sy[0]) / (sy[-1] - sy[0])
    sx = (1 - frac ** 1.4) * 0.97 * n + (frac ** 1.4) * tributary_x \
        + 7 * np.sin(sy / 17.0) * (1 - frac)
    ds = _polyline_dist(n, np.c_[sx, sy])
    lab[ds <= (1.5 + 0.45 * _noise(rng, n, 20))] = WATER
    lake = ((((xx - 0.155 * n) / (0.080 * n)) ** 2
             + ((yy - 0.845 * n) / (0.048 * n)) ** 2)
            < 1 + 0.30 * _noise(rng, n, 5))
    lab[lake] = WATER

    aux = {"vig": np.clip(vig, 0.15, 1.05), "fid": fid, "reg": reg}
    return lab, aux


def synthesize(lab, aux, seed=3, noise=1.0, gain=1.0):
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    vig = aux["vig"]
    bands = np.zeros((5, n, n))
    for cls, spec in SPECTRA.items():
        m = lab == cls
        if not m.any():
            continue
        for k in range(5):
            bands[k][m] = spec[k]

    veg = np.isin(lab, [CROP, FOREST, GRASS])
    fc = np.clip(vig, 0.05, 1.0)
    soil_bg = np.array(SPECTRA[SOIL])
    dryness = 0.74 + 0.20 * _noise(rng, n, 18)
    for k in range(5):
        mixed = fc * bands[k] + (1.0 - fc) * soil_bg[k] * dryness
        bands[k] = np.where(veg, mixed, bands[k])

    amp = np.zeros((n, n))
    for cls, a in TEX_AMP.items():
        amp[lab == cls] = a
    tex = 1 + amp * noise * (0.55 * _noise(rng, n, 1.0)
                             + 0.30 * _noise(rng, n, 2.5)
                             + 0.25 * _noise(rng, n, 6.0))
    illum = 1 + 0.05 * _noise(rng, n, 35)
    yg, xg = np.mgrid[0:n, 0:n] / (n - 1.0) - 0.5
    vign = 1 - 0.11 * (xg ** 2 + yg ** 2) / 0.5

    for k in range(5):
        b = bands[k] * tex * illum * vign * gain
        b *= 1 + 0.22 * amp * _noise(rng, n, 0.8)
        stripe = (1 + 0.010 * np.sin(np.arange(n) / 3.0 + 1.7 * k)
                  + 0.006 * rng.standard_normal(n))
        b *= stripe[None, :]
        b += 0.003 * rng.standard_normal((n, n))
        sp = rng.random((n, n))
        b[sp < 3e-4] *= 2.2
        b[sp > 1 - 3e-4] *= 0.35
        bands[k] = b
    return np.clip(bands, 0.0015, 1.0)


def image_axes(ax, color="#333333", lw=0.7):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


# ---------------------------------------------------------------------------
# Segmentation simulation
# ---------------------------------------------------------------------------
CONF = {CROP: GRASS, GRASS: CROP, FOREST: CROP, URBAN: SOIL, SOIL: URBAN,
        WATER: FOREST}


def segment(gt, seed, disp, blob_rate, blob_sigma):
    """Boundary displacement plus confusion blobs = realistic network output."""
    n = gt.shape[0]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    dy = disp * _noise(rng, n, 5.0)
    dx = disp * _noise(rng, n, 5.0)
    sy = np.clip(np.round(yy + dy).astype(int), 0, n - 1)
    sx = np.clip(np.round(xx + dx).astype(int), 0, n - 1)
    pred = gt[sy, sx].copy()
    b = _noise(rng, n, blob_sigma)
    hit = b > np.quantile(b, 1.0 - blob_rate)
    for a, c in CONF.items():
        pred[hit & (gt == a)] = c
    return pred


def miou(gt, pred, classes):
    ious = []
    for c in classes:
        inter = np.count_nonzero((gt == c) & (pred == c))
        union = np.count_nonzero((gt == c) | (pred == c))
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious))


def main():
    set_pub_style(base_font=8)
    crop = 160
    # (seed, x0, y0, rot90) - different scenes, windows and orientations
    tiles = [(11, 96, 84, 0), (23, 6, 52, 1), (41, 52, 0, 3)]
    seg_ids = [WATER, CROP, FOREST, GRASS, URBAN, SOIL]

    samples = []
    for k, (sd, x0, y0, rot) in enumerate(tiles):
        lab, aux = build_labels(256, seed=sd)
        bands = synthesize(lab, aux, seed=3 + 5 * k)
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        gt = np.rot90(lab[sl], rot).copy()
        gt[gt == ROAD] = URBAN
        lims = [np.percentile(bands[i], [2.0, 97.5]) for i in range(5)]
        rgb = np.dstack([np.clip((bands[i][sl] - lims[i][0])
                                 / (lims[i][1] - lims[i][0]), 0, 1) ** g
                         for i, g in zip((2, 1, 0), (0.72, 0.74, 0.80))])
        samples.append((np.rot90(rgb, rot).copy(), gt))

    methods = [("U-Net", dict(disp=1.6, blob_rate=0.052, blob_sigma=4.0)),
               ("DeepLabv3+", dict(disp=1.15, blob_rate=0.034,
                                   blob_sigma=6.0)),
               ("Ours (SwinSeg)", dict(disp=0.85, blob_rate=0.026,
                                       blob_sigma=7.0))]
    preds = {nm: [] for nm, _ in methods}
    for j, (nm, kw) in enumerate(methods):
        for k, (_rgb, gt) in enumerate(samples):
            preds[nm].append(segment(gt, seed=100 + 7 * j + k, **kw))

    scores = {}
    for nm, _ in methods:
        g = np.concatenate([s[1].ravel() for s in samples])
        p = np.concatenate([q.ravel() for q in preds[nm]])
        scores[nm] = miou(g, p, seg_ids)
        print(f"  {nm:16s} mIoU = {scores[nm]:.3f}")

    lut = np.array([to_rgb(SEG_COLORS[i]) if i < 6 else (0.4, 0.4, 0.4)
                    for i in range(7)])

    def overlay(rgb, lab_map, alpha=0.55):
        return np.clip((1 - alpha) * rgb + alpha * lut[lab_map], 0, 1)

    cols = ["True colour", "Reference", "U-Net", "DeepLabv3+",
            "Ours (SwinSeg)"]
    fig = plt.figure(figsize=(7.3, 5.05))
    gs = fig.add_gridspec(3, 5, wspace=0.045, hspace=0.055,
                          left=0.055, right=0.988, top=0.885, bottom=0.175)

    for r, (rgb, gt) in enumerate(samples):
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            image_axes(ax)
            if c == 0:
                ax.imshow(rgb, interpolation="nearest")
            elif c == 1:
                ax.imshow(overlay(rgb, gt), interpolation="nearest")
            else:
                pm = preds[methods[c - 2][0]][r]
                ax.imshow(overlay(rgb, pm), interpolation="nearest")
                if c == 4:
                    ax.contour(gaussian_filter((pm != gt).astype(float), 0.6),
                               levels=[0.5], colors="#ed1c24",
                               linewidths=0.45)
            if r == 0:
                ax.set_title(f"({'abcde'[c]})  {cols[c]}", fontsize=7.3,
                             fontweight="bold", pad=3.2)
            if c == 0:
                ax.set_ylabel(f"Tile {r + 1}", fontsize=7.0,
                              fontweight="bold", labelpad=3)
            if r == 2 and c >= 2:
                nm = methods[c - 2][0]
                best = scores[nm] == max(scores.values())
                ax.text(0.5, -0.055, f"mIoU {scores[nm]:.3f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.8, fontweight="bold" if best else "normal",
                        color="#1a6b45" if best else "#444444")

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4,
                     label=nm) for c, nm in zip(SEG_COLORS, SEG_NAMES)]
    handles.append(Patch(facecolor="none", edgecolor="#ed1c24",
                         linewidth=0.9,
                         label="Error outline (last column)"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=7,
                     frameon=False, fontsize=6.6, handlelength=1.25,
                     handleheight=0.95, columnspacing=1.15,
                     bbox_to_anchor=(0.52, 0.008))
    leg.set_clip_on(False)


    savefig(fig, "179_rs_segmentation_grid.png")


if __name__ == "__main__":
    main()
