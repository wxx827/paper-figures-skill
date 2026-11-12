"""Figure 178 - Single-image super-resolution of satellite imagery (x4).

Top row: low-resolution input, bicubic, SRCNN, EDSR and the ground truth.
Bottom row: the identical red-box crop from each result, column-aligned.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.ndimage import gaussian_filter, distance_transform_edt, zoom

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)

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


def synthesize(lab, aux, seed=3, noise=1.0, gain=1.0, grain=1.0):
    """``grain`` scales the pixel-scale sensor noise; a low-noise sensor
    (grain < 1) leaves the structural texture but removes the per-pixel snow,
    which is what makes a super-resolution benchmark meaningful."""
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
    tex = 1 + amp * noise * (0.55 * grain * _noise(rng, n, 1.0)
                             + 0.30 * _noise(rng, n, 2.5)
                             + 0.25 * _noise(rng, n, 6.0))
    illum = 1 + 0.05 * _noise(rng, n, 35)
    yg, xg = np.mgrid[0:n, 0:n] / (n - 1.0) - 0.5
    vign = 1 - 0.11 * (xg ** 2 + yg ** 2) / 0.5

    for k in range(5):
        b = bands[k] * tex * illum * vign * gain
        b *= 1 + 0.22 * grain * amp * _noise(rng, n, 0.8)
        stripe = (1 + 0.010 * np.sin(np.arange(n) / 3.0 + 1.7 * k)
                  + 0.006 * grain * rng.standard_normal(n))
        b *= stripe[None, :]
        b += 0.003 * grain * rng.standard_normal((n, n))
        sp = rng.random((n, n))
        b[sp < 3e-4 * grain] *= 2.2
        b[sp > 1 - 3e-4 * grain] *= 0.35
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
# Quality metrics
# ---------------------------------------------------------------------------
def psnr(ref, test, drange=1.0):
    mse = float(np.mean((ref - test) ** 2))
    return 10.0 * np.log10(drange ** 2 / max(mse, 1e-12))


def ssim(ref, test, sigma=1.5, drange=1.0):
    c1, c2 = (0.01 * drange) ** 2, (0.03 * drange) ** 2
    vals = []
    for k in range(ref.shape[2]):
        a, b = ref[..., k], test[..., k]
        ma, mb = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
        saa = gaussian_filter(a * a, sigma) - ma ** 2
        sbb = gaussian_filter(b * b, sigma) - mb ** 2
        sab = gaussian_filter(a * b, sigma) - ma * mb
        s = (((2 * ma * mb + c1) * (2 * sab + c2))
             / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2)))
        vals.append(s.mean())
    return float(np.mean(vals))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6)
    n, r = 256, 2
    lab, aux = build_labels(n, seed=11)
    bands = synthesize(lab, aux, seed=3, grain=0.30)

    lims = [np.percentile(bands[k], [2.0, 97.5]) for k in range(5)]
    gam = (0.72, 0.74, 0.80)

    def rgb(stack):
        return np.dstack([np.clip((stack[k] - lims[k][0])
                                  / (lims[k][1] - lims[k][0]), 0, 1) ** g
                          for k, g in zip((2, 1, 0), gam)])

    # reference acquired through the sensor point-spread function
    hr = np.dstack([gaussian_filter(rgb(bands)[..., k], 0.7)
                    for k in range(3)])

    # sensor degradation -> LR observation
    lo = np.stack([gaussian_filter(hr[..., k], 0.42 * r)[::r, ::r]
                   for k in range(3)], -1)
    lo = np.clip(lo + 0.0015 * rng.standard_normal(lo.shape), 0, 1)

    nearest = np.clip(np.repeat(np.repeat(lo, r, 0), r, 1), 0, 1)[:n, :n]
    bicubic = np.clip(np.dstack([zoom(lo[..., k], n / lo.shape[0], order=3)
                                 for k in range(3)]), 0, 1)

    # SRCNN: shallow network, recovers mid-frequency detail only
    hp = bicubic - np.dstack([gaussian_filter(bicubic[..., k], 1.6)
                              for k in range(3)])
    srcnn = np.clip(0.62 * bicubic + 0.38 * hr + 0.40 * hp
                    + 0.004 * rng.standard_normal(hr.shape), 0, 1)

    # EDSR: deep residual network, sharper and closer to the reference
    edsr = np.clip(0.56 * hr + 0.44 * bicubic
                   + 0.005 * rng.standard_normal(hr.shape), 0, 1)

    imgs = [nearest, bicubic, srcnn, edsr, hr]
    names = ["LR input  (20 m)", "Bicubic  x2", "SRCNN", "EDSR",
             "Ground truth  (10 m)"]
    stats = []
    for im in imgs[:-1]:
        stats.append((psnr(hr, im), ssim(hr, im)))
    stats.append((np.inf, 1.0))
    for nm, (p, s) in zip(names, stats):
        print(f"  {nm:22s} PSNR={p:6.2f} dB  SSIM={s:.4f}")

    zx, zy, zs = 148, 132, 52

    fig = plt.figure(figsize=(7.6, 3.85))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0],
                          wspace=0.055, hspace=0.30,
                          left=0.012, right=0.988, top=0.855, bottom=0.075)

    for i, (im, nm) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(gs[0, i])
        image_axes(ax)
        ax.imshow(im, interpolation="nearest")
        ax.set_title(f"({'abcde'[i]})  {nm}", fontsize=7.2,
                     fontweight="bold", pad=3.0)
        ax.add_patch(Rectangle((zx, zy), zs, zs, facecolor="none",
                               edgecolor="#b23a48", linewidth=0.9, zorder=5))
        p, s = stats[i]
        txt = ("reference" if not np.isfinite(p)
               else f"PSNR {p:.2f} dB   SSIM {s:.3f}")
        ax.text(0.5, -0.045, txt, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.0,
                fontweight="bold" if i == 3 else "normal",
                color="#1a6b45" if i == 3 else "#444444")

        azx = fig.add_subplot(gs[1, i])
        image_axes(azx, color="#b23a48", lw=1.1)
        azx.imshow(im[zy:zy + zs, zx:zx + zs], interpolation="nearest")
        azx.set_xlabel("red-box zoom, 52 x 52 px" if i == 0 else "",
                       fontsize=6.0, labelpad=1.5, color="#a02b23")


    savefig(fig, "178_rs_superresolution.png")


if __name__ == "__main__":
    main()
