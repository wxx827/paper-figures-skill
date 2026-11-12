"""Figure 173 - Multispectral band-composite comparison.

Six renderings of the same synthetic Sentinel-2-like scene: true colour,
standard false colour, SWIR composite, NDVI, NDWI and a single NIR band.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)
CLASS_NAMES = ["Water", "Cropland", "Forest", "Grassland",
               "Built-up", "Bare soil", "Road"]

# Mean surface reflectance for Blue, Green, Red, NIR, SWIR1
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
    """Euclidean distance (px) to a rasterised polyline."""
    m = np.zeros((n, n), bool)
    pts = np.asarray(pts, float)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        k = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
        ii = np.clip(np.round(np.linspace(y0, y1, k)).astype(int), 0, n - 1)
        jj = np.clip(np.round(np.linspace(x0, x1, k)).astype(int), 0, n - 1)
        m[ii, jj] = True
    return distance_transform_edt(~m)


def build_labels(n=256, seed=11):
    """Land-cover label map with fields, river, roads, city and forest."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    # --- agricultural mosaic: four farm blocks with distinct orientations --
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

    # --- forest patches ----------------------------------------------------
    f2 = _fbm(rng, n, (5, 13, 30), (0.34, 0.34, 0.32))
    zone = _noise(rng, n, 40) > -0.35
    forest = gaussian_filter((f2 > 0.68).astype(float), 2.0) > 0.42
    lab[forest & zone] = FOREST
    vig = np.where(lab == FOREST, 0.86 + 0.09 * _noise(rng, n, 4), vig)
    vig = np.where(lab == GRASS, 0.46 + 0.10 * _noise(rng, n, 6), vig)

    # --- centre-pivot irrigated circles ------------------------------------
    wob = _noise(rng, n, 3)
    for (cx, cy, rr) in [(0.125, 0.45, 0.050), (0.235, 0.56, 0.040),
                         (0.075, 0.63, 0.034), (0.185, 0.66, 0.030)]:
        m = np.hypot(xx - cx * n, yy - cy * n) < rr * n * (1 + 0.03 * wob)
        lab[m] = CROP
        vig[m] = 0.93 + 0.05 * wob[m]

    # --- city plus a small town -------------------------------------------
    th = np.deg2rad(-8)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    blocks = ((u % 11.0) < 9.3) & ((v % 9.0) < 7.5)
    for (cx, cy, rad, thr) in [(0.70, 0.63, 0.155, 0.60), (0.36, 0.30, 0.062, 0.62)]:
        blob = (np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2)
                         / (2 * (rad * n) ** 2)))
                + 0.30 * _fbm(rng, n, (7, 17), (0.5, 0.5)))
        city = blob > thr
        lab[city & blocks] = URBAN
        lab[city & ~blocks] = ROAD

    # --- road network ------------------------------------------------------
    roads = [[(0, 0.20 * n), (0.44 * n, 0.34 * n), (0.70 * n, 0.63 * n),
              (n - 1, 0.74 * n)],
             [(0.70 * n, 0.63 * n), (0.62 * n, 0.05 * n)],
             [(0.05 * n, 0.95 * n), (0.45 * n, 0.72 * n), (0.70 * n, 0.63 * n)],
             [(0.02 * n, 0.55 * n), (n - 1, 0.44 * n)]]
    for pl in roads:
        lab[_polyline_dist(n, pl) <= 1.1] = ROAD

    # --- river and reservoir ----------------------------------------------
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
    """Turn a label map into a 5-band reflectance stack."""
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

    # Linear canopy/soil mixing: fractional cover drives the whole spectrum,
    # so NDVI spans a realistic 0.2-0.85 instead of saturating near 1.
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


def make_scene(n=256, seed=11, tex_seed=3):
    lab, aux = build_labels(n, seed)
    return synthesize(lab, aux, tex_seed), lab, aux


def stretch(a, lo=1.5, hi=98.5, gamma=1.0):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1) ** gamma


def composite(bands, idx, gamma=0.82, lo=1.5, hi=98.5):
    g = gamma if np.iterable(gamma) else (gamma,) * 3
    return np.dstack([stretch(bands[i], lo, hi, gi) for i, gi in zip(idx, g)])


def ndvi(b):
    return (b[3] - b[2]) / (b[3] + b[2] + 1e-9)


def ndwi(b):
    return (b[1] - b[3]) / (b[1] + b[3] + 1e-9)


# ---------------------------------------------------------------------------
# Map furniture
# ---------------------------------------------------------------------------
def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")


def add_scale_bar(ax, n, mpp=10.0, length_m=1000.0, label="1 km", fs=6.0):
    px = length_m / mpp
    x0 = n - px - 0.06 * n
    y0 = 0.935 * n
    h = 0.020 * n
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * px / 2, y0), px / 2, h, zorder=6,
                               facecolor="white" if i else "#111111",
                               edgecolor="#111111", linewidth=0.5))
    ax.text(x0 + px / 2, y0 - 0.012 * n, label, ha="center", va="bottom",
            fontsize=fs, color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="#1a1a1a")])


def add_north_arrow(ax, n, fs=7.0):
    cx, cy = 0.935 * n, 0.115 * n
    s = 0.052 * n
    tri = Polygon([[cx, cy - s], [cx - 0.42 * s, cy + 0.55 * s],
                   [cx, cy + 0.22 * s], [cx + 0.42 * s, cy + 0.55 * s]],
                  closed=True, facecolor="white", edgecolor="#1a1a1a",
                  linewidth=0.6, zorder=6)
    ax.add_patch(tri)
    ax.text(cx, cy - 0.72 * s, "N", ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])


def inset_cbar(fig, ax, mappable, label, ticks):
    ax.add_patch(FancyBboxPatch(
        (0.030, 0.788), 0.54, 0.190, transform=ax.transAxes,
        boxstyle="round,pad=0.008,rounding_size=0.02", facecolor="white",
        edgecolor="#8a8a8a", linewidth=0.5, alpha=0.92, zorder=8))
    cax = ax.inset_axes([0.065, 0.868, 0.455, 0.034])
    cax.set_zorder(9)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8, pad=1.0,
                      color="#333333")
    ax.text(0.065, 0.945, label, transform=ax.transAxes, fontsize=6.0,
            ha="left", va="center", color="#1a1a1a", zorder=9,
            fontweight="bold")


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n, mpp = 256, 10.0
    bands, lab, aux = make_scene(n=n, seed=11, tex_seed=3)

    ndvi_map = ndvi(bands)
    ndwi_map = ndwi(bands)

    cmap_ndvi = LinearSegmentedColormap.from_list(
        "ndvi", ["#7a4a1e", "#c8a35a", "#f2eecb", "#9dcf78", "#3f8f43",
                 "#12481f"])
    cmap_ndwi = LinearSegmentedColormap.from_list(
        "ndwi", ["#6b4a20", "#c9b083", "#f4f1e6", "#9ec9e8", "#2a6fae",
                 "#123a63"])

    fig = plt.figure(figsize=(7.4, 5.5))
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.30,
                          left=0.025, right=0.975, top=0.925, bottom=0.055)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    titles = ["True colour  (R-G-B)", "Standard false colour  (NIR-R-G)",
              "SWIR composite  (SWIR-NIR-R)", "NDVI  (NIR-Red)/(NIR+Red)",
              "NDWI  (Green-NIR)/(Green+NIR)", "Single band  NIR (B8)"]
    subs = ["B4/B3/B2 - visual interpretation",
            "B8/B4/B3 - vegetation vigour, red = canopy",
            "B11/B8/B4 - moisture and built-up separation",
            "Green vegetation > 0.3, water < 0",
            "Open water > 0, land < 0",
            "B8 reflectance, 10 m GSD"]

    for ax, t, s in zip(axes, titles, subs):
        image_axes(ax)
        ax.set_title(t, fontsize=7.6, fontweight="bold", pad=3.0)
        ax.text(0.5, -0.055, s, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color="#555555")

    axes[0].imshow(composite(bands, (2, 1, 0), gamma=(0.72, 0.74, 0.80),
                             lo=2.0, hi=97.5))
    axes[1].imshow(composite(bands, (3, 2, 1), gamma=(0.85, 0.78, 0.78)))
    axes[2].imshow(composite(bands, (4, 3, 2), gamma=(0.85, 0.85, 0.78)))
    im3 = axes[3].imshow(ndvi_map, cmap=cmap_ndvi, vmin=-0.6, vmax=0.9)
    im4 = axes[4].imshow(ndwi_map, cmap=cmap_ndwi, vmin=-0.8, vmax=0.8)
    im5 = axes[5].imshow(bands[3], cmap="gray", vmin=0.0, vmax=0.55)

    inset_cbar(fig, axes[3], im3, "NDVI", [-0.5, 0.0, 0.5, 0.9])
    inset_cbar(fig, axes[4], im4, "NDWI", [-0.8, -0.3, 0.2, 0.7])
    inset_cbar(fig, axes[5], im5, r"$\rho$", [0.0, 0.2, 0.4])

    for i, ax in enumerate(axes):
        add_scale_bar(ax, n, mpp, 1000.0, "1 km")
        add_panel_label(ax, f"({'abcdef'[i]})", x=0.0, y=1.045, size=9)
    add_north_arrow(axes[0], n)


    savefig(fig, "173_rs_composite_bands.png")


if __name__ == "__main__":
    main()
