"""Figure 181 - Hyperspectral data cube.

A 120-band cube rendered in 3D (false-colour top face, two spectral slice
walls), single-band slices at four diagnostic wavelengths and the endmember
spectra extracted from the cube.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import (set_pub_style, PALETTES, savefig, add_panel_label,
                      style_axes)

# ---------------------------------------------------------------------------
# Synthetic scene generator (self-contained)
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


# ---------------------------------------------------------------------------
# Continuous laboratory spectra -> 120-band cube
# ---------------------------------------------------------------------------
LIB = {
    WATER: ([400, 450, 500, 550, 600, 650, 700, 760, 850, 1000, 1200, 1500,
             1800, 2100, 2400, 2500],
            [0.055, 0.062, 0.058, 0.050, 0.036, 0.026, 0.018, 0.013, 0.010,
             0.008, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003]),
    CROP:  ([400, 450, 500, 550, 600, 650, 680, 700, 720, 750, 800, 900, 1000,
             1100, 1250, 1350, 1400, 1450, 1500, 1650, 1750, 1850, 1930, 2000,
             2100, 2200, 2350, 2500],
            [0.030, 0.032, 0.045, 0.092, 0.070, 0.042, 0.035, 0.055, 0.190,
             0.415, 0.455, 0.470, 0.450, 0.455, 0.400, 0.290, 0.075, 0.150,
             0.270, 0.245, 0.200, 0.090, 0.020, 0.075, 0.145, 0.130, 0.080,
             0.062]),
    FOREST: ([400, 450, 500, 550, 600, 650, 680, 700, 720, 750, 800, 900,
              1000, 1100, 1250, 1350, 1400, 1450, 1500, 1650, 1750, 1850,
              1930, 2000, 2100, 2200, 2350, 2500],
             [0.024, 0.026, 0.035, 0.070, 0.052, 0.032, 0.027, 0.042, 0.145,
              0.330, 0.365, 0.375, 0.360, 0.360, 0.310, 0.215, 0.055, 0.110,
              0.195, 0.168, 0.135, 0.060, 0.014, 0.050, 0.098, 0.086, 0.052,
              0.040]),
    GRASS: ([400, 450, 500, 550, 600, 650, 680, 700, 720, 750, 800, 900, 1000,
             1100, 1250, 1350, 1400, 1450, 1500, 1650, 1750, 1850, 1930, 2000,
             2100, 2200, 2350, 2500],
            [0.052, 0.058, 0.078, 0.118, 0.108, 0.086, 0.078, 0.098, 0.175,
             0.262, 0.285, 0.298, 0.292, 0.300, 0.278, 0.230, 0.085, 0.150,
             0.245, 0.268, 0.230, 0.115, 0.030, 0.100, 0.215, 0.190, 0.128,
             0.105]),
    URBAN: ([400, 450, 500, 550, 600, 650, 700, 760, 850, 1000, 1200, 1350,
             1400, 1450, 1500, 1650, 1750, 1850, 1930, 2000, 2100, 2200, 2350,
             2500],
            [0.112, 0.124, 0.136, 0.148, 0.152, 0.158, 0.168, 0.182, 0.196,
             0.212, 0.232, 0.240, 0.130, 0.180, 0.245, 0.272, 0.258, 0.170,
             0.055, 0.150, 0.272, 0.268, 0.246, 0.232]),
    SOIL:  ([400, 450, 500, 550, 600, 650, 700, 760, 850, 1000, 1200, 1350,
             1400, 1450, 1500, 1650, 1750, 1850, 1930, 2000, 2100, 2160, 2210,
             2280, 2350, 2500],
            [0.096, 0.122, 0.152, 0.184, 0.212, 0.238, 0.262, 0.288, 0.306,
             0.328, 0.352, 0.360, 0.190, 0.260, 0.360, 0.392, 0.372, 0.245,
             0.070, 0.210, 0.368, 0.372, 0.298, 0.336, 0.318, 0.290]),
}
ENDMEMBERS = [("Vegetation", CROP, "#4c9f38"), ("Forest", FOREST, "#0e4d2a"),
              ("Built-up", URBAN, "#b03a2e"), ("Bare soil", SOIL, "#8c5b2f"),
              ("Water", WATER, "#1f6fa8")]
BAD_BANDS = [(1340, 1465), (1790, 1960)]


def build_cube(n=96, nb=120, seed=11, tex_seed=3):
    lab, aux = build_labels(256, seed=seed)
    y0, x0 = 32, 44
    lab = lab[y0:y0 + 2 * n:2, x0:x0 + 2 * n:2].copy()
    vig = aux["vig"][y0:y0 + 2 * n:2, x0:x0 + 2 * n:2].copy()
    rng = np.random.default_rng(tex_seed)

    wl = np.linspace(400, 2500, nb)
    ref = np.zeros((7, nb))
    for cls, (w, r) in LIB.items():
        ref[cls] = np.clip(PchipInterpolator(w, r)(wl), 0.0, 1.0)
    ref[ROAD] = ref[URBAN] * 0.78

    cube = ref[lab]                                   # (n, n, nb)
    veg = np.isin(lab, [CROP, FOREST, GRASS])
    fc = np.clip(vig, 0.05, 1.0)[..., None]
    dry = (0.74 + 0.20 * _noise(rng, n, 12))[..., None]
    cube = np.where(veg[..., None], fc * cube + (1 - fc) * ref[SOIL] * dry,
                    cube)

    amp = np.zeros((n, n))
    for cls, a in TEX_AMP.items():
        amp[lab == cls] = a
    tex = 1 + amp * (0.55 * _noise(rng, n, 1.0) + 0.30 * _noise(rng, n, 2.5)
                     + 0.25 * _noise(rng, n, 5.0))
    cube *= tex[..., None]

    # detector noise, rising in the water-vapour windows
    snr = np.ones(nb) * 0.0035
    for a, b in BAD_BANDS:
        snr += 0.030 * np.exp(-0.5 * ((wl - 0.5 * (a + b)) / 45.0) ** 2)
    cube += snr * rng.standard_normal(cube.shape)
    return np.clip(cube, 0.0005, 1.2), lab, wl


def face_image(arr, cmap, vmin, vmax):
    return cmap(Normalize(vmin, vmax)(arr))


def stretch(a, lo=2.0, hi=98.0, gamma=0.8):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1) ** gamma


def main():
    set_pub_style(base_font=8)
    n, nb = 96, 120
    cube, lab, wl = build_cube(n=n, nb=nb)
    print(f"  cube {cube.shape}, {wl[0]:.0f}-{wl[-1]:.0f} nm, "
          f"{wl[1] - wl[0]:.1f} nm sampling")

    def band(w):
        return int(np.argmin(np.abs(wl - w)))

    false_rgb = np.dstack([stretch(cube[..., band(w)])
                           for w in (860, 660, 550)])
    cmap_side = LinearSegmentedColormap.from_list(
        "side", PALETTES["cb_sequential_blue"][::-1])

    fig = plt.figure(figsize=(7.6, 4.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.52, 1.0],
                          height_ratios=[0.74, 1.0],
                          wspace=0.02, hspace=0.40,
                          left=0.005, right=0.975, top=0.885, bottom=0.095)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axb = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])

    # ---- (a) the cube -----------------------------------------------------
    st = 2
    xs = np.arange(0, n, st)
    zs = np.arange(0, nb, st)
    xf = np.arange(n)
    Xf, Yf = np.meshgrid(xf, xf)
    ax3.plot_surface(Xf, Yf, np.full_like(Xf, nb, dtype=float), rstride=1,
                     cstride=1, facecolors=false_rgb, shade=False, zorder=2)

    wall_v = (0.0, 0.42)
    XZ, ZX = np.meshgrid(xs, zs)
    front = cube[0, ::st, ::st].T                      # (nz, nx)
    ax3.plot_surface(XZ, np.zeros_like(XZ, dtype=float), ZX, rstride=1,
                     cstride=1, shade=False, zorder=1,
                     facecolors=face_image(front, cmap_side, *wall_v))
    YZ, ZY = np.meshgrid(xs, zs)
    right = cube[::st, -1, ::st].T
    ax3.plot_surface(np.full_like(YZ, n, dtype=float), YZ, ZY, rstride=1,
                     cstride=1, shade=False, zorder=1,
                     facecolors=face_image(right, cmap_side, *wall_v))

    for (a, b) in [((0, 0, nb), (n, 0, nb)), ((n, 0, nb), (n, n, nb)),
                   ((n, n, nb), (0, n, nb)), ((0, n, nb), (0, 0, nb)),
                   ((0, 0, 0), (n, 0, 0)), ((n, 0, 0), (n, n, 0)),
                   ((0, 0, 0), (0, 0, nb)), ((n, 0, 0), (n, 0, nb)),
                   ((n, n, 0), (n, n, nb))]:
        ax3.plot(*zip(a, b), color="#2b2b2b", linewidth=0.5, zorder=4)

    ax3.set_box_aspect((1, 1, 0.72))
    ax3.view_init(elev=21, azim=-58)
    ax3.set_xlim(0, n)
    ax3.set_ylim(0, n)
    ax3.set_zlim(0, nb)
    ax3.set_zticks([band(w) for w in (500, 900, 1300, 1700, 2100, 2500)])
    ax3.set_zticklabels(["500", "900", "1300", "1700", "2100", "2500"],
                        fontsize=6.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zlabel("")
    ax3.tick_params(axis="z", pad=-2)
    ax3.grid(False)
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1, 1, 1, 0))
        pane.line.set_color((1, 1, 1, 0))
    ax3.set_title("120-band cube  |  top: false colour 860/660/550 nm",
                  fontsize=7.6, fontweight="bold", pad=-2, y=0.99)
    ax3.text2D(0.02, 0.97, "(a)", transform=ax3.transAxes, fontsize=9.5,
               fontweight="bold", va="top")
    ax3.text2D(0.905, 0.135, "Wavelength\n(nm)", transform=ax3.transAxes,
               fontsize=6.3, color="#1a1a1a", ha="center", va="center",
               linespacing=1.25)
    ax3.text2D(0.02, 0.075, "side walls: reflectance of the cube edges "
               "across all 120 bands", transform=ax3.transAxes, fontsize=6.0,
               color="#666666", ha="left", va="center")
    ax3.text2D(0.60, 0.86, f"{n} x {n} px\n20 m GSD",
               transform=ax3.transAxes, fontsize=6.0, color="#555555",
               ha="left", va="center", linespacing=1.3)

    # ---- (b) single-band slices -------------------------------------------
    axb.set_xticks([])
    axb.set_yticks([])
    for s in axb.spines.values():
        s.set_visible(False)
    axb.set_title("Single-band slices", fontsize=7.6, fontweight="bold",
                  pad=3.0)
    add_panel_label(axb, "(b)", x=-0.03, y=1.03, size=9.5)
    picks = [(490, "blue, haze"), (660, "red, chlorophyll"),
             (860, "NIR, biomass"), (1650, "SWIR, moisture")]
    for i, (w, note) in enumerate(picks):
        r, c = divmod(i, 4)
        sx = axb.inset_axes([0.005 + 0.253 * i, 0.135, 0.225, 0.735])
        sx.imshow(stretch(cube[..., band(w)], gamma=0.85), cmap="gray")
        sx.set_aspect("equal")
        sx.set_xticks([])
        sx.set_yticks([])
        for s in sx.spines.values():
            s.set_linewidth(0.6)
            s.set_color("#333333")
        sx.set_title(f"{w} nm", fontsize=6.0, fontweight="bold", pad=1.8)
        sx.text(0.5, -0.045, note, transform=sx.transAxes, ha="center",
                va="top", fontsize=6.0, color="#666666")

    # ---- (c) endmember spectra --------------------------------------------
    for name, cls, col in ENDMEMBERS:
        m = lab == cls
        if m.sum() < 20:
            continue
        mu = cube[m].mean(0)
        sd = cube[m].std(0)
        axe.fill_between(wl, mu - sd, mu + sd, color=col, alpha=0.15,
                         linewidth=0)
        axe.plot(wl, mu, color=col, linewidth=1.15, label=name)
    for a, b in BAD_BANDS:
        axe.axvspan(a, b, color="#d8d8d8", alpha=0.55, linewidth=0, zorder=0)
    axe.set_xlim(400, 2500)
    axe.set_ylim(0, 0.50)
    axe.set_xticks([500, 900, 1300, 1700, 2100, 2500])
    axe.set_xlabel("Wavelength (nm)", fontsize=7.0, labelpad=1.5)
    axe.set_ylabel("Reflectance", fontsize=7.0, labelpad=1.5)
    axe.tick_params(labelsize=6.2)
    axe.grid(axis="y", linewidth=0.35, alpha=0.55)
    style_axes(axe)
    axe.set_title("Endmember spectra extracted from the cube",
                  fontsize=7.4, fontweight="bold", pad=3.0)
    add_panel_label(axe, "(c)", x=-0.115, y=1.04, size=9.5)
    leg = axe.legend(loc="upper right", ncol=2, fontsize=6.0, frameon=True,
                     framealpha=0.93, edgecolor="#cccccc", handlelength=1.3,
                     labelspacing=0.25, columnspacing=0.9, borderpad=0.3)
    leg.get_frame().set_linewidth(0.5)
    axe.text(1650, 0.052, "noisy H$_2$O bands", fontsize=6.0, color="#777777",
             ha="center", va="bottom")


    savefig(fig, "181_rs_hyperspectral_cube.png")


if __name__ == "__main__":
    main()
