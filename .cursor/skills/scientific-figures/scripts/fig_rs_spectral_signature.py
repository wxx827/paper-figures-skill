"""Figure 175 - Spectral signatures and feature-space separability.

Laboratory-style reflectance curves (400-2500 nm) with +/-1 sigma envelopes,
Sentinel-2 band positions, the red / NIR feature space with 95% ellipses and
a Jeffries-Matusita separability matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import (set_pub_style, PALETTES, savefig, add_panel_label,
                      style_axes)

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)
CLASS_NAMES = ["Water", "Cropland", "Forest", "Grassland",
               "Built-up", "Bare soil", "Road"]

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


def make_scene(n=256, seed=11, tex_seed=3):
    lab, aux = build_labels(n, seed)
    return synthesize(lab, aux, tex_seed), lab, aux


# ---------------------------------------------------------------------------
# Continuous laboratory spectra (400-2500 nm)
# ---------------------------------------------------------------------------
LIB = {
    "Water":      dict(color="#1f6fa8",
                       wl=[400, 450, 500, 550, 600, 650, 700, 760, 850, 1000,
                           1200, 1500, 1800, 2100, 2400, 2500],
                       rf=[0.055, 0.062, 0.058, 0.050, 0.036, 0.026, 0.018,
                           0.013, 0.010, 0.008, 0.006, 0.005, 0.004, 0.004,
                           0.003, 0.003]),
    "Cropland":   dict(color="#4c9f38",
                       wl=[400, 450, 500, 550, 600, 650, 680, 700, 720, 750,
                           800, 900, 1000, 1100, 1250, 1350, 1400, 1450, 1500,
                           1650, 1750, 1850, 1930, 2000, 2100, 2200, 2350,
                           2500],
                       rf=[0.030, 0.032, 0.045, 0.092, 0.070, 0.042, 0.035,
                           0.055, 0.190, 0.415, 0.455, 0.470, 0.450, 0.455,
                           0.400, 0.290, 0.075, 0.150, 0.270, 0.245, 0.200,
                           0.090, 0.020, 0.075, 0.145, 0.130, 0.080, 0.062]),
    "Forest":     dict(color="#0e4d2a",
                       wl=[400, 450, 500, 550, 600, 650, 680, 700, 720, 750,
                           800, 900, 1000, 1100, 1250, 1350, 1400, 1450, 1500,
                           1650, 1750, 1850, 1930, 2000, 2100, 2200, 2350,
                           2500],
                       rf=[0.024, 0.026, 0.035, 0.070, 0.052, 0.032, 0.027,
                           0.042, 0.145, 0.330, 0.365, 0.375, 0.360, 0.360,
                           0.310, 0.215, 0.055, 0.110, 0.195, 0.168, 0.135,
                           0.060, 0.014, 0.050, 0.098, 0.086, 0.052, 0.040]),
    "Grassland":  dict(color="#c9a227",
                       wl=[400, 450, 500, 550, 600, 650, 680, 700, 720, 750,
                           800, 900, 1000, 1100, 1250, 1350, 1400, 1450, 1500,
                           1650, 1750, 1850, 1930, 2000, 2100, 2200, 2350,
                           2500],
                       rf=[0.052, 0.058, 0.078, 0.118, 0.108, 0.086, 0.078,
                           0.098, 0.175, 0.262, 0.285, 0.298, 0.292, 0.300,
                           0.278, 0.230, 0.085, 0.150, 0.245, 0.268, 0.230,
                           0.115, 0.030, 0.100, 0.215, 0.190, 0.128, 0.105]),
    "Built-up":   dict(color="#b03a2e",
                       wl=[400, 450, 500, 550, 600, 650, 700, 760, 850, 1000,
                           1200, 1350, 1400, 1450, 1500, 1650, 1750, 1850,
                           1930, 2000, 2100, 2200, 2350, 2500],
                       rf=[0.112, 0.124, 0.136, 0.148, 0.152, 0.158, 0.168,
                           0.182, 0.196, 0.212, 0.232, 0.240, 0.130, 0.180,
                           0.245, 0.272, 0.258, 0.170, 0.055, 0.150, 0.272,
                           0.268, 0.246, 0.232]),
    "Bare soil":  dict(color="#8c5b2f",
                       wl=[400, 450, 500, 550, 600, 650, 700, 760, 850, 1000,
                           1200, 1350, 1400, 1450, 1500, 1650, 1750, 1850,
                           1930, 2000, 2100, 2160, 2210, 2280, 2350, 2500],
                       rf=[0.096, 0.122, 0.152, 0.184, 0.212, 0.238, 0.262,
                           0.288, 0.306, 0.328, 0.352, 0.360, 0.190, 0.260,
                           0.360, 0.392, 0.372, 0.245, 0.070, 0.210, 0.368,
                           0.372, 0.298, 0.336, 0.318, 0.290]),
}

# Sentinel-2 MSI: (name, centre nm, bandwidth nm)
S2_BANDS = [("B1", 443, 20), ("B2", 490, 65), ("B3", 560, 35),
            ("B4", 665, 30), ("B5", 705, 15), ("B6", 740, 15),
            ("B7", 783, 20), ("B8", 842, 115), ("B8a", 865, 20),
            ("B9", 945, 20), ("B11", 1610, 90), ("B12", 2190, 180)]
S2_LABELLED = {"B2", "B3", "B4", "B8", "B11", "B12"}
H2O_WINDOWS = [(1340, 1465), (1790, 1960)]


def jm_distance(xa, xb):
    """Jeffries-Matusita distance between two Gaussian classes (0-2)."""
    # work in percent reflectance so the 5x5 determinants stay well scaled
    xa, xb = 100.0 * xa, 100.0 * xb
    ma, mb = xa.mean(0), xb.mean(0)
    ca, cb = np.cov(xa.T), np.cov(xb.T)
    cm = 0.5 * (ca + cb)
    d = (mb - ma)[:, None]
    reg = 1e-6 * np.eye(cm.shape[0])
    b = (0.125 * float((d.T @ np.linalg.inv(cm + reg) @ d).item())
         + 0.5 * np.log(max(np.linalg.det(cm + reg), 1e-30)
                        / max(np.sqrt(np.linalg.det(ca + reg)
                                      * np.linalg.det(cb + reg)), 1e-30)))
    return 2.0 * (1.0 - np.exp(-b))


def conf_ellipse(ax, x, y, color, nsig=2.447):
    cov = np.cov(x, y)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    e = Ellipse((x.mean(), y.mean()), 2 * nsig * np.sqrt(w[-1]),
                2 * nsig * np.sqrt(w[0]), angle=ang, facecolor="none",
                edgecolor=color, linewidth=1.1, zorder=4)
    ax.add_patch(e)
    return e


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(21)
    bands, lab, aux = make_scene(n=256, seed=11, tex_seed=3)

    names = ["Water", "Cropland", "Forest", "Grassland", "Built-up",
             "Bare soil"]
    ids = [WATER, CROP, FOREST, GRASS, URBAN, SOIL]
    colors = [LIB[k]["color"] for k in names]

    fig = plt.figure(figsize=(7.5, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0],
                          height_ratios=[1.0, 0.80],
                          wspace=0.26, hspace=0.62,
                          left=0.065, right=0.985, top=0.885, bottom=0.105)
    gl = gs[:, 0].subgridspec(2, 1, height_ratios=[1.0, 0.115], hspace=0.045)
    ax_sp = fig.add_subplot(gl[0])
    ax_bd = fig.add_subplot(gl[1], sharex=ax_sp)
    ax_fs = fig.add_subplot(gs[0, 1])
    ax_jm = fig.add_subplot(gs[1, 1])

    # ---- (a) spectral library --------------------------------------------
    wlg = np.linspace(400, 2500, 900)
    for nm in names:
        d = LIB[nm]
        f = PchipInterpolator(d["wl"], d["rf"])
        y = np.clip(f(wlg), 0.0, 1.0)
        sd = 0.11 * y + 0.006 + 0.02 * np.exp(-((wlg - 800) / 260.0) ** 2) * y
        ax_sp.fill_between(wlg, y - sd, y + sd, color=d["color"], alpha=0.16,
                           linewidth=0)
        ax_sp.plot(wlg, y, color=d["color"], linewidth=1.5, label=nm,
                   zorder=3)

    trans = blended_transform_factory(ax_sp.transData, ax_sp.transAxes)
    for a, b in H2O_WINDOWS:
        ax_sp.add_patch(Rectangle((a, 0.0), b - a, 1.0, transform=trans,
                                  facecolor="#d8d8d8", alpha=0.55,
                                  edgecolor="none", zorder=1))
        ax_sp.text(0.5 * (a + b), 0.985, "H$_2$O", transform=trans,
                   ha="center", va="top", fontsize=6.0, color="#777777",
                   zorder=5)

    ax_sp.set_xlim(400, 2500)
    ax_sp.set_ylim(0, 0.62)
    ax_sp.set_ylabel("Surface reflectance (0-1)", labelpad=2)
    ax_sp.grid(axis="y", linewidth=0.4, alpha=0.55)
    style_axes(ax_sp)
    ax_sp.tick_params(labelbottom=False)
    ax_sp.set_title("Mean spectral signatures with $\\pm$1$\\sigma$ envelope",
                    fontsize=8.4, fontweight="bold", pad=4)
    add_panel_label(ax_sp, "(a)", x=-0.085, y=1.02, size=9.5)
    leg = ax_sp.legend(loc="upper right", bbox_to_anchor=(1.0, 0.945),
                       ncol=2, fontsize=6.5, frameon=True,
                       framealpha=0.94, edgecolor="#cccccc", handlelength=1.4,
                       labelspacing=0.28, columnspacing=1.0, borderpad=0.35)
    leg.get_frame().set_linewidth(0.5)
    ax_sp.annotate("red edge", xy=(722, 0.220), xytext=(830, 0.095),
                   fontsize=6.0, color="#2f6b28",
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#2f6b28"))
    ax_sp.annotate("NIR plateau", xy=(1010, 0.462), xytext=(760, 0.520),
                   fontsize=6.0, color="#2f6b28",
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#2f6b28"))
    ax_sp.annotate("clay 2200 nm", xy=(2210, 0.298), xytext=(2050, 0.150),
                   fontsize=6.0, color="#7a4f28",
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#7a4f28"))

    # ---- Sentinel-2 band positions (own strip, shared x) ------------------
    for nm, c, w in S2_BANDS:
        ax_bd.add_patch(Rectangle((c - w / 2, 0.28), w, 0.62,
                                  facecolor="#3a5d8f", alpha=0.70,
                                  edgecolor="#22406b", linewidth=0.35))
        if nm in S2_LABELLED:
            ax_bd.text(c, 0.02, nm, ha="center", va="bottom", fontsize=6.0,
                       color="#22406b")
    for a, b in H2O_WINDOWS:
        ax_bd.add_patch(Rectangle((a, 0.0), b - a, 1.0, facecolor="#d8d8d8",
                                  alpha=0.55, edgecolor="none", zorder=0))
    ax_bd.set_ylim(0, 1.0)
    ax_bd.set_yticks([])
    ax_bd.set_xlabel("Wavelength (nm)", labelpad=2)
    ax_bd.set_xticks([400, 700, 1000, 1300, 1600, 1900, 2200, 2500])
    ax_bd.tick_params(length=2.2, labelsize=6.6)
    for s in ax_bd.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#666666")
    ax_bd.text(1.005, 0.5, "S-2 MSI", transform=ax_bd.transAxes,
               ha="left", va="center", fontsize=6.0, color="#22406b")

    # ---- (b) red / NIR feature space -------------------------------------
    feats = {}
    for nm, cid in zip(names, ids):
        m = np.flatnonzero(lab.ravel() == cid)
        take = rng.choice(m, size=min(420, m.size), replace=False)
        feats[nm] = np.stack([bands[k].ravel()[take] for k in range(5)], 1)

    for k in (0.2, 0.5, 0.8):
        xs = np.array([0.0, 0.45])
        ax_fs.plot(xs, xs * (1 + k) / (1 - k), color="#aaaaaa", linewidth=0.55,
                   linestyle=(0, (4, 2)), zorder=1)
        ax_fs.text(0.560 * (1 - k) / (1 + k), 0.572, f"{k:.1f}", fontsize=6.0,
                   color="#8a8a8a", ha="center", va="top", zorder=1,
                   bbox=dict(boxstyle="square,pad=0.08", facecolor="white",
                             edgecolor="none", alpha=0.85))
    ax_fs.text(0.412, 0.505, "NDVI iso-lines", fontsize=6.0, color="#8a8a8a",
               ha="right", va="center")
    ax_fs.plot([0, 0.42], [0, 0.42], color="#8a8a8a", linewidth=0.6,
               zorder=1)
    ax_fs.text(0.372, 0.352, "soil line", fontsize=6.0, color="#7a7a7a",
               ha="center", va="top")

    for nm, c in zip(names, colors):
        f = feats[nm]
        ax_fs.scatter(f[:, 2], f[:, 3], s=2.2, color=c, alpha=0.30,
                      linewidths=0, zorder=2)
        conf_ellipse(ax_fs, f[:, 2], f[:, 3], c)
        ax_fs.scatter([f[:, 2].mean()], [f[:, 3].mean()], s=13, color=c,
                      edgecolor="white", linewidth=0.6, zorder=5)
    ax_fs.set_xlim(0, 0.42)
    ax_fs.set_ylim(0, 0.58)
    ax_fs.set_xlabel(r"Red reflectance  $\rho_{665}$", labelpad=2,
                     fontsize=7.4)
    ax_fs.set_ylabel(r"NIR reflectance  $\rho_{842}$", labelpad=2,
                     fontsize=7.4)
    ax_fs.tick_params(labelsize=6.4)
    ax_fs.grid(linewidth=0.35, alpha=0.5)
    style_axes(ax_fs)
    ax_fs.set_title("Red / NIR feature space", fontsize=7.8,
                    fontweight="bold", pad=4)
    add_panel_label(ax_fs, "(b)", x=-0.22, y=1.02, size=9.5)

    # ---- (c) separability -------------------------------------------------
    jm = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j:
                jm[i, j] = jm_distance(feats[names[i]], feats[names[j]])
    cmap = LinearSegmentedColormap.from_list(
        "jm", PALETTES["cb_sequential_green"])
    masked = np.ma.masked_where(np.eye(6, dtype=bool), jm)
    ax_jm.imshow(masked, cmap=cmap, vmin=1.0, vmax=2.0,
                 interpolation="nearest")
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            ax_jm.text(j, i, f"{jm[i, j]:.2f}", ha="center", va="center",
                       fontsize=6.0,
                       color="white" if jm[i, j] > 1.75 else "#333333")
    short = ["Wat", "Crop", "For", "Grs", "Bui", "Soil"]
    ax_jm.set_xticks(range(6))
    ax_jm.set_yticks(range(6))
    ax_jm.set_xticklabels(short, fontsize=6.0)
    ax_jm.set_yticklabels(short, fontsize=6.0)
    ax_jm.tick_params(length=1.8, pad=1.2)
    for s in ax_jm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax_jm.set_title("JM separability matrix", fontsize=7.6,
                    fontweight="bold", pad=4)
    add_panel_label(ax_jm, "(c)", x=-0.30, y=1.02, size=9.5)
    iu = np.triu_indices(6, 1)
    off = jm[iu]
    lo = int(np.argmin(off))
    ax_jm.text(1.04, 0.5,
               f"range 0-2\n5 bands\n\npairs $\\geq$1.9:\n"
               f"{int((off >= 1.9).sum())} / 15\n\nweakest:\n"
               f"{short[iu[0][lo]]} / {short[iu[1][lo]]} = {off[lo]:.2f}",
               transform=ax_jm.transAxes, fontsize=6.0, color="#4a5b68",
               va="center", ha="left", linespacing=1.35)


    savefig(fig, "175_rs_spectral_signature.png")


if __name__ == "__main__":
    main()
