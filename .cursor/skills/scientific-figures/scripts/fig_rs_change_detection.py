"""Figure 177 - Bi-temporal change detection.

T1 / T2 imagery, a continuous change-probability layer, the thresholded
binary mask, a from-to change-type map and the land-cover transition
matrix, with change-area statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Patch
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               BoundaryNorm, LogNorm)
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)
CLASS_NAMES = ["Water", "Cropland", "Forest", "Grassland",
               "Built-up", "Bare soil", "Road"]
CLASS_SHORT = ["Wat", "Crop", "For", "Grs", "Bui", "Soil", "Rd"]

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
    lake_r = ((((xx - 0.155 * n) / (0.080 * n)) ** 2
               + ((yy - 0.845 * n) / (0.048 * n)) ** 2)
              / (1 + 0.30 * _noise(rng, n, 5)))
    lab[lake_r < 1] = WATER

    aux = {"vig": np.clip(vig, 0.15, 1.05), "fid": fid, "reg": reg,
           "lake_r": lake_r, "river_d": dr}
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


def stretch(a, lo=1.5, hi=98.5, gamma=1.0):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1) ** gamma


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
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


def add_north_arrow(ax, n, fs=7.0):
    cx, cy = 0.935 * n, 0.115 * n
    s = 0.052 * n
    ax.add_patch(Polygon([[cx, cy - s], [cx - 0.42 * s, cy + 0.55 * s],
                          [cx, cy + 0.22 * s], [cx + 0.42 * s, cy + 0.55 * s]],
                         closed=True, facecolor="white", edgecolor="#1a1a1a",
                         linewidth=0.6, zorder=6))
    ax.text(cx, cy - 0.72 * s, "N", ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
# Temporal evolution T1 -> T2
# ---------------------------------------------------------------------------
def evolve(lab, aux, n=256, seed=77):
    """Apply five realistic land-cover transitions between the two dates."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    lab2 = lab.copy()
    vig2 = aux["vig"].copy()
    convertible = np.isin(lab, [CROP, GRASS, SOIL])

    # 1. urban expansion in coherent blocks along the city fringe
    d_city = np.hypot(xx - 0.70 * n, yy - 0.63 * n)
    seed_f = (d_city < 0.290 * n) & (d_city > 0.130 * n) & \
        (_noise(rng, n, 13) > -0.25)
    fringe = (gaussian_filter(seed_f.astype(float), 3.5) > 0.45) & convertible
    lab2[fringe] = URBAN

    # 2. a new industrial estate next to the highway, with its own
    #    block-and-street grid so the footprint does not look stamped on
    th = np.deg2rad(12.0)
    eu = (xx - 0.455 * n) * np.cos(th) + (yy - 0.190 * n) * np.sin(th)
    ev = -(xx - 0.455 * n) * np.sin(th) + (yy - 0.190 * n) * np.cos(th)
    wob = _noise(rng, n, 6)
    estate = ((np.abs(eu) < 0.088 * n * (1 + 0.16 * wob))
              & (np.abs(ev) < 0.055 * n * (1 + 0.16 * wob)) & convertible)
    eblocks = ((eu % 13.0) < 10.0) & ((ev % 11.0) < 8.0)
    lab2[estate & eblocks] = URBAN
    lab2[estate & ~eblocks] = ROAD

    # 3. deforestation in the north-west
    defo = ((lab == FOREST) & (_noise(rng, n, 12) > 0.10)
            & (xx < 0.60 * n) & (yy < 0.64 * n))
    lab2[defo] = SOIL
    vig2[defo] = 0.10

    # 4. harvested / fallow fields (crop rotation)
    key = np.abs(aux["fid"]) % 4096
    rot = rng.random(4096)[key]
    harvest = (lab == CROP) & (rot < 0.17)
    lab2[harvest] = SOIL
    vig2[harvest] = 0.08

    # 5. reservoir filling and shoreline retreat
    newres = ((((xx - 0.585 * n) / (0.055 * n)) ** 2
               + ((yy - 0.845 * n) / (0.036 * n)) ** 2)
              < 1 + 0.25 * _noise(rng, n, 5)) & convertible
    lab2[newres] = WATER
    retreat = (aux["lake_r"] > 0.55) & (aux["lake_r"] < 1.0)
    lab2[retreat] = SOIL
    vig2[retreat] = 0.12

    # vigour drift of the unchanged vegetation (phenology, not a change)
    same = lab2 == lab
    veg = np.isin(lab2, [CROP, FOREST, GRASS]) & same
    vig2[veg] = np.clip(vig2[veg] * (0.975 + 0.05 * rng.random(veg.sum())),
                        0.12, 1.05)
    aux2 = dict(aux)
    aux2["vig"] = vig2
    return lab2, aux2


def main():
    set_pub_style(base_font=8)
    n, mpp = 256, 10.0
    px_ha = (mpp ** 2) / 1e4          # 0.01 ha per 10 m pixel
    px_km2 = (mpp ** 2) / 1e6

    lab1, aux1 = build_labels(n, seed=11)
    lab2, aux2 = evolve(lab1, aux1, n=n, seed=77)
    b1 = synthesize(lab1, aux1, seed=3)
    b2 = synthesize(lab2, aux2, seed=19, gain=1.03)

    truth = lab1 != lab2

    # ---- change-vector analysis ------------------------------------------
    d = b2 - b1
    cvm = np.sqrt((d ** 2).sum(0))
    cvm = gaussian_filter(cvm, 0.8)
    prob = 1.0 / (1.0 + np.exp(-(cvm - 0.163) / 0.026))
    prob = np.clip(gaussian_filter(prob, 1.0), 0, 1)
    det = prob > 0.5

    tp = int((det & truth).sum())
    fp = int((det & ~truth).sum())
    fn = int((~det & truth).sum())
    tn = int((~det & ~truth).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    oa = (tp + tn) / truth.size
    print(f"  changed {100 * truth.mean():.1f}%   P={prec:.3f} R={rec:.3f} "
          f"F1={f1:.3f} OA={oa:.3f}")

    # ---- change types ----------------------------------------------------
    TYPES = [("Cropland to built-up", (CROP, URBAN), "#b23a48"),
             ("Forest to bare soil", (FOREST, SOIL), "#e08214"),
             ("Cropland to bare soil", (CROP, SOIL), "#f4d06f"),
             ("New water body", (None, WATER), "#2c6fad"),
             ("Water to bare soil", (WATER, SOIL), "#7b3294"),
             ("Other transition", None, "#7f8c8d")]
    ctype = np.zeros((n, n), np.int16)      # 0 = no change
    changed = det & truth
    ctype[changed] = 6
    for i, (nm, pair, _c) in enumerate(TYPES):
        if pair is None:
            continue
        a, b = pair
        m = changed & (lab2 == b) if a is None else \
            changed & (lab1 == a) & (lab2 == b)
        ctype[m] = i + 1
    ctype[det & ~truth] = 6                 # false alarms fall in "other"

    # ---- transition matrix (ha) ------------------------------------------
    trans = np.zeros((7, 7))
    for i in range(7):
        mi = lab1 == i
        if not mi.any():
            continue
        for j in range(7):
            trans[i, j] = np.count_nonzero(mi & (lab2 == j)) * px_ha

    # ---- rendering --------------------------------------------------------
    lims = [np.percentile(b1[k], [2.0, 97.5]) for k in range(5)]
    gam = (0.72, 0.74, 0.80)

    def rgb(stack):
        return np.dstack([np.clip((stack[k] - lims[k][0])
                                  / (lims[k][1] - lims[k][0]), 0, 1) ** g
                          for k, g in zip((2, 1, 0), gam)])

    cmap_prob = LinearSegmentedColormap.from_list(
        "prob", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#8b1a1a"])
    cmap_bin = ListedColormap(["#eceff1", "#b2182b"])
    cmap_ct = ListedColormap(["#f2f2f2"] + [t[2] for t in TYPES])

    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 3, wspace=0.075, hspace=0.28,
                          left=0.022, right=0.978, top=0.900, bottom=0.075)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    for ax in axes[:5]:
        image_axes(ax)

    axes[0].imshow(rgb(b1))
    axes[0].set_title("T1  -  2019-06-14", fontsize=7.6, fontweight="bold",
                      pad=3.0)
    axes[1].imshow(rgb(b2))
    axes[1].set_title("T2  -  2024-06-21", fontsize=7.6, fontweight="bold",
                      pad=3.0)
    imp = axes[2].imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
    axes[2].set_title("Change probability", fontsize=7.6, fontweight="bold",
                      pad=3.0)
    axes[3].imshow(det.astype(int), cmap=cmap_bin, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[3].set_title("Binary change mask  (p > 0.5)", fontsize=7.6,
                      fontweight="bold", pad=3.0)
    axes[4].imshow(ctype, cmap=cmap_ct, vmin=-0.5, vmax=6.5,
                   interpolation="nearest")
    axes[4].set_title("Change type  (from - to)", fontsize=7.6,
                      fontweight="bold", pad=3.0)

    for i in range(5):
        add_scale_bar(axes[i], n, mpp, 1000.0, "1 km")
        add_panel_label(axes[i], f"({'abcde'[i]})", x=0.0, y=1.035, size=9.0)
    add_north_arrow(axes[0], n)

    # probability colourbar inside panel (c)
    axc = axes[2]
    axc.add_patch(FancyBboxPatch(
        (0.030, 0.788), 0.52, 0.190, transform=axc.transAxes,
        boxstyle="round,pad=0.008,rounding_size=0.02", facecolor="white",
        edgecolor="#8a8a8a", linewidth=0.5, alpha=0.92, zorder=8))
    cax = axc.inset_axes([0.065, 0.868, 0.445, 0.034])
    cax.set_zorder(9)
    cb = fig.colorbar(imp, cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1.0])
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8, pad=1.0,
                      color="#333333")
    axc.text(0.065, 0.945, "p(change)", transform=axc.transAxes, fontsize=6.0,
             ha="left", va="center", color="#1a1a1a", fontweight="bold",
             zorder=9)

    ch_km2 = truth.sum() * px_km2
    axes[3].text(0.030, 0.968,
                 f"changed {ch_km2:.2f} km$^2$ ({100 * truth.mean():.1f}%)\n"
                 f"precision {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}",
                 transform=axes[3].transAxes, ha="left", va="top",
                 fontsize=6.0, linespacing=1.3, zorder=9,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                           edgecolor="#b0b0b0", linewidth=0.5, alpha=0.92))

    # shared change-type legend under the bottom row
    areas = [np.count_nonzero(ctype == i + 1) * px_km2 for i in range(6)]
    handles = [Patch(facecolor=t[2], edgecolor="#333333", linewidth=0.4,
                     label=f"{t[0]}  ({a:.2f} km$^2$)")
               for t, a in zip(TYPES, areas)]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                     frameon=False, fontsize=6.2, handlelength=1.3,
                     handleheight=0.95, columnspacing=1.4, labelspacing=0.42,
                     bbox_to_anchor=(0.34, -0.008))
    leg.set_clip_on(False)

    # ---- (f) transition matrix as a supporting panel ----------------------
    axf = axes[5]
    axf.set_xticks([])
    axf.set_yticks([])
    for s in axf.spines.values():
        s.set_visible(False)
    axf.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96, transform=axf.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.03", facecolor="#f4f6f8",
        edgecolor="#c8d1d8", linewidth=0.7, zorder=0))
    axf.text(0.5, 0.995, "Transition matrix  (ha, 2019 to 2024)",
             transform=axf.transAxes, ha="center", va="top", fontsize=6.9,
             fontweight="bold", color="#1a1a1a")
    add_panel_label(axf, "(f)", x=0.0, y=1.035, size=9.0)

    sub = axf.inset_axes([0.190, 0.305, 0.68, 0.560])
    off = np.ma.masked_where(np.eye(7, dtype=bool) | (trans <= 0), trans)
    cmap_tr = LinearSegmentedColormap.from_list(
        "tr", PALETTES["cb_sequential_warm"])
    cmap_tr.set_bad("#ffffff")
    im = sub.imshow(off, cmap=cmap_tr, norm=LogNorm(vmin=0.5, vmax=200),
                    interpolation="nearest")
    for i in range(7):
        for j in range(7):
            if i == j or trans[i, j] <= 0:
                continue
            sub.text(j, i, f"{trans[i, j]:.0f}", ha="center", va="center",
                     fontsize=6.0,
                     color="white" if trans[i, j] > 60 else "#333333")
        sub.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#e4e9ed",
                                edgecolor="#ffffff", linewidth=0.4))
    sub.set_xticks(range(7))
    sub.set_yticks(range(7))
    sub.set_xticklabels(CLASS_SHORT, fontsize=6.0, rotation=45,
                        ha="right", rotation_mode="anchor")
    sub.set_yticklabels(CLASS_SHORT, fontsize=6.0)
    sub.tick_params(length=1.4, pad=0.8)
    sub.set_xlabel("to  (T2)", fontsize=6.0, labelpad=1.0)
    sub.set_ylabel("from  (T1)", fontsize=6.0, labelpad=1.0)
    for s in sub.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#666666")
    axf.text(0.5, 0.035, "grey diagonal = persistence (masked)\n"
             f"total transition area {ch_km2 * 100:.0f} ha",
             transform=axf.transAxes, ha="center", va="bottom", fontsize=6.0,
             color="#4a5b68", linespacing=1.35)


    savefig(fig, "177_rs_change_detection.png")


if __name__ == "__main__":
    main()
