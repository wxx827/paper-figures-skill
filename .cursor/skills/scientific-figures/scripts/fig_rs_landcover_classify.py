"""Figure 174 - Land-cover classification map with accuracy assessment.

Large thematic map (7 classes) plus a compact statistics column: error
matrix with producer's / user's accuracy, class area shares and an
overall-accuracy card.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, distance_transform_edt, maximum_filter

from pubstyle import (set_pub_style, PALETTES, savefig, add_panel_label,
                      style_axes)

# ---------------------------------------------------------------------------
# Synthetic multispectral scene generator (self-contained)
# ---------------------------------------------------------------------------
WATER, CROP, FOREST, GRASS, URBAN, SOIL, ROAD = range(7)
CLASS_NAMES = ["Water", "Cropland", "Forest", "Grassland",
               "Built-up", "Bare soil", "Road"]
CLASS_SHORT = ["Wat", "Crop", "For", "Grs", "Bui", "Soil", "Rd"]
CLASS_COLORS = ["#3b7ea1", "#8fbc4a", "#1d6b3f", "#d8d878",
                "#b23a48", "#d9a86c", "#6f6f6f"]

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


def stretch(a, lo=1.5, hi=98.5, gamma=1.0):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1) ** gamma


def composite(bands, idx, gamma=0.82, lo=1.5, hi=98.5):
    g = gamma if np.iterable(gamma) else (gamma,) * 3
    return np.dstack([stretch(bands[i], lo, hi, gi) for i, gi in zip(idx, g)])


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")


def add_scale_bar(ax, n, mpp=10.0, length_m=1000.0, label="1 km", fs=6.5):
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
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])


def add_north_arrow(ax, n, fs=7.5):
    cx, cy = 0.935 * n, 0.105 * n
    s = 0.050 * n
    ax.add_patch(Polygon([[cx, cy - s], [cx - 0.42 * s, cy + 0.55 * s],
                          [cx, cy + 0.22 * s], [cx + 0.42 * s, cy + 0.55 * s]],
                         closed=True, facecolor="white", edgecolor="#1a1a1a",
                         linewidth=0.6, zorder=6))
    ax.text(cx, cy - 0.72 * s, "N", ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.9, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
# Classifier simulation
# ---------------------------------------------------------------------------
CONFUSION = {
    WATER:  [(ROAD, 0.020), (FOREST, 0.010)],
    CROP:   [(GRASS, 0.055), (FOREST, 0.030), (SOIL, 0.018)],
    FOREST: [(CROP, 0.060), (GRASS, 0.020)],
    GRASS:  [(CROP, 0.090), (SOIL, 0.045)],
    URBAN:  [(ROAD, 0.070), (SOIL, 0.045)],
    SOIL:   [(GRASS, 0.070), (URBAN, 0.055)],
    ROAD:   [(URBAN, 0.115), (CROP, 0.035)],
}


def predict(lab, seed=5):
    """Simulate a classifier: errors concentrate along class boundaries."""
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    edge = (maximum_filter(lab, 3) != lab).astype(float)
    edge = gaussian_filter(edge, 1.6)
    edge /= edge.max() + 1e-9
    score = 0.55 * rng.random((n, n)) + 0.45 * edge
    pred = lab.copy()
    for cls, targets in CONFUSION.items():
        idx = np.flatnonzero(lab.ravel() == cls)
        if idx.size == 0:
            continue
        order = idx[np.argsort(-score.ravel()[idx])]
        cut = 0
        for tgt, rate in targets:
            k = int(round(rate * idx.size))
            pred.ravel()[order[cut:cut + k]] = tgt
            cut += k
    return pred


def error_matrix(truth, pred, n_samples=2800, seed=8):
    rng = np.random.default_rng(seed)
    flat = rng.choice(truth.size, size=n_samples, replace=False)
    t = truth.ravel()[flat]
    p = pred.ravel()[flat]
    cm = np.zeros((7, 7), int)
    for a, b in zip(t, p):
        cm[a, b] += 1
    return cm


def accuracy_stats(cm):
    tot = cm.sum()
    oa = np.trace(cm) / tot
    pe_ = (cm.sum(0) * cm.sum(1)).sum() / tot ** 2
    kappa = (oa - pe_) / (1 - pe_)
    pa = np.diag(cm) / np.maximum(cm.sum(1), 1)
    ua = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * pa * ua / np.maximum(pa + ua, 1e-9)
    return oa, kappa, pa, ua, f1


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n, mpp = 256, 10.0
    bands, truth, aux = make_scene(n=n, seed=11, tex_seed=3)
    pred = predict(truth, seed=5)
    cm = error_matrix(truth, pred)
    oa, kappa, pa, ua, f1 = accuracy_stats(cm)

    cmap_lc = ListedColormap(CLASS_COLORS)
    norm_lc = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), 7)

    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.24, 1.0],
                          height_ratios=[1.28, 0.86, 0.24],
                          wspace=0.20, hspace=0.42,
                          left=0.045, right=0.975, top=0.905, bottom=0.045)
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_leg = fig.add_subplot(gs[2, 0])
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_card = fig.add_subplot(gs[2, 1])
    ax_leg.axis("off")
    ax_card.axis("off")

    # ---- (a) thematic map -------------------------------------------------
    image_axes(ax_map)
    ax_map.imshow(pred, cmap=cmap_lc, norm=norm_lc, interpolation="nearest")
    ax_map.set_title("Random-forest land-cover map  (10 m, 7 classes)",
                     fontsize=8.2, fontweight="bold", pad=3.5)
    add_scale_bar(ax_map, n, mpp, 1000.0, "1 km")
    add_north_arrow(ax_map, n)
    add_panel_label(ax_map, "(a)", x=0.0, y=1.035, size=9.5)

    ins = ax_map.inset_axes([0.025, 0.025, 0.27, 0.27])
    ins.imshow(composite(bands, (2, 1, 0), gamma=(0.72, 0.74, 0.80),
                         lo=2.0, hi=97.5))
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.8)
        s.set_color("white")
    ins.set_title("True colour", fontsize=6.0, pad=1.5, color="white",
                  path_effects=[pe.withStroke(linewidth=1.8,
                                              foreground="#1a1a1a")])

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.5,
                     label=f"{nm}")
               for c, nm in zip(CLASS_COLORS, CLASS_NAMES)]
    leg = ax_leg.legend(handles=handles, loc="upper center", ncol=4,
                        frameon=False, fontsize=7.0, handlelength=1.2,
                        handleheight=1.0, columnspacing=1.1,
                        labelspacing=0.5, borderaxespad=0.0,
                        bbox_to_anchor=(0.5, 1.25))
    leg.set_clip_on(False)

    # ---- (b) error matrix -------------------------------------------------
    row_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap_cm = LinearSegmentedColormap.from_list(
        "cm", PALETTES["cb_sequential_blue"])
    ax_cm.imshow(row_norm, cmap=cmap_cm, vmin=0, vmax=1,
                 interpolation="nearest")
    for i in range(7):
        for j in range(7):
            v = cm[i, j]
            if v == 0:
                continue
            ax_cm.text(j, i, str(v), ha="center", va="center", fontsize=6.0,
                       color="white" if row_norm[i, j] > 0.5 else "#333333",
                       fontweight="bold" if i == j else "normal")
    for i in range(7):
        ax_cm.add_patch(Rectangle((6.55, i - 0.5), 1.0, 1.0, clip_on=False,
                                  facecolor="#eef2f5", edgecolor="#cccccc",
                                  linewidth=0.4))
        ax_cm.text(7.05, i, f"{100 * pa[i]:.0f}", ha="center", va="center",
                   fontsize=6.0, color="#1a1a1a", clip_on=False)
        ax_cm.add_patch(Rectangle((i - 0.5, 6.55), 1.0, 1.0, clip_on=False,
                                  facecolor="#f7f0ea", edgecolor="#cccccc",
                                  linewidth=0.4))
        ax_cm.text(i, 7.05, f"{100 * ua[i]:.0f}", ha="center", va="center",
                   fontsize=6.0, color="#1a1a1a", clip_on=False)
    ax_cm.set_xticks(range(7))
    ax_cm.set_yticks(range(7))
    ax_cm.set_xticklabels(CLASS_SHORT, fontsize=6.2)
    ax_cm.set_yticklabels(CLASS_SHORT, fontsize=6.2)
    ax_cm.tick_params(length=1.8, pad=1.5)
    ax_cm.set_xlim(-0.5, 7.55)
    ax_cm.set_ylim(7.55, -0.5)
    ax_cm.set_xlabel("Predicted class", fontsize=7.2, labelpad=1.5)
    ax_cm.set_ylabel("Reference class", fontsize=7.2, labelpad=1.5)
    ax_cm.text(7.05, -0.78, "PA %", ha="center", va="center", fontsize=6.0,
               fontweight="bold", color="#1a1a1a")
    ax_cm.text(-1.35, 7.05, "UA %", ha="center", va="center", fontsize=6.0,
               fontweight="bold", color="#1a1a1a")
    for s in ax_cm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax_cm.set_title(f"Error matrix  (n = {cm.sum():,})",
                    fontsize=7.8, fontweight="bold", pad=9.0, x=0.44)
    add_panel_label(ax_cm, "(b)", x=-0.20, y=1.035, size=9.5)

    # ---- (c) class areas --------------------------------------------------
    px_km2 = (mpp ** 2) / 1e6
    area = np.array([(pred == c).sum() for c in range(7)]) * px_km2
    share = 100 * area / area.sum()
    order = np.argsort(area)
    ax_bar.barh(np.arange(7), area[order],
                color=[CLASS_COLORS[i] for i in order],
                edgecolor="#333333", linewidth=0.5, height=0.68)
    for k, i in enumerate(order):
        ax_bar.text(area[i] + 0.045, k, f"{area[i]:.2f}  ({share[i]:.1f}%)",
                    va="center", ha="left", fontsize=6.0, color="#333333")
    ax_bar.set_yticks(np.arange(7))
    ax_bar.set_yticklabels([CLASS_NAMES[i] for i in order], fontsize=6.4)
    ax_bar.set_xlim(0, area.max() * 1.52)
    ax_bar.set_xlabel(r"Mapped area (km$^2$)", fontsize=7.2, labelpad=1.5)
    ax_bar.tick_params(length=2, pad=1.5, labelsize=6.2)
    ax_bar.grid(axis="x", linewidth=0.4, alpha=0.6)
    style_axes(ax_bar)
    ax_bar.set_title(r"Class area  (total 6.55 km$^2$)", fontsize=7.8,
                     fontweight="bold", pad=3.0)
    add_panel_label(ax_bar, "(c)", x=-0.20, y=1.06, size=9.5)

    # ---- (d) accuracy card ------------------------------------------------
    chips = [("Overall accuracy", f"{oa:.3f}"), ("Kappa", f"{kappa:.3f}"),
             ("Macro F1", f"{f1.mean():.3f}")]
    for i, (k, v) in enumerate(chips):
        x0 = 0.015 + i * 0.332
        ax_card.add_patch(FancyBboxPatch(
            (x0, 0.06), 0.305, 0.80, transform=ax_card.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            facecolor="#f2f5f8", edgecolor="#b9c6d1", linewidth=0.6,
            clip_on=False))
        ax_card.text(x0 + 0.152, 0.62, v, transform=ax_card.transAxes,
                     ha="center", va="center", fontsize=9.0,
                     fontweight="bold", color="#1b3f5c")
        ax_card.text(x0 + 0.152, 0.24, k, transform=ax_card.transAxes,
                     ha="center", va="center", fontsize=6.0, color="#4a5b68")


    savefig(fig, "174_rs_landcover_classify.png")


if __name__ == "__main__":
    main()
