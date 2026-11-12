"""Figure 182 - Cloud, cirrus and shadow screening.

Contaminated acquisition, continuous cloud probability, the discrete quality
mask and the resulting per-date valid-pixel availability of a one-year
time series.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               BoundaryNorm)
from scipy.ndimage import gaussian_filter, distance_transform_edt, shift

from pubstyle import (set_pub_style, PALETTES, savefig, add_panel_label,
                      style_axes)

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


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")


# ---------------------------------------------------------------------------
# Cloud field, contamination and screening
# ---------------------------------------------------------------------------
# top-of-atmosphere cloud reflectance: bright and nearly flat, drops in SWIR
CLOUD_SPEC = np.array([0.86, 0.85, 0.84, 0.82, 0.62])


def cloud_field(n, seed, thick_q=0.90, thin_q=0.74):
    rng = np.random.default_rng(seed)
    f = _fbm(rng, n, (6, 16, 34), (0.42, 0.34, 0.24))
    f = gaussian_filter(f, 1.2)
    tt = np.quantile(f, thick_q)
    tn = np.quantile(f, thin_q)
    tau = np.clip((f - tn) / max(tt - tn, 1e-6), 0, None)
    tau = np.clip(tau, 0, 2.4)
    alpha = 1.0 - np.exp(-1.35 * tau)          # cloud opacity 0-1
    return alpha


SUN_OFFSET = (26, -34)          # shadow displacement for the solar geometry


def contaminate(bands, alpha, sun=SUN_OFFSET):
    shadow = shift(alpha, sun, order=1, mode="constant", cval=0.0)
    shadow = np.clip(shadow - alpha, 0, 1)
    out = bands.copy()
    for k in range(5):
        out[k] = out[k] * (1 - 0.62 * shadow)
        out[k] = (1 - alpha) * out[k] + alpha * CLOUD_SPEC[k]
    return np.clip(out, 0.0015, 1.2), shadow


def main():
    set_pub_style(base_font=8)
    n, mpp = 256, 10.0
    lab, aux = build_labels(n, seed=11)
    clear = synthesize(lab, aux, seed=3)
    alpha = cloud_field(n, seed=202)
    toa, shadow = contaminate(clear, alpha)

    # ---- detector: brightness + whiteness, then geometric shadow search ---
    vis = toa[:3].mean(0)
    white = 1.0 - (toa[:3].max(0) - toa[:3].min(0)) / (vis + 1e-6)
    score = 3.2 * (vis - 0.22) + 1.0 * (white - 0.86)
    prob = np.clip(gaussian_filter(1 / (1 + np.exp(-7.0 * score)), 1.0), 0, 1)

    cloudy = prob >= 0.35
    proj = shift(cloudy.astype(float), SUN_OFFSET, order=1, mode="constant",
                 cval=0.0)
    proj = gaussian_filter(proj, 2.5) > 0.25
    ndwi = (toa[1] - toa[3]) / (toa[1] + toa[3] + 1e-9)
    dark = (vis < 0.070) & (toa[4] < 0.17)

    mask = np.zeros((n, n), np.int16)                   # 0 clear land
    mask[(ndwi > 0.05) & ~cloudy] = 1                   # water
    mask[dark & proj & ~cloudy & (ndwi <= 0.05)] = 2    # cloud shadow
    mask[(prob >= 0.35) & (prob < 0.68)] = 3            # thin cloud
    mask[prob >= 0.68] = 4                              # thick cloud

    t_cloud, d_cloud = alpha > 0.12, mask >= 3
    t_shad, d_shad = shadow > 0.30, mask == 2
    pod = (d_cloud & t_cloud).sum() / max(t_cloud.sum(), 1)
    far = (d_cloud & ~t_cloud).sum() / max(d_cloud.sum(), 1)
    pod_s = (d_shad & t_shad).sum() / max(t_shad.sum(), 1)
    frac = [100.0 * np.count_nonzero(mask == i) / mask.size for i in range(5)]
    print(f"  clear {frac[0]:.1f}%  water {frac[1]:.1f}%  shadow "
          f"{frac[2]:.1f}%  thin {frac[3]:.1f}%  thick {frac[4]:.1f}%")
    print(f"  cloud POD {pod:.3f}  FAR {far:.3f}  shadow POD {pod_s:.3f}")

    MASK_COLORS = ["#ded7c4", "#2f7fb8", "#4c4368", "#cdd8ee", "#8ea6c4"]
    MASK_NAMES = ["Clear land", "Water", "Cloud shadow", "Thin cloud",
                  "Thick cloud"]
    cmap_mask = ListedColormap(MASK_COLORS)

    lims = [np.percentile(clear[k], [2.0, 97.5]) for k in range(5)]

    def rgb(stack):
        return np.dstack([np.clip((stack[k] - lims[k][0])
                                  / (lims[k][1] - lims[k][0]), 0, 1) ** g
                          for k, g in zip((2, 1, 0), (0.72, 0.74, 0.80))])

    cmap_prob = LinearSegmentedColormap.from_list(
        "prob", ["#f7f7f7", "#dbe4ef", "#9dbdd8", "#f4c48a", "#d98032",
                 "#8c2f2f"])

    fig = plt.figure(figsize=(7.2, 5.55))
    gs = fig.add_gridspec(2, 2, wspace=0.075, hspace=0.19,
                          left=0.035, right=0.965, top=0.905, bottom=0.105)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    for ax in (axa, axb, axc):
        image_axes(ax)

    axa.imshow(rgb(toa))
    axa.set_title("TOA reflectance  -  2023-07-08", fontsize=7.8,
                  fontweight="bold", pad=3.0)
    stroke = [pe.withStroke(linewidth=1.9, foreground="#1a1a1a")]
    inner = np.zeros((n, n))
    inner[40:n - 40, 40:n - 40] = 1.0
    cy, cx = np.unravel_index(np.argmax(gaussian_filter(alpha, 6) * inner),
                              alpha.shape)
    sy_, sx_ = np.unravel_index(np.argmax(gaussian_filter(shadow, 4) * inner),
                                shadow.shape)
    for txt, (py, px) in [("thick cloud", (cy, cx)),
                          ("cast shadow", (sy_, sx_))]:
        ty = py + (54 if py < n / 2 else -54)
        tx = px + (58 if px < n / 2 else -58)
        ty = int(np.clip(ty, 24, n - 20))
        tx = int(np.clip(tx, 36, n - 36))
        axa.annotate(txt, xy=(px, py), xytext=(tx, ty), fontsize=6.0,
                     color="white", ha="center", path_effects=stroke,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="white",
                                     path_effects=stroke))

    imp = axb.imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
    axb.set_title("Cloud probability", fontsize=7.8, fontweight="bold",
                  pad=3.0)
    axb.add_patch(FancyBboxPatch(
        (0.030, 0.788), 0.54, 0.190, transform=axb.transAxes,
        boxstyle="round,pad=0.008,rounding_size=0.02", facecolor="white",
        edgecolor="#8a8a8a", linewidth=0.5, alpha=0.92, zorder=8))
    cax = axb.inset_axes([0.065, 0.868, 0.455, 0.034])
    cax.set_zorder(9)
    cb = fig.colorbar(imp, cax=cax, orientation="horizontal",
                      ticks=[0, 0.35, 0.68, 1.0])
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8, pad=1.0,
                      color="#333333")
    axb.text(0.065, 0.945, "p(cloud)", transform=axb.transAxes, fontsize=6.0,
             ha="left", va="center", color="#1a1a1a", fontweight="bold",
             zorder=9)
    axb.text(0.965, 0.030,
             f"cloud POD {pod:.2f}   FAR {far:.2f}\nshadow POD {pod_s:.2f}",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=6.0,
             linespacing=1.3, zorder=9,
             bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                       edgecolor="#b0b0b0", linewidth=0.5, alpha=0.92))

    axc.imshow(mask, cmap=cmap_mask, vmin=-0.5, vmax=4.5,
               interpolation="nearest")
    axc.set_title("Quality mask  (5 classes)", fontsize=7.8,
                  fontweight="bold", pad=3.0)
    axc.text(0.030, 0.030,
             f"valid (clear + water) {frac[0] + frac[1]:.1f}%",
             transform=axc.transAxes, ha="left", va="bottom", fontsize=6.0,
             zorder=9,
             bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                       edgecolor="#b0b0b0", linewidth=0.5, alpha=0.92))

    for i, ax in enumerate((axa, axb, axc, axd)):
        add_panel_label(ax, f"({'abcd'[i]})", x=0.0, y=1.035, size=9.0)

    # ---- (d) time-series availability -------------------------------------
    axd.set_xticks([])
    axd.set_yticks([])
    for s in axd.spines.values():
        s.set_visible(False)
    axd.add_patch(FancyBboxPatch(
        (0.005, 0.015), 0.99, 0.97, transform=axd.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.03", facecolor="#f4f6f8",
        edgecolor="#c8d1d8", linewidth=0.7, zorder=0))
    axd.text(0.5, 0.985, "Time-series data availability",
             transform=axd.transAxes, ha="center", va="top", fontsize=7.4,
             fontweight="bold", color="#1a1a1a")

    rng = np.random.default_rng(9)
    dates = ["03-12", "03-27", "04-11", "04-26", "05-11", "05-26", "06-10",
             "06-25", "07-08", "07-25", "08-09", "08-24"]
    seas = 0.22 + 0.26 * np.sin(np.linspace(0.2, 3.1, len(dates)))
    thick = np.clip(seas * (0.35 + 1.1 * rng.random(len(dates))), 0.01, 0.50)
    thin = np.clip(0.32 * thick + 0.07 * rng.random(len(dates)), 0.005, 0.20)
    shad = np.clip(0.30 * thick + 0.01 * rng.random(len(dates)), 0.0, 0.18)
    k = dates.index("07-08")
    thick[k] = frac[4] / 100.0
    thin[k] = frac[3] / 100.0
    shad[k] = frac[2] / 100.0
    valid = np.clip(1 - thick - thin - shad, 0.02, 1.0)

    sub = axd.inset_axes([0.175, 0.165, 0.775, 0.720])
    y = np.arange(len(dates))
    left = np.zeros(len(dates))
    parts = [(valid, "#7fb069"), (shad, MASK_COLORS[2]),
             (thin, MASK_COLORS[3]), (thick, MASK_COLORS[4])]
    for vals, col in parts:
        sub.barh(y, 100 * vals, left=100 * left, color=col,
                 edgecolor="#555555", linewidth=0.35, height=0.72)
        left = left + vals
    for i, v in enumerate(valid):
        sub.text(101.5, i, f"{100 * v:.0f}%", va="center", ha="left",
                 fontsize=6.0, color="#333333",
                 fontweight="bold" if i == k else "normal")
    sub.set_yticks(y)
    sub.set_yticklabels(dates, fontsize=6.0)
    sub.get_yticklabels()[k].set_fontweight("bold")
    sub.get_yticklabels()[k].set_color("#b03a2e")
    sub.set_xlim(0, 112)
    sub.set_xticks([0, 25, 50, 75, 100])
    sub.invert_yaxis()
    sub.tick_params(length=1.6, pad=1.2, labelsize=6.0)
    sub.set_xlabel("Share of scene pixels (%)", fontsize=6.2, labelpad=1.5)
    sub.grid(axis="x", linewidth=0.3, alpha=0.55)
    style_axes(sub)
    sub.text(-0.235, k, "shown\nabove", transform=sub.get_yaxis_transform(),
             ha="center", va="center", fontsize=6.0, color="#b03a2e",
             linespacing=1.25)
    axd.text(0.5, 0.945, f"{len(dates)} acquisitions, mean valid fraction "
             f"{100 * valid.mean():.0f}%", transform=axd.transAxes,
             ha="center", va="top", fontsize=6.0, color="#4a5b68")

    handles = [Patch(facecolor=c, edgecolor="#555555", linewidth=0.4,
                     label=nm) for c, nm in zip(MASK_COLORS, MASK_NAMES)]
    handles.insert(2, Patch(facecolor="#7fb069", edgecolor="#555555",
                            linewidth=0.4, label="Valid: clear + water"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                     frameon=False, fontsize=6.4, handlelength=1.25,
                     handleheight=0.95, columnspacing=1.3,
                     bbox_to_anchor=(0.5, 0.005))
    leg.set_clip_on(False)


    savefig(fig, "182_rs_cloud_mask.png")


if __name__ == "__main__":
    main()
