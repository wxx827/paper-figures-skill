"""Figure 176 - Pan-sharpening comparison (Wald protocol).

Low-resolution multispectral input, high-resolution panchromatic band and
three fusion results, each with an identically placed zoom inset, plus a
compact ERGAS / SAM / Q4 summary panel.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.ndimage import gaussian_filter, distance_transform_edt, zoom

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
    s = n / 256.0

    anchors = np.array([[0.20, 0.22], [0.74, 0.16],
                        [0.26, 0.76], [0.82, 0.70]]) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1])
                      + 12 * s * _noise(rng, n, 22 * s) for a in anchors])
    reg = np.argmin(dists, axis=0)

    fid = np.zeros((n, n), int)
    for r, (ang, cu, cv) in enumerate([(17, 27, 35), (-26, 31, 22),
                                       (39, 23, 29), (6, 35, 26)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 4.5 * s * _noise(rng, n, 16 * s)
        v = -xx * np.sin(th) + yy * np.cos(th) + 4.5 * s * _noise(rng, n, 16 * s)
        f = (np.floor(u / (cu * s)).astype(int) * 211
             + np.floor(v / (cv * s)).astype(int) * 37 + r * 1013)
        fid = np.where(reg == r, f, fid)

    lab = np.full((n, n), CROP, np.int16)
    key = np.abs(fid) % 4096
    lut_v = rng.random(4096)
    lut_k = rng.random(4096)
    vig = 0.34 + 0.64 * lut_v[key]
    kind = lut_k[key]
    lab[kind < 0.10] = SOIL
    lab[(kind >= 0.10) & (kind < 0.24)] = GRASS

    f2 = _fbm(rng, n, (5 * s, 13 * s, 30 * s), (0.34, 0.34, 0.32))
    zone = _noise(rng, n, 40 * s) > -0.35
    forest = gaussian_filter((f2 > 0.68).astype(float), 2.0 * s) > 0.42
    lab[forest & zone] = FOREST
    vig = np.where(lab == FOREST, 0.86 + 0.09 * _noise(rng, n, 4 * s), vig)
    vig = np.where(lab == GRASS, 0.46 + 0.10 * _noise(rng, n, 6 * s), vig)

    wob = _noise(rng, n, 3 * s)
    for (cx, cy, rr) in [(0.125, 0.45, 0.050), (0.235, 0.56, 0.040),
                         (0.075, 0.63, 0.034), (0.185, 0.66, 0.030)]:
        m = np.hypot(xx - cx * n, yy - cy * n) < rr * n * (1 + 0.03 * wob)
        lab[m] = CROP
        vig[m] = 0.93 + 0.05 * wob[m]

    th = np.deg2rad(-8)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    blocks = ((u % (11.0 * s)) < 9.3 * s) & ((v % (9.0 * s)) < 7.5 * s)
    for (cx, cy, rad, thr) in [(0.70, 0.63, 0.155, 0.60),
                               (0.36, 0.30, 0.062, 0.62)]:
        blob = (np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2)
                         / (2 * (rad * n) ** 2)))
                + 0.30 * _fbm(rng, n, (7 * s, 17 * s), (0.5, 0.5)))
        city = blob > thr
        lab[city & blocks] = URBAN
        lab[city & ~blocks] = ROAD

    roads = [[(0, 0.20 * n), (0.44 * n, 0.34 * n), (0.70 * n, 0.63 * n),
              (n - 1, 0.74 * n)],
             [(0.70 * n, 0.63 * n), (0.62 * n, 0.05 * n)],
             [(0.05 * n, 0.95 * n), (0.45 * n, 0.72 * n), (0.70 * n, 0.63 * n)],
             [(0.02 * n, 0.55 * n), (n - 1, 0.44 * n)]]
    for pl in roads:
        lab[_polyline_dist(n, pl) <= 1.1 * s] = ROAD

    ty = np.linspace(0, n - 1, 240)
    tx = (0.30 * n + 0.105 * n * np.sin(2 * np.pi * ty / n * 1.05)
          + 0.045 * n * np.sin(2 * np.pi * ty / n * 2.6 + 1.3))
    dr = _polyline_dist(n, np.c_[tx, ty])
    lab[dr <= (3.3 + 1.1 * _noise(rng, n, 26 * s)) * s] = WATER
    sy = np.linspace(0.30 * n, 0.615 * n, 160)
    tributary_x = np.interp(sy, ty, tx)
    frac = (sy - sy[0]) / (sy[-1] - sy[0])
    sx = (1 - frac ** 1.4) * 0.97 * n + (frac ** 1.4) * tributary_x \
        + 7 * s * np.sin(sy / (17.0 * s)) * (1 - frac)
    ds = _polyline_dist(n, np.c_[sx, sy])
    lab[ds <= (1.5 + 0.45 * _noise(rng, n, 20 * s)) * s] = WATER
    lake = ((((xx - 0.155 * n) / (0.080 * n)) ** 2
             + ((yy - 0.845 * n) / (0.048 * n)) ** 2)
            < 1 + 0.30 * _noise(rng, n, 5 * s))
    lab[lake] = WATER

    aux = {"vig": np.clip(vig, 0.15, 1.05), "fid": fid, "reg": reg}
    return lab, aux


def synthesize(lab, aux, seed=3, noise=1.0, gain=1.0):
    n = lab.shape[0]
    s = n / 256.0
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
    dryness = 0.74 + 0.20 * _noise(rng, n, 18 * s)
    for k in range(5):
        mixed = fc * bands[k] + (1.0 - fc) * soil_bg[k] * dryness
        bands[k] = np.where(veg, mixed, bands[k])

    amp = np.zeros((n, n))
    for cls, a in TEX_AMP.items():
        amp[lab == cls] = a
    tex = 1 + amp * noise * (0.55 * _noise(rng, n, 1.0 * s)
                             + 0.30 * _noise(rng, n, 2.5 * s)
                             + 0.25 * _noise(rng, n, 6.0 * s))
    illum = 1 + 0.05 * _noise(rng, n, 35 * s)
    yg, xg = np.mgrid[0:n, 0:n] / (n - 1.0) - 0.5
    vign = 1 - 0.11 * (xg ** 2 + yg ** 2) / 0.5

    for k in range(5):
        b = bands[k] * tex * illum * vign * gain
        b *= 1 + 0.22 * amp * _noise(rng, n, 0.8 * s)
        stripe = (1 + 0.010 * np.sin(np.arange(n) / (3.0 * s) + 1.7 * k)
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


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")


# ---------------------------------------------------------------------------
# Fusion algorithms and quality metrics
# ---------------------------------------------------------------------------
def degrade(stack, ratio=4):
    """MTF-like low-pass + decimation + bicubic re-interpolation."""
    lo = np.stack([gaussian_filter(b, 0.42 * ratio)[::ratio, ::ratio]
                   for b in stack])
    up = np.stack([zoom(b, ratio, order=3) for b in lo])
    return lo, np.clip(up, 1e-4, None)


def ergas(ref, test, ratio=4):
    acc = 0.0
    for r, t in zip(ref, test):
        acc += (np.sqrt(np.mean((r - t) ** 2)) / r.mean()) ** 2
    return 100.0 / ratio * np.sqrt(acc / len(ref))


def sam_deg(ref, test):
    r = ref.reshape(len(ref), -1)
    t = test.reshape(len(test), -1)
    num = (r * t).sum(0)
    den = np.linalg.norm(r, axis=0) * np.linalg.norm(t, axis=0) + 1e-12
    return float(np.degrees(np.arccos(np.clip(num / den, -1, 1))).mean())


def q_index(ref, test):
    """Universal image quality index averaged over bands (Q4 style)."""
    vals = []
    for r, t in zip(ref, test):
        mr, mt = r.mean(), t.mean()
        vr, vt = r.var(), t.var()
        cv = ((r - mr) * (t - mt)).mean()
        vals.append((4 * cv * mr * mt)
                    / ((vr + vt) * (mr ** 2 + mt ** 2) + 1e-12))
    return float(np.mean(vals))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4)
    n, ratio = 256, 4
    gt, lab, aux = make_scene(n=n, seed=11, tex_seed=3)
    ms_gt = gt[:4]                       # B, G, R, NIR at 2.5 m (reference)
    ms_lo, ms_up = degrade(ms_gt, ratio)  # 10 m MS, bicubic to 2.5 m

    # Panchromatic band: broad visible+NIR response at full resolution
    pan = (0.22 * gt[0] + 0.30 * gt[1] + 0.32 * gt[2] + 0.16 * gt[3])
    pan = pan + 0.002 * rng.standard_normal((n, n))

    inten = ms_up.mean(0)
    pan_m = (pan - pan.mean()) * (inten.std() / pan.std()) + inten.mean()

    # Brovey: band-dependent intensity ratio -> strong hue shift
    expo = np.array([1.55, 1.15, 0.95, 0.70])[:, None, None]
    brovey = np.clip(ms_up * (pan_m / (inten + 1e-6)) ** expo, 1e-4, 1.2)

    # Gram-Schmidt: spectrally faithful, but the global injection gain
    # over- / under-sharpens locally (an achromatic, hue-preserving error)
    d = pan_m - inten
    gs = np.stack([ms_up[k] + (np.cov(ms_up[k].ravel(), inten.ravel())[0, 1]
                               / (inten.var() + 1e-9)) * d for k in range(4)])
    gs = 0.70 * gs + 0.30 * ms_gt
    ach = gaussian_filter(np.random.default_rng(9).standard_normal((n, n)), 2.2)
    gs = np.clip(gs * (1 + 0.09 * ach / ach.std()), 1e-4, 1.2)

    # CNN fusion: sharpest and most accurate, but the residual concentrates
    # in NIR, the band the panchromatic detector barely covers
    hue = np.stack([gaussian_filter(
        np.random.default_rng(31 + i).standard_normal((n, n)), 5.0)
        for i in range(4)])
    hue /= hue.std()
    amp = np.array([0.020, 0.020, 0.020, 0.130])[:, None, None]
    cnn = 0.68 * ms_gt + 0.32 * gs
    cnn = cnn * (1 + amp * hue) + 0.004 * rng.standard_normal(ms_gt.shape)
    cnn = np.clip(cnn, 1e-4, 1.2)

    results = [("Brovey transform", brovey), ("Gram-Schmidt", gs),
               ("CNN fusion (ours)", cnn)]
    mets = {nm: (ergas(ms_gt, r, ratio), sam_deg(ms_gt, r), q_index(ms_gt, r))
            for nm, r in results}
    mets["Bicubic MS"] = (ergas(ms_gt, ms_up, ratio), sam_deg(ms_gt, ms_up),
                          q_index(ms_gt, ms_up))
    for k, v in mets.items():
        print(f"  {k:20s} ERGAS={v[0]:.2f}  SAM={v[1]:.2f} deg  Q4={v[2]:.3f}")

    # ---- rendering with a single shared stretch --------------------------
    lims = [np.percentile(ms_gt[k], [2.0, 97.5]) for k in range(4)]
    gam = (0.72, 0.74, 0.80)

    def rgb(stack):
        out = []
        for i, (k, g) in enumerate(zip((2, 1, 0), gam)):
            lo, hi = lims[k]
            out.append(np.clip((stack[k] - lo) / (hi - lo), 0, 1) ** g)
        return np.dstack(out)

    plo, phi = np.percentile(pan, [2.0, 97.5])
    pan_img = np.clip((pan - plo) / (phi - plo), 0, 1) ** 0.85

    # zoom window (identical for every panel)
    zx, zy, zs = 148, 132, 52

    fig = plt.figure(figsize=(7.5, 5.5))
    gs_ = fig.add_gridspec(2, 3, wspace=0.075, hspace=0.26,
                           left=0.022, right=0.978, top=0.905, bottom=0.035)
    axes = [fig.add_subplot(gs_[i, j]) for i in range(2) for j in range(3)]

    panels = [("Low-resolution MS  (10 m)", rgb(ms_up), None),
              ("Panchromatic  (2.5 m)", pan_img, "gray"),
              ("Brovey transform", rgb(brovey), None),
              ("Gram-Schmidt (GS)", rgb(gs), None),
              ("CNN fusion (ours)", rgb(cnn), None)]

    for i, (title, img, cm) in enumerate(panels):
        ax = axes[i]
        image_axes(ax)
        ax.imshow(img, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None,
                  interpolation="nearest")
        ax.set_title(title, fontsize=7.3, fontweight="bold", pad=3.0)
        ax.add_patch(Rectangle((zx, zy), zs, zs, facecolor="none",
                               edgecolor="#b23a48", linewidth=0.9, zorder=5))
        ins = ax.inset_axes([0.015, 0.015, 0.40, 0.40])
        sub = img[zy:zy + zs, zx:zx + zs]
        ins.imshow(sub, cmap=cm, vmin=0 if cm else None,
                   vmax=1 if cm else None, interpolation="nearest")
        ins.set_xticks([])
        ins.set_yticks([])
        for s in ins.spines.values():
            s.set_linewidth(1.0)
            s.set_color("#b23a48")
        if i >= 2:
            key = ["Brovey transform", "Gram-Schmidt",
                   "CNN fusion (ours)"][i - 2]
            e, sa, q = mets[key]
            ax.text(0.985, 0.982,
                    f"ERGAS {e:.2f}\nSAM {sa:.2f}$^\\circ$\nQ4 {q:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.0, color="#1a1a1a", zorder=7, linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                              edgecolor="#b0b0b0", linewidth=0.5, alpha=0.90))
        add_panel_label(ax, f"({'abcdef'[i]})", x=0.0, y=1.035, size=9.0)

    axes[0].text(0.985, 0.982, "reference GSD 2.5 m\ndegraded to 10 m",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=6.0, color="#1a1a1a", zorder=7, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                           edgecolor="#b0b0b0", linewidth=0.5, alpha=0.90))

    # ---- (f) supporting metric panel --------------------------------------
    axf = axes[5]
    axf.set_xticks([])
    axf.set_yticks([])
    for s in axf.spines.values():
        s.set_visible(False)
    axf.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96, transform=axf.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.03", facecolor="#f4f6f8",
        edgecolor="#c8d1d8", linewidth=0.7, zorder=0))
    axf.text(0.5, 0.995, "Quantitative assessment (Wald protocol)",
             transform=axf.transAxes, ha="center", va="top", fontsize=6.9,
             fontweight="bold", color="#1a1a1a")
    add_panel_label(axf, "(f)", x=0.0, y=1.035, size=9.0)

    meth = ["Bicubic MS", "Brovey transform", "Gram-Schmidt",
            "CNN fusion (ours)"]
    short = ["Bicubic", "Brovey", "GS", "Ours"]
    cols = ["#9aa5ad", "#d98032", "#4c78a8", "#2e7d5b"]
    runtime = [0.03, 0.05, 0.24, 4.80]
    specs = [("ERGAS", 0, r"$\downarrow$", "{:.2f}"),
             ("SAM (deg)", 1, r"$\downarrow$", "{:.2f}"),
             ("Q4", 2, r"$\uparrow$", "{:.3f}"),
             ("GPU runtime / 1024$^2$ tile (s)", 3, r"$\downarrow$",
              "{:.2f}")]
    for r, (nm, idx, arrow, fmt) in enumerate(specs):
        sub = axf.inset_axes([0.175, 0.715 - r * 0.235, 0.70, 0.160])
        vals = runtime if idx == 3 else [mets[m][idx] for m in meth]
        sub.barh(np.arange(4), vals, color=cols, edgecolor="#333333",
                 linewidth=0.4, height=0.74)
        for i, v in enumerate(vals):
            sub.text(v + 0.025 * max(vals), i, fmt.format(v), va="center",
                     ha="left", fontsize=6.0, color="#333333")
        sub.set_yticks(np.arange(4))
        sub.set_yticklabels(short, fontsize=6.0)
        sub.set_xlim(0, max(vals) * 1.32)
        sub.set_xticks([])
        sub.invert_yaxis()
        sub.tick_params(length=1.5, pad=1.0, labelsize=6.0)
        style_axes(sub)
        sub.spines["bottom"].set_linewidth(0.5)
        sub.set_title(f"{nm}  {arrow} better", fontsize=6.0, pad=1.6,
                      color="#33424d")


    savefig(fig, "176_rs_pansharpening.png")


if __name__ == "__main__":
    main()
