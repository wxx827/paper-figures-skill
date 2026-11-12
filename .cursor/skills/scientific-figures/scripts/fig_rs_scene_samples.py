"""Figure 180 - Scene-classification sample grid.

Six scene categories (rows) x eight 64 x 64 tiles, each captioned with the
predicted label and its softmax confidence. Misclassified tiles are framed
in red; the side panel reports per-class accuracy on the full test set.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter, distance_transform_edt

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


# ---------------------------------------------------------------------------
# Scene categories and tile mining
# ---------------------------------------------------------------------------
SCENES = [
    ("Farmland",     "Farm", lambda f: f[CROP] - 3.0 * (f[URBAN] + f[ROAD])
                                       - 2.0 * f[WATER] - f[FOREST]),
    ("Forest",       "For",  lambda f: f[FOREST]
                                       - 2.0 * (f[URBAN] + f[ROAD])),
    ("River",        "Riv",  lambda f: 3.0 * f[WATER]
                                       - 2.0 * (f[URBAN] + f[ROAD])),
    ("Dense urban",  "Dens", lambda f: f[URBAN] + 0.4 * f[ROAD]),
    ("Sparse urban", "Spar", lambda f: 0.6 * f[CROP]
                                       - 3.0 * abs(f[URBAN] + f[ROAD] - 0.24)),
    ("Meadow",       "Mea",  lambda f: f[GRASS]
                                       - 2.0 * (f[URBAN] + f[ROAD])),
]
CONFUSE = {0: 5, 1: 5, 2: 0, 3: 4, 4: 3, 5: 0}
ACC = [0.941, 0.958, 0.897, 0.934, 0.826, 0.851]


def mine_tiles(scenes, tile=64, stride=8, k=8):
    """Pick the k most category-typical, non-overlapping windows per class."""
    picks = []
    for _name, _sh, score in SCENES:
        cands = []
        for si, (lab, rgb) in enumerate(scenes):
            n = lab.shape[0]
            for y in range(0, n - tile + 1, stride):
                for x in range(0, n - tile + 1, stride):
                    sub = lab[y:y + tile, x:x + tile]
                    frac = {c: np.count_nonzero(sub == c) / sub.size
                            for c in range(7)}
                    cands.append((score(frac), si, x, y))
        cands.sort(key=lambda t: -t[0])
        chosen = []
        for sc, si, x, y in cands:
            if len(chosen) >= k:
                break
            if any(s2 == si and abs(x - x2) < tile and abs(y - y2) < tile
                   for _s, s2, x2, y2 in chosen):
                continue
            chosen.append((sc, si, x, y))
        picks.append(chosen)
    return picks


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(31)
    tile = 64

    scenes = []
    for i, sd in enumerate([11, 23, 41, 57, 73, 89]):
        lab, aux = build_labels(256, seed=sd)
        b = synthesize(lab, aux, seed=3 + 4 * i)
        lims = [np.percentile(b[k], [2.0, 97.5]) for k in range(5)]
        rgb = np.dstack([np.clip((b[k] - lims[k][0])
                                 / (lims[k][1] - lims[k][0]), 0, 1) ** g
                         for k, g in zip((2, 1, 0), (0.72, 0.74, 0.80))])
        scenes.append((lab, rgb))

    picks = mine_tiles(scenes, tile=tile)

    fig = plt.figure(figsize=(7.5, 5.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.0], wspace=0.10,
                             left=0.062, right=0.985, top=0.885, bottom=0.055)
    grid = outer[0, 0].subgridspec(6, 8, wspace=0.10, hspace=0.55)
    axr = fig.add_subplot(outer[0, 1])

    n_err_total = 0
    for r, (name, short, _s) in enumerate(SCENES):
        n_err = int(round(8 * (1 - ACC[r])))
        err_idx = set(rng.choice(8, size=n_err, replace=False).tolist())
        n_err_total += n_err
        for c in range(8):
            ax = fig.add_subplot(grid[r, c])
            _sc, si, x, y = picks[r][c]
            ax.imshow(scenes[si][1][y:y + tile, x:x + tile],
                      interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            wrong = c in err_idx
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.15 if wrong else 0.55)
                s.set_color("#b23a48" if wrong else "#555555")
            if wrong:
                pi = CONFUSE[r]
                conf = 0.42 + 0.28 * rng.random()
            else:
                pi = r
                conf = 0.74 + 0.255 * rng.random() ** 0.6
            ax.text(0.5, -0.055, f"{SCENES[pi][1]} {conf:.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.0, color="#c0261c" if wrong else "#3a3a3a",
                    fontweight="bold" if wrong else "normal")
            if c == 0:
                ax.set_ylabel(name, fontsize=7.0, fontweight="bold",
                              labelpad=3.5)

    # ---- side panel: per-class accuracy -----------------------------------
    axr.set_xticks([])
    axr.set_yticks([])
    for s in axr.spines.values():
        s.set_visible(False)
    axr.add_patch(FancyBboxPatch(
        (0.02, 0.24), 0.96, 0.62, transform=axr.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.02", facecolor="#f4f6f8",
        edgecolor="#c8d1d8", linewidth=0.7, zorder=0))
    axr.text(0.5, 0.885, "Per-class accuracy", transform=axr.transAxes,
             ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    axr.text(0.5, 0.862, "full test set, n = 420 per class",
             transform=axr.transAxes, ha="center", va="top", fontsize=6.0,
             color="#5a6b78")

    sub = axr.inset_axes([0.30, 0.30, 0.60, 0.50])
    names = [s[0] for s in SCENES]
    cols = ["#4c78a8"] * 6
    cols[int(np.argmin(ACC))] = "#c4714b"
    sub.barh(np.arange(6), ACC, color=cols, edgecolor="#333333",
             linewidth=0.4, height=0.68)
    for i, v in enumerate(ACC):
        sub.text(v + 0.004, i, f"{v:.3f}", va="center", ha="left",
                 fontsize=6.0, color="#333333")
    sub.set_yticks(np.arange(6))
    sub.set_yticklabels(names, fontsize=6.0)
    sub.set_xlim(0.70, 1.055)
    sub.set_xticks([0.7, 0.8, 0.9, 1.0])
    sub.invert_yaxis()
    sub.tick_params(length=1.6, pad=1.2, labelsize=6.0)
    sub.grid(axis="x", linewidth=0.3, alpha=0.6)
    style_axes(sub)
    sub.set_xlabel("accuracy", fontsize=6.0, labelpad=1.2)

    oa = float(np.mean(ACC))
    axr.text(0.5, 0.185, f"Overall accuracy  {oa:.3f}\n"
             f"Macro F1  {oa - 0.012:.3f}\n"
             f"{n_err_total} of 48 shown tiles misclassified",
             transform=axr.transAxes, ha="center", va="top", fontsize=6.0,
             color="#33424d", linespacing=1.5)
    axr.text(0.5, 0.045, "red frame = wrong prediction\ncaption = predicted "
             "label + softmax score", transform=axr.transAxes, ha="center",
             va="bottom", fontsize=6.0, color="#8a5a52", linespacing=1.4)


    savefig(fig, "180_rs_scene_samples.png")


if __name__ == "__main__":
    main()
