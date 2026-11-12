"""Figure 342 - One backbone, five remote-sensing outputs.

A single peri-urban tile enters a shared encoder and five task heads emit
land cover, building footprints, roads, an object height model and a change
map.  Arrows fan from the encoder to each output; a compact bar strip under
the row reports the metric each head is scored with.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (Rectangle, FancyBboxPatch, FancyArrowPatch,
                                Patch)
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig

# house red already carries the built-up class and the building footprint
# here, so the zoom chrome takes house purple instead
ZOOM_RED = "#8e6bbf"
GSD = 1.0
WATER, CROP, FOREST, URBAN, SOIL = range(5)
LC_NAMES = ["Water", "Cropland", "Forest", "Built-up", "Bare soil"]
LC_SHORT = ["Wat", "Crop", "For", "Built", "Soil"]
LC_COLORS = ["#3b7ea1", "#8fbc4a", "#1d6b3f", "#b23a48", "#d9a86c"]
BUILD_C = "#b23a48"
ROAD_C = "#c9a227"
CHG_COLORS = ["#8c2f2f", "#2e7d5b"]
CHG_NAMES = ["New building", "Vegetation loss"]


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


def scene(n=192, seed=15):
    """Peri-urban tile: fields, forest, a river, a road net and a village."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    # --- farmland parcels --------------------------------------------------
    reg = np.argmin(np.stack(
        [np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
         for a in rng.uniform(0.12, 0.88, size=(3, 2)) * n]), axis=0)
    fid = np.zeros((n, n), int)
    for r in range(3):
        th = np.deg2rad(rng.uniform(-55, 55))
        cu, cv = rng.uniform(20, 30), rng.uniform(24, 36)
        u = xx * np.cos(th) + yy * np.sin(th) + 2.6 * _noise(rng, n, 14)
        v = -xx * np.sin(th) + yy * np.cos(th) + 2.6 * _noise(rng, n, 14)
        fid = np.where(reg == r,
                       np.floor(u / cu).astype(int) * 211
                       + np.floor(v / cv).astype(int) * 37 + r * 1013, fid)
    key = np.abs(fid) % 4096
    vig = rng.random(4096)[key] ** 0.62
    kind = rng.random(4096)[key]

    lab = np.full((n, n), CROP, np.int16)
    lab[kind < 0.15] = SOIL
    forest = gaussian_filter((_fbm(rng, n, (4, 10, 22)) > 0.82).astype(float),
                             1.4) > 0.45
    lab[forest] = FOREST

    ty = np.linspace(0, n - 1, 220)
    tx = 0.22 * n + 0.075 * n * np.sin(2 * np.pi * ty / n * 1.2 + 0.9)
    river = _polyline_dist(n, np.c_[tx, ty]) <= 2.6
    lab[river] = WATER

    # --- road network ------------------------------------------------------
    t = np.linspace(0, n - 1, 260)
    lines = [(np.c_[t, 0.34 * n + 0.06 * n * np.sin(2 * np.pi * t / n * 0.9)],
              1.9),
             (np.c_[0.66 * n + 0.05 * n * np.sin(2 * np.pi * t / n * 1.1),
                    t], 1.7),
             (np.c_[t, 0.80 * n + 0.04 * n * np.sin(2 * np.pi * t / n * 1.4
                                                    + 2.0)], 1.4)]
    road = np.zeros((n, n), bool)
    for pl, w in lines:
        road |= _polyline_dist(n, pl) <= w

    # --- village -----------------------------------------------------------
    village = (np.exp(-(((xx - 0.68 * n) ** 2 + (yy - 0.60 * n) ** 2)
                        / (2 * (0.185 * n) ** 2)))
               + 0.26 * _fbm(rng, n, (5, 13), (0.5, 0.5))) > 0.62
    th = np.deg2rad(rng.uniform(-30, 30))
    u = xx * np.cos(th) + yy * np.sin(th) + 1.4 * _noise(rng, n, 12)
    v = -xx * np.sin(th) + yy * np.cos(th) + 1.4 * _noise(rng, n, 12)
    cu, pv = 17.0, 11.0
    fu, fv = np.mod(u, cu), np.mod(v, pv)
    street = fu < 3.4
    iu, iv = np.floor(u / cu).astype(int), np.floor(v / pv).astype(int)
    bkey = np.abs(iu * 7919 + iv * 131) % 4096
    lut_s, lut_d, lut_h = (rng.random(4096) for _ in range(3))
    depth = 3.4 + 2.0 + (cu - 3.4 - 4.0) * (0.55 + 0.40 * lut_d[bkey])
    build = (village & (fu > 5.4) & (fu < depth) & (fv > 2.0)
             & (fv < pv - 2.0) & (lut_s[bkey] > 0.18) & ~street & ~river)
    road |= village & street & ~build
    lab[build] = URBAN
    lab[road] = URBAN

    # --- reflectance -------------------------------------------------------
    veg_spec, soil_spec = (0.034, 0.082, 0.034), (0.202, 0.182, 0.154)
    fc = np.clip(vig, 0.04, 1.0)
    bands = np.stack([fc * veg_spec[k] + (1 - fc) * soil_spec[k]
                      for k in range(3)])
    for k, val in enumerate((0.028, 0.046, 0.060)):
        bands[k] = np.where(river, val, bands[k])
    for k, val in enumerate((0.034, 0.066, 0.038)):
        bands[k] = np.where(forest & ~build & ~road, val, bands[k])
    for k, val in enumerate((0.132, 0.136, 0.142)):
        bands[k] = np.where(road, val, bands[k])
    bright = 0.13 + 0.24 * lut_h[bkey]
    terra = lut_h[bkey] > 0.66
    for k, f in enumerate((1.0 + 0.42 * terra, 1.0 - 0.14 * terra,
                           1.0 - 0.34 * terra)):
        bands[k] = np.where(build, bright * f, bands[k])
    sh = np.zeros((n, n), bool)
    for d in range(1, 3):
        sh |= np.roll(np.roll(build, d, 0), d, 1)
    sh &= ~build
    for k in range(3):
        bands[k] = np.where(sh, bands[k] * 0.45, bands[k])

    amp = np.where(forest, 0.24, np.where(build, 0.10, 0.08))
    tex = 1 + amp * (0.6 * _noise(rng, n, 1.0) + 0.4 * _noise(rng, n, 2.8))
    for k in range(3):
        bands[k] = bands[k] * tex * (1 + 0.04 * _noise(rng, n, 26))
    lo, hi = 0.012, 0.290
    rgb = np.dstack([np.clip((bands[i] - lo) / (hi - lo), 0, 1) ** 0.78
                     for i in range(3)])

    # --- height model ------------------------------------------------------
    hgt = np.zeros((n, n))
    canopy = forest & ~build & ~road
    hgt[canopy] = np.clip(7.5 + 3.0 * _fbm(rng, n, (2, 5)), 3.0, 13.0)[canopy]
    hgt = gaussian_filter(hgt, 1.2)
    hgt[build] = (7.0 + 16.0 * lut_h[bkey])[build]
    hgt = np.clip(hgt, 0, 24)

    # --- change against an earlier acquisition -----------------------------
    new_b = build & (lut_s[bkey] > 0.80)
    cleared = forest & (_noise(rng, n, 9) > 1.05) & ~build & ~road
    chg = np.zeros((n, n), np.int16)
    chg[new_b] = 1
    chg[cleared] = 2
    return dict(rgb=rgb, lab=lab, build=build, road=road, hgt=hgt, chg=chg,
                forest=forest)


def degrade_mask(m, seed, sigma, thr, noise):
    n = m.shape[0]
    rng = np.random.default_rng(seed)
    f = gaussian_filter(m.astype(float), sigma) + noise * _noise(rng, n, 2.2)
    return f > thr


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def overlay(rgb, mask, color, alpha=0.62):
    out = rgb.copy()
    out[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(to_rgb(color))
    return np.clip(out, 0, 1)


def chip_legend(ax, entries, y=0.022, fs=6.0):
    """Compact colour-chip strip on a translucent band inside a panel."""
    ax.add_patch(FancyBboxPatch(
        (0.018, y - 0.008), 0.964, 0.088, transform=ax.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.02", facecolor="white",
        edgecolor="#a8a8a8", linewidth=0.45, alpha=0.90, zorder=8))
    k = len(entries)
    for i, (col, name) in enumerate(entries):
        x = 0.040 + i * (0.925 / k)
        ax.add_patch(Rectangle((x, y + 0.018), 0.030, 0.046,
                               transform=ax.transAxes, facecolor=col,
                               edgecolor="#555555", linewidth=0.35,
                               zorder=9))
        ax.text(x + 0.040, y + 0.040, name, transform=ax.transAxes,
                fontsize=fs, va="center", ha="left", color="#1a1a1a",
                zorder=9)


def main():
    set_pub_style(base_font=8)
    n = 192
    s = scene(n, seed=15)

    lc_pred = s["lab"].copy()
    rng = np.random.default_rng(31)
    edge = gaussian_filter((np.abs(np.gradient(s["lab"].astype(float))[0])
                            + np.abs(np.gradient(s["lab"].astype(float))[1])
                            > 0).astype(float), 1.2)
    sc = 0.6 * (_noise(rng, n, 5.0) - _noise(rng, n, 5.0).min()) + 0.4 * edge
    hit = sc > np.quantile(sc, 1 - 0.113)
    for a, b in {CROP: SOIL, SOIL: CROP, FOREST: CROP, URBAN: SOIL,
                 WATER: FOREST}.items():
        lc_pred[hit & (s["lab"] == a)] = b

    b_pred = degrade_mask(s["build"], 51, 0.85, 0.47, 0.10)
    r_pred = degrade_mask(s["road"], 52, 0.80, 0.42, 0.09)
    hs = gaussian_filter(s["hgt"], 0.6)
    # Noise scales with the signal so bare ground stays flat instead of
    # filling the panel with speckle.
    h_pred = hs + (0.18 + 0.08 * hs) * _noise(np.random.default_rng(53), n,
                                              2.0)
    h_pred = np.clip(h_pred, 0, 24)
    c_pred = s["chg"].copy()
    cm = degrade_mask(s["chg"] > 0, 54, 0.85, 0.50, 0.11)
    c_pred = np.where(cm, np.where(s["chg"] > 0, s["chg"],
                                   1 + (_noise(np.random.default_rng(55), n,
                                               4.0) > 0.6).astype(int)), 0)

    metrics = [("Land cover", "OA", 0.887), ("Building", "IoU", 0.782),
               ("Road", "IoU", 0.714), ("Height", r"R$^2$", 0.906),
               ("Change", "F1", 0.803)]

    lut = np.array([to_rgb(c) for c in LC_COLORS])
    lc_img = np.clip(0.42 * s["rgb"] + 0.58 * lut[lc_pred], 0, 1)
    cmap_h = LinearSegmentedColormap.from_list(
        "h", PALETTES["cb_sequential_warm"])
    chg_lut = np.array([(1.0, 1.0, 1.0)] + [to_rgb(c) for c in CHG_COLORS])
    chg_img = np.clip(0.16 * s["rgb"] + 0.84 * chg_lut[c_pred], 0, 1)
    grey = s["rgb"].mean(axis=2, keepdims=True) * 0.62 + 0.34
    chg_img[c_pred == 0] = np.repeat(grey, 3, axis=2)[c_pred == 0]

    fig = plt.figure(figsize=(10.6, 6.3))
    ax_in = fig.add_axes([0.040, 0.575, 0.190, 0.320])
    gs_o = fig.add_gridspec(1, 5, left=0.040, right=0.985, top=0.505,
                            bottom=0.185, wspace=0.035)
    gs_m = fig.add_gridspec(1, 5, left=0.040, right=0.985, top=0.143,
                            bottom=0.062, wspace=0.035)

    # ---- input ------------------------------------------------------------
    image_axes(ax_in)
    ax_in.imshow(s["rgb"], interpolation="nearest")
    ax_in.set_title("(a)  Input tile", fontsize=8.0, fontweight="bold",
                    pad=3.4)
    px = 100.0 / GSD
    ax_in.add_patch(Rectangle((0.035 * n, 0.930 * n), px, 0.018 * n,
                              facecolor="white", edgecolor="#111111",
                              linewidth=0.5, zorder=6))
    ax_in.text(0.035 * n + px / 2, 0.918 * n, "100 m", ha="center",
               va="bottom", fontsize=6.0, color="white", zorder=7,
               path_effects=[pe.withStroke(linewidth=1.7,
                                           foreground="#1a1a1a")])

    # ---- encoder block and fan --------------------------------------------
    enc = FancyBboxPatch((0.268, 0.690), 0.150, 0.098,
                         transform=fig.transFigure,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         facecolor="#eef2f5", edgecolor="#3b5b92",
                         linewidth=0.9, zorder=3)
    fig.patches.append(enc)
    fig.text(0.343, 0.755, "Shared encoder", ha="center", va="center",
             fontsize=7.6, fontweight="bold", color="#1b3f5c", zorder=4)
    fig.text(0.343, 0.716, "ViT-B + FPN neck", ha="center", va="center",
             fontsize=6.4, color="#4a5b68", zorder=4)
    fig.patches.append(FancyArrowPatch(
        (0.232, 0.735), (0.264, 0.739), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#3b5b92",
        zorder=3))

    titles = ["(b)  Land cover", "(c)  Building", "(d)  Road",
              "(e)  Object height", "(f)  Change"]
    out_axes = []
    for c in range(5):
        ax = fig.add_subplot(gs_o[0, c])
        out_axes.append(ax)
        image_axes(ax)
        if c == 0:
            ax.imshow(lc_img, interpolation="nearest")
            chip_legend(ax, list(zip(LC_COLORS, LC_SHORT)))
        elif c == 1:
            ax.imshow(overlay(s["rgb"], b_pred, BUILD_C), interpolation="nearest")
            ax.contour(b_pred.astype(float), levels=[0.5], colors="white",
                       linewidths=0.45)
            chip_legend(ax, [(BUILD_C, "Building footprint")])
        elif c == 2:
            ax.imshow(overlay(s["rgb"], r_pred, ROAD_C, 0.85),
                      interpolation="nearest")
            chip_legend(ax, [(ROAD_C, "Road surface")])
        elif c == 3:
            im = ax.imshow(h_pred, cmap=cmap_h, vmin=0, vmax=24,
                           interpolation="nearest")
            ax.add_patch(FancyBboxPatch(
                (0.018, 0.014), 0.964, 0.100, transform=ax.transAxes,
                boxstyle="round,pad=0.004,rounding_size=0.02",
                facecolor="white", edgecolor="#a8a8a8", linewidth=0.45,
                alpha=0.90, zorder=8))
            cax = ax.inset_axes([0.075, 0.045, 0.640, 0.030])
            cax.set_zorder(9)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                              ticks=[0, 12, 24])
            cb.outline.set_linewidth(0.5)
            cb.outline.set_edgecolor("#333333")
            cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.6, pad=0.8,
                              color="#333333")
            ax.text(0.760, 0.062, "height (m)", transform=ax.transAxes,
                    fontsize=6.0, ha="left", va="center", zorder=9)
        else:
            ax.imshow(chg_img, interpolation="nearest")
            ax.contour((c_pred > 0).astype(float), levels=[0.5],
                       colors="#1a1a1a", linewidths=0.4)
            chip_legend(ax, list(zip(CHG_COLORS, ["New build.", "Veg. loss"])))
        ax.set_title(titles[c], fontsize=8.0, fontweight="bold", pad=3.4)

        p = ax.get_position()
        fig.patches.append(FancyArrowPatch(
            (0.343, 0.686), (0.5 * (p.x0 + p.x1), p.y1 + 0.036),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
            lw=0.9, color="#7d8f9e",
            connectionstyle=f"arc3,rad={0.16 * (c - 2) / 2.2:.3f}",
            zorder=2))

    # ---- compact metric strip --------------------------------------------
    for c, (task, mname, val) in enumerate(metrics):
        ax = fig.add_subplot(gs_m[0, c])
        ax.barh([0], [1.0], 0.62, color="#eef0f2", edgecolor="#cccccc",
                linewidth=0.4)
        ax.barh([0], [val], 0.62, color="#5f7688", edgecolor="#333333",
                linewidth=0.4)
        ax.text(0.014, 0.72, f"{task}  -  {mname}", transform=ax.transAxes,
                fontsize=6.6, va="bottom", ha="left", color="#333333")
        ax.text(val - 0.02, 0, f"{val:.3f}", va="center", ha="right",
                fontsize=6.8, fontweight="bold", color="white")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.62, 0.98)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4,
                     label=nm) for c, nm in zip(LC_COLORS, LC_NAMES)]
    leg = fig.legend(handles=handles, loc="upper left", ncol=1,
                     frameon=False, fontsize=7.0, handlelength=1.2,
                     handleheight=1.0, bbox_to_anchor=(0.828, 0.892))
    leg.set_clip_on(False)
    fig.text(0.470, 0.870, "Five decoders, one shared representation",
             fontsize=8.0, fontweight="bold", ha="left", va="bottom",
             color="#1b3f5c")
    fig.text(0.470, 0.838,
             "semantic  ·  instance-like  ·  linear  ·  regression  ·  "
             "bi-temporal",
             fontsize=6.8, color="#555555", ha="left", va="bottom")

    savefig(fig, "342_rsdl_multitask_output.png")


if __name__ == "__main__":
    main()
