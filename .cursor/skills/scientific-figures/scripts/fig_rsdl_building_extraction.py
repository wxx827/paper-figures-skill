"""Figure 333 - Building footprint extraction benchmark.

Three very-high-resolution tiles (rows) x six views (columns): true-colour
image, reference footprints and four segmentation networks.  Every cell
carries the same red zoom box so the reader can compare boundary quality at
identical locations; per-tile IoU sits inside each cell and the pooled
IoU / F1 per method runs along the bottom.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter, label as cc_label

from pubstyle import set_pub_style, savefig

# ---------------------------------------------------------------------------
# Shared land-cover semantics (kept identical across the 333-342 batch)
# ---------------------------------------------------------------------------
LC = {"water": "#3b7ea1", "crop": "#8fbc4a", "forest": "#1d6b3f",
      "grass": "#d8d878", "building": "#b23a48", "soil": "#d9a86c",
      "road": "#6f6f6f"}
# house red is already the built-up land-cover class here, so the zoom chrome
# takes house purple instead
ZOOM_RED = "#8e6bbf"
ZOOM_ORANGE = "#c9a227"
GSD = 0.5          # m per pixel, VHR aerial-like


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(3, 8, 20), weights=(0.5, 0.32, 0.18)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _rot(xx, yy, ang):
    th = np.deg2rad(ang)
    return xx * np.cos(th) + yy * np.sin(th), -xx * np.sin(th) + yy * np.cos(th)


# ---------------------------------------------------------------------------
# Synthetic VHR urban scene (self-contained)
# ---------------------------------------------------------------------------
DISTRICTS = [
    # (angle, block pitch across street, building pitch along street,
    #  cross-street pitch, road width, skip rate)
    (14.0, 34.0, 22.0, 128.0, 6.4, 0.16),
    (-38.0, 29.0, 17.0, 104.0, 5.8, 0.20),
    (62.0, 52.0, 44.0, 150.0, 7.0, 0.34),
]


def urban_scene(n=176, seed=7):
    """Return (rgb, building_mask, road_mask) for a synthetic VHR tile."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    anchors = rng.uniform(0.12, 0.88, size=(3, 2)) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 14 * _noise(rng, n, 26)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)

    build = np.zeros((n, n), bool)
    road = np.zeros((n, n), bool)
    bid = np.zeros((n, n), int)

    for r, (ang, cu, pv, cv, wr, skip) in enumerate(DISTRICTS):
        here = reg == r
        if not here.any():
            continue
        jit = 1.8 * _noise(rng, n, 16)
        u, v = _rot(xx, yy, ang + 3.0 * rng.standard_normal())
        u = u + jit
        v = v + jit

        fu = np.mod(u, cu)
        fv = np.mod(v, pv)
        street = (fu < wr) | (np.mod(v, cv) < wr + 1.6)

        iu = np.floor(u / cu).astype(int)
        iv = np.floor(v / pv).astype(int)
        key = np.abs(iu * 7919 + iv * 131 + r * 104729) % 4096
        lut_skip = rng.random(4096)
        lut_d = rng.random(4096)
        lut_l = rng.random(4096)

        setback = 2.2
        depth = wr + setback + (cu - wr - 2 * setback) * (0.58 + 0.40 * lut_d[key])
        gap = 2.0 + 3.2 * lut_l[key]
        inside = (fu > wr + setback) & (fu < depth) & (fv > gap) & (fv < pv - gap)
        keep = lut_skip[key] > skip
        b = here & inside & keep & ~street
        build |= b
        road |= here & street
        bid = np.where(b, iu * 7919 + iv * 131 + r * 104729, bid)

    # --- background land cover -------------------------------------------
    veg = gaussian_filter((_fbm(rng, n, (4, 11, 26)) > 0.72).astype(float),
                          1.3) > 0.45
    veg &= ~build & ~road
    soil = (_noise(rng, n, 8) > 0.85) & ~build & ~road & ~veg

    # --- per-band reflectance --------------------------------------------
    bands = np.zeros((3, n, n))
    base = {"grass": [0.088, 0.132, 0.086], "soil": [0.215, 0.198, 0.170],
            "forest": [0.038, 0.070, 0.042], "road": [0.130, 0.134, 0.140]}
    for k in range(3):
        bands[k] = base["grass"][k]
        bands[k][soil] = base["soil"][k]
        bands[k][veg] = base["forest"][k]
        bands[k][road] = base["road"][k]

    key_b = np.abs(bid) % 4096
    lut_roof = rng.random(4096)
    lut_hue = rng.random(4096)
    bright = 0.13 + 0.26 * lut_roof[key_b]
    hue = lut_hue[key_b]
    terra = hue > 0.70                      # terracotta tile
    metal = hue < 0.14                      # bluish metal sheet
    roof_r = bright * (1.0 + 0.42 * terra - 0.10 * metal)
    roof_g = bright * (1.0 - 0.14 * terra - 0.02 * metal)
    roof_b = bright * (1.0 - 0.34 * terra + 0.08 * metal)
    for k, rf in enumerate((roof_r, roof_g, roof_b)):
        bands[k] = np.where(build, rf, bands[k])

    # --- cast shadows to the south-east -----------------------------------
    sh = np.zeros((n, n), bool)
    for d in range(1, 5):
        sh |= np.roll(np.roll(build, d, axis=0), d, axis=1)
    sh &= ~build
    for k in range(3):
        bands[k] = np.where(sh, bands[k] * 0.42 + 0.012, bands[k])

    tex = 1 + 0.12 * _noise(rng, n, 1.0) + 0.06 * _noise(rng, n, 3.0)
    illum = 1 + 0.05 * _noise(rng, n, 30)
    for k in range(3):
        b = bands[k] * tex * illum
        b += 0.0035 * rng.standard_normal((n, n))
        bands[k] = b

    # One shared stretch across R/G/B keeps the colour balance neutral.
    p0, p1 = np.percentile(bands, [1.2, 99.0])
    rgb = np.dstack([np.clip((bands[i] - p0) / max(p1 - p0, 1e-9), 0, 1) ** g
                     for i, g in zip((0, 1, 2), (0.78, 0.78, 0.84))])
    return rgb, build, road


# ---------------------------------------------------------------------------
# Network simulation
# ---------------------------------------------------------------------------
def predict_mask(gt, seed, sigma, noise, thr, disp, drop, merge=0.0, fp=0.0):
    """Omission + boundary wobble + corner rounding + spurious blobs.

    Working per connected component keeps the output rectangular enough to
    still read as buildings; only ``sigma`` rounds the corners off.
    """
    n = gt.shape[0]
    rng = np.random.default_rng(seed)
    inst, k = cc_label(gt)
    keep = np.ones(k + 1, bool)
    keep[0] = False
    if drop > 0 and k > 0:
        keep[1:] = rng.random(k) > drop
    src = keep[inst].astype(float)

    if merge > 0:
        src = (gaussian_filter(src, merge) > 0.30).astype(float)

    field = gaussian_filter(src, sigma)
    field = field + noise * _noise(rng, n, 1.6) + 0.6 * noise * _noise(rng, n, 4.0)
    yy, xx = np.mgrid[0:n, 0:n]
    dy = disp * _noise(rng, n, 7.0)
    dx = disp * _noise(rng, n, 7.0)
    sy = np.clip(np.round(yy + dy).astype(int), 0, n - 1)
    sx = np.clip(np.round(xx + dx).astype(int), 0, n - 1)
    pred = field[sy, sx] > thr

    if fp > 0:
        halo = gaussian_filter(gt.astype(float), 6.0) > 0.06
        blob = _noise(rng, n, 2.6)
        pred |= halo & ~gt & (blob > np.quantile(blob, 1.0 - fp))
    return pred


def iou_f1(gt, pr):
    inter = np.count_nonzero(gt & pr)
    union = np.count_nonzero(gt | pr)
    prec = inter / max(np.count_nonzero(pr), 1)
    rec = inter / max(np.count_nonzero(gt), 1)
    return inter / max(union, 1), 2 * prec * rec / max(prec + rec, 1e-9)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def overlay(rgb, mask, color, alpha=0.45):
    col = np.array(plt.matplotlib.colors.to_rgb(color))
    out = rgb.copy()
    out[mask] = (1 - alpha) * rgb[mask] + alpha * col
    return np.clip(out, 0, 1)


def add_zoom(ax, disp, mask, box, loc=(0.575, 0.022, 0.405, 0.405),
             color=ZOOM_RED, outline="white"):
    """Red box on the panel + magnified crop pinned to the lower-right."""
    x0, y0, w, h = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                           edgecolor=color, linewidth=0.9, zorder=7))
    ins = ax.inset_axes(loc)
    ins.set_zorder(8)
    ins.imshow(disp[y0:y0 + h, x0:x0 + w], interpolation="nearest")
    if mask is not None and mask[y0:y0 + h, x0:x0 + w].any():
        ins.contour(mask[y0:y0 + h, x0:x0 + w].astype(float), levels=[0.5],
                    colors=outline, linewidths=0.7)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color(color)
    return ins


def stamp(ax, text, weight="normal"):
    ax.text(0.028, 0.030, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.4, color="white", fontweight=weight,
            zorder=9,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


def add_scale_bar(ax, n, length_m=50.0, label="50 m"):
    px = length_m / GSD
    x0, y0 = 0.032 * n, 0.930 * n
    h = 0.018 * n
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * px / 2, y0), px / 2, h, zorder=6,
                               facecolor="white" if i else "#111111",
                               edgecolor="#111111", linewidth=0.5))
    ax.text(x0 + px / 2, y0 - 0.014 * n, label, ha="center", va="bottom",
            fontsize=6.0, color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n = 176
    tiles = [(7, (96, 18, 44, 44)), (23, (22, 92, 44, 44)),
             (54, (18, 20, 44, 44))]

    methods = [
        ("U-Net", dict(sigma=1.46, noise=0.180, thr=0.46, disp=1.20,
                       drop=0.110, merge=1.50, fp=0.056)),
        ("HRNet", dict(sigma=1.20, noise=0.150, thr=0.47, disp=1.00,
                       drop=0.085, merge=1.10, fp=0.045)),
        ("SegFormer", dict(sigma=1.00, noise=0.130, thr=0.48, disp=0.88,
                           drop=0.070, merge=0.85, fp=0.033)),
        ("Ours (BFE-Net)", dict(sigma=0.78, noise=0.095, thr=0.49, disp=0.55,
                                drop=0.035, merge=0.30, fp=0.018)),
    ]
    # Tile 2 is the one scene where the transformer baseline edges us out.
    tweak = {(3, 1): dict(disp=0.82, drop=0.072, fp=0.032)}

    samples = []
    for k, (sd, box) in enumerate(tiles):
        rgb, build, road = urban_scene(n, seed=sd)
        preds = []
        for j, (_nm, kw) in enumerate(methods):
            kw = {**kw, **tweak.get((j, k), {})}
            preds.append(predict_mask(build, seed=400 + 11 * j + k, **kw))
        samples.append(dict(rgb=rgb, gt=build, preds=preds, box=box))

    pooled = []
    for j in range(len(methods)):
        g = np.concatenate([s["gt"].ravel() for s in samples])
        p = np.concatenate([s["preds"][j].ravel() for s in samples])
        pooled.append(iou_f1(g, p))
    for (nm, _), (i, f) in zip(methods, pooled):
        print(f"  {nm:16s} IoU = {i:.3f}   F1 = {f:.3f}")

    cols = ["Image", "Reference", "U-Net", "HRNet", "SegFormer",
            "Ours (BFE-Net)"]
    fig = plt.figure(figsize=(10.4, 5.9))
    gs = fig.add_gridspec(3, 6, wspace=0.028, hspace=0.028,
                          left=0.036, right=0.988, top=0.878, bottom=0.128)
    axes = np.empty((3, 6), object)

    for r, s in enumerate(samples):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            image_axes(ax)
            ax.patch.set_visible(False)
            if c == 0:
                disp, mask = s["rgb"], None
            elif c == 1:
                mask = s["gt"]
                disp = overlay(s["rgb"], mask, LC["building"], 0.45)
            else:
                mask = s["preds"][c - 2]
                disp = overlay(s["rgb"], mask, LC["building"], 0.45)
            ax.imshow(disp, interpolation="nearest")
            if mask is not None:
                ax.contour(mask.astype(float), levels=[0.5], colors="white",
                           linewidths=0.45)
            add_zoom(ax, disp, mask, s["box"])
            if c == 0:
                add_scale_bar(ax, n)
                ax.set_ylabel(f"Tile {r + 1}", fontsize=7.0,
                              fontweight="bold", labelpad=3)
            if c >= 2:
                i_t, _f = iou_f1(s["gt"], mask)
                best = i_t >= max(iou_f1(s["gt"], s["preds"][j])[0]
                                  for j in range(4)) - 1e-9
                stamp(ax, f"IoU {i_t:.3f}",
                      weight="bold" if best else "normal")
            if r == 0:
                ax.set_title(f"({'abcdef'[c]})  {cols[c]}", fontsize=7.6,
                             fontweight="bold" if c == 5 else "bold",
                             pad=3.4,
                             color="#7a1f2b" if c == 5 else "#1a1a1a")

    # ---- highlight the "Ours" column --------------------------------------
    p_top = axes[0, 5].get_position()
    p_bot = axes[2, 5].get_position()
    pad = 0.006
    fig.patches.append(Rectangle(
        (p_top.x0 - pad, p_bot.y0 - 0.092), p_top.width + 2 * pad,
        (p_top.y1 + 0.042) - (p_bot.y0 - 0.092),
        transform=fig.transFigure, facecolor="#f2efe9", edgecolor="none",
        zorder=-1))

    # ---- pooled metrics under each method column --------------------------
    for j, (nm, _) in enumerate(methods):
        p = axes[2, j + 2].get_position()
        xc = 0.5 * (p.x0 + p.x1)
        best = pooled[j][0] == max(q[0] for q in pooled)
        fig.text(xc, p.y0 - 0.018, f"IoU  {pooled[j][0]:.3f}", ha="center",
                 va="top", fontsize=7.4,
                 fontweight="bold" if best else "normal",
                 color="#7a1f2b" if best else "#333333")
        fig.text(xc, p.y0 - 0.062, f"F1  {pooled[j][1]:.3f}", ha="center",
                 va="top", fontsize=7.0,
                 fontweight="bold" if best else "normal",
                 color="#7a1f2b" if best else "#555555")

    handles = [Patch(facecolor=LC["building"], alpha=0.55,
                     edgecolor="white", linewidth=0.7,
                     label="Predicted building"),
               Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                     label="Zoom window (identical per row)")]
    leg = fig.legend(handles=handles, loc="lower left", ncol=1,
                     frameon=False, fontsize=6.8, handlelength=1.3,
                     handleheight=1.0, bbox_to_anchor=(0.036, 0.006))
    leg.set_clip_on(False)


    savefig(fig, "333_rsdl_building_extraction.png")


if __name__ == "__main__":
    main()
