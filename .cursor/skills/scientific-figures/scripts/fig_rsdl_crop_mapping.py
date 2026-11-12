"""Figure 335 - Parcel-level crop type mapping.

A large thematic crop map with field boundaries and a red zoom window, an
error matrix for the seven crop classes and the NDVI phenology that the
classifier keys on.  Overall accuracy and Kappa sit under the map.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Patch
from matplotlib.colors import (ListedColormap, BoundaryNorm,
                               LinearSegmentedColormap)
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label, \
    style_axes

ZOOM_RED = "#b23a48"
GSD = 10.0                                   # m per pixel (Sentinel-2 like)

CROPS = ["Maize", "Soybean", "Winter wheat", "Rice", "Cotton", "Alfalfa",
         "Orchard", "Non-crop"]
CROP_SHORT = ["Mze", "Soy", "Wht", "Rice", "Ctn", "Alf", "Orc", "Non"]
CROP_COLORS = ["#c9a227", "#8fbc4a", "#d98c5f", "#3b7ea1", "#8e6bbf",
               "#2e7d5b", "#8c2f2f", "#b9b9b9"]
# peak DOY, peak NDVI, season width (days), green-up DOY
PHENO = [(212, 0.86, 58, 150), (232, 0.81, 52, 172), (140, 0.78, 46, 78),
         (238, 0.74, 44, 178), (226, 0.70, 50, 165), (196, 0.66, 92, 110),
         (204, 0.62, 104, 118), (200, 0.28, 120, 120)]


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


def crop_scene(n=256, seed=17):
    """Field-parcel mosaic labelled with crop types."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    anchors = np.array([[0.20, 0.22], [0.76, 0.18], [0.24, 0.78],
                        [0.80, 0.72]]) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)

    fid = np.zeros((n, n), int)
    for r, (ang, cu, cv) in enumerate([(16, 17, 23), (-25, 20, 14),
                                       (41, 14, 19), (5, 22, 16)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 2.4 * _noise(rng, n, 14)
        v = -xx * np.sin(th) + yy * np.cos(th) + 2.4 * _noise(rng, n, 14)
        f = (np.floor(u / cu).astype(int) * 211
             + np.floor(v / cv).astype(int) * 37 + r * 1013)
        fid = np.where(reg == r, f, fid)

    key = np.abs(fid) % 4096
    draw = rng.random(4096)
    weights = np.array([0.20, 0.17, 0.15, 0.11, 0.09, 0.09, 0.08, 0.11])
    edges = np.concatenate([[0], np.cumsum(weights)])
    lut = np.digitize(draw, edges[1:-1])
    lab = lut[key].astype(np.int16)

    # centre-pivot circles are always maize or alfalfa
    wob = _noise(rng, n, 3)
    for (cx, cy, rr, cls) in [(0.14, 0.46, 0.052, 0), (0.25, 0.57, 0.041, 5),
                              (0.08, 0.64, 0.035, 0), (0.20, 0.68, 0.030, 5)]:
        m = np.hypot(xx - cx * n, yy - cy * n) < rr * n * (1 + 0.03 * wob)
        lab[m] = cls

    # river and a small settlement are non-crop
    ty = np.linspace(0, n - 1, 240)
    tx = (0.34 * n + 0.10 * n * np.sin(2 * np.pi * ty / n * 1.05)
          + 0.04 * n * np.sin(2 * np.pi * ty / n * 2.6 + 1.3))
    lab[_polyline_dist(n, np.c_[tx, ty]) <= 3.0] = 7
    town = (np.hypot(xx - 0.74 * n, yy - 0.66 * n)
            < 0.085 * n * (1 + 0.30 * _fbm(rng, n, (6, 15), (0.5, 0.5))))
    lab[town] = 7
    return lab, fid


def simulate_pred(lab, fid, seed=9):
    """Confusions concentrate on small parcels and along field edges."""
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    edge = (np.abs(np.gradient(fid.astype(float))[0])
            + np.abs(np.gradient(fid.astype(float))[1])) > 0
    edge = gaussian_filter(edge.astype(float), 1.4)
    edge /= edge.max() + 1e-9
    score = 0.5 * rng.random((n, n)) + 0.5 * edge
    conf = {0: [(1, 0.055), (4, 0.022)], 1: [(0, 0.070), (5, 0.024)],
            2: [(5, 0.060), (7, 0.020)], 3: [(1, 0.050), (7, 0.028)],
            4: [(1, 0.075), (0, 0.035)], 5: [(2, 0.065), (6, 0.030)],
            6: [(5, 0.080), (7, 0.030)], 7: [(6, 0.045), (2, 0.028)]}
    pred = lab.copy()
    flat = pred.ravel()
    for cls, targets in conf.items():
        idx = np.flatnonzero(lab.ravel() == cls)
        if idx.size == 0:
            continue
        order = idx[np.argsort(-score.ravel()[idx])]
        cut = 0
        for tgt, rate in targets:
            k = int(round(rate * idx.size))
            flat[order[cut:cut + k]] = tgt
            cut += k
    return pred


def error_matrix(truth, pred, n_samples=3200, seed=4):
    rng = np.random.default_rng(seed)
    sel = rng.choice(truth.size, size=n_samples, replace=False)
    cm = np.zeros((8, 8), int)
    for a, b in zip(truth.ravel()[sel], pred.ravel()[sel]):
        cm[a, b] += 1
    return cm


def accuracy_stats(cm):
    tot = cm.sum()
    oa = np.trace(cm) / tot
    pe_ = (cm.sum(0) * cm.sum(1)).sum() / tot ** 2
    kappa = (oa - pe_) / (1 - pe_)
    pa = np.diag(cm) / np.maximum(cm.sum(1), 1)
    ua = np.diag(cm) / np.maximum(cm.sum(0), 1)
    return oa, kappa, pa, ua


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def add_scale_bar(ax, n, length_m=1000.0, label="1 km"):
    px = length_m / GSD
    x0, y0 = 0.035 * n, 0.930 * n
    h = 0.018 * n
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * px / 2, y0), px / 2, h, zorder=6,
                               facecolor="white" if i else "#111111",
                               edgecolor="#111111", linewidth=0.5))
    ax.text(x0 + px / 2, y0 - 0.012 * n, label, ha="center", va="bottom",
            fontsize=6.2, color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


def add_north_arrow(ax, n):
    cx, cy = 0.945 * n, 0.095 * n
    s = 0.045 * n
    ax.add_patch(Polygon([[cx, cy - s], [cx - 0.42 * s, cy + 0.55 * s],
                          [cx, cy + 0.22 * s], [cx + 0.42 * s, cy + 0.55 * s]],
                         closed=True, facecolor="white", edgecolor="#1a1a1a",
                         linewidth=0.6, zorder=6))
    ax.text(cx, cy - 0.75 * s, "N", ha="center", va="bottom", fontsize=6.8,
            fontweight="bold", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n = 256
    lab, fid = crop_scene(n, seed=17)
    pred = simulate_pred(lab, fid, seed=9)
    cm = error_matrix(lab, pred)
    oa, kappa, pa, ua = accuracy_stats(cm)
    print(f"  OA = {oa:.3f}   Kappa = {kappa:.3f}")

    cmap_lc = ListedColormap(CROP_COLORS)
    norm_lc = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), 8)
    # Field boundaries come from the parcel geometry, not from the noisy
    # prediction, so the cadastral lines stay clean.
    bnd = np.zeros((n, n), bool)
    bnd[:, :-1] |= fid[:, :-1] != fid[:, 1:]
    bnd[:-1, :] |= fid[:-1, :] != fid[1:, :]

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.00],
                          height_ratios=[1.00, 0.88], wspace=0.13,
                          hspace=0.30, left=0.038, right=0.978, top=0.885,
                          bottom=0.150)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_ph = fig.add_subplot(gs[1, 1])

    # ---- (a) crop map -----------------------------------------------------
    image_axes(ax_map)
    ax_map.imshow(pred, cmap=cmap_lc, norm=norm_lc, interpolation="nearest")
    ax_map.imshow(np.dstack([np.zeros((n, n))] * 3 + [0.30 * bnd]),
                  interpolation="nearest")
    ax_map.set_title("Crop type map, 7 classes at 10 m",
                     fontsize=8.4, fontweight="bold", pad=3.5)
    add_scale_bar(ax_map, n)
    add_north_arrow(ax_map, n)
    add_panel_label(ax_map, "(a)", x=0.0, y=1.030, size=9.5)

    zx, zy, zs = 150, 34, 58
    ax_map.add_patch(Rectangle((zx - 0.5, zy - 0.5), zs, zs, fill=False,
                               edgecolor=ZOOM_RED, linewidth=1.0, zorder=7))
    ins = ax_map.inset_axes([0.615, 0.022, 0.365, 0.365])
    ins.set_zorder(8)
    ins.imshow(pred[zy:zy + zs, zx:zx + zs], cmap=cmap_lc, norm=norm_lc,
               interpolation="nearest")
    ins.imshow(np.dstack([np.zeros((zs, zs))] * 3
                         + [0.45 * bnd[zy:zy + zs, zx:zx + zs]]),
               interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.1)
        s.set_color(ZOOM_RED)
    chips = [("Overall accuracy", f"{oa:.3f}"), ("Kappa", f"{kappa:.3f}"),
             ("Macro F1", f"{np.mean(2 * pa * ua / np.maximum(pa + ua, 1e-9)):.3f}")]
    p = ax_map.get_position()
    for i, (k, v) in enumerate(chips):
        x0 = p.x0 + i * (p.width / 3.0)
        fig.patches.append(FancyBboxPatch(
            (x0 + 0.006, 0.022), p.width / 3.0 - 0.012, 0.062,
            transform=fig.transFigure,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="#f2f5f8", edgecolor="#b9c6d1", linewidth=0.6,
            zorder=-1))

    # ---- (b) error matrix -------------------------------------------------
    row_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap_cm = LinearSegmentedColormap.from_list(
        "cm", PALETTES["cb_sequential_blue"])
    ax_cm.imshow(row_norm, cmap=cmap_cm, vmin=0, vmax=1,
                 interpolation="nearest")
    for i in range(8):
        for j in range(8):
            if cm[i, j] == 0:
                continue
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=6.0,
                       color="white" if row_norm[i, j] > 0.5 else "#333333",
                       fontweight="bold" if i == j else "normal")
        ax_cm.add_patch(Rectangle((7.56, i - 0.5), 1.0, 1.0, clip_on=False,
                                  facecolor="#eef2f5", edgecolor="#cccccc",
                                  linewidth=0.4))
        ax_cm.text(8.06, i, f"{100 * pa[i]:.0f}", ha="center", va="center",
                   fontsize=6.0, color="#1a1a1a", clip_on=False)
        ax_cm.add_patch(Rectangle((i - 0.5, 7.56), 1.0, 1.0, clip_on=False,
                                  facecolor="#f7f0ea", edgecolor="#cccccc",
                                  linewidth=0.4))
        ax_cm.text(i, 8.06, f"{100 * ua[i]:.0f}", ha="center", va="center",
                   fontsize=6.0, color="#1a1a1a", clip_on=False)
    ax_cm.set_xticks(range(8))
    ax_cm.set_yticks(range(8))
    ax_cm.set_xticklabels(CROP_SHORT, fontsize=6.2)
    ax_cm.set_yticklabels(CROP_SHORT, fontsize=6.2)
    for t, c in zip(ax_cm.get_xticklabels(), CROP_COLORS):
        t.set_color(c)
    for t, c in zip(ax_cm.get_yticklabels(), CROP_COLORS):
        t.set_color(c)
    ax_cm.tick_params(length=1.8, pad=1.5)
    ax_cm.set_xlim(-0.5, 8.56)
    ax_cm.set_ylim(8.56, -0.5)
    ax_cm.set_xlabel("predicted class", fontsize=7.2, labelpad=1.5)
    ax_cm.set_ylabel("reference class", fontsize=7.2, labelpad=1.5)
    ax_cm.text(8.06, -0.80, "PA %", ha="center", va="center", fontsize=6.0,
               fontweight="bold")
    ax_cm.text(-1.45, 8.06, "UA %", ha="center", va="center", fontsize=6.0,
               fontweight="bold")
    for s in ax_cm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax_cm.set_title(f"Error matrix, n = {cm.sum():,} pixels", fontsize=7.9,
                    fontweight="bold", pad=9.0, x=0.50)
    add_panel_label(ax_cm, "(b)", x=-0.17, y=1.050, size=9.5)

    # ---- (c) NDVI phenology ----------------------------------------------
    doy = np.arange(5, 366, 5)
    rng = np.random.default_rng(3)
    for c, (name, col, (pk, amp, wd, gu)) in enumerate(
            zip(CROPS, CROP_COLORS, PHENO)):
        base = 0.16 + 0.03 * np.sin(doy / 60.0)
        curve = base + amp * np.exp(-0.5 * ((doy - pk) / (wd * 0.62)) ** 2)
        if c < 7:
            curve += 0.12 * amp * np.exp(-0.5 * ((doy - gu) / 26.0) ** 2)
        sd = 0.05 + 0.03 * np.exp(-0.5 * ((doy - pk) / (wd * 0.8)) ** 2)
        ax_ph.fill_between(doy, curve - sd, curve + sd, color=col,
                           alpha=0.16, linewidth=0)
        ax_ph.plot(doy, curve, color=col, lw=1.5 if c < 7 else 1.1,
                   ls="-" if c < 7 else "--", label=name, zorder=3 - 0.1 * c)
    ax_ph.set_xlim(0, 366)
    ax_ph.set_ylim(0.05, 1.24)
    ax_ph.set_xticks([15, 74, 135, 196, 258, 319])
    ax_ph.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
                          fontsize=6.6)
    ax_ph.set_xlabel("day of year (-)", fontsize=7.2, labelpad=1.5)
    ax_ph.set_ylabel("NDVI (-)", fontsize=7.2, labelpad=2)
    ax_ph.tick_params(labelsize=6.6, length=2, pad=1.5)
    ax_ph.grid(linewidth=0.4, alpha=0.55)
    ax_ph.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper center",
                 handlelength=1.2, columnspacing=0.9, labelspacing=0.25,
                 borderaxespad=0.1, bbox_to_anchor=(0.5, 1.015))
    style_axes(ax_ph)
    ax_ph.set_title("Class-mean NDVI phenology (shaded: 1 s.d.)",
                    fontsize=7.9, fontweight="bold", pad=3.0)
    add_panel_label(ax_ph, "(c)", x=-0.14, y=1.045, size=9.5)

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4,
                     label=nm) for c, nm in zip(CROP_COLORS, CROPS)]
    leg = fig.legend(handles=handles, loc="lower left", ncol=3,
                     frameon=False, fontsize=7.0, handlelength=1.2,
                     handleheight=1.0, columnspacing=1.0, labelspacing=0.42,
                     bbox_to_anchor=(0.600, 0.006))
    leg.set_clip_on(False)


    savefig(fig, "335_rsdl_crop_mapping.png")


if __name__ == "__main__":
    main()
