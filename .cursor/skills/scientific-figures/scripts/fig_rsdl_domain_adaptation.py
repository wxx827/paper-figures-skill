"""Figure 340 - Cross-domain land-cover mapping.

Top row: a labelled source scene, an unlabelled target scene from another
sensor and region, the source model transferred directly, and the same model
after unsupervised adaptation.  Bottom row: the matching feature embeddings,
which show the domain gap opening and then closing.  The right column
carries the accuracy bars.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

# house red already carries the built-up class and the target domain here, so
# the zoom chrome takes house purple instead
ZOOM_RED = "#8e6bbf"
WATER, CROP, FOREST, URBAN, SOIL = range(5)
LC_NAMES = ["Water", "Cropland", "Forest", "Built-up", "Bare soil"]
LC_COLORS = ["#3b7ea1", "#8fbc4a", "#1d6b3f", "#b23a48", "#d9a86c"]
SRC_BLUE = "#3b5b92"
TGT_RED = "#b23a48"


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


def domain_scene(n=176, seed=3, pitch=(20, 27), gain=(1.0, 1.0, 1.0),
                 haze=0.0, urban_frac=0.11):
    """Land-cover labels plus an RGB rendering with a sensor-specific cast."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    anchors = rng.uniform(0.12, 0.88, size=(3, 2)) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)
    fid = np.zeros((n, n), int)
    for r in range(3):
        ang = rng.uniform(-55, 55)
        cu = pitch[0] * rng.uniform(0.8, 1.25)
        cv = pitch[1] * rng.uniform(0.8, 1.25)
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 2.6 * _noise(rng, n, 14)
        v = -xx * np.sin(th) + yy * np.cos(th) + 2.6 * _noise(rng, n, 14)
        fid = np.where(reg == r,
                       np.floor(u / cu).astype(int) * 211
                       + np.floor(v / cv).astype(int) * 37 + r * 1013, fid)
    key = np.abs(fid) % 4096
    vig = (rng.random(4096)[key]) ** 0.7
    kind = rng.random(4096)[key]

    lab = np.full((n, n), CROP, np.int16)
    lab[kind < 0.14] = SOIL
    forest = gaussian_filter((_fbm(rng, n, (4, 11, 24)) > 0.78).astype(float),
                             1.4) > 0.45
    lab[forest] = FOREST
    vig = np.where(forest, 0.92, vig)

    ty = np.linspace(0, n - 1, 200)
    tx = (rng.uniform(0.22, 0.40) * n
          + 0.09 * n * np.sin(2 * np.pi * ty / n * rng.uniform(0.8, 1.3)
                              + rng.uniform(0, 6)))
    lab[_polyline_dist(n, np.c_[tx, ty]) <= 2.2] = WATER
    cx, cy = rng.uniform(0.55, 0.82), rng.uniform(0.55, 0.82)
    town = (np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2)
                     / (2 * (urban_frac * n) ** 2)))
            + 0.30 * _fbm(rng, n, (5, 13), (0.5, 0.5))) > 0.64
    lab[town] = URBAN

    veg_spec = (0.036, 0.082, 0.036)
    soil_spec = (0.204, 0.184, 0.156)
    fc = np.clip(vig, 0.04, 1.0)
    bands = np.stack([fc * veg_spec[k] + (1 - fc) * soil_spec[k]
                      for k in range(3)])
    for k, val in enumerate((0.028, 0.046, 0.058)):
        bands[k] = np.where(lab == WATER, val, bands[k])
    for k, val in enumerate((0.034, 0.066, 0.038)):
        bands[k] = np.where(lab == FOREST, val, bands[k])
    for k, val in enumerate((0.164, 0.158, 0.154)):
        bands[k] = np.where(lab == URBAN, val, bands[k])
    amp = np.where(lab == FOREST, 0.24, np.where(lab == URBAN, 0.36, 0.08))
    tex = 1 + amp * (0.6 * _noise(rng, n, 1.0) + 0.4 * _noise(rng, n, 2.8))
    for k in range(3):
        bands[k] = bands[k] * tex * gain[k] * (1 + 0.04 * _noise(rng, n, 28))
        bands[k] = bands[k] * (1 - haze) + haze * 0.20

    lo, hi = 0.012, 0.285
    rgb = np.dstack([np.clip((bands[i] - lo) / (hi - lo), 0, 1) ** 0.78
                     for i in range(3)])
    return lab, rgb


def classify(lab, seed, err, blob_sigma, conf):
    """Classifier whose error rate is exactly ``err``, concentrated on
    class boundaries and on coherent blobs rather than on stray pixels."""
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    b = _noise(rng, n, blob_sigma)
    b = (b - b.min()) / (np.ptp(b) + 1e-9)
    edge = gaussian_filter((np.abs(np.gradient(lab.astype(float))[0])
                            + np.abs(np.gradient(lab.astype(float))[1]) > 0)
                           .astype(float), 1.3)
    edge = edge / (edge.max() + 1e-9)
    score = 0.60 * b + 0.40 * edge
    hit = score > np.quantile(score, 1.0 - err)
    pred = lab.copy()
    for a, c in conf.items():
        pred[hit & (lab == a)] = c
    return pred


def overlay(rgb, lab, alpha=0.52):
    lut = np.array([plt.matplotlib.colors.to_rgb(c) for c in LC_COLORS])
    return np.clip((1 - alpha) * rgb + alpha * lut[lab], 0, 1)


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def add_zoom(ax, disp, box, loc=(0.595, 0.020, 0.385, 0.385)):
    x0, y0, w, h = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                           edgecolor=ZOOM_RED, linewidth=0.9, zorder=7))
    ins = ax.inset_axes(loc)
    ins.set_zorder(8)
    ins.imshow(disp[y0:y0 + h, x0:x0 + w], interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color(ZOOM_RED)
    return ins


def clusters(rng, centres, spread, m=150):
    pts, lab = [], []
    for c, ctr in enumerate(centres):
        p = rng.normal(ctr, spread, size=(m, 2))
        p += 0.35 * rng.standard_normal((m, 2)) ** 3
        pts.append(p)
        lab.append(np.full(m, c))
    return np.vstack(pts), np.concatenate(lab)


def main():
    set_pub_style(base_font=8)
    n = 176
    box = (96, 22, 52, 52)

    src_lab, src_rgb = domain_scene(n, seed=3, pitch=(21, 28),
                                    gain=(1.0, 1.0, 1.0), haze=0.0)
    tgt_lab, tgt_rgb = domain_scene(n, seed=44, pitch=(12, 16),
                                    gain=(1.20, 0.95, 0.80), haze=0.24,
                                    urban_frac=0.13)

    conf = {CROP: SOIL, FOREST: CROP, URBAN: SOIL, SOIL: CROP,
            WATER: FOREST}
    src_pred = classify(src_lab, 11, 0.079, 5.0, conf)
    direct = classify(tgt_lab, 12, 0.366, 4.0, conf)
    adapted = classify(tgt_lab, 13, 0.158, 5.5, conf)

    acc = {"Source -> source": float((src_pred == src_lab).mean()),
           "Source -> target": float((direct == tgt_lab).mean()),
           "After adaptation": float((adapted == tgt_lab).mean()),
           "Target supervised": 0.0}
    acc["Target supervised"] = acc["After adaptation"] + 0.046
    for k, v in acc.items():
        print(f"  {k:20s} OA = {v:.3f}")

    fig = plt.figure(figsize=(10.2, 5.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.80],
                          height_ratios=[1.0, 0.72], wspace=0.075,
                          hspace=0.235, left=0.036, right=0.982, top=0.862,
                          bottom=0.130)

    panels = [
        ("Source domain", overlay(src_rgb, src_lab), "reference labels"),
        ("Target domain", tgt_rgb, "no labels available"),
        ("Direct transfer", overlay(tgt_rgb, direct),
         f"OA {acc['Source -> target']:.3f}"),
        ("After adaptation", overlay(tgt_rgb, adapted),
         f"OA {acc['After adaptation']:.3f}"),
    ]
    top_axes = []
    for c, (title, img, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        top_axes.append(ax)
        image_axes(ax)
        ax.patch.set_visible(False)
        ax.imshow(img, interpolation="nearest")
        add_zoom(ax, img, box)
        ax.set_title(f"({'abcd'[c]})  {title}", fontsize=7.8,
                     fontweight="bold", pad=3.4,
                     color="#7a1f2b" if c == 3 else "#1a1a1a")
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.8,
                fontweight="bold" if c == 3 else "normal",
                color="#7a1f2b" if c == 3 else "#555555")

    # ---- bottom row: feature embeddings -----------------------------------
    rng = np.random.default_rng(5)
    centres = np.array([[-3.1, 2.4], [1.9, 3.2], [3.4, -1.4],
                        [-2.6, -2.9], [0.4, -0.4]])
    src_pts, src_cls = clusters(rng, centres, 0.62)
    tgt_pts, tgt_cls = clusters(rng, centres + np.array([4.6, 3.4]), 1.02)
    tgt_align, _ = clusters(rng, centres + np.array([0.30, 0.22]), 0.86)

    emb = [("Source features", src_pts, src_cls, None),
           ("Target features", tgt_pts, tgt_cls, None),
           ("Before adaptation", (src_pts, tgt_pts), None, "domain"),
           ("After adaptation", (src_pts, tgt_align), None, "domain")]

    lim = np.array([[-9.5, 11.5], [-8.0, 10.5]])
    for c, (title, data, cls, mode) in enumerate(emb):
        ax = fig.add_subplot(gs[1, c])
        if mode is None:
            for k in range(5):
                m = cls == k
                ax.scatter(data[m, 0], data[m, 1], s=3.0,
                           color=LC_COLORS[k], alpha=0.72, linewidths=0)
        else:
            a, b = data
            ax.scatter(a[:, 0], a[:, 1], s=3.0, color=SRC_BLUE, alpha=0.55,
                       linewidths=0)
            ax.scatter(b[:, 0], b[:, 1], s=3.0, color=TGT_RED, alpha=0.55,
                       linewidths=0)
            gap = np.linalg.norm(a.mean(0) - b.mean(0))
            ax.text(0.03, 0.955, f"MMD gap {gap:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.4, fontweight="bold",
                    color="#7a1f2b" if c == 3 else "#333333")
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color("#8a8a8a")
        ax.set_title(f"({'efgh'[c]})  {title}", fontsize=7.4,
                     fontweight="bold", pad=3.0,
                     color="#7a1f2b" if c == 3 else "#1a1a1a")
        if c == 0:
            ax.set_ylabel("t-SNE dim 2 (-)", fontsize=6.8, labelpad=3)
        ax.set_xlabel("t-SNE dim 1 (-)", fontsize=6.8, labelpad=2)

    # ---- (i) accuracy bars ------------------------------------------------
    ax_b = fig.add_subplot(gs[:, 4])
    names = ["Source -> source", "Source -> target", "After adaptation",
             "Target supervised"]
    vals = np.array([acc["Source -> source"], acc["Source -> target"],
                     acc["After adaptation"], acc["Target supervised"]])
    cols = ["#9aa7b1", "#c6ccd1", "#b23a48", "#5f7688"]
    y = np.arange(4)[::-1]
    ax_b.barh(y, vals, 0.42, color=cols, edgecolor="#333333", linewidth=0.5)
    for yi, v, nm in zip(y, vals, names):
        best = nm == "After adaptation"
        ax_b.text(0.014, yi + 0.26, nm, va="bottom", ha="left",
                  fontsize=6.8, color="#7a1f2b" if best else "#333333",
                  fontweight="bold" if best else "normal")
        ax_b.text(v + 0.014, yi, f"{v:.3f}", va="center", fontsize=6.8,
                  color="#7a1f2b" if best else "#333333",
                  fontweight="bold" if best else "normal")
    ax_b.annotate("", xy=(vals[2], y[1] - 0.34), xytext=(vals[1], y[1] - 0.34),
                  arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#7a1f2b",
                                  shrinkA=0, shrinkB=0))
    ax_b.text(0.5 * (vals[1] + vals[2]), y[1] - 0.44,
              f"+{100 * (vals[2] - vals[1]):.1f} pp", ha="center",
              va="top", fontsize=6.6, fontweight="bold", color="#7a1f2b")
    ax_b.set_yticks([])
    ax_b.set_ylim(-0.75, 3.62)
    ax_b.set_xlim(0, 1.16)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xlabel("overall accuracy (-)", fontsize=7.4, labelpad=1.5)
    ax_b.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_b.grid(axis="x", linewidth=0.4, alpha=0.6)
    style_axes(ax_b)
    ax_b.set_title("(i)  Domain gap closed", fontsize=7.8,
                   fontweight="bold", pad=3.4)

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4,
                     label=nm) for c, nm in zip(LC_COLORS, LC_NAMES)]
    handles += [Line2D([], [], ls="none", marker="o", ms=4, mfc=SRC_BLUE,
                       mec="none", label="Source sample"),
                Line2D([], [], ls="none", marker="o", ms=4, mfc=TGT_RED,
                       mec="none", label="Target sample"),
                Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                      label="Zoom window")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=8,
                     frameon=False, fontsize=6.8, handlelength=1.2,
                     handleheight=1.0, columnspacing=1.1,
                     bbox_to_anchor=(0.50, 0.004))
    leg.set_clip_on(False)


    savefig(fig, "340_rsdl_domain_adaptation.png")


if __name__ == "__main__":
    main()
