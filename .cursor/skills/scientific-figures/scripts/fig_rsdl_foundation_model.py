"""Figure 341 - A remote-sensing foundation model.

Top left: the masked-autoencoder pretext task, original / 70% masked /
reconstruction, with the same red window on all three.  Top right: how the
pretrained backbone transfers to six downstream tasks.  Bottom: downstream
accuracy against the amount of unlabelled pretraining imagery and against
the masking ratio.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

ZOOM_RED = "#b23a48"
WATER, CROP, FOREST, URBAN, SOIL = range(5)
TASKS = ["Land cover\nsegmentation", "Building\nextraction",
         "Road\nextraction", "Crop type\nmapping", "Change\ndetection",
         "Cloud\nmasking"]
MODELS = ["Random init", "ImageNet init", "Ours (RS-FM)"]
MODEL_COLORS = ["#9aa7b1", "#3b5b92", "#b23a48"]
SCORES = np.array([[0.601, 0.552, 0.573, 0.588, 0.616, 0.702],
                   [0.702, 0.664, 0.681, 0.694, 0.723, 0.795],
                   [0.812, 0.788, 0.764, 0.831, 0.842, 0.877]])


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


def scene(n=192, seed=23):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    anchors = rng.uniform(0.12, 0.88, size=(3, 2)) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)
    fid = np.zeros((n, n), int)
    for r in range(3):
        th = np.deg2rad(rng.uniform(-55, 55))
        cu, cv = rng.uniform(15, 24), rng.uniform(19, 30)
        u = xx * np.cos(th) + yy * np.sin(th) + 2.6 * _noise(rng, n, 14)
        v = -xx * np.sin(th) + yy * np.cos(th) + 2.6 * _noise(rng, n, 14)
        fid = np.where(reg == r,
                       np.floor(u / cu).astype(int) * 211
                       + np.floor(v / cv).astype(int) * 37 + r * 1013, fid)
    key = np.abs(fid) % 4096
    vig = rng.random(4096)[key] ** 0.65
    kind = rng.random(4096)[key]

    lab = np.full((n, n), CROP, np.int16)
    lab[kind < 0.14] = SOIL
    forest = gaussian_filter((_fbm(rng, n, (4, 11, 24)) > 0.76).astype(float),
                             1.4) > 0.45
    lab[forest] = FOREST
    vig = np.where(forest, 0.92, vig)
    ty = np.linspace(0, n - 1, 200)
    tx = 0.30 * n + 0.10 * n * np.sin(2 * np.pi * ty / n * 1.1 + 0.6)
    lab[_polyline_dist(n, np.c_[tx, ty]) <= 2.6] = WATER
    town = (np.exp(-(((xx - 0.72 * n) ** 2 + (yy - 0.70 * n) ** 2)
                     / (2 * (0.12 * n) ** 2)))
            + 0.30 * _fbm(rng, n, (5, 13), (0.5, 0.5))) > 0.63
    lab[town] = URBAN

    veg_spec, soil_spec = (0.036, 0.082, 0.036), (0.204, 0.184, 0.156)
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
        bands[k] = bands[k] * tex * (1 + 0.04 * _noise(rng, n, 28))
    lo, hi = 0.012, 0.285
    return np.dstack([np.clip((bands[i] - lo) / (hi - lo), 0, 1) ** 0.78
                      for i in range(3)])


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def add_zoom(ax, disp, box, loc=(0.600, 0.020, 0.380, 0.380)):
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


def main():
    set_pub_style(base_font=8)
    n, patch, ratio = 192, 16, 0.70
    img = scene(n, seed=23)
    rng = np.random.default_rng(9)

    g = n // patch
    keep = rng.random(g * g) > ratio
    mask = np.repeat(np.repeat(keep.reshape(g, g), patch, 0), patch, 1)
    masked = img.copy()
    masked[~mask] = 0.90
    grid = np.zeros((n, n), bool)
    grid[::patch, :] = True
    grid[:, ::patch] = True
    masked[grid & ~mask] = 0.78

    rec = np.dstack([gaussian_filter(img[..., c], 1.35) for c in range(3)])
    rec = rec + 0.012 * rng.standard_normal(rec.shape)
    recon = np.clip(np.where(mask[..., None], img, rec), 0, 1)

    box = (112, 24, 48, 48)
    fig = plt.figure(figsize=(10.0, 5.9))
    gs_a = fig.add_gridspec(1, 3, left=0.040, right=0.545, top=0.845,
                            bottom=0.500, wspace=0.040)
    gs_b = fig.add_gridspec(1, 2, left=0.075, right=0.978, top=0.400,
                            bottom=0.100, wspace=0.26)

    titles = ["Input tile", f"Masked ({int(100 * ratio)}%)", "Reconstruction"]
    for c, (t, im) in enumerate(zip(titles, [img, masked, recon])):
        ax = fig.add_subplot(gs_a[0, c])
        image_axes(ax)
        ax.imshow(im, interpolation="nearest")
        add_zoom(ax, im, box)
        ax.set_title(f"({'abc'[c]})  {t}", fontsize=7.8, fontweight="bold",
                     pad=3.4)

    # ---- (d) downstream radar --------------------------------------------
    ax_r = fig.add_axes([0.640, 0.522, 0.300, 0.352], projection="polar")
    ang = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    for v, nm, cc in zip(SCORES, MODELS, MODEL_COLORS):
        vc = np.concatenate([v, v[:1]])
        ax_r.plot(ang_c, vc, color=cc, lw=1.5, label=nm,
                  marker="o", ms=3.0, mec="white", mew=0.5)
        ax_r.fill(ang_c, vc, color=cc, alpha=0.13)
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels(TASKS, fontsize=6.4)
    ax_r.tick_params(axis="x", pad=1.0)
    ax_r.set_ylim(0.40, 0.95)
    ax_r.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_r.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=6.0,
                         color="#777777")
    ax_r.grid(linewidth=0.45, alpha=0.7)
    ax_r.spines["polar"].set_linewidth(0.6)
    ax_r.spines["polar"].set_color("#cccccc")
    ax_r.set_title("(d)  Fine-tuning across six downstream tasks",
                   fontsize=7.8, fontweight="bold", pad=13.0)
    leg = ax_r.legend(frameon=False, fontsize=6.6, ncol=3,
                      loc="upper center", bbox_to_anchor=(0.5, -0.085),
                      handlelength=1.3, columnspacing=1.0)
    leg.set_clip_on(False)

    # ---- (e) pretraining scale -------------------------------------------
    ax_e = fig.add_subplot(gs_b[0, 0])
    vol = np.array([0.05, 0.2, 1.0, 4.0, 16.0])
    curves = {
        "Land cover segmentation": np.array([0.681, 0.724, 0.768, 0.797,
                                             0.812]),
        "Building extraction": np.array([0.640, 0.692, 0.741, 0.772, 0.788]),
        "Crop type mapping": np.array([0.694, 0.744, 0.789, 0.818, 0.831]),
    }
    cc3 = ["#2e7d5b", "#b23a48", "#c9a227"]
    for (nm, v), c in zip(curves.items(), cc3):
        ax_e.plot(vol, v, color=c, lw=1.6, marker="o", ms=3.6, mec="white",
                  mew=0.5, label=nm)
        ax_e.fill_between(vol, v - 0.014, v + 0.014, color=c, alpha=0.14,
                          linewidth=0)
    ax_e.axhline(0.702, color="#808080", lw=0.9, ls="--")
    ax_e.text(0.055, 0.706, "ImageNet init baseline", fontsize=6.2,
              color="#666666", va="bottom")
    ax_e.set_xscale("log")
    ax_e.set_xticks(vol)
    ax_e.set_xticklabels(["50 k", "200 k", "1 M", "4 M", "16 M"],
                         fontsize=6.8)
    ax_e.minorticks_off()
    ax_e.set_ylim(0.60, 0.875)
    ax_e.set_xlabel("unlabelled pretraining tiles (-)", fontsize=7.4,
                    labelpad=1.5)
    ax_e.set_ylabel("downstream score (-)", fontsize=7.4, labelpad=2)
    ax_e.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_e.grid(linewidth=0.4, alpha=0.6)
    ax_e.legend(frameon=False, fontsize=6.6, loc="lower right",
                handlelength=1.4)
    style_axes(ax_e)
    ax_e.set_title("Returns keep coming but flatten past 4 M tiles",
                   fontsize=7.8, fontweight="bold", pad=3.0)
    add_panel_label(ax_e, "(e)", x=-0.115, y=1.05, size=11)

    # ---- (f) mask ratio ablation -----------------------------------------
    ax_f = fig.add_subplot(gs_b[0, 1])
    mr = np.array([0.25, 0.40, 0.55, 0.70, 0.85, 0.95])
    lc = np.array([0.744, 0.771, 0.797, 0.812, 0.795, 0.741])
    bd = np.array([0.719, 0.746, 0.774, 0.788, 0.769, 0.708])
    ax_f.plot(100 * mr, lc, color="#2e7d5b", lw=1.6, marker="o", ms=3.6,
              mec="white", mew=0.5, label="Land cover segmentation")
    ax_f.plot(100 * mr, bd, color="#b23a48", lw=1.6, marker="s", ms=3.4,
              mec="white", mew=0.5, label="Building extraction")
    best = int(np.argmax(lc))
    ax_f.axvline(100 * mr[best], color=ZOOM_RED, lw=0.9, ls=":")
    ax_f.annotate(f"best at {int(100 * mr[best])}%",
                  (100 * mr[best], lc[best]), textcoords="offset points",
                  xytext=(7, 5), fontsize=6.4, color=ZOOM_RED,
                  fontweight="bold")
    ax_f.set_ylim(0.68, 0.865)
    ax_f.set_xlabel("masking ratio (%)", fontsize=7.4, labelpad=1.5)
    ax_f.set_ylabel("downstream score (-)", fontsize=7.4, labelpad=2)
    ax_f.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_f.grid(linewidth=0.4, alpha=0.6)
    ax_f.legend(frameon=False, fontsize=6.6, loc="lower center",
                handlelength=1.4)
    style_axes(ax_f)
    ax_f.set_title("High masking ratios pretrain best", fontsize=7.8,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_f, "(f)", x=-0.115, y=1.05, size=11)

    handles = [Patch(facecolor="#e6e6e6", edgecolor="#8a8a8a", linewidth=0.5,
                     label="Masked 16 x 16 patch"),
               Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                     label="Zoom window (identical in a, b, c)")]
    leg2 = fig.legend(handles=handles, loc="lower left", ncol=1,
                      frameon=False, fontsize=6.8, handlelength=1.3,
                      bbox_to_anchor=(0.040, 0.455))
    leg2.set_clip_on(False)


    savefig(fig, "341_rsdl_foundation_model.png")


if __name__ == "__main__":
    main()
