"""Figure 337 - SAR-guided reconstruction of cloud-obscured optical imagery.

Top row: the cloudy acquisition, the co-registered SAR image, two
reconstructions and the cloud-free reference.  A subordinate row (58%
height) magnifies the same window in every column, and three compact
panels quantify PSNR, SSIM and how they degrade with cloud cover.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, style_axes

ZOOM_RED = "#b23a48"
GSD = 10.0
WATER, CROP, FOREST, GRASS, URBAN, SOIL = range(6)


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


def scene(n=224, seed=13):
    """Land-cover labels, per-parcel vigour and a true-colour reference."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    anchors = np.array([[0.22, 0.20], [0.78, 0.22], [0.30, 0.80],
                        [0.80, 0.74]]) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 12 * _noise(rng, n, 22)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)
    fid = np.zeros((n, n), int)
    for r, (ang, cu, cv) in enumerate([(17, 22, 29), (-26, 26, 18),
                                       (39, 18, 24), (6, 28, 21)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 3.0 * _noise(rng, n, 15)
        v = -xx * np.sin(th) + yy * np.cos(th) + 3.0 * _noise(rng, n, 15)
        fid = np.where(reg == r,
                       np.floor(u / cu).astype(int) * 211
                       + np.floor(v / cv).astype(int) * 37 + r * 1013, fid)
    key = np.abs(fid) % 4096
    vig = 0.28 + 0.68 * rng.random(4096)[key]
    kind = rng.random(4096)[key]

    lab = np.full((n, n), CROP, np.int16)
    lab[kind < 0.11] = SOIL
    lab[(kind >= 0.11) & (kind < 0.24)] = GRASS
    forest = gaussian_filter((_fbm(rng, n, (5, 13, 28)) > 0.74).astype(float),
                             1.5) > 0.44
    lab[forest] = FOREST
    vig = np.where(forest, 0.90, vig)

    ty = np.linspace(0, n - 1, 240)
    tx = (0.30 * n + 0.10 * n * np.sin(2 * np.pi * ty / n * 1.05)
          + 0.04 * n * np.sin(2 * np.pi * ty / n * 2.6 + 1.3))
    lab[_polyline_dist(n, np.c_[tx, ty]) <= 2.6] = WATER
    town = (np.exp(-(((xx - 0.74 * n) ** 2 + (yy - 0.68 * n) ** 2)
                     / (2 * (0.105 * n) ** 2)))
            + 0.30 * _fbm(rng, n, (5, 13), (0.5, 0.5))) > 0.66
    lab[town] = URBAN

    veg_spec = (0.034, 0.082, 0.034)
    soil_spec = (0.198, 0.178, 0.150)
    fc = np.clip(vig, 0.04, 1.0)
    bands = np.stack([fc * veg_spec[k] + (1 - fc) * soil_spec[k]
                      for k in range(3)])
    for k, val in enumerate((0.030, 0.048, 0.058)):
        bands[k] = np.where(lab == WATER, val, bands[k])
    for k, val in enumerate((0.036, 0.068, 0.040)):
        bands[k] = np.where(lab == FOREST, val, bands[k])
    for k, val in enumerate((0.168, 0.162, 0.158)):
        bands[k] = np.where(lab == URBAN, val, bands[k])
    amp = np.where(lab == FOREST, 0.24, np.where(lab == URBAN, 0.38, 0.08))
    tex = 1 + amp * (0.6 * _noise(rng, n, 1.0) + 0.4 * _noise(rng, n, 3.0))
    for k in range(3):
        bands[k] = bands[k] * tex * (1 + 0.04 * _noise(rng, n, 30))

    p0, p1 = np.percentile(bands, [1.2, 99.0])
    ref = np.dstack([np.clip((bands[i] - p0) / max(p1 - p0, 1e-9), 0, 1) ** g
                     for i, g in zip((0, 1, 2), (0.80, 0.80, 0.86))])
    return lab, vig, ref


def make_sar(lab, vig, seed=21):
    """Single-look-ish VV backscatter with multiplicative speckle."""
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    sig0 = np.full((n, n), -11.0)
    sig0[lab == WATER] = -22.0
    sig0[lab == CROP] = -10.5 - 4.0 * (1 - vig[lab == CROP])
    sig0[lab == GRASS] = -13.0
    sig0[lab == FOREST] = -7.5
    sig0[lab == SOIL] = -14.5
    sig0[lab == URBAN] = -2.0
    lin = 10 ** (sig0 / 10.0)
    speckle = rng.gamma(shape=4.0, scale=0.25, size=(n, n))
    lin = lin * speckle
    db = 10 * np.log10(np.maximum(lin, 1e-6))
    db = gaussian_filter(db, 0.55)
    return np.clip((db + 25.0) / 27.0, 0, 1)


def cloud_field(n, seed=8):
    rng = np.random.default_rng(seed)
    f = _fbm(rng, n, (6, 15, 34), (0.42, 0.34, 0.24))
    a = np.clip((f - 0.28) * 1.45, 0, 1)
    a = gaussian_filter(a, 2.2)
    a = np.clip(a / max(a.max(), 1e-9), 0, 1) ** 0.85
    core = a > 0.62
    return a, core


def ssim(a, b, sigma=2.0):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    saa = gaussian_filter(a * a, sigma) - mu_a ** 2
    sbb = gaussian_filter(b * b, sigma) - mu_b ** 2
    sab = gaussian_filter(a * b, sigma) - mu_a * mu_b
    s = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
         / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)))
    return s


def psnr(a, b, mask):
    mse = float(np.mean((a[mask] - b[mask]) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


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
    h = 0.020 * n
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
    n = 224
    lab, vig, ref = scene(n, seed=13)
    sar = make_sar(lab, vig, seed=21)
    alpha, core = cloud_field(n, seed=8)
    rng = np.random.default_rng(2)

    cloud_rgb = np.dstack([0.93 + 0.05 * _noise(rng, n, 3.0)] * 3)
    cloud_rgb = np.clip(cloud_rgb, 0, 1)
    obs = np.clip(ref * (1 - alpha[..., None])
                  + cloud_rgb * alpha[..., None], 0, 1)

    def reconstruct(sig_blur, bias, sar_bleed, noise, seed):
        r = np.random.default_rng(seed)
        rec = np.dstack([gaussian_filter(ref[..., c], sig_blur)
                         for c in range(3)])
        rec = rec * (1.0 + bias * np.array([1.0, -0.4, -0.8]))
        rec = rec + sar_bleed * (sar - sar.mean())[..., None]
        rec = rec + noise * r.standard_normal(rec.shape)
        w = np.clip(alpha * 1.35, 0, 1)[..., None]
        return np.clip(obs * (1 - w) + rec * w, 0, 1)

    base = reconstruct(1.95, 0.052, 0.062, 0.015, 41)
    ours = reconstruct(0.80, 0.014, 0.018, 0.005, 42)

    m = core
    stats = {}
    for nm, img in (("Fusion baseline", base), ("Ours (SAR-Diff)", ours)):
        g_a = img.mean(axis=2)
        g_b = ref.mean(axis=2)
        stats[nm] = (psnr(img, ref, np.repeat(m[..., None], 3, axis=2)),
                     float(ssim(g_a, g_b)[m].mean()))
        print(f"  {nm:18s} PSNR = {stats[nm][0]:.2f} dB   "
              f"SSIM = {stats[nm][1]:.3f}")

    # PSNR against cloud-cover fraction (independent cloud realisations)
    # Nested masks from one cloud field keep the trend monotonic; wider gaps
    # mean less optical context, so both models degrade.
    fracs = np.array([0.10, 0.20, 0.30, 0.45, 0.60, 0.75])
    curves = {"Fusion baseline": [], "Ours (SAR-Diff)": []}
    for i, f in enumerate(fracs):
        m2 = alpha > np.quantile(alpha, 1 - f)
        m3 = np.repeat(m2[..., None], 3, axis=2)
        for nm, (sb, bi, sl, no, sd) in (
                ("Fusion baseline", (1.95, 0.052, 0.062, 0.015, 41)),
                ("Ours (SAR-Diff)", (0.80, 0.014, 0.018, 0.005, 42))):
            r = np.random.default_rng(sd)
            grow = 0.72 + 0.85 * f
            rec = np.dstack([gaussian_filter(ref[..., c], sb * grow)
                             for c in range(3)])
            rec = rec * (1.0 + bi * grow * np.array([1.0, -0.4, -0.8]))
            rec = rec + sl * grow * (sar - sar.mean())[..., None]
            rec = rec + no * grow * r.standard_normal(rec.shape)
            curves[nm].append(psnr(np.clip(rec, 0, 1), ref, m3))

    cols = ["Optical (cloudy)", "SAR (VV, dB)", "Fusion baseline",
            "Ours (SAR-Diff)", "Reference (cloud-free)"]
    box = (118, 26, 62, 62)

    fig = plt.figure(figsize=(10.0, 5.8))
    gs_a = fig.add_gridspec(1, 5, left=0.036, right=0.988, top=0.876,
                            bottom=0.545, wspace=0.030)
    gs_b = fig.add_gridspec(1, 5, left=0.036, right=0.988, top=0.500,
                            bottom=0.310, wspace=0.030)
    gs_c = fig.add_gridspec(1, 3, left=0.072, right=0.978, top=0.238,
                            bottom=0.078, wspace=0.30)

    imgs = [obs, sar, base, ours, ref]
    top_axes = []
    for c, (title, img) in enumerate(zip(cols, imgs)):
        ax = fig.add_subplot(gs_a[0, c])
        top_axes.append(ax)
        image_axes(ax)
        ax.patch.set_visible(False)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2], box[3],
                               fill=False, edgecolor=ZOOM_RED, linewidth=1.0,
                               zorder=7))
        ax.set_title(f"({'abcde'[c]})  {title}", fontsize=7.6,
                     fontweight="bold", pad=3.4,
                     color="#7a1f2b" if c == 3 else "#1a1a1a")
        if c == 0:
            add_scale_bar(ax, n)
            ax.contour(core.astype(float), levels=[0.5], colors=["#3b5b92"],
                       linewidths=0.5, linestyles="--")

        axz = fig.add_subplot(gs_b[0, c])
        image_axes(axz)
        for s in axz.spines.values():
            s.set_linewidth(1.0)
            s.set_color(ZOOM_RED)
        crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        if img.ndim == 2:
            axz.imshow(crop, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
        else:
            axz.imshow(crop, interpolation="nearest")
        if c == 0:
            axz.set_ylabel("Zoom 3.6x", fontsize=6.8, fontweight="bold",
                           labelpad=3, color="#7a1f2b")
        if title in stats:
            p_, s_ = stats[title]
            best = title.startswith("Ours")
            axz.text(0.5, -0.075, f"PSNR {p_:.2f} dB    SSIM {s_:.3f}",
                     transform=axz.transAxes, ha="center", va="top",
                     fontsize=7.0, fontweight="bold" if best else "normal",
                     color="#7a1f2b" if best else "#444444")

    p_t = top_axes[3].get_position()
    p_z = fig.axes[-1].get_position()
    fig.patches.append(Rectangle(
        (p_t.x0 - 0.006, 0.262), p_t.width + 0.012, (p_t.y1 + 0.040) - 0.262,
        transform=fig.transFigure, facecolor="#f2efe9", edgecolor="none",
        zorder=-1))

    # ---- subordinate quantitative row -------------------------------------
    names = list(stats)
    bar_c = ["#7d8f9e", "#b23a48"]

    ax1 = fig.add_subplot(gs_c[0, 0])
    x = np.arange(2)
    ax1.bar(x, [stats[k][0] for k in names], 0.52, color=bar_c,
            edgecolor="#333333", linewidth=0.5)
    for xi, k in zip(x, names):
        ax1.text(xi, stats[k][0] + 0.25, f"{stats[k][0]:.2f}", ha="center",
                 va="bottom", fontsize=6.6, color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Baseline", "Ours"], fontsize=7.0)
    ax1.set_ylim(0, max(stats[k][0] for k in names) * 1.22)
    ax1.set_ylabel("PSNR (dB)", fontsize=7.4, labelpad=2)
    ax1.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax1)
    ax1.set_title("(f)  Reconstruction fidelity", fontsize=7.8,
                  fontweight="bold", pad=3.0)

    ax2 = fig.add_subplot(gs_c[0, 1])
    ax2.bar(x, [stats[k][1] for k in names], 0.52, color=bar_c,
            edgecolor="#333333", linewidth=0.5)
    for xi, k in zip(x, names):
        ax2.text(xi, stats[k][1] + 0.012, f"{stats[k][1]:.3f}", ha="center",
                 va="bottom", fontsize=6.6, color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Baseline", "Ours"], fontsize=7.0)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("SSIM (-)", fontsize=7.4, labelpad=2)
    ax2.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax2)
    ax2.set_title("(g)  Structural similarity", fontsize=7.8,
                  fontweight="bold", pad=3.0)

    ax3 = fig.add_subplot(gs_c[0, 2])
    for nm, cc, mk in zip(names, bar_c, ("s", "o")):
        ax3.plot(100 * fracs, curves[nm], color=cc, lw=1.5, marker=mk,
                 ms=3.4, mec="white", mew=0.5, label=nm)
    ax3.set_xlabel("cloud cover (%)", fontsize=7.4, labelpad=1.5)
    ax3.set_ylabel("PSNR (dB)", fontsize=7.4, labelpad=2)
    ax3.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax3.grid(linewidth=0.4, alpha=0.6)
    ax3.legend(frameon=False, fontsize=6.6, loc="lower left",
               handlelength=1.4)
    style_axes(ax3)
    ax3.set_title("(h)  Sensitivity to cloud cover", fontsize=7.8,
                  fontweight="bold", pad=3.0)

    handles = [Patch(facecolor="none", edgecolor="#3b5b92", linewidth=0.9,
                     linestyle="--", label="Cloud core (metrics computed "
                                           "inside)"),
               Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                     label="Zoom window (identical in all columns)")]
    leg = fig.legend(handles=handles, loc="lower left", ncol=2,
                     frameon=False, fontsize=6.8, handlelength=1.4,
                     bbox_to_anchor=(0.072, 0.004))
    leg.set_clip_on(False)


    savefig(fig, "337_rsdl_sar_optical_fusion.png")


if __name__ == "__main__":
    main()
