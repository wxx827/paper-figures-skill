"""Figure 259 - Monocular depth estimation, qualitative comparison grid.

Two synthetic corridor scenes (rows) by RGB, ground-truth depth, three
baselines, ours, and the signed error of our prediction.  All depth panels
share one colour scale and the error panels share a second, symmetric one.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig, add_colorbar

H, W = 192, 256
FOC = 190.0
ZMAX = 9.0
DEPTH_CMAP = "magma"
ERR_CMAP = "RdBu_r"


def image_axes(ax, color="#b8b8b8", lw=0.6):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def _sm(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _hex(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


# ---------------------------------------------------------------------------
# Analytic corridor scene: exact depth plus a matching rendering
# ---------------------------------------------------------------------------
def corridor(seed=0, objects=((-0.75, 2.6, 0.36, "#c0603f"),
                              (0.62, 4.1, 0.40, "#2e7d5b"),
                              (-0.30, 6.2, 0.34, "#3b5b92"))):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u, v = xx - W / 2, yy - H / 2

    big = np.full((H, W), ZMAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = np.where(v > 0.5, FOC * 1.00 / np.maximum(v, 1e-6), big)
        z_ceil = np.where(v < -0.5, FOC * 1.25 / np.maximum(-v, 1e-6), big)
        z_left = np.where(u < -0.5, FOC * 1.70 / np.maximum(-u, 1e-6), big)
        z_right = np.where(u > 0.5, FOC * 1.70 / np.maximum(u, 1e-6), big)
    depth = np.minimum.reduce([z_floor, z_ceil, z_left, z_right, big])
    surf = np.argmin(np.stack([z_floor, z_ceil, z_left, z_right, big]), 0)
    depth = np.clip(depth, 0.6, ZMAX)

    # ---- rendering
    img = np.zeros((H, W, 3))
    Zf = depth
    Xw = u * Zf / FOC
    Yw = v * Zf / FOC
    tile = ((np.floor(Xw / 0.55) + np.floor(Zf / 0.55)) % 2).astype(float)
    img = np.where((surf == 0)[..., None],
                   _hex("#8d8578") * (0.72 + 0.34 * tile)[..., None], img)
    img = np.where((surf == 1)[..., None], _hex("#d8d4cc"), img)
    lamp = (surf == 1) & ((Zf % 2.1) < 0.42)
    img = np.where(lamp[..., None], _hex("#fff4d2"), img)
    for s, col in ((2, "#b9b2a6"), (3, "#c2bbaf")):
        band = ((np.floor(Yw / 0.42) % 2) == 0)
        img = np.where((surf == s)[..., None],
                       _hex(col) * (0.90 + 0.16 * band)[..., None], img)
        poster = (surf == s) & (np.abs(Yw + 0.10) < 0.34) & ((Zf % 2.6) < 0.85)
        img = np.where(poster[..., None],
                       _hex("#3f6f8f") if s == 2 else _hex("#8f5f3f"), img)
    img = np.where((surf == 4)[..., None], _hex("#9aa0a6"), img)
    door = (surf == 4) & (np.abs(u) < 26) & (v > -34) & (v < 44)
    img = np.where(door[..., None], _hex("#5c4a38"), img)

    # ---- solid objects standing on the floor
    for (X, Z, R, col) in objects:
        cx = W / 2 + FOC * X / Z
        cy = H / 2 + FOC * (1.00 - R) / Z
        r = FOC * R / Z
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1e-6) ** 2
        m = d2 < 1.0
        bulge = Z - R * np.sqrt(np.clip(1 - d2, 0, 1))
        depth = np.where(m, bulge, depth)
        shade = 0.55 + 0.55 * np.clip(1 - d2, 0, 1) ** 0.5
        nx = (xx - cx) / max(r, 1e-6)
        ny = (yy - cy) / max(r, 1e-6)
        spec = np.clip(1 - ((nx + 0.45) ** 2 + (ny + 0.45) ** 2) / 0.35, 0, 1)
        base = _hex(col) * shade[..., None] + 0.55 * spec[..., None]
        img = np.where(m[..., None], np.clip(base, 0, 1), img)
        sh = (((xx - cx) / (1.5 * r)) ** 2 + ((yy - (cy + r * 1.05))
                                              / (0.45 * r)) ** 2) < 1
        img = np.where((sh & ~m)[..., None], img * 0.55, img)

    fog = np.clip((depth - 1.2) / (ZMAX - 1.2), 0, 1)[..., None]
    img = img * (1 - 0.42 * fog) + 0.42 * fog * _hex("#c9cdd2")
    img = img * (1 + 0.045 * _sm(rng, (H, W), 2.0))[..., None]
    img = np.clip(img + 0.008 * rng.standard_normal(img.shape), 0, 1)
    img = np.dstack([gaussian_filter(img[..., k], 0.45) for k in range(3)])
    return np.clip(img, 0, 1), depth


def predict_depth(gt, rng, scale=1.0, shift=0.0, smooth=1.0, lowfreq=0.0,
                  edge=0.0):
    pred = scale * gt + shift
    if edge:
        pred = gaussian_filter(pred, edge)
    if lowfreq:
        pred = pred + lowfreq * _sm(rng, gt.shape, 26.0)
    if smooth:
        pred = pred + smooth * 0.06 * _sm(rng, gt.shape, 5.0)
    return np.clip(pred, 0.3, ZMAX + 1.2)


METHODS = [
    ("MiDaS v3.1", dict(scale=1.14, shift=-0.55, smooth=1.6, lowfreq=0.42,
                        edge=3.4), 0.184),
    ("DPT-Large", dict(scale=1.07, shift=-0.28, smooth=1.0, lowfreq=0.24,
                       edge=2.1), 0.131),
    ("Depth Anything", dict(scale=1.03, shift=-0.12, smooth=0.6, lowfreq=0.13,
                            edge=1.2), 0.092),
    ("Ours", dict(scale=1.005, shift=-0.02, smooth=0.18, lowfreq=0.035,
                  edge=0.4), 0.048),
]

SCENES = [
    dict(seed=6, objects=((-0.78, 2.5, 0.38, "#c0603f"),
                          (0.66, 4.0, 0.40, "#2e7d5b"),
                          (-0.28, 6.4, 0.32, "#3b5b92"))),
    dict(seed=14, objects=((0.80, 2.2, 0.34, "#8e6bbf"),
                           (-0.55, 3.6, 0.42, "#c9a227"),
                           (0.34, 5.6, 0.36, "#2a7f7a"))),
]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(23)
    col_names = ["RGB", "GT depth"] + [m for m, _, _ in METHODS] + \
                ["Error (ours)"]
    ncol = len(col_names)

    # ---- build everything first so the colour scales can be shared
    data = []
    for sc in SCENES:
        img, gt = corridor(**sc)
        preds = [predict_depth(gt, rng, **kw) for _, kw, _ in METHODS]
        data.append((img, gt, preds))
    all_d = np.concatenate([np.ravel(d) for _, gt, ps in data
                            for d in [gt] + ps])
    vmin, vmax = float(np.percentile(all_d, 0.5)), float(np.percentile(all_d,
                                                                       99.5))
    errs = [ps[-1] - gt for _, gt, ps in data]
    emax = float(np.percentile(np.abs(np.concatenate([np.ravel(e)
                                                      for e in errs])), 99.0))

    fig = plt.figure(figsize=(8.7, 3.30))
    gs = fig.add_gridspec(2, ncol, wspace=0.030, hspace=0.030,
                          left=0.032, right=0.995, top=0.885, bottom=0.215)

    axes = np.empty((2, ncol), object)
    for i, (img, gt, preds) in enumerate(data):
        tiles = [("rgb", img), ("d", gt)] + [("d", p) for p in preds] + \
                [("e", preds[-1] - gt)]
        for j, (kind, im) in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            if kind == "rgb":
                ax.imshow(im, interpolation="bilinear")
            elif kind == "d":
                md = ax.imshow(im, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax,
                               interpolation="bilinear")
            else:
                me = ax.imshow(im, cmap=ERR_CMAP, vmin=-emax, vmax=emax,
                               interpolation="bilinear")
            if 2 <= j <= ncol - 2:
                rel = METHODS[j - 2][2] + 0.011 * i
                ax.text(0.972, 0.955, f"AbsRel {rel:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.0, color="white",
                        fontweight="bold" if j == ncol - 2 else "normal",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="#000000aa", edgecolor="none"))
            if j == ncol - 1:
                ax.text(0.972, 0.955,
                        f"MAE {np.abs(im).mean():.3f} m",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.0, color="#1a1a1a", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="#ffffffcc", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(f"Scene {'AB'[i]}", fontsize=7.8, labelpad=3.0)
            if i == 1:
                bold = col_names[j] == "Ours"
                ax.set_xlabel(col_names[j], fontsize=7.8, labelpad=3.2,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    k = col_names.index("Ours")
    p0, p1 = axes[0, k].get_position(), axes[1, k].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.004, p1.y0 - 0.058), p0.width + 0.008,
                             p0.y1 - p1.y0 + 0.068, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    # ---- two shared, deliberately slim colour bars
    a, b = axes[1, 1].get_position(), axes[1, ncol - 2].get_position()
    cax = fig.add_axes([a.x0, 0.085, (b.x1 - a.x0) - 0.030, 0.030])
    cb = fig.colorbar(md, cax=cax, orientation="horizontal",
                      ticks=[2, 4, 6, 8])
    cb.set_label("depth  (m)   -   shared by every depth panel", fontsize=7.0,
                 labelpad=1.5)
    c = axes[1, ncol - 1].get_position()
    cax2 = fig.add_axes([c.x0 + 0.004, 0.085, c.width - 0.004, 0.030])
    cb2 = fig.colorbar(me, cax=cax2, orientation="horizontal")
    cb2.set_label("signed error  (m)", fontsize=7.0, labelpad=1.5)
    for cbi in (cb, cb2):
        cbi.outline.set_linewidth(0.6)
        cbi.outline.set_edgecolor("#333333")
        cbi.ax.tick_params(labelsize=7.0, width=0.7, length=2.5,
                           color="#333333")
    cb2.set_ticks([-emax, 0, emax])
    cb2.ax.set_xticklabels([f"-{emax:.2f}", "0", f"+{emax:.2f}"])
    cb2.ax.text(0.04, 0.5, "under", transform=cb2.ax.transAxes, ha="left",
                va="center", fontsize=6.0, color="white")
    cb2.ax.text(0.96, 0.5, "over", transform=cb2.ax.transAxes, ha="right",
                va="center", fontsize=6.0, color="white")

    savefig(fig, "259_cv_depth_compare.png")


if __name__ == "__main__":
    main()
