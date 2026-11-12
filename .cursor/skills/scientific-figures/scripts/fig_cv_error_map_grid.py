"""Figure 262 - Error-map comparison matrix for optical-flow estimation.

Three sequences (rows) by input frame and three estimators (columns).  The main
tile carries the colour-coded flow, the lower-right inset the end-point-error
map; all error insets share one colour scale so the tiles stay comparable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig

N = 200
ERR_CMAP = "inferno"
INSET_EDGE = "#2f3b46"
INSET = (0.575, 0.0, 0.425, 0.425)


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
# Scene + ground-truth flow field
# ---------------------------------------------------------------------------
def sequence(seed, objects):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)

    # textured background frame: luminance-dominated so it reads as a photo
    lum = 0.58 + 0.20 * (0.6 * _sm(rng, (N, N), 6.0)
                         + 0.4 * _sm(rng, (N, N), 14.0))
    grain = 0.16 * _sm(rng, (N, N), 1.1)
    chroma = 0.05 * np.dstack([_sm(rng, (N, N), 11.0) for _ in range(3)])
    img = np.clip((lum + grain)[..., None] * (_hex("#9c9885") + chroma), 0, 1)

    # global camera motion
    cx, cy = N / 2, N / 2
    om = rng.uniform(-0.010, 0.010)
    u = 2.4 * np.cos(rng.uniform(0, 6)) - om * (yy - cy)
    v = 2.4 * np.sin(rng.uniform(0, 6)) + om * (xx - cx)

    for (ox, oy, orr, ou, ov, col) in objects:
        d = np.sqrt(((xx - ox) / orr) ** 2 + ((yy - oy) / orr) ** 2)
        m = d < 1.0
        shade = 0.60 + 0.55 * np.clip(1 - d, 0, 1) ** 0.6
        img = np.where(m[..., None],
                       np.clip(_hex(col) * shade[..., None]
                               + 0.12 * grain[..., None], 0, 1), img)
        soft = np.clip(1.4 - d, 0, 1) ** 1.6
        u = u * (1 - soft) + ou * soft
        v = v * (1 - soft) + ov * soft

    u = u + 0.55 * _sm(rng, (N, N), 22)
    v = v + 0.55 * _sm(rng, (N, N), 22)
    img = np.clip(np.dstack([gaussian_filter(img[..., k], 0.45)
                             for k in range(3)]), 0, 1)
    return img, np.dstack([u, v])


def flow_rgb(flow, vmax):
    mag = np.hypot(flow[..., 0], flow[..., 1])
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    h = (ang + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = np.clip(mag / vmax, 0, 1) ** 0.75
    return hsv_to_rgb(np.dstack([h, s, v]))


def predict_flow(gt, rng, smooth=1.0, bias=0.0, edge=0.0, noise=0.0):
    out = gt.copy()
    if edge:
        out = np.dstack([gaussian_filter(out[..., k], edge) for k in range(2)])
    if bias:
        out = out + bias * np.dstack([_sm(rng, (N, N), 30),
                                      _sm(rng, (N, N), 30)])
    if smooth:
        out = out + smooth * 0.25 * np.dstack([_sm(rng, (N, N), 7),
                                               _sm(rng, (N, N), 7)])
    if noise:
        out = out + noise * rng.standard_normal(out.shape)
    return out


METHODS = [("PWC-Net", dict(smooth=1.7, bias=1.25, edge=4.5, noise=0.10)),
           ("RAFT", dict(smooth=0.9, bias=0.60, edge=2.2, noise=0.05)),
           ("Ours", dict(smooth=0.25, bias=0.16, edge=0.6, noise=0.015))]

SEQUENCES = [
    dict(seed=51, objects=[(66, 78, 30, -6.5, 3.0, "#c0603f"),
                           (140, 130, 26, 5.5, -4.0, "#3b5b92")]),
    dict(seed=57, objects=[(112, 66, 34, 4.0, 6.0, "#2e7d5b"),
                           (58, 148, 24, -5.0, -5.5, "#8e6bbf")]),
    dict(seed=63, objects=[(96, 108, 40, 7.0, -2.0, "#c9a227"),
                           (166, 54, 22, -4.5, 5.0, "#2a7f7a")]),
]


def color_wheel(ax, n=120):
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    h = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    v = np.clip(r, 0, 1) ** 0.75
    rgb = hsv_to_rgb(np.dstack([h, np.ones_like(h), v]))
    rgb = np.where((r > 1)[..., None], 1.0, rgb)
    ax.imshow(rgb, interpolation="bilinear", extent=(-1, 1, -1, 1))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(77)

    data = []
    for sq in SEQUENCES:
        img, gt = sequence(**sq)
        preds = [predict_flow(gt, rng, **kw) for _, kw in METHODS]
        data.append((img, gt, preds))

    fmax = max(float(np.percentile(np.hypot(f[..., 0], f[..., 1]), 99.0))
               for _, gt, ps in data for f in [gt] + ps)
    errs = [[np.hypot(*(p - gt).transpose(2, 0, 1)) for p in ps]
            for _, gt, ps in data]
    emax = float(np.percentile(np.concatenate([np.ravel(e) for row in errs
                                               for e in row]), 99.0))

    col_names = ["Frame + GT flow"] + [m for m, _ in METHODS]
    ncol = len(col_names)

    fig = plt.figure(figsize=(7.1, 5.30))
    gs = fig.add_gridspec(3, ncol, wspace=0.032, hspace=0.032,
                          left=0.062, right=0.995, top=0.918, bottom=0.140)

    axes = np.empty((3, ncol), object)
    for i, (img, gt, preds) in enumerate(data):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            if j == 0:
                ax.imshow(img, interpolation="bilinear")
                axi = ax.inset_axes(INSET)
                axi.imshow(flow_rgb(gt, fmax), interpolation="bilinear",
                           aspect="auto")
                tag, tagcol = "GT flow", "white"
            else:
                p = preds[j - 1]
                ax.imshow(flow_rgb(p, fmax), interpolation="bilinear")
                axi = ax.inset_axes(INSET)
                mp = axi.imshow(errs[i][j - 1], cmap=ERR_CMAP, vmin=0,
                                vmax=emax, interpolation="bilinear",
                                aspect="auto")
                tag, tagcol = "|EPE|", "white"
                mae = float(errs[i][j - 1].mean())
                ax.text(0.032, 0.962, f"EPE {mae:.2f} px",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.2, color="white",
                        fontweight="bold" if j == ncol - 1 else "normal",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.20",
                                  facecolor="#000000aa", edgecolor="none"))
            axi.set_xticks([])
            axi.set_yticks([])
            for s in axi.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.4)
                s.set_color(INSET_EDGE)
            axi.set_zorder(7)
            axi.text(0.05, 0.94, tag, transform=axi.transAxes, ha="left",
                     va="top", fontsize=6.0, color=tagcol, zorder=9,
                     bbox=dict(boxstyle="square,pad=0.16",
                               facecolor="#00000099", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(f"Sequence {'ABC'[i]}", fontsize=7.8,
                              labelpad=3.0)
            if i == 2:
                bold = col_names[j] == "Ours"
                ax.set_xlabel(col_names[j], fontsize=8.0, labelpad=3.2,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    k = col_names.index("Ours")
    p0, p1 = axes[0, k].get_position(), axes[2, k].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.005, p1.y0 - 0.040), p0.width + 0.010,
                             p0.y1 - p1.y0 + 0.048, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    # ---- shared error bar plus a small flow colour wheel, both subordinate
    a = axes[2, 1].get_position()
    b = axes[2, ncol - 1].get_position()
    cax = fig.add_axes([a.x0, 0.055, b.x1 - a.x0, 0.024])
    cb = fig.colorbar(mp, cax=cax, orientation="horizontal")
    cb.set_label("end-point error  (px)   -   shared by every inset",
                 fontsize=6.8, labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")

    c = axes[2, 0].get_position()
    wh = 0.062
    axw = fig.add_axes([c.x0 + 0.004, 0.018, wh, wh * 7.1 / 5.3])
    color_wheel(axw)
    axw.text(1.18, 0.5, "flow colour code:\nhue = direction,\nvalue = speed",
             transform=axw.transAxes, ha="left", va="center", fontsize=6.0,
             color="#666666", linespacing=1.35)

    savefig(fig, "262_cv_error_map_grid.png")


if __name__ == "__main__":
    main()
