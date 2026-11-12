"""Figure 300 - Content x style transfer matrix.

The classic arrangement: an empty top-left corner, five style references
along the first row (orange frames), four content images down the first
column (blue frames) and the 4 x 5 block of transfers in between.  Each
style has its own transfer operator - directional impasto strokes, flat
posterised ukiyo-e with ink outlines, a grouted mosaic, a sumi-e wash and a
neon gradient remap - so the columns stay visually distinct while the rows
keep their identity.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig

C_REAL = "#2e7d5b"
C_OURS = "#3b5b92"
C_BAD = "#b23a48"
C_NOISE = "#8e6bbf"
C_STYLE = "#d98032"
C_INK = "#1a1a1a"

# ---------------------------------------------------------------------------
# Procedural "generated portrait" engine
# ---------------------------------------------------------------------------
BASE = dict(
    cx=0.500, cy=0.500, rx=0.236, ry=0.302, tilt=0.00,
    skin_h=0.068, skin_s=0.360, skin_v=0.880,
    hair_h=0.070, hair_s=0.600, hair_v=0.300,
    bg_h=0.560, bg_s=0.260, bg_v=0.780,
    hair_line=-0.42, fringe=0.62,
    eye_x=0.400, eye_y=-0.170, eye_r=0.150,
    iris_h=0.090, iris_s=0.520, iris_v=0.400,
    brow_y=-0.360, brow_c=0.090,
    mouth_y=0.470, mouth_w=0.430, mouth_c=0.070,
    light=0.34, cloth_h=0.055,
)


def _sm(d, w=0.02):
    return np.clip(0.5 - d / max(float(w), 1e-6), 0.0, 1.0)


def _fbm(rng, n, scales=(1.2, 3.0, 8.0), weights=(0.45, 0.33, 0.22)):
    out = np.zeros((n, n))
    for s, w in zip(scales, weights):
        a = gaussian_filter(rng.standard_normal((n, n)), s, mode="wrap")
        out += w * a / (a.std() + 1e-9)
    return out / (out.std() + 1e-9)


def _over(img, a, col):
    col = np.asarray(col, float)
    if col.ndim == 1:
        col = col[None, None, :]
    return img * (1.0 - a[..., None]) + col * a[..., None]


def render_face(p, n=144, seed=0, tex=1.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    px = 1.0 / n

    bg_a = hsv_to_rgb([p["bg_h"] % 1.0, p["bg_s"] * 0.50,
                       min(p["bg_v"] + 0.15, 1.0)])
    bg_b = hsv_to_rgb([(p["bg_h"] + 0.07) % 1.0, p["bg_s"], p["bg_v"] * 0.66])
    t = yy[..., None]
    img = bg_a * (1 - t) + bg_b * t
    img = img * (1 + 0.055 * _fbm(rng, n, (5, 14), (0.5, 0.5)))[..., None]

    ca, sa = np.cos(p["tilt"]), np.sin(p["tilt"])
    dx, dy = xx - p["cx"], yy - p["cy"]
    ux = dx * ca + dy * sa
    uy = -dx * sa + dy * ca
    rx, ry = p["rx"], p["ry"]

    skin = hsv_to_rgb([p["skin_h"] % 1.0, p["skin_s"], p["skin_v"]])
    hair = hsv_to_rgb([p["hair_h"] % 1.0, p["hair_s"], p["hair_v"]])

    cloth = hsv_to_rgb([(p["bg_h"] + 0.45) % 1.0, 0.42, 0.50 + p["cloth_h"]])
    dsh = (np.sqrt(((xx - p["cx"]) / (rx * 2.15)) ** 2
                   + ((yy - (p["cy"] + ry * 1.95)) / (ry * 1.05)) ** 2) - 1.0)
    img = _over(img, _sm(dsh * rx * 2.15, 3 * px), cloth)

    dnk = (np.sqrt((ux / (rx * 0.44)) ** 2
                   + ((uy - ry * 0.92) / (ry * 0.50)) ** 2) - 1.0)
    img = _over(img, _sm(dnk * rx * 0.44, 3 * px), skin * 0.78)

    rr = rx * (1.0 - 0.20 * np.clip(uy / ry, 0, 1) ** 2)
    dfc = np.sqrt((ux / rr) ** 2 + (uy / ry) ** 2) - 1.0
    face = _sm(dfc * rx, 2.6 * px)
    shade = (1.0 + p["light"] * (-0.42 * ux / rx - 0.30 * uy / ry)
             - 0.34 * np.clip(np.hypot(ux / rx, uy / ry) - 0.70, 0, None))
    img = _over(img, face, skin[None, None, :]
                * np.clip(shade, 0.52, 1.26)[..., None])

    hrx, hry = rx * 1.11, ry * 1.09
    dh = np.sqrt((ux / hrx) ** 2 + ((uy + ry * 0.10) / hry) ** 2) - 1.0
    cap = _sm(dh * hrx, 2.6 * px)
    top = _sm(uy - p["hair_line"] * ry, 0.022)
    side = 1.0 - _sm(np.abs(ux) - rx * p["fringe"], 0.030)
    lower = _sm(uy - ry * 0.34, 0.06)
    hmask = cap * np.clip(top + side * lower, 0, 1)
    strand = 0.5 + 0.5 * np.sin(52 * (0.32 * ux + uy)
                                + 2.4 * _fbm(rng, n, (2, 6), (0.6, 0.4)))
    img = _over(img, hmask,
                hair[None, None, :] * (0.76 + 0.44 * strand)[..., None])

    dn = (np.sqrt((ux / (rx * 0.135)) ** 2
                  + ((uy - ry * 0.13) / (ry * 0.21)) ** 2) - 1.0)
    img = _over(img, _sm(dn * rx * 0.135, 0.030) * 0.24, skin * 0.68)

    ex, ey = p["eye_x"] * rx, p["eye_y"] * ry
    er = p["eye_r"] * rx
    iris = hsv_to_rgb([p["iris_h"] % 1.0, p["iris_s"], p["iris_v"]])
    for s in (-1.0, 1.0):
        axl = ux - s * ex
        da = (np.sqrt((axl / (er * 1.60)) ** 2
                      + ((uy - ey) / (er * 0.80)) ** 2) - 1.0)
        alm = _sm(da * er * 1.60, 2.0 * px)
        img = _over(img, alm * 0.96, (0.94, 0.92, 0.90))
        di = np.hypot(axl - s * er * 0.04, uy - ey) / (er * 0.74) - 1.0
        img = _over(img, _sm(di * er * 0.74, 2.0 * px) * alm, iris)
        dp = np.hypot(axl - s * er * 0.04, uy - ey) / (er * 0.33) - 1.0
        img = _over(img, _sm(dp * er * 0.33, 2.0 * px) * alm, (0.07, 0.06, 0.08))
        dl = np.hypot(axl - s * er * 0.30,
                      uy - ey + er * 0.26) / (er * 0.18) - 1.0
        img = _over(img, _sm(dl * er * 0.18, 2.0 * px) * alm * 0.92, (1, 1, 1))
        dls = np.abs(uy - ey + er * 0.74) - er * 0.12
        lash = _sm(dls, 2.2 * px) * _sm(np.abs(axl) - er * 1.55, 0.020)
        img = _over(img, lash * 0.85, hair * 0.55)

    for s in (-1.0, 1.0):
        bxn = (ux - s * ex) / (er * 1.95)
        tgt = p["brow_y"] * ry - p["brow_c"] * ry * (1 - np.clip(bxn, -1, 1) ** 2)
        db = np.abs(uy - tgt) - er * 0.17
        img = _over(img, _sm(db, 2.2 * px) * _sm(np.abs(bxn) - 1.0, 0.26) * 0.94,
                    hair * 0.82)

    mw, my = p["mouth_w"] * rx, p["mouth_y"] * ry
    mxn = ux / mw
    tgt = my - p["mouth_c"] * ry * (1 - np.clip(mxn, -1, 1) ** 2)
    lip = hsv_to_rgb([(p["skin_h"] - 0.028) % 1.0,
                      min(p["skin_s"] + 0.32, 0.95), p["skin_v"] * 0.78])
    dm = np.abs(uy - tgt) - ry * 0.032
    img = _over(img, _sm(dm, 2.5 * px) * _sm(np.abs(mxn) - 1.0, 0.18), lip)
    dml = np.abs(uy - tgt) - ry * 0.006
    img = _over(img, _sm(dml, 2.0 * px) * _sm(np.abs(mxn) - 0.9, 0.16) * 0.6,
                lip * 0.45)

    img = img * (1 + 0.040 * tex * _fbm(rng, n, (0.9, 2.6), (0.6, 0.4)))[..., None]
    vg = 1 - 0.13 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.5
    img = img * vg[..., None]
    img = img + 0.012 * tex * rng.standard_normal((n, n, 1))
    return np.clip(img, 0.0, 1.0)


def sample_params(rng, jitter=1.0):
    p = dict(BASE)
    p["cx"] += 0.018 * jitter * rng.standard_normal()
    p["cy"] += 0.016 * jitter * rng.standard_normal()
    p["rx"] *= 1 + 0.09 * jitter * rng.standard_normal()
    p["ry"] *= 1 + 0.08 * jitter * rng.standard_normal()
    p["tilt"] += 0.10 * jitter * rng.standard_normal()
    p["skin_h"] = 0.042 + 0.038 * rng.random()
    p["skin_s"] = 0.26 + 0.30 * rng.random()
    p["skin_v"] = 0.62 + 0.32 * rng.random()
    p["hair_h"] = float(rng.choice([0.055, 0.080, 0.105, 0.020, 0.640]))
    p["hair_s"] = 0.22 + 0.58 * rng.random()
    p["hair_v"] = 0.14 + 0.48 * rng.random()
    p["bg_h"] = rng.random()
    p["bg_s"] = 0.16 + 0.34 * rng.random()
    p["bg_v"] = 0.54 + 0.36 * rng.random()
    p["hair_line"] = -0.66 + 0.20 * rng.random()
    p["fringe"] = 0.70 + 0.20 * rng.random()
    p["eye_x"] = 0.370 + 0.060 * rng.random()
    p["eye_y"] = -0.200 + 0.055 * rng.random()
    p["eye_r"] = 0.130 + 0.038 * rng.random()
    p["iris_h"] = float(rng.choice([0.09, 0.30, 0.56, 0.075]))
    p["iris_s"] = 0.35 + 0.35 * rng.random()
    p["iris_v"] = 0.28 + 0.34 * rng.random()
    p["mouth_y"] = 0.430 + 0.075 * rng.random()
    p["mouth_w"] = 0.380 + 0.110 * rng.random()
    p["mouth_c"] = 0.020 + 0.110 * rng.random()
    p["brow_y"] = -0.395 + 0.060 * rng.random()
    p["cloth_h"] = 0.30 * rng.random()
    return p


# ---------------------------------------------------------------------------
# Style operators
# ---------------------------------------------------------------------------
def _lum(img):
    return img @ np.array([0.299, 0.587, 0.114])


def _saturate(img, k):
    L = _lum(img)[..., None]
    return np.clip(L + k * (img - L), 0, 1)


def _edges(img, sigma=1.3, pct=97.0):
    L = gaussian_filter(_lum(img), sigma)
    gy, gx = np.gradient(L)
    m = np.hypot(gx, gy)
    return np.clip(m / (np.percentile(m, pct) + 1e-9), 0, 1)


def st_impasto(img, rng):
    n = img.shape[0]
    ang = gaussian_filter(rng.standard_normal((n, n)), 14.0)
    ang = 1.4 * ang / (ang.std() + 1e-9) + 0.7
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    amp = n * 0.022
    co = np.array([np.clip(yy + amp * np.sin(ang), 0, n - 1),
                   np.clip(xx + amp * np.cos(ang), 0, n - 1)])
    out = np.dstack([map_coordinates(img[..., k], co, order=1, mode="nearest")
                     for k in range(3)])
    ph = np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / (n / 26.0))
    out = out * (1 + 0.22 * ph[..., None])
    out = out * np.array([1.10, 1.00, 0.86]) + 0.015
    return np.clip(_saturate(out, 1.40), 0, 1)


def st_ukiyo(img, rng):
    q = 4
    base = np.dstack([gaussian_filter(img[..., k], 1.4) for k in range(3)])
    out = np.round(base * (q - 1)) / (q - 1)
    out = _saturate(out, 1.22)
    out = 0.88 * out + 0.12 * np.array([0.97, 0.91, 0.76])
    e = _edges(img, 2.2, 98.5)
    e = gaussian_filter((e > 0.40).astype(float), 0.6)
    e = np.clip(e * 1.7, 0, 1)
    return np.clip(out * (1 - 0.88 * e[..., None])
                   + 0.88 * e[..., None] * np.array([0.11, 0.09, 0.13]), 0, 1)


def st_mosaic(img, rng):
    n = img.shape[0]
    b = max(5, n // 20)
    h = (n // b) * b
    blk = img[:h, :h].reshape(h // b, b, h // b, b, 3).mean(axis=(1, 3))
    blk = np.clip(blk + rng.uniform(-0.05, 0.05, blk.shape), 0, 1)
    up = np.repeat(np.repeat(blk, b, axis=0), b, axis=1)
    out = img.copy()
    out[:h, :h] = up
    g = (np.arange(n) % b < 1).astype(float)
    grout = np.maximum(g[:, None], g[None, :])
    out = out * (1 - 0.30 * grout[..., None]) \
        + 0.30 * grout[..., None] * np.array([0.14, 0.13, 0.16])
    out = 0.5 + (out - 0.5) * 1.16
    return np.clip(_saturate(out, 1.34), 0, 1)


def st_ink(img, rng):
    n = img.shape[0]
    L = _lum(img)
    lo, hi = np.percentile(L, 4), np.percentile(L, 96)
    L = np.clip((L - lo) / (hi - lo + 1e-9), 0, 1) ** 0.80
    L = gaussian_filter(L, 1.0)
    e = _edges(img, 2.0, 96.0)
    ink = np.clip(1.14 * L - 0.95 * e, 0, 1)
    paper = np.array([0.955, 0.932, 0.868])
    out = ink[..., None] * paper + (1 - ink[..., None]) \
        * np.array([0.13, 0.12, 0.16])
    out = out * (1 + 0.035 * _fbm(rng, n, (2.5, 7.0), (0.6, 0.4)))[..., None]
    return np.clip(out, 0, 1)


NEON = LinearSegmentedColormap.from_list(
    "neon", [(0.02, 0.02, 0.11), (0.26, 0.05, 0.44), (0.86, 0.12, 0.54),
             (0.22, 0.84, 0.90), (0.93, 0.99, 1.00)])


def st_neon(img, rng):
    L = _lum(img)
    L = np.clip((L - np.percentile(L, 2)) /
                (np.percentile(L, 98) - np.percentile(L, 2) + 1e-9), 0, 1)
    out = NEON(L)[..., :3]
    glow = gaussian_filter(_edges(img, 1.2, 96.0), 1.7)
    return np.clip(out + 0.55 * glow[..., None]
                   * np.array([0.35, 0.95, 1.00]), 0, 1)


STYLES = [("Impasto", st_impasto), ("Ukiyo-e", st_ukiyo),
          ("Mosaic", st_mosaic), ("Sumi-e", st_ink), ("Neon", st_neon)]


def art_canvas(rng, n):
    """Abstract painting used as the style reference swatch."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    h0 = rng.random()
    img = (hsv_to_rgb([h0, 0.30, 0.90]) * (1 - yy[..., None])
           + hsv_to_rgb([(h0 + 0.18) % 1, 0.52, 0.55]) * yy[..., None])
    for _ in range(3):
        cx, cy = rng.uniform(0.15, 0.85, 2)
        rx_, ry_ = rng.uniform(0.26, 0.52, 2)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        m = _sm(np.sqrt((u / rx_) ** 2 + (v / ry_) ** 2) - 1.0, 0.22)
        col = hsv_to_rgb([rng.random(), rng.uniform(0.40, 0.85),
                          rng.uniform(0.50, 0.95)])
        img = _over(img, m * rng.uniform(0.50, 0.85), col)
    img = img * (1 + 0.06 * _fbm(rng, n, (9, 22), (0.55, 0.45)))[..., None]
    return np.clip(img, 0, 1)


def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    N = 170

    contents = [render_face(sample_params(np.random.default_rng(1800 + 6 * r)),
                            n=N, seed=1800 + 6 * r) for r in range(4)]
    refs = [fn(art_canvas(np.random.default_rng(300 + 9 * c), N),
               np.random.default_rng(311 + 9 * c))
            for c, (_nm, fn) in enumerate(STYLES)]

    FW, FH = 7.4, 6.20
    fig = plt.figure(figsize=(FW, FH))
    wgap, hgap, sep = 0.009, 0.009, 0.022
    th = 0.1462
    tw = th * FH / FW
    grid_w = 6 * tw + sep + 4 * wgap
    left = (1.0 - grid_w) / 2.0
    xs = [left] + [left + tw + sep + c * (tw + wgap) for c in range(5)]

    y_top = 0.892
    ys = [y_top] + [y_top - th - sep - r * (th + hgap) for r in range(4)]
    y_bot = ys[-1] - th

    # ---- corner cell -------------------------------------------------------
    axc = fig.add_axes([xs[0], ys[0] - th, tw, th])
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.plot([0.03, 0.97], [0.97, 0.03], color="#d6d6dc", lw=0.8)
    axc.text(0.94, 0.86, "style", ha="right", va="center", fontsize=7.6,
             fontweight="bold", color=C_STYLE)
    axc.annotate("", xy=(1.06, 0.86), xytext=(0.96, 0.86),
                 xycoords="axes fraction", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color=C_STYLE, lw=1.1))
    axc.text(0.10, 0.18, "content", ha="center", va="center", fontsize=7.6,
             fontweight="bold", color=C_OURS, rotation=90)
    axc.annotate("", xy=(0.10, -0.09), xytext=(0.10, 0.02),
                 xycoords="axes fraction", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.1))

    # ---- style reference row ----------------------------------------------
    for c, (nm, _fn) in enumerate(STYLES):
        ax = fig.add_axes([xs[c + 1], ys[0] - th, tw, th])
        image_axes(ax, color=C_STYLE, lw=1.6)
        ax.imshow(refs[c], vmin=0.0, vmax=1.0, interpolation="nearest")
        fig.text(xs[c + 1] + tw / 2, ys[0] + 0.011, nm, ha="center",
                 va="bottom", fontsize=7.8, fontweight="bold", color=C_STYLE)

    # ---- content column + transfers ---------------------------------------
    for r in range(4):
        top = ys[r + 1]
        ax = fig.add_axes([xs[0], top - th, tw, th])
        image_axes(ax, color=C_OURS, lw=1.6)
        ax.imshow(contents[r], vmin=0.0, vmax=1.0, interpolation="nearest")
        fig.text(xs[0] - 0.010, top - th / 2, f"content {r + 1}",
                 rotation=90, ha="right", va="center", fontsize=6.8,
                 color=C_OURS)

        for c, (nm, fn) in enumerate(STYLES):
            axs_ = fig.add_axes([xs[c + 1], top - th, tw, th])
            image_axes(axs_, color="#d0d0d6", lw=0.45)
            out = fn(contents[r], np.random.default_rng(41 + 13 * r + c))
            axs_.imshow(out, vmin=0.0, vmax=1.0, interpolation="nearest")

    fig.text(left, y_bot - 0.022,
             "Content identity, pose and layout survive every column; the "
             "style column controls palette, stroke geometry and edge "
             "treatment only.",
             fontsize=6.4, color="#777777", ha="left", va="top")

    savefig(fig, "300_style_transfer_matrix.png")


if __name__ == "__main__":
    main()
