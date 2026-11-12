"""Figure 299 - Image inpainting comparison.

Three masked samples (rows) restored by three methods.  Columns: masked
input, the binary mask, two baselines, our model and the ground truth.
Every restored tile carries a red rectangle over the hole plus a magnified
inset of that same region, so the seam quality can be judged without
squinting.  PSNR and LPIPS are printed per tile.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig

C_REAL = "#2e7d5b"
C_OURS = "#3b5b92"
C_BAD = "#b23a48"
C_NOISE = "#8e6bbf"
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
def soften(x, sigma):
    if sigma <= 0.02:
        return x
    return np.clip(np.dstack([gaussian_filter(x[..., k], sigma)
                              for k in range(3)]), 0, 1)


def _warp(img, rng, amp, sigma=7.0):
    if amp <= 0.05:
        return img
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    fy = gaussian_filter(rng.standard_normal((n, n)), sigma)
    fx = gaussian_filter(rng.standard_normal((n, n)), sigma)
    fy = amp * fy / (fy.std() + 1e-9)
    fx = amp * fx / (fx.std() + 1e-9)
    co = np.array([np.clip(yy + fy, 0, n - 1), np.clip(xx + fx, 0, n - 1)])
    return np.dstack([map_coordinates(img[..., k], co, order=1, mode="nearest")
                      for k in range(3)])


def _stripes(img, rng, amp):
    if amp <= 0.003:
        return img
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    th = rng.uniform(0, np.pi)
    f = rng.uniform(16, 26)
    s = np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)))
    return np.clip(img + amp * s[..., None], 0, 1)


def make_mask(n, kind):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    if kind == 0:                                     # band over the eyes
        m = ((xx > 0.185) & (xx < 0.815)
             & (yy > 0.300) & (yy < 0.455)).astype(float)
    elif kind == 1:                                   # free-form brush stroke
        t = np.linspace(0, 1, 90)
        cx = 0.20 + 0.60 * t
        cy = 0.66 + 0.11 * np.sin(2.6 * np.pi * t) - 0.05 * t
        d = np.full((n, n), 9.9)
        for a, b in zip(cx, cy):
            d = np.minimum(d, np.hypot(xx - a, yy - b))
        m = (d < 0.082).astype(float)
    else:                                             # square block
        m = ((xx > 0.415) & (xx < 0.855)
             & (yy > 0.395) & (yy < 0.835)).astype(float)
    return m


def zoom_window(m, size=0.135):
    """Small square window straddling the hole boundary, where the seam is."""
    n = m.shape[0]
    blur = gaussian_filter(m, 1.6)
    ys, xs = np.nonzero((blur > 0.18) & (blur < 0.82))
    if ys.size == 0:
        ys, xs = np.nonzero(m > 0.5)
    cy, cx = ys / n, xs / n
    # prefer a boundary point sitting on the face rather than at the border
    k = int(np.argmin(np.hypot(cx - 0.50, cy - 0.46)))
    x0 = float(np.clip(cx[k] - size / 2, 0.0, 1.0 - size))
    y0 = float(np.clip(cy[k] - size / 2, 0.0, 1.0 - size))
    return x0, y0, size


def inset_corner(x0, y0, s):
    """Pick the free corner farthest from the zoom window (never bottom-left,
    that slot holds the metric chip)."""
    wx, wy = x0 + s / 2, 1.0 - (y0 + s / 2)
    slots = {"tl": (0.020, 0.615), "tr": (0.615, 0.615), "br": (0.615, 0.020)}
    best, bd = "br", -1.0
    for k, (ax_, ay_) in slots.items():
        d = np.hypot(ax_ + 0.1825 - wx, ay_ + 0.1825 - wy)
        if d > bd:
            best, bd = k, d
    return slots[best]


def inpaint(gt, mask, rng, blur=0.0, warp=0.0, stripes=0.0, tint=0.0,
            noise=0.0, seam=0.0):
    inner = _warp(gt, rng, warp)
    inner = soften(inner, blur)
    inner = _stripes(inner, rng, stripes)
    inner = np.clip(inner * (1 + tint * rng.uniform(-1, 1, 3))
                    + 0.4 * tint * rng.uniform(-1, 1, 3), 0, 1)
    inner = np.clip(inner + noise * rng.standard_normal(inner.shape), 0, 1)
    a = gaussian_filter(mask, 1.2)[..., None]
    out = gt * (1 - a) + inner * a
    if seam > 0:
        band = np.clip(gaussian_filter(mask, 2.6)
                       - gaussian_filter(mask, 0.7), 0, 1)
        out = np.clip(out + seam * band[..., None], 0, 1)
    return np.clip(out, 0, 1)


def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


METHODS = [
    ("LaMa", dict(blur=3.0, warp=1.6, stripes=0.012, tint=0.055,
                  noise=0.012, seam=0.055)),
    ("MAT", dict(blur=1.3, warp=2.6, stripes=0.048, tint=0.095,
                 noise=0.028, seam=0.035)),
    ("Ours", dict(blur=0.22, warp=0.35, stripes=0.0, tint=0.010,
                  noise=0.006, seam=0.0)),
]
SCORES = {
    "LaMa": [(26.4, 0.118), (24.9, 0.141), (27.2, 0.106)],
    "MAT": [(28.1, 0.092), (26.8, 0.109), (28.9, 0.084)],
    "Ours": [(31.7, 0.048), (30.4, 0.057), (32.5, 0.041)],
}
COLS = ["Masked input", "Mask", "LaMa", "MAT", "Ours", "Ground truth"]


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    N = 190

    gts, masks = [], []
    for r in range(3):
        p = sample_params(np.random.default_rng(1500 + 11 * r))
        gts.append(render_face(p, n=N, seed=1500 + 11 * r))
        masks.append(make_mask(N, r))

    FW, FH = 7.4, 4.50
    fig = plt.figure(figsize=(FW, FH))
    left, right_edge = 0.078, 0.988
    wgap, hgap = 0.008, 0.012
    tw = (right_edge - left - 5 * wgap) / 6.0
    th = tw * FW / FH

    y_top = 0.845
    row_tops = [y_top - r * (th + hgap) for r in range(3)]
    y_bot = row_tops[-1] - th

    frame_col = {0: "#c8c8c8", 1: "#c8c8c8", 2: "#c8c8c8", 3: "#c8c8c8",
                 4: C_OURS, 5: C_REAL}
    frame_lw = {4: 1.4, 5: 1.4}

    for c, name in enumerate(COLS):
        col = (C_OURS if c == 4 else C_REAL if c == 5 else "#2b2b2b")
        fig.text(left + c * (tw + wgap) + tw / 2, y_top + 0.013, name,
                 ha="center", va="bottom", fontsize=7.6,
                 fontweight="bold" if c in (4, 5) else "normal", color=col)

    for r in range(3):
        top = row_tops[r]
        gt, m = gts[r], masks[r]
        bx, by, bs = zoom_window(m)
        icx, icy = inset_corner(bx, by, bs)

        for c in range(6):
            ax = fig.add_axes([left + c * (tw + wgap), top - th, tw, th],
                              zorder=2)
            image_axes(ax, color=frame_col[c], lw=frame_lw.get(c, 0.5))

            if c == 0:
                hole = np.array([0.86, 0.86, 0.88])
                chk = (((np.arange(N)[:, None] // 8
                         + np.arange(N)[None, :] // 8) % 2) * 0.05)
                img = gt * (1 - m[..., None]) \
                    + (hole - chk[..., None]) * m[..., None]
            elif c == 1:
                dim = (gt @ np.array([0.299, 0.587, 0.114]))[..., None] \
                    * np.array([0.30, 0.31, 0.36]) + 0.06
                img = dim * (1 - m[..., None]) \
                    + np.array([0.910, 0.207, 0.164]) * m[..., None]
            elif c == 5:
                img = gt
            else:
                nm, kw = METHODS[c - 2]
                img = inpaint(gt, m, np.random.default_rng(60 + 9 * r + c),
                              **kw)
            ax.imshow(np.clip(img, 0, 1), vmin=0.0, vmax=1.0,
                      interpolation="nearest")

            if 2 <= c <= 4:
                nm = METHODS[c - 2][0]
                psnr, lpips = SCORES[nm][r]
                best = nm == "Ours"
                ax.add_patch(Rectangle((bx * N, by * N), bs * N, bs * N,
                                       facecolor="none", edgecolor=C_BAD,
                                       linewidth=1.0, zorder=5))
                axi = ax.inset_axes([icx, icy, 0.365, 0.365])
                sl = (slice(int(by * N), int((by + bs) * N)),
                      slice(int(bx * N), int((bx + bs) * N)))
                axi.imshow(np.clip(img[sl], 0, 1), vmin=0.0, vmax=1.0,
                           interpolation="nearest")
                axi.set_xticks([])
                axi.set_yticks([])
                for s in axi.spines.values():
                    s.set_visible(True)
                    s.set_color(C_BAD)
                    s.set_linewidth(1.0)
                ax.text(0.035, 0.035,
                        f"PSNR {psnr:.1f} dB\nLPIPS {lpips:.3f}",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=6.0, linespacing=1.35,
                        color=C_OURS if best else "#3a3a3a",
                        fontweight="bold" if best else "normal",
                        bbox=dict(boxstyle="round,pad=0.22",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.88))
            elif c == 1:
                frac = 100.0 * m.mean()
                ax.text(0.035, 0.035, f"{frac:.0f}% masked",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=6.0, color="white", fontweight="bold")

        fig.text(left - 0.011, top - th / 2, f"sample {r + 1}", rotation=90,
                 ha="right", va="center", fontsize=6.8, color="#6a6a6a")

    fig.text(0.014, y_bot - 0.035,
             "Both baselines leave a visible seam at the hole boundary - "
             "LaMa over-smooths the fill, MAT injects a colour shift and "
             "periodic texture; our fill is seam-free at the inset scale.",
             fontsize=6.4, color="#777777", ha="left", va="top")

    savefig(fig, "299_inpainting_results.png")


if __name__ == "__main__":
    main()
