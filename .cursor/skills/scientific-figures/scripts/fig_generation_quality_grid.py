"""Figure 298 - Sample-quality comparison across generative model families.

Five families (rows) decode the same six latents (columns), so the rows can
be read side by side.  Each family is given its own characteristic failure
mode rather than a single "quality" slider: the GAN is sharp but warped and
checkered, the VAE is heavily blurred and washed out, the flow drifts in
colour, the diffusion model is nearly clean and our model is clean.  The
right-hand block reports FID / IS / Precision / Recall as value chips with
normalised bars.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
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
# Family-specific corruption
# ---------------------------------------------------------------------------
def soften(x, sigma):
    if sigma <= 0.02:
        return x
    return np.clip(np.dstack([gaussian_filter(x[..., k], sigma)
                              for k in range(3)]), 0, 1)


def _warp(img, rng, amp, sigma=9.0):
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


def _artifacts(img, rng, amp):
    if amp <= 0.004:
        return img
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    th = rng.uniform(0, np.pi)
    f = rng.uniform(13, 24)
    stripe = np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)))
    blk = np.sign(np.sin(2 * np.pi * 7 * xx) * np.sin(2 * np.pi * 7 * yy))
    m = gaussian_filter(rng.standard_normal((n, n)), 10.0)
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    return np.clip(img + amp * ((0.6 * stripe + 0.4 * blk)
                                * (0.30 + 0.70 * m))[..., None], 0, 1)


def corrupt(img, rng, blur=0.0, warp=0.0, arti=0.0, color=0.0, noise=0.0,
            washout=0.0):
    out = _warp(img, rng, warp)
    out = soften(out, blur)
    out = _artifacts(out, rng, arti)
    out = out * (1.0 + color * rng.uniform(-1, 1, 3)) \
        + 0.3 * color * rng.uniform(-1, 1, 3)
    out = 0.5 + (out - 0.5) * (1.0 - washout)
    out = out + noise * rng.standard_normal(out.shape)
    return np.clip(out, 0.0, 1.0)


MODELS = [
    ("GAN", "StyleGAN2", dict(blur=0.45, warp=5.5, arti=0.105, color=0.10,
                              noise=0.030, washout=0.02),
     dict(fid=21.4, is_=5.83, p=0.624, r=0.358)),
    ("VAE", "NVAE", dict(blur=3.30, warp=1.4, arti=0.0, color=0.05,
                         noise=0.010, washout=0.24),
     dict(fid=48.7, is_=4.12, p=0.541, r=0.512)),
    ("Diffusion", "ADM", dict(blur=0.55, warp=0.9, arti=0.016, color=0.030,
                              noise=0.013, washout=0.03),
     dict(fid=9.62, is_=7.02, p=0.708, r=0.583)),
    ("Flow", "Glow", dict(blur=1.65, warp=2.4, arti=0.048, color=0.145,
                          noise=0.052, washout=0.13),
     dict(fid=32.1, is_=5.06, p=0.577, r=0.618)),
    ("Ours", "latent rectified flow",
     dict(blur=0.14, warp=0.3, arti=0.0, color=0.014, noise=0.008,
          washout=0.0),
     dict(fid=5.41, is_=7.64, p=0.781, r=0.664)),
]

METRICS = [("FID $\\downarrow$", "fid", (5.0, 55.0), True),
           ("IS $\\uparrow$", "is_", (3.8, 8.0), False),
           ("Prec $\\uparrow$", "p", (0.50, 0.82), False),
           ("Rec $\\uparrow$", "r", (0.30, 0.70), False)]


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
    N = 150
    NS = 6

    latents = [sample_params(np.random.default_rng(1200 + 7 * k))
               for k in range(NS)]
    clean = [render_face(p, n=N, seed=1200 + 7 * k)
             for k, p in enumerate(latents)]

    FW, FH = 7.4, 4.45
    fig = plt.figure(figsize=(FW, FH))
    left, grid_right = 0.152, 0.700
    wgap, hgap = 0.007, 0.010
    tw = (grid_right - left - (NS - 1) * wgap) / NS
    th = tw * FW / FH

    y_top = 0.858
    row_tops = [y_top - r * (th + hgap) for r in range(len(MODELS))]
    y_bot = row_tops[-1] - th

    mx0, mx1 = 0.722, 0.990
    mw = (mx1 - mx0) / 4.0

    # headers
    fig.text(left + (grid_right - left) / 2, y_top + 0.016,
             "six uncurated samples   (identical latents in every row)",
             ha="center", va="bottom", fontsize=7.6, fontweight="bold",
             color="#2b2b2b")
    fig.text((mx0 + mx1) / 2, y_top + 0.016,
             "quality metrics   (50k samples)", ha="center", va="bottom",
             fontsize=7.6, fontweight="bold", color="#2b2b2b")
    fig.add_artist(plt.Line2D([grid_right + 0.011] * 2,
                              [y_bot - 0.010, y_top + 0.010],
                              color="#c9c9d0", lw=0.8, ls=(0, (3, 3)),
                              transform=fig.transFigure, zorder=1))

    for r, (name, sub, kw, mets) in enumerate(MODELS):
        top = row_tops[r]
        is_ours = name == "Ours"
        if is_ours:
            fig.patches.append(FancyBboxPatch(
                (0.014, top - th - 0.009), 0.980, th + 0.018,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                transform=fig.transFigure, facecolor="#e8eff7",
                edgecolor="#c6d6e8", linewidth=0.6, zorder=0))

        for c in range(NS):
            ax = fig.add_axes([left + c * (tw + wgap), top - th, tw, th],
                              zorder=2)
            image_axes(ax, color=C_OURS if is_ours else "#cbcbcb",
                       lw=1.1 if is_ours else 0.45)
            ax.imshow(corrupt(clean[c],
                              np.random.default_rng(70 + 13 * r + c), **kw),
                      vmin=0.0, vmax=1.0, interpolation="nearest")

        fig.text(left - 0.014, top - th / 2 + 0.017, name, ha="right",
                 va="center", fontsize=8.6,
                 fontweight="bold" if is_ours else "normal",
                 color=C_OURS if is_ours else "#2b2b2b")
        fig.text(left - 0.014, top - th / 2 - 0.019, sub, ha="right",
                 va="center", fontsize=6.4, fontstyle="italic",
                 color=C_OURS if is_ours else "#909090")

        # ---- metric chips
        axm = fig.add_axes([mx0, top - th, mx1 - mx0, th], zorder=2)
        axm.set_xlim(0, 4)
        axm.set_ylim(0, 1)
        axm.axis("off")
        for j, (lab, key, (lo, hi), invert) in enumerate(METRICS):
            v = mets[key]
            nrm = (v - lo) / (hi - lo)
            nrm = 1.0 - nrm if invert else nrm
            nrm = float(np.clip(nrm, 0.04, 1.0))
            bc = C_OURS if is_ours else "#a8a8a8"
            axm.text(j + 0.5, 0.855, lab, ha="center", va="center",
                     fontsize=6.0, color="#8a8a8a")
            axm.text(j + 0.5, 0.545, f"{v:.2f}" if key != "fid"
                     else f"{v:.1f}", ha="center", va="center",
                     fontsize=8.0 if is_ours else 7.6,
                     fontweight="bold" if is_ours else "normal",
                     color=C_OURS if is_ours else "#2b2b2b")
            axm.add_patch(Rectangle((j + 0.13, 0.185), 0.74, 0.105,
                                    facecolor="#e4e4e8", edgecolor="none"))
            axm.add_patch(Rectangle((j + 0.13, 0.185), 0.74 * nrm, 0.105,
                                    facecolor=bc, edgecolor="none"))

    fig.text(0.014, y_bot - 0.040,
             "Each family keeps its own signature failure: adversarial "
             "training gives sharp but warped and checkered output, the VAE "
             "blurs and desaturates, the flow drifts in colour.",
             fontsize=6.4, color="#777777", ha="left", va="top")

    savefig(fig, "298_generation_quality_grid.png")


if __name__ == "__main__":
    main()
