"""Figure 297 - GAN training dynamics dashboard.

(a) generator and discriminator losses with a smoothed trend and a spread
band, (b) FID and Inception Score against iteration, (c) samples from a
fixed latent batch at five checkpoints, (d) mode-collapse diagnostics.  A
collapse episode around 44k-52k iterations is highlighted in red across all
four panels so the reader can follow one event through loss, metric, sample
and diagnostic space.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, add_panel_label, smooth_line

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
    f = rng.uniform(12, 22)
    stripe = np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)))
    blk = np.sign(np.sin(2 * np.pi * 6 * xx) * np.sin(2 * np.pi * 6 * yy))
    m = gaussian_filter(rng.standard_normal((n, n)), 10.0)
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    return np.clip(img + amp * ((0.62 * stripe + 0.38 * blk)
                                * (0.30 + 0.70 * m))[..., None], 0, 1)


def degrade(img, rng, q):
    d = float(np.clip(1.0 - q, 0.0, 1.0))
    out = _warp(img, rng, 9.0 * d ** 1.4)
    out = soften(out, 3.6 * d ** 1.35)
    out = _artifacts(out, rng, 0.150 * d ** 1.2)
    gains = 1.0 + 0.26 * d ** 1.2 * rng.uniform(-1, 1, 3)
    out = out * gains + 0.06 * d ** 1.2 * rng.uniform(-1, 1, 3)
    out = 0.5 + (out - 0.5) * (1.0 - 0.32 * d)
    out = out + 0.10 * d ** 1.3 * rng.standard_normal(out.shape)
    return np.clip(out, 0.0, 1.0)


def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def style_panel(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("#444444")
    ax.tick_params(labelsize=6.6, length=2.6, width=0.7)
    ax.grid(True, color="#ececf0", lw=0.55, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
COLLAPSE = (44.0, 52.0)      # thousands of iterations


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(297)

    # ---- synthetic training log -------------------------------------------
    it = np.linspace(0.3, 100.0, 620)          # thousands of iterations
    col = (it > COLLAPSE[0]) & (it < COLLAPSE[1])
    bump = np.exp(-0.5 * ((it - 48.0) / 2.6) ** 2)

    g_base = 1.35 + 1.9 * np.exp(-it / 9.0) + 0.16 * np.sin(it / 7.0)
    g_loss = g_base + 2.5 * bump + 0.20 * rng.standard_normal(it.size) \
        * (1 + 1.6 * bump)
    d_base = 0.92 - 0.30 * np.exp(-it / 14.0) - 0.0012 * it
    d_loss = d_base - 0.62 * bump + 0.13 * rng.standard_normal(it.size) \
        * (1 + 1.2 * bump)

    fid_it = np.arange(2.0, 101.0, 4.0)
    fid = 118 * np.exp(-fid_it / 17.0) + 8.4 \
        + 26 * np.exp(-0.5 * ((fid_it - 48.0) / 3.4) ** 2) \
        + 0.9 * np.random.default_rng(5).standard_normal(fid_it.size)
    iscore = 3.05 + 4.55 * (1 - np.exp(-fid_it / 21.0)) \
        - 1.75 * np.exp(-0.5 * ((fid_it - 48.0) / 3.4) ** 2) \
        + 0.06 * np.random.default_rng(6).standard_normal(fid_it.size)

    diag_it = np.arange(1.0, 101.0, 2.0)
    modes = 8 * (1 - np.exp(-diag_it / 12.0)) \
        - 5.0 * np.exp(-0.5 * ((diag_it - 48.0) / 3.2) ** 2)
    modes = np.clip(np.round(modes + 0.18 * np.random.default_rng(7)
                             .standard_normal(diag_it.size)), 0, 8)
    div = 0.62 * (1 - np.exp(-diag_it / 10.0)) \
        - 0.40 * np.exp(-0.5 * ((diag_it - 48.0) / 3.0) ** 2) \
        + 0.012 * np.random.default_rng(8).standard_normal(diag_it.size)
    div = np.clip(div, 0.02, 0.75)

    # ---- checkpoints -------------------------------------------------------
    ckpts = [1, 5, 20, 60, 100]
    qual = [0.03, 0.26, 0.60, 0.88, 0.99]
    ident = [render_face(sample_params(np.random.default_rng(920 + 4 * k)),
                         n=132, seed=920 + 4 * k) for k in range(4)]
    tiles = [[degrade(ident[r], np.random.default_rng(50 + 9 * c + r), qual[c])
              for c in range(5)] for r in range(4)]

    # ---- canvas ------------------------------------------------------------
    FW, FH = 7.4, 5.40
    fig = plt.figure(figsize=(FW, FH))

    axA = fig.add_axes([0.070, 0.645, 0.375, 0.260])
    axB = fig.add_axes([0.590, 0.645, 0.355, 0.260])

    # ---------------- (a) adversarial losses --------------------------------
    axA.axvspan(*COLLAPSE, color=C_BAD, alpha=0.085, lw=0, zorder=1)
    for y, c, lab in ((g_loss, C_OURS, "Generator  $\\mathcal{L}_G$"),
                      (d_loss, C_NOISE, "Discriminator  $\\mathcal{L}_D$")):
        axA.plot(it, y, color=c, lw=0.5, alpha=0.28, zorder=2)
        xs, ys = smooth_line(it, y, frac=0.028, n=400)
        _, sd = smooth_line(it, (y - np.interp(it, xs, ys)) ** 2,
                            frac=0.028, n=400)
        sd = np.sqrt(np.maximum(sd, 0))
        axA.fill_between(xs, ys - sd, ys + sd, color=c, alpha=0.18, lw=0,
                         zorder=3)
        axA.plot(xs, ys, color=c, lw=1.7, zorder=4, label=lab)
    axA.set_xlim(0, 100)
    axA.set_ylim(-0.3, 5.4)
    axA.set_xlabel("iteration  (x $10^3$)", fontsize=7.2, labelpad=1.8)
    axA.set_ylabel("loss", fontsize=7.2, labelpad=2.0)
    style_panel(axA)
    axA.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.4, labelspacing=0.32)
    axA.annotate("collapse episode", xy=(44.5, 4.35), xytext=(6, 4.85),
                 fontsize=6.4, color=C_BAD, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.8,
                                 shrinkA=2, shrinkB=2,
                                 connectionstyle="arc3,rad=-0.15"))
    add_panel_label(axA, "(a)", x=-0.135, y=1.03, size=9.5)
    axA.set_title("adversarial losses", fontsize=7.6, pad=3.5,
                  color="#3a3a3a", loc="left")

    # ---------------- (b) FID / IS ------------------------------------------
    axB.axvspan(*COLLAPSE, color=C_BAD, alpha=0.085, lw=0, zorder=1)
    axB.plot(fid_it, fid, color=C_OURS, lw=2.0, marker="o", ms=2.8,
             zorder=4, label="FID $\\downarrow$")
    axB.set_xlim(0, 100)
    axB.set_ylim(0, 135)
    axB.set_xlabel("iteration  (x $10^3$)", fontsize=7.2, labelpad=1.8)
    axB.set_ylabel("FID", fontsize=7.2, labelpad=2.0, color=C_OURS)
    axB.tick_params(axis="y", colors=C_OURS)
    style_panel(axB)

    axB2 = axB.twinx()
    axB2.plot(fid_it, iscore, color=C_REAL, lw=1.5, ls=(0, (4, 2)),
              marker="s", ms=2.5, zorder=4, label="IS $\\uparrow$")
    axB2.set_ylim(2.4, 8.4)
    axB2.set_ylabel("Inception Score", fontsize=7.2, labelpad=2.0,
                    color=C_REAL)
    axB2.tick_params(axis="y", colors=C_REAL, labelsize=6.6, length=2.6,
                     width=0.7)
    axB2.spines["top"].set_visible(False)
    axB2.spines["right"].set_color(C_REAL)
    axB2.spines["right"].set_linewidth(0.8)
    axB2.grid(False)

    kbest = int(np.argmin(fid))
    axB.scatter([fid_it[kbest]], [fid[kbest]], s=34, facecolor="white",
                edgecolor=C_OURS, linewidth=1.5, zorder=6)
    axB.annotate(f"best FID {fid[kbest]:.1f}",
                 xy=(fid_it[kbest], fid[kbest]), xytext=(-6, 22),
                 textcoords="offset points", fontsize=6.3, color=C_OURS,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.8,
                                 shrinkA=1, shrinkB=3))
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5,
               frameon=False, handlelength=1.4, labelspacing=0.32,
               bbox_to_anchor=(1.0, 0.86))
    add_panel_label(axB, "(b)", x=-0.135, y=1.03, size=9.5)
    axB.set_title("sample-quality metrics", fontsize=7.6, pad=3.5,
                  color="#3a3a3a", loc="left")

    # ---------------- (c) checkpoint samples --------------------------------
    gl, gt = 0.070, 0.470
    tw, wg, hg = 0.0665, 0.0072, 0.0100
    th = tw * FW / FH
    for r in range(4):
        for c in range(5):
            ax = fig.add_axes([gl + c * (tw + wg),
                               gt - (r + 1) * th - r * hg, tw, th], zorder=2)
            image_axes(ax, color="#c8c8c8", lw=0.45)
            ax.imshow(tiles[r][c], vmin=0.0, vmax=1.0, interpolation="nearest")
    gw = 5 * tw + 4 * wg
    gh = 4 * th + 3 * hg

    axbar = fig.add_axes([gl, gt + 0.014, gw, 0.017])
    cmapc = LinearSegmentedColormap.from_list("c", ["#e2e2e6", C_OURS])
    axbar.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cmapc,
                 extent=[0, 0.978, 0, 1], interpolation="bilinear")
    axbar.fill([0.978, 1.0, 0.978], [-0.55, 0.5, 1.55], color=C_OURS, lw=0,
               clip_on=False)
    axbar.set_xlim(0, 1)
    axbar.set_ylim(0, 1)
    axbar.set_xticks([])
    axbar.set_yticks([])
    for s in axbar.spines.values():
        s.set_visible(False)
    for c, k in enumerate(ckpts):
        fig.text(gl + c * (tw + wg) + tw / 2, gt + 0.040, f"{k}k",
                 ha="center", va="bottom", fontsize=6.9, fontweight="bold",
                 color="#2b2b2b")
    fig.text(gl, gt + 0.076, "(c)", ha="left", va="bottom", fontsize=9.5,
             fontweight="bold", color=C_INK)
    fig.text(gl + 0.036, gt + 0.077, "fixed latent batch decoded at five "
             "checkpoints", ha="left", va="bottom", fontsize=7.6,
             color="#3a3a3a")
    fig.text(gl - 0.010, gt - gh / 2, "4 fixed latents $z_{1..4}$",
             rotation=90, ha="right", va="center", fontsize=6.6,
             color="#8a8a8a")

    # ---------------- (d) mode-collapse diagnostics -------------------------
    axD = fig.add_axes([0.590, 0.118, 0.355, 0.278])
    axD.axvspan(*COLLAPSE, color=C_BAD, alpha=0.085, lw=0, zorder=1)
    axD.step(diag_it, modes, where="mid", color=C_REAL, lw=1.7, zorder=4,
             label="modes covered (of 8)")
    axD.set_xlim(0, 100)
    axD.set_ylim(-0.4, 9.2)
    axD.set_yticks([0, 2, 4, 6, 8])
    axD.set_xlabel("iteration  (x $10^3$)", fontsize=7.2, labelpad=1.8)
    axD.set_ylabel("modes covered", fontsize=7.2, labelpad=2.0, color=C_REAL)
    axD.tick_params(axis="y", colors=C_REAL)
    style_panel(axD)

    axD2 = axD.twinx()
    axD2.plot(diag_it, div, color=C_NOISE, lw=1.5, zorder=3,
              label="intra-batch LPIPS")
    axD2.axhline(0.30, color=C_BAD, lw=0.9, ls=(0, (3, 2)), zorder=2)
    axD2.text(99, 0.315, "collapse threshold", ha="right", va="bottom",
              fontsize=6.0, color=C_BAD)
    axD2.set_ylim(0, 0.80)
    axD2.set_ylabel("intra-batch LPIPS", fontsize=7.2, labelpad=2.0,
                    color=C_NOISE)
    axD2.tick_params(axis="y", colors=C_NOISE, labelsize=6.6, length=2.6,
                     width=0.7)
    axD2.spines["top"].set_visible(False)
    axD2.spines["right"].set_color(C_NOISE)
    axD2.spines["right"].set_linewidth(0.8)
    axD2.grid(False)

    h1, l1 = axD.get_legend_handles_labels()
    h2, l2 = axD2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=6.5,
               frameon=False, handlelength=1.4, labelspacing=0.32,
               bbox_to_anchor=(1.0, 0.02))
    add_panel_label(axD, "(d)", x=-0.135, y=1.05, size=9.5)
    axD.set_title("mode-collapse diagnostics", fontsize=7.6, pad=3.5,
                  color="#3a3a3a", loc="left")


    savefig(fig, "297_gan_training_dynamics.png")


if __name__ == "__main__":
    main()
