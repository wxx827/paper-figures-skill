"""Figure 293 - Forward / reverse diffusion process.

Top row: the forward Markov chain q(x_t | x_{t-1}) turning a sample into
isotropic Gaussian noise over 8 sampled time steps.  Bottom row: the learned
reverse chain p_theta(x_{t-1} | x_t) generating a fresh sample from noise.
A small attached panel on the right traces log-SNR(t) for the cosine
schedule and marks the eight sampled steps.

All imagery is synthetic (procedural portrait engine + exact DDPM noising
formula); the figure is a layout reference, not a scientific result.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

# ---- shared generative-model palette --------------------------------------
C_REAL = "#2e7d5b"
C_OURS = "#3b5b92"
C_BAD = "#b23a48"
C_NOISE = "#8e6bbf"
C_GREY = "#8a8a8a"
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
    """Smooth step: 1 well inside (d < 0), 0 well outside."""
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
    """Rasterise one abstract portrait from a smooth parameter dict."""
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

    # shoulders / garment
    cloth = hsv_to_rgb([(p["bg_h"] + 0.45) % 1.0, 0.42, 0.50 + p["cloth_h"]])
    dsh = (np.sqrt(((xx - p["cx"]) / (rx * 2.15)) ** 2
                   + ((yy - (p["cy"] + ry * 1.95)) / (ry * 1.05)) ** 2) - 1.0)
    img = _over(img, _sm(dsh * rx * 2.15, 3 * px), cloth)

    # neck
    dnk = (np.sqrt((ux / (rx * 0.44)) ** 2
                   + ((uy - ry * 0.92) / (ry * 0.50)) ** 2) - 1.0)
    img = _over(img, _sm(dnk * rx * 0.44, 3 * px), skin * 0.78)

    # face (tapered jaw)
    rr = rx * (1.0 - 0.20 * np.clip(uy / ry, 0, 1) ** 2)
    dfc = np.sqrt((ux / rr) ** 2 + (uy / ry) ** 2) - 1.0
    face = _sm(dfc * rx, 2.6 * px)
    shade = (1.0 + p["light"] * (-0.42 * ux / rx - 0.30 * uy / ry)
             - 0.34 * np.clip(np.hypot(ux / rx, uy / ry) - 0.70, 0, None))
    img = _over(img, face, skin[None, None, :]
                * np.clip(shade, 0.52, 1.26)[..., None])

    # hair: cap above the hair line plus side fringes
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

    # nose shadow
    dn = (np.sqrt((ux / (rx * 0.135)) ** 2
                  + ((uy - ry * 0.13) / (ry * 0.21)) ** 2) - 1.0)
    img = _over(img, _sm(dn * rx * 0.135, 0.030) * 0.24, skin * 0.68)

    # eyes
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
        m_i = _sm(di * er * 0.74, 2.0 * px) * alm
        img = _over(img, m_i, iris)
        dp = np.hypot(axl - s * er * 0.04, uy - ey) / (er * 0.33) - 1.0
        img = _over(img, _sm(dp * er * 0.33, 2.0 * px) * alm, (0.07, 0.06, 0.08))
        dl = np.hypot(axl - s * er * 0.30,
                      uy - ey + er * 0.26) / (er * 0.18) - 1.0
        img = _over(img, _sm(dl * er * 0.18, 2.0 * px) * alm * 0.92, (1, 1, 1))
        dls = np.abs(uy - ey + er * 0.74) - er * 0.12
        lash = _sm(dls, 2.2 * px) * _sm(np.abs(axl) - er * 1.55, 0.020)
        img = _over(img, lash * 0.85, hair * 0.55)

    # brows
    for s in (-1.0, 1.0):
        bxn = (ux - s * ex) / (er * 1.95)
        tgt = p["brow_y"] * ry - p["brow_c"] * ry * (1 - np.clip(bxn, -1, 1) ** 2)
        db = np.abs(uy - tgt) - er * 0.17
        brow = _sm(db, 2.2 * px) * _sm(np.abs(bxn) - 1.0, 0.26)
        img = _over(img, brow * 0.94, hair * 0.82)

    # mouth
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

    # global film grain / vignette
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
# Diffusion maths
# ---------------------------------------------------------------------------
def alpha_bar_cosine(tfrac, s=0.008):
    f = np.cos((np.asarray(tfrac, float) + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(f / f0, 1e-5, 1.0)


def make_eps(rng, n, corr=0.55, mono=0.55):
    """Noise field: mildly spatially correlated and partly achromatic, which
    is what published diffusion strips actually look like at print size."""
    e = rng.standard_normal((n, n, 3))
    if corr > 0:
        e = np.dstack([gaussian_filter(e[..., k], corr) for k in range(3)])
    g = e.mean(axis=2, keepdims=True)
    e = mono * np.repeat(g, 3, axis=2) + (1 - mono) * e
    return e / (e.std() + 1e-9)


def q_sample(x0, ab, eps):
    """Exact DDPM forward marginal, images kept in [0, 1] for display."""
    z = x0 * 2.0 - 1.0
    xt = np.sqrt(ab) * z + np.sqrt(1.0 - ab) * eps
    return np.clip((xt + 1.0) / 2.0, 0.0, 1.0)


def soften(x0, sigma):
    if sigma <= 0.02:
        return x0
    out = np.dstack([gaussian_filter(x0[..., k], sigma) for k in range(3)])
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def tag(ax, text, color, x=0.045, y=0.955):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.0, color=color, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.86))


def grid_axes(fig, left, top, cw, ch, ncol, wgap):
    return [fig.add_axes([left + c * (cw + wgap), top - ch, cw, ch])
            for c in range(ncol)]


def arrow_bar(fig, x0, y0, w, h, c_from, c_to):
    ax = fig.add_axes([x0, y0, w, h])
    cmap = LinearSegmentedColormap.from_list("ab", [c_from, c_to])
    ax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cmap,
              extent=[0, 0.972, 0, 1], interpolation="bilinear")
    ax.fill([0.972, 1.0, 0.972], [-0.55, 0.5, 1.55], color=c_to,
            lw=0, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240293)

    N = 144
    T = 1000
    # t chosen so that sqrt(1 - alpha_bar) - i.e. the *visible* noise level -
    # advances in even perceptual increments across the strip
    ts = np.array([0, 40, 85, 150, 250, 420, 640, 1000])
    abs_t = alpha_bar_cosine(ts / T)

    # forward chain: one real sample, one shared noise field
    p_fwd = sample_params(np.random.default_rng(7))
    x0 = render_face(p_fwd, n=N, seed=3)
    eps_f = make_eps(np.random.default_rng(11), N)
    fwd = [q_sample(x0, a, eps_f) for a in abs_t]

    # reverse chain: a *different* identity is generated out of noise
    p_rev = sample_params(np.random.default_rng(23))
    xg = render_face(p_rev, n=N, seed=5)
    eps_r = make_eps(np.random.default_rng(31), N)
    rev = []
    for a, tt in zip(abs_t[::-1], ts[::-1]):
        # the network's running x0-estimate is coarse early on
        x0_hat = soften(xg, 4.6 * (tt / T) ** 1.35)
        rev.append(q_sample(x0_hat, a, eps_r))

    # ---- canvas -----------------------------------------------------------
    FW, FH = 7.4, 2.88
    fig = plt.figure(figsize=(FW, FH))
    left, right_edge = 0.055, 0.720
    ncol, wgap = 8, 0.0075
    cw = (right_edge - left - (ncol - 1) * wgap) / ncol
    ch = cw * FW / FH
    bar_h = 0.030

    y_bar1 = 0.855
    y_row1 = y_bar1 - 0.014
    y_bar2 = 0.492
    y_row2 = y_bar2 - 0.014

    arrow_bar(fig, left, y_bar1, right_edge - left, bar_h, "#d5dbe6", C_NOISE)
    arrow_bar(fig, left, y_bar2, right_edge - left, bar_h,
              lighten_hex(C_NOISE, 0.62), C_OURS)

    fig.text(left + (right_edge - left) / 2, y_bar1 + bar_h + 0.018,
             "Forward process   $q(x_t\\,|\\,x_{t-1}) = "
             "\\mathcal{N}(\\sqrt{1-\\beta_t}\\,x_{t-1},\\ \\beta_t I)$",
             ha="center", va="bottom", fontsize=7.4, color="#3d3350")
    fig.text(left + (right_edge - left) / 2, y_bar2 + bar_h + 0.018,
             "Reverse process   $p_\\theta(x_{t-1}\\,|\\,x_t) = "
             "\\mathcal{N}(\\mu_\\theta(x_t,t),\\ \\Sigma_\\theta(x_t,t))$",
             ha="center", va="bottom", fontsize=7.4, color="#24425e")

    axes1 = grid_axes(fig, left, y_row1, cw, ch, ncol, wgap)
    axes2 = grid_axes(fig, left, y_row2, cw, ch, ncol, wgap)

    for k, ax in enumerate(axes1):
        image_axes(ax)
        ax.imshow(fwd[k], vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.text(0.5, -0.085, f"$t={ts[k]}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.7, color="#444444")
    tag(axes1[0], "real $x_0$", C_REAL)
    tag(axes1[-1], "$x_T\\!\\sim\\!\\mathcal{N}(0,I)$", C_NOISE)
    for s in axes1[0].spines.values():
        s.set_color(C_REAL)
        s.set_linewidth(1.5)
    for s in axes1[-1].spines.values():
        s.set_color(C_NOISE)
        s.set_linewidth(1.5)

    rev_ts = ts[::-1]
    for k, ax in enumerate(axes2):
        image_axes(ax)
        ax.imshow(rev[k], vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.text(0.5, -0.085, f"$t={rev_ts[k]}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.7, color="#444444")
    for s in axes2[-1].spines.values():
        s.set_color(C_OURS)
        s.set_linewidth(1.5)
    tag(axes2[-1], "sample $\\hat{x}_0$", C_OURS)
    tag(axes2[0], "$x_T$", C_NOISE)

    fig.text(left - 0.011, y_row1 - ch / 2, "Forward", rotation=90,
             ha="right", va="center", fontsize=7.4, fontweight="bold",
             color="#3d3350")
    fig.text(left - 0.011, y_row2 - ch / 2, "Reverse", rotation=90,
             ha="right", va="center", fontsize=7.4, fontweight="bold",
             color="#24425e")

    # ---- attached panel: log-SNR of the cosine schedule -------------------
    blk_lo, blk_hi = y_row2 - ch, y_row1
    px0, pw = 0.822, 0.152
    ph = (blk_hi - blk_lo) * 0.58
    py0 = blk_lo + ((blk_hi - blk_lo) - ph) / 2
    fig.patches.append(FancyBboxPatch(
        (0.752, py0 - 0.088), 0.240, ph + 0.165,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#f3f3f6",
        edgecolor="#dcdce2", linewidth=0.6, zorder=0))

    axp = fig.add_axes([px0, py0, pw, ph], zorder=1)
    axp.set_facecolor("none")
    tg = np.linspace(0, 1, 400)
    for nm, sched, col, ls, lw in [
            ("cosine", alpha_bar_cosine(tg), C_NOISE, "-", 1.6)]:
        snr = np.log10(sched / (1 - sched + 1e-12) + 1e-12)
        axp.plot(tg * T, snr, color=col, lw=lw, ls=ls, zorder=3)
    snr_pts = np.log10(abs_t / (1 - abs_t + 1e-12) + 1e-12)
    axp.scatter(ts, snr_pts, s=13, facecolor="white", edgecolor=C_NOISE,
                linewidth=1.0, zorder=4)
    axp.axhline(0, color=C_GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    axp.text(T, 0.35, "SNR = 1", ha="right", va="bottom", fontsize=6.2,
             color="#666666")
    axp.set_xlim(-25, T + 25)
    axp.set_ylim(-3.4, 3.8)
    axp.set_xlabel("time step  $t$", fontsize=6.8, labelpad=1.2)
    axp.set_ylabel("$\\log_{10}$ SNR($t$)", fontsize=6.8, labelpad=1.2)
    axp.set_xticks([0, 500, 1000])
    axp.set_yticks([-3, 0, 3])
    axp.tick_params(labelsize=6.2, length=2.2, width=0.6)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axp.spines[s].set_linewidth(0.7)
        axp.spines[s].set_color("#555555")
    axp.set_title("cosine schedule", fontsize=6.8, pad=2.5, color="#3d3350")

    # ---- titles -----------------------------------------------------------
    fig.text(0.055, 0.185, "cosine $\\bar{\\alpha}_t$ schedule   |   "
             "$T=1000$, 8 sampled steps   |   a single noise field is shared "
             "along each row, so the trajectory is continuous   |   "
             "identical display range [0, 1] for every tile",
             fontsize=6.3, color="#777777", ha="left", va="top")

    savefig(fig, "293_diffusion_process.png")


def lighten_hex(color, amount=0.5):
    import matplotlib as mpl
    r, g, b = mpl.colors.to_rgb(color)
    f = float(np.clip(amount, 0, 1))
    return mpl.colors.to_hex((r + (1 - r) * f, g + (1 - g) * f,
                              b + (1 - b) * f))


if __name__ == "__main__":
    main()
