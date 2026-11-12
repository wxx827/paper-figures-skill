"""Figure 301 - Noise schedule analysis.

(a) the signal retention curve alpha-bar(t) for a linear, a cosine and a
sigmoid schedule, (b) the same three schedules in log-SNR, (c) what a
sample actually looks like at four fractions of the trajectory under each
schedule, and (d) the FID that each schedule reaches at two step budgets.
Colour is tied to the schedule and reused in all four panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, add_panel_label

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
# Schedules
# ---------------------------------------------------------------------------
def ab_linear(u, T=1000, b0=1e-4, b1=0.02):
    grid = np.linspace(0, 1, T + 1)
    betas = b0 + (b1 - b0) * grid
    ab = np.concatenate([[1.0], np.cumprod(1.0 - betas[1:])])
    return np.clip(np.interp(u, grid, ab), 1e-5, 1.0)


def ab_cosine(u, s=0.008):
    f = np.cos((np.asarray(u, float) + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(f / f0, 1e-5, 1.0)


def ab_sigmoid(u, lo=-4.0, hi=4.0, tau=0.62):
    def sg(x):
        return 1.0 / (1.0 + np.exp(-x / tau))
    v0, v1 = sg(lo), sg(hi)
    return np.clip((v1 - sg(lo + (hi - lo) * np.asarray(u, float)))
                   / (v1 - v0), 1e-5, 1.0)


SCHEDULES = [
    ("linear", ab_linear, "#9a9a9a", "-", 1.3),
    ("sigmoid", ab_sigmoid, C_NOISE, (0, (5, 2)), 1.4),
    ("cosine", ab_cosine, C_OURS, "-", 2.0),
]
FID = {"linear": (18.6, 12.8), "sigmoid": (13.1, 9.4), "cosine": (8.7, 6.9)}
TFRAC = [0.15, 0.30, 0.45, 0.60]


def make_eps(rng, n, corr=0.55, mono=0.55):
    e = rng.standard_normal((n, n, 3))
    e = np.dstack([gaussian_filter(e[..., k], corr) for k in range(3)])
    g = e.mean(axis=2, keepdims=True)
    e = mono * np.repeat(g, 3, axis=2) + (1 - mono) * e
    return e / (e.std() + 1e-9)


def q_sample(x0, ab, eps):
    z = x0 * 2.0 - 1.0
    return np.clip((np.sqrt(ab) * z + np.sqrt(1.0 - ab) * eps + 1.0) / 2.0,
                   0.0, 1.0)


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
    ax.grid(True, color="#ececf0", lw=0.55)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    N = 150

    x0 = render_face(sample_params(np.random.default_rng(58)), n=N, seed=58)
    eps = make_eps(np.random.default_rng(37), N)

    FW, FH = 7.4, 5.20
    fig = plt.figure(figsize=(FW, FH))
    u = np.linspace(0, 1, 400)

    # ---------------- (a) alpha-bar -----------------------------------------
    axA = fig.add_axes([0.078, 0.620, 0.372, 0.275])
    for nm, fn, c, ls, lw in SCHEDULES:
        axA.plot(u, fn(u), color=c, ls=ls, lw=lw, label=nm,
                 zorder=4 if nm == "cosine" else 3)
    for tf in TFRAC:
        axA.axvline(tf, color="#d9d9de", lw=0.6, ls=(0, (2, 2)), zorder=1)
    axA.set_xlim(0, 1)
    axA.set_ylim(-0.03, 1.05)
    axA.set_xlabel("normalised time  $t/T$", fontsize=7.2, labelpad=1.8)
    axA.set_ylabel("$\\bar{\\alpha}_t$   (signal retained)", fontsize=7.2,
                   labelpad=2.0)
    style_panel(axA)
    axA.legend(loc="upper right", fontsize=6.8, frameon=False,
               handlelength=1.8, labelspacing=0.34)
    axA.text(0.035, 0.27, "cosine keeps signal longer\nthrough the "
             "mid-trajectory", transform=axA.transAxes, fontsize=6.2,
             color=C_OURS, ha="left", va="top", linespacing=1.4)
    add_panel_label(axA, "(a)", x=-0.135, y=1.03, size=9.5)
    axA.set_title("signal retention", fontsize=7.6, pad=3.5, color="#3a3a3a",
                  loc="left")

    # ---------------- (b) log SNR -------------------------------------------
    axB = fig.add_axes([0.593, 0.620, 0.372, 0.275])
    for nm, fn, c, ls, lw in SCHEDULES:
        a = fn(u)
        axB.plot(u, np.log10(a / (1 - a + 1e-12) + 1e-12), color=c, ls=ls,
                 lw=lw, label=nm, zorder=4 if nm == "cosine" else 3)
    axB.axhline(0, color="#8a8a8a", lw=0.7, ls=(0, (3, 2)), zorder=2)
    axB.text(0.985, 0.20, "SNR = 1", ha="right", va="bottom", fontsize=6.1,
             color="#777777")
    axB.set_xlim(0, 1)
    axB.set_ylim(-3.6, 4.2)
    axB.set_xlabel("normalised time  $t/T$", fontsize=7.2, labelpad=1.8)
    axB.set_ylabel("$\\log_{10}$ SNR($t$)", fontsize=7.2, labelpad=2.0)
    style_panel(axB)
    add_panel_label(axB, "(b)", x=-0.135, y=1.03, size=9.5)
    axB.set_title("log signal-to-noise ratio", fontsize=7.6, pad=3.5,
                  color="#3a3a3a", loc="left")

    # ---------------- (c) sample appearance ---------------------------------
    gl, gt = 0.128, 0.455
    tw, wg, hg = 0.0843, 0.0060, 0.0075
    th = tw * FW / FH
    for r, (nm, fn, c, ls, lw) in enumerate(SCHEDULES):
        top = gt - r * (th + hg)
        for k, tf in enumerate(TFRAC):
            ax = fig.add_axes([gl + k * (tw + wg), top - th, tw, th],
                              zorder=2)
            image_axes(ax, color=c if nm == "cosine" else "#cccccc",
                       lw=1.1 if nm == "cosine" else 0.45)
            ax.imshow(q_sample(x0, float(fn(np.array([tf]))[0]), eps),
                      vmin=0.0, vmax=1.0, interpolation="nearest")
        fig.text(gl - 0.010, top - th / 2, nm, ha="right", va="center",
                 fontsize=7.2, color=c,
                 fontweight="bold" if nm == "cosine" else "normal")
    gw = 4 * tw + 3 * wg
    gh = 3 * th + 2 * hg
    for k, tf in enumerate(TFRAC):
        fig.text(gl + k * (tw + wg) + tw / 2, gt + 0.012, f"$t/T = {tf:.2f}$",
                 ha="center", va="bottom", fontsize=6.8, color="#3a3a3a")
    fig.text(0.030, gt + 0.062, "(c)", ha="left", va="bottom", fontsize=9.5,
             fontweight="bold", color=C_INK)
    fig.text(0.078, 0.500 + 0.017,
             "the same sample and the same noise field under each schedule",
             ha="left", va="bottom", fontsize=7.6, color="#3a3a3a")
    fig.text(gl + gw / 2, gt - gh - 0.020,
             "linear has already destroyed the image at $t/T = 0.30$",
             ha="center", va="top", fontsize=6.2, color="#8a8a8a",
             fontstyle="italic")

    # ---------------- (d) FID bars ------------------------------------------
    axD = fig.add_axes([0.640, gt - gh + 0.010, 0.325, gh - 0.010])
    xs = np.arange(3.0)
    bw = 0.34
    for j, (lab, alpha) in enumerate([("50 steps", 0.42), ("250 steps", 1.0)]):
        vals = [FID[nm][j] for nm, *_ in SCHEDULES]
        cols = [c for _nm, _fn, c, _ls, _lw in SCHEDULES]
        axD.bar(xs + (j - 0.5) * bw, vals, bw * 0.92,
                color=cols, alpha=alpha, edgecolor="white", linewidth=0.6,
                label=lab, zorder=3)
        for x, v in zip(xs + (j - 0.5) * bw, vals):
            axD.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=6.0, color="#3a3a3a")
    axD.set_xticks(xs)
    axD.set_xticklabels([nm for nm, *_ in SCHEDULES], fontsize=7.0)
    for lab, (nm, _fn, c, _ls, _lw) in zip(axD.get_xticklabels(), SCHEDULES):
        lab.set_color(c)
        if nm == "cosine":
            lab.set_fontweight("bold")
    axD.set_ylim(0, 22)
    axD.set_ylabel("FID $\\downarrow$", fontsize=7.2, labelpad=2.0)
    style_panel(axD)
    axD.grid(True, axis="y", color="#ececf0", lw=0.55)
    axD.grid(False, axis="x")
    handles = [Rectangle((0, 0), 1, 1, facecolor="#6a6a6a", alpha=a)
               for a in (0.42, 1.0)]
    axD.legend(handles, ["50 steps", "250 steps"], loc="upper right",
               fontsize=6.4, frameon=False, handlelength=1.2,
               labelspacing=0.28)
    add_panel_label(axD, "(d)", x=-0.155, y=1.05, size=9.5)
    axD.set_title("final sample quality", fontsize=7.6, pad=3.5,
                  color="#3a3a3a", loc="left")


    savefig(fig, "301_noise_schedule.png")


if __name__ == "__main__":
    main()
