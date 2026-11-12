"""Figure 294 - Sampler x step-budget quality matrix.

Three ODE/SDE samplers (rows) are run at five function-evaluation budgets
(columns) from the same initial latent, so every tile shows the *same*
identity degrading only because the trajectory is discretised more coarsely.
FID is printed inside each tile; the attached panel on the right traces
FID against NFE for the three samplers.

Everything is procedurally synthesised - a layout reference, not a result.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
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
# Sampling-error / quality degradation model
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
    f = rng.uniform(14, 26)
    stripe = np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)))
    blk = np.sign(np.sin(2 * np.pi * 7 * xx) * np.sin(2 * np.pi * 7 * yy))
    m = gaussian_filter(rng.standard_normal((n, n)), 10.0)
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    field = (0.62 * stripe + 0.38 * blk) * (0.30 + 0.70 * m)
    return np.clip(img + amp * field[..., None], 0, 1)


def degrade(img, rng, q):
    """q = 1 -> pristine, q = 0 -> heavily under-sampled."""
    d = float(np.clip(1.0 - q, 0.0, 1.0))
    out = _warp(img, rng, 7.0 * d ** 1.5)
    out = soften(out, 3.0 * d ** 1.45)
    out = _artifacts(out, rng, 0.135 * d ** 1.25)
    gains = 1.0 + 0.22 * d ** 1.2 * rng.uniform(-1, 1, 3)
    out = out * gains + 0.05 * d ** 1.2 * rng.uniform(-1, 1, 3)
    out = 0.5 + (out - 0.5) * (1.0 - 0.28 * d)
    out = out + 0.085 * d ** 1.3 * rng.standard_normal(out.shape)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def arrow_bar(fig, x0, y0, w, h, c_from, c_to):
    ax = fig.add_axes([x0, y0, w, h])
    cmap = LinearSegmentedColormap.from_list("ab", [c_from, c_to])
    ax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cmap,
              extent=[0, 0.978, 0, 1], interpolation="bilinear")
    ax.fill([0.978, 1.0, 0.978], [-0.55, 0.5, 1.55], color=c_to, lw=0,
            clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# ---------------------------------------------------------------------------
STEPS = np.array([10, 20, 50, 100, 250])
SAMPLERS = [
    ("DDPM", "ancestral", [0.06, 0.22, 0.56, 0.83, 0.97], "#9a9a9a", False),
    ("DDIM", "$\\eta = 0$", [0.36, 0.63, 0.87, 0.955, 0.99], "#565656", False),
    ("DPM-Solver++", "Ours", [0.74, 0.91, 0.975, 0.993, 1.0], C_OURS, True),
]
FIDS = {
    "DDPM": [86.4, 51.7, 19.8, 10.6, 6.72],
    "DDIM": [31.5, 16.9, 9.14, 7.05, 6.11],
    "DPM-Solver++": [9.86, 6.83, 5.62, 5.31, 5.18],
}


def main():
    set_pub_style(base_font=8)
    N = 176

    p = sample_params(np.random.default_rng(9))
    x_ref = render_face(p, n=N, seed=4)

    tiles = {}
    for r, (nm, _sub, qs, _c, _o) in enumerate(SAMPLERS):
        for c, q in enumerate(qs):
            tiles[(r, c)] = degrade(x_ref, np.random.default_rng(400 + 17 * r + c),
                                    q)

    FW, FH = 7.4, 3.95
    fig = plt.figure(figsize=(FW, FH))
    left, right_edge = 0.168, 0.680
    ncol, wgap, hgap = 5, 0.009, 0.016
    cw = (right_edge - left - (ncol - 1) * wgap) / ncol
    ch = cw * FW / FH

    y_top = 0.800
    row_tops = [y_top - r * (ch + hgap) for r in range(3)]
    y_bot = row_tops[-1] - ch

    # "Ours" row highlight
    fig.patches.append(FancyBboxPatch(
        (0.022, row_tops[2] - ch - 0.010),
        (right_edge - 0.022) + 0.010, ch + 0.020,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        transform=fig.transFigure, facecolor="#e8eff7", edgecolor="#c6d6e8",
        linewidth=0.6, zorder=0))

    arrow_bar(fig, left, y_top + 0.017, right_edge - left, 0.022,
              "#f6d9d6", C_REAL)
    fig.text(left + (right_edge - left) / 2, y_top + 0.052,
             "increasing sampling budget   (number of function evaluations)",
             ha="center", va="bottom", fontsize=7.2, color="#3a3a3a")

    for c, s in enumerate(STEPS):
        fig.text(left + c * (cw + wgap) + cw / 2, y_top + 0.094,
                 f"{s} steps", ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color=C_INK)

    for r, (nm, sub, qs, col, is_ours) in enumerate(SAMPLERS):
        top = row_tops[r]
        for c in range(ncol):
            ax = fig.add_axes([left + c * (cw + wgap), top - ch, cw, ch],
                              zorder=2)
            image_axes(ax, color=C_OURS if is_ours else "#c8c8c8",
                       lw=1.2 if is_ours else 0.5)
            ax.imshow(tiles[(r, c)], vmin=0.0, vmax=1.0,
                      interpolation="nearest")
            fid = FIDS[nm][c]
            best = fid == min(FIDS[k][c] for k in FIDS)
            fcol = C_BAD if fid > 30 else (C_OURS if best else "#3a3a3a")
            ax.text(0.045, 0.045, f"FID {fid:.1f}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=6.4, color=fcol,
                    fontweight="bold" if (best or fid > 30) else "normal",
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                              edgecolor="none", alpha=0.88))
        fig.text(left - 0.016, top - ch / 2 + 0.017, nm, ha="right",
                 va="center", fontsize=8.0,
                 fontweight="bold" if is_ours else "normal",
                 color=C_OURS if is_ours else "#2b2b2b")
        fig.text(left - 0.016, top - ch / 2 - 0.020, sub, ha="right",
                 va="center", fontsize=6.7,
                 color=C_OURS if is_ours else "#8a8a8a",
                 fontstyle="italic")

    # ---- attached panel: FID vs NFE ---------------------------------------
    ph = (y_top - y_bot) * 0.58
    py0 = y_bot + ((y_top - y_bot) - ph) / 2
    fig.patches.append(FancyBboxPatch(
        (0.706, py0 - 0.088), 0.288, ph + 0.180,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#f3f3f6", edgecolor="#dcdce2",
        linewidth=0.6, zorder=0))
    axp = fig.add_axes([0.776, py0, 0.192, ph], zorder=1)
    axp.set_facecolor("none")
    for nm, sub, qs, col, is_ours in SAMPLERS:
        axp.plot(STEPS, FIDS[nm], marker="o", ms=3.4, color=col,
                 lw=2.0 if is_ours else 1.3,
                 zorder=4 if is_ours else 3,
                 markerfacecolor="white" if is_ours else col,
                 markeredgewidth=1.2 if is_ours else 0.8,
                 label=nm + (" (Ours)" if is_ours else ""))
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_xticks(STEPS)
    axp.set_xticklabels([str(s) for s in STEPS])
    axp.set_yticks([5, 10, 20, 50, 100])
    axp.set_yticklabels(["5", "10", "20", "50", "100"])
    axp.minorticks_off()
    axp.set_xlabel("NFE (sampling steps)", fontsize=6.9, labelpad=1.5)
    axp.set_ylabel("FID $\\downarrow$", fontsize=6.9, labelpad=1.5)
    axp.tick_params(labelsize=6.2, length=2.2, width=0.6)
    axp.grid(True, which="major", color="#e2e2e6", lw=0.5)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axp.spines[s].set_linewidth(0.7)
        axp.spines[s].set_color("#555555")
    axp.legend(loc="upper right", fontsize=6.0, frameon=False,
               handlelength=1.3, labelspacing=0.30, borderaxespad=0.1)
    axp.set_title("FID vs. step budget", fontsize=6.9, pad=3.0,
                  color="#3a3a3a")

    fig.text(0.022, 0.158, "same initial latent in every tile   |   "
             "FID over 50k samples   |   identical display range [0, 1]   |   "
             "our solver matches the 250-step reference at 20 NFE",
             fontsize=6.3, color="#777777", ha="left", va="top")

    savefig(fig, "294_sampling_steps_compare.png")


if __name__ == "__main__":
    main()
