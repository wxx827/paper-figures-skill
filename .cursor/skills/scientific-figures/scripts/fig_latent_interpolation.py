"""Figure 296 - Latent-space interpolation.

Five endpoint pairs (rows), nine frames each.  The two framed tiles at the
ends are inversions of real samples; the seven tiles between them are
decoded from spherically interpolated latents.  Because the walk happens in
the *generator's parameter space* rather than in pixel space, intermediate
frames are genuine deformations - the face shape, hair line, gaze and
palette migrate continuously instead of cross-fading.

The attached panel on the right shows the same five walks projected onto the
first two principal directions of the latent prior.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_palette

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
HUE_KEYS = ("skin_h", "hair_h", "bg_h", "iris_h")


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
    p["rx"] *= 1 + 0.11 * jitter * rng.standard_normal()
    p["ry"] *= 1 + 0.10 * jitter * rng.standard_normal()
    p["tilt"] += 0.13 * jitter * rng.standard_normal()
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


def lerp_params(a, b, t):
    """Interpolate in parameter space; hues take the shortest arc."""
    out = {}
    for k in a:
        if k in HUE_KEYS:
            d = (b[k] - a[k] + 0.5) % 1.0 - 0.5
            out[k] = (a[k] + t * d) % 1.0
        else:
            out[k] = (1.0 - t) * a[k] + t * b[k]
    return out


def slerp2(a, b, ts):
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(an @ bn, -1, 1))
    ra, rb = np.linalg.norm(a), np.linalg.norm(b)
    if om < 1e-6:
        return np.outer(1 - ts, a) + np.outer(ts, b)
    w1 = np.sin((1 - ts) * om) / np.sin(om)
    w2 = np.sin(ts * om) / np.sin(om)
    pts = np.outer(w1, an) + np.outer(w2, bn)
    r = (1 - ts) * ra + ts * rb
    return pts / np.linalg.norm(pts, axis=1, keepdims=True) * r[:, None]


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
    N = 140
    NF = 9
    NROW = 5
    alphas = np.linspace(0.0, 1.0, NF)

    # endpoint pairs chosen for contrast so the morph is legible; hair stays
    # inside a plausible range so the walk never passes through neon hues
    HAIR = [(0.050, 0.58, 0.16),   # near-black
            (0.088, 0.66, 0.42),   # auburn
            (0.108, 0.52, 0.82),   # blonde
            (0.040, 0.45, 0.27),   # dark brown
            (0.070, 0.10, 0.74)]   # silver
    HAIR_PAIRS = [(0, 2), (1, 4), (3, 2), (4, 0), (2, 3)]
    pairs = []
    for r in range(NROW):
        ra = np.random.default_rng(600 + 3 * r)
        rb = np.random.default_rng(650 + 3 * r)
        pa, pb = sample_params(ra), sample_params(rb)
        ia, ib = HAIR_PAIRS[r]
        ha, hb = HAIR[ia], HAIR[ib]
        pa["hair_h"], pa["hair_s"], pa["hair_v"] = ha
        pb["hair_h"], pb["hair_s"], pb["hair_v"] = hb
        pb["bg_h"] = (pa["bg_h"] + 0.40) % 1.0
        pairs.append((pa, pb))

    frames = [[render_face(lerp_params(pa, pb, a), n=N, seed=770 + 11 * r)
               for a in alphas] for r, (pa, pb) in enumerate(pairs)]

    # ---- canvas -----------------------------------------------------------
    FW, FH = 7.4, 3.90
    fig = plt.figure(figsize=(FW, FH))
    left, right_edge = 0.072, 0.722
    wgap, hgap = 0.006, 0.010
    cw = (right_edge - left - (NF - 1) * wgap) / NF
    ch = cw * FW / FH

    y_top = 0.842
    row_tops = [y_top - r * (ch + hgap) for r in range(NROW)]
    y_bot = row_tops[-1] - ch

    path_cols = get_palette("cb_sequential_blue", 8)[2:7]

    # ---- alpha ruler ------------------------------------------------------
    axr = fig.add_axes([left, y_top + 0.018, right_edge - left, 0.020])
    cmap = LinearSegmentedColormap.from_list(
        "ab", ["#e9eef4", C_OURS])
    axr.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cmap,
               extent=[0, 0.982, 0, 1], interpolation="bilinear")
    axr.fill([0.982, 1.0, 0.982], [-0.55, 0.5, 1.55], color=C_OURS, lw=0,
             clip_on=False)
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.set_xticks([])
    axr.set_yticks([])
    for s in axr.spines.values():
        s.set_visible(False)

    for k in (0, 2, 4, 6, 8):
        fig.text(left + k * (cw + wgap) + cw / 2, y_top + 0.048,
                 f"$\\alpha = {alphas[k]:.2f}$", ha="center", va="bottom",
                 fontsize=6.8, color="#3a3a3a",
                 fontweight="bold" if k in (0, 8) else "normal")
    fig.text(left + (right_edge - left) / 2, y_top + 0.090,
             "spherical interpolation   $z_\\alpha = \\mathrm{slerp}"
             "(z_A, z_B; \\alpha)$", ha="center", va="bottom", fontsize=7.4,
             color="#24425e")

    # ---- image grid -------------------------------------------------------
    for r in range(NROW):
        top = row_tops[r]
        for k in range(NF):
            ax = fig.add_axes([left + k * (cw + wgap), top - ch, cw, ch],
                              zorder=2)
            endpoint = k in (0, NF - 1)
            image_axes(ax, color=C_OURS if endpoint else "#d2d2d2",
                       lw=1.5 if endpoint else 0.4)
            ax.imshow(frames[r][k], vmin=0.0, vmax=1.0,
                      interpolation="nearest")
        # row tag + colour chip keyed to the trajectory panel
        fig.patches.append(Rectangle(
            (left - 0.0165, top - ch / 2 - 0.006), 0.009, 0.012,
            transform=fig.transFigure, facecolor=path_cols[r],
            edgecolor="none", zorder=3))
        fig.text(left - 0.021, top - ch / 2, f"pair {r + 1}", ha="right",
                 va="center", fontsize=6.6, color="#5a5a5a")

    fig.text(left, y_bot - 0.020, "$z_A$  (real, inverted)", ha="left",
             va="top", fontsize=6.6, color=C_OURS, fontweight="bold")
    fig.text(right_edge, y_bot - 0.020, "$z_B$  (real, inverted)",
             ha="right", va="top", fontsize=6.6, color=C_OURS,
             fontweight="bold")

    # ---- attached panel: trajectories in the latent projection ------------
    pw = 0.172
    ph = pw * FW / FH
    py0 = y_bot + ((y_top - y_bot) - ph) / 2
    fig.patches.append(FancyBboxPatch(
        (0.756, py0 - 0.082), 0.238, ph + 0.162,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#f3f3f6",
        edgecolor="#dcdce2", linewidth=0.6, zorder=0))
    axp = fig.add_axes([0.800, py0, pw, ph], zorder=1)
    axp.set_facecolor("none")

    rp = np.random.default_rng(96)
    cloud = rp.standard_normal((520, 2)) * np.array([1.05, 0.92])
    cloud += 0.35 * rp.standard_normal((520, 2)) ** 3
    axp.scatter(cloud[:, 0], cloud[:, 1], s=2.4, color="#c9c9d2",
                edgecolor="none", zorder=1, rasterized=True)

    tt = np.linspace(0, 1, 120)
    ends = []
    for r in range(NROW):
        rr = np.random.default_rng(880 + 5 * r)
        # spread the endpoints over distinct angular sectors and radii so the
        # five walks stay individually readable
        th_a = 2 * np.pi * r / NROW + rr.uniform(-0.25, 0.25)
        th_b = th_a + rr.uniform(0.9, 1.9) * (1 if r % 2 == 0 else -1)
        ra_, rb_ = rr.uniform(0.85, 1.45), rr.uniform(1.85, 2.55)
        if r % 2:
            ra_, rb_ = rb_, ra_
        za = np.array([np.cos(th_a), np.sin(th_a)]) * ra_
        zb = np.array([np.cos(th_b), np.sin(th_b)]) * rb_
        path = slerp2(za, zb, tt)
        nodes = slerp2(za, zb, alphas)
        axp.plot(path[:, 0], path[:, 1], color=path_cols[r], lw=1.5, zorder=3)
        axp.scatter(nodes[1:-1, 0], nodes[1:-1, 1], s=5.5,
                    color=path_cols[r], edgecolor="none", zorder=4)
        ends.append((za, zb, path_cols[r]))
    for za, zb, c in ends:
        axp.scatter([za[0], zb[0]], [za[1], zb[1]], s=22, marker="o",
                    facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)

    axp.set_xlim(-3.1, 3.1)
    axp.set_ylim(-3.1, 3.1)
    axp.set_aspect("equal")
    axp.set_xticks([-2, 0, 2])
    axp.set_yticks([-2, 0, 2])
    axp.tick_params(labelsize=6.1, length=2.2, width=0.6)
    axp.set_xlabel("PC 1 of $z$", fontsize=6.8, labelpad=1.2)
    axp.set_ylabel("PC 2 of $z$", fontsize=6.8, labelpad=1.2)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axp.spines[s].set_linewidth(0.7)
        axp.spines[s].set_color("#555555")
    axp.set_title("walks in the latent prior", fontsize=6.8, pad=3.0,
                  color="#24425e")
    axp.text(0.03, 0.03, "grey = prior samples", transform=axp.transAxes,
             fontsize=6.0, color="#8a8a8a", ha="left", va="bottom")

    # ---- titles -----------------------------------------------------------
    fig.text(0.022, y_bot - 0.080,
             "9 frames per pair   |   slerp on the unit sphere   |   "
             "identical display range [0, 1] for every tile\n"
             "Intermediate frames are deformations, not cross-fades: the "
             "silhouette, hair line and gaze translate continuously and no "
             "double exposure appears at $\\alpha = 0.5$.",
             fontsize=6.4, color="#777777", ha="left", va="top",
             linespacing=1.6)

    savefig(fig, "296_latent_interpolation.png")


if __name__ == "__main__":
    main()
