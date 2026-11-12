"""Figure 295 - Conditional generation grid.

Four conditioning modalities (rows): free-form text, a discrete class label,
a Canny edge map and a depth map.  The left-most tile of each row is the
conditioning signal itself; the four tiles to its right are samples drawn
with different random seeds under that same condition, so each row reads as
"consistency with the condition" (left-to-right) and "sample diversity"
(within a row).

Structural conditions (edges, depth) constrain geometry and let appearance
vary; semantic conditions (text, class) constrain appearance and let
geometry vary.  All imagery is procedurally synthesised.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

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

GEOM_KEYS = ("cx", "cy", "rx", "ry", "tilt", "eye_x", "eye_y", "eye_r",
             "brow_y", "brow_c", "mouth_y", "mouth_w", "mouth_c",
             "hair_line", "fringe")
APPEAR_KEYS = ("skin_h", "skin_s", "skin_v", "hair_h", "hair_s", "hair_v",
               "bg_h", "bg_s", "bg_v", "iris_h", "iris_s", "iris_v",
               "cloth_h", "light")


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


def _frame(p, n):
    """Shared geometry fields used by both the RGB and the depth renderer."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    ca, sa = np.cos(p["tilt"]), np.sin(p["tilt"])
    dx, dy = xx - p["cx"], yy - p["cy"]
    ux = dx * ca + dy * sa
    uy = -dx * sa + dy * ca
    return xx, yy, ux, uy


def render_face(p, n=144, seed=0, tex=1.0):
    rng = np.random.default_rng(seed)
    xx, yy, ux, uy = _frame(p, n)
    px = 1.0 / n

    bg_a = hsv_to_rgb([p["bg_h"] % 1.0, p["bg_s"] * 0.50,
                       min(p["bg_v"] + 0.15, 1.0)])
    bg_b = hsv_to_rgb([(p["bg_h"] + 0.07) % 1.0, p["bg_s"], p["bg_v"] * 0.66])
    t = yy[..., None]
    img = bg_a * (1 - t) + bg_b * t
    img = img * (1 + 0.055 * _fbm(rng, n, (5, 14), (0.5, 0.5)))[..., None]

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


def render_depth(p, n=144):
    """Analytic depth proxy in [0, 1] (far = 0, near = 1)."""
    xx, yy, ux, uy = _frame(p, n)
    px = 1.0 / n
    rx, ry = p["rx"], p["ry"]
    d = np.full((n, n), 0.03)

    dsh = (np.sqrt(((xx - p["cx"]) / (rx * 2.15)) ** 2
                   + ((yy - (p["cy"] + ry * 1.95)) / (ry * 1.05)) ** 2) - 1.0)
    d = np.maximum(d, 0.30 * _sm(dsh * rx * 2.15, 3 * px))

    hrx, hry = rx * 1.14, ry * 1.12
    dh = np.sqrt((ux / hrx) ** 2 + ((uy + ry * 0.10) / hry) ** 2) - 1.0
    hcap = _sm(dh * hrx, 2.6 * px)
    d = np.maximum(d, hcap * (0.44 + 0.10 * np.clip(-uy / ry, 0, 1)))

    dnk = (np.sqrt((ux / (rx * 0.46)) ** 2
                   + ((uy - ry * 0.92) / (ry * 0.52)) ** 2) - 1.0)
    d = np.maximum(d, 0.52 * _sm(dnk * rx * 0.46, 3 * px))

    rr = rx * (1.0 - 0.20 * np.clip(uy / ry, 0, 1) ** 2)
    q = np.clip(1.0 - (ux / rr) ** 2 - (uy / ry) ** 2, 0, 1)
    face = _sm(np.sqrt((ux / rr) ** 2 + (uy / ry) ** 2) - 1.0, 2.6 * px)
    d = np.maximum(d, face * (0.66 + 0.30 * np.sqrt(q)))

    dn = (np.sqrt((ux / (rx * 0.16)) ** 2
                  + ((uy - ry * 0.10) / (ry * 0.26)) ** 2) - 1.0)
    d = d + 0.075 * _sm(dn * rx * 0.16, 0.045) * face
    return np.clip(gaussian_filter(d, 0.8), 0, 1)


def render_edges(img):
    g = img @ np.array([0.299, 0.587, 0.114])
    g = gaussian_filter(g, 1.5)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    m = m / (np.percentile(m, 99.2) + 1e-9)
    e = (np.clip(m, 0, 1) > 0.34).astype(float)
    # drop isolated speckle: keep pixels with enough local support
    e = e * (gaussian_filter(e, 1.6) > 0.09)
    e = gaussian_filter(e, 0.55)
    return np.clip(e * 2.0, 0, 1)


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


def inherit(base, donor, keys):
    p = dict(base)
    for k in keys:
        p[k] = donor[k]
    return p


# ---------------------------------------------------------------------------
def image_axes(ax, color="#c8c8c8", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def card_axes(ax, face="#f4f2f8", edge=C_NOISE, lw=1.2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(face)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(edge)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    N = 168

    rows = []

    # -- row 1: free-form text ---------------------------------------------
    r1 = np.random.default_rng(101)
    cond1 = sample_params(r1)
    cond1.update(hair_h=0.055, hair_s=0.30, hair_v=0.16,
                 bg_h=0.50, bg_s=0.42, bg_v=0.62)
    s1 = [render_face(inherit(sample_params(np.random.default_rng(110 + k)),
                              cond1, ("hair_h", "hair_s", "hair_v",
                                      "bg_h", "bg_s", "bg_v")),
                      n=N, seed=110 + k) for k in range(4)]
    rows.append(dict(kind="text", name="Text prompt",
                     desc="free-form caption,\nCLIP text encoder",
                     payload='"studio portrait of a\nperson with short dark\n'
                             'hair, teal backdrop"',
                     samples=s1))

    # -- row 2: discrete class label ---------------------------------------
    r2 = np.random.default_rng(202)
    cond2 = sample_params(r2)
    cond2.update(hair_h=0.115, hair_s=0.66, hair_v=0.74,
                 mouth_c=0.130, bg_h=0.07, bg_s=0.20, bg_v=0.86)
    s2 = [render_face(inherit(sample_params(np.random.default_rng(210 + k)),
                              cond2, ("hair_h", "hair_s", "hair_v",
                                      "mouth_c", "bg_h", "bg_s", "bg_v")),
                      n=N, seed=210 + k) for k in range(4)]
    rows.append(dict(kind="class", name="Class label",
                     desc="one-hot over 40\nattribute classes",
                     payload="y = 17", samples=s2))

    # -- row 3: Canny edge map ---------------------------------------------
    r3 = np.random.default_rng(303)
    cond3 = sample_params(r3)
    edge_src = render_face(cond3, n=N, seed=303)
    edges = render_edges(edge_src)
    s3 = [render_face(inherit(sample_params(np.random.default_rng(310 + k)),
                              cond3, GEOM_KEYS), n=N, seed=310 + k)
          for k in range(4)]
    rows.append(dict(kind="edge", name="Edge map",
                     desc="Canny, low = 100 /\nhigh = 200",
                     payload=edges, samples=s3))

    # -- row 4: depth map ---------------------------------------------------
    r4 = np.random.default_rng(404)
    cond4 = sample_params(r4)
    depth = render_depth(cond4, n=N)
    s4 = [render_face(inherit(sample_params(np.random.default_rng(410 + k)),
                              cond4, GEOM_KEYS), n=N, seed=410 + k)
          for k in range(4)]
    rows.append(dict(kind="depth", name="Depth map",
                     desc="monocular depth,\nnear = white",
                     payload=depth, samples=s4))

    # ---- canvas -----------------------------------------------------------
    FW, FH = 7.4, 5.70
    fig = plt.figure(figsize=(FW, FH))
    left, right_edge = 0.205, 0.988
    sepgap = 0.026
    wgap, hgap = 0.008, 0.014
    cw = (right_edge - left - sepgap - 3 * wgap) / 5.0
    ch = cw * FW / FH
    x_cond = left
    x_smp = [left + cw + sepgap + k * (cw + wgap) for k in range(4)]

    y_top = 0.895
    row_tops = [y_top - r * (ch + hgap) for r in range(4)]
    y_bot = row_tops[-1] - ch

    # headers
    fig.text(x_cond + cw / 2, y_top + 0.014, "Conditioning input",
             ha="center", va="bottom", fontsize=7.8, fontweight="bold",
             color=C_NOISE)
    smp_lo, smp_hi = x_smp[0], x_smp[-1] + cw
    fig.text((smp_lo + smp_hi) / 2, y_top + 0.014,
             "Generated samples   (4 random seeds, identical condition)",
             ha="center", va="bottom", fontsize=7.8, fontweight="bold",
             color=C_OURS)

    # dashed separator between the condition column and the samples
    fig.add_artist(plt.Line2D(
        [x_cond + cw + sepgap / 2] * 2, [y_bot - 0.012, y_top + 0.010],
        color="#c3c3cb", lw=0.8, ls=(0, (3, 3)),
        transform=fig.transFigure, zorder=1))

    grey = plt.get_cmap("Greys_r")

    for r, spec in enumerate(rows):
        top = row_tops[r]
        # ---- condition tile
        axc = fig.add_axes([x_cond, top - ch, cw, ch], zorder=2)
        if spec["kind"] == "text":
            card_axes(axc)
            axc.text(0.5, 0.62, spec["payload"], ha="center", va="center",
                     fontsize=6.6, color="#3d3350", linespacing=1.5,
                     fontstyle="italic")
            axc.text(0.5, 0.145, "77 tokens  |  CFG $w=7.5$", ha="center",
                     va="center", fontsize=6.0, color="#8a80a0")
        elif spec["kind"] == "class":
            card_axes(axc)
            axc.text(0.5, 0.70, spec["payload"], ha="center", va="center",
                     fontsize=13.0, fontweight="bold", color="#3d3350")
            axc.text(0.5, 0.545, "blond  |  smiling", ha="center",
                     va="center", fontsize=6.4, color="#6b6280")
            for k in range(20):
                on = (k == 8)
                axc.add_patch(Rectangle(
                    (0.075 + k * 0.0425, 0.27), 0.034, 0.075,
                    facecolor=C_NOISE if on else "#dedae8",
                    edgecolor="none", zorder=3))
            axc.text(0.5, 0.175, "one-hot embedding", ha="center", va="top",
                     fontsize=6.0, color="#8a80a0")
        elif spec["kind"] == "edge":
            image_axes(axc, color=C_NOISE, lw=1.2)
            e = spec["payload"]
            rgb = (e[..., None] * np.array([1.0, 1.0, 1.0])
                   + (1 - e[..., None]) * np.array([0.055, 0.050, 0.090]))
            axc.imshow(np.clip(rgb, 0, 1), vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        else:
            image_axes(axc, color=C_NOISE, lw=1.2)
            axc.imshow(spec["payload"], cmap=grey, vmin=0.0, vmax=1.0,
                       interpolation="nearest")

        # ---- samples
        for k in range(4):
            ax = fig.add_axes([x_smp[k], top - ch, cw, ch], zorder=2)
            image_axes(ax, color="#cfcfcf", lw=0.5)
            ax.imshow(spec["samples"][k], vmin=0.0, vmax=1.0,
                      interpolation="nearest")
            if r == 3:
                ax.text(0.5, -0.055, f"seed {1000 + 7 * k}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.2, color="#909090")

        # ---- row label block
        fig.text(left - 0.018, top - ch / 2 + 0.042, spec["name"],
                 ha="right", va="center", fontsize=8.4, fontweight="bold",
                 color=C_INK)
        fig.text(left - 0.018, top - ch / 2 - 0.004, spec["desc"],
                 ha="right", va="center", fontsize=6.5, color="#7a7a7a",
                 linespacing=1.45)
        tagtxt = ("appearance fixed,\ngeometry free"
                  if spec["kind"] in ("text", "class")
                  else "geometry fixed,\nappearance free")
        fig.text(left - 0.018, top - ch / 2 - 0.062, tagtxt, ha="right",
                 va="center", fontsize=6.1, color=C_OURS, linespacing=1.4,
                 fontstyle="italic")

    fig.text(0.022, y_bot - 0.052,
             "Structural conditions pin down pose and silhouette while colour, "
             "hair and lighting stay free; semantic conditions do the "
             "opposite.",
             fontsize=6.4, color="#777777", ha="left", va="top")

    savefig(fig, "295_conditional_generation.png")


if __name__ == "__main__":
    main()
