"""Figure 273 - Grad-CAM comparison grid.

Three test images (rows) x five views (columns): the grayscale input and the
class-activation maps read out at three depths of the backbone plus the
proposed head.  Every CAM is drawn with the same ``magma`` normalisation so the
sharpening of the evidence toward the object is a real, comparable effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

HEAT = "magma"
ALPHA = 0.50


# ---------------------------------------------------------------------------
# Synthetic "photograph" generator
# ---------------------------------------------------------------------------
def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(1.5, 4.0, 10.0), weights=(0.45, 0.33, 0.22)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _ellipse(yy, xx, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _soft(mask, sigma=0.9):
    return np.clip(gaussian_filter(mask.astype(float), sigma), 0, 1)


def _paint(img, mask, value, texture):
    return img * (1 - mask) + mask * np.clip(value + texture, 0.02, 0.99)


def synth_photo(kind, seed, n=128):
    """Return (grayscale image in 0..1, subject centre, subject radii)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    v = yy / (n - 1.0)

    # --- defocused background: sky gradient over a ground plane ------------
    horizon = 0.56 + 0.05 * rng.standard_normal()
    sky = 0.86 - 0.30 * (v / horizon)
    ground = 0.44 - 0.16 * ((v - horizon) / (1 - horizon))
    bg = np.where(v < horizon, sky, ground)
    bg = bg + 0.11 * _fbm(rng, n, (3, 9, 22), (0.4, 0.35, 0.25))
    for _ in range(5):                       # distant clutter / bokeh blobs
        cx, cy = rng.uniform(0, n), rng.uniform(0, horizon * n)
        rr = rng.uniform(0.06, 0.18) * n
        bg += rng.uniform(-0.13, 0.13) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rr ** 2)))
    img = gaussian_filter(bg, 2.6)           # background is out of focus

    tex = 0.034 * _fbm(rng, n, (0.9, 2.4, 6.0), (0.44, 0.33, 0.23))

    if kind == "dog":
        cx, cy, rx, ry = 0.44 * n, 0.545 * n, 0.235 * n, 0.135 * n
        body = _ellipse(yy, xx, cx, cy, rx, ry, -6)
        neck = _ellipse(yy, xx, 0.625 * n, 0.455 * n, 0.075 * n, 0.095 * n, -22)
        head = _ellipse(yy, xx, 0.705 * n, 0.355 * n, 0.098 * n, 0.088 * n)
        snout = _ellipse(yy, xx, 0.815 * n, 0.385 * n, 0.070 * n, 0.042 * n, 6)
        ear = _ellipse(yy, xx, 0.655 * n, 0.288 * n, 0.038 * n, 0.070 * n, 16)
        tail = _ellipse(yy, xx, 0.205 * n, 0.435 * n, 0.090 * n, 0.024 * n, 42)
        legs = np.zeros((n, n), bool)
        for lx, lh in ((0.315, 0.145), (0.395, 0.140),
                       (0.545, 0.145), (0.620, 0.140)):
            legs |= _ellipse(yy, xx, lx * n, (0.545 + lh) * n,
                             0.030 * n, lh * n)
        sub = body | neck | head | snout | ear | tail | legs
        img = _paint(img, _soft(sub, 0.8), 0.30, tex * 1.1)
        img = _paint(img, _soft(head | snout | ear, 0.7), 0.37, tex * 0.9)
        img = _paint(img, _soft(_ellipse(yy, xx, 0.868 * n, 0.378 * n,
                                         0.021 * n, 0.017 * n), 0.45),
                     0.08, tex * 0.3)
        ext = (0.30 * n, 0.22 * n)
        ctr = (0.58 * n, 0.49 * n)

    elif kind == "car":
        img = _paint(img, _soft(yy > 0.70 * n, 1.4), 0.30, tex * 0.9)
        cx, cy = 0.50 * n, 0.60 * n
        chassis = _ellipse(yy, xx, cx, cy, 0.325 * n, 0.088 * n)
        chassis |= ((xx > 0.20 * n) & (xx < 0.80 * n)
                    & (yy > 0.55 * n) & (yy < 0.685 * n))
        cabin = _ellipse(yy, xx, 0.50 * n, 0.525 * n, 0.175 * n, 0.085 * n)
        cabin &= yy < 0.575 * n
        sub = chassis | cabin
        img = _paint(img, _soft(sub, 0.7), 0.24, tex * 1.2)
        glass = _ellipse(yy, xx, 0.50 * n, 0.530 * n, 0.140 * n, 0.060 * n)
        img = _paint(img, _soft(glass & (yy < 0.545 * n), 0.6), 0.66, tex)
        for wx in (0.315, 0.685):
            img = _paint(img, _soft(_ellipse(yy, xx, wx * n, 0.690 * n,
                                             0.062 * n, 0.058 * n), 0.6),
                         0.09, tex * 0.5)
            img = _paint(img, _soft(_ellipse(yy, xx, wx * n, 0.690 * n,
                                             0.028 * n, 0.026 * n), 0.4),
                         0.55, tex * 0.4)
        ext = (0.34 * n, 0.16 * n)
        ctr = (cx, 0.605 * n)

    else:                                     # bird
        cx, cy = 0.52 * n, 0.55 * n
        body = _ellipse(yy, xx, cx, cy, 0.155 * n, 0.215 * n, 16)
        head = _ellipse(yy, xx, 0.60 * n, 0.295 * n, 0.088 * n, 0.082 * n)
        wing = _ellipse(yy, xx, 0.475 * n, 0.545 * n, 0.085 * n, 0.155 * n, 24)
        tail = _ellipse(yy, xx, 0.385 * n, 0.815 * n, 0.055 * n, 0.135 * n, 22)
        sub = body | head | wing | tail
        img = _paint(img, _soft(sub, 0.8), 0.30, tex * 1.9)
        img = _paint(img, _soft(wing, 0.7), 0.20, tex * 2.3)
        beak = _ellipse(yy, xx, 0.685 * n, 0.320 * n, 0.052 * n, 0.028 * n, -14)
        img = _paint(img, _soft(beak, 0.5), 0.70, tex * 0.5)
        img = _paint(img, _soft(_ellipse(yy, xx, 0.622 * n, 0.272 * n,
                                         0.019 * n, 0.017 * n), 0.4),
                     0.08, tex * 0.3)
        branch = (np.abs(yy - (0.86 * n + 0.05 * n * np.sin(xx / 18.0)))
                  < 0.022 * n)
        img = _paint(img, _soft(branch, 0.8), 0.26, tex * 1.1)
        ext = (0.22 * n, 0.30 * n)
        ctr = (0.545 * n, 0.50 * n)

    img += 0.012 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo), 0, 1), ctr, ext


# ---------------------------------------------------------------------------
# Class-activation maps
# ---------------------------------------------------------------------------
def cam(n, peaks, blur=2.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    m = np.zeros((n, n))
    for cx, cy, sx, sy, amp, ang in peaks:
        t = np.deg2rad(ang)
        u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
        m += amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return gaussian_filter(m, blur)


def build_cams(n, ctr, ext, seed):
    """Shallow -> deep: scattered texture response collapsing onto the object."""
    rng = np.random.default_rng(seed)
    cx, cy = ctr
    ex, ey = ext
    out = {}

    peaks = []
    for _ in range(16):                       # layer2: edge / texture clutter
        peaks.append((rng.uniform(0.05, 0.95) * n, rng.uniform(0.05, 0.95) * n,
                      rng.uniform(0.045, 0.085) * n,
                      rng.uniform(0.045, 0.085) * n,
                      rng.uniform(0.30, 0.55), rng.uniform(0, 180)))
    out["Layer 2"] = cam(n, peaks, 1.6)

    peaks = []
    for _ in range(6):                        # layer3: a few broad regions
        on = rng.random() < 0.5
        px = cx + rng.uniform(-0.55, 0.55) * ex if on else rng.uniform(0.1, 0.9) * n
        py = cy + rng.uniform(-0.55, 0.55) * ey if on else rng.uniform(0.1, 0.9) * n
        peaks.append((px, py, rng.uniform(0.09, 0.15) * n,
                      rng.uniform(0.09, 0.15) * n,
                      rng.uniform(0.45, 0.80) if on else rng.uniform(0.30, 0.55),
                      rng.uniform(0, 180)))
    out["Layer 3"] = cam(n, peaks, 2.2)

    out["Layer 4"] = cam(n, [                 # layer4: object + one distractor
        (cx + 0.16 * ex, cy - 0.10 * ey, 0.44 * ex, 0.46 * ey, 0.92, 0),
        (cx - 0.55 * ex, cy + 0.35 * ey, 0.26 * ex, 0.30 * ey, 0.42, 0),
        (0.14 * n, 0.20 * n, 0.11 * n, 0.09 * n, 0.34, 0),
    ], 2.6)

    out["Ours"] = cam(n, [                    # ours: one tight, on-object blob
        (cx, cy, 0.33 * ex, 0.34 * ey, 1.00, 0),
        (cx + 0.28 * ex, cy - 0.34 * ey, 0.15 * ex, 0.16 * ey, 0.30, 0),
    ], 2.2)

    # Deeper read-outs are also more confident: fix each map's peak so the
    # shared colour scale encodes depth, not the accident of blob overlap.
    for key, peak in (("Layer 2", 0.42), ("Layer 3", 0.60),
                      ("Layer 4", 0.80), ("Ours", 1.00)):
        out[key] = out[key] * (peak / out[key].max())
    return out


def image_axes(ax, color="#c9c9c9", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def corner_tag(ax, text, bold=False, color="#1a1a1a"):
    ax.text(0.035, 0.035, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.1, color=color,
            fontweight="bold" if bold else "normal", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#b8b8b8", linewidth=0.4, alpha=0.90))


def main():
    set_pub_style(base_font=8)
    n = 128
    rows = [("dog", 11, "Retriever", [0.34, 0.51, 0.72, 0.94]),
            ("car", 23, "Sports car", [0.28, 0.46, 0.68, 0.91]),
            ("bird", 37, "Macaw", [0.31, 0.44, 0.63, 0.89])]
    cols = ["Input", "Layer 2", "Layer 3", "Layer 4", "Ours"]

    scenes, maps = [], []
    for k, (kind, sd, _, _) in enumerate(rows):
        img, ctr, ext = synth_photo(kind, sd, n)
        scenes.append(img)
        maps.append(build_cams(n, ctr, ext, seed=200 + 13 * k))

    # One normalisation for every CAM in the figure.
    vmax = max(m.max() for d in maps for m in d.values())
    norm = Normalize(vmin=0.0, vmax=1.0)
    for d in maps:
        for key in d:
            d[key] = d[key] / vmax

    fig = plt.figure(figsize=(7.2, 4.62))
    gs = fig.add_gridspec(3, 5, wspace=0.035, hspace=0.035,
                          left=0.062, right=0.905, top=0.868, bottom=0.058)

    for r, (kind, sd, cls, confs) in enumerate(rows):
        img = scenes[r]
        for c, name in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            image_axes(ax)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="bilinear")
            if c == 0:
                corner_tag(ax, f"GT: {cls}")
            else:
                ax.imshow(maps[r][name], cmap=HEAT, norm=norm, alpha=ALPHA,
                          interpolation="bilinear")
                best = c == 4
                corner_tag(ax, f"{cls}  {confs[c - 1]:.2f}", bold=best,
                           color="#1a5c38" if best else "#1a1a1a")
            if r == 0:
                ax.set_title(name, fontsize=7.8, pad=3.4,
                             fontweight="bold" if name == "Ours" else "normal",
                             color="#12503a" if name == "Ours" else "#1a1a1a")
            if c == 0:
                ax.set_ylabel(f"Sample {r + 1}", fontsize=7.2,
                              fontweight="bold", labelpad=4)

    cax = fig.add_axes([0.918, 0.058, 0.014, 0.810])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT), cax=cax,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("normalised class activation (-)", fontsize=7.0,
                 color="#1a1a1a", labelpad=3.5)


    savefig(fig, "273_gradcam_grid.png")


if __name__ == "__main__":
    main()
