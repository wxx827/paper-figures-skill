"""Procedural scene library shared by the WM qualitative figures (1492-1497).

Four figures in this family used to render the same programmatic motorway, so
a client flipping through them could tell they all came out of one function.
This module keeps the shared machinery in one place but gives each scene
*kind* its own skeleton - horizon, focal length, camera height, subject layout
and texture direction all differ - and the per-figure camera dictionary varies
exposure, white balance, contrast and grain on top of that.

    from _wm_scenes import build
    sc = build("tabletop", seed=11)
    sc["rgb"], sc["focus"], sc["anchors"]

``anchors`` names the salient objects in pixel coordinates, so a caller can
put an attention blob, a red box or a failure callout on real content instead
of on a hand-typed guess.

The finishing pass is what turns "blurred" into "photographed": a depth-of-
field blend driven by the scene's own focus map, an edge-preserving
Perona-Malik pass that removes noise without softening boundaries, and an
unsharp mask applied only where the subject is meant to be sharp.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

KINDS = ("road", "tabletop", "indoor", "aerial", "shelf", "ego", "studio",
         "conveyor")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def rgb(c):
    return np.array(mpl.colors.to_rgb(c))


def octave_noise(rng, shape, base=3.0, octaves=4, persistence=0.55):
    """Sum of band-limited Gaussian fields; the texture backbone of every tile."""
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        sigma = max(0.7, min(shape) / (base * (2.0 ** k)))
        n = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
        out += amp * n / (n.std() + 1e-9)
        tot += amp
        amp *= persistence
    return out / tot


def grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    return yy, xx, xx / w, yy / h


def srect(h, w, cx, cy, hw, hh, r=2.0, soft=0.7, ang=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = xx - cx, yy - cy
    if ang:
        ca, sa = np.cos(ang), np.sin(ang)
        dx, dy = dx * ca + dy * sa, -dx * sa + dy * ca
    ex = np.clip(np.abs(dx) - (hw - r), 0, None)
    ey = np.clip(np.abs(dy) - (hh - r), 0, None)
    return gaussian_filter(np.clip(0.5 - (np.hypot(ex, ey) - r), 0, 1), soft)


def sellipse(h, w, cx, cy, rx, ry, soft=0.9):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    d = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2
    return gaussian_filter(np.clip(1.35 - d, 0, 1) ** 0.5, soft)


def paint(img, mask, colour, alpha=1.0):
    m = (mask[..., None] * alpha)
    return img * (1 - m) + np.asarray(colour) * m


def scapsule(h, w, p0, p1, r, soft=0.8):
    """Soft capsule (thick line segment) mask - limbs, fingers, handles."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    bx, by = p1[0] - p0[0], p1[1] - p0[1]
    px, py = xx - p0[0], yy - p0[1]
    tt = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-9), 0.0, 1.0)
    d = np.hypot(px - tt * bx, py - tt * by)
    return gaussian_filter(np.clip(0.5 + r - d, 0, 1), soft)


def rim(mask, width=1.2):
    """Outer halo of a soft mask.

    Section 1.6 of BRIEF_WM: every silhouette needs a dark contour or the
    object reads as a stain rather than a thing.  Multiply the image down by
    this and the primitive suddenly has an edge.
    """
    return np.clip(gaussian_filter(np.asarray(mask, dtype=float), width)
                   - np.asarray(mask, dtype=float), 0, 1)


def ink(img, mask, amount=0.42, width=1.2):
    """Darken the outer contour of ``mask`` in place-ish (returns a copy)."""
    return img * (1 - amount * rim(mask, width)[..., None])


# ---------------------------------------------------------------------------
# Camera / finishing
# ---------------------------------------------------------------------------
def camera(**kw):
    # ``kappa`` is the edge threshold of the diffusion pass: gradients above it
    # survive.  Too generous and fine texture (window grids, book spines) is
    # read as noise and wiped, which is how a scene turns into cardboard.
    #
    # ``sat`` was added for the first-person family: a real head-mounted
    # camera indoors is strongly tungsten, and the batch colour budget caps
    # warm / high-chroma area at 8 percent.  Pulling chroma out in the camera
    # (rather than desaturating every material by hand) keeps the scene code
    # readable.  It defaults to 1.0, so every pre-existing preset is untouched.
    cam = dict(exposure=1.00, wb=(1.00, 1.00, 1.00), contrast=1.08,
               vignette=0.26, grain=0.009, dof=2.0, sharpen=0.90, smooth=2,
               kappa=0.038, sat=1.00)
    cam.update(kw)
    return cam


def perona_malik(img, iters=2, kappa=0.055, gamma=0.20):
    """Edge-preserving diffusion: kills noise, leaves boundaries standing."""
    out = np.asarray(img, dtype=float).copy()
    for _ in range(int(iters)):
        d = [np.roll(out, 1, 0) - out, np.roll(out, -1, 0) - out,
             np.roll(out, -1, 1) - out, np.roll(out, 1, 1) - out]
        out = out + gamma * sum(np.exp(-(x / kappa) ** 2) * x for x in d)
    return out


def depth_of_field(img, focus, sigma=2.0):
    """Two-tap defocus: the further a pixel is from the focal plane, the softer."""
    if sigma <= 0:
        return img
    b1 = np.dstack([gaussian_filter(img[..., c], 0.42 * sigma) for c in range(3)])
    b2 = np.dstack([gaussian_filter(img[..., c], sigma) for c in range(3)])
    w = np.clip(1.0 - focus, 0, 1)[..., None]
    near = img * (1 - np.clip(2 * w, 0, 1)) + b1 * np.clip(2 * w, 0, 1)
    far = np.clip(2 * (w - 0.5), 0, 1)
    return near * (1 - far) + b2 * far


def unsharp(img, focus, sigma=1.0, amount=0.9):
    if amount <= 0:
        return img
    lo = np.dstack([gaussian_filter(img[..., c], sigma) for c in range(3)])
    return img + amount * focus[..., None] * (img - lo)


def finish(img, focus, cam, rng):
    h, w = img.shape[:2]
    _, _, u, v = grid(h, w)
    out = depth_of_field(img, focus, cam["dof"])
    out = perona_malik(out, iters=cam["smooth"], kappa=cam["kappa"])
    out = unsharp(out, focus, amount=cam["sharpen"])
    out = out * np.asarray(cam["wb"])
    s = cam.get("sat", 1.0)
    if s != 1.0:
        lum = (out @ np.array([0.30, 0.59, 0.11]))[..., None]
        out = lum + (out - lum) * s
    out = (out - 0.5) * cam["contrast"] + 0.5
    out = out * cam["exposure"]
    vig = 1.0 - cam["vignette"] * (((u - 0.5) * 1.85) ** 2 + ((v - 0.5) * 1.4) ** 2)
    out = out * vig[..., None] + cam["grain"] * rng.standard_normal(out.shape)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Scene: road  (first-person dash camera, wide lens, cool daylight)
# ---------------------------------------------------------------------------
def _road(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    hz = p.get("horizon", 0.455)
    vp = p.get("vpoint", 0.50) * w

    s = np.clip(v / hz, 0, 1)[..., None]
    sky = np.array([0.40, 0.51, 0.66]) * (1 - s) + np.array([0.82, 0.83, 0.80]) * s
    sky = sky + 0.10 * np.clip(octave_noise(rng, (h, w), 2.2, 3),
                               0, None)[..., None] * (1 - s)
    g = np.clip((v - hz) / (1 - hz), 1e-3, 1)
    road = np.array([0.30, 0.30, 0.32]) * (0.78 + 0.95 * g[..., None])
    road = road + 0.045 * octave_noise(rng, (h, w), 6.0, 4)[..., None]
    img = np.where((v < hz)[..., None], sky, road)

    sw = p.get("block", 13)
    ns = w // sw + 1
    es = np.abs(np.linspace(-1, 1, ns))
    hs = (0.02 + 0.31 * es ** 1.7) * (0.55 + 0.80 * rng.random(ns))
    sid = np.minimum(np.arange(w) // sw, ns - 1)
    top = hz - np.clip(hs, 0.008, 0.42)[sid]
    tone = rng.uniform(0.40, 0.66, ns)[sid]
    lit = (rng.random((h // 3 + 3, w // 3 + 3)) > 0.52).astype(float)
    win = lit[(yy // 3).astype(int) % lit.shape[0],
              (xx // 3).astype(int) % lit.shape[1]]
    win = win * (((xx.astype(int) % 3) < 2) & ((yy.astype(int) % 3) < 2))
    face = tone[None, :, None] * np.ones((h, 1, 3)) * np.array([1.05, 1.0, 0.93])
    face = face * (0.86 + 0.26 * win[..., None])
    face = face * (1.0 - 0.30 * ((xx.astype(int) % sw) < 1))[..., None]
    face = face + 0.028 * octave_noise(rng, (h, w), 9.0, 3)[..., None]
    bmask = gaussian_filter(((v < hz) & (v > top[None, :])).astype(float), 0.6)
    img = img * (1 - bmask[..., None]) + face * bmask[..., None]

    z = 0.40 / (g + 0.02)
    halfw = (0.010 + 0.95 * g) * w
    for sgn in (-1.0, 1.0):
        wdt = 0.45 + 2.4 * g
        for frac, dashed in ((0.33, True), (0.97, False)):
            m = (np.abs(xx - (vp + sgn * frac * halfw)) <
                 wdt * (1.6 if dashed else 1.0)) & (v > hz + 0.008)
            if dashed:
                m = m & ((((z * 0.75) + 2.4 * t) % 1.0) < 0.44)
            img = paint(img, gaussian_filter(m.astype(float), 0.55),
                        rgb("#d8d5c6"), 0.90)

    anchors, focus = [], np.zeros((h, w))
    cars = [(0.34, 0.78, "#41505f"), (0.66, 0.56, "#6b4a3c"),
            (0.53, 0.96, "#3f4650")]
    for (lx, d0, col) in cars:
        d = np.clip(d0 - 0.32 * t, 0.12, 1.0)
        sc = (1.0 - d) ** 1.30
        cy = hz * h + (0.06 + 0.80 * sc) * (1 - hz) * h
        cx = vp + (lx - 0.5) * w * (0.24 + 1.30 * sc)
        bw, bh = 4.5 + 24.0 * sc, 2.2 + 8.6 * sc
        base = rgb(col)
        shade = 0.80 + 0.55 * np.clip((v - hz) / (1 - hz), 0, 1)[..., None]
        img = img * (1 - 0.50 * sellipse(h, w, cx, cy + 1.05 * bh, 1.15 * bw,
                                         0.40 * bh, 1.6)[..., None])
        body = srect(h, w, cx, cy, bw, bh, r=0.30 * bh)[..., None]
        img = img * (1 - body) + base * shade * body
        cab = srect(h, w, cx, cy - 1.45 * bh, 0.66 * bw, 0.72 * bh,
                    r=0.30 * bh)[..., None]
        img = img * (1 - cab) + base * shade * 1.05 * cab
        glass = srect(h, w, cx, cy - 1.55 * bh, 0.55 * bw, 0.40 * bh,
                      r=0.2 * bh, soft=0.6)[..., None]
        img = img * (1 - glass) + rgb("#8fa2b0") * 0.75 * glass
        for ws in (-1.0, 1.0):
            wh = srect(h, w, cx + ws * 0.72 * bw, cy + 0.85 * bh,
                       0.16 * bw, 0.34 * bh, r=0.2 * bh)[..., None]
            img = img * (1 - 0.85 * wh) + rgb("#22252a") * 0.85 * wh
        img = img + 0.30 * sellipse(h, w, cx - 0.70 * bw, cy - 0.15 * bh,
                                    0.13 * bw, 0.20 * bh, 0.7)[..., None] * \
            np.array([1.0, 0.86, 0.62])
        anchors.append(dict(name="vehicle", cx=cx, cy=cy, rx=1.4 * bw, ry=2.2 * bh))
        focus = np.maximum(focus, sellipse(h, w, cx, cy, 2.6 * bw, 3.4 * bh, 3.0))

    band = np.exp(-0.5 * ((v - (hz + 0.30)) / 0.20) ** 2)
    skyline = np.exp(-0.5 * ((v - (hz - 0.07)) / 0.11) ** 2)
    focus = np.clip(np.maximum.reduce([focus, 0.85 * band, 0.68 * skyline]),
                    0, 1)
    return img, focus, anchors


# ---------------------------------------------------------------------------
# Scene: tabletop  (short lens looking down at a desk, many discrete objects)
# ---------------------------------------------------------------------------
def _tabletop(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    hz = p.get("horizon", 0.34)

    wall = np.array([0.62, 0.60, 0.57]) * (0.88 + 0.30 * (v / hz)[..., None])
    wall = wall + 0.035 * octave_noise(rng, (h, w), 4.0, 3)[..., None]
    g = np.clip((v - hz) / (1 - hz), 1e-3, 1)
    top = np.array([0.46, 0.36, 0.27]) * (0.80 + 0.70 * g[..., None])
    lines = 0.5 + 0.5 * np.sin((xx - 0.5 * w) / (2.6 + 7.0 * g) + 3.0 * g)
    top = top * (0.93 + 0.11 * lines[..., None])
    top = top + 0.035 * octave_noise(rng, (h, w), 7.0, 4)[..., None]
    img = np.where((v < hz)[..., None], wall, top)
    img = paint(img, gaussian_filter((np.abs(v - hz) < 0.008).astype(float), 0.7),
                np.array([0.30, 0.26, 0.22]), 0.55)
    sheen = np.exp(-0.5 * ((u - 0.30) / 0.26) ** 2) * np.clip(g, 0, 1)
    img = img + 0.10 * sheen[..., None] * np.array([1.0, 0.97, 0.90])

    # (name, u, depth 0..1 near=1, kind, colour)
    props = [("bottle", 0.845, 0.30, "bottle", "#4c6b63"),
             ("laptop", 0.315, 0.44, "laptop", "#54595f"),
             ("mug", 0.620, 0.60, "mug", "#8d5f4a"),
             ("bowl", 0.150, 0.70, "bowl", "#7d8790"),
             ("plate", 0.760, 0.76, "plate", "#9aa0a4"),
             ("book", 0.430, 0.88, "book", "#6a5570"),
             ("pen", 0.640, 0.955, "pen", "#31363c")]
    anchors, focus = [], np.zeros((h, w))
    for (name, ux, d, kind, col) in sorted(props, key=lambda r: r[2]):
        s = 0.85 + 2.50 * d
        cy = (hz + 0.055 + (0.80 - hz) * d) * h
        cx = (0.5 + (ux - 0.5) * (0.55 + 0.75 * d)) * w
        base = rgb(col)
        img = img * (1 - 0.42 * sellipse(h, w, cx, cy + 3.0 * s, 9.0 * s,
                                         2.6 * s, 1.5)[..., None])
        if kind == "bottle":
            body = srect(h, w, cx, cy - 7.0 * s, 3.1 * s, 8.0 * s, r=1.6 * s)
            neck = srect(h, w, cx, cy - 16.0 * s, 1.2 * s, 4.0 * s, r=0.6 * s)
            m = np.clip(body + neck, 0, 1)
            rx, ry = 4.0 * s, 18.0 * s
        elif kind == "laptop":
            base_m = srect(h, w, cx, cy - 1.4 * s, 9.5 * s, 1.5 * s, r=0.7 * s)
            scr = srect(h, w, cx + 0.7 * s, cy - 9.5 * s, 8.6 * s, 6.6 * s,
                        r=0.9 * s, ang=-0.05)
            m = np.clip(base_m + scr, 0, 1)
            rx, ry = 10.0 * s, 12.0 * s
        elif kind == "mug":
            body = srect(h, w, cx, cy - 3.6 * s, 3.4 * s, 4.0 * s, r=1.1 * s)
            handle = (sellipse(h, w, cx + 4.4 * s, cy - 3.6 * s, 2.0 * s, 2.0 * s)
                      - sellipse(h, w, cx + 4.2 * s, cy - 3.6 * s, 1.0 * s,
                                 1.0 * s))
            m = np.clip(body + np.clip(handle, 0, 1), 0, 1)
            rx, ry = 5.6 * s, 5.6 * s
        elif kind == "bowl":
            m = sellipse(h, w, cx, cy - 2.0 * s, 6.0 * s, 3.0 * s, 0.8)
            rx, ry = 6.6 * s, 3.8 * s
        elif kind == "plate":
            m = sellipse(h, w, cx, cy - 0.6 * s, 6.4 * s, 2.4 * s, 0.8)
            rx, ry = 7.0 * s, 3.2 * s
        elif kind == "book":
            m = srect(h, w, cx, cy - 1.4 * s, 8.6 * s, 2.2 * s, r=0.5 * s,
                      ang=0.16)
            rx, ry = 9.2 * s, 3.4 * s
        else:                                     # pen
            m = srect(h, w, cx, cy - 0.8 * s, 6.0 * s, 0.7 * s, r=0.35 * s,
                      ang=-0.32)
            rx, ry = 6.4 * s, 2.0 * s
        lit = 0.92 + 0.34 * np.exp(-0.5 * ((xx - (cx - 2.4 * s)) /
                                           (4.0 * s)) ** 2)
        img = img * (1 - m[..., None]) + base[None, None, :] * lit[..., None] \
            * m[..., None]
        if kind == "laptop":
            scr_in = srect(h, w, cx + 0.7 * s, cy - 9.6 * s, 7.4 * s, 5.5 * s,
                           r=0.5 * s, ang=-0.05)
            img = paint(img, scr_in, rgb("#9fb2c4") * 0.92, 0.95)
        if kind in ("bowl", "plate"):
            img = paint(img, sellipse(h, w, cx, cy - 2.4 * s, 0.62 * rx,
                                      0.55 * ry, 0.8), base * 0.78, 0.9)
        anchors.append(dict(name=name, cx=cx, cy=cy - 0.35 * ry, rx=rx, ry=ry))
        focus = np.maximum(focus, sellipse(h, w, cx, cy - 0.3 * ry,
                                           2.0 * rx, 2.0 * ry, 2.5))
    focus = np.clip(np.maximum(focus, 0.55 * np.clip((v - hz) / 0.35, 0, 1) *
                               np.clip((0.95 - v) / 0.25, 0, 1)), 0, 1)
    return img, focus, anchors


# ---------------------------------------------------------------------------
# Scene: indoor  (living room, frontal, low camera, warm bounce light)
# ---------------------------------------------------------------------------
def _indoor(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    hz = p.get("horizon", 0.54)

    wall = np.array([0.60, 0.585, 0.55]) * (1.02 - 0.16 * (v / hz)[..., None])
    wall = wall + 0.030 * octave_noise(rng, (h, w), 4.0, 3)[..., None]
    g = np.clip((v - hz) / (1 - hz), 1e-3, 1)
    floor = np.array([0.40, 0.33, 0.27]) * (0.85 + 0.55 * g[..., None])
    floor = floor * (0.95 + 0.09 * (0.5 + 0.5 * np.sin(
        (xx - 0.5 * w) / (3.0 + 9.0 * g)))[..., None])
    floor = floor + 0.030 * octave_noise(rng, (h, w), 7.0, 3)[..., None]
    img = np.where((v < hz)[..., None], wall, floor)

    anchors, focus = [], np.zeros((h, w))
    wx, wy = 0.74 * w, 0.26 * h
    win = srect(h, w, wx, wy, 0.115 * w, 0.155 * h, r=0.010 * w)
    img = paint(img, win, np.array([0.97, 0.97, 0.94]), 0.98)
    img = paint(img, srect(h, w, wx, wy, 0.122 * w, 0.163 * h, r=0.010 * w)
                - win, rgb("#7c7873"), 0.9)
    for k in (-1, 1):
        img = paint(img, srect(h, w, wx + k * 0.058 * w, wy, 0.004 * w,
                               0.150 * h), rgb("#6e6a66"), 0.8)
    img = img + 0.16 * sellipse(h, w, wx, wy + 0.30 * h, 0.30 * w,
                                0.26 * h, 4.0)[..., None] * \
        np.array([1.0, 0.98, 0.92])
    anchors.append(dict(name="window", cx=wx, cy=wy, rx=0.15 * w, ry=0.20 * h))

    fx, fy = 0.22 * w, 0.22 * h
    frame = srect(h, w, fx, fy, 0.075 * w, 0.095 * h, r=0.006 * w)
    img = paint(img, frame, rgb("#4d443c"), 0.95)
    img = paint(img, srect(h, w, fx, fy, 0.062 * w, 0.080 * h, r=0.004 * w),
                np.array([0.55, 0.58, 0.56]), 0.9)
    anchors.append(dict(name="picture", cx=fx, cy=fy, rx=0.10 * w, ry=0.12 * h))

    sx, sy = 0.42 * w, 0.74 * h
    img = img * (1 - 0.40 * sellipse(h, w, sx, sy + 0.11 * h, 0.30 * w,
                                     0.05 * h, 2.0)[..., None])
    seat = srect(h, w, sx, sy, 0.26 * w, 0.085 * h, r=0.020 * w)
    backr = srect(h, w, sx, sy - 0.115 * h, 0.26 * w, 0.070 * h, r=0.020 * w)
    img = paint(img, backr, rgb("#5b6672") * 0.90, 1.0)
    img = paint(img, seat, rgb("#5b6672"), 1.0)
    for k in (-1, 1):                              # armrests
        img = paint(img, srect(h, w, sx + k * 0.245 * w, sy - 0.030 * h,
                               0.032 * w, 0.070 * h, r=0.014 * w),
                    rgb("#66717d"), 1.0)
    for k in (-1, 0, 1):                           # back cushions
        img = paint(img, srect(h, w, sx + k * 0.140 * w, sy - 0.100 * h,
                               0.062 * w, 0.042 * h, r=0.016 * w),
                    rgb("#69747f"), 0.95)
    anchors.append(dict(name="sofa", cx=sx, cy=sy - 0.04 * h, rx=0.30 * w,
                        ry=0.16 * h))
    focus = np.maximum(focus, sellipse(h, w, sx, sy - 0.04 * h, 0.46 * w,
                                       0.30 * h, 3.0))

    lx = 0.88 * w
    img = paint(img, srect(h, w, lx, 0.60 * h, 0.005 * w, 0.16 * h),
                rgb("#3f3a35"), 0.9)
    shade = srect(h, w, lx, 0.415 * h, 0.048 * w, 0.045 * h, r=0.014 * w)
    img = paint(img, shade, np.array([0.86, 0.82, 0.72]), 0.95)
    img = img + 0.20 * sellipse(h, w, lx, 0.46 * h, 0.14 * w, 0.12 * h,
                                3.5)[..., None] * np.array([1.0, 0.92, 0.78])
    anchors.append(dict(name="lamp", cx=lx, cy=0.44 * h, rx=0.09 * w,
                        ry=0.13 * h))

    rug = sellipse(h, w, 0.46 * w, 0.94 * h, 0.40 * w, 0.10 * h, 1.6)
    img = paint(img, rug, rgb("#7a6a5c"), 0.55)
    focus = np.clip(np.maximum(focus, 0.5 * np.exp(
        -0.5 * ((v - 0.70) / 0.22) ** 2)), 0, 1)
    return img, focus, anchors


# ---------------------------------------------------------------------------
# Scene: aerial  (drone looking down at a landscape; deep depth of field)
# ---------------------------------------------------------------------------
def _aerial(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    pan = p.get("pan", 34.0) * t
    ys = yy + pan
    persp = 0.72 + 0.55 * v                       # slight oblique compression

    field = gaussian_filter(rng.standard_normal((h + 160, w)), 7.0)
    field = field / (field.std() + 1e-9)
    off = int(pan) % 120
    fld = field[off:off + h]
    tone = np.digitize(fld, [-0.85, -0.25, 0.25, 0.85]) / 4.0
    greens = np.array([[0.34, 0.36, 0.27], [0.40, 0.43, 0.30],
                       [0.47, 0.46, 0.35], [0.36, 0.40, 0.31],
                       [0.44, 0.41, 0.32]])
    img = greens[np.clip((tone * 4).astype(int), 0, 4)]
    img = img * (0.90 + 0.22 * octave_noise(rng, (h, w), 9.0, 4)[..., None])
    img = img * (0.86 + 0.28 * persp[..., None])

    # a curving road ribbon plus a water strip
    cx_road = 0.50 * w + 0.16 * w * np.sin((ys / h) * 2.4 + 0.6)
    road = np.abs(xx - cx_road) < (0.036 * w) * persp
    img = paint(img, gaussian_filter(road.astype(float), 0.7),
                np.array([0.33, 0.33, 0.34]), 0.95)
    dash = road & (np.abs(xx - cx_road) < 0.004 * w) & (((ys / 7.0) % 1.0) < 0.5)
    img = paint(img, gaussian_filter(dash.astype(float), 0.5),
                np.array([0.82, 0.80, 0.70]), 0.85)
    water = np.abs(ys - (0.80 * h + 0.10 * h * np.sin(xx / w * 3.1))) < 0.035 * h
    img = paint(img, gaussian_filter(water.astype(float), 0.9),
                np.array([0.30, 0.40, 0.50]), 0.92)

    anchors, focus = [], np.zeros((h, w))
    rs = np.random.default_rng(int(p.get("plan", 7)))
    for k in range(7):
        bx = rs.uniform(0.08, 0.92) * w
        by = (rs.uniform(0.05, 0.95) * h - pan * 0.6) % h
        bw_, bh_ = rs.uniform(0.035, 0.085) * w, rs.uniform(0.030, 0.070) * h
        if abs(bx - (0.50 * w + 0.16 * w * np.sin((by + pan) / h * 2.4 + 0.6))) \
                < 0.07 * w:
            bx += 0.16 * w * (1 if bx < 0.5 * w else -1)
        ang = rs.uniform(-0.25, 0.25)
        m = srect(h, w, bx, by, bw_, bh_, r=0.006 * w, ang=ang)
        sh = srect(h, w, bx + 0.022 * w, by + 0.028 * h, bw_, bh_,
                   r=0.006 * w, ang=ang)
        img = img * (1 - 0.45 * np.clip(sh - m, 0, 1)[..., None])
        roof = rs.uniform(0.46, 0.68)
        img = paint(img, m, np.array([roof, roof * 0.95, roof * 0.88]), 1.0)
        img = paint(img, np.clip(m - srect(h, w, bx, by, 0.86 * bw_,
                                           0.84 * bh_, r=0.004 * w, ang=ang),
                                 0, 1), np.array([0.30, 0.28, 0.26]), 0.55)
        anchors.append(dict(name="building", cx=bx, cy=by, rx=1.5 * bw_,
                            ry=1.6 * bh_))
        focus = np.maximum(focus, sellipse(h, w, bx, by, 3.0 * bw_, 3.0 * bh_,
                                           2.5))
    for k in range(14):
        tx, ty = rs.uniform(0.03, 0.97) * w, (rs.uniform(0, 1) * h - pan) % h
        r_ = rs.uniform(0.012, 0.026) * w
        img = img * (1 - 0.35 * sellipse(h, w, tx + 0.012 * w, ty + 0.014 * h,
                                         r_, 0.8 * r_, 1.1)[..., None])
        img = paint(img, sellipse(h, w, tx, ty, r_, 0.85 * r_, 0.9),
                    np.array([0.22, 0.30, 0.21]), 0.92)
    car_cols = ["#9aa6b2", "#b0b4b8", "#8d5f4a", "#59636e"]
    for k in range(4):
        vy = ((0.18 + 0.24 * k) * h - pan * 1.6) % h
        vx = 0.50 * w + 0.16 * w * np.sin((vy + pan) / h * 2.4 + 0.6)
        img = img * (1 - 0.30 * srect(h, w, vx + 0.006 * w, vy + 0.008 * h,
                                      0.008 * w, 0.020 * h,
                                      r=0.003 * w)[..., None])
        img = paint(img, srect(h, w, vx, vy, 0.008 * w, 0.020 * h, r=0.003 * w),
                    rgb(car_cols[k % 4]), 0.95)
        anchors.append(dict(name="car", cx=vx, cy=vy, rx=0.022 * w,
                            ry=0.034 * h))
    focus = np.clip(np.maximum(focus, 0.80 * np.clip((v - 0.05) / 0.30, 0, 1)),
                    0, 1)
    return img, focus, anchors


# ---------------------------------------------------------------------------
# Scene: shelf  (frontal close-up, strong horizontal / vertical structure)
# ---------------------------------------------------------------------------
def _shelf(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    img = np.ones((h, w, 3)) * np.array([0.30, 0.25, 0.21])
    img = img * (0.88 + 0.26 * octave_noise(rng, (h, w), 5.0, 3)[..., None])

    anchors, focus = [], np.zeros((h, w))
    nsh = p.get("shelves", 3)
    for s in range(nsh):
        y0 = (0.10 + 0.86 * (s + 1) / nsh) * h
        img = paint(img, srect(h, w, 0.5 * w, y0, 0.52 * w, 0.016 * h,
                               r=0.004 * w), np.array([0.48, 0.40, 0.32]), 0.98)
        img = img * (1 - 0.30 * srect(h, w, 0.5 * w, y0 - 0.055 * h, 0.52 * w,
                                      0.040 * h, r=0.006 * w, soft=2.0)[..., None])
        x = 0.055 * w
        rs = np.random.default_rng(int(p.get("plan", 3)) * 17 + s)
        while x < 0.94 * w:
            bw_ = rs.uniform(0.012, 0.034) * w
            bh_ = rs.uniform(0.085, 0.135) * h
            ang = rs.uniform(-0.06, 0.06) if rs.random() < 0.18 else 0.0
            col = rs.choice(["#6b4a3c", "#3f5566", "#5d6b52", "#7a5f70",
                             "#8a7a53", "#44484d"])
            cy = y0 - 0.012 * h - bh_ / 2
            m = srect(h, w, x + bw_ / 2, cy, bw_ / 2, bh_ / 2, r=0.002 * w,
                      ang=ang)
            img = paint(img, m, rgb(col) * rs.uniform(0.86, 1.12), 1.0)
            img = paint(img, np.clip(m - srect(h, w, x + bw_ / 2, cy,
                                               0.34 * bw_, 0.46 * bh_,
                                               r=0.002 * w, ang=ang), 0, 1),
                        rgb(col) * 0.72, 0.55)
            if bw_ > 0.024 * w:
                img = paint(img, srect(h, w, x + bw_ / 2, cy - 0.22 * bh_,
                                       0.26 * bw_, 0.012 * h, r=0.001 * w),
                            np.array([0.86, 0.83, 0.74]), 0.75)
            anchors.append(dict(name="book", cx=x + bw_ / 2, cy=cy,
                                rx=1.2 * bw_, ry=0.6 * bh_))
            x += bw_ + rs.uniform(0.002, 0.008) * w
        if s == nsh // 2:
            focus = np.maximum(focus, np.exp(-0.5 * ((v - y0 / h + 0.06) /
                                                     0.16) ** 2))
    focus = np.clip(np.maximum(focus, 0.35), 0, 1)
    return img, focus, anchors


# ---------------------------------------------------------------------------
# Hands  (shared by the ego skeleton and by every overlay drawn on top of it)
# ---------------------------------------------------------------------------
# MANO joint order: 0 wrist, then thumb / index / middle / ring / pinky, four
# joints each, proximal to distal.  The *same* array both paints the hand into
# the raster and positions the keypoint overlay a figure draws afterwards, so
# the skeleton can never drift off the pixels it is supposed to annotate.
HAND_NAMES = ("wrist", "thumb", "index", "middle", "ring", "pinky")
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_TIPS = (4, 8, 12, 16, 20)

# canonical right hand, wrist at the origin, fingers pointing up-screen (-y),
# in units of the hand scale
_HAND_BASE = {
    "thumb": [(-0.40, -0.18), (-0.64, -0.44), (-0.78, -0.66), (-0.86, -0.86)],
    "index": [(-0.30, -0.96), (-0.34, -1.32), (-0.36, -1.56), (-0.37, -1.74)],
    "middle": [(-0.04, -1.02), (-0.05, -1.42), (-0.06, -1.68), (-0.06, -1.88)],
    "ring": [(0.21, -0.98), (0.24, -1.34), (0.26, -1.57), (0.27, -1.75)],
    "pinky": [(0.43, -0.88), (0.49, -1.17), (0.52, -1.36), (0.54, -1.51)],
}


def hand_joints(cx, cy, s, ang=0.0, curl=0.0, side="r"):
    """21 keypoints of one hand, in pixels.

    ``curl`` in 0..1 folds the four fingers towards the palm (0 = flat, 1 =
    fist), ``ang`` rolls the whole hand, ``side`` mirrors it.
    """
    sgn = -1.0 if side == "l" else 1.0
    out = [np.array([0.0, 0.0])]
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        pts = [np.array(q, dtype=float) for q in _HAND_BASE[name]]
        c = curl * (0.55 if name == "thumb" else 1.0)
        acc = 0.0
        chain = [pts[0]]
        for i in range(1, len(pts)):
            seg = pts[i] - pts[i - 1]
            acc += c * (0.60 + 0.18 * i)
            ca, sa = np.cos(acc), np.sin(acc)
            chain.append(chain[-1] + np.array([seg[0] * ca - seg[1] * sa,
                                               seg[0] * sa + seg[1] * ca]))
        out.extend(chain)
    P = np.asarray(out)
    P[:, 0] *= sgn
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]])
    return (P * s) @ R.T + np.array([cx, cy])


def draw_hand(img, J, s, skin, forearm=None, light=1.0):
    """Paint a hand described by ``J`` (21x2 px) into ``img``."""
    h, w = img.shape[:2]
    m = np.zeros((h, w))
    if forearm is not None:
        m = np.maximum(m, scapsule(h, w, forearm, J[0], 0.34 * s, 1.0))
    for tip in (5, 9, 13, 17):                      # palm web
        m = np.maximum(m, scapsule(h, w, J[0], J[tip], 0.27 * s, 0.9))
    m = np.maximum(m, scapsule(h, w, J[5], J[17], 0.24 * s, 0.9))
    for a, b in HAND_BONES:
        if a == 0:
            continue
        r = 0.155 * s if b % 4 == 2 else (0.135 * s if b % 4 == 3
                                          else 0.115 * s)
        m = np.maximum(m, scapsule(h, w, J[a], J[b], r, 0.8))
    m = np.clip(m, 0, 1)
    shadow = gaussian_filter(m, 3.2) * 0.5
    out = img * (1 - 0.42 * shadow[..., None])
    lit = light * (1.06 - 0.34 * (np.mgrid[0:h, 0:w][0] / h))
    out = out * (1 - m[..., None]) + np.asarray(skin)[None, None, :] * \
        lit[..., None] * m[..., None]
    return ink(out, m, 0.46, 1.3), m


# ---------------------------------------------------------------------------
# Scene: ego  (head-mounted first person: work surface, props, hands entering
# from the bottom of the frame).  ``room`` swaps the whole set dressing, so
# four figures can each get their own space instead of four seeds of one.
# ---------------------------------------------------------------------------
_EGO_ROOMS = {
    "kitchen": dict(
        horizon=0.385, wall="#5e666d", surf="#7d8081", back="tile",
        skin="#9d9089",
        # (name, u, depth 0..1 near=1, family, colour, size)
        props=[("pan", 0.330, 0.46, "pan", "#3f454c", 1.20),
               ("kettle", 0.800, 0.28, "cyl", "#828a94", 1.05),
               ("board", 0.520, 0.76, "slab", "#94918a", 1.45),
               ("jar", 0.075, 0.38, "cyl", "#556468", 0.85),
               ("bowl", 0.930, 0.58, "disc", "#8f959b", 0.95)],
        hands=[dict(u=0.330, v=1.00, s=0.150, ang=-0.30, curl=0.30, side="l"),
               dict(u=0.700, v=1.04, s=0.155, ang=0.26, curl=0.20, side="r")],
    ),
    "desk": dict(
        horizon=0.300, wall="#6a7076", surf="#565b60", back="pinboard",
        skin="#968b84",
        props=[("monitor", 0.450, 0.20, "panel", "#383e45", 1.35),
               ("keyboard", 0.475, 0.78, "slab", "#5f656c", 1.60),
               ("mug", 0.845, 0.56, "cyl", "#71808d", 0.85),
               ("notebook", 0.095, 0.62, "slab", "#868b91", 1.15),
               ("phone", 0.780, 0.84, "slab", "#42474d", 0.60)],
        hands=[dict(u=0.345, v=1.03, s=0.140, ang=-0.16, curl=0.42, side="l"),
               dict(u=0.640, v=1.05, s=0.142, ang=0.12, curl=0.38, side="r")],
    ),
    "workbench": dict(
        horizon=0.430, wall="#565c62", surf="#75726c", back="pegboard",
        skin="#9a8e85",
        props=[("drill", 0.600, 0.56, "tool", "#454c55", 1.45),
               ("crate", 0.135, 0.42, "box", "#6d747d", 1.15),
               ("clamp", 0.880, 0.34, "tool", "#7e848b", 0.95),
               ("washer", 0.370, 0.72, "disc", "#8d9299", 0.75),
               ("tin", 0.945, 0.62, "cyl", "#5a666b", 0.90)],
        hands=[dict(u=0.520, v=0.99, s=0.160, ang=-0.46, curl=0.58, side="l"),
               dict(u=0.840, v=1.07, s=0.152, ang=0.20, curl=0.16, side="r")],
    ),
    "lab": dict(
        horizon=0.365, wall="#616a72", surf="#7a8083", back="shelfstrip",
        skin="#948a83",
        props=[("module", 0.235, 0.50, "box", "#5b636d", 1.05),
               ("beaker", 0.845, 0.34, "cyl", "#7c8b95", 0.90),
               ("tray", 0.505, 0.74, "slab", "#8d9298", 1.50),
               ("probe", 0.665, 0.60, "tool", "#3f464e", 1.30),
               ("dish", 0.070, 0.66, "disc", "#909599", 0.80)],
        hands=[dict(u=0.780, v=1.05, s=0.150, ang=0.34, curl=0.24, side="r")],
    ),
    "store": dict(
        horizon=0.615, wall="#666d74", surf="#767a7d", back="aisle",
        skin="#968c84",
        props=[("carton", 0.310, 0.44, "box", "#69717b", 1.00),
               ("can", 0.560, 0.52, "cyl", "#5f6d73", 0.85),
               ("packet", 0.775, 0.62, "slab", "#8a8f95", 1.05),
               ("jar", 0.905, 0.40, "cyl", "#78828c", 0.80),
               ("tub", 0.115, 0.58, "disc", "#8d9298", 0.85)],
        hands=[dict(u=0.615, v=1.12, s=0.140, ang=0.12, curl=0.26, side="r")],
    ),
    "living": dict(
        horizon=0.545, wall="#6c7278", surf="#635f5b", back="livingwall",
        skin="#978d85",
        props=[("tray", 0.465, 0.62, "slab", "#868b91", 1.40),
               ("cup", 0.680, 0.70, "cyl", "#71808b", 0.85),
               ("book", 0.240, 0.78, "slab", "#585f67", 1.10),
               ("remote", 0.780, 0.86, "tool", "#43484f", 0.75),
               ("vase", 0.945, 0.50, "cyl", "#7d868f", 1.00)],
        hands=[],
    ),
}


def _prop(h, w, family, cx, cy, s, ang=0.0):
    """Masks for one prop family; ``cy`` is where the object meets the surface.

    Returns ``(body, detail, highlight, centre_y, rx, ry)``.
    """
    if family == "cyl":
        bh = 2.5 * s
        body = np.clip(srect(h, w, cx, cy - bh / 2, 0.78 * s, bh / 2,
                             r=0.20 * s)
                       + sellipse(h, w, cx, cy - bh, 0.78 * s, 0.26 * s, 0.7),
                       0, 1)
        det = sellipse(h, w, cx, cy - bh, 0.54 * s, 0.17 * s, 0.7)
        hi = srect(h, w, cx - 0.42 * s, cy - 0.62 * bh, 0.10 * s, 0.60 * bh,
                   r=0.05 * s, soft=0.9)
        return body, det, hi, cy - 1.25 * s, 1.05 * s, 1.45 * s
    if family in ("disc", "pan"):
        body = sellipse(h, w, cx, cy - 0.34 * s, 1.52 * s, 0.56 * s, 0.8)
        if family == "pan":
            body = np.clip(body + scapsule(
                h, w, (cx + 1.35 * s, cy - 0.42 * s),
                (cx + 3.20 * s, cy - 0.62 * s), 0.18 * s, 0.8), 0, 1)
        det = sellipse(h, w, cx, cy - 0.40 * s, 1.12 * s, 0.38 * s, 0.8)
        hi = sellipse(h, w, cx - 0.55 * s, cy - 0.52 * s, 0.42 * s, 0.13 * s,
                      0.8)
        return body, det, hi, cy - 0.34 * s, 1.75 * s, 0.85 * s
    if family == "slab":
        body = srect(h, w, cx, cy - 0.34 * s, 1.95 * s, 0.34 * s, r=0.12 * s,
                     ang=ang)
        det = srect(h, w, cx, cy - 0.50 * s, 1.72 * s, 0.13 * s, r=0.06 * s,
                    ang=ang)
        hi = srect(h, w, cx, cy - 0.62 * s, 1.80 * s, 0.05 * s, r=0.02 * s,
                   ang=ang)
        return body, det, hi, cy - 0.36 * s, 2.10 * s, 0.62 * s
    if family == "panel":
        scr = srect(h, w, cx, cy - 2.05 * s, 2.30 * s, 1.42 * s, r=0.10 * s)
        stand = np.clip(srect(h, w, cx, cy - 0.50 * s, 0.16 * s, 0.42 * s,
                              r=0.05 * s)
                        + srect(h, w, cx, cy - 0.10 * s, 0.85 * s, 0.11 * s,
                                r=0.05 * s), 0, 1)
        body = np.clip(scr + stand, 0, 1)
        det = srect(h, w, cx, cy - 2.07 * s, 2.10 * s, 1.24 * s, r=0.05 * s)
        hi = srect(h, w, cx - 1.30 * s, cy - 2.05 * s, 0.24 * s, 1.10 * s,
                   r=0.05 * s, soft=1.2)
        return body, det, hi, cy - 1.90 * s, 2.55 * s, 1.90 * s
    if family == "box":
        body = srect(h, w, cx, cy - 0.92 * s, 1.15 * s, 0.92 * s, r=0.08 * s)
        det = srect(h, w, cx, cy - 1.72 * s, 1.15 * s, 0.14 * s, r=0.06 * s)
        hi = srect(h, w, cx - 0.72 * s, cy - 0.95 * s, 0.13 * s, 0.72 * s,
                   r=0.05 * s, soft=1.0)
        return body, det, hi, cy - 0.92 * s, 1.40 * s, 1.20 * s
    # tool: an elongated body with a heavier head
    a = ang if ang else -0.42
    dx, dy = np.cos(a) * 1.25 * s, np.sin(a) * 1.25 * s
    body = np.clip(
        scapsule(h, w, (cx - dx, cy - 0.36 * s - dy),
                 (cx + dx, cy - 0.36 * s + dy), 0.34 * s, 0.8)
        + scapsule(h, w, (cx + 0.55 * dx, cy - 0.36 * s + 0.55 * dy),
                   (cx + dx, cy - 0.36 * s + dy), 0.46 * s, 0.8), 0, 1)
    det = scapsule(h, w, (cx + 0.90 * dx, cy - 0.36 * s + 0.90 * dy),
                   (cx + 1.35 * dx, cy - 0.36 * s + 1.35 * dy), 0.15 * s, 0.7)
    hi = scapsule(h, w, (cx - 0.80 * dx, cy - 0.52 * s - 0.80 * dy),
                  (cx + 0.30 * dx, cy - 0.52 * s + 0.30 * dy), 0.08 * s, 0.8)
    return body, det, hi, cy - 0.40 * s, 1.60 * s, 0.75 * s


def _ego(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    room = dict(_EGO_ROOMS[p.get("room", "kitchen")])
    for key in ("props", "hands", "back", "wall", "surf", "skin"):
        if key in p:
            room[key] = p[key]
    hz = float(p.get("horizon", room["horizon"]))
    wallc, surfc = rgb(room["wall"]), rgb(room["surf"])

    wall = wallc[None, None, :] * (1.12 - 0.36 * (v / max(hz, 1e-3))[..., None])
    wall = wall + 0.028 * octave_noise(rng, (h, w), 4.0, 3)[..., None]
    g = np.clip((v - hz) / (1 - hz), 1e-3, 1)
    surf = surfc[None, None, :] * (0.72 + 0.64 * g[..., None])
    surf = surf + 0.026 * octave_noise(rng, (h, w), 8.0, 4)[..., None]
    lin = 0.5 + 0.5 * np.sin((xx - 0.5 * w) / (2.0 + 10.0 * g) + 2.4 * g)
    surf = surf * (0.968 + 0.052 * lin[..., None])
    img = np.where((v < hz)[..., None], wall, surf)
    img = paint(img, gaussian_filter((np.abs(v - hz) < 0.007).astype(float),
                                     0.7), surfc * 0.40, 0.80)

    back = room["back"]
    if back == "tile":
        tw, th = 0.086 * w, 0.078 * h
        grout = (((xx % tw) < 1.2) | ((yy % th) < 1.2)) & (v < hz)
        img = paint(img, gaussian_filter(grout.astype(float), 0.5),
                    wallc * 1.26, 0.42)
        cab = srect(h, w, 0.50 * w, 0.030 * h, 0.62 * w, 0.105 * h,
                    r=0.008 * w)
        img = paint(img, cab, wallc * 0.70, 0.96)
        img = ink(img, cab, 0.40)
        for k in (-1, 1):
            img = paint(img, srect(h, w, 0.5 * w + k * 0.205 * w, 0.030 * h,
                                   0.0018 * w, 0.100 * h), wallc * 0.44, 0.85)
    elif back == "pegboard":
        dots = (((xx + 0.019 * w) % (0.038 * w)) < 1.7) & \
               (((yy + 0.024 * h) % (0.050 * h)) < 1.7) & (v < hz * 0.97)
        img = paint(img, gaussian_filter(dots.astype(float), 0.45),
                    wallc * 0.58, 0.55)
        for hx, hl, hw_ in ((0.155, 0.20, 0.009), (0.245, 0.14, 0.007),
                            (0.855, 0.24, 0.010)):
            m = srect(h, w, hx * w, (0.035 + hl / 2) * h, hw_ * w, hl / 2 * h,
                      r=0.003 * w)
            img = paint(img, m, wallc * 0.50, 0.96)
            img = ink(img, m, 0.38)
    elif back == "pinboard":
        bd = srect(h, w, 0.660 * w, 0.135 * h, 0.235 * w, 0.115 * h,
                   r=0.007 * w)
        img = paint(img, bd, wallc * 0.74, 0.95)
        img = ink(img, bd, 0.42)
        rs = np.random.default_rng(5)
        for k in range(7):
            m = srect(h, w, rs.uniform(0.46, 0.86) * w,
                      rs.uniform(0.045, 0.225) * h, 0.026 * w, 0.024 * h,
                      r=0.002 * w, ang=rs.uniform(-0.10, 0.10))
            img = paint(img, m, np.array([0.80, 0.81, 0.83]), 0.92)
            img = ink(img, m, 0.30)
    elif back == "shelfstrip":
        y0 = 0.215 * h
        board = srect(h, w, 0.5 * w, y0, 0.60 * w, 0.011 * h, r=0.003 * w)
        img = paint(img, board, wallc * 0.62, 0.96)
        img = ink(img, board, 0.42)
        rs = np.random.default_rng(9)
        x = 0.115 * w
        while x < 0.86 * w:
            bw_ = rs.uniform(0.030, 0.062) * w
            bh_ = rs.uniform(0.055, 0.105) * h
            m = srect(h, w, x + bw_ / 2, y0 - 0.008 * h - bh_ / 2, bw_ / 2,
                      bh_ / 2, r=0.003 * w)
            img = paint(img, m, wallc * rs.uniform(0.80, 1.14), 0.96)
            img = ink(img, m, 0.38)
            x += bw_ + rs.uniform(0.010, 0.028) * w
        img = img * (1 - 0.22 * srect(h, w, 0.5 * w, y0 + 0.035 * h,
                                      0.60 * w, 0.030 * h, r=0.006 * w,
                                      soft=2.4)[..., None])
    elif back == "aisle":
        rs = np.random.default_rng(13)
        for row in range(3):
            y0 = (0.085 + 0.185 * row) * h
            board = srect(h, w, 0.5 * w, y0 + 0.082 * h, 0.62 * w, 0.008 * h,
                          r=0.002 * w)
            x = 0.075 * w
            while x < 0.92 * w:
                bw_ = rs.uniform(0.026, 0.058) * w
                bh_ = rs.uniform(0.052, 0.078) * h
                m = srect(h, w, x + bw_ / 2, y0 + 0.078 * h - bh_ / 2,
                          bw_ / 2, bh_ / 2, r=0.002 * w)
                img = paint(img, m,
                            np.clip(wallc * rs.uniform(0.62, 1.72), 0, 1),
                            0.95)
                img = ink(img, m, 0.34)
                x += bw_ + rs.uniform(0.004, 0.014) * w
            img = paint(img, board, wallc * 0.56, 0.96)
            img = ink(img, board, 0.40)
    elif back == "livingwall":
        tv = srect(h, w, 0.560 * w, 0.185 * h, 0.235 * w, 0.115 * h,
                   r=0.006 * w)
        img = paint(img, tv, np.array([0.155, 0.170, 0.195]), 0.97)
        img = paint(img, srect(h, w, 0.560 * w, 0.185 * h, 0.222 * w,
                               0.103 * h, r=0.004 * w),
                    np.array([0.215, 0.245, 0.290]), 0.85)
        img = ink(img, tv, 0.45)
        pic = srect(h, w, 0.145 * w, 0.215 * h, 0.062 * w, 0.082 * h,
                    r=0.004 * w)
        img = paint(img, pic, wallc * 0.72, 0.95)
        img = paint(img, srect(h, w, 0.145 * w, 0.215 * h, 0.050 * w,
                               0.068 * h, r=0.003 * w), wallc * 1.16, 0.9)
        img = ink(img, pic, 0.36)

    anchors, focus = [], np.zeros((h, w))
    for (name, ux, d, family, col, sz) in sorted(room["props"],
                                                 key=lambda r: r[2]):
        s = sz * (0.026 + 0.078 * d) * w
        cy = (hz + 0.045 + (0.92 - hz) * d) * h
        cx = (0.5 + (ux - 0.5) * (0.66 + 0.74 * d)) * w
        base = rgb(col)
        body, det, hi, ccy, rx, ry = _prop(h, w, family, cx, cy, s)
        img = img * (1 - 0.46 * sellipse(h, w, cx, cy + 0.16 * s, 1.55 * s,
                                         0.42 * s, 1.6)[..., None])
        lit = 1.02 + 0.26 * np.exp(-0.5 * ((xx - (cx - 1.1 * s)) /
                                           (1.9 * s)) ** 2)
        img = img * (1 - body[..., None]) + base[None, None, :] * \
            lit[..., None] * body[..., None]
        img = paint(img, det, base * 0.74, 0.85)
        img = img + 0.13 * hi[..., None]
        img = ink(img, body, 0.50, 1.25)
        anchors.append(dict(name=name, cx=cx, cy=ccy, rx=rx, ry=ry))
        focus = np.maximum(focus, sellipse(h, w, cx, ccy, 2.1 * rx, 2.1 * ry,
                                           2.4))

    skin = rgb(room["skin"])
    for hd in room["hands"]:
        s = hd["s"] * w
        J = hand_joints(hd["u"] * w, hd["v"] * h, s, ang=hd.get("ang", 0.0),
                        curl=hd.get("curl", 0.0), side=hd.get("side", "r"))
        fore = (J[0][0] + (0.9 * s if hd.get("side") == "r" else -0.9 * s),
                J[0][1] + 2.6 * s)
        img, m = draw_hand(img, J, s, skin, forearm=fore)
        anchors.append(dict(name="hand_" + hd.get("side", "r"),
                            cx=float(J[:, 0].mean()), cy=float(J[:, 1].mean()),
                            rx=1.5 * s, ry=1.7 * s, joints=J))
        focus = np.maximum(focus, gaussian_filter(m, 3.0))

    focus = np.maximum(focus, 0.55 * np.clip((v - hz) / 0.28, 0, 1) *
                       np.clip((1.02 - v) / 0.30, 0, 1))
    if back in ("aisle", "shelfstrip"):
        # in these two rooms the wall *is* the subject, so it has to stay in
        # the focal plane instead of being thrown away as background
        focus = np.maximum(focus, 0.85 * np.clip((hz - v) / 0.10, 0, 1) *
                           np.clip((v - 0.02) / 0.09, 0, 1))
    return img, np.clip(focus, 0, 1), anchors


# ---------------------------------------------------------------------------
# Scene: conveyor  (fixed oblique camera over a production line)
#
# Added for the video / temporal family: unlike ``tabletop``, ``indoor`` and
# ``shelf`` this skeleton actually *moves* with ``t`` - the belt slats travel,
# the parts ride along at the belt speed, and a pusher arm sweeps on its own
# cycle - so a caller can render a real clip instead of re-seeding a still.
# Three independent motion sources in one static frame is exactly what a
# motion-segmentation or scene-flow figure needs.
#
# ``anchors`` names every part plus the pusher head, so callers can attach
# masks, boxes and velocity arrows to content rather than to guessed pixels.
# ---------------------------------------------------------------------------
CONVEYOR_TOP, CONVEYOR_BOT = 0.415, 0.845      # belt edges, in v units
CONVEYOR_SPEED = 0.62                          # belt travel per unit t
CONVEYOR_PARTS = [(0.02, "box", "#5c6672", 1.00),
                  (0.30, "cyl", "#6f7883", 0.86),
                  (0.58, "box", "#4d5763", 1.16),
                  (0.83, "disc", "#78818b", 0.82)]


def _pusher_pose(t, h, w):
    """Pivot point and head of the sweep arm at time ``t`` (pixels)."""
    ang = -0.62 + 0.80 * np.sin(2 * np.pi * (0.5 * t + 0.10))
    px, py = 0.285 * w, 0.055 * h
    L = 0.315 * h
    return (px, py), (px + L * np.sin(ang), py + L * np.cos(ang))


def conveyor_parts(t, h, w):
    """Where the parts sit at time ``t`` - the same maths ``_conveyor`` uses."""
    out = []
    for (x0, fam, col, sz) in CONVEYOR_PARTS:
        px = ((x0 + CONVEYOR_SPEED * t) % 1.20 - 0.10) * w
        py = (0.5 * (CONVEYOR_TOP + CONVEYOR_BOT) + 0.085) * h
        out.append((px, py, sz * 0.055 * w, fam, col))
    return out


def _conveyor(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    top = float(p.get("belt_top", CONVEYOR_TOP))
    bot = float(p.get("belt_bot", CONVEYOR_BOT))

    wallc = rgb(p.get("wall", "#4a5158"))
    img = wallc[None, None, :] * (0.82 + 0.42 * (v / max(top, 1e-3))[..., None])
    img = img + 0.026 * octave_noise(rng, (h, w), 4.5, 3)[..., None]
    for sx in (0.115, 0.470, 0.865):                  # frame uprights
        m = srect(h, w, sx * w, 0.19 * h, 0.011 * w, 0.20 * h)
        img = paint(img, m, wallc * 0.74, 0.96)
        img = ink(img, m, 0.40)
    pan = srect(h, w, 0.690 * w, 0.150 * h, 0.100 * w, 0.076 * h, r=0.008 * w)
    img = paint(img, pan, wallc * 0.62, 0.97)
    img = paint(img, srect(h, w, 0.690 * w, 0.150 * h, 0.086 * w, 0.060 * h,
                           r=0.005 * w), np.array([0.46, 0.55, 0.62]), 0.92)
    img = ink(img, pan, 0.44)
    for cx in (0.628, 0.660, 0.692):                  # indicator lamps
        img = img + 0.20 * sellipse(h, w, cx * w, 0.212 * h, 0.006 * w,
                                    0.007 * h, 0.6)[..., None] * \
            np.array([0.72, 0.86, 0.96])
    img = img + 0.12 * np.exp(-0.5 * ((v - 0.045) / 0.045) ** 2)[..., None] * \
        np.array([0.94, 0.96, 1.00])

    g = np.clip((v - top) / max(bot - top, 1e-3), 0, 1)
    belt = np.array([0.250, 0.268, 0.290])[None, None, :] * \
        (0.84 + 0.46 * g[..., None])
    slats = 0.5 + 0.5 * np.cos(2 * np.pi * (xx / w * 15.0 - 2.8 * t))
    belt = belt * (0.92 + 0.17 * (slats ** 3)[..., None])
    belt = belt + 0.022 * octave_noise(rng, (h, w), 8.0, 4)[..., None]
    img = np.where(((v >= top) & (v <= bot))[..., None], belt, img)
    floor = np.array([0.330, 0.348, 0.372])[None, None, :] * \
        (0.94 + 0.26 * np.clip((v - bot) / max(1 - bot, 1e-3), 0, 1)[..., None])
    img = np.where((v > bot)[..., None], floor, img)
    for ry, rh in ((top, 0.011), (bot, 0.013)):       # side rails
        m = srect(h, w, 0.5 * w, ry * h, 0.62 * w, rh * h)
        img = paint(img, m, np.array([0.58, 0.61, 0.65]), 0.96)
        img = ink(img, m, 0.36)

    anchors, focus = [], np.zeros((h, w))
    for (px, py, s, fam, col) in conveyor_parts(t, h, w):
        body, det, hi, ccy, rx, ry = _prop(h, w, fam, px, py, s)
        img = img * (1 - 0.48 * sellipse(h, w, px, py + 0.14 * s, 1.5 * s,
                                         0.40 * s, 1.6)[..., None])
        base = rgb(col)
        lit = 1.04 + 0.24 * np.exp(-0.5 * ((xx - (px - 1.1 * s)) /
                                           (1.9 * s)) ** 2)
        img = img * (1 - body[..., None]) + base[None, None, :] * \
            lit[..., None] * body[..., None]
        img = paint(img, det, base * 0.76, 0.85)
        img = img + 0.12 * hi[..., None]
        img = ink(img, body, 0.50, 1.25)
        anchors.append(dict(name="part", cx=px, cy=ccy, rx=rx, ry=ry))
        focus = np.maximum(focus, sellipse(h, w, px, ccy, 2.1 * rx, 2.1 * ry,
                                           2.4))

    pv, ph = _pusher_pose(t, h, w)
    arm = np.clip(scapsule(h, w, pv, ph, 0.015 * w, 0.9)
                  + sellipse(h, w, ph[0], ph[1], 0.030 * w, 0.026 * h, 0.8),
                  0, 1)
    img = img * (1 - 0.34 * gaussian_filter(arm, 3.0)[..., None])
    img = paint(img, arm, np.array([0.520, 0.548, 0.590]), 0.97)
    img = ink(img, arm, 0.48, 1.3)
    anchors.append(dict(name="pusher", cx=ph[0], cy=ph[1], rx=0.055 * w,
                        ry=0.055 * h))
    focus = np.maximum(focus, gaussian_filter(arm, 3.0))

    focus = np.maximum(focus, 0.78 * np.clip((v - top + 0.06) / 0.10, 0, 1) *
                       np.clip((bot + 0.07 - v) / 0.12, 0, 1))
    return img, np.clip(focus, 0, 1), anchors


# ---------------------------------------------------------------------------
# Human bodies  (shared by the mesh-recovery and mocap figures)
# ---------------------------------------------------------------------------
JOINTS = ("pelvis", "spine", "neck", "head",
          "l_shoulder", "l_elbow", "l_wrist",
          "r_shoulder", "r_elbow", "r_wrist",
          "l_hip", "l_knee", "l_ankle", "l_toe",
          "r_hip", "r_knee", "r_ankle", "r_toe")
# (from, to, limb radius in metres) - a capsule radius, so the torso entries
# are half a torso *width*, not half a torso thickness
BONES = [(0, 1, .150), (1, 2, .140), (2, 3, .068),
         (2, 4, .072), (4, 5, .052), (5, 6, .042),
         (2, 7, .072), (7, 8, .052), (8, 9, .042),
         (0, 10, .098), (10, 11, .082), (11, 12, .056), (12, 13, .042),
         (0, 14, .098), (14, 15, .082), (15, 16, .056), (16, 17, .042)]


def human_pose(kind="walk", phase=0.0, forward=0.0):
    """An 18-joint body in metres: X right, Y up, Z away from the camera.

    Legs and arms are solved by two-link forward kinematics from the hip and
    the shoulder, so a pose is described by four angles per limb rather than
    by 54 hand-typed numbers - which is what keeps four poses looking like
    four configurations of one skeleton.
    """
    ph = 2 * np.pi * phase
    P = np.zeros((18, 3))
    lean, hipy, sw, arm, kbend, abend = 0.0, 0.94, 0.0, 0.0, 0.0, 0.0
    twist = 0.0
    if kind == "walk":
        sw, arm = 0.34, 0.30
        hipy = 0.945 - 0.022 * abs(np.sin(ph))
        lean = 0.06
    elif kind == "reach":
        lean, hipy = 0.16, 0.915
        arm = -1.55
    elif kind == "crouch":
        hipy, lean = 0.60, 0.34
        kbend, abend = 1.30, 0.55
    elif kind == "turn":
        twist, lean = 0.85, 0.05
        sw, arm = 0.16, 0.55

    pelvis = np.array([0.0, hipy, forward])
    P[0] = pelvis
    up = np.array([np.sin(lean) * 0.0, np.cos(lean), np.sin(lean)])
    P[1] = pelvis + 0.245 * up
    P[2] = pelvis + 0.500 * up
    P[3] = pelvis + 0.690 * up + np.array([0.0, 0.0, 0.03 * np.sin(lean)])

    ct, st = np.cos(twist), np.sin(twist)

    def rot(vec):
        return np.array([vec[0] * ct + vec[2] * st, vec[1],
                         -vec[0] * st + vec[2] * ct])

    for side, sgn, ai in ((0, -1.0, 4), (1, 1.0, 7)):
        sh = P[2] + rot(np.array([sgn * 0.185, -0.055, 0.0]))
        a0 = arm * (1 if side == 0 else -1) * (np.sin(ph) if kind == "walk"
                                               else 1.0)
        if kind == "reach":
            a0 = -1.45 - 0.25 * side
        elbow = sh + 0.285 * rot(np.array([sgn * 0.10, -np.cos(a0),
                                           np.sin(a0)]))
        b0 = a0 + (0.55 if kind != "reach" else -0.35)
        wrist = elbow + 0.265 * rot(np.array([sgn * 0.06, -np.cos(b0),
                                              np.sin(b0)]))
        P[ai], P[ai + 1], P[ai + 2] = sh, elbow, wrist

    for side, sgn, li in ((0, -1.0, 10), (1, 1.0, 14)):
        hip = pelvis + np.array([sgn * 0.105, -0.020, 0.0])
        th = sw * (np.sin(ph + (0 if side == 0 else np.pi))
                   if kind == "walk" else 0.0)
        if kind == "crouch":
            th = 0.62
        if kind == "turn":
            th = sgn * 0.18
        kn = hip + 0.435 * np.array([0.0, -np.cos(th), np.sin(th)])
        if kind == "crouch":
            tk = th - kbend
        elif kind == "walk":
            # a smooth knee cycle: a clipped sine flat-lines at zero for half
            # the stride, which reads as a stiff prosthetic in an overlay
            tk = th - max(0.03, 0.34 + 0.36 * np.sin(
                ph + 1.15 + (0.0 if side == 0 else np.pi)))
        else:
            tk = th - 0.08                     # a standing knee, barely bent
        an = kn + 0.415 * np.array([0.0, -np.cos(tk), np.sin(tk)])
        an[1] = max(an[1], 0.062)
        toe = an + np.array([0.0, -0.048, 0.155 + 0.02 * abend])
        P[li], P[li + 1], P[li + 2], P[li + 3] = hip, kn, an, toe
    return P


def body_radii(dz=4.0, f=1.30, scale=1.0):
    """Limb radii in normalised *height* units for a body at depth ``dz``.

    The radii in ``BONES`` are metres; a caller that forgets to divide by the
    projection depth gets a snowman.  Always pair this with ``project``.
    """
    return np.array([b[2] for b in BONES]) * f / float(dz) * float(scale)


def project(P, f=1.30, cam_y=1.05, cam_z=-4.0, yaw=0.0, cx=0.5, cy=0.62,
            aspect=1.0):
    """Weak-perspective projection to normalised (u, v) frame coordinates.

    ``aspect`` is the panel's width / height in pixels; dividing the
    horizontal term by it keeps the body isotropic in a non-square tile.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    ca, sa = np.cos(yaw), np.sin(yaw)
    X = P[:, 0] * ca + P[:, 2] * sa
    Z = -P[:, 0] * sa + P[:, 2] * ca
    dz = np.maximum(Z - cam_z, 0.35)
    return np.stack([cx + f * X / dz / max(aspect, 1e-6),
                     cy - f * (P[:, 1] - cam_y) / dz], axis=1), dz


# ---------------------------------------------------------------------------
# Scene: studio  (motion-capture volume: cyclorama, marked floor, truss rig)
# ---------------------------------------------------------------------------
def _studio(h, w, rng, t, p):
    yy, xx, u, v = grid(h, w)
    hz = float(p.get("horizon", 0.520))

    back = np.array([0.565, 0.588, 0.618])
    fl = np.array([0.352, 0.368, 0.392])
    wall = back[None, None, :] * (0.86 + 0.34 * (v / max(hz, 1e-3))[..., None])
    g = np.clip((v - hz) / (1 - hz), 1e-3, 1)
    floor = fl[None, None, :] * (0.90 + 0.52 * g[..., None])
    img = np.where((v < hz)[..., None], wall, floor)
    img = img + 0.022 * octave_noise(rng, (h, w), 5.0, 3)[..., None]
    # the cyclorama coving: a soft bright band instead of a hard corner
    img = img + 0.028 * np.exp(-0.5 * ((v - hz) / 0.022) ** 2)[..., None]

    # perspective floor grid: lines converge on the vanishing point
    vp = 0.5 * w
    for k in range(-4, 5):
        xf = vp + k * 0.30 * w
        m = np.abs(xx - (vp + (xf - vp) * np.clip(g, 0, 1) ** 0.85)) < \
            (0.35 + 1.3 * g)
        img = paint(img, gaussian_filter((m & (v > hz)).astype(float), 0.6),
                    fl * 1.55, 0.40)
    for d in (0.16, 0.34, 0.56, 0.82):
        m = (np.abs(g - d) < 0.006 + 0.010 * d) & (v > hz)
        img = paint(img, gaussian_filter(m.astype(float), 0.6), fl * 1.45,
                    0.32)
    # taped capture-volume boundary
    m = (np.abs(g - 0.70) < 0.010) & (v > hz) & (np.abs(u - 0.5) < 0.40)
    img = paint(img, gaussian_filter(m.astype(float), 0.7),
                np.array([0.640, 0.660, 0.690]), 0.60)

    # truss + two soft key lights along the top edge
    img = paint(img, srect(h, w, 0.5 * w, 0.012 * h, 0.62 * w, 0.014 * h,
                           r=0.004 * w), np.array([0.245, 0.258, 0.278]), 0.9)
    for lx in (0.255, 0.745):
        img = paint(img, srect(h, w, lx * w, 0.036 * h, 0.024 * w, 0.020 * h,
                               r=0.006 * w),
                    np.array([0.300, 0.315, 0.335]), 0.95)
        img = img + 0.14 * sellipse(h, w, lx * w, 0.20 * h, 0.24 * w,
                                    0.24 * h, 5.0)[..., None] * \
            np.array([0.96, 0.97, 1.00])

    anchors, focus = [], np.zeros((h, w))
    focus = np.maximum(focus, 0.55 * np.exp(-0.5 * ((v - 0.66) / 0.26) ** 2))

    body = p.get("body")
    if body is not None:
        pts = np.asarray(body, dtype=float) * np.array([w, h])
        # radii arrive in normalised height units (see ``body_radii``)
        radii = np.asarray(p.get("radii", body_radii()), dtype=float) * h
        torso = rgb(p.get("shirt", "#59626e"))
        legs = rgb(p.get("trousers", "#434a54"))
        skin = rgb(p.get("skin", "#8e857c"))
        limb = np.zeros((h, w))
        gy = float(max(pts[13][1], pts[17][1]))
        gx = float(0.5 * (pts[13][0] + pts[17][0]))
        img = img * (1 - 0.42 * sellipse(h, w, gx, gy + 0.004 * h,
                                         0.070 * w, 0.014 * h,
                                         1.8)[..., None])
        for (a, b, _), r in zip(BONES, radii):
            r = max(r, 1.0)
            col = skin if b in (3, 6, 9) else (legs if a >= 10 or b >= 10
                                               else torso)
            m = scapsule(h, w, pts[a], pts[b], r, 0.75)
            lit = 1.12 - 0.26 * (yy / h) + 0.16 * np.exp(
                -0.5 * ((xx - (pts[a][0] - 1.7 * r)) / (2.6 * r)) ** 2)
            img = img * (1 - m[..., None]) + col[None, None, :] * \
                lit[..., None] * m[..., None]
            limb = np.maximum(limb, m)
        hr = float(p.get("head_r", 1.50 * radii[2]))
        head = sellipse(h, w, pts[3][0], pts[3][1] - 0.10 * hr, hr,
                        1.20 * hr, 0.8)
        img = img * (1 - head[..., None]) + skin[None, None, :] * 1.04 * \
            head[..., None]
        limb = np.clip(limb + head, 0, 1)
        img = ink(img, limb, 0.55, 1.3)
        anchors.append(dict(name="person", cx=float(pts[:, 0].mean()),
                            cy=float(pts[:, 1].mean()),
                            rx=float(np.ptp(pts[:, 0])) * 0.7,
                            ry=float(np.ptp(pts[:, 1])) * 0.6, joints=pts))
        focus = np.maximum(focus, gaussian_filter(limb, 4.0))
    return img, np.clip(focus, 0, 1), anchors


# ---------------------------------------------------------------------------
_BUILDERS = {"road": _road, "tabletop": _tabletop, "indoor": _indoor,
             "aerial": _aerial, "shelf": _shelf, "ego": _ego,
             "studio": _studio, "conveyor": _conveyor}

_CAM_PRESETS = {
    # wide dash lens, cool overcast daylight
    "road": dict(wb=(0.98, 1.00, 1.05), exposure=1.00, contrast=1.08,
                 vignette=0.28, dof=1.3, sharpen=0.95, kappa=0.030),
    # short macro-ish lens, tungsten indoor, shallow depth of field
    "tabletop": dict(wb=(1.06, 1.00, 0.93), exposure=1.04, contrast=1.16,
                     vignette=0.32, dof=2.2, sharpen=1.25, grain=0.011,
                     kappa=0.032),
    # normal lens, mixed daylight / bounce, medium depth of field
    "indoor": dict(wb=(1.03, 1.00, 0.98), exposure=1.02, contrast=1.06,
                   vignette=0.30, dof=2.2, sharpen=0.95),
    # long lens from altitude, flat light, nearly everything in focus
    "aerial": dict(wb=(0.99, 1.01, 1.02), exposure=1.06, contrast=1.14,
                   vignette=0.18, dof=1.1, sharpen=1.05, grain=0.008),
    # close frontal, soft window light
    "shelf": dict(wb=(1.04, 1.00, 0.96), exposure=1.03, contrast=1.10,
                  vignette=0.30, dof=2.6, sharpen=1.10),
    # head-mounted wide lens.  Deliberately cool and low-chroma: an indoor
    # first-person rig is tungsten-orange out of the box, and the batch
    # colour budget allows at most 8 percent warm area per figure.
    "ego": dict(wb=(0.965, 1.00, 1.075), exposure=1.02, contrast=1.10,
                vignette=0.34, dof=1.9, sharpen=1.10, grain=0.010,
                sat=0.80, kappa=0.031),
    # capture volume: long-ish lens, flat fill light, nearly neutral grey
    "studio": dict(wb=(0.985, 1.00, 1.035), exposure=1.02, contrast=1.09,
                   vignette=0.22, dof=1.5, sharpen=1.05, grain=0.008,
                   sat=0.86, kappa=0.031),
    # fixed industrial camera: stopped down, cool fluorescent light, low
    # chroma - a factory line is grey metal and the batch budget is cool
    "conveyor": dict(wb=(0.955, 1.00, 1.070), exposure=1.05, contrast=1.12,
                     vignette=0.24, dof=1.1, sharpen=1.20, grain=0.009,
                     sat=0.72, kappa=0.030),
}


def build(kind, seed, shape=(138, 204), t=0.0, cam=None, **params):
    """Render one finished scene; see the module docstring for the contract."""
    if kind not in _BUILDERS:
        raise ValueError(f"unknown scene kind {kind!r}; choose from {KINDS}")
    h, w = shape
    rng = np.random.default_rng(seed)
    img, focus, anchors = _BUILDERS[kind](h, w, rng, t, params)
    c = camera(**_CAM_PRESETS[kind])
    if cam:
        c.update(cam)
    out = finish(np.clip(img, 0, 1.4), np.clip(focus, 0, 1), c,
                 np.random.default_rng(seed + 991))
    return dict(rgb=out, focus=np.clip(focus, 0, 1), anchors=anchors, cam=c,
                kind=kind)


def grey(img, lo=0.10, hi=0.86):
    """Luminance, remapped into a printable range for heat-map overlays."""
    lum = np.asarray(img) @ np.array([0.30, 0.59, 0.11])
    return np.clip(lo + hi * np.clip(lum, 0, 1), 0, 1)
