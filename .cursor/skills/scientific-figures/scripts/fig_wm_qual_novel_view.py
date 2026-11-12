"""Figure 1088 - Novel-view synthesis: RGB, depth, normal and error per scene.

Three synthetic rooms are rendered from one procedural depth buffer, so the
four columns stay physically consistent: the depth map is back-projected to 3D
points and the surface normal is the cross product of its two tangents, the
shading in the RGB column follows those normals, and the error column is the
residual of a deliberately imperfect depth prediction.  Depth and error use one
shared colour range across all three scenes, with a single colour-bar strip on
the right.

Colour: the depth ramp is this batch's own blue ladder (near-white to saturated
navy) and the error ramp is a reversed ``PuBu``.  The normal column keeps the
standard RGB-encodes-XYZ convention, because that is how every reader
recognises a normal map, but its chroma is pulled 26 % towards mid grey
(``NORMAL_CHROMA``): at full saturation the green floor is a neon block that
outweighs the other 47 figures of this batch on a contact sheet, while the four
orientation classes stay just as separable once it is toned down.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig
from figkit import image_axes, shared_norm

H, W = 150, 176
Z_FAR = 11.5
FLOOR_K = 0.62                   # floor plane constant: z = FLOOR_K / (v - hz)
CAM_H = 1.6                      # camera height above the floor, metres
FOCAL = FLOOR_K * H / CAM_H      # implied focal length in pixels

# WM blue batch ladder (BRIEF_WM_BLUE.md section 2), reversed so that the ramp
# runs near-white -> saturated navy as the plotted value increases.
BLUE_LADDER = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7",
               "#7fb0dc", "#b3d0ea", "#dce9f5", "#f2f7fb"]
DEPTH_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "wm_blue", BLUE_LADDER[::-1])
NORMAL_CHROMA = 0.74             # see normal_rgb: 26 % chroma pull to mid grey
COLS = ["RGB (novel view)", "Inverse depth", "Surface normal",
        "Abs. depth error"]
SCENES = [("Scene A  ·  office", 11, "#b9a892"),
          ("Scene B  ·  lounge", 27, "#a8b3a0"),
          ("Scene C  ·  corridor", 43, "#b0a9b8")]


def octave_noise(rng, shape, base=3.0, octaves=4, persistence=0.55):
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        sigma = max(0.7, min(shape) / (base * (2.0 ** k)))
        n = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
        out += amp * n / (n.std() + 1e-9)
        tot += amp
        amp *= persistence
    return out / tot


def _rgb(c):
    return np.array(mpl.colors.to_rgb(c))


def rrect(cx, cy, hw, hh, r=2.0, soft=0.6):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx = np.clip(np.abs(xx - cx) - (hw - r), 0, None)
    dy = np.clip(np.abs(yy - cy) - (hh - r), 0, None)
    return gaussian_filter(np.clip(0.5 - (np.hypot(dx, dy) - r), 0, 1), soft)


# ---------------------------------------------------------------------------
def room_depth(seed):
    """Perspective room (floor, ceiling, two side walls, back wall) + props."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u, v = xx / W, yy / H
    hz = 0.40 + 0.05 * rng.random()

    z = np.full((H, W), Z_FAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(v > hz, FLOOR_K / np.maximum(v - hz, 1e-3), Z_FAR)
        ceil = np.where(v < hz, 0.50 / np.maximum(hz - v, 1e-3), Z_FAR)
        wl = np.where(u < 0.5, 2.05 / np.maximum(0.5 - u, 1e-3), Z_FAR)
        wr = np.where(u > 0.5, 2.05 / np.maximum(u - 0.5, 1e-3), Z_FAR)
    z = np.minimum.reduce([z, floor, ceil, wl, wr])

    props = []
    for _ in range(int(rng.integers(3, 5))):
        px = rng.uniform(0.16, 0.84)
        zo = rng.uniform(2.4, 6.0)
        sc = 1.0 / zo
        hw = W * (0.34 + 0.50 * rng.random()) * sc
        hh = H * (0.60 + 0.80 * rng.random()) * sc
        cy = (hz + FLOOR_K / zo) * H - hh
        cx = 0.5 * W + (px - 0.5) * W * (0.35 + 2.4 / zo)
        m = rrect(cx, cy, hw, hh, r=max(1.2, 0.10 * hw), soft=0.45) > 0.5
        # three visible facets, so the box has real normals instead of a decal
        side = np.sign(cx - 0.5 * W) or 1.0
        lat = np.clip(side * (xx - (cx + side * 0.30 * hw)) / (0.70 * hw), 0, 1)
        topf = np.clip(((cy - 0.62 * hh) - yy) / (0.38 * hh), 0, 1)
        zp = zo + 1.15 * lat + 0.85 * topf
        # paint the albedo only where the box really wins the depth test,
        # otherwise the floor edge shows straight through it
        vis = m & (zp < z)
        z = np.where(vis, zp, z)
        props.append((vis, lat, topf))
    for k in range(len(props) - 1):
        later = np.logical_or.reduce([p[0] for p in props[k + 1:]])
        props[k] = (props[k][0] & ~later,) + props[k][1:]
    z = np.clip(gaussian_filter(z, 0.5), 0.9, Z_FAR)
    return z, props, rng, hz


def normals_from_depth(z, hz):
    """True surface normals: back-project to 3D, then cross the two tangents.

    Taking the image-space gradient of ``z`` directly and pinning the third
    component to a constant is what flattens a normal map into one uniform
    lavender: the constant dominates, every surface ends up facing the camera
    and the floor / side walls stop being distinguishable.  Back-projecting
    through the pinhole first keeps the real orientation, so the floor comes
    out green, the back wall lavender and the two side walls red and teal.
    """
    zs = gaussian_filter(z, 0.8)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    p = np.dstack([(xx - 0.5 * W) * zs / FOCAL,
                   (yy - hz * H) * zs / FOCAL, zs])
    n = np.cross(np.gradient(p, axis=1), np.gradient(p, axis=0))
    n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    # orient every normal towards the camera at the origin
    flip = np.sum(n * p, axis=2, keepdims=True) > 0
    return np.where(flip, -n, n)


def normal_rgb(n):
    """Standard tangent-space encoding: +x red, up green, towards camera blue.

    The channel triple is pulled towards mid grey by ``NORMAL_CHROMA`` after
    encoding.  This is a pure contrast scale about 0.5, so it is monotone and
    invertible per channel: the mapping from orientation to hue is unchanged
    and only the chroma comes down.
    """
    rgb = np.dstack([n[..., 0], -n[..., 1], -n[..., 2]]) * 0.5 + 0.5
    return np.clip(0.5 + NORMAL_CHROMA * (rgb - 0.5), 0, 1)


def render(seed, tint):
    z, props, rng, hz = room_depth(seed)
    n = normals_from_depth(z, hz)

    # ---- albedo: wall paint + floor + one colour per prop
    alb = np.ones((H, W, 3)) * _rgb(tint)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    v = yy / H
    alb = np.where((v > hz)[..., None], _rgb("#7d7268") * 1.05, alb)
    grain = octave_noise(rng, (H, W), base=5.0, octaves=4)
    alb = alb * (1.0 + 0.11 * grain[..., None])
    prop_cols = ["#5d6b7a", "#8a6a52", "#4f6b5c", "#7a6076"]
    for k, (m, lat, topf) in enumerate(props):
        c = _rgb(prop_cols[k % len(prop_cols)])
        c = c[None, None, :] * (1.0 - 0.26 * lat - 0.14 * topf)[..., None]
        alb = np.where(m[..., None], c, alb)

    # key from above-left-front plus a weak fill from the right, so that with
    # real normals neither side wall drops to pure ambient
    key = np.array([-0.35, -0.80, -0.45])
    fill = np.array([0.60, -0.20, -0.75])
    key, fill = key / np.linalg.norm(key), fill / np.linalg.norm(fill)
    lam = 0.75 * np.clip(n @ key, 0, 1) + 0.35 * np.clip(n @ fill, 0, 1)
    gz = np.hypot(*np.gradient(gaussian_filter(z, 0.8)))
    ao = np.clip(1.0 - 0.95 * np.clip(gz, 0, 1.0), 0.22, 1.0)  # contact shadows
    ao = gaussian_filter(ao, 0.6)
    img = alb * (0.40 + 0.82 * lam[..., None]) * ao[..., None]
    u = xx / W
    vig = 1.0 - 0.26 * (((u - 0.5) * 1.9) ** 2 + ((v - 0.5) * 1.6) ** 2)
    img = (img * vig[..., None] - 0.5) * 1.14 + 0.5
    img = img + 0.011 * rng.standard_normal((H, W, 3))

    # ---- imperfect prediction: smooth bias + blow-ups on depth edges
    edge = np.abs(np.gradient(z)[0]) + np.abs(np.gradient(z)[1])
    edge = gaussian_filter(np.clip(edge, 0, 2.2), 1.1)
    bias = 0.17 * octave_noise(rng, (H, W), base=2.5, octaves=3)
    err = np.abs(bias * (0.4 + z / 4.0)) + 0.26 * edge + \
        0.05 * np.clip(z - 5.0, 0, None) * np.abs(octave_noise(rng, (H, W), 6, 3))
    err = gaussian_filter(err, 0.8)
    # The depth column is plotted as 1/z, not z.  Metric depth in a room is
    # heavily long-tailed - the back wall sits at Z_FAR while the near floor is
    # under 2 m - so a linear ramp squeezes every foreground surface into the
    # darkest sliver of the colour map and the floor gradient disappears.
    # Inverting is the standard monocular-depth convention and spreads the
    # near field over most of the ramp; the colour bar is labelled 1/m to match.
    return np.clip(img, 0, 1), 1.0 / z, normal_rgb(n), err


def main():
    set_pub_style(base_font=8)

    scenes = [render(seed, tint) for (_, seed, tint) in SCENES]
    # the nearest ceiling / floor pixels are a long tail; clipping them spreads
    # the room itself across the whole ladder instead of the pale third of it
    dnorm = shared_norm([s[1] for s in scenes], robust=(1.0, 96.0))
    enorm = shared_norm([s[3] for s in scenes], robust=(1.0, 99.0))

    fig = plt.figure(figsize=(10.0, 6.2))
    gs = fig.add_gridspec(3, 4, wspace=0.04, hspace=0.10,
                          left=0.058, right=0.878, top=0.900, bottom=0.058)

    im_d = im_e = None
    for i, ((name, _, _), (rgbimg, z, nrm, err)) in enumerate(zip(SCENES, scenes)):
        tiles = [(rgbimg, None, None), (z, DEPTH_CMAP, dnorm),
                 (nrm, None, None), (err, "PuBu_r", enorm)]
        for j, (arr, cmap, norm) in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            image_axes(ax, color="#9a9a9a", lw=0.55)
            ax.set_aspect("auto")
            h = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto",
                          interpolation="bilinear")
            if j == 1:
                im_d = h
            if j == 3:
                im_e = h
            if i == 0:
                ax.set_title(COLS[j], fontsize=8.6, pad=3.6)
            if j == 0:
                ax.set_ylabel(name, fontsize=8.2, labelpad=3.6)
        # both stamps are read off the arrays that are actually drawn
        mae = float(err.mean())
        ax0, axe = fig.axes[-4], fig.axes[-1]
        ax0.text(0.028, 0.955, f"PSNR {30.6 - 11.0 * mae:.1f} dB",
                 transform=ax0.transAxes, ha="left", va="top", fontsize=6.2,
                 color="white", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.20", facecolor="#00000099",
                           edgecolor="none"))
        axe.text(0.028, 0.955, f"MAE {mae:.2f} m", transform=axe.transAxes,
                 ha="left", va="top", fontsize=6.2, color="white", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.20", facecolor="#00000099",
                           edgecolor="none"))

    # ---- shared colour-bar strip: two equal bars, centred as a pair
    CB_X, CB_W, CB_H, CB_GAP = 0.896, 0.014, 0.300, 0.070
    top_y = 0.5 + CB_GAP / 2
    cax_d = fig.add_axes([CB_X, top_y, CB_W, CB_H])
    cb_d = fig.colorbar(im_d, cax=cax_d, extend="max")
    cb_d.set_label("Inverse depth  (1/m)   ·   dark = near", fontsize=7.0,
                   labelpad=4)
    cax_e = fig.add_axes([CB_X, top_y - CB_GAP - CB_H, CB_W, CB_H])
    cb_e = fig.colorbar(im_e, cax=cax_e)
    cb_e.set_label("Abs. depth error  (m)", fontsize=7.4, labelpad=4)
    for cb in (cb_d, cb_e):
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#333333")
        cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.4, color="#333333")

    savefig(fig, "1088_wm_qual_novel_view.png")


if __name__ == "__main__":
    main()
