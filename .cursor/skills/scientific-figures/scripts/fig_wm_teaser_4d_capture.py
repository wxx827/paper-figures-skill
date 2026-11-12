"""4D reconstruction teaser: camera ring -> deformable Gaussians -> renders.

Left is the capture rig, drawn as a perspective ring of hand-built camera
glyphs all sighted on the subject in the middle.  The centre panel is the 4D
representation itself: the same actor resolved as a Gaussian splat cloud at
four instants, fading in along an internal time axis.  Right is what the
representation buys you -- four free-viewpoint renders at those instants,
each stamped with its azimuth, plus a strip of headline metrics underneath.

Renders are procedural (multi-octave value noise + Gaussian blur + muted ramp
+ vignette) with a bright subject blob that walks across the frame, so the
four thumbnails belong to one continuous take.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Ellipse, Polygon)
from scipy.ndimage import gaussian_filter, zoom
from pubstyle import set_pub_style, savefig, lighten, get_cmap

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DARK = "#3a3a3a"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

CMAP_STUDIO = LinearSegmentedColormap.from_list(
    "wm_studio", ["#33383f", "#5b636d", "#8d949c", "#c2c7cc", "#eceef0"])
SPLAT = get_cmap("cb_sequential_blue")

R_BOX = 0.08
W, H = 11.9, 3.5
Y_CAP = 3.24
BAND0, BAND1 = 1.20, 3.10


# ---- canvas ----------------------------------------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


# ---- procedural imagery ----------------------------------------------------
def value_noise(rng, ny, nx, base=3, octaves=4, decay=0.55):
    out = np.zeros((ny, nx))
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        gh = max(2, base * 2 ** k)
        gw = max(2, int(round(gh * nx / ny)))
        small = rng.random((gh, gw))
        big = zoom(small, (ny / gh, nx / gw), order=3)[:ny, :nx]
        if big.shape != (ny, nx):
            big = np.pad(big, ((0, ny - big.shape[0]), (0, nx - big.shape[1])),
                         mode="edge")
        out += amp * big
        tot += amp
        amp *= decay
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-9)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def studio_shot(rng, sx, ny=132, nx=88):
    """Portrait studio frame: one bright standing subject on a dim wall."""
    yy = np.linspace(0, 1, ny)[:, None]
    xx = np.linspace(0, 1, nx)[None, :]
    field = 0.30 + 0.12 * yy + 0.0 * xx
    field += 0.12 * _sig(22 * (yy - 0.80))
    field += 0.36 * np.exp(-(((xx - sx) / 0.135) ** 4
                             + ((yy - 0.60) / 0.210) ** 4))
    field += 0.30 * np.exp(-(((xx - sx) / 0.050) ** 2
                             + ((yy - 0.29) / 0.052) ** 2))
    field += 0.16 * np.exp(-(((xx - sx) / 0.200) ** 4
                             + ((yy - 0.50) / 0.070) ** 4))   # shoulders
    field -= 0.16 * np.exp(-(((xx - sx) / 0.085) ** 2
                             + ((yy - 0.385) / 0.016) ** 2))  # neck
    field -= 0.15 * np.exp(-(((xx - sx) / 0.014) ** 2
                             + ((yy - 0.735) / 0.075) ** 2))  # gap, legs
    field -= 0.12 * np.exp(-(((xx - sx) / 0.17) ** 2
                             + ((yy - 0.87) / 0.030) ** 2))
    field += 0.11 * (value_noise(rng, ny, nx, base=3) - 0.5)
    field = gaussian_filter(field, 1.0)
    return (field - field.min()) / (np.ptp(field) + 1e-9)


def colorize(field, cmap, vignette=0.26, gain=1.1):
    ny, nx = field.shape
    yy = np.linspace(-1, 1, ny)[:, None]
    xx = np.linspace(-1, 1, nx)[None, :]
    rgb = cmap(np.clip((field - 0.5) * gain + 0.5, 0, 1))[..., :3]
    vig = 1.0 - vignette * (xx ** 2 + yy ** 2) / 2.0
    return np.clip(rgb * vig[..., None], 0, 1)


# ---- primitives ------------------------------------------------------------
def box(ax, x0, y0, w, h, fc="white", ec=RULE, lw=0.9, ls="-", r=R_BOX,
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def photo(ax, rgb, x0, y0, w, h, ec="white", lw=1.0, r=0.05, zorder=6):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 2)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=[x0, x0 + w, y0, y0 + h], zorder=zorder,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def arrow(ax, p0, p1, color=RULE, lw=1.3, ms=12, zorder=8):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                        zorder=zorder)
    ax.add_patch(a)
    return a


def caption(ax, x, text):
    ax.text(x, Y_CAP, text, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK)


# ---- rig -------------------------------------------------------------------
def camera_glyph(ax, cx, cy, ang, s=0.115, color=BLUE, zorder=6):
    c, sn = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -sn], [sn, c]])

    def put(pts, fc, z):
        p = (np.asarray(pts) @ rot.T) + np.array([cx, cy])
        ax.add_patch(Polygon(p, closed=True, facecolor=fc, edgecolor="white",
                             linewidth=0.5, zorder=z))

    put([(-1.05 * s, -0.55 * s), (0.0, -0.55 * s), (0.0, 0.55 * s),
         (-1.05 * s, 0.55 * s)], color, zorder)
    put([(0.0, -0.34 * s), (0.62 * s, -0.55 * s), (0.62 * s, 0.55 * s),
         (0.0, 0.34 * s)], lighten(color, 0.38), zorder)


def subject_silhouette(ax, x, y, s=1.0, color=DARK, zorder=7, alpha=1.0):
    ax.add_patch(Circle((x, y + 0.30 * s), 0.075 * s, facecolor=color,
                        edgecolor="none", zorder=zorder, alpha=alpha))
    ax.add_patch(FancyBboxPatch(
        (x - 0.075 * s, y - 0.06 * s), 0.15 * s, 0.32 * s,
        boxstyle="round,pad=0,rounding_size=0.04", facecolor=color,
        edgecolor="none", zorder=zorder, alpha=alpha))
    for dx in (-0.042, 0.042):
        ax.add_patch(FancyBboxPatch(
            (x + dx * s - 0.030 * s, y - 0.34 * s), 0.060 * s, 0.30 * s,
            boxstyle="round,pad=0,rounding_size=0.025", facecolor=color,
            edgecolor="none", zorder=zorder - 1, alpha=alpha))


def draw_rig(ax):
    caption(ax, 1.70, "Multi-view capture rig")
    cx, cy, rx, ry = 1.70, 2.28, 1.20, 0.60
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, facecolor=lighten(BLUE, 0.94),
                         edgecolor=lighten(BLUE, 0.55), linewidth=0.9,
                         linestyle=(0, (5, 3)), zorder=2))
    th = np.linspace(0, 2 * np.pi, 14, endpoint=False) + 0.12
    for t in th:
        x, y = cx + rx * np.cos(t), cy + ry * np.sin(t)
        ang = np.arctan2(cy - y, cx - x)
        camera_glyph(ax, x, y, ang, zorder=4 if y > cy else 8)
    for t in th[::4]:
        x, y = cx + rx * np.cos(t), cy + ry * np.sin(t)
        ax.plot([x, cx], [y, cy], color=lighten(BLUE, 0.55), lw=0.7,
                ls=(0, (2, 2.4)), zorder=3)
    subject_silhouette(ax, cx, cy + 0.06, s=1.15, zorder=7)
    ax.text(cx, 1.38, "14 synchronised views  ·  30 fps", ha="center",
            va="center", fontsize=7.0, color=MUTED)


# ---- 4D representation -----------------------------------------------------
PARTS = [((0.000, 0.440), (0.040, 0.044), 0.16),
         ((0.000, 0.090), (0.062, 0.120), 0.31),
         ((-0.140, 0.090), (0.019, 0.105), 0.11),
         ((0.140, 0.090), (0.019, 0.105), 0.11),
         ((-0.055, -0.290), (0.023, 0.140), 0.155),
         ((0.055, -0.290), (0.023, 0.140), 0.155)]


def splat_cloud(rng, n=300, phase=0.0):
    """Point cloud in local units: height ~1.0, half-width ~0.16."""
    w = np.array([p[2] for p in PARTS])
    w = w / w.sum()
    pick = rng.choice(len(PARTS), size=n, p=w)
    pts = np.empty((n, 2))
    for k, (mu, sg, _) in enumerate(PARTS):
        m = pick == k
        swing = 0.045 * np.sin(phase) * (1 if k in (2, 5) else
                                         -1 if k in (3, 4) else 0)
        # truncate at 2 sigma so the silhouette keeps a crisp outline
        z = np.clip(rng.normal(0, 1, (m.sum(), 2)), -2, 2)
        pts[m, 0] = mu[0] + swing + z[:, 0] * sg[0]
        pts[m, 1] = mu[1] + z[:, 1] * sg[1]
    return pts


def draw_scene(ax, rng):
    caption(ax, 5.12, "Dynamic 4D representation")
    x0, x1 = 3.58, 6.66
    frame = box(ax, x0, BAND0, x1 - x0, BAND1 - BAND0,
                fc=lighten(GREEN, 0.95), ec=lighten(GREEN, 0.45), lw=1.1,
                ls=(0, (6, 3)), r=0.12, zorder=2)
    centers = np.linspace(x0 + 0.50, x1 - 0.50, 4)
    for k, (cx, a) in enumerate(zip(centers, (0.32, 0.52, 0.74, 0.96))):
        pts = splat_cloud(rng, phase=k * 0.9)
        sz = rng.uniform(1.6, 5.6, pts.shape[0])
        col = SPLAT(0.28 + 0.55 * (0.5 - pts[:, 1]))
        sc = ax.scatter(cx + pts[:, 0] * 1.20, 2.46 + pts[:, 1] * 1.20, s=sz,
                        c=col, alpha=a, linewidths=0, zorder=5)
        sc.set_clip_path(frame)
    ay = 1.54
    arrow(ax, (x0 + 0.22, ay), (x1 - 0.18, ay), color=RULE, lw=1.0, ms=11,
          zorder=6)
    for k, cx in enumerate(centers):
        ax.plot([cx, cx], [ay, ay + 0.09], color=RULE, lw=0.9, zorder=6)
        ax.text(cx, 1.34, f"$t_{k + 1}$", ha="center", va="center",
                fontsize=6.8, color=SUB)
    ax.text(5.12, 0.56, "42k anisotropic Gaussians  ·  deformation field "
                        "$\\Delta(x,\\,t)$", ha="center", va="center",
            fontsize=7.0, color=MUTED)
    return x0, x1


# ---- renders ---------------------------------------------------------------
def draw_renders(ax, rng):
    x0 = 7.08
    fw, gap, fh = 1.06, 0.09, 1.60
    caption(ax, x0 + (4 * fw + 3 * gap) / 2, "Free-viewpoint renders")
    times = ["0.0 s", "0.4 s", "0.8 s", "1.2 s"]
    azim = ["$\\phi$ = 15$\\degree$", "$\\phi$ = 75$\\degree$",
            "$\\phi$ = 150$\\degree$", "$\\phi$ = 220$\\degree$"]
    for k in range(4):
        x = x0 + k * (fw + gap)
        rgb = colorize(studio_shot(rng, 0.32 + 0.11 * k), CMAP_STUDIO)
        photo(ax, rgb, x, BAND1 - fh, fw, fh, ec="white", lw=1.0, r=0.05,
              zorder=6)
        chip = box(ax, x + 0.06, BAND1 - 0.28, 0.52, 0.20, fc="white",
                   ec="none", r=0.035, zorder=9, alpha=0.86)
        ax.text(x + 0.32, BAND1 - 0.18, azim[k], ha="center", va="center",
                fontsize=6.0, color=INK, zorder=10)
        chip.set_clip_on(False)
        ax.text(x + fw / 2, 1.38, f"$t$ = {times[k]}", ha="center",
                va="center", fontsize=6.8, color=SUB)
    stats = [("32.4 dB", "PSNR"), ("0.941", "SSIM"), ("38 fps", "render")]
    cw, cg = 1.42, 0.12
    sx = x0 + ((4 * fw + 3 * gap) - (3 * cw + 2 * cg)) / 2
    for k, (val, lab) in enumerate(stats):
        x = sx + k * (cw + cg)
        box(ax, x, 0.34, cw, 0.46, fc="#f5f5f5", ec=HAIR, lw=0.8, r=0.07,
            zorder=3)
        ax.text(x + cw / 2, 0.64, val, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=INK, zorder=5)
        ax.text(x + cw / 2, 0.46, lab, ha="center", va="center", fontsize=6.5,
                color=MUTED, zorder=5)
    return x0


def main():
    set_pub_style()
    rng = np.random.default_rng(476)
    fig, ax = canvas(W, H)

    draw_rig(ax)
    sx0, sx1 = draw_scene(ax, rng)
    rx0 = draw_renders(ax, rng)

    arrow(ax, (3.02, 2.20), (sx0 - 0.06, 2.20), lw=1.3, ms=13)
    ax.text((3.02 + sx0) / 2 - 0.03, 2.40, "fit", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    arrow(ax, (sx1 + 0.06, 2.20), (rx0 - 0.06, 2.20), lw=1.3, ms=13)
    ax.text((sx1 + rx0) / 2, 2.40, "render", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(1.70, 0.56, "4 s take  ·  1080p  ·  hardware sync", ha="center",
            va="center", fontsize=7.0, color=MUTED)

    savefig(fig, "1071_wm_teaser_4d_capture.png")


if __name__ == "__main__":
    main()
