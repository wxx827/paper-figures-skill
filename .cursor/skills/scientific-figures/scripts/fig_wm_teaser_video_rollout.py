"""Video world model teaser: context frames + action tokens -> film-strip rollout.

Fig.1-style banner in the visual idiom of recent video world-model papers:
observed context on the left, an action-token strip underneath, a single
gradient-filled method block in the middle carrying three hand-drawn icons,
and the generated rollout on the right rendered as a film strip.  A shared
time axis runs along the bottom and ties observed frames to generated ones.

All "frames" are procedurally synthesised (multi-octave value noise, Gaussian
blur, muted colour ramp, vignette) from one panorama, so the rollout pans
coherently instead of showing four unrelated colour blocks.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from scipy.ndimage import gaussian_filter, zoom
from pubstyle import set_pub_style, savefig, lighten

# ---- house colours ---------------------------------------------------------
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DARK = "#3a3a3a"
BLUE = "#3b5b92"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
RED = "#b23a48"

# Muted "photograph" ramps: lightness increases monotonically (DESIGN §2.2).
CMAP_SCENE = LinearSegmentedColormap.from_list(
    "wm_scene", ["#2b3a2f", "#4d6b52", "#8aa585", "#c6d3bd", "#eef1e6"])

R_BOX = 0.07          # one rounding radius for the whole figure (inches)
LW_BOX = 0.9


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
        big = zoom(small, (ny / gh, nx / gw), order=3)
        big = big[:ny, :nx]
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


def soft_rect(xx, yy, x0, x1, y0, y1, k=90):
    return (_sig(k * (xx - x0)) * _sig(k * (x1 - xx))
            * _sig(k * (yy - y0)) * _sig(k * (y1 - yy)))


def panorama(rng, ny=96, nx=460, horizon=0.46, n_obj=9):
    """One wide synthetic scene; crops of it become the video frames.

    Sky / ground split plus soft-edged standing objects, then blurred: after
    the colour ramp this reads as an out-of-focus photograph rather than as
    abstract noise.
    """
    yy = np.linspace(0, 1, ny)[:, None]
    xx = np.linspace(0, 1, nx)[None, :]
    hz = horizon + 0.02 * np.sin(6.0 * np.pi * xx)
    sky = 0.92 - 0.22 * (yy / np.maximum(hz, 1e-3))
    ground = 0.46 - 0.20 * (yy - hz)
    field = np.where(yy < hz, sky, ground)
    for _ in range(n_obj):
        cx = rng.uniform(0.02, 0.98)
        w = rng.uniform(0.012, 0.055)
        top = float(np.mean(hz)) - rng.uniform(0.06, 0.34)
        field += rng.uniform(-0.44, 0.30) * soft_rect(
            xx, yy, cx - w, cx + w, top, float(np.mean(hz)) + 0.06)
    field += 0.26 * (value_noise(rng, ny, nx, base=4) - 0.5)
    field = gaussian_filter(field, 1.05)
    return (field - field.min()) / (np.ptp(field) + 1e-9)


def colorize(field, cmap, vignette=0.30, gain=1.0):
    ny, nx = field.shape
    yy = np.linspace(-1, 1, ny)[:, None]
    xx = np.linspace(-1, 1, nx)[None, :]
    f = np.clip((field - 0.5) * gain + 0.5, 0, 1)
    rgb = cmap(f)[..., :3]
    vig = 1.0 - vignette * (xx ** 2 + yy ** 2) / 2.0
    return np.clip(rgb * vig[..., None], 0, 1)


def frame(ax, rgb, x0, y0, w, h, ec="white", lw=1.0, r=R_BOX, zorder=4):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 2)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=[x0, x0 + w, y0, y0 + h], zorder=zorder,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


# ---- primitives ------------------------------------------------------------
def box(ax, x0, y0, w, h, fc="white", ec=RULE, lw=LW_BOX, ls="-", r=R_BOX,
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def gradient_box(ax, x0, y0, w, h, top, bottom, ec=RULE, lw=1.1, r=R_BOX,
                 zorder=3):
    p = box(ax, x0, y0, w, h, fc="none", ec=ec, lw=lw, r=r, zorder=zorder + 2)
    ramp = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("g", [top, bottom])
    im = ax.imshow(ramp, extent=[x0, x0 + w, y0, y0 + h], cmap=cmap,
                   origin="lower", zorder=zorder, interpolation="bilinear")
    im.set_clip_path(p)
    return p


def arrow(ax, p0, p1, color=RULE, lw=1.2, ms=12, ls="-", rad=0.0, zorder=8):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


# ---- hand-drawn icons ------------------------------------------------------
def icon_gear(ax, cx, cy, r, color, teeth=8, zorder=9):
    n = teeth * 4
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.where(np.arange(n) % 4 < 2, r, r * 0.74)
    pts = np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                         zorder=zorder))
    ax.add_patch(Circle((cx, cy), r * 0.33, facecolor="white",
                        edgecolor="none", zorder=zorder + 1))


def icon_stack(ax, cx, cy, s, color, zorder=9):
    """Three offset frames -- the 'latent sequence' glyph."""
    for k, a in zip((2, 1, 0), (0.35, 0.62, 1.0)):
        ax.add_patch(FancyBboxPatch(
            (cx - s * 0.62 + k * s * 0.16, cy - s * 0.46 - k * s * 0.12),
            s * 1.0, s * 0.72,
            boxstyle="round,pad=0,rounding_size=0.02", facecolor="white",
            edgecolor=color, linewidth=0.9, alpha=a, zorder=zorder + (2 - k)))


def icon_atom(ax, cx, cy, r, color, zorder=9):
    from matplotlib.patches import Ellipse
    for a in (0, 60, 120):
        ax.add_patch(Ellipse((cx, cy), 2 * r, 0.78 * r, angle=a,
                             facecolor="none", edgecolor=color, linewidth=0.9,
                             zorder=zorder))
    ax.add_patch(Circle((cx, cy), r * 0.24, facecolor=color, edgecolor="none",
                        zorder=zorder + 1))


# ---- figure ----------------------------------------------------------------
W, H = 12.0, 3.9
Y_CAP = 3.62            # group caption baseline
FILM_X0, FILM_X1 = 6.92, 11.62
TIME_Y = 0.62


def caption(ax, x, text, color=INK, size=8.4, weight="bold"):
    ax.text(x, Y_CAP, text, ha="center", va="center", fontsize=size,
            color=color, fontweight=weight)


def draw_context(ax, frames):
    caption(ax, 1.54, "Observed context")
    for k, (rgb, dx) in enumerate(zip(frames, (0.34, 0.17, 0.0))):
        frame(ax, rgb, 0.46 + dx, 1.74 + dx * 0.62, 1.74, 1.30,
              lw=1.0, zorder=4 + k * 3)
    ax.text(1.33, 1.68, "3 frames  ·  256 px", ha="center", va="top",
            fontsize=6.4, color=MUTED)

    ax.text(1.50, 1.44, "Action tokens", ha="center", va="center",
            fontsize=7.4, color=SUB)
    cols = [lighten(BLUE, a) for a in (0.62, 0.50, 0.38, 0.26, 0.14, 0.0)]
    x = 0.46
    for k, c in enumerate(cols):
        box(ax, x, 0.92, 0.30, 0.32, fc=c, ec="white", lw=0.7, r=0.045,
            zorder=5)
        ax.text(x + 0.15, 1.08, f"$a_{k + 1}$", ha="center", va="center",
                fontsize=6.4, color="white", zorder=6)
        x += 0.345
    box(ax, 0.42, 0.88, 2.16, 0.40, fc="none", ec=lighten(BLUE, 0.45),
        lw=0.8, ls=(0, (5, 2.6)), r=0.075, zorder=4)


def draw_method(ax):
    x0, y0, w, h = 3.26, 1.24, 3.04, 1.88
    gradient_box(ax, x0, y0, w, h, lighten(GOLD, 0.80), "#ffffff", ec=GOLD,
                 lw=1.2, r=0.10)
    ax.text(x0 + w / 2, y0 + 1.58, "DreamStream", ha="center", va="center",
            fontsize=12.2, fontweight="bold", color=INK, zorder=8)
    ax.text(x0 + w / 2, y0 + 1.34, "action-conditioned video world model",
            ha="center", va="center", fontsize=7.4, color=SUB, zorder=8)

    chips = [("Tokenizer", icon_gear), ("Dynamics", icon_stack),
             ("Decoder", icon_atom)]
    cw, ch = 0.84, 0.60
    for k, (name, drawer) in enumerate(chips):
        cx = x0 + 0.52 + k * 1.00
        box(ax, cx - cw / 2, y0 + 0.52, cw, ch, fc="white", ec=HAIR, lw=0.8,
            r=0.055, zorder=7)
        if drawer is icon_stack:
            drawer(ax, cx, y0 + 0.84, 0.26, GOLD, zorder=8)
        else:
            drawer(ax, cx, y0 + 0.83, 0.16, GOLD, zorder=8)
        ax.text(cx, y0 + 0.38, name, ha="center", va="center", fontsize=6.6,
                color=SUB, zorder=8)
    ax.text(x0 + w / 2, y0 + 0.17, "24 latent frames  ·  12 denoising steps",
            ha="center", va="center", fontsize=6.6, color=MUTED, zorder=8)
    return x0, y0, w, h


def draw_film(ax, frames):
    caption(ax, (FILM_X0 + FILM_X1) / 2, "Generated rollout")
    fy0, fh = 1.78, 1.52
    box(ax, FILM_X0, fy0, FILM_X1 - FILM_X0, fh, fc=DARK, ec="none", r=0.09,
        zorder=3)
    nh = 15
    hx = np.linspace(FILM_X0 + 0.10, FILM_X1 - 0.26, nh)
    for x in hx:
        for y in (fy0 + fh - 0.20, fy0 + 0.07):
            box(ax, x, y, 0.16, 0.13, fc="#f2f2f2", ec="none", r=0.025,
                zorder=4)
    inner_y, inner_h = fy0 + 0.27, 0.98
    fw = 1.06
    xs = FILM_X0 + 0.095 + np.arange(4) * (fw + 0.09)
    labels = ["$t{+}8$", "$t{+}24$", "$t{+}40$", "$t{+}64$"]
    for x, rgb, lab in zip(xs, frames, labels):
        frame(ax, rgb, x, inner_y, fw, inner_h, ec="#f2f2f2", lw=0.8,
              r=0.045, zorder=5)
        ax.text(x + fw / 2, 1.62, lab, ha="center", va="top", fontsize=6.8,
                color=SUB)
    return xs, fw


def draw_timeline(ax, xs, fw):
    arrow(ax, (0.46, TIME_Y), (FILM_X1, TIME_Y), color=RULE, lw=1.1, ms=12,
          zorder=6)
    ticks = [(1.05, "$t{-}2$", MUTED), (1.80, "$t{-}1$", MUTED),
             (2.55, "$t$", INK)]
    ticks += [(x + fw / 2, lab, GOLD) for x, lab in
              zip(xs, ["$t{+}8$", "$t{+}24$", "$t{+}40$", "$t{+}64$"])]
    for x, lab, col in ticks:
        ax.plot([x, x], [TIME_Y, TIME_Y + 0.11], color=col, lw=1.0,
                zorder=6, solid_capstyle="butt")
        ax.text(x, 0.40, lab, ha="center", va="center", fontsize=6.6,
                color=col)
    ax.plot([6.61, 6.61], [0.50, 0.94], color=HAIR, lw=0.9, ls=(0, (3, 2.4)),
            zorder=5)
    ax.text(3.10, 0.83, "observed", ha="right", va="center", fontsize=7.0,
            color=MUTED, style="italic")
    ax.text(6.78, 0.83, "autoregressive rollout in latent space", ha="left",
            va="center", fontsize=7.0, color=MUTED, style="italic")


def main():
    set_pub_style()
    rng = np.random.default_rng(474)
    pano = panorama(rng)
    crops = [colorize(pano[:, s:s + 128], CMAP_SCENE, gain=1.15)
             for s in (0, 26, 52, 130, 186, 242, 300)]

    fig, ax = canvas(W, H)
    draw_context(ax, crops[:3][::-1])
    mx0, my0, mw, mh = draw_method(ax)
    xs, fw = draw_film(ax, crops[3:])

    arrow(ax, (2.66, 2.62), (mx0 - 0.05, 2.42), lw=1.2, ms=12)
    arrow(ax, (2.66, 1.08), (mx0 - 0.05, 1.70), lw=1.2, ms=12)
    arrow(ax, (mx0 + mw + 0.05, 2.42), (FILM_X0 - 0.05, 2.42), lw=1.4, ms=13)

    draw_timeline(ax, xs, fw)
    savefig(fig, "1069_wm_teaser_video_rollout.png")


if __name__ == "__main__":
    main()
