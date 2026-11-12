"""Driving world model teaser: surround views -> forecast occupancy + plan.

Left is the sensor rig as six procedurally synthesised surround-view frames.
The centre column is the model itself -- a gradient block fed from below by a
dashed "action conditions" container holding three hand-drawn dials.  Right
are three forecast horizons, each a top-down semantic occupancy raster with
the planned ego trajectory drawn on top and a planning-cost verdict beneath.
A full-width legend bar closes the figure.

The occupancy rasters are built as integer label maps and shown through a
ListedColormap, so the classes stay flat and readable at thumbnail size while
the camera frames stay photographic (value noise + blur + muted ramp).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Polygon)
from scipy.ndimage import gaussian_filter, zoom
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DARK = "#3a3a3a"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
PURPLE = "#8e6bbf"

CMAP_STREET = LinearSegmentedColormap.from_list(
    "wm_street", ["#2f3740", "#525d68", "#848f99", "#bcc4cb", "#e9edf0"])

# Semantic occupancy classes, flat and low-saturation.
OCC_NAMES = ["free space", "drivable", "vehicle", "pedestrian", "building",
             "vegetation"]
OCC_COLORS = ["#eef0f2", "#d2d7dc", BLUE, GOLD, lighten(PURPLE, 0.42),
              lighten(GREEN, 0.52)]
OCC_CMAP = ListedColormap(OCC_COLORS)

R_BOX = 0.08
W, H = 12.0, 4.05
Y_CAP = 3.80
BAND_TOP = 3.58


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


def street_shot(rng, vx, bl, br, bright=0.0, n_car=2, ny=124, nx=180):
    """Street frame drawn as geometry, then textured.

    ``vx`` is the vanishing point the lane edges converge on and ``bl`` / ``br``
    are where those edges cross the bottom of the frame, so the same routine
    gives a centred road for the forward views and a road sliding off to one
    side for the oblique ones.  Structure first (horizon, road wedge, painted
    lines, vehicles, roadside blocks), noise and depth of field last.
    """
    yy = np.linspace(0, 1, ny)[:, None]
    xx = np.linspace(0, 1, nx)[None, :]
    hz = 0.44
    field = np.where(yy < hz, 0.88 - 0.15 * (yy / hz),
                     0.47 - 0.05 * (yy - hz)) + 0.0 * xx

    # roadside blocks standing on the horizon
    for _ in range(5):
        cx = rng.uniform(-0.05, 1.05)
        w = rng.uniform(0.05, 0.15)
        top = hz - rng.uniform(0.10, 0.32)
        field += rng.uniform(-0.28, 0.16) * (
            _sig(80 * (xx - cx + w)) * _sig(80 * (cx + w - xx))
            * _sig(80 * (yy - top)) * _sig(80 * (hz + 0.01 - yy)))

    # road wedge converging on the vanishing point
    t = np.clip((yy - hz) / (1 - hz), 0, 1)
    xl, xr = vx + (bl - vx) * t, vx + (br - vx) * t
    road = _sig(70 * (xx - xl)) * _sig(70 * (xr - xx)) * _sig(70 * (yy - hz))
    field = field * (1 - road) + road * (0.34 + 0.07 * t)
    for e in (xl, xr):
        field += 0.26 * road * np.exp(-((xx - e) / (0.008 + 0.022 * t)) ** 2)
    xc = 0.5 * (xl + xr)
    dash = (np.sin(2 * np.pi * 5.0 * t ** 0.7) > 0.15).astype(float)
    field += 0.30 * road * dash * np.exp(
        -((xx - xc) / (0.006 + 0.020 * t)) ** 2)

    # vehicles sitting on the road, scaled by depth
    for k in range(n_car):
        d = float(np.clip(0.26 + 0.32 * k + rng.uniform(-0.07, 0.07),
                          0.14, 0.86))
        ty = hz + d * (1 - hz)
        s = 0.05 + 0.26 * d
        cxk = float(xc[int(ty * (ny - 1)), 0]) + rng.uniform(-0.55, 0.55) * s
        car = (_sig(90 * (s * 0.60 - np.abs(xx - cxk)))
               * _sig(90 * (yy - (ty - s * 0.58))) * _sig(90 * (ty - yy)))
        field -= 0.36 * car
        field += 0.18 * car * np.exp(
            -((yy - (ty - s * 0.52)) / (0.008 + 0.020 * s)) ** 2)

    field += 0.10 * (value_noise(rng, ny, nx, base=4) - 0.5)
    far = gaussian_filter(field, 1.7)
    near = gaussian_filter(field, 0.55)
    dof = _sig(20 * (hz + 0.03 - yy))
    field = far * dof + near * (1 - dof)
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return np.clip(0.05 + 0.90 * field + bright, 0, 1)


def colorize(field, cmap, vignette=0.28, gain=1.12):
    ny, nx = field.shape
    yy = np.linspace(-1, 1, ny)[:, None]
    xx = np.linspace(-1, 1, nx)[None, :]
    rgb = cmap(np.clip((field - 0.5) * gain + 0.5, 0, 1))[..., :3]
    vig = 1.0 - vignette * (xx ** 2 + yy ** 2) / 2.0
    return np.clip(rgb * vig[..., None], 0, 1)


# ---- occupancy raster ------------------------------------------------------
def occupancy(rng, step, ny=84, nx=66):
    """Top-down semantic label map; traffic advances with ``step``."""
    lab = np.full((ny, nx), 5, dtype=int)                 # vegetation
    for _ in range(9):                                    # building blocks
        by, bx = rng.integers(2, ny - 14), rng.integers(1, nx - 12)
        lab[by:by + rng.integers(7, 15), bx:bx + rng.integers(6, 12)] = 4
    lab[:, 24:42] = 1                                     # main corridor
    lab[36:52, :] = 1                                     # cross street
    lab[:, 32:34] = 0                                     # centre line
    lab[43:45, :] = 0
    for k in range(4):                                    # other vehicles
        vy = (14 + 19 * k + 7 * step) % (ny - 10)
        vx = 26 if k % 2 else 36
        lab[vy:vy + 7, vx:vx + 4] = 2
    lab[30 - 2 * step:34 - 2 * step, 44:47] = 2           # crossing car
    for k in range(3):                                    # pedestrians
        py = 38 + 3 * k
        px = 18 + 2 * step + 4 * k
        lab[py:py + 2, px:px + 2] = 3
    return lab


def trajectory(step):
    """Planned path in panel-normalised coordinates, longer each horizon."""
    reach = 0.30 + 0.20 * step
    t = np.linspace(0, 1, 90)
    x = 0.500 + 0.085 * np.sin(np.pi * t) * (1 + 0.35 * step)
    y = 0.940 - reach * t
    return x, y


# ---- primitives ------------------------------------------------------------
def box(ax, x0, y0, w, h, fc="white", ec=RULE, lw=0.9, ls="-", r=R_BOX,
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def raster(ax, arr, x0, y0, w, h, cmap=None, ec="white", lw=1.0, r=0.05,
           zorder=5, **kw):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 3)
    ax.add_patch(clip)
    im = ax.imshow(arr, extent=[x0, x0 + w, y0, y0 + h], cmap=cmap,
                   zorder=zorder, **kw)
    im.set_clip_path(clip)
    return clip


def gradient_box(ax, x0, y0, w, h, top, bottom, ec=RULE, lw=1.2, r=0.10,
                 zorder=3):
    p = box(ax, x0, y0, w, h, fc="none", ec=ec, lw=lw, r=r, zorder=zorder + 2)
    ramp = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("g", [top, bottom])
    im = ax.imshow(ramp, extent=[x0, x0 + w, y0, y0 + h], cmap=cmap,
                   origin="lower", zorder=zorder, interpolation="bilinear")
    im.set_clip_path(p)
    return p


def arrow(ax, p0, p1, color=RULE, lw=1.3, ms=12, zorder=9):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                        zorder=zorder)
    ax.add_patch(a)
    return a


def caption(ax, x, text):
    ax.text(x, Y_CAP, text, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)


def tick_mark(ax, x, y, s=0.055, color=GREEN, lw=1.4, zorder=9):
    ax.plot([x - s, x - s * 0.25, x + s], [y, y - s * 0.8, y + s * 0.85],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def cross_mark(ax, x, y, s=0.048, color=RED, lw=1.4, zorder=9):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)


# ---- hand-drawn dials ------------------------------------------------------
def icon_speedo(ax, cx, cy, r, color=BLUE, zorder=8):
    th = np.linspace(np.pi * 1.15, -np.pi * 0.15, 60)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=color, lw=1.0,
            zorder=zorder)
    for t in np.linspace(np.pi * 1.15, -np.pi * 0.15, 5):
        ax.plot([cx + r * 0.78 * np.cos(t), cx + r * np.cos(t)],
                [cy + r * 0.78 * np.sin(t), cy + r * np.sin(t)], color=color,
                lw=0.7, zorder=zorder)
    a = np.pi * 0.30
    ax.plot([cx, cx + r * 0.72 * np.cos(a)], [cy, cy + r * 0.72 * np.sin(a)],
            color=RED, lw=1.1, zorder=zorder + 1)
    ax.add_patch(Circle((cx, cy), r * 0.13, facecolor=color, edgecolor="none",
                        zorder=zorder + 2))


def icon_wheel(ax, cx, cy, r, color=BLUE, zorder=8):
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=color,
                        linewidth=1.4, zorder=zorder))
    for a in (np.pi * 0.5, np.pi * 1.17, np.pi * 1.83):
        ax.plot([cx, cx + r * 0.86 * np.cos(a)],
                [cy, cy + r * 0.86 * np.sin(a)], color=color, lw=1.0,
                zorder=zorder)
    ax.add_patch(Circle((cx, cy), r * 0.26, facecolor=color, edgecolor="none",
                        zorder=zorder + 1))


def icon_route(ax, cx, cy, r, color=BLUE, zorder=8):
    xs = np.array([-1.0, -0.35, -0.35, 0.45, 0.45]) * r
    ys = np.array([-0.9, -0.9, 0.15, 0.15, 0.85]) * r
    ax.plot(cx + xs, cy + ys, color=color, lw=1.1, solid_joinstyle="miter",
            zorder=zorder)
    ax.add_patch(Circle((cx + xs[0], cy + ys[0]), r * 0.15, facecolor=color,
                        edgecolor="none", zorder=zorder + 1))
    ax.plot([cx + xs[-1], cx + xs[-1]], [cy + ys[-1], cy + ys[-1] + r * 0.30],
            color=color, lw=0.9, zorder=zorder)
    ax.add_patch(Polygon([(cx + xs[-1], cy + ys[-1] + r * 0.30),
                          (cx + xs[-1] + r * 0.42, cy + ys[-1] + r * 0.20),
                          (cx + xs[-1], cy + ys[-1] + r * 0.10)], closed=True,
                         facecolor=RED, edgecolor="none", zorder=zorder + 1))


# ---- panels ----------------------------------------------------------------
# name, vanishing point, left / right lane edge at the bottom, exposure, cars
CAM_VIEWS = [("FRONT LEFT", 0.98, -0.55, 0.80, 0.03, 2),
             ("FRONT", 0.52, -0.30, 1.30, 0.00, 3),
             ("FRONT RIGHT", 0.04, 0.20, 1.55, -0.02, 2),
             ("BACK LEFT", 0.92, -0.45, 0.90, -0.06, 1),
             ("BACK", 0.46, -0.35, 1.35, -0.09, 2),
             ("BACK RIGHT", 0.08, 0.10, 1.50, -0.04, 2)]


def draw_cameras(ax, rng):
    cw, chh, gx = 0.86, 0.60, 0.07
    x_left = 0.38
    caption(ax, x_left + (3 * cw + 2 * gx) / 2, "Surround-view input")
    for k, (name, vx, bl, br, bright, n_car) in enumerate(CAM_VIEWS):
        r_, c_ = divmod(k, 3)
        x = x_left + c_ * (cw + gx)
        y = BAND_TOP - chh - r_ * (chh + 0.29)
        rgb = colorize(street_shot(rng, vx, bl, br, bright, n_car),
                       CMAP_STREET, vignette=0.22, gain=1.05)
        raster(ax, rgb, x, y, cw, chh, ec="white", lw=0.9, r=0.04, zorder=5,
               interpolation="bilinear")
        ax.text(x + cw / 2, y - 0.10, name, ha="center", va="center",
                fontsize=6.0, color=MUTED)
    ax.text(x_left + (3 * cw + 2 * gx) / 2, 1.72,
            "6 cameras  ·  12 Hz  ·  1600 x 900", ha="center", va="center",
            fontsize=7.0, color=MUTED)
    return x_left + 3 * cw + 2 * gx


def draw_model(ax):
    x0, x1 = 3.55, 6.45
    caption(ax, (x0 + x1) / 2, "Driving world model")
    by0, bh = 2.52, 1.06
    gradient_box(ax, x0, by0, x1 - x0, bh, lighten(GOLD, 0.80), "#ffffff",
                 ec=GOLD, lw=1.2, r=0.10)
    ax.text((x0 + x1) / 2, by0 + 0.72, "OccDreamer", ha="center", va="center",
            fontsize=12.0, fontweight="bold", color=INK, zorder=8)
    ax.text((x0 + x1) / 2, by0 + 0.48, "BEV latent dynamics + planner",
            ha="center", va="center", fontsize=7.4, color=SUB, zorder=8)
    ax.text((x0 + x1) / 2, by0 + 0.22, "200 x 200 grid  ·  0.5 m cells",
            ha="center", va="center", fontsize=6.6, color=MUTED, zorder=8)

    cy0, chh = 1.42, 0.84
    box(ax, x0, cy0, x1 - x0, chh, fc=lighten(BLUE, 0.95),
        ec=lighten(BLUE, 0.45), lw=1.0, ls=(0, (5, 3)), r=0.09, zorder=3)
    ax.text((x0 + x1) / 2, cy0 + chh - 0.14, "action conditions", ha="center",
            va="center", fontsize=7.0, color=BLUE, zorder=6)
    dials = [("velocity", icon_speedo), ("steering", icon_wheel),
             ("command", icon_route)]
    for k, (name, drawer) in enumerate(dials):
        cx = x0 + 0.60 + k * 0.85
        drawer(ax, cx, cy0 + 0.42, 0.15)
        ax.text(cx, cy0 + 0.13, name, ha="center", va="center", fontsize=6.4,
                color=SUB, zorder=8)
    arrow(ax, ((x0 + x1) / 2, cy0 + chh + 0.03), ((x0 + x1) / 2, by0 - 0.03),
          lw=1.2, ms=11)
    return x0, x1, by0, bh


def draw_forecasts(ax, rng):
    pw, gap = 1.46, 0.16
    x0 = 6.95
    ph = 1.86
    caption(ax, x0 + (3 * pw + 2 * gap) / 2, "Forecast occupancy + plan")
    verdicts = [("cost 0.42", True), ("cost 0.61", True),
                ("collision", False)]
    for k in range(3):
        x = x0 + k * (pw + gap)
        y = BAND_TOP - ph
        clip = raster(ax, occupancy(rng, k), x, y, pw, ph, cmap=OCC_CMAP,
                      ec="white", lw=1.1, r=0.05, zorder=5,
                      interpolation="nearest", vmin=0, vmax=5)
        tx, ty = trajectory(k)
        px, py = x + tx * pw, y + (1 - ty) * ph
        halo, = ax.plot(px, py, color="white", lw=3.0, zorder=9,
                        solid_capstyle="round")
        line, = ax.plot(px, py, color=RED, lw=1.6, zorder=10,
                        solid_capstyle="round")
        halo.set_clip_path(clip)
        line.set_clip_path(clip)
        for f in (0.34, 0.67, 1.0):
            i = int(f * (len(px) - 1))
            dot = Circle((px[i], py[i]), 0.032, facecolor="white",
                         edgecolor=RED, linewidth=0.9, zorder=11)
            ax.add_patch(dot)
            dot.set_clip_path(clip)
        ego = Polygon([(x + 0.50 * pw, y + 0.115 * ph),
                       (x + 0.455 * pw, y + 0.045 * ph),
                       (x + 0.545 * pw, y + 0.045 * ph)], closed=True,
                      facecolor=RED, edgecolor="white", linewidth=0.7,
                      zorder=11)
        ax.add_patch(ego)
        ax.text(x + pw / 2, 1.56, f"$t$ + {k + 1}.0 s", ha="center",
                va="center", fontsize=8.0, fontweight="bold", color=INK)
        lab, ok = verdicts[k]
        box(ax, x + 0.24, 1.20, pw - 0.48, 0.26,
            fc=lighten(GREEN if ok else RED, 0.88), ec="none", r=0.05,
            zorder=4)
        if ok:
            tick_mark(ax, x + 0.40, 1.33)
        else:
            cross_mark(ax, x + 0.40, 1.33)
        ax.text(x + 0.54, 1.33, lab, ha="left", va="center", fontsize=6.8,
                color=GREEN if ok else RED, zorder=6)
    return x0, 3 * pw + 2 * gap


def draw_legend(ax):
    ly0, lh = 0.28, 0.52
    box(ax, 0.38, ly0, 11.24, lh, fc="#f5f5f5", ec=HAIR, lw=0.8, r=0.07,
        zorder=2)
    ax.text(0.62, ly0 + lh / 2, "Occupancy classes", ha="left", va="center",
            fontsize=7.2, fontweight="bold", color=INK, zorder=4)
    x = 2.28
    for name, col in zip(OCC_NAMES, OCC_COLORS):
        box(ax, x, ly0 + lh / 2 - 0.075, 0.15, 0.15, fc=col, ec=HAIR, lw=0.6,
            r=0.025, zorder=4)
        ax.text(x + 0.22, ly0 + lh / 2, name, ha="left", va="center",
                fontsize=6.8, color=SUB, zorder=4)
        x += 0.24 + 0.062 * len(name) + 0.32
    ax.plot([9.95, 10.35], [ly0 + lh / 2, ly0 + lh / 2], color=RED, lw=1.6,
            zorder=4)
    ax.add_patch(Circle((10.15, ly0 + lh / 2), 0.032, facecolor="white",
                        edgecolor=RED, linewidth=0.9, zorder=5))
    ax.text(10.45, ly0 + lh / 2, "planned ego trajectory", ha="left",
            va="center", fontsize=6.8, color=SUB, zorder=4)


def main():
    set_pub_style()
    rng = np.random.default_rng(1477)
    fig, ax = canvas(W, H)

    cam_right = draw_cameras(ax, rng)
    mx0, mx1, by0, bh = draw_model(ax)
    fx0, _ = draw_forecasts(ax, rng)

    arrow(ax, (cam_right + 0.08, 2.90), (mx0 - 0.06, 3.00), lw=1.3, ms=13)
    arrow(ax, (mx1 + 0.06, by0 + bh / 2), (fx0 - 0.07, by0 + bh / 2), lw=1.4,
          ms=13)
    ax.text((mx1 + fx0) / 2, by0 + bh / 2 + 0.20, "roll out", ha="center",
            va="center", fontsize=6.8, color=MUTED)
    draw_legend(ax)

    savefig(fig, "1072_wm_teaser_occ_forecast.png")


if __name__ == "__main__":
    main()
