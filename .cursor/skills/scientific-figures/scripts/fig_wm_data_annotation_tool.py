"""Screenshot-style mock-up of the annotation tool used to label the corpus.

Every widget is drawn from matplotlib primitives: rounded rectangles with a
white-to-grey vertical ramp for the raised look, hairline borders, hand-built
icons (cursor, box, polygon, brush, eraser, point, zoom, undo, redo), a layer
list with eye and lock glyphs, dropdowns, sliders with round knobs, check
boxes with a stroked tick, radio buttons and a progress bar.

The document under the cursor is a procedurally synthesised worktop scene
(signed-distance hob, pan, cutting board, jar and knife, multi-scale value
noise, shallow depth of field).  One box is mid-drag with its eight handles
live and a size read-out attached; the two committed boxes keep the muted tag
style of the finished annotations.

Laid out in inches on an equal-aspect canvas so every corner radius, border
weight and gap is literally the same number everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (Arc, Circle, FancyBboxPatch, Polygon,
                                Rectangle)
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
LINE = "#cccccc"
HAIR = "#d5d5d5"
CHROME = "#f0f0f0"
SHELL = "#f5f5f5"
DARK = "#3a3a3a"

UI = "#3b5b92"          # selection / active widget
LIVE = "#b23a48"        # the object currently being dragged
DONE = ["#c9a227", "#2a7f7a"]

SEED = 1535  # original id; figure number is 1130
AR = 1.4516

W, H = 10.4, 6.2
WIN = (0.28, 0.24, 9.84, 5.36)          # x, y, w, h
TITLEBAR_Y, MENU_Y, TOOL_Y = 5.24, 4.98, 4.52
MAIN_Y0, STATUS_Y1 = 0.62, 0.62
LEFT_X1, RIGHT_X0 = 2.05, 7.30
IMG_W = 4.95
IMG_H = IMG_W / AR
IMG_X0 = LEFT_X1 + (RIGHT_X0 - LEFT_X1 - IMG_W) / 2
IMG_Y1 = MAIN_Y0 + (TOOL_Y - MAIN_Y0 + IMG_H) / 2
RAD = 0.055


# ---- procedural document ---------------------------------------------------
def _blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(int(3 * sigma), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, p)
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, p)


def _blur3(a, sigma):
    return np.dstack([_blur(a[..., c], sigma) for c in range(a.shape[2])])


def _value_noise(rng, ny, nx, octaves=(3, 6, 12, 24),
                 weights=(1.0, .55, .32, .18)):
    out = np.zeros((ny, nx))
    for o, wt in zip(octaves, weights):
        g = rng.random((o, o))
        ty = np.linspace(0, o - 1, ny)
        tx = np.linspace(0, o - 1, nx)
        iy0, ix0 = np.floor(ty).astype(int), np.floor(tx).astype(int)
        iy1 = np.minimum(iy0 + 1, o - 1)
        ix1 = np.minimum(ix0 + 1, o - 1)
        fy, fx = (ty - iy0)[:, None], (tx - ix0)[None, :]
        top = g[iy0][:, ix0] * (1 - fx) + g[iy0][:, ix1] * fx
        bot = g[iy1][:, ix0] * (1 - fx) + g[iy1][:, ix1] * fx
        out += wt * (top * (1 - fy) + bot * fy)
    return (out - out.min()) / max(np.ptp(out), 1e-9)


def _capsule(xx, yy, p0, p1, r):
    bx, by = p1[0] - p0[0], p1[1] - p0[1]
    px, py = xx - p0[0], yy - p0[1]
    t = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-9), 0.0, 1.0)
    return np.hypot(px - t * bx, py - t * by) - r


def _rbox(xx, yy, cx, cy, hw, hh, r):
    qx = np.abs(xx - cx) - (hw - r)
    qy = np.abs(yy - cy) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def _cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


def worktop_scene(seed=SEED, ny=210):
    """Overhead worktop: hob with a pan, cutting board, knife and a jar."""
    nx = int(round(ny * AR))
    rng = np.random.default_rng(seed)
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * AR
    soft = 2.2 / ny

    top = np.array([0.700, 0.675, 0.635])
    back = np.array([0.560, 0.560, 0.545])
    img = np.where((yy < 0.20)[..., None],
                   back * (0.94 + 0.20 * yy / 0.20)[..., None],
                   top * (1.06 - 0.20 * (yy - 0.20) / 0.80)[..., None])
    tile = _cover(np.abs(yy - 0.20) - 0.005, soft)
    img *= (1.0 - 0.26 * tile)[..., None]

    hob = _rbox(xx, yy, 0.42, 0.545, 0.325, 0.290, 0.045)
    pan_out = np.hypot(xx - 0.420, yy - 0.520) - 0.205
    pan_in = np.hypot(xx - 0.420, yy - 0.520) - 0.158
    handle = _capsule(xx, yy, (0.230, 0.545), (0.020, 0.505), 0.026)
    board = _rbox(xx, yy, 1.075, 0.615, 0.300, 0.225, 0.038)
    veg_a = np.hypot(xx - 0.960, yy - 0.560) - 0.057
    veg_b = np.hypot(xx - 1.155, yy - 0.640) - 0.049
    blade = _capsule(xx, yy, (0.880, 0.815), (1.215, 0.782), 0.017)
    grip = _capsule(xx, yy, (1.225, 0.781), (1.330, 0.771), 0.021)
    jar = np.minimum(_rbox(xx, yy, 1.315, 0.335, 0.070, 0.095, 0.026),
                     _rbox(xx, yy, 1.315, 0.243, 0.048, 0.020, 0.008))

    layers = [(hob, np.array([0.300, 0.305, 0.320])),
              (board, np.array([0.660, 0.520, 0.360])),
              (handle, np.array([0.230, 0.235, 0.250])),
              (pan_out, np.array([0.345, 0.350, 0.365])),
              (pan_in, np.array([0.470, 0.420, 0.315])),
              (veg_a, np.array([0.760, 0.450, 0.215])),
              (veg_b, np.array([0.400, 0.520, 0.300])),
              (blade, np.array([0.760, 0.770, 0.785])),
              (grip, np.array([0.290, 0.270, 0.255])),
              (jar, np.array([0.620, 0.660, 0.660]))]
    shade = (1.16 - 0.26 * yy)[..., None]
    subject = np.zeros_like(yy)
    for sdf, col in layers:
        m = _cover(sdf, soft)
        img = img * (1 - m[..., None]) + col * shade * m[..., None]
        subject = np.clip(subject + m, 0, 1)

    ring = _cover(np.abs(np.hypot(xx - 0.420, yy - 0.520) - 0.182) - 0.008,
                  soft)
    img *= (1.0 - 0.20 * ring)[..., None]

    tex = 1.0 + 0.070 * (_value_noise(rng, ny, nx) - 0.5) * 2.0
    grain = 1.0 + 0.019 * rng.normal(0.0, 1.0, (ny, nx))
    img *= (tex * grain)[..., None]

    sharp = _blur(subject, 1.1)[..., None]
    img = _blur3(img, 0.35) * sharp + _blur3(img, 1.7) * (1.0 - sharp)
    r2 = ((xx - AR / 2) / AR) ** 2 + (yy - 0.5) ** 2
    img *= (1.0 - 0.30 * r2)[..., None]
    return np.clip(img, 0.0, 1.0)


# ---- widget primitives -----------------------------------------------------
def raised(ax, x, y, w, h, ec=LINE, lw=0.7, r=RAD, top="white",
           bottom=CHROME, zorder=3):
    """Rounded rectangle with a vertical white-to-grey ramp."""
    base = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(base)
    cmap = LinearSegmentedColormap.from_list("ui", [bottom, top])
    im = ax.imshow(np.linspace(0, 1, 96).reshape(-1, 1),
                   extent=(x, x + w, y, y + h), origin="lower", cmap=cmap,
                   vmin=0, vmax=1, zorder=zorder + 0.1,
                   interpolation="bilinear", aspect="auto")
    im.set_clip_path(base)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.2))
    return base


def flat(ax, x, y, w, h, fc=CHROME, ec="none", lw=0.7, r=RAD, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=zorder))


def hline(ax, x0, x1, y, color=HAIR, lw=0.7, zorder=4):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=zorder)


# ---- hand-drawn icons ------------------------------------------------------
def ic_cursor(ax, cx, cy, s, c):
    pts = [(cx - .28 * s, cy + .46 * s), (cx - .28 * s, cy - .42 * s),
           (cx - .02 * s, cy - .16 * s), (cx + .16 * s, cy - .44 * s),
           (cx + .30 * s, cy - .36 * s), (cx + .12 * s, cy - .10 * s),
           (cx + .34 * s, cy - .06 * s)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=c, edgecolor=c,
                         linewidth=0.5, joinstyle="miter", zorder=6))


def ic_box(ax, cx, cy, s, c):
    ax.add_patch(Rectangle((cx - .40 * s, cy - .32 * s), .80 * s, .64 * s,
                           facecolor="none", edgecolor=c, linewidth=1.0,
                           zorder=6))
    for dx in (-.40, .40):
        for dy in (-.32, .32):
            ax.add_patch(Rectangle((cx + dx * s - .07 * s,
                                    cy + dy * s - .07 * s), .14 * s, .14 * s,
                                   facecolor=c, edgecolor="none", zorder=7))


def ic_poly(ax, cx, cy, s, c):
    ang = np.deg2rad([90, 162, 234, 306, 18])
    pts = [(cx + .42 * s * np.cos(a), cy + .42 * s * np.sin(a)) for a in ang]
    ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=c,
                         linewidth=1.0, zorder=6))
    for px, py in pts:
        ax.add_patch(Circle((px, py), .075 * s, facecolor=c, edgecolor="none",
                            zorder=7))


def ic_brush(ax, cx, cy, s, c):
    ax.plot([cx - .26 * s, cx + .34 * s], [cy - .30 * s, cy + .34 * s],
            color=c, lw=1.6, solid_capstyle="round", zorder=6)
    ax.add_patch(Polygon([(cx - .40 * s, cy - .42 * s),
                          (cx - .12 * s, cy - .34 * s),
                          (cx - .32 * s, cy - .14 * s)], closed=True,
                         facecolor=c, edgecolor="none", zorder=6))


def ic_eraser(ax, cx, cy, s, c):
    ax.add_patch(Polygon([(cx - .44 * s, cy - .12 * s),
                          (cx - .06 * s, cy + .40 * s),
                          (cx + .44 * s, cy + .06 * s),
                          (cx + .06 * s, cy - .42 * s)], closed=True,
                         facecolor="none", edgecolor=c, linewidth=1.0,
                         zorder=6))
    ax.plot([cx - .25 * s, cx + .25 * s], [cy + .14 * s, cy - .12 * s],
            color=c, lw=0.8, zorder=6)


def ic_point(ax, cx, cy, s, c):
    for a in np.deg2rad([0, 90, 180, 270]):
        ax.plot([cx + .16 * s * np.cos(a), cx + .44 * s * np.cos(a)],
                [cy + .16 * s * np.sin(a), cy + .44 * s * np.sin(a)],
                color=c, lw=1.0, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle((cx, cy), .11 * s, facecolor=c, edgecolor="none",
                        zorder=6))


def ic_zoom(ax, cx, cy, s, c):
    ax.add_patch(Circle((cx - .08 * s, cy + .10 * s), .28 * s,
                        facecolor="none", edgecolor=c, linewidth=1.0,
                        zorder=6))
    ax.plot([cx + .12 * s, cx + .38 * s], [cy - .10 * s, cy - .36 * s],
            color=c, lw=1.3, solid_capstyle="round", zorder=6)


def _curved(ax, cx, cy, s, c, mirror=False):
    """Undo / redo glyph: a flat arc with a head sitting on its own tip."""
    sgn = -1.0 if mirror else 1.0
    ax.add_patch(Arc((cx, cy - .10 * s), .74 * s, .58 * s, angle=0,
                     theta1=20, theta2=160, edgecolor=c, linewidth=1.1,
                     zorder=6))
    px = cx - sgn * .348 * s
    py = cy - .10 * s + .099 * s
    ax.add_patch(Polygon([(px - sgn * .04 * s, py - .21 * s),
                          (px - sgn * .20 * s, py + .07 * s),
                          (px + sgn * .12 * s, py + .07 * s)], closed=True,
                         facecolor=c, edgecolor="none", zorder=6))


def ic_undo(ax, cx, cy, s, c):
    _curved(ax, cx, cy, s, c, mirror=False)


def ic_redo(ax, cx, cy, s, c):
    _curved(ax, cx, cy, s, c, mirror=True)


def ic_eye(ax, cx, cy, s, c, crossed=False):
    t = np.linspace(-1, 1, 40)
    ax.plot(cx + .48 * s * t, cy + .30 * s * (1 - t ** 2), color=c, lw=0.8,
            zorder=6)
    ax.plot(cx + .48 * s * t, cy - .30 * s * (1 - t ** 2), color=c, lw=0.8,
            zorder=6)
    ax.add_patch(Circle((cx, cy), .13 * s, facecolor=c, edgecolor="none",
                        zorder=6))
    if crossed:
        ax.plot([cx - .44 * s, cx + .44 * s], [cy - .36 * s, cy + .36 * s],
                color=c, lw=0.9, zorder=7)


def ic_lock(ax, cx, cy, s, c):
    ax.add_patch(Rectangle((cx - .26 * s, cy - .34 * s), .52 * s, .42 * s,
                           facecolor="none", edgecolor=c, linewidth=0.8,
                           zorder=6))
    ax.add_patch(Arc((cx, cy + .08 * s), .34 * s, .36 * s, theta1=0,
                     theta2=180, edgecolor=c, linewidth=0.8, zorder=6))


def tick(ax, cx, cy, s, c):
    ax.plot([cx - .30 * s, cx - .06 * s, cx + .32 * s],
            [cy + .02 * s, cy - .26 * s, cy + .30 * s], color=c, lw=1.3,
            solid_capstyle="round", solid_joinstyle="miter", zorder=7)


def caret(ax, cx, cy, s, c):
    ax.add_patch(Polygon([(cx - .30 * s, cy + .14 * s),
                          (cx + .30 * s, cy + .14 * s), (cx, cy - .22 * s)],
                         closed=True, facecolor=c, edgecolor="none",
                         zorder=7))


# ---- panels ----------------------------------------------------------------
def chrome(ax):
    flat(ax, WIN[0], WIN[1], WIN[2], WIN[3], fc=SHELL, ec=LINE, lw=0.9,
         r=0.09, zorder=1)
    flat(ax, WIN[0], TITLEBAR_Y, WIN[2], WIN[1] + WIN[3] - TITLEBAR_Y,
         fc=CHROME, ec="none", r=0.09, zorder=2)
    flat(ax, WIN[0], TITLEBAR_Y, WIN[2], 0.10, fc=CHROME, ec="none", r=0.0,
         zorder=2)
    hline(ax, WIN[0], WIN[0] + WIN[2], TITLEBAR_Y)
    ax.text(WIN[0] + 0.18, TITLEBAR_Y + 0.18,
            "VideoAnnotator  3.1   —   session S014 / clip 0271.mp4",
            ha="left", va="center", fontsize=7.6, fontweight="bold",
            color=RULE, zorder=5)
    for k in range(3):
        ax.add_patch(Circle((WIN[0] + WIN[2] - 0.62 + 0.20 * k,
                             TITLEBAR_Y + 0.18), 0.048, facecolor="white",
                            edgecolor=MUTED, linewidth=0.7, zorder=5))

    for k, name in enumerate(["File", "Edit", "View", "Label", "Track",
                              "Help"]):
        ax.text(WIN[0] + 0.20 + 0.66 * k, MENU_Y + 0.13, name, ha="left",
                va="center", fontsize=7.0, color=SUB, zorder=5)
    hline(ax, WIN[0], WIN[0] + WIN[2], MENU_Y)

    icons = [ic_cursor, ic_box, ic_poly, ic_brush, ic_eraser, ic_point,
             ic_zoom, ic_undo, ic_redo]
    x = WIN[0] + 0.20
    for k, fn in enumerate(icons):
        if k in (3, 6, 7):
            ax.plot([x + 0.03, x + 0.03], [TOOL_Y + 0.09, MENU_Y - 0.09],
                    color=HAIR, lw=0.7, zorder=4)
            x += 0.14
        active = (k == 1)
        if active:
            flat(ax, x, TOOL_Y + 0.06, 0.34, 0.34, fc=lighten(UI, 0.80),
                 ec=UI, lw=0.9, r=0.05, zorder=3)
        else:
            raised(ax, x, TOOL_Y + 0.06, 0.34, 0.34, r=0.05)
        fn(ax, x + 0.17, TOOL_Y + 0.23, 0.30, UI if active else RULE)
        x += 0.40
    ax.text(WIN[0] + WIN[2] - 0.20, TOOL_Y + 0.23,
            "B  box      P  polygon      SPACE  next frame", ha="right",
            va="center", fontsize=6.6, color=FAINT, zorder=5)
    hline(ax, WIN[0], WIN[0] + WIN[2], TOOL_Y)

    ax.plot([LEFT_X1, LEFT_X1], [MAIN_Y0, TOOL_Y], color=HAIR, lw=0.7,
            zorder=4)
    ax.plot([RIGHT_X0, RIGHT_X0], [MAIN_Y0, TOOL_Y], color=HAIR, lw=0.7,
            zorder=4)
    hline(ax, WIN[0], WIN[0] + WIN[2], STATUS_Y1)


def layer_panel(ax):
    flat(ax, WIN[0], MAIN_Y0, LEFT_X1 - WIN[0], TOOL_Y - MAIN_Y0, fc=SHELL,
         ec="none", r=0.0, zorder=2)
    ax.text(WIN[0] + 0.16, TOOL_Y - 0.20, "Layers", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=RULE, zorder=5)
    raised(ax, LEFT_X1 - 0.36, TOOL_Y - 0.30, 0.20, 0.20, r=0.04)
    ax.plot([LEFT_X1 - 0.32, LEFT_X1 - 0.20], [TOOL_Y - 0.20, TOOL_Y - 0.20],
            color=RULE, lw=0.9, zorder=6)
    ax.plot([LEFT_X1 - 0.26, LEFT_X1 - 0.26], [TOOL_Y - 0.26, TOOL_Y - 0.14],
            color=RULE, lw=0.9, zorder=6)

    rows = [("pan", LIVE, False, False, True),
            ("cutting board", DONE[0], False, True, False),
            ("jar", DONE[1], False, False, False),
            ("hob", MUTED, True, True, False),
            ("worktop", MUTED, True, True, False)]
    y = TOOL_Y - 0.46
    for name, col, hidden, locked, sel in rows:
        if sel:
            flat(ax, WIN[0] + 0.06, y - 0.135, LEFT_X1 - WIN[0] - 0.12, 0.27,
                 fc=lighten(UI, 0.86), ec=lighten(UI, 0.60), lw=0.7, r=0.04,
                 zorder=3)
        ic_eye(ax, WIN[0] + 0.22, y, 0.20, FAINT if hidden else RULE,
               crossed=hidden)
        ax.add_patch(Rectangle((WIN[0] + 0.38, y - 0.055), 0.11, 0.11,
                               facecolor=col, edgecolor="white",
                               linewidth=0.5, zorder=6))
        ax.text(WIN[0] + 0.56, y, name, ha="left", va="center", fontsize=6.8,
                color=INK if sel else SUB, zorder=6,
                fontweight="bold" if sel else "normal")
        if locked:
            ic_lock(ax, LEFT_X1 - 0.20, y, 0.20, FAINT)
        y -= 0.30

    hline(ax, WIN[0] + 0.10, LEFT_X1 - 0.10, y + 0.04)
    ax.text(WIN[0] + 0.16, y - 0.18, "Track", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=RULE, zorder=5)
    ax.text(WIN[0] + 0.16, y - 0.40, "id  obj_0271_03", ha="left",
            va="center", fontsize=6.4, family="monospace", color=SUB,
            zorder=5)
    ax.text(WIN[0] + 0.16, y - 0.60, "linked over 46 frames", ha="left",
            va="center", fontsize=6.4, color=MUTED, zorder=5)
    # miniature timeline of the track
    tx0, tx1, ty = WIN[0] + 0.16, LEFT_X1 - 0.16, y - 0.82
    ax.plot([tx0, tx1], [ty, ty], color=HAIR, lw=2.6,
            solid_capstyle="butt", zorder=5)
    ax.plot([tx0, tx0 + 0.62 * (tx1 - tx0)], [ty, ty], color=UI, lw=2.6,
            solid_capstyle="butt", zorder=6)
    for f in (0.18, 0.34, 0.51, 0.62, 0.78):
        ax.plot([tx0 + f * (tx1 - tx0)] * 2, [ty - 0.055, ty + 0.055],
                color=RULE, lw=0.6, zorder=7)
    ax.text(tx0, ty - 0.17, "frame 118 of 300", ha="left", va="center",
            fontsize=6.2, color=FAINT, zorder=5)


def canvas_panel(ax):
    flat(ax, LEFT_X1, MAIN_Y0, RIGHT_X0 - LEFT_X1, TOOL_Y - MAIN_Y0,
         fc=DARK, ec="none", r=0.0, zorder=2)
    ax.imshow(worktop_scene(), extent=(IMG_X0, IMG_X0 + IMG_W,
                                       IMG_Y1 - IMG_H, IMG_Y1),
              origin="upper", interpolation="bilinear", zorder=3)

    def px(u):
        return IMG_X0 + u * IMG_W / AR

    def py(v):
        return IMG_Y1 - v * IMG_W / AR

    # committed boxes with muted tags
    for (x0, x1, y0, y1), col, name in [
            ((0.745, 1.400, 0.365, 0.860), DONE[0], "cutting board"),
            ((1.228, 1.408, 0.215, 0.450), DONE[1], "jar")]:
        ax.add_patch(Rectangle((px(x0), py(y1)), (x1 - x0) * IMG_W / AR,
                               (y1 - y0) * IMG_W / AR, facecolor="none",
                               edgecolor=col, linewidth=1.1, zorder=5))
        ax.text(px(x0) + 0.05, py(y0) - 0.09, name, ha="left", va="center",
                fontsize=6.2, color="white", zorder=7,
                bbox=dict(boxstyle="square,pad=0.20", facecolor=col,
                          edgecolor="none"))

    # the box under the cursor: live edges, eight handles, size read-out
    bx0, bx1, by0, by1 = 0.170, 0.672, 0.268, 0.782
    x0, x1 = px(bx0), px(bx1)
    y0, y1 = py(by1), py(by0)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=LIVE,
                           alpha=0.10, edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=LIVE, linewidth=1.3, zorder=6))
    for hx in (x0, (x0 + x1) / 2, x1):
        for hy in (y0, (y0 + y1) / 2, y1):
            if hx == (x0 + x1) / 2 and hy == (y0 + y1) / 2:
                continue
            ax.add_patch(Rectangle((hx - 0.036, hy - 0.036), 0.072, 0.072,
                                   facecolor="white", edgecolor=LIVE,
                                   linewidth=0.9, zorder=8))
    for xg in (x0, x1):
        ax.plot([xg, xg], [MAIN_Y0 + 0.04, TOOL_Y - 0.04], color=LIVE,
                lw=0.55, alpha=0.45, linestyle=(0, (4, 3)), zorder=4)
    for yg in (y0, y1):
        ax.plot([LEFT_X1 + 0.04, RIGHT_X0 - 0.04], [yg, yg], color=LIVE,
                lw=0.55, alpha=0.45, linestyle=(0, (4, 3)), zorder=4)
    ax.text(x0, y1 + 0.15, "pan", ha="left", va="center", fontsize=6.4,
            color="white", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="square,pad=0.22", facecolor=LIVE,
                      edgecolor="none"))
    ax.text(x1 + 0.20, y0 - 0.26, "412 x 419 px", ha="left", va="center",
            fontsize=6.2, color="white", family="monospace", zorder=9,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="#3a3a3add",
                      edgecolor="none"))
    ic_cursor(ax, x1 + 0.075, y0 - 0.035, 0.32, "white")
    ic_cursor(ax, x1 + 0.070, y0 - 0.030, 0.29, RULE)

    ax.text(LEFT_X1 + 0.10, TOOL_Y - 0.16, "clip 0271  ·  frame 118",
            ha="left", va="center", fontsize=6.4, color="#d5d5d5", zorder=6)
    ax.text(RIGHT_X0 - 0.10, TOOL_Y - 0.16, "100 %", ha="right", va="center",
            fontsize=6.4, color="#d5d5d5", zorder=6)


def property_panel(ax):
    x0, x1 = RIGHT_X0, WIN[0] + WIN[2]
    flat(ax, x0, MAIN_Y0, x1 - x0, TOOL_Y - MAIN_Y0, fc=SHELL, ec="none",
         r=0.0, zorder=2)
    lx, rx = x0 + 0.18, x1 - 0.18
    ax.text(lx, TOOL_Y - 0.20, "Object properties", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=RULE, zorder=5)

    y = TOOL_Y - 0.54
    ax.text(lx, y + 0.20, "Category", ha="left", va="center", fontsize=6.6,
            color=MUTED, zorder=5)
    raised(ax, lx, y - 0.13, rx - lx, 0.26, r=0.045)
    ax.text(lx + 0.10, y, "pan", ha="left", va="center", fontsize=6.9,
            color=INK, zorder=6)
    ax.text(rx - 0.30, y, "cookware", ha="right", va="center", fontsize=6.2,
            color=FAINT, zorder=6)
    caret(ax, rx - 0.14, y + 0.01, 0.26, RULE)

    y -= 0.46
    ax.text(lx, y + 0.20, "Instance id", ha="left", va="center", fontsize=6.6,
            color=MUTED, zorder=5)
    flat(ax, lx, y - 0.13, rx - lx, 0.26, fc="white", ec=UI, lw=0.9, r=0.045,
         zorder=3)
    ax.text(lx + 0.10, y, "obj_0271_03", ha="left", va="center", fontsize=6.7,
            family="monospace", color=INK, zorder=6)
    ax.plot([lx + 0.86, lx + 0.86], [y - 0.075, y + 0.075], color=UI, lw=0.9,
            zorder=6)

    for label, frac, val in [("Occlusion", 0.35, "0.35"),
                             ("Motion blur", 0.12, "0.12")]:
        y -= 0.44
        ax.text(lx, y + 0.17, label, ha="left", va="center", fontsize=6.6,
                color=MUTED, zorder=5)
        ax.text(rx, y + 0.17, val, ha="right", va="center", fontsize=6.4,
                family="monospace", color=SUB, zorder=5)
        ax.plot([lx, rx], [y - 0.02, y - 0.02], color=HAIR, lw=2.2,
                solid_capstyle="round", zorder=5)
        kx = lx + frac * (rx - lx)
        ax.plot([lx, kx], [y - 0.02, y - 0.02], color=UI, lw=2.2,
                solid_capstyle="round", zorder=6)
        ax.add_patch(Circle((kx, y - 0.02), 0.062, facecolor="white",
                            edgecolor=UI, linewidth=1.1, zorder=7))

    y -= 0.32
    hline(ax, lx, rx, y + 0.06)
    for label, checked in [("truncated by frame", False),
                           ("part of a crowd", False),
                           ("verified by reviewer", True)]:
        y -= 0.25
        if checked:
            flat(ax, lx, y - 0.075, 0.15, 0.15, fc=UI, ec=UI, lw=0.8,
                 r=0.028, zorder=4)
            tick(ax, lx + 0.075, y, 0.16, "white")
        else:
            flat(ax, lx, y - 0.075, 0.15, 0.15, fc="white", ec=LINE, lw=0.8,
                 r=0.028, zorder=4)
        ax.text(lx + 0.24, y, label, ha="left", va="center", fontsize=6.6,
                color=SUB, zorder=5)

    y -= 0.30
    ax.text(lx, y, "Quality", ha="left", va="center", fontsize=6.6,
            color=MUTED, zorder=5)
    for k, (lab, on) in enumerate([("good", True), ("fair", False),
                                   ("poor", False)]):
        cx = lx + 0.62 + k * 0.68
        ax.add_patch(Circle((cx, y), 0.072, facecolor="white",
                            edgecolor=UI if on else LINE, linewidth=0.9,
                            zorder=5))
        if on:
            ax.add_patch(Circle((cx, y), 0.034, facecolor=UI,
                                edgecolor="none", zorder=6))
        ax.text(cx + 0.13, y, lab, ha="left", va="center", fontsize=6.5,
                color=SUB, zorder=5)

    by = MAIN_Y0 + 0.16
    raised(ax, lx, by, 0.92, 0.30, r=0.05)
    ax.text(lx + 0.46, by + 0.15, "Skip", ha="center", va="center",
            fontsize=6.9, color=SUB, zorder=6)
    flat(ax, lx + 1.04, by, rx - lx - 1.04, 0.30, fc=UI, ec=UI, lw=0.8,
         r=0.05, zorder=4)
    ax.text((lx + 1.04 + rx) / 2, by + 0.15, "Save and next", ha="center",
            va="center", fontsize=6.9, fontweight="bold", color="white",
            zorder=6)


def status_bar(ax):
    y = WIN[1] + 0.19
    ax.text(WIN[0] + 0.18, y, "3 objects on this frame  ·  1 in progress",
            ha="left", va="center", fontsize=6.5, color=SUB, zorder=5)
    px0, px1 = 4.10, 6.60
    ax.plot([px0, px1], [y, y], color=HAIR, lw=3.0, solid_capstyle="round",
            zorder=5)
    ax.plot([px0, px0 + 0.39 * (px1 - px0)], [y, y], color=UI, lw=3.0,
            solid_capstyle="round", zorder=6)
    ax.text(px1 + 0.14, y, "118 / 300 frames", ha="left", va="center",
            fontsize=6.5, color=SUB, zorder=5)
    ax.text(WIN[0] + WIN[2] - 0.18, y,
            "autosaved 12 s ago  ·  mean 41 s per object", ha="right",
            va="center", fontsize=6.5, color=MUTED, zorder=5)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(WIN[0], 6.00, "Annotation interface used to build the corpus",
            ha="left", va="center", fontsize=11.6, fontweight="bold",
            color=INK)
    ax.text(WIN[0], 5.78, "box tool active: the operator drags one object "
                          "while the track, the attributes and the quality "
                          "flag stay editable on the right",
            ha="left", va="center", fontsize=7.8, color=MUTED)

    chrome(ax)
    layer_panel(ax)
    canvas_panel(ax)
    property_panel(ax)
    status_bar(ax)

    savefig(fig, "1130_wm_data_annotation_tool.png")


if __name__ == "__main__":
    main()
