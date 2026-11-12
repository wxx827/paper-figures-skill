"""Figure 343 - Image-text alignment inside a vision-language model.

Left: the input photograph cut into a patch grid, with the patches that carry
alignment mass tinted by concept group.  Right: the caption tokens as rounded
cards.  Bezier links between the two encode the alignment score in line width
and opacity.  The subordinate row underneath carries the full patch x token
score matrix and the per-token peak score.

This module also hosts the synthetic scene generator, the rounded text-card
helper and the shared colour vocabulary that figures 344-352 import, so the
whole VLM batch speaks one visual language.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Circle)
from scipy.ndimage import gaussian_filter

from pubstyle import (set_pub_style, savefig, get_cmap, add_colorbar,
                      lighten)

# ---------------------------------------------------------------------------
# Shared vocabulary for the whole VLM batch (343-352)
# ---------------------------------------------------------------------------
H, W = 168, 224                      # scene design units; boxes use these
SS = 2                               # supersampling of the rendered array

BG_QUESTION = "#f0f0f0"              # question / prompt cards
BG_ANSWER = "#dce8f5"                # model prediction cards
BG_GT = "#dcf0e2"                    # ground-truth cards
BG_ERROR = "#f7dcdc"                 # wrong prediction cards
BG_NOTE = "#f5f5f5"                  # commentary / grouping blocks

C_OK = "#2e7d5b"                     # correct
C_BAD = "#b23a48"                    # wrong
C_PRED = "#3b5b92"                   # prediction box
C_GT = "#1f6a47"                     # ground-truth box
C_INK = "#1a1a1a"
C_SUB = "#555555"
C_FAINT = "#8a8a8a"
C_EDGE = "#c8c8c8"

# concept groups used to colour tokens, links and boxes across the batch
C_PERSON = "#3b5b92"
C_OBJECT = "#b23a48"
C_ACTION = "#2e7d5b"
C_SCENE = "#c9a227"

MONO = {"fontfamily": "monospace"}


def _hex(c):
    return np.array(mpl.colors.to_rgb(c))


def _sm(rng, sigma):
    a = gaussian_filter(rng.standard_normal((H * SS, W * SS)), sigma * SS,
                        mode="wrap")
    return a / (a.std() + 1e-9)


def _blob(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _bar(xx, yy, x0, x1, y0, y1):
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def _person(xx, yy, cx, by, h):
    """Simplified standing figure; ``by`` is the ground line."""
    head = _blob(xx, yy, cx, by - 0.90 * h, 0.10 * h, 0.12 * h)
    torso = _blob(xx, yy, cx, by - 0.58 * h, 0.14 * h, 0.22 * h)
    arms = _blob(xx, yy, cx, by - 0.63 * h, 0.26 * h, 0.05 * h)
    legs = (((np.abs(xx - (cx - 0.08 * h)) < 0.05 * h) |
             (np.abs(xx - (cx + 0.08 * h)) < 0.05 * h)) &
            (yy > by - 0.40 * h) & (yy <= by))
    return head | torso | arms | legs


def _quadruped(xx, yy, cx, by, h, face=1):
    """Simplified four-legged animal silhouette (horse / dog / cat)."""
    body = _blob(xx, yy, cx, by - 0.58 * h, 0.50 * h, 0.20 * h)
    neck = _blob(xx, yy, cx + face * 0.40 * h, by - 0.78 * h, 0.12 * h,
                 0.20 * h)
    head = _blob(xx, yy, cx + face * 0.53 * h, by - 0.95 * h, 0.16 * h,
                 0.09 * h)
    tail = _blob(xx, yy, cx - face * 0.50 * h, by - 0.68 * h, 0.11 * h,
                 0.035 * h)
    legs = np.zeros_like(body)
    for d in (-0.34, -0.20, 0.20, 0.34):
        legs |= ((np.abs(xx - (cx + d * h)) < 0.045 * h) &
                 (yy > by - 0.60 * h) & (yy <= by))
    return body | neck | head | tail | legs


SCENE_KINDS = ("beach", "street", "park", "table", "room")


def make_scene(seed, kind="beach"):
    """Procedural photograph + named object boxes.

    Layered geometry (sky gradient, ground, rectangular buildings, round tree
    crowns, simplified vehicles / animals) plus multi-scale Gaussian texture,
    a sharp subject and a slightly defocused background.  Boxes are returned
    as ``(x0, y0, x1, y1)`` in pixel coordinates with y measured downward, the
    convention ``imshow`` uses.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(float) / SS
    fine, med, coarse = _sm(rng, 1.0), _sm(rng, 3.2), _sm(rng, 8.0)
    img = np.zeros((H * SS, W * SS, 3))
    boxes = {}
    subject = np.zeros((H * SS, W * SS), bool)

    def grad(y0, y1, ctop, cbot, mask=None):
        g = np.clip((yy - y0) / max(y1 - y0, 1.0), 0, 1)[..., None]
        v = _hex(ctop) * (1 - g) + _hex(cbot) * g
        m = np.ones_like(yy, bool) if mask is None else mask
        img[:] = np.where(m[..., None], v, img)

    def paint(mask, color, amp=0.10, tex=None, shade=1.0):
        t = med if tex is None else tex
        f = (np.clip(1.0 + amp * t, 0.25, 1.9) * shade)[..., None]
        img[:] = np.where(mask[..., None], np.clip(_hex(color) * f, 0, 1), img)

    def box(name, m, pad=1.5):
        ys, xs = np.where(m)
        if ys.size:
            boxes[name] = (max(xs.min() / SS - pad, 0),
                           max(ys.min() / SS - pad, 0),
                           min(xs.max() / SS + pad, W - 1),
                           min(ys.max() / SS + pad, H - 1))

    if kind == "beach":
        hz = 74
        grad(0, hz + 6, "#4a7cb0", "#cfe0ec")
        cl = np.clip(coarse * 0.45 + 0.05, 0, 1) * (yy < hz - 6)
        img[:] = np.clip(img + 0.17 * cl[..., None], 0, 1)
        sun = _blob(xx, yy, 188, 26, 10, 10)
        paint(sun, "#f7e6b4", amp=0.02)
        box("sun", sun, 3)
        sea = (yy >= hz) & (yy < hz + 28)
        paint(sea, "#2c6b80", amp=0.20, tex=fine)
        foam = sea & (((yy * 3 + 6 * np.sin(xx / 9.0)) % 11) < 1.8)
        paint(foam, "#d3e6ea", amp=0.05)
        box("sea", sea)
        sand = yy >= hz + 28
        grad(hz + 28, H, "#dcc59b", "#bda476", mask=sand)
        img[:] = np.clip(img * (1 + 0.07 * fine * sand)[..., None], 0, 1)
        boxes["sand"] = (0, hz + 28, W - 1, H - 1)
        hx, hy, hh = 124.0, 154.0, 72.0
        shad = _blob(xx, yy, hx, hy + 2, 0.60 * hh, 0.07 * hh)
        img[:] = np.where(shad[..., None], img * 0.72, img)
        horse = _quadruped(xx, yy, hx, hy, hh, face=1)
        paint(horse, "#5f4026", amp=0.12)
        rider = _person(xx, yy, hx + 5, hy - 0.60 * hh, 46.0)
        paint(rider, "#33506e", amp=0.09)
        box("horse", horse)
        box("man", rider)
        subject |= horse | rider

    elif kind == "street":
        hz = 70
        grad(0, hz + 8, "#5b86b8", "#d5e3ee")
        x = 0
        while x < W:
            bw = int(rng.integers(20, 42))
            top = hz - int(rng.integers(22, 58))
            m = _bar(xx, yy, x, x + bw, max(top, 0), hz + 2)
            paint(m, ["#9a8f84", "#8b8279", "#a89c8e"][int(rng.integers(0, 3))],
                  amp=0.06, shade=0.86 + 0.26 * rng.random())
            win = m & (xx % 8 < 4) & (yy % 10 < 5) & (yy < hz - 5)
            paint(win, "#e2d3a4", amp=0.10)
            x += bw + int(rng.integers(2, 6))
        t = np.clip((yy - hz) / (H - hz), 0, 1)
        left, right = 96 + (10 - 96) * t, 128 + (214 - 128) * t
        road = (yy >= hz) & (xx >= left) & (xx <= right)
        walk = (yy >= hz) & ~road
        paint(walk, "#a9a49c", amp=0.09)
        paint(road, "#4e5158", amp=0.07)
        lane = (road & (np.abs(xx - (left + right) / 2) < 1.2 + 1.8 * t) &
                (((yy * (0.7 + 1.5 * t)) % 20) < 10))
        paint(lane, "#ded6bd", amp=0.03)
        boxes["road"] = (10, hz, 214, H - 1)

        def car(cx, by, s, col, name):
            body = _bar(xx, yy, cx - 17 * s, cx + 17 * s, by - 11 * s,
                        by - 2 * s)
            roof = _bar(xx, yy, cx - 10 * s, cx + 9 * s, by - 19 * s,
                        by - 11 * s)
            wh = (_blob(xx, yy, cx - 11 * s, by - 2 * s, 4 * s, 3.4 * s) |
                  _blob(xx, yy, cx + 11 * s, by - 2 * s, 4 * s, 3.4 * s))
            glass = _bar(xx, yy, cx - 8 * s, cx + 7 * s, by - 17.5 * s,
                         by - 12 * s)
            paint(body | roof, col, amp=0.05)
            paint(glass, "#b9cdd8", amp=0.04)
            paint(wh, "#2f3136", amp=0.03)
            m = body | roof | wh
            box(name, m)
            return m

        subject |= car(148, 152, 1.30, "#b23a48", "red_car")
        subject |= car(72, 116, 0.70, "#3b5b92", "blue_car")
        pole = _bar(xx, yy, 214, 217, 74, 132)
        head = _bar(xx, yy, 209, 223, 50, 76)
        paint(pole | head, "#3a3a3a", amp=0.03)
        for k, c in enumerate(("#b23a48", "#c9a227", "#2e7d5b")):
            paint(_blob(xx, yy, 216, 56 + 8 * k, 3.4, 3.4), c, amp=0.02)
        box("traffic_light", head, 2)
        ped = _person(xx, yy, 20.0, 142.0, 44.0)
        paint(ped, "#4a3a52", amp=0.06)
        box("person", ped)
        subject |= ped

    elif kind == "park":
        hz = 58
        grad(0, hz + 6, "#6b95c4", "#dbe8f0")
        grass = yy >= hz
        grad(hz, H, "#6d9c54", "#48713a", mask=grass)
        img[:] = np.clip(img * (1 + 0.11 * fine * grass)[..., None], 0, 1)
        pt = np.clip((yy - hz) / (H - hz), 0, 1)
        path = (yy >= hz + 6) & (np.abs(xx - (146 + 46 * pt)) < 5 + 32 * pt)
        paint(path, "#c9b795", amp=0.09)
        for (tx, ty, r) in ((36, 104, 27), (88, 88, 18), (204, 96, 23)):
            paint(_bar(xx, yy, tx - 2.6, tx + 2.6, ty - 4, ty + 8), "#5a4326",
                  amp=0.06)
            crown = (((xx - tx) / r) ** 2 + ((yy - (ty - 8 - r)) / (0.9 * r))
                     ** 2) < 1 + 0.28 * med
            paint(crown, "#2f6b39", amp=0.30, tex=fine)
        bench = (_bar(xx, yy, 20, 82, 116, 124) |
                 _bar(xx, yy, 20, 82, 100, 112) |
                 _bar(xx, yy, 24, 28, 124, 140) |
                 _bar(xx, yy, 74, 78, 124, 140))
        paint(bench, "#8a6236", amp=0.07)
        box("bench", bench)
        dog = _quadruped(xx, yy, 122, 150, 26, face=-1)
        paint(dog, "#9a7448", amp=0.10)
        box("dog", dog)
        per = _person(xx, yy, 182.0, 156.0, 54.0)
        paint(per, "#3b5b92", amp=0.07)
        box("person", per)
        fris = _blob(xx, yy, 150, 86, 8, 3.2)
        paint(fris, "#c9a227", amp=0.02)
        box("frisbee", fris, 2.5)
        subject |= dog | per | fris

    elif kind == "table":
        grad(0, 84, "#d2c6b5", "#bcac97")
        paint(_bar(xx, yy, 0, W, 64, 74), "#a89a86", amp=0.05)
        top = yy >= 78
        grad(78, H, "#9c7442", "#7b5a33", mask=top)
        img[:] = np.clip(img * (1 + 0.10 * fine * top)[..., None], 0, 1)
        paint(top & ((yy % 15) < 1.1), "#6a4c2c", amp=0.04)
        shad = _blob(xx, yy, 78, 140, 48, 15) | _blob(xx, yy, 170, 136, 34, 12)
        img[:] = np.where(shad[..., None], img * 0.82, img)
        plate = _blob(xx, yy, 76, 132, 44, 22)
        paint(plate, "#e6e2da", amp=0.03)
        paint(_blob(xx, yy, 76, 132, 37, 17), "#f2efe9", amp=0.02)
        box("plate", plate)
        apples = np.zeros_like(yy, bool)
        for i, (ax_, ay_) in enumerate(((50, 124), (76, 130), (102, 124),
                                        (64, 141))):
            a = _blob(xx, yy, ax_, ay_, 11, 10)
            paint(a, "#b23a48", amp=0.08)
            paint(_blob(xx, yy, ax_ - 3.4, ay_ - 4, 3.4, 2.4), "#dd9a94",
                  amp=0.02)
            box(f"apple{i + 1}", a)
            apples |= a
        box("apples", apples)
        bowl = _blob(xx, yy, 170, 128, 32, 19) & (yy >= 122)
        paint(bowl, "#3d6b8e", amp=0.07)
        paint(_blob(xx, yy, 170, 124, 30, 8), "#7ba2bd", amp=0.03)
        box("bowl", bowl)
        cup = (_bar(xx, yy, 184, 204, 94, 124) |
               _blob(xx, yy, 208, 106, 7, 8))
        paint(cup, "#e8e4dc", amp=0.04)
        box("cup", cup)
        subject |= apples | bowl | plate | cup

    else:  # "room"
        grad(0, 100, "#d9d2c6", "#c5bcad")
        paint(_bar(xx, yy, 0, W, 96, 104), "#b1a693", amp=0.05)
        floor = yy >= 104
        grad(104, H, "#a8875f", "#7c6042", mask=floor)
        img[:] = np.clip(img * (1 + 0.09 * fine * floor)[..., None], 0, 1)
        paint(_blob(xx, yy, 92, 152, 80, 20), "#8f6a70", amp=0.06)
        wnd = _bar(xx, yy, 16, 72, 18, 68)
        paint(wnd, "#a9c8dd", amp=0.06)
        paint(wnd & ((np.abs(xx - 44) < 1.8) | (np.abs(yy - 43) < 1.8)),
              "#f2efe8", amp=0.02)
        paint(_bar(xx, yy, 12, 76, 68, 73), "#e6e0d4", amp=0.03)
        box("window", wnd)
        seat = _bar(xx, yy, 92, 196, 94, 126)
        back = _bar(xx, yy, 92, 196, 64, 94)
        arms = ((_blob(xx, yy, 100, 96, 10, 32) & (yy < 128)) |
                (_blob(xx, yy, 188, 96, 10, 32) & (yy < 128)))
        sofa = seat | back | arms
        img[:] = np.where((_blob(xx, yy, 144, 130, 58, 8))[..., None],
                          img * 0.80, img)
        paint(sofa, "#4e6070", amp=0.07)
        paint(back, "#61768a", amp=0.06)
        for cx0 in (110, 148):
            paint(_bar(xx, yy, cx0, cx0 + 32, 68, 92), "#72879b", amp=0.05)
        paint(seat & (yy < 98), "#3f5162", amp=0.03)
        paint(arms, "#5a6d7e", amp=0.05)
        paint(_bar(xx, yy, 96, 102, 126, 136) |
              _bar(xx, yy, 186, 192, 126, 136), "#4a3a2c", amp=0.03)
        box("sofa", sofa)
        cat = _quadruped(xx, yy, 152, 118, 26, face=-1)
        paint(cat, "#c07a3a", amp=0.10)
        box("cat", cat)
        shade_ = _bar(xx, yy, 198, 222, 28, 50)
        lamp = _bar(xx, yy, 208, 212, 50, 126) | shade_
        paint(lamp, "#8a8175", amp=0.05)
        paint(shade_, "#e0c98d", amp=0.04)
        box("lamp", lamp)
        tbl = (_bar(xx, yy, 16, 78, 118, 126) |
               _bar(xx, yy, 22, 28, 126, 150) |
               _bar(xx, yy, 66, 72, 126, 150))
        paint(tbl, "#7a5a38", amp=0.07)
        box("table", tbl)
        subject |= cat | sofa

    # depth of field: lower in the frame reads as nearer, and the subject is
    # always in focus
    focus = np.clip((yy / H - 0.34) / 0.26, 0, 1)
    if subject.any():
        focus = np.maximum(focus, np.clip(
            gaussian_filter(subject.astype(float), 2.2 * SS) * 2.8, 0, 1))
    blur = np.dstack([gaussian_filter(img[..., k], 1.5 * SS) for k in range(3)])
    img = img * focus[..., None] + blur * (1 - focus[..., None])
    vign = 1 - 0.15 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5)
    img = img * vign[..., None]
    img = np.clip(img + 0.013 * rng.standard_normal(img.shape), 0, 1)
    img = np.dstack([gaussian_filter(img[..., k], 0.32 * SS)
                     for k in range(3)])
    return np.clip(img, 0, 1), boxes


def attention_map(boxes, weights=None, spread=1.0, floor=0.05, seed=0):
    """Smooth attention field peaked on the given ``(x0, y0, x1, y1)`` boxes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    nz = gaussian_filter(rng.standard_normal((H, W)), 9.0, mode="wrap")
    a = floor * (1.0 + 0.6 * nz / (nz.std() + 1e-9))
    if weights is None:
        weights = [1.0] * len(boxes)
    for b, w in zip(boxes, weights):
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        rx = max((b[2] - b[0]) / 2.0, 7.0) * spread
        ry = max((b[3] - b[1]) / 2.0, 7.0) * spread
        a = a + w * np.exp(-0.5 * (((xx - cx) / (0.85 * rx + 5)) ** 2 +
                                   ((yy - cy) / (0.85 * ry + 5)) ** 2))
    a = gaussian_filter(a, 3.0)
    return np.clip(a / a.max(), 0, 1)


# ---------------------------------------------------------------------------
# Drawing helpers shared by the batch
# ---------------------------------------------------------------------------
def inch_canvas(fig, rect):
    """Axes whose data coordinates are inches, running top-down.

    Everything drawn in it keeps its physical proportions, which is what makes
    the diagram-style panels of this batch line up across figures.
    """
    ax = fig.add_axes(rect)
    w_in, h_in = rect[2] * fig.get_figwidth(), rect[3] * fig.get_figheight()
    ax.set_xlim(0, w_in)
    ax.set_ylim(h_in, 0)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return ax, w_in, h_in


def draw_photo(ax, img, x, y, w, h=None, ec="#b8b8b8", lw=0.6, zorder=2,
               alpha=1.0):
    """Place a synthetic photograph with its hairline frame; returns the box."""
    if h is None:
        h = w * img.shape[0] / float(img.shape[1])
    ax.imshow(img, extent=(x, x + w, y + h, y), origin="upper", zorder=zorder,
              interpolation="bilinear", alpha=alpha)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=lw, zorder=zorder + 4))
    return x, y, w, h


def box_axes(b, x, y, w, h):
    """Map a scene box (design pixels) into the inch coords of a placed photo."""
    sx, sy = w / float(W), h / float(H)
    return (x + b[0] * sx, y + b[1] * sy, (b[2] - b[0]) * sx,
            (b[3] - b[1]) * sy)


def region_box(ax, b, x, y, w, h, color=C_PRED, dashed=False, lw=1.3,
               zorder=9, halo=True, label=None, fs=6.2, above=True):
    """Draw a detection-style box over a placed photo, with a white halo so it
    survives any background."""
    bx, by, bw, bh = box_axes(b, x, y, w, h)
    r = Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=color,
                  linewidth=lw, zorder=zorder,
                  linestyle=(0, (2.6, 1.6)) if dashed else "-")
    if halo:
        r.set_path_effects([pe.withStroke(linewidth=lw + 1.5,
                                          foreground="white", alpha=0.55)])
    ax.add_patch(r)
    if label:
        ly = by - 0.035 if above else by + bh + 0.035
        ax.text(bx + 0.02, ly, label, fontsize=fs, color="white",
                ha="left", va="bottom" if above else "top", zorder=zorder + 1,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=color,
                          edgecolor="none"))
    return bx, by, bw, bh


def text_card(ax, x, y, w, h, text, fc=BG_QUESTION, ec=C_EDGE, fs=7.0,
              color=C_INK, weight="normal", mono=False, ha="left", va="center",
              pad=0.07, lw=0.6, zorder=3, rounding=0.05, alpha=1.0,
              linespacing=1.25):
    """Rounded card with a label - the house text element of this batch."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha,
        mutation_aspect=1.0))
    tx = {"left": x + pad, "center": x + w / 2.0, "right": x + w - pad}[ha]
    kw = dict(MONO) if mono else {}
    ax.text(tx, y + h / 2.0, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, zorder=zorder + 2, linespacing=linespacing,
            **kw)


def tick(ax, cx, cy, s=0.075, color=C_OK, lw=1.5, zorder=8):
    """Geometric check mark (no emoji anywhere in this batch)."""
    ax.plot([cx - s, cx - 0.15 * s, cx + s],
            [cy + 0.1 * s, cy + 0.85 * s, cy - 0.85 * s],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder, clip_on=False)


def cross(ax, cx, cy, s=0.065, color=C_BAD, lw=1.5, zorder=8):
    """Geometric cross mark."""
    ax.plot([cx - s, cx + s], [cy - s, cy + s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder, clip_on=False)
    ax.plot([cx - s, cx + s], [cy + s, cy - s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder, clip_on=False)


def panel_tag(fig, ax, label, dx=0.020, dy=0.010, size=11):
    """Panel label placed at a fixed *figure* offset, identical for all axes."""
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, label, fontsize=size, fontweight="bold",
             ha="left", va="bottom")


def bg_block(fig, x0, y0, x1, y1, color=BG_NOTE):
    """Light grey grouping block behind subordinate panels."""
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0,
                             transform=fig.transFigure, facecolor=color,
                             edgecolor="none", zorder=-1))


# ---------------------------------------------------------------------------
# Figure 343
# ---------------------------------------------------------------------------
TOKENS = ["a", "man", "riding", "a", "brown", "horse", "on", "the", "beach"]
TOK_COLOR = {1: C_PERSON, 2: C_ACTION, 4: C_OBJECT, 5: C_OBJECT, 8: C_SCENE}
# token index -> [(patch row, patch col, alignment score)]
LINKS = {
    0: [(2, 6, 0.14)],
    1: [(2, 4, 0.88), (3, 4, 0.57)],
    2: [(3, 4, 0.49), (4, 4, 0.36)],
    3: [(3, 6, 0.15)],
    4: [(4, 4, 0.54), (4, 5, 0.43)],
    5: [(4, 4, 0.91), (4, 5, 0.69), (3, 5, 0.45)],
    6: [(5, 5, 0.18)],
    7: [(2, 5, 0.13)],
    8: [(5, 6, 0.82), (4, 6, 0.64), (5, 5, 0.44)],
}
PROWS, PCOLS = 6, 8


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)
    photo, _boxes = make_scene(11, "beach")

    figsize = (9.8, 6.1)
    fig = plt.figure(figsize=figsize)
    bg_block(fig, 0.020, 0.045, 0.980, 0.312)

    ax, cw, ch = inch_canvas(fig, [0.045, 0.335, 0.935, 0.560])

    # ---- input image with its patch grid ---------------------------------
    ix, iy, iw = 0.14, 0.44, 3.45
    ih = iw * H / W
    draw_photo(ax, photo, ix, iy, iw)
    pw, ph = iw / PCOLS, ih / PROWS

    strongest = {}
    for ti, links in LINKS.items():
        for (r, c, s) in links:
            if s > strongest.get((r, c), (0.0, None))[0]:
                strongest[(r, c)] = (s, ti)

    for r in range(PROWS):
        for c in range(PCOLS):
            ax.add_patch(Rectangle((ix + c * pw, iy + r * ph), pw, ph,
                                   facecolor="none", edgecolor="white",
                                   linewidth=0.35, alpha=0.45, zorder=4))
    for (r, c), (s, ti) in strongest.items():
        col = TOK_COLOR.get(ti, C_FAINT)
        ax.add_patch(Rectangle((ix + c * pw, iy + r * ph), pw, ph,
                               facecolor=col, alpha=0.10 + 0.20 * s,
                               edgecolor=col, linewidth=1.0, zorder=5))
        ax.text(ix + (c + 0.5) * pw, iy + (r + 0.5) * ph, f"p{r}{c}",
                ha="center", va="center", fontsize=6.2, color="#1a1a1a",
                zorder=6, **MONO,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white",
                                            alpha=0.85)])

    ax.text(ix, iy - 0.13, "Image patches  (6 x 8 grid)", fontsize=8.0,
            fontweight="bold", ha="left", va="bottom", color=C_INK)
    ax.text(ix + iw, iy - 0.13, "ViT-B/16 encoder", fontsize=6.6,
            ha="right", va="bottom", color=C_FAINT)

    # ---- caption tokens ---------------------------------------------------
    tx, tw, th, gap = 6.38, 2.55, 0.26, 0.040
    ty0 = 0.44
    ax.text(tx, ty0 - 0.13, "Caption tokens", fontsize=8.0, fontweight="bold",
            ha="left", va="bottom", color=C_INK)
    ax.text(tx + tw, ty0 - 0.13, "peak score", fontsize=6.6, ha="right",
            va="bottom", color=C_FAINT)

    tok_y = {}
    for i, tokn in enumerate(TOKENS):
        y = ty0 + i * (th + gap)
        tok_y[i] = y + th / 2.0
        col = TOK_COLOR.get(i)
        peak = max(s for _r, _c, s in LINKS[i])
        text_card(ax, tx, y, tw, th, f"{tokn}",
                  fc=lighten(col, 0.86) if col else "#f4f4f4",
                  ec=col if col else "#d8d8d8", fs=8.0, mono=True,
                  color=col if col else C_SUB,
                  weight="bold" if col else "normal", pad=0.14, lw=0.8)
        ax.text(tx + tw - 0.12, y + th / 2.0, f"{peak:.2f}", ha="right",
                va="center", fontsize=6.4, color=C_SUB if col else C_FAINT,
                zorder=6, **MONO)

    # ---- alignment links --------------------------------------------------
    for i, links in LINKS.items():
        col = TOK_COLOR.get(i, "#9a9a9a")
        for (r, c, s) in sorted(links, key=lambda v: v[2]):
            x0 = ix + (c + 0.5) * pw
            y0 = iy + (r + 0.5) * ph
            arc = 0.16 if tok_y[i] > y0 else -0.16
            fa = FancyArrowPatch((x0, y0), (tx - 0.045, tok_y[i]),
                                 connectionstyle=f"arc3,rad={arc}",
                                 arrowstyle="-", mutation_scale=1.0,
                                 linewidth=0.35 + 2.4 * s,
                                 color=col, alpha=0.18 + 0.62 * s, zorder=7,
                                 capstyle="round")
            fa.set_path_effects([pe.withStroke(linewidth=0.35 + 2.4 * s + 1.1,
                                               foreground="white", alpha=0.55)])
            ax.add_patch(fa)
            ax.add_patch(Circle((x0, y0), 0.032, facecolor=col,
                                edgecolor="white", linewidth=0.5, zorder=8))

    # ---- legend strip spanning the full canvas width ----------------------
    ly = ty0 + len(TOKENS) * (th + gap) + 0.10
    for k, (name, col) in enumerate((("person", C_PERSON), ("action", C_ACTION),
                                     ("object", C_OBJECT), ("scene", C_SCENE))):
        x = ix + 0.94 * k
        ax.add_patch(Rectangle((x, ly), 0.20, 0.11, facecolor=col,
                               edgecolor="none", zorder=4))
        ax.text(x + 0.26, ly + 0.055, name, fontsize=7.0, va="center",
                ha="left", color=C_SUB)
    ax.text(tx + tw, ly + 0.055,
            "link width and opacity  $\\propto$  alignment score",
            fontsize=7.0, va="center", ha="right", color=C_SUB)

    # ---- subordinate panel: full score matrix ----------------------------
    patch_ids = sorted(strongest.keys())
    M = np.zeros((len(patch_ids), len(TOKENS)))
    for j, links in LINKS.items():
        for (r, c, s) in links:
            M[patch_ids.index((r, c)), j] = s
    M += 0.10 * np.abs(rng.standard_normal(M.shape)) * (M == 0)
    M = np.clip(M, 0, 1)

    axm = fig.add_axes([0.075, 0.088, 0.415, 0.180])
    axm.set_facecolor("none")
    im = axm.imshow(M, cmap=get_cmap("cb_sequential_blue"), vmin=0.0, vmax=1.0,
                    aspect="auto", interpolation="nearest")
    axm.set_xticks(range(len(TOKENS)))
    axm.set_xticklabels(TOKENS, fontsize=6.4, rotation=30, ha="right", **MONO)
    axm.set_yticks(range(len(patch_ids)))
    axm.set_yticklabels([f"p{r}{c}" for (r, c) in patch_ids], fontsize=6.0,
                        **MONO)
    axm.tick_params(length=1.8, width=0.6)
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#b8b8b8")
    axm.set_title("Patch x token alignment matrix", fontsize=8.5,
                  fontweight="bold", pad=3.5)
    # axm is only ~1.1 in tall, so the house aspect=22 would shrink the bar to
    # a 0.05 in hairline; scale the aspect down with the panel height.
    cb = add_colorbar(fig, im, axm, label="alignment score (-)",
                      ticks=[0, 0.5, 1.0], aspect=9)

    # ---- subordinate panel: per-token peak -------------------------------
    axb = fig.add_axes([0.655, 0.088, 0.235, 0.180])
    axb.set_facecolor("none")
    peaks = [max(s for _r, _c, s in LINKS[i]) for i in range(len(TOKENS))]
    ypos = np.arange(len(TOKENS))[::-1]
    cols = [TOK_COLOR.get(i, "#c4c4c4") for i in range(len(TOKENS))]
    axb.barh(ypos, peaks, height=0.66, color=cols, edgecolor="white",
             linewidth=0.4)
    for y, v in zip(ypos, peaks):
        axb.text(v + 0.02, y, f"{v:.2f}", va="center", ha="left", fontsize=6.0,
                 color=C_SUB, **MONO)
    axb.set_yticks(ypos)
    axb.set_yticklabels(TOKENS, fontsize=6.4, **MONO)
    axb.set_xlim(0, 1.16)
    axb.set_xticks([0, 0.5, 1.0])
    axb.tick_params(labelsize=6.4, length=2.0, width=0.6)
    axb.set_xlabel("peak alignment score (-)", fontsize=7.0)
    axb.axvline(0.30, color="#808080", lw=0.7, ls=(0, (3, 2)), zorder=0)
    axb.text(0.33, len(TOKENS) - 1.1, "grounding threshold", fontsize=6.0,
             color=C_FAINT, ha="left", va="center")
    axb.set_title("Peak score per token", fontsize=8.5, fontweight="bold",
                  pad=3.5)
    axb.spines["left"].set_linewidth(0.6)
    axb.spines["bottom"].set_linewidth(0.6)

    for a, lab in ((ax, "(a)"), (axm, "(b)"), (axb, "(c)")):
        panel_tag(fig, a, lab)
    savefig(fig, "343_vlm_image_text_align.png")


if __name__ == "__main__":
    main()
