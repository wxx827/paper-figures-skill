"""Dataset representation: image regions, region descriptions and scene graph.

Three panels read left to right.  A procedurally synthesised counter scene
carries three numbered region boxes; each box is paired by colour with a
description card in the middle column and joined to it by a thin leader line
(a line, never an arrow -- the arrows are reserved for the graph).  The right
column folds those sentences into a scene graph whose nodes are typed by
colour (object / attribute / relationship) and whose edges are right-angled
polylines with small solid heads.

The scene is built from hard-edged signed-distance geometry rather than from
blurred blobs, which is what lets a reader name the objects at thumbnail size:
a horizon for the counter edge, a mullioned window on the wall, a capsule
bottle, a ceramic bowl (elliptical mouth over a half-ellipse body, with fruit
resting inside and the near lip redrawn over it), a mug (cylinder plus a
C-shaped handle), and a cutting board with slices and a knife filling the near
right corner.  Every silhouette gets a dark contour; the grain is applied
mostly to the room, and only the room is blurred, so the objects stay crisp
against a soft background.

Everything is laid out in inches on an equal-aspect canvas, so corner radii,
gaps and line weights are identical in every panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"

OBJ = "#b23a48"        # house red    - objects
ATTR = "#3b5b92"       # house blue   - attributes
REL = "#2e7d5b"        # house green  - relationships
REGION = ["#c9a227", "#8e6bbf", "#2a7f7a"]     # gold / purple / teal

SEED = 1534  # original id; figure number is 1129
AR = 1.417             # scene width / height

# ---- canvas geometry (inches) ---------------------------------------------
W, H = 11.0, 5.2
IMG_X0, IMG_Y1 = 0.40, 4.22
IMG_W = 4.55
SCALE = IMG_W / AR                     # inches per scene unit
IMG_H = SCALE
CARD_X0, CARD_W = 5.22, 2.14
CHAIN_X, ATTR_X = 8.50, 9.95
CHAIN_W, ATTR_W = 1.35, 1.05
NODE_H = 0.34
ROWS = [4.05, 3.46, 2.87, 2.28, 1.69, 1.10]

# region boxes in scene units: (x0, x1, y0, y1) with y measured downward
BOXES = [(0.68, 1.28, 0.06, 0.54),     # window
         (0.34, 0.50, 0.235, 0.655),   # bottle
         (0.02, 1.395, 0.575, 0.975)]  # counter
CARD_Y = [3.65, 2.65, 1.65]
CARDS = [("tall window above the", "kitchen counter"),
         ("green glass bottle", "standing on the counter"),
         ("wooden counter with", "a ceramic fruit bowl")]

# image-level annotation totals; the three cards above are the sample of it
# that this figure shows, so both strings are built from these constants
N_REGIONS, N_OBJ, N_REL, N_ATTR = 41, 18, 22, 9
WORDS = [len(l1.split()) + len(l2.split()) for l1, l2 in CARDS]


# ---- procedural scene ------------------------------------------------------
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


def _box(xx, yy, cx, cy, hw, hh, r):
    qx = np.abs(xx - cx) - (hw - r)
    qy = np.abs(yy - cy) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def _ellipse(xx, yy, cx, cy, a, b):
    """Approximate signed distance to an axis-aligned ellipse."""
    return (np.hypot((xx - cx) / a, (yy - cy) / b) - 1.0) * min(a, b)


def _cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


CERAMIC = (0.875, 0.860, 0.830)
CERAMIC_IN = (0.650, 0.635, 0.610)
MUG = (0.450, 0.505, 0.520)
MUG_IN = (0.280, 0.325, 0.340)
BOARD = (0.745, 0.615, 0.420)
BOARD_EDGE = (0.540, 0.420, 0.270)
FRUITS = [(0.842, 0.628, 0.050, (0.805, 0.470, 0.185)),
          (0.926, 0.617, 0.043, (0.625, 0.245, 0.225)),
          (0.978, 0.634, 0.037, (0.430, 0.545, 0.290))]


def counter_scene(seed=SEED, ny=260):
    """Kitchen counter: wall + window behind, bottle and bowl in front."""
    nx = int(round(ny * AR))
    rng = np.random.default_rng(seed)
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * AR
    soft = 1.8 / ny
    hz = 0.615

    wall = np.array([0.735, 0.715, 0.680])
    top = np.array([0.560, 0.415, 0.290])
    bg = np.where((yy < hz)[..., None],
                  wall * (0.90 + 0.26 * (yy / hz))[..., None],
                  top * (1.14 - 0.30 * np.clip((yy - hz) / (1 - hz), 0, 1))
                  [..., None])
    bg *= 1.0 - 0.30 * _cover(np.abs(yy - hz) - 0.004, soft)[..., None]

    # window: sky gradient inside a pale frame, plus two mullions
    frame = _box(xx, yy, 0.98, 0.295, 0.300, 0.225, 0.016)
    glass = _box(xx, yy, 0.98, 0.295, 0.272, 0.197, 0.010)
    sky_hi = np.array([0.760, 0.815, 0.860])
    sky_lo = np.array([0.560, 0.650, 0.735])
    t = np.clip((yy - 0.10) / 0.40, 0, 1)[..., None]
    sky = sky_hi * (1 - t) + sky_lo * t
    sky = sky * (1.0 - 0.16 * _cover(
        np.abs(_capsule(xx, yy, (0.62, 0.44), (1.34, 0.40), 0.0)) - 0.030,
        soft)[..., None])
    m_frame, m_glass = _cover(frame, soft), _cover(glass, soft)
    bg = bg * (1 - m_frame[..., None]) + np.array([0.845, 0.830, 0.795]) \
        * m_frame[..., None]
    bg = bg * (1 - m_glass[..., None]) + sky * m_glass[..., None]
    mull = np.minimum(_box(xx, yy, 0.98, 0.295, 0.010, 0.197, 0.004),
                      _box(xx, yy, 0.98, 0.295, 0.272, 0.008, 0.004))
    m_mull = _cover(mull, soft) * m_glass
    bg = bg * (1 - m_mull[..., None]) + np.array([0.830, 0.815, 0.785]) \
        * m_mull[..., None]

    # wood grain: low-frequency streaks that follow the counter top
    grain_lines = 0.5 + 0.5 * np.sin(38.0 * yy + 3.0 * np.sin(4.0 * xx))
    wood = (yy > hz) * grain_lines
    bg *= (1.0 - 0.075 * wood)[..., None]

    # contact shadows on the counter top, one per standing object
    sh = np.exp(-(((xx - 0.44) / 0.115) ** 2 + ((yy - 0.640) / 0.022) ** 2))
    sh += np.exp(-(((xx - 0.90) / 0.195) ** 2 + ((yy - 0.768) / 0.026) ** 2))
    sh += np.exp(-(((xx - 0.21) / 0.100) ** 2 + ((yy - 0.798) / 0.020) ** 2))
    sh += np.exp(-(((xx - 1.16) / 0.235) ** 2 + ((yy - 0.912) / 0.022) ** 2))
    bg *= (1.0 - 0.32 * np.clip(sh, 0, 1))[..., None]

    # ---- foreground objects: hard-edged geometry, painted back to front ----
    shade = (1.24 - 0.36 * yy)[..., None]
    st = {"img": bg.copy(), "mask": np.zeros_like(yy),
          "edge": np.zeros_like(yy)}

    def paint(sdf, col, tone=1.0, outline=True):
        m = _cover(sdf, soft)
        st["img"] = st["img"] * (1 - m[..., None]) \
            + np.asarray(col) * tone * shade * m[..., None]
        st["mask"] = np.clip(st["mask"] + m, 0, 1)
        if outline:
            st["edge"] = np.maximum(st["edge"],
                                    _cover(np.abs(sdf) - 0.0040, soft))

    def glow(sdf, amount):
        st["img"] = np.clip(
            st["img"] + amount * _cover(sdf, soft)[..., None], 0, 1)

    # mug: cylinder (top ellipse + wall + base ellipse) with a C-shaped handle
    mx, mtop, mbot, mr = 0.212, 0.652, 0.800, 0.070
    m_wall = _box(xx, yy, mx, (mtop + mbot) / 2, mr, (mbot - mtop) / 2, 0.008)
    m_lip = _ellipse(xx, yy, mx, mtop, mr, 0.024)
    m_base = _ellipse(xx, yy, mx, mbot, mr, 0.021)
    ring = np.abs(np.hypot(xx - (mx + mr - 0.010), yy - 0.722) - 0.060) - 0.017
    m_handle = np.maximum(ring, (mx + mr - 0.008) - xx)
    paint(np.minimum.reduce([m_wall, m_lip, m_base, m_handle]), MUG)
    paint(_ellipse(xx, yy, mx, mtop, mr * 0.78, 0.017), MUG_IN, outline=False)
    glow(np.maximum(_box(xx, yy, mx - 0.034, 0.726, 0.009, 0.050, 0.009),
                    m_wall), 0.20)

    # bowl: elliptical mouth over a half-ellipse body, fruit resting inside
    bx, by, ra, rb = 0.900, 0.652, 0.158, 0.042
    b_body = np.maximum(_ellipse(xx, yy, bx, by, ra, 0.118), by - yy)
    b_rim = _ellipse(xx, yy, bx, by, ra, rb)
    b_in = _ellipse(xx, yy, bx, by, ra - 0.017, rb - 0.011)
    paint(np.minimum(b_body, b_rim), CERAMIC)
    paint(b_in, CERAMIC_IN, outline=False)
    paint(np.maximum(b_in, (by - 0.004) - yy), CERAMIC_IN, tone=0.82,
          outline=False)
    for fx, fy, fr, col in FRUITS:
        paint(np.hypot(xx - fx, yy - fy) - fr, col)
    # the near lip is redrawn over the fruit so they sit *in* the bowl
    paint(np.maximum.reduce([b_rim, -b_in, (by + 0.002) - yy]), CERAMIC,
          outline=False)
    for fx, fy, fr, _c in FRUITS:
        glow(np.hypot(xx - (fx - 0.34 * fr), yy - (fy - 0.40 * fr))
             - 0.26 * fr, 0.26)

    # cutting board with two slices and a knife, filling the near right corner
    paint(_box(xx, yy, 1.160, 0.850, 0.205, 0.046, 0.024), BOARD)
    paint(_box(xx, yy, 1.160, 0.902, 0.205, 0.014, 0.008), BOARD_EDGE,
          outline=False)
    paint(np.hypot(xx - 1.072, yy - 0.844) - 0.026, (0.545, 0.665, 0.350))
    paint(np.hypot(xx - 1.128, yy - 0.858) - 0.022, (0.760, 0.395, 0.325))
    paint(_capsule(xx, yy, (1.128, 0.826), (1.258, 0.818), 0.015),
          (0.800, 0.810, 0.820))
    paint(_capsule(xx, yy, (1.262, 0.818), (1.334, 0.814), 0.016),
          (0.255, 0.235, 0.220))

    # bottle: body capsule + neck + cap, with a specular stripe
    bottle = np.minimum(_capsule(xx, yy, (0.42, 0.585), (0.42, 0.395), 0.062),
                        _capsule(xx, yy, (0.42, 0.400), (0.42, 0.288), 0.021))
    paint(bottle, (0.205, 0.385, 0.280))
    paint(_box(xx, yy, 0.42, 0.268, 0.026, 0.020, 0.007),
          (0.375, 0.345, 0.300))
    glow(np.maximum(_box(xx, yy, 0.392, 0.470, 0.011, 0.150, 0.010), bottle),
         0.26)

    # one dark contour around every silhouette: this is what makes the
    # objects readable at thumbnail size
    st["img"] *= (1.0 - 0.38 * st["edge"])[..., None]

    noise = (1.0 + 0.075 * (_value_noise(rng, ny, nx) - 0.5) * 2.0) \
        * (1.0 + 0.020 * rng.normal(0.0, 1.0, (ny, nx)))
    # objects keep most of their flat colour; the room carries the grain
    st["img"] *= (1.0 + (noise - 1.0) * (1.0 - 0.55 * st["mask"]))[..., None]
    bg *= noise[..., None]

    sharp = _blur(st["mask"], 0.55)[..., None]
    out = _blur3(st["img"], 0.12) * sharp + _blur3(bg, 1.95) * (1.0 - sharp)
    r2 = ((xx - AR / 2) / AR) ** 2 + (yy - 0.5) ** 2
    out *= (1.0 - 0.30 * r2)[..., None]
    return np.clip(out, 0.0, 1.0)


# ---- drawing helpers -------------------------------------------------------
def sx(u):
    return IMG_X0 + u * SCALE


def sy(v):
    return IMG_Y1 - v * SCALE


def node(ax, x, y, w, text, color, fontsize=7.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - NODE_H / 2), w, NODE_H,
                                boxstyle="round,pad=0,rounding_size=0.075",
                                facecolor=lighten(color, 0.82),
                                edgecolor=color, linewidth=0.9, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=5)


def head(ax, x, y, direction, color=RULE, size=0.052):
    """Small solid triangle terminating an orthogonal edge."""
    if direction == "down":
        pts = [(x, y), (x - size, y + 1.8 * size), (x + size, y + 1.8 * size)]
    else:
        pts = [(x, y), (x - 1.8 * size, y - size), (x - 1.8 * size, y + size)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                         zorder=6))


def edge(ax, pts, direction, color=RULE):
    px, py = zip(*pts)
    ax.plot(px, py, color=color, lw=0.85, solid_joinstyle="miter",
            solid_capstyle="butt", zorder=3)
    head(ax, pts[-1][0], pts[-1][1], direction, color=color)


def column_header(ax, x, w, text):
    ax.text(x, 4.47, text, ha="left", va="bottom", fontsize=8.4,
            fontweight="bold", color=RULE)
    ax.plot([x, x + w], [4.41, 4.41], color=HAIR, lw=0.8, zorder=1)


# ---- panels ----------------------------------------------------------------
def draw_image(ax):
    ax.imshow(counter_scene(), extent=(IMG_X0, IMG_X0 + IMG_W,
                                       IMG_Y1 - IMG_H, IMG_Y1),
              origin="upper", interpolation="bilinear", zorder=2)
    ax.add_patch(Rectangle((IMG_X0, IMG_Y1 - IMG_H), IMG_W, IMG_H,
                           facecolor="none", edgecolor=RULE, linewidth=0.8,
                           zorder=6))
    for i, (x0, x1, y0, y1) in enumerate(BOXES):
        col = REGION[i]
        bx, by = sx(x0), sy(y1)
        bw, bh = (x1 - x0) * SCALE, (y1 - y0) * SCALE
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="none",
                               edgecolor=col, linewidth=1.3, zorder=7))
        ax.add_patch(FancyBboxPatch((bx + 0.035, by + bh - 0.215), 0.20, 0.18,
                                    boxstyle="round,pad=0,rounding_size=0.035",
                                    facecolor=col, edgecolor="white",
                                    linewidth=0.6, zorder=8))
        ax.text(bx + 0.135, by + bh - 0.125, str(i + 1), ha="center",
                va="center", fontsize=6.4, fontweight="bold", color="white",
                zorder=9)
    ax.text(IMG_X0, IMG_Y1 - IMG_H - 0.20,
            f"image 004512  ·  2048 x 1445 px  ·  {len(CARDS)} of {N_REGIONS} "
            f"annotated regions shown", ha="left", va="center", fontsize=6.8,
            color=MUTED)


def draw_cards(ax):
    ch = 0.78
    for i, (l1, l2) in enumerate(CARDS):
        col = REGION[i]
        cy = CARD_Y[i]
        ax.add_patch(FancyBboxPatch((CARD_X0, cy - ch / 2), CARD_W, ch,
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor="white", edgecolor=col,
                                    linewidth=1.1, zorder=4))
        ax.add_patch(FancyBboxPatch((CARD_X0 + 0.09, cy + 0.06), 0.20, 0.18,
                                    boxstyle="round,pad=0,rounding_size=0.035",
                                    facecolor=col, edgecolor="none", zorder=5))
        ax.text(CARD_X0 + 0.19, cy + 0.15, str(i + 1), ha="center",
                va="center", fontsize=6.4, fontweight="bold", color="white",
                zorder=6)
        ax.text(CARD_X0 + 0.38, cy + 0.15, l1, ha="left", va="center",
                fontsize=7.4, color=INK, zorder=6)
        ax.text(CARD_X0 + 0.38, cy - 0.09, l2, ha="left", va="center",
                fontsize=7.4, color=INK, zorder=6)
        ax.text(CARD_X0 + 0.38, cy - 0.28, f"region {i + 1}  ·  "
                f"{len(l1.split()) + len(l2.split())} tokens", ha="left",
                va="center", fontsize=6.1, color=FAINT, zorder=6)
        # leader line back to its box, same colour, no arrow head
        x0, x1, y0, y1 = BOXES[i]
        ax.plot([CARD_X0 - 0.02, sx(x1)], [cy, sy((y0 + y1) / 2)],
                color=col, lw=0.8, alpha=0.95, zorder=8,
                solid_capstyle="butt")
    ax.text(CARD_X0, 0.88,
            f"{N_REGIONS} regions  ·  {N_OBJ} objects  ·  {N_REL} "
            f"relationships\n{N_ATTR} attributes  ·  "
            f"{np.mean(WORDS):.1f} words per description",
            ha="left", va="center", fontsize=6.8, color=MUTED,
            linespacing=1.6)


def draw_graph(ax):
    n_obj, n_attr, n_rel = OBJ, ATTR, REL
    chain = [("bottle", n_obj), ("standing on", n_rel), ("counter", n_obj),
             ("below", n_rel), ("window", n_obj), ("tall", n_attr)]
    side = {0: ("green", n_attr), 2: ("wooden", n_attr),
            3: ("supports", n_rel), 4: ("bowl", n_obj),
            5: ("ceramic", n_attr)}

    half = NODE_H / 2
    # trunk edges
    for i in range(len(chain) - 1):
        edge(ax, [(CHAIN_X, ROWS[i] - half),
                  (CHAIN_X, ROWS[i + 1] + half)], "down")
    # branch: counter -> supports -> bowl -> ceramic
    edge(ax, [(CHAIN_X, ROWS[2] - half), (CHAIN_X, ROWS[2] - half - 0.16),
              (ATTR_X, ROWS[2] - half - 0.16), (ATTR_X, ROWS[3] + half)],
         "down")
    edge(ax, [(ATTR_X, ROWS[3] - half), (ATTR_X, ROWS[4] + half)], "down")
    edge(ax, [(ATTR_X, ROWS[4] - half), (ATTR_X, ROWS[5] + half)], "down")
    # attribute stubs
    for r in (0, 2):
        edge(ax, [(CHAIN_X + CHAIN_W / 2, ROWS[r]),
                  (ATTR_X - ATTR_W / 2, ROWS[r])], "right")

    for i, (txt, col) in enumerate(chain):
        node(ax, CHAIN_X, ROWS[i], CHAIN_W, txt, col)
    for r, (txt, col) in side.items():
        node(ax, ATTR_X, ROWS[r], ATTR_W, txt, col)

    # legend
    lx = CHAIN_X - CHAIN_W / 2
    for k, (col, lab) in enumerate([(OBJ, "objects"), (ATTR, "attributes"),
                                    (REL, "relationships")]):
        x = lx + k * 0.98
        ax.add_patch(FancyBboxPatch((x, 0.60), 0.20, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.045",
                                    facecolor=lighten(col, 0.82),
                                    edgecolor=col, linewidth=0.8, zorder=4))
        ax.text(x + 0.26, 0.68, lab, ha="left", va="center", fontsize=6.9,
                color=SUB, zorder=4)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.40, 5.00, "Data representation: from region boxes to a scene "
                        "graph", ha="left", va="center", fontsize=11.6,
            fontweight="bold", color=INK)
    ax.text(0.40, 4.76, "every region description is grounded in one box and "
                        "contributes objects, attributes and relationships to "
                        "the per-image graph", ha="left", va="center",
            fontsize=7.8, color=MUTED)

    column_header(ax, IMG_X0, IMG_W, "Image with region boxes")
    column_header(ax, CARD_X0, CARD_W, "Region descriptions")
    column_header(ax, CHAIN_X - CHAIN_W / 2,
                  ATTR_X + ATTR_W / 2 - CHAIN_X + CHAIN_W / 2, "Scene graph")

    draw_image(ax)
    draw_cards(ax)
    draw_graph(ax)

    savefig(fig, "1129_wm_data_scene_graph.png")


if __name__ == "__main__":
    main()
