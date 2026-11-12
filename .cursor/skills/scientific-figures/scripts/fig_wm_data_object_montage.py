"""Object sample montage: one annotated instance per class, 5 x 6 grid.

Every cell is a small procedural close-up -- a horizon, a wall and table
gradient, multi-scale value noise and a blurred background behind a sharp
object built from signed-distance primitives (tall vessel, wide slab, disc,
cup with a handle, dome, stemmed shade, stacked boxes, angled stick).  The
box drawn on each cell is measured from the rendered object mask, so it is a
genuinely tight box rather than a hand-placed rectangle.

The thirty classes fall into three families, each with its own colour and its
own header rule: two dense families of twelve and the long tail of six.  The
class chip sits in the bottom-right corner of the cell, white on the image,
the way sample sheets in detection papers print it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"

GOLD = "#c9a227"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"

SEED = 1537  # original id; figure number is 1132
AR = 1.40
W, H = 9.6, 7.2
GX0, GX1 = 0.55, 8.95
GAP = 0.06
NCOL = 6
CELL_W = (GX1 - GX0 - (NCOL - 1) * GAP) / NCOL
CELL_H = CELL_W / AR

# archetype, body colour, accent colour
MAT = {
    "clay": (0.545, 0.355, 0.250), "steel": (0.455, 0.480, 0.515),
    "white": (0.870, 0.860, 0.835), "wood": (0.500, 0.360, 0.220),
    "navy": (0.210, 0.270, 0.380), "olive": (0.335, 0.390, 0.235),
    "rust": (0.585, 0.290, 0.195), "slate": (0.240, 0.250, 0.275),
    "sand": (0.735, 0.640, 0.440), "teal": (0.185, 0.385, 0.375),
}
SCALE = 1.45          # how much of the crop the object should occupy

# class, archetype, material, backdrop.  Backdrops are cycled deliberately so
# that no two neighbouring cells share a scene recipe.
GROUPS = [
    ("Kitchen and tableware", GOLD, [
        ("mug", "cup", "clay", "counter"), ("bowl", "dome", "white", "studio"),
        ("plate", "disc", "white", "shelf"), ("kettle", "tall", "steel",
                                              "floor"),
        ("pan", "disc", "slate", "outdoor"), ("jar", "tall", "sand",
                                              "counter"),
        ("bottle", "tall", "olive", "shelf"), ("glass", "stem", "white",
                                               "studio"),
        ("teapot", "cup", "teal", "floor"), ("tray", "wide", "wood",
                                             "counter"),
        ("blender", "boxy", "steel", "outdoor"), ("toaster", "boxy", "slate",
                                                  "shelf")]),
    ("Furniture and appliances", BLUE, [
        ("chair", "boxy", "wood", "floor"), ("stool", "stem", "wood",
                                             "outdoor"),
        ("lamp", "stem", "sand", "studio"), ("side table", "wide", "wood",
                                             "counter"),
        ("shelf", "wide", "clay", "shelf"), ("monitor", "boxy", "slate",
                                             "studio"),
        ("laptop", "wide", "steel", "counter"), ("keyboard", "wide", "slate",
                                                 "floor"),
        ("printer", "boxy", "white", "shelf"), ("fan", "disc", "steel",
                                                "outdoor"),
        ("clock", "disc", "white", "studio"), ("speaker", "tall", "navy",
                                               "floor")]),
    ("Long-tail classes", PURPLE, [
        ("ladle", "stick", "steel", "counter"), ("whisk", "stick", "steel",
                                                 "studio"),
        ("grater", "tall", "steel", "shelf"), ("corkscrew", "stick", "rust",
                                               "floor"),
        ("timer", "disc", "rust", "outdoor"), ("kitchen scale", "wide",
                                               "white", "counter")]),
]


# ---- procedural cells ------------------------------------------------------
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


def _value_noise(rng, ny, nx, octaves=(3, 6, 12), weights=(1.0, .55, .30)):
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


def _parts(arch, xx, yy, rng, cx, base):
    """Signed-distance parts of one archetype, sorted back to front."""
    s = SCALE
    j = rng.uniform(-0.02, 0.02) * s
    if arch == "tall":
        hh = rng.uniform(0.165, 0.215) * s
        hw = rng.uniform(0.062, 0.085) * s
        body = _rbox(xx, yy, cx, base - hh, hw, hh, min(hw * 0.6, 0.06))
        neck = _capsule(xx, yy, (cx, base - 2 * hh),
                        (cx, base - 2 * hh - rng.uniform(0.05, 0.10) * s),
                        hw * 0.34)
        return [(np.minimum(body, neck), 1.00)]
    if arch == "wide":
        hh = rng.uniform(0.045, 0.070) * s
        hw = rng.uniform(0.180, 0.230) * s
        slab = _rbox(xx, yy, cx, base - hh, hw, hh, 0.030)
        top = _rbox(xx, yy, cx + j, base - 2 * hh - 0.040 * s, hw * 0.62,
                    0.040 * s, 0.020)
        return [(slab, 1.00), (top, 1.16)]
    if arch == "disc":
        r = rng.uniform(0.135, 0.168) * s
        return [(np.hypot(xx - cx, yy - (base - r)) - r, 1.00),
                (np.hypot(xx - cx, yy - (base - r)) - r * 0.46, 1.20)]
    if arch == "cup":
        hw = rng.uniform(0.085, 0.105) * s
        hh = rng.uniform(0.090, 0.115) * s
        body = _rbox(xx, yy, cx, base - hh, hw, hh, 0.040)
        h1 = _capsule(xx, yy, (cx + hw, base - hh - 0.060 * s),
                      (cx + hw + 0.095 * s, base - hh), 0.024)
        h2 = _capsule(xx, yy, (cx + hw + 0.095 * s, base - hh),
                      (cx + hw, base - hh + 0.060 * s), 0.024)
        return [(np.minimum.reduce([body, h1, h2]), 1.00)]
    if arch == "dome":
        r = rng.uniform(0.135, 0.165) * s
        disc = np.hypot(xx - cx, yy - (base - r)) - r
        return [(np.maximum(disc, (base - r) - yy), 1.00),
                (_rbox(xx, yy, cx, base - r + 0.012, r * 0.92, 0.016, 0.008),
                 0.80)]
    if arch == "stem":
        r = rng.uniform(0.100, 0.128) * s
        shade = np.hypot(xx - cx, yy - (base - 0.330 * s)) - r
        stem = _capsule(xx, yy, (cx, base - 0.320 * s), (cx, base - 0.028),
                        0.024)
        foot = _rbox(xx, yy, cx, base - 0.024, 0.100 * s, 0.024, 0.010)
        return [(stem, 0.84), (foot, 0.84), (shade, 1.10)]
    if arch == "boxy":
        hw = rng.uniform(0.130, 0.162) * s
        hh = rng.uniform(0.095, 0.125) * s
        body = _rbox(xx, yy, cx, base - hh, hw, hh, 0.034)
        cap = _rbox(xx, yy, cx + j, base - 2 * hh - 0.075 * s, hw * 0.70,
                    0.075 * s, 0.028)
        return [(body, 1.00), (cap, 1.22)]
    # stick
    tilt = rng.uniform(0.20, 0.28) * s
    shaft = _capsule(xx, yy, (cx - 0.180 * s, base - 0.040),
                     (cx + 0.180 * s, base - 0.040 - tilt), 0.026)
    knob = np.hypot(xx - (cx + 0.186 * s), yy - (base - 0.040 - tilt)) \
        - 0.062 * s
    return [(shaft, 1.00), (knob, 0.86)]


def _backdrop(kind, xx, yy, rng, soft):
    """One of five scene recipes, so no two neighbouring cells match.

    Returns the background image and the surface height the object rests on.
    """
    if kind == "studio":                       # seamless sweep, no horizon
        tone = np.array([0.815, 0.805, 0.790]) * rng.uniform(0.93, 1.03)
        img = np.repeat((tone[None, None, :]), yy.shape[0], axis=0)
        img = np.repeat(img, yy.shape[1], axis=1) \
            * (1.14 - 0.30 * yy)[..., None]
        return img, rng.uniform(0.84, 0.89)
    if kind == "outdoor":                      # pale sky over a grey ground
        hz = rng.uniform(0.30, 0.38)
        sky = np.array([0.775, 0.800, 0.820]) * rng.uniform(0.96, 1.04)
        ground = np.array([0.505, 0.520, 0.470]) * rng.uniform(0.94, 1.06)
        img = np.where((yy < hz)[..., None],
                       sky * (1.06 - 0.16 * yy / hz)[..., None],
                       ground * (1.14 - 0.30 * (yy - hz) / (1 - hz))
                       [..., None])
        return img, rng.uniform(0.80, 0.86)
    if kind == "floor":                        # cool wall, high horizon
        hz = rng.uniform(0.24, 0.32)
        wall = np.array([0.605, 0.615, 0.635]) * rng.uniform(0.94, 1.05)
        floor = np.array([0.470, 0.450, 0.430]) * rng.uniform(0.94, 1.06)
        img = np.where((yy < hz)[..., None],
                       wall * (0.92 + 0.20 * yy / hz)[..., None],
                       floor * (1.22 - 0.36 * (yy - hz) / (1 - hz))
                       [..., None])
        img *= (1.0 - 0.20 * _cover(np.abs(yy - hz) - 0.004, soft))[..., None]
        return img, rng.uniform(0.82, 0.88)
    if kind == "shelf":                        # dim wall with blurred clutter
        hz = rng.uniform(0.62, 0.70)
        wall = np.array([0.545, 0.530, 0.505]) * rng.uniform(0.94, 1.06)
        board = np.array([0.610, 0.500, 0.375]) * rng.uniform(0.94, 1.05)
        img = np.where((yy < hz)[..., None],
                       wall * (0.88 + 0.26 * yy / hz)[..., None],
                       board * (1.10 - 0.24 * (yy - hz) / (1 - hz))
                       [..., None])
        for _ in range(int(rng.integers(2, 5))):
            bx = rng.uniform(0.08, AR - 0.08)
            bw = rng.uniform(0.055, 0.115)
            bh = rng.uniform(0.10, 0.24)
            blob = _rbox(xx, yy, bx, hz - bh, bw, bh, 0.03)
            m = _cover(blob, soft * 3.0)
            col = np.array([rng.uniform(0.35, 0.62), rng.uniform(0.35, 0.58),
                            rng.uniform(0.32, 0.55)])
            img = img * (1 - m[..., None]) + col * m[..., None]
        img *= (1.0 - 0.26 * _cover(np.abs(yy - hz) - 0.006, soft))[..., None]
        return img, rng.uniform(0.84, 0.90)
    hz = rng.uniform(0.40, 0.50)               # "counter": warm wood worktop
    wall = np.array([0.760, 0.750, 0.725]) * rng.uniform(0.94, 1.05)
    table = np.array([0.665, 0.610, 0.535]) * rng.uniform(0.93, 1.06)
    img = np.where((yy < hz)[..., None],
                   wall * (0.90 + 0.24 * yy / hz)[..., None],
                   table * (1.10 - 0.26 * (yy - hz) / (1 - hz))[..., None])
    img *= (1.0 - 0.24 * _cover(np.abs(yy - hz) - 0.005, soft))[..., None]
    return img, rng.uniform(0.82, 0.87)


def object_tile(arch, mat, back, seed, ny=104):
    """Close-up of one object; returns the image and its tight mask box."""
    nx = int(round(ny * AR))
    rng = np.random.default_rng(seed)
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * AR
    soft = 2.4 / ny

    img, base = _backdrop(back, xx, yy, rng, soft)

    cx = AR / 2 + rng.uniform(-0.04, 0.04)
    shadow = np.exp(-(((xx - cx) / 0.28) ** 2
                      + ((yy - base) / 0.036) ** 2))
    img *= (1.0 - rng.uniform(0.28, 0.46) * shadow)[..., None]

    body = np.array(MAT[mat])
    mask = np.zeros_like(yy)
    # key light: side varies per cell, so the lit face is not always the left
    side = 1.0 if rng.random() < 0.5 else -1.0
    strength = rng.uniform(0.14, 0.26)
    shade = ((1.20 + rng.uniform(0.0, 0.10) - 0.30 * yy)
             * (1.0 + strength * side * (0.5 - xx / AR)))[..., None]
    for sdf, tone in _parts(arch, xx, yy, rng, cx, base):
        m = _cover(sdf, soft)
        img = img * (1 - m[..., None]) + body * tone * shade * m[..., None]
        rim = _cover(np.abs(sdf) - 0.009, soft)
        img *= (1.0 - 0.26 * rim)[..., None]
        mask = np.clip(mask + m, 0, 1)

    tex = 1.0 + rng.uniform(0.05, 0.10) * (_value_noise(rng, ny, nx) - 0.5) * 2
    grain = 1.0 + rng.uniform(0.012, 0.028) * rng.normal(0.0, 1.0, (ny, nx))
    img *= (tex * grain)[..., None]

    sharp = _blur(mask, 1.0)[..., None]
    img = _blur3(img, 0.30) * sharp \
        + _blur3(img, rng.uniform(1.2, 2.2)) * (1.0 - sharp)
    r2 = ((xx - AR / 2) / AR) ** 2 + (yy - 0.5) ** 2
    img = np.clip(img * (1.0 - rng.uniform(0.18, 0.34) * r2)[..., None],
                  0.0, 1.0)

    solid = mask > 0.5
    rows = np.where(solid.any(axis=1))[0]
    cols = np.where(solid.any(axis=0))[0]
    pad = 2
    v0 = max(rows[0] - pad, 0) / (ny - 1.0)
    v1 = min(rows[-1] + pad, ny - 1) / (ny - 1.0)
    u0 = max(cols[0] - pad, 0) / (nx - 1.0)
    u1 = min(cols[-1] + pad, nx - 1) / (nx - 1.0)
    return img, (u0, u1, v0, v1)


# ---- montage ---------------------------------------------------------------
def draw_cell(ax, x, y, cls, arch, mat, back, color, seed):
    """x, y is the top-left corner of the cell in inches."""
    img, (u0, u1, v0, v1) = object_tile(arch, mat, back, seed)
    ax.imshow(img, extent=(x, x + CELL_W, y - CELL_H, y), origin="upper",
              interpolation="bilinear", zorder=3)
    ax.add_patch(Rectangle((x, y - CELL_H), CELL_W, CELL_H, facecolor="none",
                           edgecolor="white", linewidth=0.7, zorder=6))
    ax.add_patch(Rectangle((x + u0 * CELL_W, y - v1 * CELL_H),
                           (u1 - u0) * CELL_W, (v1 - v0) * CELL_H,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           zorder=5))
    ax.text(x + CELL_W - 0.05, y - CELL_H + 0.115, cls, ha="right",
            va="center", fontsize=6.2, color=INK, zorder=7,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.94))


def instance_counts(rng):
    """Per-class instance counts; every printed total is summed from these."""
    counts = []
    for gi, (_, _, classes) in enumerate(GROUPS):
        n = len(classes)
        if gi < 2:
            c = rng.integers(140, 420, size=n)
        else:
            c = rng.integers(18, 39, size=n)
        counts.append(np.asarray(c, dtype=int))
    return counts


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    counts = instance_counts(rng)
    n_classes = sum(len(g[2]) for g in GROUPS)
    n_inst = int(sum(c.sum() for c in counts))
    tail = counts[-1]
    tile_shape = object_tile(*GROUPS[0][2][0][1:], SEED)[0].shape

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(GX0, 6.98, "One annotated instance per object class",
            ha="left", va="center", fontsize=11.6, fontweight="bold",
            color=INK)
    ax.text(GX0, 6.76, f"{n_classes} classes in {len(GROUPS)} families, "
                       f"{n_inst:,} instances in total; the rectangle on each "
                       f"crop is the stored tight box, measured from the "
                       f"instance mask", ha="left", va="center", fontsize=7.8,
            color=MUTED)

    y = 6.42
    for gi, (name, color, classes) in enumerate(GROUPS):
        ax.add_patch(FancyBboxPatch((GX0, y - 0.145), 0.155, 0.145,
                                    boxstyle="round,pad=0,rounding_size=0.035",
                                    facecolor=lighten(color, 0.55),
                                    edgecolor=color, linewidth=0.8, zorder=4))
        ax.text(GX0 + 0.23, y - 0.070, name, ha="left", va="center",
                fontsize=8.2, fontweight="bold", color=RULE, zorder=4)
        ax.text(GX1, y - 0.070,
                f"{len(classes)} classes  ·  {counts[gi].sum():,} instances",
                ha="right", va="center", fontsize=6.8, color=MUTED, zorder=4)
        ax.plot([GX0, GX1], [y - 0.215, y - 0.215], color=HAIR, lw=0.8,
                zorder=3)
        y -= 0.30

        for k, (cls, arch, mat, back) in enumerate(classes):
            r, c = divmod(k, NCOL)
            draw_cell(ax, GX0 + c * (CELL_W + GAP),
                      y - r * (CELL_H + GAP), cls, arch, mat, back, color,
                      SEED + 37 * gi + 7 * k)
        nrow = int(np.ceil(len(classes) / NCOL))
        y -= nrow * CELL_H + (nrow - 1) * GAP + 0.30

    ax.text(GX0, y + 0.10,
            f"crops are {tile_shape[1]} x {tile_shape[0]} px, sampled "
            f"uniformly over the training split; the {len(tail)} long-tail "
            f"classes hold between {tail.min()} and {tail.max()} instances "
            f"each", ha="left", va="center", fontsize=6.8, color=FAINT)
    ax.text(GX1, y + 0.10, f"seed {SEED}", ha="right", va="center",
            fontsize=6.8, color=FAINT)
    savefig(fig, "1132_wm_data_object_montage.png")


if __name__ == "__main__":
    main()
