"""Dataset structure: splits, sessions and the sequences inside one session.

Three nested rounded containers read left to right, each one expanding a
single element of the container before it.  The expansion is drawn the way
zoom cascades are drawn in dataset papers: two hairlines leaving the parent
row and a very pale wedge filling the space between them, so the eye follows
the path train -> session S014 -> its sequence strip without an arrow.

Level counts sit at the right end of every level header.  The sequence strip
is a black film bar with sprocket holes and four procedurally synthesised
frames of the same room, the actor translating across the frames; a coverage
ribbon underneath marks which frames carry boxes, masks and depth.  A grey
summary card runs along the bottom with the corpus totals.

Inches on an equal-aspect canvas: one radius, one hairline weight, one gap.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
LINE = "#cccccc"
HAIR = "#d5d5d5"
CARD = "#f5f5f5"
DARK = "#3a3a3a"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

SEED = 1536  # original id; figure number is 1131
AR = 1.42
W, H = 11.0, 5.2
Y0, Y1 = 1.34, 4.28
COL = [(0.45, 2.55), (3.15, 6.05), (6.65, 10.55)]
RAD = 0.075

SPLITS = [("train", 8736, 0.700, BLUE), ("val", 1872, 0.150, GREEN),
          ("test", 1872, 0.150, GOLD)]
SESSIONS = [("S014", "kitchen A", "42 min", 38, True),
            ("S015", "kitchen A", "27 min", 24, False),
            ("S016", "workshop B", "51 min", 46, False),
            ("S017", "office C", "33 min", 31, False),
            ("S018", "kitchen D", "45 min", 40, False)]
TOTALS = [("214.6 h", "recorded video"), ("23.1 M", "frames"),
          ("4.82 M", "boxes"), ("1.16 M", "instance masks"),
          ("412", "capture sessions")]


# ---- procedural frames -----------------------------------------------------
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


def room_frame(t, seed=SEED, ny=104):
    """One frame of the session: the actor walks past the counter."""
    nx = int(round(ny * AR))
    rng = np.random.default_rng(seed + int(100 * t))
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * AR
    soft = 2.6 / ny
    hz = 0.58

    wall = np.array([0.700, 0.685, 0.655])
    floor = np.array([0.545, 0.480, 0.415])
    img = np.where((yy < hz)[..., None],
                   wall * (0.92 + 0.22 * yy / hz)[..., None],
                   floor * (1.10 - 0.24 * (yy - hz) / (1 - hz))[..., None])
    img *= (1.0 - 0.26 * _cover(np.abs(yy - hz) - 0.005, soft))[..., None]

    counter = _rbox(xx, yy, 1.10, 0.700, 0.330, 0.130, 0.030)
    cab = _rbox(xx, yy, 0.24, 0.400, 0.170, 0.170, 0.030)
    pot = np.hypot(xx - 0.905, yy - 0.560) - 0.052
    actor_x = 0.30 + 0.62 * t
    body = _capsule(xx, yy, (actor_x, 0.560), (actor_x + 0.02, 0.330), 0.070)
    head = np.hypot(xx - (actor_x + 0.025), yy - 0.262) - 0.055

    subject = np.zeros_like(yy)
    shade = (1.14 - 0.24 * yy)[..., None]
    for sdf, col in [(cab, np.array([0.620, 0.560, 0.470])),
                     (counter, np.array([0.660, 0.625, 0.560])),
                     (pot, np.array([0.400, 0.470, 0.430])),
                     (body, np.array([0.300, 0.360, 0.470])),
                     (head, np.array([0.640, 0.520, 0.430]))]:
        m = _cover(sdf, soft)
        img = img * (1 - m[..., None]) + col * shade * m[..., None]
        subject = np.clip(subject + m, 0, 1)

    img *= (1.0 + 0.065 * (_value_noise(rng, ny, nx) - 0.5) * 2.0)[..., None]
    sharp = _blur(subject, 1.0)[..., None]
    img = _blur3(img, 0.30) * sharp + _blur3(img, 1.3) * (1.0 - sharp)
    r2 = ((xx - AR / 2) / AR) ** 2 + (yy - 0.5) ** 2
    img *= (1.0 - 0.26 * r2)[..., None]
    return np.clip(img * (0.97 + 0.05 * t), 0.0, 1.0)


# ---- containers ------------------------------------------------------------
def container(ax, x0, x1, title, count, zorder=2):
    ax.add_patch(FancyBboxPatch((x0, Y0), x1 - x0, Y1 - Y0,
                                boxstyle=f"round,pad=0,rounding_size={RAD}",
                                facecolor="white", edgecolor=LINE,
                                linewidth=0.9, zorder=zorder))
    ax.text(x0, Y1 + 0.14, title, ha="left", va="bottom", fontsize=8.4,
            fontweight="bold", color=RULE, zorder=zorder + 1)
    ax.text(x1, Y1 + 0.15, count, ha="right", va="bottom", fontsize=7.2,
            color=MUTED, zorder=zorder + 1)


def wedge(ax, p_top, p_bot, q_top, q_bot, color):
    ax.add_patch(Polygon([p_top, q_top, q_bot, p_bot], closed=True,
                         facecolor=lighten(color, 0.93), edgecolor="none",
                         zorder=1))
    for a, b in ((p_top, q_top), (p_bot, q_bot)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=lighten(color, 0.45),
                lw=0.8, zorder=2, solid_capstyle="butt")


def level_splits(ax):
    x0, x1 = COL[0]
    container(ax, x0, x1, "Split", "3")
    ax.add_patch(FancyBboxPatch((x0 + 0.14, Y1 - 0.62), x1 - x0 - 0.28, 0.44,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=CARD, edgecolor=HAIR, linewidth=0.7,
                                zorder=3))
    ax.text(x0 + 0.26, Y1 - 0.30, "RoomScenes-4D", ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=INK, zorder=4)
    ax.text(x0 + 0.26, Y1 - 0.49, "12,480 clips  ·  disjoint subjects",
            ha="left", va="center", fontsize=6.4, color=MUTED, zorder=4)

    box = None
    for k, (name, n, frac, col) in enumerate(SPLITS):
        top = Y1 - 0.78 - k * 0.72
        ax.add_patch(FancyBboxPatch((x0 + 0.14, top - 0.60),
                                    x1 - x0 - 0.28, 0.60,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=lighten(col, 0.93),
                                    edgecolor=col, linewidth=0.9, zorder=3))
        ax.text(x0 + 0.28, top - 0.19, name, ha="left", va="center",
                fontsize=7.6, fontweight="bold", color=INK, zorder=4)
        ax.text(x1 - 0.28, top - 0.19, f"{n:,} clips", ha="right",
                va="center", fontsize=6.6, color=SUB, zorder=4)
        bx0, bx1 = x0 + 0.28, x1 - 0.72
        ax.plot([bx0, bx1], [top - 0.42, top - 0.42], color="white", lw=3.4,
                solid_capstyle="butt", zorder=4)
        ax.plot([bx0, bx0 + frac * (bx1 - bx0)], [top - 0.42, top - 0.42],
                color=col, lw=3.4, solid_capstyle="butt", zorder=5)
        ax.text(x1 - 0.28, top - 0.42, f"{frac * 100:.0f} %", ha="right",
                va="center", fontsize=6.4, color=SUB, zorder=5)
        if k == 0:
            box = (x1 - 0.14, top, top - 0.60)
    return box


def level_sessions(ax):
    x0, x1 = COL[1]
    container(ax, x0, x1, "Sessions in train", "289")
    row = None
    for k, (sid, place, dur, nseq, sel) in enumerate(SESSIONS):
        top = Y1 - 0.16 - k * 0.50
        col = BLUE if sel else HAIR
        ax.add_patch(FancyBboxPatch((x0 + 0.14, top - 0.44),
                                    x1 - x0 - 0.28, 0.44,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=lighten(BLUE, 0.93) if sel
                                    else "white", edgecolor=col,
                                    linewidth=0.9 if sel else 0.7, zorder=3))
        ax.text(x0 + 0.28, top - 0.22, sid, ha="left", va="center",
                fontsize=7.2, family="monospace",
                fontweight="bold" if sel else "normal", color=INK, zorder=4)
        ax.text(x0 + 0.92, top - 0.22, place, ha="left", va="center",
                fontsize=6.6, color=SUB, zorder=4)
        ax.text(x1 - 0.86, top - 0.22, dur, ha="right", va="center",
                fontsize=6.6, color=SUB, zorder=4)
        ax.text(x1 - 0.28, top - 0.22, f"{nseq} seq", ha="right",
                va="center", fontsize=6.6, color=MUTED, zorder=4)
        if sel:
            row = (x1 - 0.14, top, top - 0.44)
    ax.text(x0 + 0.28, Y0 + 0.20, "284 further sessions  ·  4 environments  "
                                  "·  31 subjects", ha="left", va="center",
            fontsize=6.6, color=FAINT, zorder=4)
    return row


def level_sequences(ax):
    x0, x1 = COL[2]
    container(ax, x0, x1, "Sequences in session S014", "38")
    ax.text(x0 + 0.16, Y1 - 0.22, "sequence 07  ·  take 2  ·  1440 x 1014 px "
                                  "·  30 fps", ha="left", va="center",
            fontsize=6.8, color=SUB, zorder=4)

    # film strip: black bar, sprocket holes, four synthesised frames
    fh = 0.56
    fw = fh * AR
    n = 4
    gap = 0.07
    strip_w = n * fw + (n - 1) * gap + 0.20
    sx0 = x0 + (x1 - x0 - strip_w) / 2
    sy1 = Y1 - 0.40
    sy0 = sy1 - (fh + 0.36)
    ax.add_patch(FancyBboxPatch((sx0, sy0), strip_w, sy1 - sy0,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=DARK, edgecolor="none", zorder=3))
    holes = int((strip_w - 0.14) // 0.20)
    for k in range(holes):
        hx = sx0 + 0.10 + k * 0.20
        for hy in (sy0 + 0.055, sy1 - 0.115):
            ax.add_patch(FancyBboxPatch((hx, hy), 0.10, 0.06,
                                        boxstyle="round,pad=0,"
                                                 "rounding_size=0.018",
                                        facecolor="white", edgecolor="none",
                                        zorder=4))
    for k in range(n):
        fx = sx0 + 0.10 + k * (fw + gap)
        fy = sy0 + 0.18
        ax.imshow(room_frame(k / (n - 1.0)),
                  extent=(fx, fx + fw, fy, fy + fh), origin="upper",
                  interpolation="bilinear", zorder=5)
        ax.text(fx + 0.05, fy + 0.08, f"t = {k * 42:03d}", ha="left",
                va="center", fontsize=6.0, color="white", zorder=6)

    # coverage ribbon: which modality is annotated on which stretch
    ry = sy0 - 0.30
    tx0, tx1 = x0 + 0.94, x1 - 0.22
    for k, (lab, col, spans) in enumerate([
            ("boxes", BLUE, [(0.00, 1.00)]),
            ("masks", GREEN, [(0.06, 0.42), (0.55, 0.94)]),
            ("depth", PURPLE, [(0.00, 0.34), (0.62, 1.00)])]):
        y = ry - k * 0.24
        ax.text(x0 + 0.86, y, lab, ha="right", va="center", fontsize=6.5,
                color=SUB, zorder=4)
        ax.plot([tx0, tx1], [y, y], color=CARD, lw=3.6, solid_capstyle="butt",
                zorder=4)
        for a, b in spans:
            ax.plot([tx0 + a * (tx1 - tx0), tx0 + b * (tx1 - tx0)], [y, y],
                    color=col, lw=3.6, solid_capstyle="butt", zorder=5)
    ax.plot([tx0, tx1], [ry - 0.80, ry - 0.80], color=HAIR, lw=0.7, zorder=4)
    for f in np.linspace(0, 1, 5):
        x = tx0 + f * (tx1 - tx0)
        ax.plot([x, x], [ry - 0.80, ry - 0.86], color=HAIR, lw=0.7, zorder=4)
        ax.text(x, ry - 0.98, f"{int(f * 1260)}", ha="center", va="center",
                fontsize=6.0, color=FAINT, zorder=4)
    ax.text((tx0 + tx1) / 2, ry - 1.16, "frame index within the sequence",
            ha="center", va="center", fontsize=6.4, color=MUTED, zorder=4)


def totals_card(ax):
    x0, x1 = COL[0][0], COL[2][1]
    ax.add_patch(FancyBboxPatch((x0, 0.40), x1 - x0, 0.62,
                                boxstyle=f"round,pad=0,rounding_size={RAD}",
                                facecolor=CARD, edgecolor=HAIR, linewidth=0.8,
                                zorder=2))
    step = (x1 - x0) / len(TOTALS)
    for k, (value, label) in enumerate(TOTALS):
        cx = x0 + step * (k + 0.5)
        ax.text(cx, 0.80, value, ha="center", va="center", fontsize=9.6,
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, 0.57, label, ha="center", va="center", fontsize=6.8,
                color=MUTED, zorder=3)
        if k:
            ax.plot([x0 + step * k] * 2, [0.52, 0.90], color=HAIR, lw=0.7,
                    zorder=3)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(COL[0][0], 4.98, "Structure of the corpus: split, session, "
                             "sequence", ha="left", va="center", fontsize=11.6,
            fontweight="bold", color=INK)
    ax.text(COL[0][0], 4.76, "each level expands one element of the level "
                             "before it; counts for the whole level are "
                             "printed at the right edge of its header",
            ha="left", va="center", fontsize=7.8, color=MUTED)

    split_anchor = level_splits(ax)
    session_anchor = level_sessions(ax)
    wedge(ax, (split_anchor[0], split_anchor[1]),
          (split_anchor[0], split_anchor[2]), (COL[1][0], Y1), (COL[1][0], Y0),
          BLUE)
    wedge(ax, (session_anchor[0], session_anchor[1]),
          (session_anchor[0], session_anchor[2]), (COL[2][0], Y1),
          (COL[2][0], Y0), BLUE)
    level_sequences(ax)
    totals_card(ax)

    savefig(fig, "1131_wm_data_structure.png")


if __name__ == "__main__":
    main()
