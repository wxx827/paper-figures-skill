"""Real-robot evaluation grid: four manipulation tasks x three logged trials.

Synthetic imagery: no photograph is used anywhere.  Each cell is rendered
from signed-distance primitives (a capsule chain for the arm, rounded boxes,
rings and half-discs for the props), given material by low-amplitude
multi-scale value noise, and finished with a depth-of-field pair - the
background is blurred hard while the subject gets an unsharp pass.  Swap
``scene`` for a frame loader and the rest of the figure is unchanged.

Camera recipe D, chosen so this figure cannot be mistaken for the warm
wide-angle bench of 1528: a cold grey industrial cell, perforated tool wall
instead of a domestic cabinet, overhead key light pooling on the bench,
raised camera so the bench line sits high, long lens so the vignette is
mild, and an anthracite arm with a bright machined gripper - the inverse of
the white arm with the dark gripper used elsewhere in the group.

One noise field per task row is reused across its three trials, so a row
reads as three attempts at the same bench rather than three different rooms.

The trial outcome is drawn twice: as a hand-drawn tick or cross in the corner
of the cell, and - for the failures - in the geometry itself, where the prop
ends up toppled, dropped or never moved.  The narrow right-hand column
carries the success rate over the full 20-trial evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
HAIR = "#d5d5d5"

SEED = 1529  # original id; figure number is 1124
FW, FH = 10.4, 6.4
LEFT, RIGHT, TOP, BOT = 0.155, 0.982, 0.905, 0.062
WSP, HSP = 0.050, 0.100
RATIOS = [1.0, 1.0, 1.0, 0.55]
NROW, NCOL = 4, len(RATIOS)


def _cell_geometry():
    """Exact tile inches, so overlays drawn in pixel units stay isotropic."""
    tw = (RIGHT - LEFT) * FW
    th = (TOP - BOT) * FH
    u = tw / (sum(RATIOS) * (1.0 + WSP * (NCOL - 1) / NCOL))
    h = th / (NROW + HSP * (NROW - 1))
    return u, h


TILE_W, TILE_H = _cell_geometry()
AR = TILE_W / TILE_H
NY = 152
NX = int(round(NY * AR))

TASKS = [
    ("Pick up the mug", 95.0, (True, True, True)),
    ("Open the drawer", 80.0, (True, False, True)),
    ("Stack two blocks", 70.0, (True, True, False)),
    ("Pour into the bowl", 45.0, (False, True, False)),
]
N_TRIALS = 20

# --- camera / lighting recipe D: cold industrial cell, overhead key --------
HZ = 0.335                 # raised camera -> more bench, less wall
BALANCE = np.array([0.950, 0.988, 1.064])
EXPOSURE = 1.05
VIGNETTE = 0.20            # long lens
BG_SIGMA, FG_SIGMA = 2.4, 0.30
SHARPEN = 0.58
POOL_GAIN = 0.15           # overhead key, kept mild so the bench does not blow
CONTRAST, PIVOT = 1.24, 0.58   # cold cells still need a printable tonal range

SHOULDER = (AR - 0.20, 0.215)
BASE_Y = 0.860
POOL = (0.94, 0.560)       # centre of the overhead light pool on the bench

WALL = np.array([0.580, 0.608, 0.650])
TABLE = np.array([0.648, 0.666, 0.694])
PEGBOARD = np.array([0.512, 0.540, 0.578])
HOLE = np.array([0.255, 0.275, 0.308])
RACK = np.array([0.330, 0.352, 0.388])
SEAM = np.array([0.412, 0.432, 0.462])
ANTHRACITE = np.array([0.252, 0.270, 0.302])    # arm shell
MACHINED = np.array([0.800, 0.815, 0.836])      # gripper and joints
BLUEWARE = np.array([0.235, 0.360, 0.520])      # the mug
AMBER = np.array([0.660, 0.492, 0.155])         # the upper block
TEAL = np.array([0.180, 0.420, 0.415])
PORCELAIN = np.array([0.838, 0.856, 0.880])
GLASS = np.array([0.505, 0.575, 0.618])


# ---- signed-distance primitives -------------------------------------------
def capsule(xx, yy, p0, p1, r):
    bx, by = p1[0] - p0[0], p1[1] - p0[1]
    px, py = xx - p0[0], yy - p0[1]
    t = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-9), 0.0, 1.0)
    return np.hypot(px - t * bx, py - t * by) - r


def rbox(xx, yy, cx, cy, hw, hh, r, ang=0.0):
    px, py = xx - cx, yy - cy
    if ang:
        c, s = np.cos(ang), np.sin(ang)
        px, py = c * px + s * py, -s * px + c * py
    qx = np.abs(px) - (hw - r)
    qy = np.abs(py) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def ring(xx, yy, cx, cy, r, w):
    return np.abs(np.hypot(xx - cx, yy - cy) - r) - w


def cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


def value_noise(rng, ny, nx, octaves=(3, 6, 13, 26),
                weights=(1.0, .55, .30, .17)):
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


def blur3(a, sigma):
    return np.dstack([gaussian_filter(a[..., c], sigma) for c in range(3)])


YI, XI = np.mgrid[0:NY, 0:NX]
YY = YI / (NY - 1.0)
XX = XI / (NX - 1.0) * AR
SOFT = 2.3 / NY

_rng = np.random.default_rng(SEED)
FIELDS = [value_noise(_rng, NY, NX) for _ in range(NROW)]
GRAIN = _rng.normal(0.0, 1.0, (NY, NX))


def arm(ee, opening):
    sx, sy = SHOULDER
    dx, dy = ee[0] - sx, ee[1] - sy
    d = float(np.hypot(dx, dy))
    nx_, ny_ = -dy / d, dx / d
    if ny_ > 0:
        nx_, ny_ = -nx_, -ny_
    arc = 0.150 + 0.110 * (1.0 - min(d / 1.05, 1.0))
    ex = 0.5 * (sx + ee[0]) + nx_ * arc
    ey = 0.5 * (sy + ee[1]) + ny_ * arc
    links = np.minimum.reduce([
        rbox(XX, YY, sx, BASE_Y, 0.095, 0.034, 0.015),
        capsule(XX, YY, (sx, BASE_Y), (sx, sy), 0.046),
        capsule(XX, YY, (sx, sy), (ex, ey), 0.040),
        capsule(XX, YY, (ex, ey), ee, 0.032),
    ])
    joints = np.minimum(capsule(XX, YY, (sx, sy), (sx, sy), 0.029),
                        capsule(XX, YY, (ex, ey), (ex, ey), 0.026))
    fx, fy = ee
    grip = np.minimum.reduce([
        capsule(XX, YY, (fx, fy - 0.012), (fx, fy + 0.018), 0.022),
        capsule(XX, YY, (fx - opening, fy + 0.018), (fx + opening, fy + 0.018),
                0.011),
        capsule(XX, YY, (fx - opening, fy + 0.018), (fx - opening, fy + 0.076),
                0.010),
        capsule(XX, YY, (fx + opening, fy + 0.018), (fx + opening, fy + 0.076),
                0.010),
    ])
    return links, joints, grip


# ---- per-task prop layouts -------------------------------------------------
def task_layers(task, trial, ok, jx, jy):
    """Return (layers, end-effector, opening, shadow anchors).

    ``layers`` is a list of (sdf, colour, brightness) composited in order.
    ``jx`` / ``jy`` are the small per-trial jitters that keep the three
    attempts from looking like copies.
    """
    L = []
    if task == 0:                                   # pick up the mug
        mx, my = 0.90 + jx, 0.628 + jy
        if ok:
            L.append((np.minimum(
                rbox(XX, YY, mx, my, 0.066, 0.080, 0.018),
                ring(XX, YY, mx + 0.098, my + 0.006, 0.042, 0.013)),
                BLUEWARE, 1.0))
            ee = (mx, my - 0.152)
        else:                                       # dropped: lying on its side
            L.append((np.minimum(
                rbox(XX, YY, mx, my + 0.096, 0.078, 0.066, 0.018, -0.88),
                ring(XX, YY, mx - 0.082, my + 0.122, 0.040, 0.013)),
                BLUEWARE, 1.0))
            ee = (mx + 0.045, my - 0.062)
        op, anchors = 0.076, [(mx, 0.790)]
    elif task == 1:                                 # open the drawer
        cx, cy = 0.92 + jx, 0.556 + jy
        pull = 0.108 if ok else 0.026
        L.append((rbox(XX, YY, cx, cy, 0.205, 0.162, 0.018), PORCELAIN, 0.86))
        L.append((rbox(XX, YY, cx, cy + 0.028 + pull, 0.196, 0.012, 0.006),
                  SEAM, 0.85))
        L.append((rbox(XX, YY, cx, cy + 0.078 + pull, 0.192, 0.062, 0.014),
                  PORCELAIN, 1.14))
        L.append((rbox(XX, YY, cx, cy + 0.078 + pull, 0.044, 0.018, 0.008),
                  ANTHRACITE, 1.0))
        ee, op = (cx, cy + 0.002 + pull), 0.052
        anchors = [(cx, 0.760)]
    elif task == 2:                                 # stack two blocks
        bx, by = 0.88 + jx, 0.652 + jy
        L.append((rbox(XX, YY, bx, by, 0.086, 0.064, 0.014), TEAL, 1.0))
        if ok:
            L.append((rbox(XX, YY, bx + 0.006, by - 0.124, 0.080, 0.060,
                           0.014), AMBER, 1.05))
            ee, op = (bx + 0.006, by - 0.186), 0.074
            anchors = [(bx, 0.790)]
        else:                                       # toppled off the stack
            L.append((rbox(XX, YY, bx + 0.255, by + 0.052, 0.080, 0.060,
                           0.014, -0.66), AMBER, 1.05))
            ee, op = (bx + 0.120, by - 0.150), 0.074
            anchors = [(bx, 0.790), (bx + 0.25, 0.800)]
    else:                                           # pour into the bowl
        bwx, bwy = 0.80 + jx, 0.646 + jy
        r_b = 0.108
        half = np.maximum(np.hypot(XX - bwx, YY - bwy) - r_b, -(YY - bwy))
        L.append((half, PORCELAIN, 1.02))           # bowl body
        L.append((np.maximum(np.hypot(XX - bwx, YY - bwy) - (r_b - 0.022),
                             -(YY - bwy)), PORCELAIN, 0.74))   # inner shadow
        L.append((rbox(XX, YY, bwx, bwy, r_b, 0.013, 0.010), PORCELAIN, 1.16))
        tilt = -0.95 if ok else -0.12
        btx, bty = bwx + 0.250, bwy - 0.215
        hh = 0.108
        mouth = (btx + np.sin(tilt) * hh, bty - np.cos(tilt) * hh)
        neck = (btx + np.sin(tilt) * (hh + 0.052),
                bty - np.cos(tilt) * (hh + 0.052))
        L.append((np.minimum(rbox(XX, YY, btx, bty, 0.050, hh, 0.020, tilt),
                             capsule(XX, YY, mouth, neck, 0.021)), GLASS, 1.0))
        if ok:                                      # a thin falling stream
            L.append((capsule(XX, YY, neck, (bwx + 0.030, bwy - 0.004),
                              0.008), PORCELAIN, 1.20))
        ee, op = (btx + 0.052, bty - 0.052), 0.062
        anchors = [(bwx, 0.786)]
    return L, ee, op, anchors


def scene(task, trial, ok):
    rng = np.random.default_rng(SEED + 31 * task + trial)
    jx = float(rng.uniform(-0.035, 0.035))
    jy = float(rng.uniform(-0.014, 0.014))

    up = 0.92 + 0.16 * (YY / HZ)
    dn = 1.08 - 0.26 * np.clip((YY - HZ) / (1.0 - HZ), 0, 1)
    bg = np.where((YY < HZ)[..., None], WALL * up[..., None],
                  TABLE * dn[..., None])
    bg *= 1.0 - 0.26 * cover(np.abs(YY - HZ) - 0.004, SOFT)[..., None]

    # perforated tool wall: panel, hole lattice, rack beam, two hung tools
    peg = rbox(XX, YY, 0.44, 0.150, 0.415, 0.128, 0.014)
    m_peg = cover(peg, SOFT)[..., None]
    bg = bg * (1 - m_peg) + m_peg * PEGBOARD * (1.05 - 0.16 * YY)[..., None]
    pitch = 0.058
    hx = np.mod(XX - 0.062, pitch) - pitch / 2.0
    hy = np.mod(YY - 0.052, pitch) - pitch / 2.0
    holes = np.maximum(np.hypot(hx, hy) - 0.0135, peg + 0.020)
    m_hole = cover(holes, SOFT)[..., None]
    bg = bg * (1 - m_hole) + m_hole * HOLE

    beam = rbox(XX, YY, 0.46, 0.297, 0.470, 0.013, 0.006)
    m_beam = cover(beam, SOFT)[..., None]
    bg = bg * (1 - m_beam) + m_beam * RACK
    tools = np.minimum.reduce([
        capsule(XX, YY, (0.60, 0.190), (0.60, 0.286), 0.011),
        ring(XX, YY, 0.60, 0.178, 0.019, 0.007),
        rbox(XX, YY, 0.695, 0.238, 0.026, 0.050, 0.010)])
    m_tool = cover(tools, SOFT)[..., None]
    bg = bg * (1 - m_tool) + m_tool * RACK * 0.88

    # bench: front lip, one welded seam, a bin and a coil of cable on it
    bg *= 1.0 - 0.20 * cover(np.abs(YY - 0.958) - 0.005, SOFT)[..., None]
    bg *= 1.0 - 0.10 * cover(np.abs(YY - 0.545) - 0.003, SOFT)[..., None]
    bin_ = np.minimum(rbox(XX, YY, 0.115, 0.612, 0.072, 0.058, 0.010),
                      rbox(XX, YY, 0.115, 0.552, 0.078, 0.010, 0.004))
    m_bin = cover(bin_, SOFT)[..., None]
    bg = bg * (1 - m_bin) + m_bin * RACK * 1.18
    coil = np.minimum(ring(XX, YY, 0.395, 0.605, 0.052, 0.011),
                      ring(XX, YY, 0.395, 0.605, 0.028, 0.009))
    m_coil = cover(coil, SOFT)[..., None]
    bg = bg * (1 - m_coil) + m_coil * np.array([0.215, 0.230, 0.258])

    layers, ee, opening, anchors = task_layers(task, trial, ok, jx, jy)

    # overhead key: shadows sit tight underneath rather than trailing sideways
    shade = np.exp(-(((XX - SHOULDER[0]) / 0.14) ** 2
                     + ((YY - 0.912) / 0.026) ** 2))
    for ax_, ay_ in anchors:
        shade += 0.9 * np.exp(-(((XX - ax_) / 0.075) ** 2
                                + ((YY - ay_) / 0.017) ** 2))
    bg *= (1.0 - 0.34 * np.clip(shade, 0, 1))[..., None]

    lit = (1.16 - 0.24 * YY)[..., None]
    fg = bg.copy()
    sharp = np.zeros((NY, NX))
    for sdf, col, gain in layers:
        m = cover(sdf, SOFT)
        fg = fg * (1 - m[..., None]) + m[..., None] * col * gain * lit
        sharp = np.clip(sharp + m, 0, 1)

    links, joints, grip = arm(ee, opening)
    m_lnk = cover(links, SOFT)
    m_jnt = cover(joints, SOFT)
    m_grp = cover(grip, SOFT)
    rim = cover(links + 0.010, SOFT) - cover(links + 0.001, SOFT)
    fg = fg * (1 - m_lnk[..., None]) + m_lnk[..., None] * ANTHRACITE * lit
    fg = fg * (1 - m_jnt[..., None]) + m_jnt[..., None] * MACHINED * 0.86 * lit
    fg += (0.13 * rim * np.clip(1.0 - 2.2 * YY, 0, 1))[..., None]
    fg = fg * (1 - m_grp[..., None]) + m_grp[..., None] * MACHINED * 1.06 * lit
    sharp = np.clip(sharp + m_lnk + m_jnt + m_grp, 0, 1)

    field = np.roll(FIELDS[task], (5 * trial, -9 * trial), axis=(0, 1))
    tex = (1.0 + 0.068 * (field - 0.5) * 2.0
           + 0.016 * np.roll(GRAIN, 3 * trial, axis=1))[..., None]
    fg, bg = fg * tex, bg * tex

    sharp = gaussian_filter(sharp, 0.8)[..., None]
    out = blur3(fg, FG_SIGMA) * sharp + blur3(bg, BG_SIGMA) * (1.0 - sharp)
    out = np.clip(out + SHARPEN * (out - blur3(out, 0.95)) * sharp, 0.0, 1.0)
    pool = 1.0 + POOL_GAIN * np.exp(-(((XX - POOL[0]) / 0.62) ** 2
                                      + ((YY - POOL[1]) / 0.46) ** 2))
    r2 = ((XX - AR / 2) / AR) ** 2 + (YY - 0.5) ** 2
    out = out * pool[..., None] * (1.0 - VIGNETTE * r2)[..., None]
    out = out * EXPOSURE * BALANCE
    out = PIVOT + CONTRAST * (out - PIVOT)
    return np.clip(out * (1.0 + 0.009 * rng.normal(0, 1, (NY, NX)))[..., None],
                   0.0, 1.0)


# ---- overlays --------------------------------------------------------------
def outcome_badge(ax, ok):
    r = 0.088 * NY
    cx, cy = NX - 1.62 * r, 1.62 * r
    col = GREEN if ok else RED
    ax.add_patch(Circle((cx, cy), r, facecolor="white", alpha=0.90,
                        edgecolor="none", zorder=6))
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=col,
                        linewidth=1.0, zorder=7))
    if ok:
        ax.plot([cx - 0.46 * r, cx - 0.10 * r, cx + 0.52 * r],
                [cy + 0.04 * r, cy + 0.44 * r, cy - 0.46 * r], color=col,
                lw=1.7, solid_capstyle="round", solid_joinstyle="miter",
                zorder=8)
    else:
        for sx in (1, -1):
            ax.plot([cx - 0.40 * r * sx, cx + 0.40 * r * sx],
                    [cy - 0.40 * r, cy + 0.40 * r], color=col, lw=1.7,
                    solid_capstyle="round", zorder=8)


def image_cell(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, NX)
    ax.set_ylim(NY, 0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#5f5f5f")


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(NROW, NCOL, width_ratios=RATIOS, left=LEFT,
                          right=RIGHT, top=TOP, bottom=BOT, wspace=WSP,
                          hspace=HSP)

    for i, (name, rate, outcomes) in enumerate(TASKS):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(scene(i, j, outcomes[j]), interpolation="bilinear",
                      aspect="auto", extent=(0, NX, NY, 0), zorder=1)
            image_cell(ax)
            outcome_badge(ax, outcomes[j])
            if i == 0:
                ax.set_title(f"Trial {j + 1}", fontsize=8.6,
                             fontweight="bold", pad=4.0)
            if j == 0:
                ax.text(-0.035, 0.62, name, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8.8,
                        fontweight="bold", color=INK)
                ax.text(-0.035, 0.30, f"n = {N_TRIALS} trials",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=6.8, color=MUTED)

        axr = fig.add_subplot(gs[i, 3])
        axr.barh([0], [100.0], height=0.34, color="#f0f0f0", edgecolor=HAIR,
                 linewidth=0.6, zorder=1)
        axr.barh([0], [rate], height=0.34, color=BLUE, edgecolor="none",
                 zorder=2)
        axr.text(104.0, 0, f"{rate:.0f} %", va="center", ha="left",
                 fontsize=7.6, fontweight="bold", color=INK, zorder=3)
        axr.set_xlim(0, 132)
        axr.set_ylim(-0.62, 0.62)
        axr.set_yticks([])
        axr.spines["left"].set_visible(False)
        axr.spines["bottom"].set_color("#8a8a8a")
        axr.spines["bottom"].set_linewidth(0.6)
        if i == 0:
            axr.set_title("Success rate", fontsize=8.6, fontweight="bold",
                          pad=4.0)
        if i == NROW - 1:
            axr.set_xticks([0, 50, 100])
            axr.tick_params(labelsize=6.6, length=2.2)
            axr.set_xlabel("succeeded (%)", fontsize=7.0, labelpad=1.5)
        else:
            axr.set_xticks([0, 50, 100])
            axr.set_xticklabels([])
            axr.tick_params(length=2.2)

    savefig(fig, "1124_wm_emb_task_grid.png")


if __name__ == "__main__":
    main()
