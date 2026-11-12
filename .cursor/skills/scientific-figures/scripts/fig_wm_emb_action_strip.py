"""Nature-style keyframe strip of a table-top pick-and-place rollout.

Synthetic imagery: no photograph is used anywhere, and no frame is a flat
colour patch either.  Every tile is rendered from signed-distance primitives
in ``frame_rgb`` - swap that one function for a video loader and the rest of
the figure is unchanged.

Six frames (a)-(f) follow one 7-DoF arm through approach, descent, grasp,
lift, transport and release.  The tiles use capsules for the links and
rounded boxes for the base and the manipulated block, are given material by
low-amplitude multi-scale value noise, and are finally split into two depth
zones - the background is blurred hard, the arm and the block stay sharp.
That depth-of-field split is what makes a synthetic frame read as a camera
image.

All six frames share one underlying noise field with a per-frame shift, so
they look like six moments of the same scene rather than six scenes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREEN = "#2e7d5b"

SEED = 1528  # original id; figure number is 1123
FW, FH = 11.6, 2.8
LEFT, RIGHT, TOP, BOT = 0.010, 0.990, 0.772, 0.238
WSP = 0.045
NCOL = 6

TILE_W = (RIGHT - LEFT) * FW / (NCOL + (NCOL - 1) * WSP)
TILE_H = (TOP - BOT) * FH
AR = TILE_W / TILE_H                      # tile width / height, isotropic px

NY = 210
NX = int(round(NY * AR))

HZ = 0.455                                # wall / table horizon
SHOULDER = (AR - 0.17, 0.300)
BASE_Y = 0.855
TABLE_Y = 0.735                           # top face the props stand on
TRAY = (0.66, 0.752)

# end-effector track, gripper opening, whether the block rides the gripper
FRAMES = [
    ((0.280, 0.452), 0.072, False, "Approach the block", "t = 0.0 s"),
    ((0.280, 0.596), 0.072, False, "Descend and align", "t = 1.4 s"),
    ((0.280, 0.612), 0.038, False, "Close the gripper", "t = 2.1 s"),
    ((0.305, 0.468), 0.038, True, "Lift clear of the table", "t = 2.9 s"),
    ((0.500, 0.420), 0.038, True, "Transport to the tray", "t = 4.2 s"),
    ((0.660, 0.548), 0.072, False, "Release on the tray", "t = 5.6 s"),
]
BLOCK_HOME = (0.280, 0.686)
BLOCK_HH = 0.049


# ---- signed-distance primitives -------------------------------------------
def capsule(xx, yy, p0, p1, r):
    bx, by = p1[0] - p0[0], p1[1] - p0[1]
    px, py = xx - p0[0], yy - p0[1]
    t = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-9), 0.0, 1.0)
    return np.hypot(px - t * bx, py - t * by) - r


def rbox(xx, yy, cx, cy, hw, hh, r):
    qx = np.abs(xx - cx) - (hw - r)
    qy = np.abs(yy - cy) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


def value_noise(rng, ny, nx, octaves=(3, 6, 12, 26), weights=(1.0, .55, .3, .17)):
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


# ---- one keyframe ----------------------------------------------------------
YI, XI = np.mgrid[0:NY, 0:NX]
YY = YI / (NY - 1.0)
XX = XI / (NX - 1.0) * AR
SOFT = 2.3 / NY

_rng = np.random.default_rng(SEED)
FIELD = value_noise(_rng, NY, NX)
GRAIN = _rng.normal(0.0, 1.0, (NY, NX))

WALL = np.array([0.660, 0.685, 0.720])
TABLE = np.array([0.690, 0.655, 0.600])
CABINET = np.array([0.880, 0.885, 0.895])
SEAM = np.array([0.470, 0.485, 0.510])
HANDLE = np.array([0.600, 0.215, 0.250])
STEEL = np.array([0.900, 0.905, 0.915])
SLATE = np.array([0.185, 0.200, 0.235])
BLOCK = np.array([0.620, 0.320, 0.230])
PLATE = np.array([0.860, 0.870, 0.885])
MUG = np.array([0.230, 0.290, 0.400])


def arm_masks(ee, opening, ny=NY):
    """Capsule chain: column -> upper arm -> forearm -> wrist -> two fingers."""
    sx, sy = SHOULDER
    dx, dy = ee[0] - sx, ee[1] - sy
    d = float(np.hypot(dx, dy))
    ux, uy = dx / d, dy / d
    nx_, ny_ = -uy, ux
    if ny_ > 0:                               # normal must bow up-screen
        nx_, ny_ = -nx_, -ny_
    arc = 0.155 + 0.105 * (1.0 - min(d / 0.82, 1.0))
    ex = 0.5 * (sx + ee[0]) + nx_ * arc
    ey = 0.5 * (sy + ee[1]) + ny_ * arc

    links = np.minimum.reduce([
        rbox(XX, YY, sx, BASE_Y, 0.098, 0.036, 0.016),
        capsule(XX, YY, (sx, BASE_Y), (sx, sy), 0.049),
        capsule(XX, YY, (sx, sy), (ex, ey), 0.042),
        capsule(XX, YY, (ex, ey), ee, 0.034),
    ])
    joints = np.minimum.reduce([
        capsule(XX, YY, (sx, sy), (sx, sy), 0.031),
        capsule(XX, YY, (ex, ey), (ex, ey), 0.028),
    ])
    fx, fy = ee
    grip = np.minimum.reduce([
        capsule(XX, YY, (fx, fy - 0.012), (fx, fy + 0.020), 0.024),
        capsule(XX, YY, (fx - opening, fy + 0.020), (fx + opening, fy + 0.020),
                0.012),
        capsule(XX, YY, (fx - opening, fy + 0.020), (fx - opening, fy + 0.083),
                0.011),
        capsule(XX, YY, (fx + opening, fy + 0.020), (fx + opening, fy + 0.083),
                0.011),
    ])
    return links, joints, grip


def frame_rgb(k):
    ee, opening, held, _, _ = FRAMES[k]
    rng = np.random.default_rng(SEED + 17 * k)

    # --- background: wall, cabinet, table, back-of-scene props -------------
    up = 0.88 + 0.24 * (YY / HZ)
    dn = 1.16 - 0.44 * np.clip((YY - HZ) / (1.0 - HZ), 0, 1)
    bg = np.where((YY < HZ)[..., None], WALL * up[..., None],
                  TABLE * dn[..., None])
    bg *= 1.0 - 0.34 * cover(np.abs(YY - HZ) - 0.004, SOFT)[..., None]

    # a four-door cabinet behind the table gives the frame scale and depth
    cab = rbox(XX, YY, 0.31, 0.255, 0.275, 0.215, 0.020)
    m_cab = cover(cab, SOFT)[..., None]
    bg = bg * (1 - m_cab) + m_cab * CABINET * (1.06 - 0.20 * YY)[..., None]
    seams = np.minimum.reduce([
        np.maximum(np.abs(XX - 0.31) - 0.0045, cab + 0.012),
        np.maximum(np.abs(YY - 0.255) - 0.0045, cab + 0.012),
    ])
    m_seam = cover(seams, SOFT)[..., None]
    bg = bg * (1 - m_seam) + m_seam * SEAM
    handles = np.minimum.reduce([
        rbox(XX, YY, hx, hy, 0.011, 0.026, 0.005)
        for hx, hy in ((0.262, 0.170), (0.262, 0.340),
                       (0.358, 0.170), (0.358, 0.340))])
    m_hnd = cover(handles, SOFT)[..., None]
    bg = bg * (1 - m_hnd) + m_hnd * HANDLE

    # table lip and two background props
    lip = np.abs(YY - 0.945) - 0.006
    bg *= 1.0 - 0.26 * cover(lip, SOFT)[..., None]
    jar = rbox(XX, YY, 0.085, 0.660, 0.036, 0.066, 0.016)
    m_jar = cover(jar, SOFT)[..., None]
    bg = bg * (1 - m_jar) + m_jar * np.array([0.505, 0.470, 0.415])
    mug = np.minimum(rbox(XX, YY, 0.945, 0.678, 0.042, 0.044, 0.014),
                     rbox(XX, YY, 0.945, 0.638, 0.044, 0.006, 0.003))
    m_mug = cover(mug, SOFT)[..., None]
    bg = bg * (1 - m_mug) + m_mug * MUG

    shade = np.exp(-(((XX - SHOULDER[0]) / 0.20) ** 2
                     + ((YY - 0.905) / 0.035) ** 2))
    bx = ee[0] if held else BLOCK_HOME[0]
    shade += 0.85 * np.exp(-(((XX - bx) / 0.10) ** 2
                             + ((YY - 0.775) / 0.024) ** 2))
    shade += 0.6 * np.exp(-(((XX - TRAY[0]) / 0.13) ** 2
                            + ((YY - 0.788) / 0.020) ** 2))
    bg *= (1.0 - 0.30 * np.clip(shade, 0, 1))[..., None]

    # --- foreground: tray, block, arm --------------------------------------
    lit = (1.30 - 0.50 * YY)[..., None]
    fg = bg.copy()

    tray = rbox(XX, YY, TRAY[0], TRAY[1], 0.115, 0.020, 0.014)
    well = rbox(XX, YY, TRAY[0], TRAY[1] - 0.004, 0.094, 0.011, 0.008)
    m_tray = cover(tray, SOFT)[..., None]
    m_well = cover(well, SOFT)[..., None]
    fg = fg * (1 - m_tray) + m_tray * PLATE * lit
    fg = fg * (1 - m_well) + m_well * PLATE * 0.82 * lit

    bcx, bcy = (ee[0], ee[1] + 0.086) if held else BLOCK_HOME
    if k == 5:
        bcx, bcy = TRAY[0], TRAY[1] - 0.020 - BLOCK_HH
    blk = rbox(XX, YY, bcx, bcy, 0.066, BLOCK_HH, 0.014)
    lid = np.maximum(blk, YY - (bcy - BLOCK_HH * 0.42))
    m_blk, m_lid = cover(blk, SOFT)[..., None], cover(lid, SOFT)[..., None]
    fg = fg * (1 - m_blk) + m_blk * BLOCK * lit
    fg = fg * (1 - m_lid) + m_lid * BLOCK * 1.26 * lit

    links, joints, grip = arm_masks(ee, opening)
    m_lnk = cover(links, SOFT)[..., None]
    m_jnt = cover(joints, SOFT)[..., None]
    m_grp = cover(grip, SOFT)[..., None]
    # a thin bright rim along the upper edge sells the glossy white shell
    rim = cover(links + 0.010, SOFT) - cover(links + 0.001, SOFT)
    fg = fg * (1 - m_lnk) + m_lnk * STEEL * lit
    fg = fg * (1 - m_jnt) + m_jnt * STEEL * 0.72 * lit
    fg += (0.10 * rim * np.clip(1.0 - 2.6 * YY, 0, 1))[..., None]
    fg = fg * (1 - m_grp) + m_grp * SLATE * lit * 1.10

    # --- material, depth of field, vignette --------------------------------
    field = np.roll(FIELD, (3 * k, -5 * k), axis=(0, 1))
    tex = (1.0 + 0.085 * (field - 0.5) * 2.0
           + 0.020 * np.roll(GRAIN, 2 * k, axis=1))[..., None]
    fg, bg = fg * tex, bg * tex

    sharp = np.clip(m_lnk[..., 0] + m_jnt[..., 0] + m_grp[..., 0]
                    + m_blk[..., 0] + 0.85 * m_tray[..., 0], 0, 1)
    sharp = gaussian_filter(sharp, 1.0)[..., None]
    out = blur3(fg, 0.45) * sharp + blur3(bg, 2.2) * (1.0 - sharp)

    key = (1.06 - 0.16 * (XX / AR))[..., None]        # key light from the left
    r2 = ((XX - AR / 2) / AR) ** 2 + (YY - 0.5) ** 2
    out = out * key * (1.0 - 0.42 * r2)[..., None]
    out = np.clip(out * (1.0 + 0.010 * rng.normal(0, 1, (NY, NX)))[..., None],
                  0.0, 1.0)
    return out


# ---- overlays --------------------------------------------------------------
def tile_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#5f5f5f")


def success_badge(ax):
    """Hand-drawn tick inside a ring, in isotropic image-pixel coordinates."""
    cx, cy, r = NX - 0.115 * NY, 0.135 * NY, 0.072 * NY
    ax.add_patch(Circle((cx, cy), r * 1.34, facecolor="white", alpha=0.90,
                        edgecolor="none", zorder=6))
    ax.add_patch(Circle((cx, cy), r * 1.34, facecolor="none", edgecolor=GREEN,
                        linewidth=1.1, zorder=7))
    ax.plot([cx - 0.62 * r, cx - 0.14 * r, cx + 0.72 * r],
            [cy + 0.06 * r, cy + 0.62 * r, cy - 0.62 * r],
            color=GREEN, lw=1.9, solid_capstyle="round",
            solid_joinstyle="miter", zorder=8)


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(1, NCOL, left=LEFT, right=RIGHT, top=TOP,
                          bottom=BOT, wspace=WSP)

    for k, (_, _, _, caption, stamp) in enumerate(FRAMES):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(frame_rgb(k), interpolation="bilinear", aspect="auto",
                  extent=(0, NX, NY, 0), zorder=1)
        ax.set_xlim(0, NX)
        ax.set_ylim(NY, 0)
        tile_frame(ax)
        ax.text(0.052 * NY, 0.075 * NY, "abcdef"[k], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK, zorder=8,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="none", alpha=0.88))
        ax.text(0.5, -0.045, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.6, color=INK)
        ax.text(0.5, -0.215, stamp, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.4, color=MUTED, family="monospace")
        if k == 5:
            success_badge(ax)

    fig.text(LEFT, 0.855, "7-DoF arm, parallel-jaw gripper, 30 Hz wrist "
             "camera; the same episode is shown throughout, sampled at the "
             "six sub-goal boundaries", ha="left", va="top", fontsize=7.2,
             color=SUB)
    savefig(fig, "1123_wm_emb_action_strip.png")


if __name__ == "__main__":
    main()
