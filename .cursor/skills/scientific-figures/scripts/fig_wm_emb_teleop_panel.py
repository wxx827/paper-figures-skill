"""Teleoperation console: two camera feeds over the live control traces.

Synthetic imagery and synthetic telemetry: no photograph is used anywhere.
The feeds come from ``front_camera`` / ``wrist_camera`` and every trace from
``telemetry`` - swap those three for real sources and the rest of the figure
is unchanged.

Scene recipe C, deliberately unlike the warm wide-angle bench of 1528 / 1529
and the cool metal studio of 1531: a domestic room with a warm wooden floor,
shot by two very different cameras at the same instant.  The front camera is
a normal lens at chest height; the wrist camera is a short-throw fisheye, so
its frame is built by evaluating the same signed-distance scene on a
barrel-distorted coordinate grid, then over-exposed and heavily vignetted the
way a close-range wrist module actually behaves.

Both feeds get the blur/sharpen depth-of-field pair rather than blur alone.

Every number in the status bar and in the annotations is read back from the
array that is plotted: the frame rate from the camera timestamps, the control
rate from the trace timestamps, the round-trip latency from the latency
samples, the aperture from the gripper trace and the peak force from the
force trace.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_palette, style_axes

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
BLUE = "#3b5b92"
ACCENT = "#b23a48"
GREEN = "#2e7d5b"
UI_BG = "#f0f0f0"
UI_BAR = "#e2e5e9"

SEED = 1532  # original id; figure number is 1127
FW, FH = 10.4, 6.4
LEFT, RIGHT, TOP, BOT = 0.068, 0.955, 0.858, 0.095
WSP, HSP = 0.130, 0.230
WR, HR = [1.45, 1.0], [1.0, 0.45]


def _tile(ratio_w, ratio_h):
    tw = (RIGHT - LEFT) * FW
    th = (TOP - BOT) * FH
    u = tw / (sum(WR) * (1.0 + WSP * (len(WR) - 1) / len(WR)))
    v = th / (sum(HR) * (1.0 + HSP * (len(HR) - 1) / len(HR)))
    return u * ratio_w, v * ratio_h


MAIN_W, MAIN_H = _tile(WR[0], HR[0])
WRIST_W, WRIST_H = _tile(WR[1], HR[0])
AR_MAIN, AR_WRIST = MAIN_W / MAIN_H, WRIST_W / WRIST_H

# --- camera recipe C: warm domestic interior --------------------------------
BALANCE = np.array([1.055, 1.005, 0.938])
FLOOR = np.array([0.560, 0.430, 0.305])
RUG = np.array([0.470, 0.470, 0.455])
WALL = np.array([0.790, 0.775, 0.742])
SHELF = np.array([0.520, 0.400, 0.288])
STEEL = np.array([0.880, 0.882, 0.884])
SLATE = np.array([0.195, 0.205, 0.232])
CLAY = np.array([0.610, 0.315, 0.228])
LEAF = np.array([0.240, 0.420, 0.270])
CERAMIC = np.array([0.855, 0.860, 0.868])


# ---- primitives ------------------------------------------------------------
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
    qx, qy = np.abs(px) - (hw - r), np.abs(py) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def ring(xx, yy, cx, cy, r, w):
    return np.abs(np.hypot(xx - cx, yy - cy) - r) - w


def cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


def value_noise(rng, ny, nx, octaves=(3, 7, 15, 30),
                weights=(1.0, .55, .30, .17)):
    out = np.zeros((ny, nx))
    for o, wt in zip(octaves, weights):
        g = rng.random((o, o))
        ty, tx = np.linspace(0, o - 1, ny), np.linspace(0, o - 1, nx)
        iy0, ix0 = np.floor(ty).astype(int), np.floor(tx).astype(int)
        iy1, ix1 = np.minimum(iy0 + 1, o - 1), np.minimum(ix0 + 1, o - 1)
        fy, fx = (ty - iy0)[:, None], (tx - ix0)[None, :]
        top = g[iy0][:, ix0] * (1 - fx) + g[iy0][:, ix1] * fx
        bot = g[iy1][:, ix0] * (1 - fx) + g[iy1][:, ix1] * fx
        out += wt * (top * (1 - fy) + bot * fy)
    return (out - out.min()) / max(np.ptp(out), 1e-9)


def blur3(a, s):
    return np.dstack([gaussian_filter(a[..., c], s) for c in range(3)])


def unsharp(img, mask, amount=0.62, sigma=0.95):
    return np.clip(img + amount * (img - blur3(img, sigma)) * mask, 0.0, 1.0)


def grids(ny, ar, barrel=0.0):
    """Pixel grid, optionally warped by a barrel term for the fisheye feed."""
    nx = int(round(ny * ar))
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * ar
    if barrel:
        cx, cy = ar / 2.0, 0.5
        dx, dy = xx - cx, yy - cy
        r2 = (dx / ar) ** 2 + dy ** 2
        f = 1.0 + barrel * r2
        xx, yy = cx + dx * f, cy + dy * f
    return nx, xx, yy


# ---- the two feeds ---------------------------------------------------------
def front_camera(ny=210):
    """Chest-height normal lens looking across the room at the arm."""
    ar = AR_MAIN
    nx, xx, yy = grids(ny, ar)
    soft, rng = 2.3 / ny, np.random.default_rng(SEED)
    hz = 0.470

    up = 0.94 + 0.12 * (yy / hz)
    dn = 1.12 - 0.34 * np.clip((yy - hz) / (1.0 - hz), 0, 1)
    bg = np.where((yy < hz)[..., None], WALL * up[..., None],
                  FLOOR * dn[..., None])
    for gx in np.arange(0.04, ar, 0.126):          # floor boards
        bg *= (1.0 - 0.10 * cover(np.abs(xx - gx - 0.10 * (yy - hz)) - 0.004,
                                  soft) * (yy > hz))[..., None]
    bg *= 1.0 - 0.26 * cover(np.abs(yy - hz) - 0.004, soft)[..., None]

    rug = rbox(xx, yy, ar * 0.46, 0.880, 0.62, 0.105, 0.03)
    m = cover(rug, soft)[..., None]
    bg = bg * (1 - m) + m * RUG

    shelf = np.minimum.reduce([
        rbox(xx, yy, 0.30, 0.300, 0.245, 0.190, 0.014),
        rbox(xx, yy, 0.30, 0.300, 0.255, 0.010, 0.005)])
    m = cover(shelf, soft)[..., None]
    bg = bg * (1 - m) + m * SHELF
    plant = np.minimum(capsule(xx, yy, (1.62, 0.560), (1.62, 0.400), 0.055),
                       ring(xx, yy, 1.62, 0.318, 0.086, 0.052))
    m = cover(plant, soft)[..., None]
    bg = bg * (1 - m) + m * np.where(yy[..., None] > 0.400, CLAY, LEAF)

    # table, mug and the arm reaching for it
    tbl = np.minimum.reduce([
        rbox(xx, yy, 0.90, 0.658, 0.335, 0.024, 0.010),
        rbox(xx, yy, 0.90, 0.700, 0.300, 0.020, 0.006),
        capsule(xx, yy, (0.625, 0.700), (0.615, 0.880), 0.018),
        capsule(xx, yy, (1.175, 0.700), (1.185, 0.880), 0.018)])
    mug = np.minimum(rbox(xx, yy, 0.80, 0.590, 0.048, 0.050, 0.014),
                     ring(xx, yy, 0.872, 0.596, 0.030, 0.011))
    ee = (0.80, 0.470)
    links = np.minimum.reduce([
        rbox(xx, yy, 1.52, 0.700, 0.090, 0.030, 0.014),
        capsule(xx, yy, (1.52, 0.690), (1.52, 0.300), 0.044),
        capsule(xx, yy, (1.52, 0.300), (1.10, 0.190), 0.038),
        capsule(xx, yy, (1.10, 0.190), ee, 0.030)])
    grip = np.minimum.reduce([
        capsule(xx, yy, ee, (ee[0], ee[1] + 0.022), 0.021),
        capsule(xx, yy, (ee[0] - 0.058, ee[1] + 0.022),
                (ee[0] + 0.058, ee[1] + 0.022), 0.011),
        capsule(xx, yy, (ee[0] - 0.058, ee[1] + 0.022),
                (ee[0] - 0.058, ee[1] + 0.082), 0.010),
        capsule(xx, yy, (ee[0] + 0.058, ee[1] + 0.022),
                (ee[0] + 0.058, ee[1] + 0.082), 0.010)])

    shade = np.exp(-(((xx - 0.90) / 0.34) ** 2 + ((yy - 0.905) / 0.030) ** 2))
    shade += np.exp(-(((xx - 1.52) / 0.18) ** 2 + ((yy - 0.740) / 0.026) ** 2))
    bg *= (1.0 - 0.30 * np.clip(shade, 0, 1))[..., None]

    lit = (1.22 - 0.34 * yy)[..., None]
    fg, sharp = bg.copy(), np.zeros((ny, nx))
    for sdf, col, gain in ((tbl, SHELF, 1.14), (mug, CERAMIC, 1.06),
                           (links, STEEL, 1.0), (grip, SLATE, 1.10)):
        m = cover(sdf, soft)
        fg = fg * (1 - m[..., None]) + m[..., None] * col * gain * lit
        sharp = np.clip(sharp + m, 0, 1)

    field = value_noise(rng, ny, nx)
    tex = (1.0 + 0.070 * (field - 0.5) * 2.0
           + 0.013 * rng.normal(0, 1, (ny, nx)))[..., None]
    fg, bg = fg * tex, bg * tex
    ms = gaussian_filter(sharp, 0.8)[..., None]
    out = unsharp(blur3(fg, 0.30) * ms + blur3(bg, 2.6) * (1.0 - ms), ms)
    r2 = ((xx - ar / 2) / ar) ** 2 + (yy - 0.5) ** 2
    return np.clip(out * BALANCE * (1.0 - 0.34 * r2)[..., None], 0, 1)


def wrist_camera(ny=210):
    """Fisheye module on the wrist: barrel grid, hot exposure, hard vignette."""
    ar = AR_WRIST
    nx, xx, yy = grids(ny, ar, barrel=-0.55)
    soft, rng = 2.3 / ny, np.random.default_rng(SEED + 7)
    cx = ar / 2.0

    bg = np.where((yy < 0.300)[..., None], WALL * 1.02,
                  FLOOR * (1.10 - 0.26 * yy)[..., None])
    tbl = rbox(xx, yy, cx, 0.640, 0.62, 0.055, 0.016)
    m = cover(tbl, soft)[..., None]
    bg = bg * (1 - m) + m * SHELF * 1.16

    mug = np.minimum(rbox(xx, yy, cx - 0.010, 0.510, 0.135, 0.140, 0.038),
                     ring(xx, yy, cx + 0.205, 0.520, 0.084, 0.030))
    fingers = np.minimum(
        rbox(xx, yy, cx - 0.235, 0.560, 0.058, 0.290, 0.026),
        rbox(xx, yy, cx + 0.235, 0.560, 0.058, 0.290, 0.026))
    palm = rbox(xx, yy, cx, 0.905, 0.330, 0.075, 0.026)

    shade = np.exp(-(((xx - cx) / 0.20) ** 2 + ((yy - 0.660) / 0.026) ** 2))
    bg *= (1.0 - 0.30 * np.clip(shade, 0, 1))[..., None]

    lit = (1.20 - 0.26 * yy)[..., None]
    fg, sharp = bg.copy(), np.zeros((ny, nx))
    for sdf, col, gain in ((mug, CERAMIC, 0.94), (fingers, SLATE, 1.18),
                           (palm, SLATE, 0.92)):
        m = cover(sdf, soft)
        fg = fg * (1 - m[..., None]) + m[..., None] * col * gain * lit
        sharp = np.clip(sharp + m, 0, 1)

    field = value_noise(rng, ny, nx)
    tex = (1.0 + 0.075 * (field - 0.5) * 2.0
           + 0.020 * rng.normal(0, 1, (ny, nx)))[..., None]
    fg, bg = fg * tex, bg * tex
    ms = gaussian_filter(sharp, 0.8)[..., None]
    out = unsharp(blur3(fg, 0.26) * ms + blur3(bg, 1.9) * (1.0 - ms), ms, 0.70)
    r2 = ((xx - cx) / ar) ** 2 + (yy - 0.5) ** 2
    out = out * 1.04 * BALANCE * (1.0 - 0.95 * r2)[..., None]
    return np.clip(out, 0, 1)


# ---- telemetry -------------------------------------------------------------
def telemetry():
    rng = np.random.default_rng(SEED + 3)
    ctrl_t = np.arange(0.0, 6.0 + 1e-9, 0.01)              # control loop
    cam_t = np.arange(0.0, 6.0 + 1e-9, 1.0 / 30.0)         # camera stream
    n = ctrl_t.size

    joints = np.zeros((7, n))
    for j in range(7):
        walk = gaussian_filter(rng.normal(0, 1, n), 55.0)
        walk = (walk - walk[0]) / max(np.ptp(walk), 1e-9)
        amp = 42.0 - 4.4 * j
        joints[j] = amp * walk + (18.0 - 6.0 * j)

    grasp_i = int(0.56 * n)
    aperture = np.full(n, 84.0)
    close = np.clip((np.arange(n) - grasp_i) / (0.32 * n / 4.0), 0.0, 1.0)
    aperture = 84.0 - 62.0 * close + 0.35 * rng.normal(0, 1, n)
    force = 11.5 * close ** 1.7 * (1.0 + 0.10 * np.sin(6.0 * ctrl_t))
    force = np.clip(force + 0.16 * rng.normal(0, 1, n), 0.0, None)
    latency = 21.0 + 4.2 * rng.standard_gamma(1.6, cam_t.size)
    return ctrl_t, cam_t, joints, aperture, force, latency, ctrl_t[grasp_i]


def feed_frame(ax, img, tag, stamp):
    ax.imshow(img, interpolation="bilinear", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#4a4a4a")
    ny, nx = img.shape[:2]
    ax.text(0.024, 0.045, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.4, fontweight="bold", color="white", zorder=6,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#1a1a1acc",
                      edgecolor="none"))
    ax.text(0.976, 0.045, stamp, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color="white", family="monospace", zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#1a1a1acc",
                      edgecolor="none"))
    return nx, ny


def main():
    set_pub_style(base_font=8)
    ctrl_t, cam_t, joints, aperture, force, latency, t_grasp = telemetry()

    fps = 1.0 / float(np.mean(np.diff(cam_t)))
    ctrl_hz = 1.0 / float(np.mean(np.diff(ctrl_t)))
    lat_mean = float(np.mean(latency))
    lat_p95 = float(np.percentile(latency, 95))
    span = float(ctrl_t[-1] - ctrl_t[0])

    fig = plt.figure(figsize=(FW, FH))
    card = FancyBboxPatch((0.010, 0.010), 0.980, 0.980,
                          boxstyle="round,pad=0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor=UI_BG,
                          edgecolor="#cfd3d8", linewidth=0.8, zorder=-5)
    fig.add_artist(card)                                   # 12.2
    card.set_in_layout(False)
    bar = FancyBboxPatch((0.022, 0.906), 0.956, 0.066,
                         boxstyle="round,pad=0,rounding_size=0.010",
                         transform=fig.transFigure, facecolor=UI_BAR,
                         edgecolor="#c3c9d0", linewidth=0.8, zorder=-4)
    fig.add_artist(bar)
    bar.set_in_layout(False)

    fig.add_artist(Circle((0.560, 0.939), 0.0052, transform=fig.transFigure,
                          facecolor=ACCENT, edgecolor="none", zorder=3))

    gs = fig.add_gridspec(2, 2, width_ratios=WR, height_ratios=HR, left=LEFT,
                          right=RIGHT, top=TOP, bottom=BOT, wspace=WSP,
                          hspace=HSP)

    # ---- (a) front camera --------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    nx, ny = feed_frame(ax0, front_camera(), "FRONT CAM  ·  1280x720",
                        f"t = {span:5.2f} s")
    ax0.plot([0.5 * nx], [0.62 * ny], marker="+", ms=11, mew=1.2,
             color="#ffffffcc", zorder=6)

    # ---- (b) wrist camera --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    nx1, ny1 = feed_frame(ax1, wrist_camera(), "WRIST CAM  ·  fisheye",
                          f"{fps:.0f} fps")
    for r in (0.10, 0.16):
        ax1.add_patch(Circle((0.5 * nx1, 0.5 * ny1), r * ny1, fill=False,
                             edgecolor="#ffffff99", linewidth=0.8, zorder=6))
    ax1.plot([0.5 * nx1], [0.5 * ny1], marker="+", ms=9, mew=1.1,
             color="#ffffffcc", zorder=6)

    # ---- (c) joint angles --------------------------------------------------
    axj = fig.add_subplot(gs[1, 0])
    cols = get_palette("cb_sequential_blue", 8)[1:]
    for j in range(joints.shape[0]):
        axj.plot(ctrl_t, joints[j], color=cols[j], lw=1.0, zorder=3,
                 label=f"J{j + 1}")
    axj.axvline(t_grasp, color=ACCENT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    axj.text(t_grasp + 0.06, joints.max(), "grasp", fontsize=6.6,
             color=ACCENT, va="top", ha="left")
    axj.set_xlim(ctrl_t[0], ctrl_t[-1])
    axj.set_ylim(joints.min() - 6, joints.max() + 14)
    axj.set_xlabel("time (s)", fontsize=7.4, labelpad=1.5)
    axj.set_ylabel("joint angle (deg)", fontsize=7.4, labelpad=2.0)
    axj.tick_params(labelsize=6.8, length=2.4)
    axj.set_title(f"Commanded joint trajectory  ·  {joints.shape[0]} axes  "
                  f"·  span {joints.min():.0f} to {joints.max():.0f} deg",
                  fontsize=8.0, fontweight="bold", loc="left", pad=3.5)
    axj.legend(ncol=7, fontsize=6.6, loc="upper left",
               bbox_to_anchor=(0.0, 1.02), handlelength=1.1,
               columnspacing=0.9, handletextpad=0.4, frameon=False)
    style_axes(axj)

    # ---- (d) gripper aperture and contact force ----------------------------
    axg = fig.add_subplot(gs[1, 1])
    axg.plot(ctrl_t, aperture, color=BLUE, lw=1.3, zorder=4)
    axg.set_xlim(ctrl_t[0], ctrl_t[-1])
    axg.set_ylim(-6, max(aperture) * 1.22)
    axg.set_xlabel("time (s)", fontsize=7.4, labelpad=1.5)
    axg.set_ylabel("aperture (mm)", fontsize=7.4, color=BLUE, labelpad=2.0)
    axg.tick_params(labelsize=6.8, length=2.4)
    axg.tick_params(axis="y", colors=BLUE)

    axf = axg.twinx()
    axf.fill_between(ctrl_t, 0, force, color=ACCENT, alpha=0.20, lw=0,
                     zorder=2)
    axf.plot(ctrl_t, force, color=ACCENT, lw=1.1, zorder=3)
    axf.set_ylim(0, max(force) * 1.42)
    axf.set_ylabel("contact force (N)", fontsize=7.4, color=ACCENT,
                   labelpad=2.0)
    axf.tick_params(labelsize=6.8, length=2.4, colors=ACCENT)
    axf.spines["top"].set_visible(False)
    axf.spines["right"].set_color(ACCENT)
    axf.spines["right"].set_linewidth(0.7)
    axg.axvline(t_grasp, color="#8a8a8a", lw=0.9, ls=(0, (3.0, 2.0)),
                zorder=1)
    axg.set_title(f"Gripper  ·  {aperture.max():.0f} to {aperture.min():.0f} "
                  f"mm  ·  peak {force.max():.1f} N", fontsize=8.0,
                  fontweight="bold", loc="left", pad=3.5)
    style_axes(axg)
    axg.spines["right"].set_visible(True)

    savefig(fig, "1127_wm_emb_teleop_panel.png")


if __name__ == "__main__":
    main()
