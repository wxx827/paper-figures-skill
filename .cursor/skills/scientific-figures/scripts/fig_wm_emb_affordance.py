"""Predicted affordance maps for four household objects, one class each.

Synthetic imagery: no photograph is used anywhere.  The frames come from
``observation`` and the maps from ``affordance`` - swap those two for a frame
loader and a model head, and the rest of the figure is unchanged.

Top row: the observation, rendered from signed-distance primitives (rounded
boxes, capsules, rings), textured with multi-scale value noise.  Bottom row:
the same frame, muted, with a semi-transparent affordance map on top - grasp,
push, pour and open.

Camera recipe (deliberately *not* the warm wide-angle bench of 1528 / 1529):
a cool-white studio over a brushed metal lab bench, camera raised so the
bench line sits high, long lens so the vignette is mild, and a shallow focus
plane.  Depth of field is a blur/sharpen pair rather than blur alone - the
background is smeared hard while the subject gets an unsharp pass, which is
what separates "shot at f/2" from "out of focus".

Every map is built before anything is drawn and they all share one colour
range and one cool ramp, so a bright patch in the third panel means the same
confidence as a bright patch in the first.  Each panel's printed peak is read
back off the array that was drawn.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"

SEED = 1531  # original id; figure number is 1126
FW, FH = 10.4, 6.4
LEFT, RIGHT, TOP, BOT = 0.062, 0.895, 0.845, 0.108
WSP, HSP = 0.045, 0.090
NROW, NCOL = 2, 4

TILE_W = (RIGHT - LEFT) * FW / (NCOL + WSP * (NCOL - 1))
TILE_H = (TOP - BOT) * FH / (NROW + HSP * (NROW - 1))
AR = TILE_W / TILE_H
NY = 190
NX = int(round(NY * AR))

CMAP = plt.get_cmap("cividis")

# --- camera / lighting recipe: cool-white studio over a metal lab bench ----
HZ = 0.345                 # raised camera -> the bench line sits high
BALANCE = np.array([0.955, 0.990, 1.062])       # cool white point
EXPOSURE = 1.05
VIGNETTE = 0.17            # long lens -> mild falloff
BG_SIGMA, FG_SIGMA = 3.0, 0.28
SHARPEN = 0.60

WALL = np.array([0.760, 0.782, 0.812])
TABLE = np.array([0.600, 0.622, 0.652])
CLAY = np.array([0.620, 0.320, 0.230])
STEEL = np.array([0.640, 0.665, 0.700])
PORCELAIN = np.array([0.870, 0.878, 0.890])
WOOD = np.array([0.575, 0.470, 0.360])
HANDLE = np.array([0.600, 0.215, 0.250])
DARK = np.array([0.230, 0.245, 0.280])

OBJECTS = ["Ceramic mug", "Storage crate", "Metal kettle", "Drawer unit"]
CLASSES = ["grasp", "push", "pour", "open"]
NOTES = ["handle, side pinch", "front face", "handle over the spout",
         "upper recessed pull"]
PEAKS = [0.94, 0.88, 0.79, 0.86]        # per-class confidence the head emits


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


def value_noise(rng, ny, nx, octaves=(3, 7, 14, 28),
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


def blur3(a, sigma):
    return np.dstack([gaussian_filter(a[..., c], sigma) for c in range(3)])


YI, XI = np.mgrid[0:NY, 0:NX]
YY = YI / (NY - 1.0)
XX = XI / (NX - 1.0) * AR
SOFT = 2.3 / NY

_rng = np.random.default_rng(SEED)
FIELD = value_noise(_rng, NY, NX)
GRAIN = _rng.normal(0.0, 1.0, (NY, NX))
CX = AR / 2.0


# ---- objects ---------------------------------------------------------------
def object_layers(k):
    """(sdf, colour, gain) stack plus the Gaussian seeds of its affordance."""
    if k == 0:                                        # mug -> grasp
        body = rbox(XX, YY, CX - 0.06, 0.560, 0.150, 0.185, 0.045)
        hand = np.maximum(ring(XX, YY, CX + 0.140, 0.560, 0.105, 0.030),
                          -(XX - (CX + 0.120)))
        rim = rbox(XX, YY, CX - 0.06, 0.382, 0.152, 0.020, 0.014)
        L = [(body, CLAY, 1.0), (hand, CLAY, 0.94), (rim, CLAY, 1.22)]
        seeds = [((CX + 0.246, 0.560), 0.070, 1.00),
                 ((CX - 0.190, 0.545), 0.062, 0.74)]
        base = 0.735
    elif k == 1:                                      # crate -> push
        body = rbox(XX, YY, CX, 0.575, 0.245, 0.170, 0.028)
        lid = rbox(XX, YY, CX, 0.428, 0.255, 0.030, 0.014)
        slat = np.minimum(rbox(XX, YY, CX, 0.545, 0.212, 0.008, 0.004),
                          rbox(XX, YY, CX, 0.640, 0.212, 0.008, 0.004))
        L = [(body, WOOD, 1.0), (lid, WOOD, 1.16), (slat, WOOD, 0.76)]
        seeds = [((CX, 0.560), 0.130, 1.00), ((CX, 0.430), 0.070, 0.44)]
        base = 0.755
    elif k == 2:                                      # kettle -> pour
        body = rbox(XX, YY, CX + 0.02, 0.590, 0.155, 0.170, 0.060)
        spout = capsule(XX, YY, (CX + 0.150, 0.520), (CX + 0.270, 0.420),
                        0.022)
        hand = np.maximum(ring(XX, YY, CX + 0.005, 0.360, 0.135, 0.024),
                          YY - 0.372)
        lid = rbox(XX, YY, CX + 0.02, 0.424, 0.070, 0.020, 0.012)
        L = [(body, STEEL, 1.0), (spout, STEEL, 1.10), (hand, DARK, 1.0),
             (lid, STEEL, 1.16)]
        seeds = [((CX + 0.005, 0.232), 0.072, 1.00),
                 ((CX + 0.256, 0.432), 0.058, 0.60)]
        base = 0.752
    else:                                             # drawer unit -> open
        body = rbox(XX, YY, CX, 0.560, 0.238, 0.215, 0.024)
        d1 = rbox(XX, YY, CX, 0.452, 0.218, 0.082, 0.016)
        d2 = rbox(XX, YY, CX, 0.652, 0.218, 0.082, 0.016)
        h1 = rbox(XX, YY, CX, 0.452, 0.062, 0.017, 0.008)
        h2 = rbox(XX, YY, CX, 0.652, 0.062, 0.017, 0.008)
        L = [(body, PORCELAIN, 0.86), (d1, PORCELAIN, 1.10),
             (d2, PORCELAIN, 1.04), (h1, HANDLE, 1.0), (h2, HANDLE, 0.92)]
        seeds = [((CX, 0.452), 0.078, 1.00), ((CX, 0.652), 0.072, 0.66)]
        base = 0.782
    return L, seeds, base


def unsharp(img, mask, amount=SHARPEN, sigma=0.95):
    """Edge-preserving pass: pull the subject's edges back after the blur."""
    hi = img - blur3(img, sigma)
    return np.clip(img + amount * hi * mask, 0.0, 1.0)


def observation(k):
    rng = np.random.default_rng(SEED + 11 * k)
    up = 0.94 + 0.14 * (YY / HZ)
    dn = 1.08 - 0.26 * np.clip((YY - HZ) / (1.0 - HZ), 0, 1)
    bg = np.where((YY < HZ)[..., None], WALL * up[..., None],
                  TABLE * dn[..., None])
    bg *= 1.0 - 0.22 * cover(np.abs(YY - HZ) - 0.004, SOFT)[..., None]
    # brushed-metal specular streak across the bench
    streak = np.exp(-((YY - 0.560) / 0.085) ** 2) * (YY > HZ)
    bg *= (1.0 + 0.09 * streak)[..., None]

    layers, _, base = object_layers(k)
    shade = np.exp(-(((XX - CX) / 0.30) ** 2 + ((YY - base) / 0.024) ** 2))
    bg *= (1.0 - 0.30 * np.clip(shade, 0, 1))[..., None]

    lit = (1.18 - 0.28 * YY)[..., None]
    fg = bg.copy()
    sharp = np.zeros((NY, NX))
    for sdf, col, gain in layers:
        m = cover(sdf, SOFT)
        fg = fg * (1 - m[..., None]) + m[..., None] * col * gain * lit
        sharp = np.clip(sharp + m, 0, 1)

    field = np.roll(FIELD, (7 * k, -11 * k), axis=(0, 1))
    tex = (1.0 + 0.062 * (field - 0.5) * 2.0
           + 0.012 * np.roll(GRAIN, 4 * k, axis=1))[..., None]
    fg, bg = fg * tex, bg * tex

    m_sharp = gaussian_filter(sharp, 0.7)[..., None]
    out = blur3(fg, FG_SIGMA) * m_sharp + blur3(bg, BG_SIGMA) * (1.0 - m_sharp)
    out = unsharp(out, m_sharp)
    key = (1.03 - 0.07 * (XX / AR))[..., None]
    r2 = ((XX - CX) / AR) ** 2 + (YY - 0.5) ** 2
    out = out * key * (1.0 - VIGNETTE * r2)[..., None] * EXPOSURE * BALANCE
    out = out * (1.0 + 0.009 * rng.normal(0, 1, (NY, NX)))[..., None]
    return np.clip(out, 0.0, 1.0), sharp


def affordance(k, support):
    """Gaussian blobs on the object, gated by the object's own silhouette."""
    _, seeds, _ = object_layers(k)
    heat = np.zeros((NY, NX))
    for (px, py), sig, amp in seeds:
        heat += amp * np.exp(-(((XX - px) ** 2 + (YY - py) ** 2)
                               / (2.0 * sig ** 2)))
    lobe = gaussian_filter(np.clip(support, 0, 1), 2.0)
    heat *= 0.22 + 0.78 * lobe
    heat = gaussian_filter(heat, 1.6)
    return heat * (PEAKS[k] / max(heat.max(), 1e-9))


def cell(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, NX)
    ax.set_ylim(NY, 0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#5f5f5f")


def peak_cross(ax, iy, ix, arm=0.055 * 190):
    for lw, col in ((2.6, "#1a1a1a"), (1.3, "white")):
        ax.plot([ix - arm, ix + arm], [iy, iy], color=col, lw=lw,
                solid_capstyle="butt", zorder=7)
        ax.plot([ix, ix], [iy - arm, iy + arm], color=col, lw=lw,
                solid_capstyle="butt", zorder=7)


def main():
    set_pub_style(base_font=8)

    # every map is computed before the shared colour range is fixed
    obs, sil = zip(*[observation(k) for k in range(NCOL)])
    heat = [affordance(k, sil[k]) for k in range(NCOL)]
    vmax = float(max(h.max() for h in heat))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(NROW, NCOL, left=LEFT, right=RIGHT, top=TOP,
                          bottom=BOT, wspace=WSP, hspace=HSP)

    for k in range(NCOL):
        ax0 = fig.add_subplot(gs[0, k])
        ax0.imshow(obs[k], interpolation="bilinear", aspect="auto",
                   extent=(0, NX, NY, 0))
        cell(ax0)
        ax0.set_title(OBJECTS[k], fontsize=8.6, fontweight="bold", pad=4.0)
        if k == 0:
            ax0.set_ylabel("Observation", fontsize=8.4, fontweight="bold",
                           labelpad=5.0)

        ax1 = fig.add_subplot(gs[1, k])
        grey = obs[k].mean(axis=2, keepdims=True)
        muted = np.clip(0.42 * obs[k] + 0.58 * grey * 1.04, 0, 1)
        ax1.imshow(muted, interpolation="bilinear", aspect="auto",
                   extent=(0, NX, NY, 0))
        rgba = CMAP(norm(heat[k]))
        rgba[..., 3] = np.clip((heat[k] / vmax - 0.10) / 0.55, 0, 1) * 0.86
        ax1.imshow(rgba, interpolation="bilinear", aspect="auto",
                   extent=(0, NX, NY, 0), zorder=3)
        cell(ax1)
        iy, ix = np.unravel_index(int(np.argmax(heat[k])), heat[k].shape)
        peak_cross(ax1, iy, ix)
        ax1.text(0.040, 0.055, CLASSES[k], transform=ax1.transAxes,
                 ha="left", va="top", fontsize=8.2, fontweight="bold",
                 color=INK, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                           edgecolor="none", alpha=0.90))
        ax1.text(0.5, -0.045, f"peak {heat[k].max():.2f}  ·  {NOTES[k]}",
                 transform=ax1.transAxes, ha="center", va="top",
                 fontsize=6.8, color=SUB)
        if k == 0:
            ax1.set_ylabel("Affordance", fontsize=8.4, fontweight="bold",
                           labelpad=5.0)

    cax = fig.add_axes([0.912, BOT + 0.030, 0.017, 0.300])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    ticks = [0.0, 0.5 * vmax, vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.4)
    cb.set_label("affordance confidence (-)", fontsize=7.0, color=INK,
                 labelpad=3.0)
    fig.text(0.912, BOT + 0.352, "shared\nacross all\nfour maps",
             fontsize=6.4, color=MUTED, ha="left", va="bottom",
             linespacing=1.35)

    savefig(fig, "1126_wm_emb_affordance.png")


if __name__ == "__main__":
    main()
