"""Figure 1087 - Qualitative rollout comparison for a video world model.

Rows are methods (ground truth, two baselines, ours), columns are rollout
horizons of the same driving clip.  Frames come from the shared scene library
(``_wm_scenes``) as the ``road`` kind - a wide dash-camera lens under cool
overcast light - and are finished with a depth-of-field pass so the mid-ground
is genuinely sharper than the sky, rather than the whole tile being soft.

Damage is a strictly increasing function of the horizon that is exactly zero at
t0, so the shared initial state claimed in the caption is the one that is
drawn.  Red hairline boxes mark the artefacts a reviewer is meant to notice;
two of them carry a 2x magnifier inset.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig
from figkit import add_zoom_inset, background_card, image_axes, roi_rect
from _wm_scenes import build, octave_noise

H, W = 138, 204                     # tile resolution (pixels)
BOX = "#d1495b"                     # artefact box / inset frame
OURS_EDGE = "#3b5b92"
TILE_EDGE = "#9a9a9a"


def warp(img, rng, amp, sigma=13.0):
    if amp <= 0:
        return img
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dy = amp * octave_noise(rng, (H, W), base=H / sigma, octaves=2)
    dx = amp * octave_noise(rng, (H, W), base=H / sigma, octaves=2)
    return np.dstack([map_coordinates(img[..., c], [yy + dy, xx + dx], order=1,
                                      mode="nearest") for c in range(3)])


def degrade(img, rng, blur=0.0, wamp=0.0, desat=0.0, tint=(0, 0, 0),
            contrast=1.0, grain=0.0, band=0.0, hi_blur=0.0):
    out = warp(img, rng, wamp)
    if blur > 0:
        out = np.dstack([gaussian_filter(out[..., c], blur) for c in range(3)])
    if hi_blur > 0:
        # softening restricted to the skyline: a sub-pixel global blur is
        # invisible once the tile is printed 1.9 in wide, but losing the roof
        # lines is not
        w = np.clip((0.55 - np.mgrid[0:H, 0:W][0] / H) / 0.25, 0, 1)[..., None]
        k = np.dstack([gaussian_filter(out[..., c], hi_blur) for c in range(3)])
        out = out * (1 - w) + k * w
    if desat > 0:
        lum = out @ np.array([0.30, 0.59, 0.11])
        out = out * (1 - desat) + lum[..., None] * desat
    out = (out - 0.5) * contrast + 0.5 + np.array(tint)
    if band > 0:                       # faint colour banding, worst in the sky
        yy = np.mgrid[0:H, 0:W][0].astype(float)
        prof = np.sin(yy / 2.6) * (1.0 - 0.75 * yy / H)
        out = out + band * prof[..., None] * np.array([0.45, 0.75, 1.0])
    if grain > 0:
        out = out + grain * rng.standard_normal(out.shape)
    return np.clip(out, 0, 1)


def scale_kw(kw, p):
    """Scale a row's full-strength damage by ``p``; ``p = 0`` is a clean frame."""
    out = {}
    for key, v in kw.items():
        if key == "contrast":
            out[key] = 1.0 + (v - 1.0) * p
        elif isinstance(v, tuple):
            out[key] = tuple(c * p for c in v)
        else:
            out[key] = v * p
    return out


def smear(img, box, rng, length=11, ghost=0.45):
    """Local temporal-smear artefact: directional blur plus an offset ghost."""
    x0, y0, bw, bh = box
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    m = (np.clip(1 - ((xx - (x0 + bw / 2)) / (0.62 * bw)) ** 2, 0, 1) *
         np.clip(1 - ((yy - (y0 + bh / 2)) / (0.62 * bh)) ** 2, 0, 1))
    m = gaussian_filter(m, 2.2)[..., None]
    k = np.dstack([gaussian_filter(img[..., c], (0.8, length)) for c in range(3)])
    sh = np.dstack([gaussian_filter(np.roll(img[..., c], int(0.34 * bw), axis=1),
                                    (0.5, 2.0)) for c in range(3)])
    out = img * (1 - m) + (k * (1 - ghost) + sh * ghost) * m
    wob = 2.6 * m[..., 0] * gaussian_filter(rng.standard_normal((H, W)), 3.0)
    wob = wob / (wob.std() + 1e-9) * m[..., 0] * 2.2
    out = np.dstack([map_coordinates(out[..., c], [yy + 0.4 * wob, xx + wob],
                                     order=1, mode="nearest") for c in range(3)])
    out = out + m * 0.010 * rng.standard_normal(img.shape)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Damage at the *last* horizon; every column scales it by (step/16)**1.6, so
# the t0 column is a clean frame in every row and the figure agrees with its
# own caption.  The two baselines drift in different cool directions - one
# towards cyan, one towards blue-violet - so their failure modes stay
# distinguishable without reaching for a warm cast.
ROWS = [
    ("Ground truth", dict()),
    ("VideoGPT-L",   dict(blur=1.95, wamp=2.7, desat=0.26,
                          tint=(-0.050, 0.006, 0.028), contrast=0.88,
                          band=0.020)),
    ("DiffWM-B",     dict(blur=1.15, wamp=1.5, desat=0.12,
                          tint=(-0.006, -0.026, 0.048), contrast=0.95,
                          band=0.012)),
    # ~20 % of DiffWM-B everywhere, plus a skyline-only softening so that the
    # row reads as a rollout rather than as one tile pasted five times
    ("Ours",         dict(blur=0.22, wamp=0.30, desat=0.024,
                          tint=(0.001, -0.005, 0.012), contrast=0.990,
                          band=0.0050, hi_blur=1.25)),
]
TIMES = [0, 4, 8, 12, 16]
# (row, col, box in pixels, whether it gets a magnifier)
MARKS = [(1, 3, (24, 72, 56, 38), True),
         (1, 4, (110, 76, 54, 36), False),
         (2, 4, (26, 76, 54, 36), True)]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4092)

    nr, nc = len(ROWS), len(TIMES)
    frames = np.empty((nr, nc), object)
    for j, tt in enumerate(TIMES):
        gt = build("road", seed=717, shape=(H, W), t=tt / 20.0)["rgb"]
        prog = (tt / TIMES[-1]) ** 1.6               # 0 at t0, 1 at the end
        for i, (_, kw) in enumerate(ROWS):
            if not kw:
                frames[i, j] = gt
                continue
            # the only thing left at prog = 0 is a trace of decoder noise
            frames[i, j] = degrade(gt, np.random.default_rng(31 + 7 * i + j),
                                   grain=0.0035, **scale_kw(kw, prog))
    for (i, j, box, _) in MARKS:
        frames[i, j] = smear(frames[i, j], box, rng,
                             length=13 if i == 1 else 8,
                             ghost=0.50 if i == 1 else 0.34)

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(nr, nc, wspace=0.04, hspace=0.10,
                          left=0.062, right=0.995, top=0.885, bottom=0.075)

    axes = np.empty((nr, nc), object)
    for i, (name, _) in enumerate(ROWS):
        ours = name == "Ours"
        for j, tt in enumerate(TIMES):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax, color=OURS_EDGE if ours else TILE_EDGE,
                       lw=0.9 if ours else 0.55)
            ax.set_aspect("auto")
            ax.imshow(frames[i, j], interpolation="bilinear", aspect="auto",
                      zorder=1)
            if i == 0:
                ax.set_title(r"$t_0$" if tt == 0 else rf"$t_0 + {tt}$",
                             fontsize=8.6, pad=3.4,
                             color="#1a1a1a" if tt else "#555555")
            if j == 0:
                ax.set_ylabel(name, fontsize=8.6, labelpad=3.6,
                              fontweight="bold" if ours or i == 0 else "normal",
                              color="#1a1a1a")

    for (i, j, box, zoom) in MARKS:
        ax = axes[i, j]
        if zoom:
            add_zoom_inset(ax, frames[i, j], box, (0.575, 0.035, 0.395, 0.395),
                           color=BOX, box_lw=1.0, border_lw=1.1,
                           interpolation="bilinear", connect=True)
        else:
            roi_rect(ax, box, color=BOX, lw=1.0)

    # ---- pale card behind the "Ours" row
    p0 = axes[3, 0].get_position()
    p1 = axes[3, nc - 1].get_position()
    background_card(fig, (p0.x0 - 0.052, p0.y0 - 0.022,
                          p1.x1 - p0.x0 + 0.057, p0.height + 0.044),
                    facecolor="#eaeff6", zorder=-5, radius=0.006)

    savefig(fig, "1087_wm_qual_rollout_grid.png")


if __name__ == "__main__":
    main()
