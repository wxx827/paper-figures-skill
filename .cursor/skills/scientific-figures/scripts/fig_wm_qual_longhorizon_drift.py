"""Figure 1092 - Long-horizon drift: one clip, ten rollout steps.

The clip is the shared library's ``aerial`` kind - a drone looking down on a
landscape, long lens, flat light, deep depth of field - which reads very
differently from the first-person dash camera of 1492 and makes accumulating
drift easy to follow: the parcel boundaries and building corners are a dense
grid of edges, so softening and warping show up immediately.

A single horizontal film strip of ten predicted frames that get progressively
blurrier, warped and colour-drifted towards blue, the residual against the
ground-truth frame underneath each of them, and a narrow LPIPS bar under that.
The bar values and the residual maps are both read off the arrays that are
drawn, so the strip cannot disagree with itself.  The reference scales hang in
the right margin: a shared colour bar for the residual maps and a track scale
for the bars.

Canvas note: ten landscape tiles in one row fix the aspect at roughly 0.18, the
E8 exception in DESIGN_SYSTEM section 1.2 (single-row comparison grid).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import image_axes, shared_norm
from _wm_scenes import build, octave_noise

H, W = 116, 172
NSTEP = 10
STEP_OF = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
BAR_CMAP = get_cmap("cb_sequential_blue")
TRACK = "#e6e8ea"


def drift(img, k, rng):
    """Accumulated rollout damage after ``k`` steps: warp, blur, cool tint."""
    if k == 0:
        return img
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    a = k / (NSTEP - 1.0)
    d = octave_noise(rng, (H, W), base=4.0, octaves=2) * (0.4 + 3.4 * a)
    out = np.dstack([map_coordinates(img[..., c], [yy + 0.5 * d, xx + d],
                                     order=1, mode="nearest") for c in range(3)])
    out = np.dstack([gaussian_filter(out[..., c], 0.20 + 2.5 * a ** 1.25)
                     for c in range(3)])
    lum = out @ np.array([0.30, 0.59, 0.11])
    out = out * (1 - 0.42 * a) + lum[..., None] * 0.42 * a       # desaturates
    out = out + np.array([-0.070, -0.018, 0.055]) * a            # drifts cool
    out = (out - 0.5) * (1.0 - 0.24 * a) + 0.5                   # flattens
    return np.clip(out + 0.006 * rng.standard_normal((H, W, 3)), 0, 1)


def main():
    set_pub_style(base_font=8)

    gts = [build("aerial", seed=331, shape=(H, W), t=k / (NSTEP - 1.0))["rgb"]
           for k in range(NSTEP)]
    preds, resid = [], []
    for k in range(NSTEP):
        p = drift(gts[k], k, np.random.default_rng(500 + k))
        preds.append(p)
        resid.append(np.abs(p - gts[k]).mean(axis=2))
    rnorm = shared_norm(resid, robust=(0.0, 99.5))
    # a monotone stand-in for LPIPS, read off the residuals that are plotted
    means = np.array([float(r.mean()) for r in resid])
    lpips = 0.04 + 0.48 * means / means.max()
    LMAX = 0.60

    fig = plt.figure(figsize=(11.0, 2.0))
    gs = fig.add_gridspec(3, NSTEP, wspace=0.045, hspace=0.14,
                          height_ratios=[1.0, 1.0, 0.30],
                          left=0.058, right=0.902, top=0.870, bottom=0.110)

    im = None
    for k in range(NSTEP):
        ax = fig.add_subplot(gs[0, k])
        image_axes(ax, color="#9a9a9a", lw=0.5)
        ax.set_aspect("auto")
        ax.imshow(preds[k], interpolation="bilinear", aspect="auto")
        ax.set_title(f"{STEP_OF[k]}", fontsize=7.4, pad=2.6,
                     color="#1a1a1a" if k == 0 else "#555555")
        if k == 0:
            ax.set_ylabel("Prediction", fontsize=6.8, labelpad=3.0)

        axr = fig.add_subplot(gs[1, k])
        image_axes(axr, color="#9a9a9a", lw=0.5)
        axr.set_aspect("auto")
        im = axr.imshow(resid[k], cmap="PuBu_r", norm=rnorm, aspect="auto",
                        interpolation="bilinear")
        if k == 0:
            axr.set_ylabel(r"$|\Delta|$ to GT", fontsize=6.8, labelpad=3.0)

        axb = fig.add_subplot(gs[2, k])
        axb.set_xlim(0, LMAX)
        axb.set_ylim(0, 1)
        axb.set_xticks([])
        axb.set_yticks([])
        for s in axb.spines.values():
            s.set_visible(False)
        axb.add_patch(Rectangle((0.0, 0.22), LMAX, 0.56, facecolor=TRACK,
                                edgecolor="none", zorder=1))
        axb.add_patch(Rectangle((0.0, 0.22), lpips[k], 0.56,
                                facecolor=BAR_CMAP(0.25 + 0.70 * lpips[k] / LMAX),
                                edgecolor="none", zorder=2))
        inside = lpips[k] > 0.60 * LMAX
        axb.text(lpips[k] - 0.018 if inside else lpips[k] + 0.018, 0.5,
                 f"{lpips[k]:.2f}", ha="right" if inside else "left",
                 va="center", fontsize=6.0, zorder=3,
                 color="#ffffff" if inside else "#333333")
        if k == 0:
            axb.set_ylabel("LPIPS", fontsize=6.8, labelpad=3.0, rotation=90,
                           va="center")

    # ---- reference scales in the right margin
    p_pred = fig.axes[0].get_position()
    p_res = fig.axes[1].get_position()
    cax = fig.add_axes([0.912, p_res.y0, 0.009, p_pred.y1 - p_res.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$|\Delta|$", fontsize=6.6, labelpad=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, color="#333333")

    p_bar = fig.axes[2].get_position()          # first bar axes
    sax = fig.add_axes([0.912, p_bar.y0, 0.076, p_bar.height])
    sax.imshow(np.linspace(0, 1, 256)[None, :], cmap=BAR_CMAP, aspect="auto",
               vmin=-0.357, vmax=1.071, extent=(0, LMAX, 0, 1))
    sax.set_yticks([])
    sax.set_xlim(0, LMAX)
    sax.set_xticks([0, 0.3, 0.6])
    sax.tick_params(labelsize=6.0, width=0.6, length=1.8, pad=1.0)
    for s in sax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("#333333")

    savefig(fig, "1092_wm_qual_longhorizon_drift.png")


if __name__ == "__main__":
    main()
