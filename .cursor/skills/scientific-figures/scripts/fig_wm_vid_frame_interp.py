"""Figure 1167 - Frame interpolation between two aerial key frames.

Two key frames are rendered from the shared ``aerial`` scene at t = T0 and
t = T1.  Three intermediate frames are *synthesised* by shifting each key frame
along the known global pan and cross-dissolving them; the ground truth for the
same instants is rendered separately.  The scene deliberately does not move
rigidly - the field pans at the nominal rate, buildings at 0.6x, cars at 1.6x -
so a single global displacement cannot align everything and the synthesis
ghosts exactly where a real interpolator ghosts.

Honesty note.  For each tau the signed residual ``d = pred - gt`` is computed
once; the error tile shows ``|d|`` averaged over channels and the PSNR printed
under it is ``10 log10(1 / mean(d**2))`` over the same array.  There is no
second sampling anywhere, so the map and the number cannot disagree.

Canvas note: one row of five landscape tiles fixes the aspect near 0.42, the
E8 exception in DESIGN_SYSTEM section 1.2 (single-row comparison grid).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig
from figkit import image_axes, shared_norm
from _wm_scenes import build

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# below 0.92 relative luminance on purpose: a 0.94 card sits exactly on the
# ink-ratio threshold and flips between "all ink" and "all white"
HAIR, CARD = "#d5d5d5", "#e6e9ec"
DEEP, MID = "#134a76", "#5f9dc6"
SEED = 1572  # original id; figure number is 1167

H, W = 124, 180
T0, T1 = 0.12, 0.36
PAN = 34.0                                   # px per unit t, the aerial preset
DELTA = PAN * (T1 - T0)                      # key-frame separation, in pixels
TAUS = [0.0, 0.25, 0.50, 0.75, 1.0]
CAM = dict(wb=(0.972, 1.000, 1.042), sat=0.84, contrast=1.12, exposure=1.04)


def render(tau):
    return build("aerial", seed=SEED, shape=(H, W), t=T0 + (T1 - T0) * tau,
                 cam=CAM, plan=11)["rgb"]


def shift(img, dy):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return np.dstack([map_coordinates(img[..., c], [yy + dy, xx], order=1,
                                      mode="nearest") for c in range(3)])


def synth(k0, k1, tau):
    """Linear-motion interpolation: shift both key frames, cross-dissolve."""
    a = shift(k0, tau * DELTA)
    b = shift(k1, -(1.0 - tau) * DELTA)
    out = (1.0 - tau) * a + tau * b
    # synthesis networks are never perfectly sharp at the blend seam
    out = np.dstack([gaussian_filter(out[..., c], 0.30 + 0.55 *
                                     np.sin(np.pi * tau)) for c in range(3)])
    return np.clip(out, 0, 1)


def main():
    set_pub_style(base_font=8.5)
    gts = [render(t) for t in TAUS]
    preds, errs, psnr = [], [], []
    for i, tau in enumerate(TAUS):
        if tau in (0.0, 1.0):
            preds.append(gts[i])
            errs.append(None)
            psnr.append(None)
            continue
        p = synth(gts[0], gts[-1], tau)
        d = p - gts[i]                       # one residual array per tau ...
        preds.append(p)
        errs.append(np.abs(d).mean(axis=2))  # ... the map ...
        psnr.append(10.0 * np.log10(1.0 / float((d ** 2).mean())))  # ... and dB
    enorm = shared_norm([e for e in errs if e is not None], robust=(0.0, 99.5))
    worst = int(np.argmin([p if p is not None else 1e9 for p in psnr]))

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.26, 1.0, 1.0],
                          left=0.045, right=0.958, top=0.845, bottom=0.192,
                          wspace=0.060, hspace=0.165)

    # ---- the time axis -----------------------------------------------------
    axt = fig.add_subplot(gs[0, :])
    axt.plot([0.0, 1.0], [0.0, 0.0], color="#9aa7b2", lw=1.1, zorder=2)
    for i, tau in enumerate(TAUS):
        key = tau in (0.0, 1.0)
        axt.plot([tau], [0.0], marker="s" if key else "D",
                 ms=6.5 if key else 5.4, mfc=DEEP if key else "white",
                 mec=DEEP, mew=1.1, zorder=4)
        axt.text(tau, 0.62, f"$\\tau$ = {tau:.2f}", ha="center", va="bottom",
                 fontsize=7.2, color=INK if key else SUB,
                 fontweight="bold" if key else "normal")
        axt.text(tau, -0.72, "key frame" if key else "synthesised",
                 ha="center", va="top", fontsize=6.6,
                 color=DEEP if key else MUTED)
    axt.set_xlim(-0.125, 1.125)
    axt.set_ylim(-1.9, 1.9)
    axt.set_xticks([])
    axt.set_yticks([])
    for s in axt.spines.values():
        s.set_visible(False)
    axt.text(-0.118, 0.0, "$t$", ha="left", va="center", fontsize=8.4,
             color=SUB)

    # ---- frames and error maps ---------------------------------------------
    im_e = None
    for i, tau in enumerate(TAUS):
        key = tau in (0.0, 1.0)
        axi = fig.add_subplot(gs[1, i])
        image_axes(axi, color=DEEP if key else "#9a9a9a",
                   lw=1.3 if key else 0.6)
        axi.set_aspect("auto")
        axi.imshow(preds[i], interpolation="bilinear", aspect="auto")
        if i == 0:
            axi.set_ylabel("frame", fontsize=7.6, labelpad=4.0, color=INK)

        axe = fig.add_subplot(gs[2, i])
        axe.set_aspect("auto")
        if key:
            image_axes(axe, color=HAIR, lw=0.6)
            axe.set_aspect("auto")
            axe.add_patch(Rectangle((0, 0), 1, 1, facecolor=CARD,
                                    edgecolor="none"))
            axe.text(0.5, 0.5, "input, not\npredicted", ha="center",
                     va="center", fontsize=7.0, color=MUTED, linespacing=1.5)
            axe.set_xlim(0, 1)
            axe.set_ylim(0, 1)
            axe.set_xticks([])
            axe.set_yticks([])
        else:
            image_axes(axe, color="#9a9a9a", lw=0.6)
            axe.set_aspect("auto")
            im_e = axe.imshow(errs[i], cmap="PuBu", norm=enorm, aspect="auto",
                              interpolation="bilinear")
            axe.set_xlabel(f"PSNR {psnr[i]:.2f} dB", fontsize=7.6,
                           labelpad=2.6,
                           color=INK if i == worst else SUB,
                           fontweight="bold" if i == worst else "normal")
            if i == worst:
                sm = gaussian_filter(errs[i], 3.0)
                iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
                ix = float(np.clip(ix, 14, W - 14))
                iy = float(np.clip(iy, 12, H - 12))
                axe.add_patch(Ellipse((ix, iy), 34, 26, facecolor="none",
                                      edgecolor=DEEP, lw=0.9,
                                      ls=(0, (3, 2)), zorder=6))
                axe.annotate("peak residual on the rigid structures:\n"
                             "buildings pan at 0.6x, cars at 1.6x,\nthe "
                             "interpolator assumes one rate",
                             xy=(ix, iy + 14),
                             xytext=(0.030, 0.045), textcoords="axes fraction",
                             fontsize=6.2, color=DEEP, ha="left", va="bottom",
                             linespacing=1.45, zorder=7,
                             bbox=dict(boxstyle="round,pad=0.24", fc="white",
                                       ec="#b9c9d8", lw=0.55, alpha=0.94),
                             arrowprops=dict(arrowstyle="-", lw=0.7,
                                             color=DEEP, shrinkA=3,
                                             shrinkB=2))
        if i == 0:
            axe.set_ylabel("$|$error$|$", fontsize=7.6, labelpad=4.0,
                           color=INK)

    # ---- one shared colour bar under the three predicted columns ----------
    p_lo = fig.axes[1 + 2 * 1 + 1].get_position()      # error tile of tau=0.25
    p_hi = fig.axes[1 + 2 * 3 + 1].get_position()      # error tile of tau=0.75
    cax = fig.add_axes([p_lo.x0, 0.108, p_hi.x1 - p_lo.x0, 0.0165])
    cb = fig.colorbar(im_e, cax=cax, orientation="horizontal")
    cb.set_label("absolute error vs. the rendered ground truth, one shared "
                 "range", fontsize=6.8, labelpad=1.8)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color="#333333")

    good = [p for p in psnr if p is not None]
    savefig(fig, "1167_wm_vid_frame_interp.png")


if __name__ == "__main__":
    main()
