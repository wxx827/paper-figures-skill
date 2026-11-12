"""Figure 1166 - Temporal consistency: three methods, five frames, one clip.

The clip is the shared library's ``shelf`` kind - a frontal close-up of a
bookcase - rendered once at ``W + PAN`` pixels wide and read out through a
sliding window, so the five frames really are five moments of one lateral
camera pan and nothing else changes between them.

Every number on the page comes off the same stack of arrays:

    frames[m][k]                      what the tiles show
    diff[m][k] = |frames[m][k] - shift(frames[m][k-1])|
                                      what the strips under the tiles show
    jitter[m][k] = 100 * diff[m][k].mean()
                                      what the bars on the right show
    printed mean = jitter[m][1:].mean()

Because the pan is a known integer shift, the previous frame is re-aligned
before differencing; what is left is flicker, not content motion.  The red
boxes are placed at the arg-max of the blurred difference map of the cell they
sit on, so a callout can never end up circling empty wall.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import image_axes, shared_norm
from _wm_scenes import build, octave_noise

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# Two greys, one either side of the 0.94 ink threshold, never on it.  CARD
# (0.91) fills the three "no previous frame" cells and is meant to count;
# LAYOUT (0.96) only groups the jitter panels, which this figure - fifteen
# photographs - does not need to lean on for density.
HAIR, CARD, LAYOUT = "#d5d5d5", "#e6e9ec", "#f5f5f5"
FLICK = "#b23a48"                      # the one accent colour on the page
DIFF_CMAP = get_cmap("cb_sequential_blue")
SEED = 1571  # original id; figure number is 1166

H, W, PAN, NF = 120, 152, 24, 5
DS = PAN // (NF - 1)                   # per-frame lateral shift, in pixels
METHODS = [("Per-frame diffusion", 1.00, "#9cc3dd"),
           ("+ warp consistency loss", 0.42, "#5f9dc6"),
           ("Ours: temporal attention", 0.10, "#134a76")]
# The shelf preset is a warm, very shallow close-up; this batch is cool and
# needs legible book spines, so the camera is pulled cold and stopped down.
#
# Legibility here is a *tone* problem, not a chroma one.  Measured on this
# raster: raising ``sat`` 0.64 -> 0.92 moves the warm area 2.8 % -> 12.2 % and
# leaves detail energy at 25.2, i.e. buys nothing; opening the exposure,
# lifting the vignette and stopping the defocus down takes luminance
# 0.271 -> 0.325 and detail 25.2 -> 31.1.  Brightening does push pixels over
# the V > 0.35 gate of the warm predicate, so the chroma comes back down to
# pay for it - the page ends up at 2.1 % warm, below the 2.6 % it started at.
CAM = dict(wb=(0.925, 1.000, 1.090), sat=0.50, contrast=1.16, dof=0.28,
           sharpen=1.55, vignette=0.10, exposure=1.34, kappa=0.030)


def clip():
    """One shelf render, five overlapping crops = a lateral camera pan."""
    sc = build("shelf", seed=SEED, shape=(H, W + PAN), plan=5, shelves=3,
               cam=CAM)
    return [sc["rgb"][:, k * DS:k * DS + W] for k in range(NF)]


def degrade(gt, level, seed):
    """Temporally incoherent damage: exposure flicker, warp, texture reroll.

    The blur term is *deterministic* in ``level`` so that it softens a method
    without contributing to the flicker metric; everything that differs from
    frame to frame is drawn fresh, which is what a per-frame generator does.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    out = []
    for k, img in enumerate(gt):
        if level <= 0:
            out.append(img)
            continue
        base = np.dstack([gaussian_filter(img[..., c], 0.22 + 0.55 * level)
                          for c in range(3)])
        dx = octave_noise(rng, (H, W), base=3.2, octaves=2) * 1.9 * level
        dy = octave_noise(rng, (H, W), base=3.2, octaves=2) * 1.1 * level
        wrp = np.dstack([map_coordinates(base[..., c], [yy + dy, xx + dx],
                                         order=1, mode="nearest")
                         for c in range(3)])
        gain = 1.0 + 0.055 * level * rng.normal()
        tint = np.array([-0.010, 0.003, 0.014]) * level * rng.normal()
        blotch = gaussian_filter(rng.standard_normal((H, W, 3)),
                                 (7.0, 7.0, 0.0))
        blotch = blotch / (blotch.std() + 1e-9) * 0.032 * level
        out.append(np.clip(wrp * gain + tint + blotch, 0, 1))
    return out


def main():
    set_pub_style(base_font=8.5)
    gt = clip()

    frames, diffs, jitter = [], [], []
    for m, (_, level, _) in enumerate(METHODS):
        seq = degrade(gt, level, SEED + 17 * m)
        dk, jk = [None], [0.0]
        for k in range(1, NF):
            # Re-align the previous frame by the known pan before
            # differencing.  A feature at column x of frame k sits at column
            # x + DS of frame k-1, so the valid overlap is [0, W - DS).
            d = np.abs(seq[k][:, :-DS] - seq[k - 1][:, DS:]).mean(axis=2)
            dk.append(d)
            jk.append(100.0 * float(d.mean()))
        frames.append(seq)
        diffs.append(dk)
        jitter.append(jk)
    jitter = np.asarray(jitter)
    dnorm = shared_norm([d for row in diffs for d in row if d is not None],
                        robust=(0.0, 99.4))
    jmax = float(jitter.max()) * 1.58

    # the two cells with the strongest local flicker, over the whole grid
    peaks = []
    for m in range(len(METHODS)):
        for k in range(1, NF):
            sm = gaussian_filter(diffs[m][k], 4.0)
            iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
            peaks.append((float(sm.max()), m, k,
                          int(np.clip(ix, 20, W - 20)),
                          int(np.clip(iy, 17, H - 17))))
    peaks.sort(reverse=True)
    boxes = {(m, k): (x, y) for _, m, k, x, y in peaks[:3]}

    fig = plt.figure(figsize=(10.4, 6.4))
    # two grids, not one: the metric column needs room for its tick labels,
    # and a shared ``wspace`` would have to loosen the image grid to get it
    outer = fig.add_gridspec(3, 5, left=0.088, right=0.795, top=0.845,
                             bottom=0.122, wspace=0.080, hspace=0.255)
    gs_m = fig.add_gridspec(3, 1, left=0.876, right=0.978, top=0.845,
                            bottom=0.122, hspace=0.255)

    im_d = None
    for m, (name, _, col) in enumerate(METHODS):
        ours = m == len(METHODS) - 1
        for k in range(NF):
            sub = outer[m, k].subgridspec(2, 1, height_ratios=[1.0, 0.19],
                                          hspace=0.10)
            axi = fig.add_subplot(sub[0])
            image_axes(axi, color="#3a3a3a" if ours else "#9a9a9a",
                       lw=0.9 if ours else 0.6)
            axi.set_aspect("auto")
            axi.imshow(frames[m][k], interpolation="bilinear", aspect="auto")
            if m == 0:
                axi.set_title(f"$t_{{{k + 1}}}$", fontsize=8.2, pad=3.0,
                              color=INK)
            if k == 0:
                axi.set_ylabel(name.replace(": ", ":\n"), fontsize=7.2,
                               labelpad=4.0, linespacing=1.35,
                               color=INK if ours else SUB,
                               fontweight="bold" if ours else "normal")
            if (m, k) in boxes:
                bx, by = boxes[(m, k)]
                axi.add_patch(Rectangle((bx - 17, by - 14), 34, 28,
                                        facecolor="none", edgecolor=FLICK,
                                        lw=1.15, zorder=6))

            axd = fig.add_subplot(sub[1])
            axd.set_xticks([])
            axd.set_yticks([])
            for s in axd.spines.values():
                s.set_linewidth(0.5)
                s.set_color("#9a9a9a")
            axd.set_xlim(0, W)
            axd.set_ylim(1, 0)
            if diffs[m][k] is None:
                axd.add_patch(Rectangle((0, 0), W, 1, facecolor=CARD,
                                        edgecolor="none"))
                axd.text(0.5 * W, 0.5, "no previous frame", fontsize=6.0,
                         color=MUTED, ha="center", va="center")
            else:
                im_d = axd.imshow(diffs[m][k], cmap=DIFF_CMAP, norm=dnorm,
                                  extent=(0, W - DS, 1, 0), aspect="auto",
                                  interpolation="bilinear")
            if k == 0:
                axd.set_ylabel("$|\\Delta|$", fontsize=6.4, labelpad=3.0,
                               rotation=0, va="center", ha="right",
                               color=MUTED)

        # ---- per-method jitter bars, aligned with the row -----------------
        axm = fig.add_subplot(gs_m[m, 0])
        axm.set_facecolor(LAYOUT)
        ypos = np.arange(1, NF)
        axm.barh(ypos, jitter[m, 1:], height=0.62, color=col,
                 edgecolor="white", linewidth=0.5, zorder=3)
        mu = float(jitter[m, 1:].mean())
        axm.plot([mu, mu], [0.55, NF - 0.38], color="#0b3d63", lw=0.9,
                 ls=(0, (3, 2)), zorder=5)
        for k in range(1, NF):
            axm.text(jitter[m, k] + 0.035 * jmax, k, f"{jitter[m, k]:.2f}",
                     fontsize=6.2, color=SUB, ha="left", va="center",
                     zorder=6,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                               alpha=0.92))
        axm.text(0.975, 0.995, f"mean {mu:.2f}", transform=axm.transAxes,
                 fontsize=7.0, color="#0b3d63", ha="right", va="top",
                 fontweight="bold" if ours else "normal", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#b9c9d8",
                           lw=0.55, alpha=0.94))
        axm.set_xlim(0, jmax)
        axm.set_ylim(NF - 0.32, 0.05)
        axm.set_yticks(ypos)
        axm.set_yticklabels([f"$t_{{{k}}}\\!\\rightarrow\\!t_{{{k + 1}}}$"
                             for k in range(1, NF)], fontsize=6.4)
        axm.tick_params(length=2.0, labelsize=6.4)
        axm.grid(True, axis="x", lw=0.45, color="#e2e2e2")
        axm.set_axisbelow(True)
        if m == 0:
            axm.set_title("temporal jitter\n(mean $|\\Delta|$, % of range)",
                          fontsize=7.4, pad=3.0, color=INK, linespacing=1.35)
        if m == len(METHODS) - 1:
            axm.set_xlabel("jitter (%)", fontsize=7.2)
        else:
            axm.set_xticklabels([])

    # ---- shared colour bar for the twelve difference strips ---------------
    cax = fig.add_axes([0.088, 0.048, 0.290, 0.0130])
    cb = fig.colorbar(im_d, cax=cax, orientation="horizontal")
    cb.set_label("frame-to-frame difference $|\\Delta|$, shared range",
                 fontsize=6.6, labelpad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, color="#333333")

    best, worst = jitter[:, 1:].mean(axis=1).min(), jitter[:, 1:].mean(axis=1).max()

    fig.add_artist(Rectangle((0.436, 0.041), 0.011, 0.021,
                             transform=fig.transFigure, facecolor="none",
                             edgecolor=FLICK, lw=1.15)).set_in_layout(False)
    savefig(fig, "1166_wm_vid_temporal_consistency.png")


if __name__ == "__main__":
    main()
