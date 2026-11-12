"""Figure 1158 - Radiance field training: five snapshots, five error maps, one curve.

There is exactly one reconstruction model, ``recon(n)``: a blur radius, a
noise amplitude and a mix against the scene mean, all three decaying with the
iteration count.  Everything else is measured off it.

    recon(n)                      -> the top row of snapshots
    |recon(n) - GT| (luminance)   -> the bottom row of error maps
    PSNR(recon(n), GT)            -> the curve on the right

The curve is evaluated at 24 iterations with the same function, so the five
marked points are literally on it, and the error maps share one colour scale
taken over all five panels at once.  Every printed dB figure is an f-string
over these arrays.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from _wm_scenes import build

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
HAIR = "#d5d5d5"
CURVE, ACCENT = "#134a76", "#0b3d63"
SEED = 1563  # original id; figure number is 1158

SHAPE = (132, 176)
IT0 = 500.0
SIG_INF, SIG0 = 0.62, 8.0
EPS_INF, EPS0 = 0.012, 0.100
MIX_INF = 0.0025               # the residual bias training never removes
SHOWN = (500, 2000, 10000, 50000, 200000)
ROMAN = ("i", "ii", "iii", "iv", "v")


def label_it(n):
    return f"{n // 1000} k" if n >= 1000 else f"{n}"


GT = build("aerial", seed=SEED, shape=SHAPE, t=0.35,
           cam=dict(wb=(0.965, 1.005, 1.070), exposure=1.02, contrast=1.10,
                    vignette=0.20))["rgb"]
MEAN = GT.mean(axis=(0, 1))
_rng = np.random.default_rng(SEED)
_mono = (gaussian_filter(_rng.standard_normal(SHAPE), 3.4) * 6.0
         + 0.45 * _rng.standard_normal(SHAPE))
NOISE = (0.78 * _mono[..., None]
         + 0.22 * (gaussian_filter(_rng.standard_normal(GT.shape),
                                   (3.4, 3.4, 0)) * 6.0
                   + 0.45 * _rng.standard_normal(GT.shape)))


def recon(n):
    r = IT0 / float(n)
    sig = SIG_INF + SIG0 * r ** 1.05
    eps = EPS_INF + EPS0 * r ** 0.95
    mix = 1.0 - (MIX_INF + 0.62 * r ** 0.90)
    base = np.dstack([gaussian_filter(GT[..., c], sig) for c in range(3)])
    img = MEAN[None, None, :] * (1.0 - mix) + mix * base + eps * NOISE
    return np.clip(img, 0.0, 1.0)


def psnr(a):
    return 10.0 * np.log10(1.0 / max(float(np.mean((a - GT) ** 2)), 1e-12))


def main():
    set_pub_style(base_font=8.5)
    imgs = [recon(n) for n in SHOWN]
    errs = [np.abs(im - GT) @ np.array([0.2126, 0.7152, 0.0722])
            for im in imgs]
    dbs = [psnr(im) for im in imgs]
    vmax = float(np.percentile(np.concatenate([e.ravel() for e in errs]),
                               99.0))

    its = np.unique(np.round(np.geomspace(300, 320000, 24)).astype(int))
    curve = np.array([psnr(recon(n)) for n in its])
    # closure: the tile captions print 1 dp, so the gaps quoted in the titles
    # have to be the difference of those printed values, not of the raw ones
    shown = [round(v, 1) for v in dbs]
    tail = shown[-1] - shown[-2]
    early = shown[2] - shown[0]

    fig = plt.figure(figsize=(11.0, 4.2))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1.62],
                          left=0.032, right=0.982, top=0.788, bottom=0.175,
                          wspace=0.055, hspace=0.255)
    cmap = get_cmap("cb_sequential_blue")

    for j, n in enumerate(SHOWN):
        axt = fig.add_subplot(gs[0, j])
        axt.imshow(imgs[j], interpolation="bilinear")
        axt.set_title(f"({ROMAN[j]})  {label_it(n)} it", fontsize=7.8,
                      fontweight="bold", pad=3.0, color=INK)
        axe = fig.add_subplot(gs[1, j])
        im = axe.imshow(errs[j], cmap=cmap, vmin=0.0, vmax=vmax,
                        interpolation="bilinear")
        axe.set_xlabel(f"PSNR {dbs[j]:.1f} dB\nmean $|\\Delta|$ "
                       f"{errs[j].mean():.3f}", fontsize=7.0, color=SUB,
                       labelpad=2.5, linespacing=1.5)
        for ax in (axt, axe):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#9aa7b2")
                s.set_linewidth(0.7)
        if j == 0:
            axt.set_ylabel("render", fontsize=7.6, labelpad=3.0)
            axe.set_ylabel("$|\\Delta|$", fontsize=7.6, labelpad=3.0)
            add_panel_label(axt, "(a)", x=-0.230, y=1.02)
            add_panel_label(axe, "(b)", x=-0.230, y=1.02)

    cax = fig.add_axes([0.055, 0.082, 0.245, 0.020])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2, color="#333333")
    cb.set_label(f"$|\\Delta|$, one scale for all five (0 to {vmax:.2f})",
                 fontsize=6.8, color=INK, labelpad=2.0)

    # ---- (c) the PSNR curve ------------------------------------------------
    axc = fig.add_subplot(gs[:, 5])
    axc.plot(its, curve, color=CURVE, lw=1.7, zorder=4)
    axc.axhline(dbs[-1], color="#8a8a8a", lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    axc.text(its[0] * 1.06, dbs[-1] - 0.9, f"converged {dbs[-1]:.1f} dB",
             ha="left", va="top", fontsize=6.8, color=SUB, zorder=6)
    axc.plot(SHOWN, dbs, ls="none", marker="o", ms=5.0, mfc="white",
             mec=ACCENT, mew=1.3, zorder=6)
    for j, (n, d) in enumerate(zip(SHOWN, dbs)):
        axc.annotate(f"({ROMAN[j]})", xy=(n, d), xytext=(n * 0.86,
                     d + 0.055 * float(np.ptp(curve))), fontsize=7.0,
                     color=ACCENT, fontweight="bold", ha="right",
                     va="bottom", zorder=7)
    axc.fill_between(its, curve.min() - 2.0, curve, where=its >= SHOWN[-2],
                     color="#eef3f7", lw=0, zorder=1)
    axc.text(3.4e4, curve.min() + 0.5,
             f"the last {SHOWN[-1] // 1000 - SHOWN[-2] // 1000} k iterations\n"
             f"buy {tail:+.1f} dB", fontsize=6.8, color=ACCENT, ha="left",
             va="bottom", linespacing=1.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#b9c9d8",
                       lw=0.6, alpha=0.94))
    axc.set_xscale("log")
    axc.set_xlim(its[0], its[-1])
    axc.set_ylim(curve.min() - 1.4, curve.max() + 2.4)
    axc.set_xlabel("training iterations", fontsize=8.0)
    axc.set_ylabel("PSNR against the held-out view (dB)", fontsize=8.0)
    axc.set_title(f"{early:+.1f} dB before {SHOWN[2] // 1000} k",
                  fontsize=8.4, fontweight="bold", pad=4.0)
    axc.tick_params(labelsize=7.0, length=2.4)
    axc.grid(True, which="major", lw=0.45, color="#ececec")
    axc.set_axisbelow(True)
    add_panel_label(axc, "(c)", x=-0.072, y=1.015)

    savefig(fig, "1158_wm_nr_training_progress.png")


if __name__ == "__main__":
    main()
