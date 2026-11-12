"""Figure 386 - Three-phase distribution: BSE image, segmentation, statistics.

One synthetic field of view rendered three ways.  The back-scattered electron
panel carries atomic-number contrast plus polishing texture; the middle panel
is the phase label map it would segment to; the right column counts what the
segmentation contains.  All three share the same frame, so a feature can be
followed left to right.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label

from pubstyle import set_pub_style, savefig, style_axes, annotate_box, COLORS
from figkit import add_scale_bar, image_axes, panel_labels_inches

UM_PER_PX = 0.16
SHAPE = (420, 420)

FERRITE, AUSTENITE, CARBIDE = 0, 1, 2
PHASE_NAMES = ["Ferrite ($\\alpha$)", "Austenite ($\\gamma$)",
               "Carbide (M$_{23}$C$_6$)"]
PHASE_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227"]
PHASE_GRAY = {FERRITE: 0.40, AUSTENITE: 0.61, CARBIDE: 0.86}


def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def laguerre_grains(shape, n_grains, seed, relax=2, sigma_r=0.30,
                    warp=4.0, warp_sigma=8.5):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.random(n_grains) * nx, rng.random(n_grains) * ny]
    radii = np.exp(sigma_r * rng.standard_normal(n_grains))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = (xx + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    wy = (yy + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    w = (radii * 0.55 * (nx * ny / n_grains) ** 0.5) ** 2
    lab = np.zeros(wx.size, dtype=np.int32)
    for _ in range(relax + 1):
        best = np.full(wx.size, np.inf)
        for k in range(n_grains):
            d = (wx - pts[k, 0]) ** 2 + (wy - pts[k, 1]) ** 2 - w[k]
            m = d < best
            best[m] = d[m]
            lab[m] = k
        cnt = np.maximum(np.bincount(lab, minlength=n_grains), 1).astype(float)
        pts = np.c_[np.bincount(lab, weights=wx, minlength=n_grains) / cnt,
                    np.bincount(lab, weights=wy, minlength=n_grains) / cnt]
    return lab.reshape(shape)


def boundary_mask(lab, width=1):
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return binary_dilation(b, np.ones((width, width), bool)) if width > 1 else b


def build_scene(seed=15):
    """Return (phase map, grain labels, BSE grey image)."""
    rng = np.random.default_rng(seed)
    n_grains = 150
    lab = laguerre_grains(SHAPE, n_grains, seed=seed)

    # Austenite islands follow a banded segregation field, grain by grain.
    band = smooth_noise(rng, SHAPE, 26.0) + 0.55 * np.sin(
        np.mgrid[0:SHAPE[0], 0:SHAPE[1]][0] / 34.0)
    cnt = np.maximum(np.bincount(lab.ravel(), minlength=n_grains), 1)
    gmean = np.bincount(lab.ravel(), weights=band.ravel(),
                        minlength=n_grains) / cnt
    phase = np.where(gmean[lab] > 0.42, AUSTENITE, FERRITE).astype(np.int8)

    # Carbides decorate grain boundaries: seeded blobs on the boundary network.
    bnd = boundary_mask(lab, 3)
    seedmap = (rng.random(SHAPE) < 0.010) & bnd
    carb = gaussian_filter(seedmap.astype(float), 1.7) > 0.035
    carb |= (gaussian_filter((rng.random(SHAPE) < 6e-4).astype(float), 1.2)
             > 0.075)
    phase[carb] = CARBIDE

    # ---- BSE rendering ----------------------------------------------------
    grey = np.zeros(SHAPE)
    for p, g in PHASE_GRAY.items():
        grey[phase == p] = g
    grey += 0.028 * (rng.random(n_grains) - 0.5)[lab]        # channelling
    grey -= 0.13 * boundary_mask(lab, 2)                     # etched grooves
    tex = (0.55 * smooth_noise(rng, SHAPE, 0.9)
           + 0.30 * smooth_noise(rng, SHAPE, 2.6)
           + 0.15 * smooth_noise(rng, SHAPE, 7.0))
    grey *= 1.0 + 0.085 * tex
    scratch = np.sin((np.mgrid[0:SHAPE[0], 0:SHAPE[1]][1] * 0.75
                      + np.mgrid[0:SHAPE[0], 0:SHAPE[1]][0] * 0.28) / 1.0)
    grey += 0.008 * scratch * (smooth_noise(rng, SHAPE, 30.0) > 0.6)
    yg, xg = np.mgrid[0:SHAPE[0], 0:SHAPE[1]] / (SHAPE[0] - 1.0) - 0.5
    grey *= 1.0 - 0.09 * (xg ** 2 + yg ** 2) / 0.5
    grey += 0.012 * rng.standard_normal(SHAPE)
    return phase, lab, np.clip(grey, 0.0, 1.0)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    phase, lab, bse = build_scene()

    frac = np.array([(phase == p).mean() for p in range(3)]) * 100.0
    px_um2 = UM_PER_PX ** 2
    cc, _ = cc_label(phase == CARBIDE)
    carb_d = 2.0 * np.sqrt(np.bincount(cc.ravel())[1:] * px_um2 / np.pi)
    n_carb = carb_d.size
    grey_by_phase = [bse[phase == p] for p in range(3)]
    grey_mean = np.array([g.mean() for g in grey_by_phase])

    cmap_p = ListedColormap(PHASE_COLORS)
    norm_p = BoundaryNorm(np.arange(-0.5, 3.5, 1.0), 3)

    fig = plt.figure(figsize=(10.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.80], wspace=0.13,
                          left=0.030, right=0.985, top=0.880, bottom=0.075)
    ax_bse = fig.add_subplot(gs[0, 0])
    ax_seg = fig.add_subplot(gs[0, 1])
    gs_r = gs[0, 2].subgridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.30)
    ax_pie = fig.add_subplot(gs_r[0])
    ax_bar = fig.add_subplot(gs_r[1])

    # ---- (a) BSE ----------------------------------------------------------
    ax_bse.imshow(bse, cmap="gray", vmin=0.10, vmax=0.98,
                  interpolation="nearest")
    image_axes(ax_bse, color=COLORS["rule"], lw=0.7)
    ax_bse.set_title("Back-scattered electron image, 15 kV",
                     fontsize=8.5, fontweight="bold", pad=3.5)
    add_scale_bar(ax_bse, 15.0 / UM_PER_PX, label="15 \u00b5m",
                  loc=(0.035, 0.045), height=0.020, fontsize=6.5)

    # ---- (b) segmentation -------------------------------------------------
    ax_seg.imshow(phase, cmap=cmap_p, norm=norm_p, interpolation="nearest")
    ov = np.zeros(SHAPE + (4,))
    ov[..., 3] = 0.85 * boundary_mask(lab, 1)
    ax_seg.imshow(ov, interpolation="nearest")
    image_axes(ax_seg, color=COLORS["rule"], lw=0.7)
    ax_seg.set_title("Threshold + watershed phase segmentation",
                     fontsize=8.5, fontweight="bold", pad=3.5)
    add_scale_bar(ax_seg, 15.0 / UM_PER_PX, label="15 \u00b5m",
                  loc=(0.035, 0.045), height=0.020, fontsize=6.5)
    ax_seg.legend(handles=[Patch(facecolor=c, edgecolor="white", linewidth=0.5,
                                 label=n)
                           for c, n in zip(PHASE_COLORS, PHASE_NAMES)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.012), ncol=3,
                  frameon=False, fontsize=7.0, handlelength=1.1,
                  handleheight=1.0, columnspacing=1.3, borderaxespad=0.0)

    # ---- (c) phase fractions ---------------------------------------------
    wedges, _ = ax_pie.pie(frac, colors=PHASE_COLORS, startangle=90,
                           counterclock=False,
                           wedgeprops=dict(width=0.40, edgecolor="white",
                                           linewidth=1.0))
    ax_pie.text(0, 0.06, f"{frac[FERRITE]:.0f}%", ha="center", va="center",
                fontsize=11.0, fontweight="bold", color=PHASE_COLORS[0])
    ax_pie.text(0, -0.20, "ferrite", ha="center", va="center", fontsize=6.8,
                color=COLORS["sub"])
    ang = 90.0
    for p in range(3):
        ang_mid = np.radians(ang - frac[p] * 3.6 / 2.0)
        ang -= frac[p] * 3.6
        r = 1.26
        ax_pie.text(r * np.cos(ang_mid), r * np.sin(ang_mid),
                    f"{frac[p]:.1f}%", ha="center", va="center",
                    fontsize=7.0, color=PHASE_COLORS[p], fontweight="bold")
    ax_pie.set_aspect("equal")
    ax_pie.set_xlim(-1.60, 1.60)
    ax_pie.set_ylim(-1.60, 1.60)
    ax_pie.set_title("Area fraction (point count)", fontsize=8.5,
                     fontweight="bold", pad=5.0)

    edges = np.linspace(0.05, 1.0, 60)
    ax_bar.hist(grey_by_phase, bins=edges, stacked=True, color=PHASE_COLORS,
                edgecolor="none")
    ax_bar.set_yscale("log")
    for p in range(3):
        ax_bar.axvline(grey_mean[p], color=PHASE_COLORS[p], lw=0.8, ls="--")
    ax_bar.set_xlabel("BSE grey level (-)", fontsize=7.5)
    ax_bar.set_ylabel("pixels", fontsize=7.5)
    ax_bar.set_xlim(0.05, 1.0)
    ax_bar.tick_params(labelsize=6.5, length=2.5)
    ax_bar.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_bar)
    ax_bar.set_title("Grey-level histogram by phase", fontsize=8.0,
                     fontweight="bold", pad=2.5)
    ax_bar.set_ylim(top=ax_bar.get_ylim()[1] * 9.0)
    annotate_box(ax_bar,
                 f"mean grey  $\\alpha$ {grey_mean[0]:.2f}   "
                 f"$\\gamma$ {grey_mean[1]:.2f}\n"
                 f"carbide {grey_mean[2]:.2f}   "
                 f"({n_carb} particles)",
                 loc="upper right", fontsize=6.2)

    panel_labels_inches(fig, [ax_bse, ax_seg, ax_pie, ax_bar],
                        dx=-0.22, dy=0.08)

    savefig(fig, "386_mat_phase_distribution.png")


if __name__ == "__main__":
    main()
