"""Figure 390 - Micro-hardness map across a V-groove weld cross-section.

An indent grid is measured, then contoured; the profile and the box plot both
read the same 285 indents, so the peak, the soft zone and the zone medians
cannot disagree with the map.  Zone membership comes from the signed distance
to the fusion boundary, which is also what shapes the hardness field.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter

from pubstyle import (set_pub_style, savefig, get_cmap, annotate_box,
                      style_axes, COLORS, CATEGORICAL)
from figkit import safe_colorbar, panel_labels_inches

W_MM, H_MM = 18.0, 15.0            # cross-section, mm
NX, NY = 19, 15                    # indent grid
ZONE_NAMES = ["Fusion zone", "HAZ", "Base metal"]
ZONE_COLORS = [CATEGORICAL[0], CATEGORICAL[3], CATEGORICAL[1]]
PROFILE_Y = 4.2                    # depth of the traverse, mm


def rasterize(step=0.05):
    """Signed distance (mm) to the fusion boundary on a fine grid."""
    nx, ny = int(W_MM / step), int(H_MM / step)
    x = (np.arange(nx) + 0.5) * step - W_MM / 2.0
    y = (np.arange(ny) + 0.5) * step
    X, Y = np.meshgrid(x, y)
    rng = np.random.default_rng(4)
    wob = gaussian_filter(rng.standard_normal(ny), 6.0)
    wob /= wob.std() + 1e-9
    half = np.clip(4.3 * (1.0 - (Y / H_MM) ** 1.7) + 0.16 * wob[:, None], 0.25,
                   None)
    fz = np.abs(X) < half
    d = (distance_transform_edt(~fz) - distance_transform_edt(fz)) * step
    return x, y, d


def hv_field(d):
    """Peak next to the fusion line, sub-critical soft zone further out."""
    hv = np.where(d < 0.0, 332.0, 225.0)
    hv = hv + 158.0 * np.exp(-((d - 0.45) / 0.95) ** 2) * (d >= 0.0)
    hv = hv - 36.0 * np.exp(-((d - 2.80) / 1.00) ** 2) * (d >= 0.0)
    return hv


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    xf, yf, df = rasterize()

    gx = np.linspace(-W_MM / 2 + 0.9, W_MM / 2 - 0.9, NX)
    gy = np.linspace(0.9, H_MM - 0.9, NY)
    GX, GY = np.meshgrid(gx, gy)
    d_ind = griddata((np.repeat(yf, xf.size), np.tile(xf, yf.size)),
                     df.ravel(), (GY.ravel(), GX.ravel()),
                     method="linear").reshape(GY.shape)
    rng = np.random.default_rng(11)
    hv = hv_field(d_ind) + rng.normal(0, 6.0, GX.shape)

    # Contour the *measurements*, not the underlying field.
    ix = np.linspace(gx[0], gx[-1], 240)
    iy = np.linspace(gy[0], gy[-1], 200)
    IX, IY = np.meshgrid(ix, iy)
    hv_i = griddata(np.c_[GX.ravel(), GY.ravel()], hv.ravel(), (IX, IY),
                    method="cubic")

    zone = np.where(d_ind < 0.0, 0, np.where(d_ind < 2.5, 1, 2))
    row = int(np.argmin(np.abs(gy - PROFILE_Y)))

    fig = plt.figure(figsize=(7.8, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.92], wspace=0.30, hspace=0.52,
                          left=0.055, right=0.965, top=0.880, bottom=0.115)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_pro = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 1])

    # ---- (a) hardness map -------------------------------------------------
    levels = np.linspace(np.floor(hv.min() / 10) * 10,
                         np.ceil(hv.max() / 10) * 10, 13)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        cf = ax_map.contourf(IX, IY, hv_i, levels=levels,
                             cmap=get_cmap("cb_sequential_warm"), extend="both")
        cf.set_rasterized(True)
    cf.set_edgecolor("face")
    ax_map.contour(IX, IY, hv_i, levels=levels[::3], colors=COLORS["rule"],
                   linewidths=0.4, alpha=0.55)
    ax_map.contour(xf, yf, df, levels=[0.0, 2.5], colors=[COLORS["ink"],
                                                          COLORS["ink"]],
                   linewidths=[1.2, 0.8], linestyles=["-", "--"])
    ax_map.scatter(GX, GY, s=2.6, facecolor="white", edgecolor=COLORS["rule"],
                   linewidth=0.3, zorder=5)
    ax_map.plot([gx[0], gx[-1]], [gy[row]] * 2, color=COLORS["accent"],
                lw=1.1, ls="--", zorder=6)
    for lx, lab in ((gx[0], "A"), (gx[-1], "B")):
        ax_map.text(lx, gy[row] + 0.35, lab, fontsize=7.0, ha="center",
                    va="bottom", color=COLORS["accent"], fontweight="bold")
    ax_map.text(0.0, 0.9, "FZ", fontsize=7.2, ha="center", va="top",
                color=COLORS["ink"], fontweight="bold")
    ax_map.text(5.6, 0.9, "HAZ", fontsize=7.2, ha="center", va="top",
                color=COLORS["ink"], fontweight="bold")
    ax_map.text(-8.4, H_MM - 0.6, "base metal", fontsize=7.2, ha="left",
                va="bottom", color=COLORS["ink"])
    ax_map.set_xlim(-W_MM / 2, W_MM / 2)
    ax_map.set_ylim(H_MM, 0)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("distance from weld centre (mm)", fontsize=8.0)
    ax_map.set_ylabel("depth below cap (mm)", fontsize=8.0)
    ax_map.tick_params(labelsize=7.0, length=2.5)
    for s in ax_map.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["rule"])
    ax_map.set_title(f"{GX.size} indents, HV0.5, 0.9 mm pitch", fontsize=8.5,
                     fontweight="bold", pad=3.5)
    safe_colorbar(fig, cf, ax_map, side="bottom", width=0.10, pad=0.36,
                  shrink=0.90, label="Vickers hardness (HV0.5)", labelsize=6.8,
                  ticks=np.arange(160, 421, 40))

    # ---- (b) traverse -----------------------------------------------------
    ax_pro.plot(gx, hv[row], marker="o", ms=3.2, lw=1.3,
                color=COLORS["rule"], mfc="white", mew=0.8, zorder=4)
    for z, c in enumerate(ZONE_COLORS):
        m = zone[row] == z
        ax_pro.scatter(gx[m], hv[row][m], s=15, color=c, zorder=5,
                       edgecolor="white", linewidth=0.4)
    pk = int(np.argmax(hv[row]))
    sf = int(np.argmin(hv[row]))
    for idx, txt, col, dy, va in (
            (pk, f"HAZ peak {hv[row][pk]:.0f} HV", COLORS["accent"], 9.0,
             "bottom"),
            (sf, f"soft zone {hv[row][sf]:.0f} HV", CATEGORICAL[1], -9.0,
             "top")):
        left = gx[idx] < 0
        ax_pro.annotate(txt, xy=(gx[idx], hv[row][idx]),
                        xytext=(4.0 if left else -4.0, dy),
                        textcoords="offset points", fontsize=6.6, color=col,
                        ha="left" if left else "right", va=va)
    ax_pro.set_xlabel("distance from weld centre (mm)", fontsize=7.5)
    ax_pro.set_ylabel("hardness (HV0.5)", fontsize=7.5)
    ax_pro.set_xlim(gx[0] - 0.4, gx[-1] + 0.4)
    lo, hi = hv[row].min(), hv[row].max()
    ax_pro.set_ylim(lo - 0.30 * (hi - lo), hi + 0.34 * (hi - lo))
    ax_pro.tick_params(labelsize=6.5, length=2.5)
    ax_pro.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_pro)
    ax_pro.set_title(f"Traverse A\u2013B at {gy[row]:.1f} mm depth",
                     fontsize=8.0, fontweight="bold", pad=2.5)

    # ---- (c) zone statistics ---------------------------------------------
    groups = [hv[zone == z] for z in range(3)]
    bp = ax_box.boxplot(groups, vert=True, widths=0.55, patch_artist=True,
                        medianprops=dict(color="white", lw=1.1),
                        flierprops=dict(marker="o", ms=1.8,
                                        markerfacecolor=COLORS["muted"],
                                        markeredgecolor="none"),
                        whiskerprops=dict(lw=0.8, color=COLORS["rule"]),
                        capprops=dict(lw=0.8, color=COLORS["rule"]),
                        boxprops=dict(lw=0.6, edgecolor=COLORS["rule"]))
    for patch, c in zip(bp["boxes"], ZONE_COLORS):
        patch.set_facecolor(c)
    ax_box.set_xticks([1, 2, 3])
    ax_box.set_xticklabels([f"{n}\nn = {g.size}"
                            for n, g in zip(ZONE_NAMES, groups)],
                           fontsize=6.5, linespacing=1.3)
    ax_box.set_ylabel("hardness (HV0.5)", fontsize=7.5)
    ax_box.tick_params(labelsize=6.5, length=2.5)
    ax_box.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_box)
    ax_box.set_title("Hardness by weld zone", fontsize=8.0,
                     fontweight="bold", pad=2.5)
    drop = 100.0 * (1.0 - groups[1].min() / np.median(groups[2]))
    annotate_box(ax_box,
                 f"HAZ peak {groups[1].max():.0f} HV\n"
                 f"soft zone {drop:.0f} % below base",
                 loc="lower left", fontsize=6.4)

    panel_labels_inches(fig, [ax_map, ax_pro, ax_box], dx=-0.34, dy=0.08)

    savefig(fig, "390_mat_hardness_mapping.png")


if __name__ == "__main__":
    main()
