"""Wrapped C-band interferogram over a subsiding basin and an inflating vent.

Main panel: the wrapped differential phase. A cyclic colour map is the only
honest choice here - phase is an angle, so the map has to close on itself,
and the industry reads fringe *counts* rather than absolute hue. Pixels below
the coherence threshold are greyed out because their phase is meaningless.

Auxiliary panels (58 % of the main panel width): the unwrapped line-of-sight
velocity along profile A-A', and the coherence histogram that justifies the
mask.

Every number on the figure - fringe spacing, peak rate, masked fraction,
signal-to-noise ratio - is computed from the arrays that are drawn.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    image_axes, halo_text, add_scale_bar)

N = 260                       # pixels per side
KM = 22.0                     # scene width
EXT = [0.0, KM, 0.0, KM]
LAMBDA_CM = 5.55              # C-band wavelength
K_PHASE = 4.0 * np.pi / LAMBDA_CM      # rad per cm of LOS displacement
T_YR = 2.5                    # temporal baseline
COH_MIN = 0.30                # masking threshold
PHASE_CMAP = "twilight"       # cyclic: mandatory for wrapped phase


def field(rng, sigma):
    z = gaussian_filter(rng.normal(size=(N, N)), sigma, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-12)


def build(rng):
    yy, xx = np.mgrid[0:N, 0:N] * (KM / N)

    # LOS velocity in cm/yr: pumping bowl, inflating vent, regional tilt
    vel = (-4.60 * np.exp(-(((xx - 8.2) / 3.4) ** 2 + ((yy - 9.0) / 2.7) ** 2))
           - 1.85 * np.exp(-(((xx - 14.6) / 1.9) ** 2 + ((yy - 5.4) / 1.7) ** 2))
           + 1.35 * np.exp(-(((xx - 16.2) / 2.4) ** 2 + ((yy - 16.0) / 2.2) ** 2))
           + 0.055 * (xx + 0.45 * yy)
           + 0.20 * field(rng, 16.0))
    vel = gaussian_filter(vel, 1.1)

    relief = field(rng, 13.0)
    water = ((xx - 4.2) ** 2 / 1.9 ** 2 + (yy - 18.4) ** 2 / 1.1 ** 2) < 1.0
    urban = np.exp(-(((xx - 8.4) / 3.2) ** 2 + ((yy - 9.2) / 2.6) ** 2)) > 0.55
    forest = (relief > np.quantile(relief, 0.70)) & ~urban & ~water

    coh = np.full((N, N), 0.55)
    coh[forest] = 0.20
    coh[urban] = 0.88
    coh[water] = 0.05
    coh += 0.10 * field(rng, 3.0)
    gy, gx = np.gradient(vel * T_YR)
    rate = np.hypot(gx, gy)
    rate /= rate.max() + 1e-12
    coh *= 1.0 - 0.40 * rate ** 3
    coh = np.clip(gaussian_filter(coh, 0.9), 0.02, 0.96)

    atmo = 1.9 * field(rng, 22.0)
    topo = 0.55 * relief
    noise = gaussian_filter(rng.normal(size=(N, N)), 0.6) * 2.6 * \
        (1.0 - coh) ** 1.7
    phase = K_PHASE * vel * T_YR + atmo + topo + noise
    return vel, coh, np.angle(np.exp(1j * phase))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(761)
    vel, coh, wrapped = build(rng)

    low = coh < COH_MIN
    show = np.ma.masked_where(low, wrapped)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (0.66, 0.95, 4.55, 4.55))
    ax_prof = axes_inches(fig, (6.45, 3.62, 3.35, 1.88))
    ax_coh = axes_inches(fig, (6.45, 0.95, 3.35, 1.72))

    # ---- (a) wrapped interferogram ---------------------------------------
    cmap = plt.get_cmap(PHASE_CMAP).copy()
    cmap.set_bad("#b3b3b3")
    im = ax.imshow(show, extent=EXT, origin="lower", cmap=cmap,
                   vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    image_axes(ax, color="#333333", lw=0.8)
    ax.set_title("Wrapped differential phase, %.1f-year baseline" % T_YR,
                 fontsize=9.0, fontweight="bold")

    a0, a1 = np.array([1.6, 3.4]), np.array([20.2, 18.4])
    ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="#1a1a1a", lw=1.2,
            ls=(0, (5, 2.5)), zorder=6)
    for lab, xy in (("A", a0), ("A'", a1)):
        ax.plot([xy[0]], [xy[1]], marker="o", ms=3.6, mfc="white",
                mec="#1a1a1a", mew=0.9, zorder=7)
        halo_text(ax, xy[0] + 0.45, xy[1] - 0.25, lab, fontsize=7.6,
                  fontweight="bold", ha="left", va="top", zorder=8)

    add_scale_bar(ax, 5.0, label="5 km", loc=(0.045, 0.050), height=0.019,
                  fontsize=6.3)
    halo_text(ax, 0.030, 0.965,
              "1 fringe = %.2f cm LOS ($\\lambda$/2)" % (LAMBDA_CM / 2.0),
              transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
              zorder=8)
    halo_text(ax, 0.030, 0.918,
              "grey: coherence $\\gamma$ < %.2f (%.1f %% of pixels)"
              % (COH_MIN, 100.0 * low.mean()), transform=ax.transAxes,
              ha="left", va="top", fontsize=6.5, zorder=8)

    cb = safe_colorbar(fig, im, ax, label="wrapped phase (rad)", width=0.12,
                       pad=0.08, labelsize=6.6,
                       ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # fringe count between the bowl centre and the scene edge
    v_ctr = float(vel[int(9.0 / KM * N), int(8.2 / KM * N)])
    v_edge = float(vel[int(20.0 / KM * N), int(1.5 / KM * N)])
    n_fr = abs(v_ctr - v_edge) * T_YR / (LAMBDA_CM / 2.0)
    ax.annotate("%.1f fringes to the bowl centre" % n_fr, xy=(8.2, 9.0),
                xytext=(9.4, 17.6), fontsize=6.5, color="#1a1a1a",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#1a1a1a",
                                shrinkA=1.0, shrinkB=2.0),
                bbox=dict(boxstyle="round,pad=0.26", fc="white",
                          ec="#cccccc", lw=0.5, alpha=0.92))

    # ---- (b) unwrapped profile along A-A' ---------------------------------
    n_p = 420
    px = np.linspace(a0[0], a1[0], n_p)
    py = np.linspace(a0[1], a1[1], n_p)
    ix = np.clip((px / KM * N).astype(int), 0, N - 1)
    iy = np.clip((py / KM * N).astype(int), 0, N - 1)
    dist = np.hypot(px - a0[0], py - a0[1])
    vp = vel[iy, ix]
    cp = coh[iy, ix]
    sig = 0.16 + 0.75 * (1.0 - cp) ** 2      # cm/yr, worse where incoherent

    ax_prof.axhline(0.0, color="#555555", lw=0.8, zorder=2)
    gap = np.where(cp < COH_MIN, np.nan, vp)
    ax_prof.fill_between(dist, vp - sig, vp + sig, color="#3b5b92",
                         alpha=0.18, lw=0, zorder=3)
    ax_prof.plot(dist, vp, color="#9cc3dd", lw=1.0, zorder=4)
    ax_prof.plot(dist, gap, color="#1f4d7a", lw=1.6, zorder=5)
    i_min = int(np.nanargmin(gap))
    ax_prof.plot([dist[i_min]], [gap[i_min]], marker="o", ms=4.4,
                 mfc="white", mec="#b23a48", mew=1.3, zorder=6)
    ax_prof.annotate("%.2f cm yr$^{-1}$" % gap[i_min],
                     xy=(dist[i_min], gap[i_min]),
                     xytext=(dist[i_min] + 2.6, gap[i_min] + 0.9),
                     fontsize=6.5, color="#b23a48", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", lw=0.8,
                                     color="#b23a48"))
    ax_prof.set_xlim(0, dist.max())
    ax_prof.set_xlabel("Distance along A-A' (km)", fontsize=7.4, labelpad=1.0)
    ax_prof.set_ylabel("LOS velocity (cm yr$^{-1}$)", fontsize=7.4)
    ax_prof.set_title("Unwrapped deformation along the profile",
                      fontsize=8.5, fontweight="bold")
    ax_prof.tick_params(labelsize=6.6)
    style_axes(ax_prof)
    ax_prof.grid(axis="y", lw=0.5, alpha=0.45)
    ax_prof.text(0.012, 0.06, "A", transform=ax_prof.transAxes,
                 fontsize=7.4, fontweight="bold")
    ax_prof.text(0.972, 0.06, "A'", transform=ax_prof.transAxes,
                 fontsize=7.4, fontweight="bold")
    ax_prof.text(0.985, 0.32, "pale line: masked, low $\\gamma$\n"
                 "band: $\\pm$1$\\sigma$ from coherence",
                 transform=ax_prof.transAxes, ha="right", va="top",
                 fontsize=6.2, color="#777777", linespacing=1.3)

    # ---- (c) coherence histogram ------------------------------------------
    c = coh.ravel()
    ax_coh.hist(c, bins=60, range=(0.0, 1.0), color="#9cc3dd",
                edgecolor="white", linewidth=0.25, zorder=3)
    ax_coh.axvline(COH_MIN, color="#b23a48", lw=1.2, ls="--", zorder=4)
    ax_coh.axvline(float(np.median(c)), color="#333333", lw=1.0,
                   ls=(0, (3, 2)), zorder=4)
    top = ax_coh.get_ylim()[1]
    ax_coh.text(COH_MIN - 0.02, top * 0.94, "mask", ha="right", va="top",
                fontsize=6.3, color="#b23a48")
    ax_coh.text(np.median(c) + 0.02, top * 0.94,
                "median %.2f" % np.median(c), ha="left", va="top",
                fontsize=6.3, color="#333333")
    ax_coh.set_xlim(0, 1)
    ax_coh.set_xlabel("Interferometric coherence $\\gamma$ (-)",
                      fontsize=7.4, labelpad=1.0)
    ax_coh.set_ylabel("Pixels (-)", fontsize=7.4)
    ax_coh.set_title("Coherence sets what can be unwrapped", fontsize=8.5,
                     fontweight="bold")
    ax_coh.tick_params(labelsize=6.6)
    style_axes(ax_coh)
    ax_coh.grid(axis="y", lw=0.5, alpha=0.45)
    g = np.clip(c[c >= COH_MIN], 0.0, 0.995)
    snr = 10.0 * np.log10(np.mean(g / (1.0 - g)))
    ax_coh.text(0.985, 0.78, "retained pixels: %.1f %%\nmean phase SNR "
                "= %.1f dB" % (100.0 * (1.0 - low.mean()), snr),
                transform=ax_coh.transAxes, ha="right", va="top",
                fontsize=6.3, color="#555555", linespacing=1.3)

    panel_labels_inches(fig, [ax, ax_prof, ax_coh], dx=-0.18, dy=0.14)


    savefig(fig, "761_rs_sar_interferogram.png")


if __name__ == "__main__":
    main()
