"""InSAR ground-deformation monitoring over a subsiding basin.

Top row (rasters, identical footprint): wrapped differential interferogram,
interferometric coherence and the unwrapped LOS velocity field.
Bottom row (analysis): velocity profile along A-A', PS time series at three
targets and the velocity histogram of the whole scene.

Coherence is generated from land cover (built-up high, cropland medium,
forest low, water ~0) and is additionally degraded where the fringe rate is
high, which is what limits unwrapping over the subsidence funnel.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label, get_cmap, add_colorbar)

N = 200                      # 100 m pixels -> 20 km square
EXT = [0.0, 20.0, 0.0, 20.0]
LAMBDA_MM = 55.6             # C-band wavelength
K_PHASE = 4.0 * np.pi / LAMBDA_MM
T_SPAN = 4.0                 # years of cumulative displacement


def norm01(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def fractal_noise(rng, shape, sigmas=(20, 8, 3), weights=(1.0, 0.5, 0.22)):
    out = np.zeros(shape)
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.normal(size=shape), s, mode="reflect")
    return norm01(out)


def build_scene(rng):
    yy, xx = np.mgrid[0:N, 0:N]
    xk, yk = xx * 0.1, yy * 0.1

    # ---- land cover -------------------------------------------------------
    river = np.abs(xk - (15.4 + 1.6 * np.sin(yk / 3.0))) < 0.30
    lake = ((xk - 4.0) ** 2 / 1.8 ** 2 + (yk - 17.0) ** 2 / 1.1 ** 2) < 1.0
    water = river | lake

    urban = np.exp(-(((xk - 9.0) / 3.6) ** 2 + ((yk - 9.6) / 3.0) ** 2))
    urban = (urban + 0.40 * (fractal_noise(rng, (N, N), (6, 2)) - 0.45)) > 0.42
    urban &= ~water
    elev = fractal_noise(rng, (N, N), (24, 9, 3))
    forest = (elev > np.quantile(elev, 0.74)) & (~urban) & (~water)
    crop = ~(urban | forest | water)

    # ---- LOS velocity: two subsidence funnels + tectonic tilt ------------
    vel = (-42.0 * np.exp(-(((xk - 9.2) / 3.1) ** 2 + ((yk - 9.4) / 2.5) ** 2))
           - 17.0 * np.exp(-(((xk - 15.0) / 1.7) ** 2 + ((yk - 15.2) / 1.5) ** 2))
           + 1.6 * norm01(xk + 0.4 * yk)
           + 2.2 * (fractal_noise(rng, (N, N), (18, 7)) - 0.5))
    vel = gaussian_filter(vel, 1.2)

    # ---- coherence --------------------------------------------------------
    coh = np.full((N, N), 0.44)
    coh[crop] = 0.42
    coh[forest] = 0.15
    coh[urban] = 0.89
    coh[water] = 0.04
    coh += 0.09 * (fractal_noise(rng, (N, N), (5, 1.5)) - 0.5)
    # only the very steepest fringe gradients cause extra decorrelation, so
    # built-up scatterers stay coherent even inside the subsidence funnel
    gy, gx = np.gradient(vel * T_SPAN)
    fringe_rate = norm01(gaussian_filter(np.hypot(gx, gy), 1.5))
    coh *= 1.0 - 0.45 * fringe_rate ** 3
    coh = np.clip(gaussian_filter(coh, 0.8), 0.02, 0.97)

    # ---- wrapped interferometric phase -----------------------------------
    atmo = 2.4 * (fractal_noise(rng, (N, N), (28, 12)) - 0.5)
    topo_res = 0.9 * (elev - 0.5)
    noise = rng.normal(0, 1.0, (N, N)) * 0.9 * (1.0 - coh) ** 1.6
    phase = K_PHASE * vel * T_SPAN + atmo + topo_res + gaussian_filter(noise, 0.5)
    wrapped = np.angle(np.exp(1j * phase))
    return dict(vel=vel, coh=coh, wrapped=wrapped, water=water, urban=urban,
                forest=forest)


def frame_raster(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#333333")


def raster_cbar(fig, mappable, ax, label, **kw):
    """House colorbar for a frame_raster() axes.

    These axes are set_aspect("equal"), so fig.colorbar(ax=...) sizes the bar
    from the axes box before the aspect shrink and leaves it detached from the
    image. An inset in axes coordinates tracks the real raster position.
    """
    cb = fig.colorbar(mappable, cax=ax.inset_axes([1.035, 0.0, 0.042, 1.0]),
                      **kw)
    cb.set_label(label, fontsize=6.8)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.3, width=0.7, length=2.5)
    return cb


def add_scalebar(ax, x0, y0, length, label, color="white", stroke="black"):
    fx = [pe.withStroke(linewidth=2.0, foreground=stroke)]
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.2,
            solid_capstyle="butt", zorder=8, path_effects=fx)
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - 0.30, y0 + 0.30], color=color, lw=1.1,
                zorder=8, path_effects=fx)
    ax.text(x0 + length / 2, y0 + 0.45, label, color=color, ha="center",
            va="bottom", fontsize=6.3, fontweight="bold", zorder=8,
            path_effects=fx)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240187)
    sc = build_scene(rng)
    vel, coh, wrapped = sc["vel"], sc["coh"], sc["wrapped"]

    # profile A-A' and PS targets (km)
    a0, a1 = np.array([1.2, 3.0]), np.array([18.4, 16.6])
    pts = {"P1": (9.2, 9.4), "P2": (12.6, 12.2), "P3": (17.6, 4.6)}
    pcol = {"P1": "#b23a48", "P2": "#c9a227", "P3": "#2e7d5b"}

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.86, 1.0],
                  hspace=0.34, wspace=0.30)

    # ---- (a) wrapped phase -----------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    im_a = axa.imshow(wrapped, origin="lower", extent=EXT,
                      cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi,
                      interpolation="nearest")
    frame_raster(axa)
    add_scalebar(axa, 1.0, 1.2, 5.0, "5 km")
    axa.set_title("Wrapped phase", fontsize=8.2)
    add_panel_label(axa, "(a)", x=-0.06, y=1.02, size=10)
    cb = raster_cbar(fig, im_a, axa, "phase (rad)",
                     ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    axa.text(0.03, 0.975, f"1 fringe = {LAMBDA_MM / 2:.1f} mm LOS",
             transform=axa.transAxes, ha="left", va="top", fontsize=6.1,
             color="white",
             path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    # ---- (b) coherence ----------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    im_b = axb.imshow(coh, origin="lower", extent=EXT, cmap="bone",
                      vmin=0, vmax=1, interpolation="bilinear")
    frame_raster(axb)
    axb.set_title("Coherence", fontsize=8.2)
    add_panel_label(axb, "(b)", x=-0.06, y=1.02, size=10)
    raster_cbar(fig, im_b, axb, "$\\gamma$ (\u2013)")
    axb.text(0.03, 0.975,
             f"built-up $\\gamma$ = {coh[sc['urban']].mean():.2f}\n"
             f"forest $\\gamma$ = {coh[sc['forest']].mean():.2f}\n"
             f"water $\\gamma$ = {coh[sc['water']].mean():.2f}",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.1,
             color="white", linespacing=1.25,
             path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    # ---- (c) LOS velocity -------------------------------------------------
    axc = fig.add_subplot(gs[0, 2])
    vlim = 30
    vshow = np.where(coh > 0.10, vel, np.nan)
    cmap_v = get_cmap("cb_diverging").copy()
    cmap_v.set_bad("#e0e0e0")
    im_c = axc.imshow(vshow, origin="lower", extent=EXT, cmap=cmap_v,
                      vmin=-vlim, vmax=vlim, interpolation="bilinear")
    frame_raster(axc)
    axc.plot([a0[0], a1[0]], [a0[1], a1[1]], color="black", lw=1.1,
             ls=(0, (5, 2)), zorder=6)
    for lab, xy, ha in (("A", a0, "right"), ("A'", a1, "left")):
        axc.text(xy[0], xy[1] + 0.5, lab, fontsize=7.2, fontweight="bold",
                 ha=ha, va="bottom", color="black", zorder=7,
                 path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    for lab, (px, py) in pts.items():
        axc.plot(px, py, marker="o", ms=4.2, mfc=pcol[lab], mec="black",
                 mew=0.7, zorder=7)
        axc.text(px + 0.6, py + 0.4, lab, fontsize=6.4, fontweight="bold",
                 color="black", zorder=7,
                 path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    add_scalebar(axc, 1.0, 1.2, 5.0, "5 km", color="black", stroke="white")
    axc.set_title("LOS velocity", fontsize=8.2)
    add_panel_label(axc, "(c)", x=-0.06, y=1.02, size=10)
    raster_cbar(fig, im_c, axc, "$v_{LOS}$ (mm yr$^{-1}$)", extend="both")

    # ---- (d) profile A-A' -------------------------------------------------
    axd = fig.add_subplot(gs[1, 0])
    nprof = 240
    px = np.linspace(a0[0], a1[0], nprof)
    py = np.linspace(a0[1], a1[1], nprof)
    ix = np.clip((px / 0.1).astype(int), 0, N - 1)
    iy = np.clip((py / 0.1).astype(int), 0, N - 1)
    dist = np.hypot(px - a0[0], py - a0[1])
    vprof = vel[iy, ix]
    sprof = 1.2 + 2.6 * (1.0 - coh[iy, ix])
    axd.axhline(0, color="#555555", lw=0.8)
    axd.fill_between(dist, vprof - sprof, vprof + sprof, color="#3b5b92",
                     alpha=0.20, lw=0)
    axd.plot(dist, vprof, color="#1f3f6b", lw=1.5)
    axd.set_xlim(0, dist.max())
    axd.set_xlabel("Distance along A\u2013A' (km)", labelpad=1.5)
    axd.set_ylabel("$v_{LOS}$ (mm yr$^{-1}$)")
    axd.set_title("Deformation profile", fontsize=8.2)
    style_axes(axd)
    axd.grid(axis="y", lw=0.5, alpha=0.5)
    axd.text(0.02, 0.06, "A", transform=axd.transAxes, fontsize=7.5,
             fontweight="bold")
    axd.text(0.965, 0.06, "A'", transform=axd.transAxes, fontsize=7.5,
             fontweight="bold")
    imin = int(np.argmin(vprof))
    axd.annotate(f"{vprof[imin]:.0f}".replace("-", "\u2212")
                 + " mm yr$^{-1}$",
                 xy=(dist[imin], vprof[imin]),
                 xytext=(dist[imin] + 2.8, vprof[imin] + 4),
                 fontsize=6.4, fontweight="bold", color="#8c2f2f",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#8c2f2f"))
    add_panel_label(axd, "(d)", x=-0.20, y=1.03, size=10)

    # ---- (e) PS time series -----------------------------------------------
    axe = fig.add_subplot(gs[1, 1])
    t = np.arange(0, T_SPAN + 1e-9, 12.0 / 365.0)
    for lab, (qx, qy) in pts.items():
        v0 = vel[int(qy / 0.1), int(qx / 0.1)]
        seas = 3.4 * np.sin(2 * np.pi * (t - 0.15))
        d = v0 * t + seas + rng.normal(0, 2.6, t.size)
        sig = 2.5 + 0.9 * np.sqrt(t)
        axe.fill_between(t, d - sig, d + sig, color=pcol[lab], alpha=0.15, lw=0)
        axe.plot(t, d, color=pcol[lab], lw=0.9, marker="o", ms=1.8,
                 label=f"{lab}: {v0:+.0f}".replace("-", "\u2212")
                 + " mm yr$^{-1}$")
    axe.axhline(0, color="#555555", lw=0.8)
    axe.set_xlim(0, T_SPAN)
    axe.set_xlabel("Time since first acquisition (yr)", labelpad=1.5)
    axe.set_ylabel("LOS displacement (mm)")
    axe.set_title("PS time series", fontsize=8.2)
    style_axes(axe)
    axe.grid(axis="y", lw=0.5, alpha=0.5)
    lg = axe.legend(loc="lower left", fontsize=6.2, frameon=True,
                    framealpha=0.92, edgecolor="#cccccc", handlelength=1.2,
                    labelspacing=0.22, borderpad=0.3)
    lg.get_frame().set_linewidth(0.5)
    add_panel_label(axe, "(e)", x=-0.22, y=1.03, size=10)

    # ---- (f) velocity histogram -------------------------------------------
    axf = fig.add_subplot(gs[1, 2])
    vv = vel[coh > 0.10].ravel()
    axf.hist(vv, bins=60, color="#7aa0c4", edgecolor="white", linewidth=0.25)
    axf.axvline(0, color="#555555", lw=0.8)
    axf.axvline(np.median(vv), color="#b23a48", lw=1.2, ls="--")
    axf.set_yscale("log")
    axf.set_xlabel("$v_{LOS}$ (mm yr$^{-1}$)", labelpad=1.5)
    axf.set_ylabel("Pixel count")
    axf.set_title("Velocity distribution", fontsize=8.2)
    style_axes(axf)
    axf.text(0.03, 0.96,
             f"median {np.median(vv):+.1f} mm yr$^{{-1}}$\n"
             f"$\\sigma$ = {vv.std():.1f} mm yr$^{{-1}}$\n"
             f"{100 * (vv < -10).mean():.1f} % below \u221210 mm yr$^{{-1}}$",
             transform=axf.transAxes, ha="left", va="top", fontsize=6.2,
             linespacing=1.3,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.95))
    add_panel_label(axf, "(f)", x=-0.22, y=1.03, size=10)

    fig.subplots_adjust(top=0.895, bottom=0.085, left=0.055, right=0.975)
    savefig(fig, "187_rs_insar_deformation.png")


if __name__ == "__main__":
    main()
