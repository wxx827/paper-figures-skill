"""Above-ground biomass inversion, validation and error structure.

Left column (rasters): retrieved AGB and its pixel-wise standard error over the
same 8 x 10 km footprint.  Right column (statistics): predicted-vs-observed
validation against field plots and the residual structure against plot biomass.

The forward model deliberately includes signal saturation above ~140 Mg ha-1,
so the retrieval underestimates dense stands and the residuals turn
systematically negative at the high end - the classic limitation of optical /
C-band AGB inversion.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, sobel

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

NY, NX = 160, 200            # 50 m pixels -> 8 km x 10 km
EXT = [0.0, 10.0, 0.0, 8.0]
B_SAT = 140.0                # onset of saturation (Mg ha-1)
ALPHA = 0.055


def norm01(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def fractal_noise(rng, shape, sigmas=(18, 7, 2.5), weights=(1.0, 0.5, 0.22)):
    out = np.zeros(shape)
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.normal(size=shape), s, mode="reflect")
    return norm01(out)


def saturating_retrieval(b):
    """Sensor response saturates for dense canopies."""
    return b - ALPHA * np.clip(b - B_SAT, 0, None) ** 1.35


def build_biomass(rng):
    yy, xx = np.mgrid[0:NY, 0:NX]
    xk, yk = xx * 0.05, yy * 0.05

    base = fractal_noise(rng, (NY, NX), (20, 8, 3))
    stands = np.zeros((NY, NX))
    for _ in range(9):
        cx, cy = rng.uniform(0.5, 9.5), rng.uniform(0.5, 7.5)
        sx, sy = rng.uniform(0.7, 1.9), rng.uniform(0.6, 1.6)
        stands += rng.uniform(0.35, 1.0) * np.exp(
            -(((xk - cx) / sx) ** 2 + ((yk - cy) / sy) ** 2))
    texture = fractal_noise(rng, (NY, NX), (5, 2, 0.9), (1.0, 0.6, 0.3))
    agb = 300.0 * norm01(0.50 * base + 0.36 * norm01(stands)
                         + 0.14 * texture) ** 1.25

    # clear-cut / plantation blocks with sharp, man-made edges
    for _ in range(11):
        x0, y0 = rng.uniform(0.2, 8.6), rng.uniform(0.2, 6.8)
        w, h = rng.uniform(0.5, 1.5), rng.uniform(0.4, 1.1)
        m = (xk > x0) & (xk < x0 + w) & (yk > y0) & (yk < y0 + h)
        agb[m] = rng.uniform(4.0, 45.0) + 12.0 * fractal_noise(
            rng, (NY, NX), (4, 1.5))[m]

    # riparian corridor: low, wet, non-forested
    river = np.abs(xk - (3.4 + 1.5 * np.sin(yk / 1.7) + 0.5 * np.sin(yk / 0.6)))
    agb *= np.clip(river / 0.55, 0.05, 1.0)

    agb = gaussian_filter(agb, 1.0)
    return np.clip(agb, 0.0, 320.0)


def frame_raster(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#333333")


def raster_cbar(fig, mappable, ax, label, **kw):
    """House colorbar for an aspect-equal raster axes.

    fig.colorbar(ax=...) sizes the bar from the axes box before set_aspect
    shrinks it, which strands the bar away from the image; an inset in axes
    coordinates follows the position the raster really occupies.
    """
    cb = fig.colorbar(mappable, cax=ax.inset_axes([1.035, 0.0, 0.042, 1.0]),
                      **kw)
    cb.set_label(label, fontsize=7.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.3, width=0.7, length=2.5)
    return cb


def add_scalebar(ax, x0, y0, length, label, color="white"):
    fx = [pe.withStroke(linewidth=2.0, foreground="black")]
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.2,
            solid_capstyle="butt", zorder=7, path_effects=fx)
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - 0.13, y0 + 0.13], color=color, lw=1.1,
                zorder=7, path_effects=fx)
    ax.text(x0 + length / 2, y0 + 0.20, label, color=color, ha="center",
            va="bottom", fontsize=6.3, fontweight="bold", zorder=7,
            path_effects=fx)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240186)

    agb = build_biomass(rng)
    retrieved = saturating_retrieval(agb) + gaussian_filter(
        rng.normal(0, 9.0, (NY, NX)), 1.2)
    retrieved = np.clip(retrieved, 0, None)

    # pixel-wise standard error: grows with biomass (saturation) and with
    # local heterogeneity (stand edges, mixed pixels)
    edge = np.hypot(sobel(gaussian_filter(agb, 1.0), 0),
                    sobel(gaussian_filter(agb, 1.0), 1))
    edge = norm01(gaussian_filter(edge, 1.5))
    se = (7.0 + 0.115 * retrieved + 42.0 * edge
          + 3.0 * fractal_noise(rng, (NY, NX), (10, 4)))
    se = gaussian_filter(se, 1.0)

    fig = plt.figure(figsize=(7.3, 5.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.12],
                  height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.30)

    # ---- (a) retrieved AGB ------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    im_a = axa.imshow(retrieved, origin="lower", extent=EXT, cmap="YlGn",
                      vmin=0, vmax=280, interpolation="bilinear")
    frame_raster(axa)
    add_scalebar(axa, 0.5, 0.55, 2.0, "2 km")
    axa.set_title("Retrieved above-ground biomass", fontsize=8.2)
    add_panel_label(axa, "(a)", x=-0.035, y=1.02, size=10)
    raster_cbar(fig, im_a, axa, "AGB (Mg ha$^{-1}$)")

    # ---- (c) standard error ----------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    im_c = axc.imshow(se, origin="lower", extent=EXT, cmap="magma_r",
                      vmin=5, vmax=np.percentile(se, 99), interpolation="bilinear")
    frame_raster(axc)
    add_scalebar(axc, 0.5, 0.55, 2.0, "2 km", color="white")
    axc.set_title("Retrieval standard error", fontsize=8.2)
    add_panel_label(axc, "(c)", x=-0.035, y=1.02, size=10)
    raster_cbar(fig, im_c, axc, "SE (Mg ha$^{-1}$)")
    axc.text(0.985, 0.03, "largest errors at stand edges\nand in dense canopy",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=6.1,
             color="white", linespacing=1.2,
             path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    # ---- field plots ------------------------------------------------------
    n = 78
    obs = np.sort(rng.uniform(8, 315, n))
    obs_err = 6.0 + 0.045 * obs
    pred = (saturating_retrieval(obs)
            + rng.normal(0, 1.0, n) * (11.0 + 0.085 * obs))
    pred = np.clip(pred, 2, None)
    resid = pred - obs

    k, b0 = np.polyfit(obs, pred, 1)
    r2 = np.corrcoef(obs, pred)[0, 1] ** 2
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    lim = [0, 340]

    # ---- (b) predicted vs observed ---------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_facecolor("#fbfbf9")
    axb.plot(lim, lim, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=2)
    axb.errorbar(obs, pred, xerr=obs_err, fmt="none", ecolor="#9bbfae",
                 elinewidth=0.6, alpha=0.9, zorder=3)
    sc = axb.scatter(obs, pred, s=17, c=pred / np.maximum(obs, 1e-6),
                     cmap="YlGn", vmin=0.55, vmax=1.15, edgecolor="#0b3d2e",
                     linewidth=0.4, zorder=4)
    xg = np.linspace(*lim, 50)
    axb.plot(xg, k * xg + b0, color="#b23a48", lw=1.4, zorder=5)
    axb.axvspan(B_SAT, lim[1], color="#c0603f", alpha=0.07, lw=0, zorder=1)
    axb.text(B_SAT + 6, 22, "saturation domain", fontsize=6.2, rotation=0,
             color="#8c2f2f", va="bottom", ha="left", fontweight="bold")
    axb.axvline(B_SAT, color="#8c2f2f", lw=0.8, ls=(0, (3, 2)), zorder=2)
    axb.set_xlim(lim)
    axb.set_ylim(lim)
    axb.set_aspect("equal")
    axb.set_xlabel("Field AGB (Mg ha$^{-1}$)", labelpad=1.5)
    axb.set_ylabel("Retrieved AGB (Mg ha$^{-1}$)")
    axb.set_title("Validation against %d field plots" % n, fontsize=8.2)
    style_axes(axb)
    axb.text(0.035, 0.975,
             f"$R^2$ = $\\bf{{{r2:.2f}}}$\n"
             f"RMSE = $\\bf{{{rmse:.1f}}}$ Mg ha$^{{-1}}$\n"
             f"rRMSE = {100 * rmse / obs.mean():.1f} %\n"
             f"bias = $\\bf{{{bias:+.1f}}}$ Mg ha$^{{-1}}$\n"
             f"slope = {k:.2f},  $n$ = {n}",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.4,
             linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.95))
    add_panel_label(axb, "(b)", x=-0.16, y=1.02, size=10)
    raster_cbar(fig, sc, axb, "retrieved / field (\u2013)")

    # ---- (d) residuals vs biomass ----------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    axd.set_facecolor("#fbfbf9")
    axd.axhline(0, color="#555555", lw=0.9)
    axd.axvspan(B_SAT, lim[1], color="#c0603f", alpha=0.07, lw=0, zorder=1)
    axd.axvline(B_SAT, color="#8c2f2f", lw=0.8, ls=(0, (3, 2)), zorder=2)
    axd.scatter(obs, resid, s=15, facecolor="#6bbf8a", edgecolor="#0b3d2e",
                linewidth=0.4, alpha=0.9, zorder=4)

    edges = np.linspace(0, 340, 9)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    mu = np.array([resid[(obs >= a) & (obs < b)].mean() if
                   ((obs >= a) & (obs < b)).sum() > 2 else np.nan
                   for a, b in zip(edges[:-1], edges[1:])])
    sd = np.array([resid[(obs >= a) & (obs < b)].std() if
                   ((obs >= a) & (obs < b)).sum() > 2 else np.nan
                   for a, b in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(mu)
    axd.fill_between(ctr[ok], (mu - sd)[ok], (mu + sd)[ok], color="#2e7d5b",
                     alpha=0.18, lw=0, zorder=3)
    axd.plot(ctr[ok], mu[ok], color="#0b3d2e", lw=1.5, marker="o", ms=3.4,
             mec="white", mew=0.6, zorder=5, label="binned mean $\\pm$ 1 SD")
    axd.plot(xg, saturating_retrieval(xg) - xg, color="#8c2f2f", lw=1.2,
             ls="--", zorder=5, label="saturation model")
    axd.set_xlim(lim)
    axd.set_xlabel("Field AGB (Mg ha$^{-1}$)", labelpad=1.5)
    axd.set_ylabel("Residual (Mg ha$^{-1}$)")
    axd.set_title("Residual structure: high-biomass underestimation",
                  fontsize=8.2)
    style_axes(axd)
    axd.grid(axis="y", lw=0.5, alpha=0.5)
    axd.legend(loc="lower left", fontsize=6.3, frameon=False,
               handlelength=1.5, labelspacing=0.25, borderaxespad=0.3)
    hi = obs > B_SAT
    def signed(v):
        return f"{v:+.1f}".replace("-", "\u2212")

    axd.text(0.985, 0.95,
             f"mean residual, AGB < {B_SAT:.0f}:  {signed(resid[~hi].mean())}\n"
             f"mean residual, AGB > {B_SAT:.0f}:  {signed(resid[hi].mean())}"
             "  Mg ha$^{-1}$",
             transform=axd.transAxes, ha="right", va="top", fontsize=6.2,
             linespacing=1.3, fontweight="bold", color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.95))
    add_panel_label(axd, "(d)", x=-0.16, y=1.02, size=10)

    fig.subplots_adjust(top=0.90, bottom=0.075, left=0.045, right=0.975)
    savefig(fig, "186_rs_biomass_inversion.png")


if __name__ == "__main__":
    main()
