"""Uncertainty budget of a gridded evapotranspiration product.

(a) retrieved daily ET, (b) its pixel-wise standard error on the same
footprint, (c) the coefficient of variation reclassified into reliability
classes, and (d) a subordinate panel (tinted background, reduced size) that
splits the error variance into its four sources for each land-cover type.

The error model is consistent across the panels: the standard error grows with
the ET magnitude and with sub-pixel heterogeneity, so the relative uncertainty
is worst over sparsely vegetated and built-up areas where ET is small.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.ndimage import gaussian_filter, sobel

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

N = 220                            # 100 m pixels -> 22 km square
EXT = [0.0, 22.0, 0.0, 22.0]

COVERS = ["Water", "Forest", "Irrigated\ncrop", "Dryland\ncrop", "Built-up"]
# variance shares (%): sensor noise, model structure, parameters, scaling
BUDGET = {
    "Water": (41, 34, 14, 11),
    "Forest": (21, 44, 26, 9),
    "Irrigated\ncrop": (26, 31, 33, 10),
    "Dryland\ncrop": (30, 33, 25, 12),
    "Built-up": (23, 40, 15, 22),
}
SRC_COL = ["#3b5b92", "#c0603f", "#c9a227", "#6b7f8c"]
SRC_NAME = ["Sensor noise", "Model structure", "Parameter uncertainty",
            "Scaling / mixed pixels"]


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
    elev = fractal_noise(rng, (N, N), (24, 10, 3))

    river = np.abs(xk - (7.0 + 2.4 * np.sin(yk / 3.4)
                         + 0.8 * np.sin(yk / 1.2))) < 0.30
    lake = ((xk - 17.6) ** 2 / 1.9 ** 2 + (yk - 4.4) ** 2 / 1.2 ** 2) < 1.0
    water = river | lake

    urban = np.exp(-(((xk - 12.4) / 2.4) ** 2 + ((yk - 8.0) / 1.9) ** 2))
    urban = (urban + 0.40 * (fractal_noise(rng, (N, N), (6, 2)) - 0.45)) > 0.44
    urban &= ~water

    forest = (elev > np.quantile(elev, 0.76)) & (~urban) & (~water)

    # centre-pivot irrigation circles
    pivot = np.zeros((N, N), bool)
    for _ in range(26):
        cx, cy = rng.uniform(2.0, 20.0), rng.uniform(2.0, 20.0)
        r = rng.uniform(0.7, 1.25)
        m = ((xk - cx) ** 2 + (yk - cy) ** 2) < r ** 2
        if not (m & (urban | water)).any():
            pivot |= m

    lc = np.full((N, N), 3, dtype=int)          # dryland crop
    lc[forest] = 1
    lc[pivot] = 2
    lc[urban] = 4
    lc[water] = 0

    et_mean = {0: 5.2, 1: 4.3, 2: 6.6, 3: 2.3, 4: 1.5}
    et = np.zeros((N, N))
    for k, v in et_mean.items():
        et[lc == k] = v
    # rain-fed parcels carry field-scale management differences
    block = (yy // 12) * 512 + (xx // 15)
    parcel = rng.uniform(0.65, 1.35, block.max() + 1)[block]
    et = np.where(lc == 3, et * parcel, et)
    et *= 0.84 + 0.32 * fractal_noise(rng, (N, N), (14, 5))
    et += 0.25 * (fractal_noise(rng, (N, N), (4, 1.4)) - 0.5)
    et = np.clip(gaussian_filter(et, 0.8), 0.15, 8.5)

    # sub-pixel heterogeneity proxy: local standard deviation of ET
    sm = gaussian_filter(et, 3.0)
    hetero = norm01(gaussian_filter(np.abs(et - sm), 2.5))
    edge = norm01(gaussian_filter(
        np.hypot(sobel(gaussian_filter(et, 1.0), 0),
                 sobel(gaussian_filter(et, 1.0), 1)), 2.0))
    se = (0.07 + 0.060 * et + 0.50 * hetero + 0.32 * edge
          + 0.34 * (lc == 4) + 0.18 * fractal_noise(rng, (N, N), (9, 3)))
    se = gaussian_filter(se, 1.1)
    return et, se, lc


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

    These axes are set_aspect("equal"); fig.colorbar(ax=...) reserves space
    from the box before that shrink and leaves the bar detached from the
    image, so anchor an inset in axes coordinates instead.
    """
    cb = fig.colorbar(mappable, cax=ax.inset_axes([1.035, 0.0, 0.042, 1.0]),
                      **kw)
    cb.set_label(label, fontsize=7.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5)
    return cb


def add_scalebar(ax, x0, y0, length, label, color="white", stroke="black"):
    fx = [pe.withStroke(linewidth=2.0, foreground=stroke)]
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.2,
            solid_capstyle="butt", zorder=8, path_effects=fx)
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - 0.32, y0 + 0.32], color=color, lw=1.1,
                zorder=8, path_effects=fx)
    ax.text(x0 + length / 2, y0 + 0.45, label, color=color, ha="center",
            va="bottom", fontsize=6.3, fontweight="bold", zorder=8,
            path_effects=fx)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240192)
    et, se, lc = build_scene(rng)
    cv = 100.0 * se / np.clip(et, 0.15, None)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.24, wspace=0.16)

    # ---- (a) ET estimate --------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    im_a = axa.imshow(et, origin="lower", extent=EXT, cmap="YlGnBu",
                      vmin=0, vmax=8, interpolation="bilinear")
    frame_raster(axa)
    add_scalebar(axa, 1.0, 1.2, 5.0, "5 km")
    axa.set_title("Retrieved evapotranspiration", fontsize=8.4)
    add_panel_label(axa, "(a)", x=-0.035, y=1.02, size=10)
    raster_cbar(fig, im_a, axa, "ET (mm d$^{-1}$)", extend="max")

    # ---- (b) standard error ----------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    im_b = axb.imshow(se, origin="lower", extent=EXT, cmap="magma_r",
                      vmin=0.1, vmax=np.percentile(se, 99),
                      interpolation="bilinear")
    frame_raster(axb)
    add_scalebar(axb, 1.0, 1.2, 5.0, "5 km")
    axb.set_title("Standard error of the estimate", fontsize=8.4)
    add_panel_label(axb, "(b)", x=-0.035, y=1.02, size=10)
    raster_cbar(fig, im_b, axb, "SE (mm d$^{-1}$)", extend="max")

    # ---- (c) CV reliability classes ---------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    bounds = [0, 10, 20, 30, 50, 200]
    names = ["< 10  (excellent)", "10\u201320  (good)", "20\u201330  (fair)",
             "30\u201350  (poor)", "> 50  (unreliable)"]
    cols = ["#1a6b52", "#7cc49a", "#f1d27a", "#e0894d", "#a63a2f"]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(bounds, cmap.N)
    axc.imshow(cv, origin="lower", extent=EXT, cmap=cmap, norm=norm,
               interpolation="nearest")
    frame_raster(axc)
    add_scalebar(axc, 1.0, 1.2, 5.0, "5 km")
    axc.set_title("Relative uncertainty class", fontsize=8.4)
    add_panel_label(axc, "(c)", x=-0.035, y=1.02, size=10)
    shares = [100.0 * ((cv >= lo) & (cv < hi)).mean()
              for lo, hi in zip(bounds[:-1], bounds[1:])]
    handles = [Patch(fc=c, ec="#555555", lw=0.4,
                     label=f"{n}   {s:.0f} %")
               for c, n, s in zip(cols, names, shares)]
    lg = axc.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(1.015, 1.0), fontsize=6.2, frameon=False,
                    handlelength=1.0, handleheight=0.85, labelspacing=0.35,
                    borderaxespad=0.0,
                    title="CV (%)   area share")
    lg.get_title().set_fontsize(6.4)
    lg.get_title().set_fontweight("bold")
    lg.set_clip_on(False)
    lg.set_zorder(12)
    for art in list(lg.get_texts()) + list(lg.legend_handles):
        art.set_clip_on(False)
        art.set_clip_path(None)

    # ---- (d) subordinate error-budget panel -------------------------------
    host = fig.add_subplot(gs[1, 1])
    host.set_xticks([])
    host.set_yticks([])
    # the host background would otherwise paint over the legend of panel (c)
    host.patch.set_visible(False)
    for s in host.spines.values():
        s.set_visible(False)
    host.add_patch(FancyBboxPatch(
        (0.30, 0.02), 0.72, 0.90, boxstyle="round,pad=0.015",
        transform=host.transAxes, facecolor="#f1f1ec", edgecolor="#d5d5cc",
        linewidth=0.7, zorder=0, clip_on=False))
    host.text(0.66, 0.935, "Error-variance budget", transform=host.transAxes,
              ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    add_panel_label(host, "(d)", x=0.245, y=0.955, size=10)

    sub = host.inset_axes([0.44, 0.20, 0.52, 0.64])
    ypos = np.arange(len(COVERS))[::-1]
    left = np.zeros(len(COVERS))
    for j, (nm, c) in enumerate(zip(SRC_NAME, SRC_COL)):
        vals = np.array([BUDGET[k][j] for k in COVERS], float)
        sub.barh(ypos, vals, left=left, height=0.60, color=c,
                 edgecolor="white", linewidth=0.5, label=nm, zorder=3)
        for yv, (v, b) in zip(ypos, zip(vals, left)):
            if v >= 14:
                sub.text(b + v / 2, yv, f"{v:.0f}", ha="center", va="center",
                         fontsize=6.0, color="white", fontweight="bold",
                         zorder=4)
        left += vals
    for yv, k in zip(ypos, COVERS):
        idx = COVERS.index(k)
        mask = lc == [0, 1, 2, 3, 4][idx]
        sub.text(103, yv, f"{se[mask].mean():.2f}", va="center", ha="left",
                 fontsize=6.0, fontweight="bold", color="#1a1a1a")
    sub.text(103, ypos[0] + 0.72, "mean SE\n(mm d$^{-1}$)", va="bottom",
             ha="left", fontsize=6.0, color="#555555", linespacing=1.15)
    sub.set_yticks(ypos)
    sub.set_yticklabels(COVERS, fontsize=6.1, linespacing=1.05)
    sub.set_xlim(0, 100)
    sub.set_xticks([0, 25, 50, 75, 100])
    sub.set_xlabel("share of error variance (%)", fontsize=6.3, labelpad=1.0)
    sub.tick_params(labelsize=6.0)
    sub.patch.set_alpha(0.0)
    style_axes(sub)
    sub.grid(axis="x", lw=0.45, alpha=0.5)
    lgd = sub.legend(loc="upper center", bbox_to_anchor=(0.44, -0.24), ncol=2,
                     fontsize=6.0, frameon=False, handlelength=1.0,
                     handleheight=0.85, columnspacing=1.0, labelspacing=0.28,
                     borderaxespad=0.0)
    lgd.set_clip_on(False)

    fig.subplots_adjust(top=0.905, bottom=0.035, left=0.035, right=0.905)
    savefig(fig, "192_rs_uncertainty_map.png")


if __name__ == "__main__":
    main()
