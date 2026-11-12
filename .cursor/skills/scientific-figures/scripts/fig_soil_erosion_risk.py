"""Catchment soil-erosion risk from a RUSLE-style factor product.

The map is a genuine factor product: five spatially varying fields (rainfall
erosivity R, erodibility K, topography LS, cover C and practice P) are built
on a synthetic DEM, multiplied cell by cell, and then classified with the
conventional six-step erosion-intensity scale.  Everything else in the figure
is read back out of that same raster - the drainage network comes from a D8
flow accumulation on the DEM, the factor bars are the variance shares of the
log factors, the scatter plots the cells against their own slope, and the
headline numbers (mean modulus, area above the tolerable rate) are computed,
not typed.

The class ramp runs pale green to deep red.  It is a hand-built, monotonically
darkening ramp rather than a stock RdYlGn: the severity order has to survive
greyscale printing.
"""
import heapq
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import (add_scale_bar, halo_text, lowess, panel_labels_inches,
                    safe_colorbar)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"

# six-step erosion intensity scale, t ha-1 a-1
BOUNDS = [0.0, 5.0, 25.0, 50.0, 80.0, 150.0, 1.0e4]
RISK = ["#d9e4d3", "#a9c49a", "#dcc27a", "#cf8f53", "#b23a48", "#7d2230"]
RISK_NAME = ["slight", "light", "moderate", "strong", "very strong",
             "severe"]
TOLERANCE = 5.0          # t ha-1 a-1, the tolerable soil-loss rate
NX, NY = 240, 190
CELL = 0.10              # km
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fractal(rng, shape, scales=(3.0, 8.0, 20.0, 48.0),
            weights=(0.14, 0.26, 0.34, 1.0)):
    """Sum of band-limited noise; the cheap way to a plausible landscape."""
    out = np.zeros(shape)
    for s, w in zip(scales, weights):
        f = gaussian_filter(rng.normal(0.0, 1.0, shape), s)
        out += w * f / (np.abs(f).max() + 1e-12)
    return out / np.abs(out).max()


def fill_sinks(dem, eps=1e-3):
    """Priority-flood depression filling.

    Without it the fractal DEM is riddled with one-cell pits, flow
    accumulation dies a few cells downslope of every ridge, and the drainage
    network comes out as disconnected stubs instead of a dendritic tree.
    """
    ny, nx = dem.shape
    filled = np.array(dem, dtype=float)
    seen = np.zeros(dem.shape, dtype=bool)
    heap = []
    for i in range(ny):
        for j in (0, nx - 1):
            seen[i, j] = True
            heapq.heappush(heap, (float(dem[i, j]), i, j))
    for j in range(nx):
        for i in (0, ny - 1):
            if not seen[i, j]:
                seen[i, j] = True
                heapq.heappush(heap, (float(dem[i, j]), i, j))
    while heap:
        e, i, j = heapq.heappop(heap)
        for di, dj in DIRS:
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx and not seen[ii, jj]:
                seen[ii, jj] = True
                filled[ii, jj] = max(dem[ii, jj], e + eps)
                heapq.heappush(heap, (float(filled[ii, jj]), ii, jj))
    return filled


def d8(dem):
    """Steepest-descent receiver index and contributing-cell count."""
    ny, nx = dem.shape
    pad = np.pad(dem, 1, constant_values=np.inf)
    best = np.full(dem.shape, -1, dtype=int)
    steep = np.zeros(dem.shape)
    for k, (di, dj) in enumerate(DIRS):
        sh = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        slope = (dem - sh) / np.hypot(di, dj)
        take = slope > steep
        steep[take] = slope[take]
        best[take] = k
    acc = np.ones(dem.size)
    for f in np.argsort(dem, axis=None)[::-1]:
        i, j = divmod(int(f), nx)
        k = best[i, j]
        if k >= 0:
            di, dj = DIRS[k]
            acc[(i + di) * nx + (j + dj)] += acc[f]
    return best, acc.reshape(dem.shape)


def basin_polygon(rng, cx, cy, r0):
    """Closed irregular catchment outline, in kilometres."""
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    bump = np.zeros_like(th)
    for k, a in zip((2, 3, 5, 7), (0.13, 0.09, 0.055, 0.035)):
        bump += a * np.sin(k * th + rng.uniform(0.0, 2.0 * np.pi))
    r = r0 * (1.0 + bump)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def main():
    set_pub_style()
    rng = np.random.default_rng(602)

    W, H = NX * CELL, NY * CELL
    xs = (np.arange(NX) + 0.5) * CELL
    ys = (np.arange(NY) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)

    # ---- terrain: a noisy cone draining to one outlet ----------------------
    x_out, y_out = 0.50 * W, 0.15 * H
    dist = np.hypot(X - x_out, Y - y_out)
    shape = (82.0 * dist / dist.max() +
             130.0 * fractal(rng, (NY, NX), (2.0, 5.0, 11.0, 26.0),
                             (0.18, 0.35, 0.55, 1.0)))
    shape = gaussian_filter(shape, 0.9)
    # relief is scaled so the catchment has a stated median gradient rather
    # than whatever the noise happened to produce
    g0y, g0x = np.gradient(shape, CELL * 1000.0)
    shape *= 0.115 / np.median(np.hypot(g0x, g0y))
    dem = 240.0 + shape - shape.min()

    gy, gx = np.gradient(dem, CELL * 1000.0)
    slope_pct = 100.0 * np.hypot(gx, gy)
    slope_rad = np.arctan(np.hypot(gx, gy))

    best, acc = d8(fill_sinks(dem))
    length = np.clip(np.sqrt(acc) * CELL * 1000.0, 15.0, 240.0)

    # ---- RUSLE factors -----------------------------------------------------
    m = np.clip(slope_pct / (slope_pct + 3.0), 0.20, 0.55)
    ls = ((length / 22.13) ** m *
          (10.8 * np.sin(slope_rad) + 0.03))
    ls = np.clip(ls, 0.05, 8.0)
    r_fac = 2000.0 + 900.0 * fractal(rng, (NY, NX), (16.0, 40.0), (0.4, 1.0))
    k_fac = 0.030 + 0.017 * fractal(rng, (NY, NX), (7.0, 22.0), (0.5, 1.0))
    cover = 0.5 + 0.5 * fractal(rng, (NY, NX), (5.0, 14.0, 30.0),
                                (0.3, 0.6, 1.0))
    c_fac = 0.006 * np.exp(4.4 * np.clip(cover, 0.0, 1.0))
    p_fac = 0.30 + 0.70 * (fractal(rng, (NY, NX), (9.0, 26.0),
                                   (0.5, 1.0)) * 0.5 + 0.5)
    soil_loss = r_fac * k_fac * ls * c_fac * p_fac

    # ---- catchment mask ----------------------------------------------------
    outline = basin_polygon(rng, 0.50 * W, 0.56 * H, 0.400 * H)
    inside = Polygon(outline).get_path().contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    field = np.where(inside, soil_loss, np.nan)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.88],
                          left=0.030, right=0.965, top=0.855, bottom=0.140,
                          wspace=0.30, hspace=0.46)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- erosion map -------------------------------------------------------
    cmap = ListedColormap(RISK)
    cmap.set_bad("#fbfbfb")
    norm = BoundaryNorm(BOUNDS, cmap.N)
    im = ax_m.imshow(field, origin="lower", extent=[0, W, 0, H], cmap=cmap,
                     norm=norm, interpolation="nearest", zorder=2)
    ax_m.add_patch(Polygon(outline, closed=True, facecolor="none",
                           edgecolor=RULE, lw=1.1, zorder=6))

    # drainage network from the same DEM that produced LS
    thr = 220.0
    seg, wid = [], []
    riv = (acc > thr) & inside
    ii, jj = np.nonzero(riv)
    for i, j in zip(ii, jj):
        k = best[i, j]
        if k < 0:
            continue
        di, dj = DIRS[k]
        if not (0 <= i + di < NY and 0 <= j + dj < NX):
            continue
        seg.append([(xs[j], ys[i]), (xs[j + dj], ys[i + di])])
        wid.append(0.25 + 0.95 * np.log10(acc[i, j] / thr + 1.0))
    ax_m.add_collection(LineCollection(seg, colors="#3d6b8e", linewidths=wid,
                                       zorder=5, alpha=0.95))

    ax_m.set_xlim(0, W)
    ax_m.set_ylim(0, H)
    ax_m.set_aspect("equal")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for s in ax_m.spines.values():
        s.set_visible(False)
    ax_m.set_title("Annual soil loss over the catchment", fontsize=9.0,
                   fontweight="bold", pad=6)
    add_scale_bar(ax_m, 5.0, "5 km", loc=(0.035, 0.045), height=0.016,
                  fontsize=6.4)
    ax_m.annotate("", xy=(0.955, 0.955), xytext=(0.955, 0.860),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0))
    halo_text(ax_m, 0.955, 0.965, "N", transform=ax_m.transAxes,
              ha="center", va="bottom", fontsize=7.0, color=INK,
              halo="white", halo_lw=1.8)

    cb = safe_colorbar(fig, im, ax_m,
                       label="annual soil loss (t ha$^{-1}$ a$^{-1}$)",
                       side="right", width=0.12, pad=0.08, shrink=0.86,
                       ticks=BOUNDS[:-1], labelsize=6.6)
    cb.ax.set_yticklabels([f"{b:.0f}" for b in BOUNDS[:-1]])
    for i, nm in enumerate(RISK_NAME):
        cb.ax.text(0.5, (i + 0.5) / len(RISK_NAME), nm, rotation=90,
                   transform=cb.ax.transAxes, ha="center", va="center",
                   fontsize=6.0, color="white" if i >= 3 else "#3a3a3a")

    vals = field[np.isfinite(field)]
    above = 100.0 * np.mean(vals > TOLERANCE)
    annotate_box(ax_m,
                 f"mean modulus  {vals.mean():.1f} t ha$^{{-1}}$ a$^{{-1}}$\n"
                 f"above tolerance ({TOLERANCE:.0f})  {above:.1f}% of area\n"
                 f"catchment area  {np.isfinite(field).sum() * CELL ** 2:.0f}"
                 " km$^{2}$",
                 loc="upper left", fontsize=6.6, margin=0.022)

    # ---- factor contributions ---------------------------------------------
    facs = [("R", r_fac), ("K", k_fac), ("LS", ls), ("C", c_fac),
            ("P", p_fac)]
    var = np.array([np.var(np.log10(f[inside])) for _, f in facs])
    share = 100.0 * var / var.sum()
    yy = np.arange(len(facs))[::-1]
    ax_f.barh(yy, share, height=0.62, color="#3b5b92", edgecolor="white",
              lw=0.6, zorder=3)
    for y, s, (nm, f) in zip(yy, share, facs):
        ax_f.text(s + 1.2, y, f"{s:.0f}%", ha="left", va="center",
                  fontsize=6.5, color=SUB)
    ax_f.set_yticks(yy)
    ax_f.set_yticklabels([nm for nm, _ in facs], fontsize=7.0)
    ax_f.set_xlim(0, share.max() * 1.28)
    ax_f.set_xlabel("share of log-factor variance (%)", fontsize=7.4)
    ax_f.tick_params(axis="x", labelsize=6.5)
    ax_f.grid(axis="x", lw=0.5, color="#e4e4e4", zorder=0)
    ax_f.set_title("Topography and cover drive the pattern", fontsize=8.5,
                   fontweight="bold")

    # ---- slope vs soil loss ------------------------------------------------
    sel = rng.choice(np.flatnonzero(inside.ravel()), size=1400, replace=False)
    sx = slope_pct.ravel()[sel]
    sy = soil_loss.ravel()[sel]
    cls = np.clip(np.digitize(sy, BOUNDS[1:-1]), 0, len(RISK) - 1)
    ax_s.scatter(sx, sy, s=5.0, c=[RISK[k] for k in cls], linewidths=0.15,
                 edgecolors="#8a8a8a", alpha=0.9, zorder=3)
    lx, ly = lowess(sx, np.log10(sy), frac=0.35, n_out=120)
    ax_s.plot(lx, 10.0 ** ly, color=RULE, lw=1.3, zorder=5)
    ax_s.axhline(TOLERANCE, color="#b23a48", lw=0.9, ls="--", zorder=4)
    ax_s.text(0.985, TOLERANCE * 1.25, "tolerable rate", ha="right",
              va="bottom", transform=ax_s.get_yaxis_transform(),
              fontsize=6.3, color="#b23a48")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("slope gradient (%)", fontsize=7.4)
    ax_s.set_ylabel("soil loss (t ha$^{-1}$ a$^{-1}$)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.5)
    ax_s.grid(lw=0.5, color="#ececec", zorder=0)
    rho = float(np.corrcoef(np.log10(sx + 1.0), np.log10(sy))[0, 1])
    ax_s.set_title(f"Slope controls the tail (r = {rho:.2f})", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color="#3d6b8e", lw=1.3, label="drainage network"),
        Line2D([0], [0], color=RULE, lw=1.1, label="catchment divide"),
        Line2D([0], [0], color=RULE, lw=1.3, ls="-",
               label="local regression of soil loss on slope")],
        loc="lower center", bbox_to_anchor=(0.5, 0.088), ncol=3,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=1.8)

    panel_labels_inches(fig, [ax_m, ax_f, ax_s], dx=-0.24, dy=0.14)


    print(f"mean={vals.mean():.2f}  median={np.median(vals):.2f}  "
          f"p95={np.percentile(vals, 95):.1f}  max={vals.max():.0f}")
    hist = np.histogram(vals, bins=BOUNDS)[0] / vals.size * 100.0
    print("class shares %:", dict(zip(RISK_NAME, hist.round(1))))
    print(f"above tolerance {above:.1f}%   factor shares "
          f"{dict(zip([n for n, _ in facs], share.round(1)))}")
    savefig(fig, "602_soil_erosion_risk.png")


if __name__ == "__main__":
    main()
