"""Airborne lidar over a 32 x 16 m transect: cloud, CHM, profile, segments.

One synthetic point cloud is generated and then never re-invented.  Forty-odd
trees are dropped on an undulating ground surface with crown shells sampled
around each stem; ground returns are thinned where the canopy is closed, which
is what produces the shadowing a real flight shows.  Height is normalised
against the terrain at every point, so "height" means height above ground in
all four views and one shared colour norm serves the cloud, the canopy height
model, the crown map and the colorbar.

The canopy height model is the per-cell maximum of that cloud, the vertical
profile is the same points binned in half-metre layers with its three modes
located by peak detection rather than by eye, and the segmentation map shows
the crowns those returns came from.  Camera is fixed at elev 18, azim -62 and
the 3D background panes are removed.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import add_scale_bar, axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GROUND = "#8a7f6a"

FW, FH = 10.4, 6.4
LX, LY = 32.0, 16.0
ELEV, AZIM = 18.0, -62.0
CELL = 1.0
BIN = 0.5
N_TREE = 26
MIN_SEP = 3.4
CMAP = get_cmap("cb_sequential_green")


def terrain(x, y):
    return (0.95 * np.sin(x / 9.0) + 0.60 * np.cos(y / 6.0 + 0.7)
            + 0.35 * np.sin(x / 3.5 + y / 4.2))


def place_trees(rng):
    xs, ys, hs = [], [], []
    for _ in range(6000):
        if len(xs) >= N_TREE:
            break
        px, py = rng.uniform(0.9, LX - 0.9), rng.uniform(0.9, LY - 0.9)
        if any((px - a) ** 2 + (py - b) ** 2 < MIN_SEP ** 2
               for a, b in zip(xs, ys)):
            continue
        xs.append(px)
        ys.append(py)
        hs.append(6.0 + 21.0 * rng.beta(2.3, 2.0))
    x, y, h = np.array(xs), np.array(ys), np.array(hs)
    return dict(x=x, y=y, h=h, r=0.78 + 0.104 * h,
                base=h * rng.uniform(0.36, 0.56, h.size))


def canopy_cover(trees, px, py):
    cov = np.zeros(px.size, dtype=bool)
    for tx, ty, tr in zip(trees["x"], trees["y"], trees["r"]):
        cov |= (px - tx) ** 2 + (py - ty) ** 2 <= tr * tr
    return cov


def build_cloud(rng, trees):
    px, py, ph = [], [], []

    for tx, ty, th, tr, tb in zip(trees["x"], trees["y"], trees["h"],
                                  trees["r"], trees["base"]):
        n = int(30 + 11 * th)
        # shell-weighted radius: a lidar pulse mostly hits the crown surface
        u = rng.random(n) ** (1.0 / 2.7)
        cost = rng.uniform(-1.0, 1.0, n)
        sint = np.sqrt(np.clip(1.0 - cost ** 2, 0.0, None))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        zc, zr = 0.5 * (tb + th), 0.5 * (th - tb)
        px.append(tx + tr * u * sint * np.cos(phi))
        py.append(ty + tr * u * sint * np.sin(phi))
        ph.append(zc + zr * u * cost + rng.normal(0.0, 0.10, n))
        # stem returns
        m = 14
        px.append(tx + rng.normal(0.0, 0.10, m))
        py.append(ty + rng.normal(0.0, 0.10, m))
        ph.append(rng.uniform(0.4, tb, m))

    gx, gy = rng.uniform(0.0, LX, 9000), rng.uniform(0.0, LY, 9000)
    keep = rng.random(gx.size) < np.where(canopy_cover(trees, gx, gy),
                                          0.12, 0.55)
    px.append(gx[keep])
    py.append(gy[keep])
    ph.append(np.abs(rng.normal(0.0, 0.06, int(keep.sum()))))

    # shrub layer: a distinct stratum, not a tail of the ground return
    ux, uy = rng.uniform(0.0, LX, 6400), rng.uniform(0.0, LY, 6400)
    keep = rng.random(ux.size) < np.where(canopy_cover(trees, ux, uy),
                                          0.34, 0.64)
    ux, uy = ux[keep], uy[keep]
    px.append(ux)
    py.append(uy)
    ph.append(np.clip(rng.normal(2.9, 0.95, ux.size), 0.4, 6.5))

    x = np.concatenate(px)
    y = np.concatenate(py)
    hn = np.concatenate(ph)
    ok = (x >= 0.0) & (x <= LX) & (y >= 0.0) & (y <= LY) & (hn >= 0.0)
    x, y, hn = x[ok], y[ok], hn[ok]
    return x, y, hn, terrain(x, y) + hn


def build_chm(x, y, hn):
    nx, ny = int(LX / CELL), int(LY / CELL)
    ix = np.clip((x / CELL).astype(int), 0, nx - 1)
    iy = np.clip((y / CELL).astype(int), 0, ny - 1)
    chm = np.zeros((ny, nx))
    np.maximum.at(chm, (iy, ix), hn)
    return gaussian_filter(chm, 0.55)


def draw_cloud(ax, x, y, z, hn, chm, norm):
    ax.view_init(elev=ELEV, azim=AZIM)
    # zoom fills the axes box; without it a 3D projection leaves a wide
    # margin of nothing on every side
    ax.set_box_aspect((LX, LY, 0.62 * LX), zoom=1.30)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="#e4e4e4", linewidth=0.45)

    zt = terrain(x, y)
    ax.scatter(x, y, z, c=hn, cmap=CMAP, norm=norm, s=1.1, alpha=0.85,
               linewidths=0.0, depthshade=False, zorder=3)

    nx, ny = chm.shape[1], chm.shape[0]
    cx = (np.arange(nx) + 0.5) * CELL
    cy = (np.arange(ny) + 0.5) * CELL
    CX, CY = np.meshgrid(cx, cy)
    CZ = terrain(CX, CY) + chm
    ax.plot_wireframe(CX, CY, CZ, rstride=1, cstride=1, linewidth=0.35,
                      color="#2e7d5b", alpha=0.55, zorder=5)

    ax.set_xlim(0.0, LX)
    ax.set_ylim(0.0, LY)
    ax.set_zlim(float(zt.min()) - 0.5, float(z.max()) + 1.0)
    ax.set_xlabel("transect distance (m)", fontsize=7.6, labelpad=0.0)
    ax.set_ylabel("across-track (m)", fontsize=7.6, labelpad=-2.0)
    ax.set_zlabel("elevation (m)", fontsize=7.6, labelpad=-6.0)
    ax.tick_params(labelsize=6.4, pad=-1.0)
    ax.set_title("Point cloud with the canopy height model as a wireframe",
                 fontsize=8.5, fontweight="bold", y=0.955)


def draw_profile(ax, hn, norm):
    edges = np.arange(0.0, hn.max() + BIN, BIN)
    counts, _ = np.histogram(hn, bins=edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    pct = 100.0 * counts / counts.sum()
    smooth = gaussian_filter1d(pct, 1.1)

    ax.fill_betweenx(mid, 0.0, pct, color="#cfe1ee", linewidth=0.0, zorder=3)
    ax.plot(smooth, mid, color="#2e74a8", lw=1.2, zorder=5)
    # pad with a zero so the ground return, which sits in the first bin, is
    # not disqualified as a boundary sample
    padded = np.concatenate([[0.0], smooth])
    peaks = np.array([], dtype=int)
    for prom in (0.35, 0.20, 0.10, 0.05):
        found, _ = find_peaks(padded, prominence=prom, distance=3)
        peaks = np.clip(found - 1, 0, smooth.size - 1)
        if peaks.size >= 3:
            break
    order = peaks[np.argsort(smooth[peaks])[::-1]][:3]
    for p in np.sort(order):
        layer = ("ground" if mid[p] < 1.5
                 else "understorey" if mid[p] < 6.0 else "canopy")
        ax.plot([smooth[p]], [mid[p]], marker="o", ms=3.4, mfc="#b23a48",
                mec="white", mew=0.5, zorder=6)
        dy = 0.95 if mid[p] < 1.0 else 0.0      # keep the ground label clear
        ax.text(smooth[p] + 0.7, mid[p] + dy, f"{layer}  {mid[p]:.1f} m",
                ha="left", va="center", fontsize=6.3, color=SUB, zorder=6)
    ax.set_ylim(0.0, hn.max() * 1.03)
    ax.set_xlim(0.0, pct.max() * 1.45)
    ax.set_xlabel("returns per 0.5 m layer (%)", fontsize=7.8)
    ax.set_ylabel("height above ground (m)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Vertical return profile is trimodal", fontsize=8.5,
                 fontweight="bold")
    return mid, smooth, np.sort(order)


def draw_segments(ax, trees, norm):
    ax.set_xlim(0.0, LX)
    ax.set_ylim(0.0, LY)
    ax.set_aspect("equal")
    ax.add_patch(mpl.patches.Rectangle((0, 0), LX, LY, facecolor="#f7f7f4",
                                       edgecolor="none", zorder=0))
    for tx, ty, th, tr in zip(trees["x"], trees["y"], trees["h"],
                              trees["r"]):
        ax.add_patch(mpl.patches.Circle(
            (tx, ty), tr, facecolor=CMAP(norm(th)), edgecolor="white",
            linewidth=0.5, alpha=0.88, zorder=3))
    ax.scatter(trees["x"], trees["y"], s=2.2, color="#3a3a3a", zorder=5)
    add_scale_bar(ax, 5.0, "5 m", loc=(0.035, 0.055), height=0.022,
                  fontsize=6.0, color="#2b2b2b", text_color=INK,
                  halo="white")
    ax.set_xlabel("transect distance (m)", fontsize=7.8)
    ax.set_ylabel("across-track (m)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(RULE)
    ax.set_title("Individual tree segmentation, plan view", fontsize=8.5,
                 fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(689)
    trees = place_trees(rng)
    x, y, hn, z = build_cloud(rng, trees)
    chm = build_chm(x, y, hn)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(max(hn.max(),
                                                         trees["h"].max())))

    fig = plt.figure(figsize=(FW, FH))
    ax3 = fig.add_axes([0.44 / FW, 1.42 / FH, 5.92 / FW, 3.84 / FH],
                       projection="3d")
    ax_p = axes_inches(fig, (7.40, 3.24, 2.64, 1.90))
    ax_s = axes_inches(fig, (7.40, 1.02, 2.64, 1.52))
    cax = axes_inches(fig, (1.75, 0.96, 3.20, 0.13))

    draw_cloud(ax3, x, y, z, hn, chm, norm)
    mid, prof, peaks = draw_profile(ax_p, hn, norm)
    draw_segments(ax_s, trees, norm)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("height above ground (m)", fontsize=7.0, color=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color=RULE)

    panel_labels_inches(fig, [ax3, ax_p, ax_s], dx=-0.42, dy=0.06)

    area_ha = LX * LY / 10000.0
    cover = float(np.mean(chm > 10.0))
    top = float(np.percentile(chm, 95))

    print(f"points={x.size} trees={trees['x'].size} "
          f"density={trees['x'].size / area_ha:.0f}/ha cover={cover:.3f} "
          f"p95={top:.2f} peaks={[round(float(mid[p]), 2) for p in peaks]}")
    savefig(fig, "689_for_lidar_canopy.png")


if __name__ == "__main__":
    main()
