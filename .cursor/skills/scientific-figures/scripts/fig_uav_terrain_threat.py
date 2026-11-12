"""Environment modelling for UAV path planning: from terrain to cost map.

(a) hill-shaded 3-D terrain, (b) superposed radar threat field,
(c) weighted composite cost map, (d) feasible mask + dilated safe corridor.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (LightSource, LinearSegmentedColormap,
                               ListedColormap, BoundaryNorm)
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy.ndimage import (gaussian_filter, gaussian_filter1d,
                           binary_dilation, binary_closing)
from scipy.interpolate import splprep, splev

from pubstyle import set_pub_style, savefig, add_panel_label

XMAX = YMAX = 10.0
PEAKS = [(2.2, 7.4, 1.30, 1.10, 980.0), (6.8, 6.2, 1.60, 1.30, 1250.0),
         (4.5, 2.6, 1.20, 1.50, 860.0), (8.4, 2.2, 1.00, 1.00, 700.0),
         (1.4, 3.0, 0.90, 1.20, 620.0), (7.0, 9.0, 1.10, 0.90, 540.0)]
THREATS = [(3.2, 4.6, 1.00, 1.00), (6.0, 4.0, 0.85, 0.85),
           (5.2, 7.6, 1.10, 0.95), (8.0, 6.6, 0.70, 0.70)]
START, GOAL = (0.6, 0.8), (9.4, 9.2)
ISOA_WP = [(0.6, 0.8), (2.3, 2.2), (4.6, 4.3), (6.6, 7.1), (8.3, 8.5),
           (9.4, 9.2)]
N = 221


def terrain_grid(n=N, seed=7):
    x = np.linspace(0, XMAX, n)
    y = np.linspace(0, YMAX, n)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, 90.0)
    for x0, y0, sx, sy, h in PEAKS:
        Z += h * np.exp(-(((X - x0) ** 2) / (2 * sx ** 2)
                          + ((Y - y0) ** 2) / (2 * sy ** 2)))
    rng = np.random.default_rng(seed)
    nz = gaussian_filter(rng.normal(0.0, 1.0, X.shape), 4.0)
    Z += 140.0 * nz / np.abs(nz).max()
    return X, Y, np.clip(Z, 0.0, None)


def threat_field(X, Y):
    """Superposed radar detection probability, saturating at 1."""
    T = np.zeros_like(X)
    for cx, cy, r, w in THREATS:
        d = np.hypot(X - cx, Y - cy)
        T += w * np.exp(-(d / (0.85 * r)) ** 2)
    return np.clip(T, 0.0, 1.0)


def isoa_track(n=420):
    wp = np.asarray(ISOA_WP, float)
    tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0, k=3)
    u = np.linspace(0, 1, n)
    xy = splev(u, tck)
    return np.asarray(xy[0]), np.asarray(xy[1])


def land_cmap():
    return LinearSegmentedColormap.from_list(
        "hypso", ["#4a7d54", "#7fa15c", "#b9b072", "#c9a06a", "#a87a55",
                  "#8e6f5e", "#c8c1bb", "#f2efeb"])


def cbar(fig, mappable, ax, label, ticks=None):
    # frame() puts these axes on set_aspect("equal"), which shrinks the axes
    # box at draw time. fig.colorbar(ax=...) sizes itself from the box before
    # that shrink and ends up stranded to the right of the map; an inset in
    # axes coordinates tracks the real position instead.
    cax = ax.inset_axes([1.035, 0.025, 0.040, 0.95])
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.set_label(label, fontsize=6.8)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2, color="#333333")
    return cb


def frame(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_aspect("equal")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))


def main():
    set_pub_style(base_font=8)
    X, Y, Z = terrain_grid()
    T = threat_field(X, Y)
    px, py = isoa_track()

    gy, gx = np.gradient(Z, YMAX / (N - 1), XMAX / (N - 1))
    slope = np.hypot(gx, gy)                      # m per km
    Zn = (Z - Z.min()) / np.ptp(Z)
    Sn = np.clip(slope / np.percentile(slope, 98), 0, 1)
    J = 0.34 * Zn + 0.46 * T + 0.20 * Sn          # composite cost, 0..1

    fig = plt.figure(figsize=(7.2, 6.3))
    cmap = land_cmap()

    # ---- (a) 3-D terrain ---------------------------------------------------
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.set_position([-0.005, 0.475, 0.50, 0.45])
    dx = (XMAX / (N - 1)) * 1000.0
    rgb = LightSource(azdeg=315, altdeg=52).shade(
        Z, cmap=cmap, blend_mode="soft", vert_exag=1.4, dx=dx, dy=dx)
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=2, cstride=2,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_zlim(0, 1500)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=42, azim=-58)
    ax.set_xlabel("x (km)", labelpad=-6, fontsize=7)
    ax.set_ylabel("y (km)", labelpad=-6, fontsize=7)
    ax.set_zlabel("h (m)", labelpad=-8, fontsize=7)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([0, 700, 1400])
    ax.tick_params(labelsize=6, pad=-3)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor("#cccccc")
        a.pane.set_linewidth(0.5)
    ax.text2D(0.52, 0.905, "Digital elevation model  $h(x,y)$", ha="center",
              transform=ax.transAxes, fontsize=8, va="bottom")
    ax.text2D(0.05, 0.895, "(a)", transform=ax.transAxes, fontsize=10,
              fontweight="bold", va="bottom")
    ax.text2D(0.60, 0.10, f"$h_{{max}}$ = {Z.max():.0f} m",
              transform=ax.transAxes, fontsize=6.5, color="#333333")

    # ---- (b) threat field --------------------------------------------------
    axb = fig.add_subplot(2, 2, 2)
    axb.set_position([0.575, 0.545, 0.335, 0.335])
    tc = LinearSegmentedColormap.from_list(
        "threat", ["#fdfbf7", "#fbe3c8", "#f5a97c", "#e2603f", "#a11417"])
    im = axb.pcolormesh(X, Y, T, cmap=tc, shading="gouraud", vmin=0, vmax=1)
    axb.contour(X, Y, Z, levels=10, colors="#9aa39a", linewidths=0.3,
                alpha=0.55, zorder=2)
    axb.contour(X, Y, T, levels=[0.1, 0.3, 0.6, 0.9], colors="#5c1113",
                linewidths=0.4, alpha=0.6, zorder=3)
    for cx, cy, r, _ in THREATS:
        axb.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#333333",
                             lw=0.8, ls=(0, (3, 1.6))))
        axb.plot([cx], [cy], marker="x", ms=3.4, mew=0.9, color="#333333")
    frame(axb)
    axb.set_title("Superposed radar threat field  $P_d(x,y)$", fontsize=8)
    axb.set_xlabel("x (km)", labelpad=1)
    axb.set_ylabel("y (km)", labelpad=1)
    add_panel_label(axb, "(b)", x=-0.16, y=1.02, size=10)
    cbar(fig, im, axb, "Detection probability  $P_d$ (-)",
         ticks=[0, 0.25, 0.5, 0.75, 1.0])

    # ---- (c) composite cost ------------------------------------------------
    axc = fig.add_subplot(2, 2, 3)
    axc.set_position([0.075, 0.105, 0.335, 0.335])
    jc = LinearSegmentedColormap.from_list(
        "cost", ["#0d2c54", "#2e6f9e", "#8fc0d9", "#f7f3e8", "#f2b56b",
                 "#d1663a", "#7a1f12"])
    imc = axc.pcolormesh(X, Y, J, cmap=jc, shading="gouraud",
                         vmin=0, vmax=float(J.max()))
    axc.contour(X, Y, J, levels=6, colors="#333333", linewidths=0.3,
                alpha=0.45)
    axc.plot(px, py, color="#b23a48", lw=2.0, zorder=6,
             label="ISOA minimum-cost track")
    frame(axc)
    axc.set_title("Weighted composite cost map", fontsize=8)
    axc.text(0.035, 0.965, "$J=0.34\\,\\hat{h}+0.46\\,P_d+0.20\\,\\hat{s}$",
             transform=axc.transAxes, ha="left", va="top", fontsize=6.8,
             zorder=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.6, alpha=0.93))
    axc.set_xlabel("x (km)", labelpad=1)
    axc.set_ylabel("y (km)", labelpad=1)
    add_panel_label(axc, "(c)", x=-0.16, y=1.02, size=10)
    cbar(fig, imc, axc, "Normalised cost  $J$ (-)")

    # ---- (d) feasible mask + corridor -------------------------------------
    axd = fig.add_subplot(2, 2, 4)
    axd.set_position([0.575, 0.105, 0.335, 0.335])
    thr = 0.52
    feasible = J < thr
    feasible = binary_closing(feasible, np.ones((5, 5)))
    core = ~binary_dilation(~feasible, np.ones((7, 7)))     # eroded safe set
    band = np.zeros_like(J)
    band[feasible] = 1.0
    band[core] = 2.0
    mc = ListedColormap(["#b9bfc7", "#e3ecdf", "#7fb08a"])
    axd.pcolormesh(X, Y, band, cmap=mc, shading="nearest",
                   norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    axd.contour(X, Y, band, levels=[0.5, 1.5], colors="#33553f",
                linewidths=0.55)
    for cx, cy, r, _ in THREATS:
        axd.add_patch(Circle((cx, cy), r, facecolor="none",
                             edgecolor="#a11417", lw=0.8, ls=(0, (3, 1.6)),
                             zorder=5))
    axd.plot(px, py, color="#b23a48", lw=2.0, zorder=6)
    axd.plot([START[0]], [START[1]], marker="^", ms=7, mfc="#2e7d5b",
             mec="#123d12", mew=0.8, ls="none", zorder=7)
    axd.plot([GOAL[0]], [GOAL[1]], marker="*", ms=12, mfc="#b23a48",
             mec="#5c1113", mew=0.8, ls="none", zorder=7)
    frame(axd)
    axd.set_title("Feasible set and dilated safe corridor", fontsize=8)
    axd.set_xlabel("x (km)", labelpad=1)
    axd.set_ylabel("y (km)", labelpad=1)
    add_panel_label(axd, "(d)", x=-0.16, y=1.02, size=10)

    frac_f = 100.0 * feasible.mean()
    frac_c = 100.0 * core.mean()
    handles = [
        Patch(facecolor="#b9bfc7", edgecolor="#7d848c", lw=0.5,
              label=f"Infeasible  $J\\geq{thr:.2f}$ ({100-frac_f:.0f} %)"),
        Patch(facecolor="#e3ecdf", edgecolor="#33553f", lw=0.5,
              label=f"Feasible ({frac_f:.0f} %)"),
        Patch(facecolor="#7fb08a", edgecolor="#33553f", lw=0.5,
              label=f"Safe corridor, 0.3 km erosion ({frac_c:.0f} %)"),
        Line2D([], [], color="#b23a48", lw=2.0, label="ISOA track"),
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Start S"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Goal G"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.002), frameon=False, fontsize=6.9,
               handlelength=1.7, columnspacing=1.4)

    iy = np.round(py / YMAX * (N - 1)).astype(int)
    ix = np.round(px / XMAX * (N - 1)).astype(int)
    on = J[iy, ix]
    print(f"feasible={frac_f:.1f}%  corridor={frac_c:.1f}%  "
          f"ISOA mean J={on.mean():.3f} max J={on.max():.3f} "
          f"(field mean {J.mean():.3f})  "
          f"track in feasible={100*feasible[iy, ix].mean():.1f}%  "
          f"in corridor={100*core[iy, ix].mean():.1f}%")
    savefig(fig, "155_uav_terrain_threat.png")


if __name__ == "__main__":
    main()
