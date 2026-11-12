"""3D UAV path planning over synthetic mountainous terrain.

Hill-shaded terrain surface + five planner trajectories + translucent
cylindrical threat (radar / no-fly) zones + ground-track projections.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import splprep, splev, RegularGridInterpolator

from pubstyle import set_pub_style, savefig

# ---------------------------------------------------------------- environment
XMAX = YMAX = 10.0                      # km
PEAKS = [(2.2, 7.4, 1.30, 1.10, 980.0),
         (6.8, 6.2, 1.60, 1.30, 1250.0),
         (4.5, 2.6, 1.20, 1.50, 860.0),
         (8.4, 2.2, 1.00, 1.00, 700.0),
         (1.4, 3.0, 0.90, 1.20, 620.0),
         (7.0, 9.0, 1.10, 0.90, 540.0)]
# (cx, cy, radius_km, ceiling_m)
THREATS = [(3.2, 4.6, 1.00, 1300.0),
           (6.0, 4.0, 0.85, 1150.0),
           (5.2, 7.6, 1.10, 1250.0),
           (8.0, 6.6, 0.70, 1000.0)]
START = (0.6, 0.8)
GOAL = (9.4, 9.2)

ALGOS = {
    "ISOA":  dict(color="#b23a48", lw=2.6, ls="-",  z=30),
    "PSO":   dict(color="#1f6fb4", lw=1.5, ls="-",  z=26),
    "GWO":   dict(color="#2e8b57", lw=1.5, ls="-",  z=25),
    "RRT*":  dict(color="#7b3294", lw=1.4, ls=(0, (5, 2)), z=24),
    "A*":    dict(color="#e07b1f", lw=1.4, ls=(0, (1.2, 1.6)), z=23),
}

WAYPOINTS = {
    "ISOA": [(0.6, 0.8), (2.3, 2.2), (4.6, 4.3), (6.6, 7.1),
             (8.3, 8.5), (9.4, 9.2)],
    "PSO":  [(0.6, 0.8), (2.8, 1.6), (5.4, 2.4), (7.6, 3.6), (9.0, 5.6),
             (9.5, 7.8), (9.4, 9.2)],
    "GWO":  [(0.6, 0.8), (1.6, 2.6), (1.8, 4.8), (3.0, 6.8), (4.4, 8.8),
             (7.0, 9.4), (9.4, 9.2)],
    "RRT*": [(0.6, 0.8), (1.8, 1.4), (2.6, 3.0), (4.2, 3.0), (4.9, 4.9),
             (5.4, 6.0), (6.9, 6.4), (7.4, 8.2), (8.6, 8.6), (9.4, 9.2)],
    "A*":   [(0.6, 0.8), (2.4, 0.9), (4.2, 2.0), (4.6, 3.4), (4.8, 5.0),
             (4.2, 6.4), (3.6, 8.2), (5.6, 9.4), (7.6, 9.6), (9.4, 9.2)],
}
KIND = {"ISOA": "spline", "PSO": "spline", "GWO": "spline",
        "RRT*": "polyline", "A*": "polyline"}


def terrain_grid(n=181, seed=7):
    x = np.linspace(0, XMAX, n)
    y = np.linspace(0, YMAX, n)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, 90.0)
    for x0, y0, sx, sy, h in PEAKS:
        Z += h * np.exp(-(((X - x0) ** 2) / (2 * sx ** 2)
                          + ((Y - y0) ** 2) / (2 * sy ** 2)))
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(0.0, 1.0, X.shape), 4.0)
    noise /= np.abs(noise).max()
    Z += 140.0 * noise
    return x, y, X, Y, np.clip(Z, 0.0, None)


def make_interp(x, y, Z):
    f = RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=None)
    return lambda px, py: f(np.column_stack([np.atleast_1d(py),
                                             np.atleast_1d(px)]))


def build_ground_track(wp, kind="spline", n=420):
    wp = np.asarray(wp, float)
    if kind == "spline":
        tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0, k=3)
        u = np.linspace(0.0, 1.0, n)
        x, y = splev(u, tck)
        return np.asarray(x), np.asarray(y)
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(wp[:, 0]), np.diff(wp[:, 1])))]
    t = np.linspace(0.0, d[-1], n)
    x = np.interp(t, d, wp[:, 0])
    y = np.interp(t, d, wp[:, 1])
    x = gaussian_filter1d(x, 2.5, mode="nearest")
    y = gaussian_filter1d(y, 2.5, mode="nearest")
    x[0], y[0] = wp[0]
    x[-1], y[-1] = wp[-1]
    return x, y


def flight_altitude(x, y, interp, clearance=130.0, grad=300.0, sigma=7.0):
    """Terrain-following altitude with a bounded climb gradient (m per km)."""
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    terr = interp(x, y)
    req = terr + clearance
    z = np.max(req[None, :] - grad * np.abs(s[:, None] - s[None, :]), axis=1)
    for _ in range(80):
        z = np.maximum(gaussian_filter1d(z, sigma, mode="nearest"), req)
    return s, terr, z


def path_metrics(x, y, z, terr):
    d3 = np.hypot(np.hypot(np.diff(x), np.diff(y)) * 1000.0, np.diff(z))
    length = d3.sum() / 1000.0                       # km
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curv = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5,
                                                    1e-9)
    tmin = min(np.min(np.hypot(x - cx, y - cy)) - r for cx, cy, r, _ in THREATS)
    return dict(length=length, smooth=float(np.mean(curv)),
                clear=float(np.min(z - terr)), tmargin=float(tmin),
                zmax=float(z.max()))


# ------------------------------------------------------------------- drawing
def draw_cylinder(ax, cx, cy, r, top, base=0.0, color="#b23a48"):
    th = np.linspace(0, 2 * np.pi, 72)
    Zc, TH = np.meshgrid(np.array([base, top]), th, indexing="ij")
    ax.plot_surface(cx + r * np.cos(TH), cy + r * np.sin(TH), Zc,
                    color=color, alpha=0.14, shade=False, linewidth=0,
                    antialiased=True, zorder=12)
    for zz, aa, lww in ((top, 0.85, 0.9), (base, 0.55, 0.8)):
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), zz, color=color,
                lw=lww, alpha=aa, ls=(0, (3, 1.6)), zorder=13)


def land_cmap():
    """Muted hypsometric tint: lowland green -> khaki -> rock -> snow."""
    return LinearSegmentedColormap.from_list(
        "hypso", ["#4a7d54", "#7fa15c", "#b9b072", "#c9a06a", "#a87a55",
                  "#8e6f5e", "#c8c1bb", "#f2efeb"])


def main():
    set_pub_style(base_font=8)
    x, y, X, Y, Z = terrain_grid()
    interp = make_interp(x, y, Z)
    cmap = land_cmap()
    vmin, vmax = float(Z.min()) - 40.0, float(Z.max())

    fig = plt.figure(figsize=(7.5, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.005, -0.035, 0.885, 1.02])
    ax.computed_zorder = False

    dx = (XMAX / (len(x) - 1)) * 1000.0
    ls = LightSource(azdeg=315, altdeg=52)
    rgb = ls.shade(Z, cmap=cmap, blend_mode="soft", vert_exag=1.4,
                   dx=dx, dy=dx, vmin=vmin, vmax=vmax)
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=2)

    for cx, cy, r, top in THREATS:
        draw_cylinder(ax, cx, cy, r, top)

    stats = {}
    for name, wp in WAYPOINTS.items():
        st = ALGOS[name]
        px, py = build_ground_track(wp, KIND[name])
        s, terr, pz = flight_altitude(px, py, interp)
        stats[name] = path_metrics(px, py, pz, terr)
        gcol, glw, gal = ("#4d4d4d", 1.1, 0.75) if name == "ISOA" \
            else ("#8a8a8a", 0.7, 0.38)
        ax.plot(px, py, np.zeros_like(px), color=gcol, lw=glw,
                ls=(0, (4, 2)), alpha=gal, zorder=15)
        ax.plot(px, py, pz, color=st["color"], lw=st["lw"], ls=st["ls"],
                solid_capstyle="round", zorder=st["z"], label=name)

    sz = float(interp(*START)[0]) + 130.0
    gz = float(interp(*GOAL)[0]) + 130.0
    ax.plot([START[0]], [START[1]], [sz], marker="^", ms=9,
            mfc="#2e7d5b", mec="#123d12", mew=0.8, ls="none", zorder=40)
    ax.plot([GOAL[0]], [GOAL[1]], [gz], marker="*", ms=15,
            mfc="#b23a48", mec="#5c1113", mew=0.8, ls="none", zorder=40)
    ax.plot([START[0], START[0]], [START[1], START[1]], [0, sz],
            color="#2e7d5b", lw=0.7, alpha=0.6, zorder=16)
    ax.plot([GOAL[0], GOAL[0]], [GOAL[1], GOAL[1]], [0, gz],
            color="#b23a48", lw=0.7, alpha=0.6, zorder=16)
    ax.text(START[0] + 0.25, START[1] - 0.1, sz + 190, "S", fontsize=9,
            fontweight="bold", color="#1a6b1a", zorder=41)
    ax.text(GOAL[0] - 0.5, GOAL[1] + 0.2, gz + 210, "G", fontsize=9,
            fontweight="bold", color="#a11417", zorder=41)

    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_zlim(0, 1800)
    ax.set_box_aspect((1.0, 1.0, 0.46))
    ax.view_init(elev=40, azim=-55)
    ax.set_xlabel("East  x (km)", labelpad=1)
    ax.set_ylabel("North  y (km)", labelpad=1)
    ax.set_zlabel("Altitude (m)", labelpad=-2)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_zticks([0, 400, 800, 1200, 1600])
    ax.tick_params(axis="both", pad=-1.5, labelsize=7)
    ax.tick_params(axis="z", pad=0.5, labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0))
        pane.pane.set_edgecolor("#bbbbbb")
        pane.pane.set_linewidth(0.5)
    ax.grid(True)
    ax.set_title("UAV 3D trajectory planning over mountainous terrain "
                 "with radar threat zones", fontsize=9.5, pad=-6)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.935, 0.30, 0.017, 0.36])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Terrain elevation (m)", fontsize=7.5, labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.2, color="#333333")

    handles = [Line2D([], [], color=ALGOS[k]["color"], lw=ALGOS[k]["lw"],
                      ls=ALGOS[k]["ls"], label=k) for k in ALGOS]
    handles += [
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Start S"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Goal G"),
        Patch(facecolor="#b23a48", alpha=0.22, edgecolor="#b23a48",
              linewidth=0.8, label="Threat cylinder"),
        Line2D([], [], color="#4d4d4d", lw=1.0, ls=(0, (4, 2)),
               label="Ground track (z = 0)"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts()[:1]:
        t.set_fontweight("bold")

    lh, top = 0.0235, 0.175
    fig.patches.append(plt.Rectangle(
        (0.745, top - 6 * lh - 0.012), 0.246, 6 * lh + 0.022,
        transform=fig.transFigure, facecolor="white", edgecolor="#cccccc",
        linewidth=0.6, alpha=0.95, zorder=4))
    fig.text(0.982, top, "Planner   L3D(km)  Cmin(m)  Tmar(m)", fontsize=6.3,
             family="monospace", va="top", ha="right", color="#333333",
             zorder=5)
    for i, (k, m) in enumerate(stats.items(), start=1):
        hot = k == "ISOA"
        fig.text(0.982, top - i * lh,
                 f"{k:<8s} {m['length']:6.2f}   {m['clear']:5.0f}   "
                 f"{m['tmargin']*1000:6.0f}", fontsize=6.3,
                 family="monospace", va="top", ha="right", zorder=5,
                 color="#b23a48" if hot else "#1a1a1a",
                 fontweight="bold" if hot else "normal")

    for k, m in stats.items():
        print(f"{k:6s} len={m['length']:.2f}km  clear={m['clear']:.0f}m  "
              f"threat_margin={m['tmargin']*1000:.0f}m  zmax={m['zmax']:.0f}m")

    savefig(fig, "153_uav_path_3d.png")


if __name__ == "__main__":
    main()
