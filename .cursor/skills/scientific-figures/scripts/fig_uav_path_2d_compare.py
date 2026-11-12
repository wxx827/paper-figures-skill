"""Top-down comparison of five UAV path planners in two scenarios.

Terrain contours + circular radar/no-fly threats + five planner tracks,
with a per-scenario metric table (length / threat cost / smoothness).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import splprep, splev

from pubstyle import set_pub_style, savefig, add_panel_label

XMAX, YMAX = 10.0, 10.0
YTOP = 12.0        # northern lowland band reserved for the metric table

ALGOS = {
    "ISOA": dict(color="#b23a48", lw=2.3, ls="-", zo=30),
    "PSO":  dict(color="#1f6fb4", lw=1.3, ls="-", zo=26),
    "GWO":  dict(color="#2e8b57", lw=1.3, ls="-", zo=25),
    "RRT*": dict(color="#7b3294", lw=1.2, ls=(0, (4.5, 2)), zo=24),
    "A*":   dict(color="#e07b1f", lw=1.2, ls=(0, (1.2, 1.5)), zo=23),
}

SCEN = {
    "a": dict(
        title="Scenario I  \u2014  sparse threats, high relief",
        peaks=[(2.2, 7.4, 1.30, 1.10, 980.0), (6.8, 6.2, 1.60, 1.30, 1250.0),
               (4.5, 2.6, 1.20, 1.50, 860.0), (8.4, 2.2, 1.00, 1.00, 700.0),
               (1.4, 3.0, 0.90, 1.20, 620.0), (7.0, 9.0, 1.10, 0.90, 540.0)],
        threats=[(3.2, 4.6, 1.00), (6.0, 4.0, 0.85), (5.2, 7.6, 1.10),
                 (8.0, 6.6, 0.70)],
        start=(0.6, 0.8), goal=(9.4, 9.2), seed=7,
        wp={"ISOA": [(0.6, 0.8), (2.3, 2.2), (4.6, 4.3), (6.6, 7.1),
                     (8.3, 8.5), (9.4, 9.2)],
            "PSO":  [(0.6, 0.8), (2.8, 1.6), (5.4, 2.4), (7.6, 3.6),
                     (9.0, 5.6), (9.5, 7.8), (9.4, 9.2)],
            "GWO":  [(0.6, 0.8), (1.6, 2.6), (1.8, 4.8), (3.0, 6.8),
                     (4.4, 8.8), (7.0, 9.4), (9.4, 9.2)],
            "RRT*": [(0.6, 0.8), (1.8, 1.4), (2.6, 3.0), (4.2, 3.0),
                     (4.9, 4.9), (5.4, 6.0), (6.9, 6.4), (7.4, 8.2),
                     (8.6, 8.6), (9.4, 9.2)],
            "A*":   [(0.6, 0.8), (2.4, 0.9), (4.2, 2.0), (4.6, 3.4),
                     (4.8, 5.0), (4.2, 6.4), (3.6, 8.2), (5.6, 9.4),
                     (7.6, 9.6), (9.4, 9.2)]}),
    "b": dict(
        title="Scenario II  \u2014  cluttered corridor, 8 threats",
        peaks=[(1.8, 4.4, 1.10, 1.00, 640.0), (5.0, 1.8, 1.30, 0.90, 720.0),
               (8.2, 5.0, 1.20, 1.40, 900.0), (3.4, 8.6, 1.00, 0.90, 680.0),
               (6.4, 7.4, 0.90, 1.10, 780.0), (9.0, 1.6, 0.90, 0.90, 560.0)],
        threats=[(2.0, 2.6, 0.62), (3.6, 5.0, 0.75), (2.4, 7.6, 0.60),
                 (5.2, 3.0, 0.70), (5.6, 6.4, 0.80), (7.4, 4.6, 0.65),
                 (7.8, 8.0, 0.70), (4.6, 8.8, 0.55)],
        start=(0.5, 0.5), goal=(9.5, 9.5), seed=19,
        wp={"ISOA": [(0.5, 0.5), (2.6, 1.4), (4.4, 4.0), (6.6, 5.6),
                     (8.6, 6.4), (9.4, 8.4), (9.5, 9.5)],
            "PSO":  [(0.5, 0.5), (3.2, 1.0), (6.2, 2.2), (8.6, 3.2),
                     (9.2, 6.0), (8.8, 7.4), (9.5, 9.5)],
            "GWO":  [(0.5, 0.5), (0.9, 3.2), (1.6, 6.2), (1.6, 8.8),
                     (4.0, 9.6), (6.6, 9.2), (9.5, 9.5)],
            "RRT*": [(0.5, 0.5), (1.6, 1.8), (3.0, 3.6), (4.4, 4.0),
                     (4.6, 5.8), (5.0, 7.4), (6.6, 8.4), (8.4, 9.0),
                     (9.5, 9.5)],
            "A*":   [(0.5, 0.5), (2.8, 0.8), (4.4, 2.0), (6.2, 3.8),
                     (6.6, 5.4), (7.0, 6.8), (6.4, 8.6), (8.2, 9.4),
                     (9.5, 9.5)]}),
}
MARGIN = {"ISOA": 0.30, "PSO": 0.16, "GWO": 0.18, "RRT*": 0.13, "A*": 0.12}
BAND = {"ISOA": 3.2, "PSO": 2.4, "GWO": 2.4, "RRT*": 1.1, "A*": 1.4}
KIND = {"ISOA": "spline", "PSO": "spline", "GWO": "spline",
        "RRT*": "poly", "A*": "poly"}


def terrain_grid(peaks, seed, n=161):
    x = np.linspace(0, XMAX, n)
    y = np.linspace(0, YTOP, int(n * YTOP / XMAX))
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, 90.0)
    for x0, y0, sx, sy, h in peaks:
        Z += h * np.exp(-(((X - x0) ** 2) / (2 * sx ** 2)
                          + ((Y - y0) ** 2) / (2 * sy ** 2)))
    rng = np.random.default_rng(seed)
    nz = gaussian_filter(rng.normal(0.0, 1.0, X.shape), 4.0)
    Z += 130.0 * nz / np.abs(nz).max()
    return X, Y, np.clip(Z, 0.0, None)


def raw_track(wp, kind="spline", n=420):
    wp = np.asarray(wp, float)
    if kind == "spline":
        tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0, k=3)
        u = np.linspace(0.0, 1.0, n)
        return np.asarray(splev(u, tck)[0]), np.asarray(splev(u, tck)[1])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(wp[:, 0]), np.diff(wp[:, 1])))]
    t = np.linspace(0.0, d[-1], n)
    x = gaussian_filter1d(np.interp(t, d, wp[:, 0]), 2.2, mode="nearest")
    y = gaussian_filter1d(np.interp(t, d, wp[:, 1]), 2.2, mode="nearest")
    x[0], y[0], x[-1], y[-1] = wp[0, 0], wp[0, 1], wp[-1, 0], wp[-1, 1]
    return x, y


def enforce_clearance(x, y, threats, margin=0.15, sigma=2.0, iters=250):
    """Elastic-band relaxation: push out of threats, re-smooth, repeat."""
    x, y = x.copy(), y.copy()
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    for _ in range(iters):
        hit = False
        for cx, cy, r in threats:
            d = np.hypot(x - cx, y - cy)
            need = r + margin
            bad = d < need
            if bad.any():
                hit = True
                dd = np.maximum(d[bad], 1e-6)
                x[bad] = cx + (x[bad] - cx) / dd * need
                y[bad] = cy + (y[bad] - cy) / dd * need
        xs = gaussian_filter1d(x, sigma, mode="nearest")
        ys = gaussian_filter1d(y, sigma, mode="nearest")
        xs[0], ys[0], xs[-1], ys[-1] = x0, y0, x1, y1
        x, y = xs, ys
        if not hit:
            break
    for cx, cy, r in threats:                      # hard final projection
        d = np.hypot(x - cx, y - cy)
        need = r + 0.35 * margin
        bad = d < need
        if bad.any():
            dd = np.maximum(d[bad], 1e-6)
            x[bad] = cx + (x[bad] - cx) / dd * need
            y[bad] = cy + (y[bad] - cy) / dd * need
    x = gaussian_filter1d(x, 0.9, mode="nearest")
    y = gaussian_filter1d(y, 0.9, mode="nearest")
    x[0], y[0], x[-1], y[-1] = x0, y0, x1, y1
    return x, y


def metrics(x, y, threats):
    ds = np.hypot(np.diff(x), np.diff(y))
    length = ds.sum()
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curv = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5,
                                                    1e-9)
    tc = 0.0
    margin = 9.9
    for cx, cy, r in threats:
        d = np.hypot(x - cx, y - cy)
        tc += float(np.sum(np.exp(-((d[:-1] / (1.35 * r)) ** 2)) * ds) * 10.0)
        margin = min(margin, float(d.min() - r))
    return dict(L=length, TC=tc, K=float(np.mean(curv)), M=margin)


def draw_scenario(ax, sc, tag):
    X, Y, Z = terrain_grid(sc["peaks"], sc["seed"])
    cmap = LinearSegmentedColormap.from_list(
        "light_land", ["#eef4ea", "#dfeada", "#e8e6c9", "#e6d8bd",
                       "#dcc8b4", "#e9e4de"])
    ax.contourf(X, Y, Z, levels=14, cmap=cmap, zorder=0)
    ax.contour(X, Y, Z, levels=14, colors="#a9a9a9", linewidths=0.35,
               alpha=0.75, zorder=1)

    for cx, cy, r in sc["threats"]:
        ax.add_patch(Circle((cx, cy), r, facecolor="#b23a48", alpha=0.17,
                            edgecolor="none", zorder=2))
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#b23a48",
                            lw=0.85, ls=(0, (3, 1.6)), zorder=3))
        ax.plot([cx], [cy], marker="x", ms=3.2, mew=0.9, color="#a11417",
                zorder=3)

    res = {}
    for name in ALGOS:
        x, y = raw_track(sc["wp"][name], KIND[name])
        x, y = enforce_clearance(x, y, sc["threats"], margin=MARGIN[name],
                                 sigma=BAND[name])
        res[name] = metrics(x, y, sc["threats"])
        st = ALGOS[name]
        ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"],
                solid_capstyle="round", zorder=st["zo"], label=name)

    sx, sy = sc["start"]
    gx, gy = sc["goal"]
    ax.plot([sx], [sy], marker="^", ms=8, mfc="#2e7d5b", mec="#123d12",
            mew=0.8, ls="none", zorder=40)
    ax.plot([gx], [gy], marker="*", ms=13, mfc="#b23a48", mec="#5c1113",
            mew=0.8, ls="none", zorder=40)
    ax.annotate("S", (sx, sy), xytext=(7, -8), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#1a6b1a", zorder=41)
    ax.annotate("G", (gx, gy), xytext=(-13, -11), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#a11417", zorder=41)

    ax.axhline(YMAX, color="#888888", lw=0.5, ls=(0, (4, 3)), zorder=4)
    top, lh = 0.982, 0.0265
    ax.add_patch(Rectangle((0.545, top - 6 * lh - 0.014), 0.445,
                           6 * lh + 0.024, transform=ax.transAxes,
                           facecolor="white", edgecolor="#bbbbbb",
                           linewidth=0.6, alpha=0.95, zorder=44))
    ax.text(0.972, top - 0.004, "Alg     L(km)   Ct    \u03ba\u0304(1/km)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
            family="monospace", color="#333333", zorder=45)
    for i, (k, m) in enumerate(res.items(), start=1):
        hot = k == "ISOA"
        ax.text(0.972, top - i * lh - 0.004,
                f"{k:<6s}{m['L']:6.2f} {m['TC']:6.2f} {m['K']:7.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
                family="monospace", zorder=45,
                color="#b23a48" if hot else "#1a1a1a",
                fontweight="bold" if hot else "normal")

    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YTOP)
    ax.set_aspect("equal")
    ax.set_xlabel("East  x (km)")
    ax.set_title(sc["title"], fontsize=8.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    print(f"[{tag}] " + "  ".join(
        f"{k}: L={m['L']:.2f} margin={m['M']*1000:+.0f}m" for k, m in
        res.items()))
    return res


def main():
    set_pub_style(base_font=8)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0))
    for ax, key, lab in zip(axes, ("a", "b"), ("(a)", "(b)")):
        draw_scenario(ax, SCEN[key], lab)
        add_panel_label(ax, lab, x=-0.13, y=1.02, size=10)
    axes[0].set_ylabel("North  y (km)")
    axes[1].tick_params(labelleft=False)

    handles = [Line2D([], [], color=ALGOS[k]["color"], lw=ALGOS[k]["lw"],
                      ls=ALGOS[k]["ls"], label=k) for k in ALGOS]
    handles += [
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Start S"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Goal G"),
        Patch(facecolor="#b23a48", alpha=0.25, edgecolor="#b23a48",
              linewidth=0.8, label="Threat zone"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=8,
                     bbox_to_anchor=(0.52, 0.005), frameon=False,
                     fontsize=7.4, handlelength=1.9, columnspacing=1.3)
    leg.get_texts()[0].set_fontweight("bold")
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 1.0))
    savefig(fig, "154_uav_path_2d_compare.png")


if __name__ == "__main__":
    main()
