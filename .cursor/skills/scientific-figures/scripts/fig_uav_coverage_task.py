"""Area-coverage mission planning for a single UAV.

(a) boustrophedon and spiral coverage patterns over the task polygon,
(b) covered fraction against flight time,
(c) accumulated sensor-footprint map (revisit count),
(d) coverage efficiency of the three strategies.

Panels (b)-(d) are all measured on the rasterised footprint of the very
paths drawn in (a), so the curves, the map and the bars are consistent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

_P = np.array([(0.55, 0.70), (4.30, 0.35), (5.65, 2.35), (5.10, 4.60),
               (2.60, 5.45), (0.35, 3.60)])
POLY = _P.mean(axis=0) + 0.82 * (_P - _P.mean(axis=0))
SWATH = 0.62          # km, sensor ground swath
RTURN = 0.25          # km, minimum turn radius of the airframe
VCRUISE = 16.0        # m/s
NG = 260              # raster resolution
LIMX, LIMY = 6.0, 6.0
STEP = 0.02           # km between path samples

STRAT = {
    "Boustrophedon (ISOA)": dict(c="#b23a48", lw=1.6),
    "Spiral (inward)": dict(c="#1f6fb4", lw=1.1),
    "Diagonal sweep": dict(c="#e07b1f", lw=1.1),
}


def grid():
    x = np.linspace(0, LIMX, NG)
    y = np.linspace(0, LIMY, NG)
    X, Y = np.meshgrid(x, y)
    inside = MplPath(POLY).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return x, y, X, Y, inside


def densify(pts, step=STEP):
    """Resample a polyline at a fixed spatial step."""
    pts = np.asarray(pts, float)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    keep = np.r_[True, np.diff(d) > 1e-9]
    pts, d = pts[keep], d[keep]
    t = np.arange(0.0, d[-1], step)
    return np.column_stack([np.interp(t, d, pts[:, 0]),
                            np.interp(t, d, pts[:, 1])])


def sweep_path(u, spacing, ov=0.07):
    """Back-and-forth sweep along ``u`` closed by tangent 180 deg turns.

    Rows are clipped to the polygon and then extended to a shared turn
    abscissa, so each turn is a semicircle of radius ``spacing / 2`` that
    meets both rows tangentially - the shape a fixed-wing actually flies.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    mpath = MplPath(POLY)
    T, C = POLY @ u, POLY @ n
    tlo, thi = T.min() - 0.3, T.max() + 0.3
    rows = []
    for c in np.arange(C.min() + spacing * 0.5, C.max(), spacing):
        ts = np.linspace(tlo, thi, 1200)
        m = mpath.contains_points(np.outer(ts, u) + c * n)
        if m.sum() < 6:
            continue
        idx = np.where(m)[0]
        rows.append((c, ts[idx[0]], ts[idx[-1]]))

    def xy(t, c):
        return t * u + c * n

    out = [xy(rows[0][1], rows[0][0])]
    for i, (c, t0, t1) in enumerate(rows):
        fwd = i % 2 == 0
        if i + 1 == len(rows):
            out.append(xy(t1 if fwd else t0, c))
            break
        cn, _, tn1 = rows[i + 1]
        tn0 = rows[i + 1][1]
        tturn = (max(t1, tn1) + ov) if fwd else (min(t0, tn0) - ov)
        out.append(xy(tturn, c))
        rad = abs(cn - c) / 2.0
        cen = xy(tturn, 0.5 * (c + cn))
        p0 = xy(tturn, c)
        a0 = np.arctan2(p0[1] - cen[1], p0[0] - cen[0])
        best, score = None, -np.inf
        for sweep in (np.pi, -np.pi):
            th = a0 + np.linspace(0.0, sweep, 48)
            pts = cen + rad * np.column_stack([np.cos(th), np.sin(th)])
            sc = float(np.dot(pts.mean(axis=0) - cen, u)) * (1 if fwd else -1)
            if sc > score:
                best, score = pts, sc
        out.extend(best)
    return densify(np.array(out))


def boustrophedon(x, y, inside, spacing):
    """Lawn-mower sweep aligned with the long axis of the task area."""
    return sweep_path((1.0, 0.0), spacing)


def spiral(X, Y, inside, spacing):
    """Inward rings taken from the distance transform of the polygon."""
    dx = LIMX / (NG - 1)
    dist = distance_transform_edt(inside) * dx
    levels = np.arange(spacing * 0.5, dist.max(), spacing)
    fig_tmp = plt.figure()
    cs = fig_tmp.gca().contour(X, Y, dist, levels=levels)
    rings = []
    for segs in cs.allsegs:
        if not segs:
            continue
        seg = max(segs, key=len)
        if len(seg) > 20:
            rings.append(seg)
    plt.close(fig_tmp)
    pts = []
    for k, ring in enumerate(rings):
        if k and len(pts):
            i0 = int(np.argmin(np.hypot(ring[:, 0] - pts[-1][0],
                                        ring[:, 1] - pts[-1][1])))
            ring = np.roll(ring, -i0, axis=0)
        pts.extend(map(tuple, ring))
    return densify(pts)


def diagonal(x, y, inside, spacing):
    """Baseline: sweep at 45 deg to the long axis - shorter and more
    numerous rows, hence many more end-of-row turns."""
    return sweep_path((1.0, 1.0), spacing)


def infeasible_fraction(p, rmin=RTURN, tol=1.06):
    """Share of the route whose turn radius is tighter than the airframe."""
    q = np.column_stack([gaussian_filter1d(p[:, 0], 2.0, mode="nearest"),
                         gaussian_filter1d(p[:, 1], 2.0, mode="nearest")])
    dx, dy = np.gradient(q[:, 0]), np.gradient(q[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    return float(np.mean(kap > tol / rmin))


def footprint(path, X, Y, inside, swath, step=STEP):
    """Sensor passes per cell and the covered fraction over time.

    Samples are much finer than the swath, so the raw hit count is divided
    by the number of samples that a single straight pass spends over one
    cell (the chord 2r), turning it into a physical pass count.
    """
    dx = LIMX / (NG - 1)
    r = swath / 2.0
    cnt = np.zeros_like(X, dtype=float)
    covered = np.zeros(X.shape, dtype=bool)
    frac, marks = [], []
    rad = int(np.ceil(r / dx))
    ii = np.round(path[:, 0] / dx).astype(int)
    jj = np.round(path[:, 1] / dx).astype(int)
    oy, ox = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disc = (ox ** 2 + oy ** 2) * dx ** 2 <= r ** 2
    ntot = int(inside.sum())
    for k in range(len(path)):
        i0, i1 = ii[k] - rad, ii[k] + rad + 1
        j0, j1 = jj[k] - rad, jj[k] + rad + 1
        if i0 < 0 or j0 < 0 or i1 > NG or j1 > NG:
            continue
        sub = disc & inside[j0:j1, i0:i1]
        cnt[j0:j1, i0:i1] += sub
        covered[j0:j1, i0:i1] |= sub
        if k % 5 == 0:
            frac.append(covered.sum() / ntot)
            marks.append(k)
    frac.append(covered.sum() / ntot)
    marks.append(len(path) - 1)
    return cnt * step / (2.0 * r), np.array(frac), np.array(marks)


def main():
    set_pub_style(base_font=8)
    x, y, X, Y, inside = grid()
    step = STEP
    spacing = SWATH * 0.85                         # 15 % lateral overlap

    paths = {
        "Boustrophedon (ISOA)": boustrophedon(x, y, inside, spacing),
        "Spiral (inward)": spiral(X, Y, inside, spacing),
        "Diagonal sweep": diagonal(x, y, inside, spacing),
    }
    res = {}
    for name, p in paths.items():
        cnt, frac, marks = footprint(p, X, Y, inside, SWATH)
        L = len(p) * step
        res[name] = dict(cnt=cnt, frac=frac, t=marks * step * 1000.0 / VCRUISE,
                         L=L, cov=frac[-1], bad=infeasible_fraction(p),
                         redun=float(cnt[inside & (cnt > 0)].mean()),
                         eff=frac[-1] * inside.sum() * (LIMX / (NG - 1)) ** 2 / L)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.965, top=0.915,
                          bottom=0.115, hspace=0.34, wspace=0.30)

    # ---- (a) patterns -----------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    axa.add_patch(MplPolygon(POLY, closed=True, facecolor="#eef2e8",
                             edgecolor="#4d6b4d", lw=1.1, zorder=1))
    pb = paths["Boustrophedon (ISOA)"]
    axa.plot(pb[:, 0], pb[:, 1], color="#b23a48", lw=1.5, zorder=5,
             solid_capstyle="round")
    ps = paths["Spiral (inward)"]
    axa.plot(ps[:, 0], ps[:, 1], color="#1f6fb4", lw=0.85, alpha=0.75,
             zorder=4)
    for frac in (0.30, 0.62):
        k = int(frac * len(pb))
        axa.add_patch(plt.Circle(pb[k], SWATH / 2.0, facecolor="#b23a48",
                                 alpha=0.16, edgecolor="#a11417", lw=0.7,
                                 ls=(0, (2.5, 1.5)), zorder=6))
    k = int(0.30 * len(pb))
    axa.annotate("", xy=(pb[k, 0], pb[k, 1] - SWATH / 2.0),
                 xytext=(pb[k, 0], pb[k, 1] + SWATH / 2.0), zorder=9,
                 arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="#a11417",
                                 mutation_scale=6))
    axa.text(pb[k, 0] + 0.12, pb[k, 1], f"{SWATH*1000:.0f} m", fontsize=6.0,
             color="#a11417", va="center", zorder=9)
    axa.plot(*pb[0], marker="^", ms=7, mfc="#2e7d5b", mec="#123d12", mew=0.8,
             ls="none", zorder=8)
    axa.plot(*pb[-1], marker="*", ms=12, mfc="#b23a48", mec="#5c1113",
             mew=0.8, ls="none", zorder=8)
    axa.annotate("", xy=(pb[60, 0], pb[60, 1]), xytext=(pb[10, 0], pb[10, 1]),
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#a11417"),
                 zorder=9)
    axa.text(0.035, 0.965, f"swath {SWATH*1000:.0f} m, lateral overlap 15 %",
             transform=axa.transAxes, ha="left", va="top", fontsize=6.3,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#bbbbbb", lw=0.6, alpha=0.93))
    axa.set_title("Coverage patterns over the task area", fontsize=8.5)
    add_panel_label(axa, "(a)", x=-0.16, y=1.02, size=10)

    # ---- (b) coverage vs time ---------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    for name, st in STRAT.items():
        r = res[name]
        axb.plot(r["t"] / 60.0, 100.0 * r["frac"], color=st["c"],
                 lw=st["lw"], zorder=5 if name.startswith("Bous") else 3,
                 label=name)
    axb.axhline(95.0, color="#666666", lw=0.7, ls=(0, (4, 2)), zorder=2)
    axb.text(0.02, 96.0, "95 % coverage requirement", fontsize=6.2,
             color="#555555", transform=axb.get_yaxis_transform(),
             va="bottom")
    rb = res["Boustrophedon (ISOA)"]
    i95 = int(np.argmax(rb["frac"] >= 0.95)) if (rb["frac"] >= 0.95).any() \
        else len(rb["frac"]) - 1
    axb.plot([rb["t"][i95] / 60.0], [100 * rb["frac"][i95]], marker="o",
             ms=4, color="#b23a48", zorder=6)
    axb.annotate(f"{rb['t'][i95]/60.0:.1f} min", xy=(rb["t"][i95] / 60.0, 95),
                 xytext=(rb["t"][i95] / 60.0 + 1.4, 62), fontsize=6.3,
                 color="#a11417",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#a11417"))
    axb.set_xlabel("Flight time (min)")
    axb.set_ylabel("Area covered (%)")
    axb.set_ylim(0, 105)
    axb.set_xlim(0, max(r["t"][-1] for r in res.values()) / 60.0 * 1.02)
    axb.set_title("Coverage growth", fontsize=8.5)
    axb.grid(lw=0.35, alpha=0.45)
    style_axes(axb)
    axb.legend(fontsize=6.2, loc="lower right", frameon=True, framealpha=0.92,
               edgecolor="#cccccc", handlelength=1.5, labelspacing=0.25,
               borderpad=0.35).get_frame().set_linewidth(0.5)
    add_panel_label(axb, "(b)", x=-0.17, y=1.02, size=10)

    # ---- (c) revisit map --------------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    cnt = res["Boustrophedon (ISOA)"]["cnt"].copy()
    revisit = np.where(inside, cnt, np.nan)
    cm = LinearSegmentedColormap.from_list(
        "rev", ["#ffffff", "#dcebd8", "#a8cfa4", "#5aa377", "#1f6b45"])
    im = axc.pcolormesh(X, Y, revisit, cmap=cm, shading="nearest",
                        vmin=0.0, vmax=2.2)
    axc.add_patch(MplPolygon(POLY, closed=True, facecolor="none",
                             edgecolor="#33553f", lw=1.1, zorder=4))
    axc.text(0.035, 0.965, "dark strips = 15 % lateral overlap\n"
             f"mean {res['Boustrophedon (ISOA)']['redun']:.2f} passes per "
             "covered cell", transform=axc.transAxes, ha="left", va="top",
             fontsize=6.1, zorder=6, linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#bbbbbb", lw=0.6, alpha=0.93))
    # axc is set_aspect("equal") below, so fig.colorbar(ax=axc) would size
    # itself from the pre-aspect box; an axes-coordinate inset follows the
    # square the map actually occupies.
    cax_c = axc.inset_axes([1.035, 0.025, 0.040, 0.95])
    cb = fig.colorbar(im, cax=cax_c, ticks=[0, 0.5, 1.0, 1.5, 2.0])
    cb.set_label("Sensor passes per cell (\u2013)", fontsize=7.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5)
    axc.set_title("Accumulated sensor footprint", fontsize=8.5)
    add_panel_label(axc, "(c)", x=-0.16, y=1.02, size=10)

    # ---- (d) efficiency ---------------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    names = list(STRAT)
    short = ["Boustro-\nphedon", "Spiral", "Diagonal"]
    eff = np.array([res[n]["eff"] for n in names])
    cov = np.array([100 * res[n]["cov"] for n in names])
    ln = np.array([res[n]["L"] for n in names])
    xp = np.arange(len(names))
    bad = np.array([res[n]["bad"] for n in names])
    bars = axd.bar(xp, eff, width=0.58, color=[STRAT[n]["c"] for n in names],
                   edgecolor="white", lw=0.6, zorder=3)
    for b, e, c, l, f in zip(bars, eff, cov, ln, bad):
        top = f"{e:.3f}"
        if f > 0.01:
            b.set_hatch("///")
            b.set_edgecolor("#333333")
            b.set_linewidth(0.7)
            top += "*"
            axd.text(b.get_x() + b.get_width() / 2, e + eff.max() * 0.105,
                     f"{100*f:.0f} % below\n$R_{{min}}$", ha="center",
                     va="bottom", fontsize=6.0, color="#a11417")
        axd.text(b.get_x() + b.get_width() / 2, e + eff.max() * 0.025,
                 top, ha="center", va="bottom", fontsize=6.6,
                 fontweight="bold")
        axd.text(b.get_x() + b.get_width() / 2, e * 0.5,
                 f"{c:.1f} %\n{l:.1f} km", ha="center", va="center",
                 fontsize=6.2, color="white", fontweight="bold")
    axd.set_xticks(xp)
    axd.set_xticklabels(short, fontsize=6.8)
    axd.get_xticklabels()[0].set_color("#b23a48")
    axd.get_xticklabels()[0].set_fontweight("bold")
    axd.set_ylabel("Coverage efficiency (km$^2$ per km flown)")
    axd.set_ylim(0, eff.max() * 1.42)
    axd.set_title("Strategy efficiency", fontsize=8.5)
    axd.grid(axis="y", lw=0.35, alpha=0.45)
    style_axes(axd)
    axd.text(0.5, 0.985, "bar labels: coverage reached / path length\n"
             "* hatched = not flyable at $R_{min}$ = "
             f"{RTURN*1000:.0f} m", transform=axd.transAxes, ha="center",
             va="top", fontsize=6.0, color="#555555", linespacing=1.35)
    add_panel_label(axd, "(d)", x=-0.17, y=1.02, size=10)

    for ax in (axa, axc):
        ax.set_xlim(0, LIMX)
        ax.set_ylim(0, LIMY)
        ax.set_aspect("equal")
        ax.set_xlabel("East  x (km)")
        ax.set_ylabel("North  y (km)")
        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_yticks(np.arange(0, 7, 1))
        ax.grid(lw=0.3, alpha=0.4)
        style_axes(ax)

    fig.legend(handles=[
        Line2D([], [], color="#b23a48", lw=1.6, label="Boustrophedon (ISOA)"),
        Line2D([], [], color="#1f6fb4", lw=1.0, label="Spiral (inward)"),
        Line2D([], [], color="#e07b1f", lw=1.0, label="Diagonal sweep"),
        Patch(facecolor="#b23a48", alpha=0.2, edgecolor="#a11417", lw=0.6,
              label="Sensor footprint disc"),
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Entry point"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Exit point")],
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.008),
        frameon=False, fontsize=6.8, handlelength=1.7, columnspacing=1.3)

    area = inside.sum() * (LIMX / (NG - 1)) ** 2
    print(f"task area = {area:.2f} km^2, swath = {SWATH*1000:.0f} m")
    for n in names:
        r = res[n]
        mono = bool(np.all(np.diff(r["frac"]) >= -1e-12))
        print(f"  {n:22s} L={r['L']:5.2f} km  cover={100*r['cov']:5.1f}%  "
              f"passes={r['redun']:.2f}  eff={r['eff']:.3f}  "
              f"below_Rmin={100*r['bad']:4.1f}%  monotone={mono}")
    savefig(fig, "160_uav_coverage_task.png")


if __name__ == "__main__":
    main()
