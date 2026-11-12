"""Trajectory smoothing: raw waypoints vs B-spline vs Dubins arc fillets.

Top row: the three geometries over the same obstacle field.
Bottom row: the matching curvature profiles against the airframe limit.
Only the Dubins construction keeps the curvature bounded everywhere, and
it does so at the price of a small, quantified detour.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy.interpolate import splprep, splev

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

WP = np.array([(0.30, 0.40), (1.60, 0.60), (2.40, 2.20), (3.90, 2.40),
               (4.60, 3.90), (6.10, 4.10), (6.80, 5.60)])
OBST = [(2.90, 1.20, 0.45), (5.30, 3.00, 0.42)]
RMIN = 0.25                                    # km, minimum turn radius
KMAX = 1.0 / RMIN                              # 1/km
VCRUISE = 24.0                                 # m/s
BANK = np.degrees(np.arctan(VCRUISE ** 2 / (RMIN * 1000.0 * 9.81)))

METHODS = ["Raw waypoint polyline", "Cubic B-spline",
           "Dubins arc fillets (ISOA)"]
MCOL = {"Raw waypoint polyline": "#6b6b6b", "Cubic B-spline": "#1f6fb4",
        "Dubins arc fillets (ISOA)": "#b23a48"}
MLW = {"Raw waypoint polyline": 1.3, "Cubic B-spline": 1.6,
       "Dubins arc fillets (ISOA)": 2.2}


def arclength(p):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def resample(p, step=0.004):
    s = arclength(p)
    t = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(t, s, p[:, 0]),
                            np.interp(t, s, p[:, 1])])


def raw_path():
    p = resample(WP)
    s = arclength(p)
    sw = arclength(WP)
    corners = []
    for i in range(1, len(WP) - 1):
        u1 = WP[i] - WP[i - 1]
        u2 = WP[i + 1] - WP[i]
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        corners.append((sw[i], np.degrees(np.arccos(
            np.clip(np.dot(u1, u2), -1.0, 1.0)))))
    return p, s, np.zeros_like(s), corners


def bspline_path(smooth=0.008, step=0.08):
    """Least-squares cubic spline through a densified copy of the route."""
    q = resample(WP, step)
    tck, _ = splprep([q[:, 0], q[:, 1]], s=smooth, k=3)
    u = np.linspace(0.0, 1.0, 4000)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    dx, dy = np.asarray(dx), np.asarray(dy)
    ddx, ddy = np.asarray(ddx), np.asarray(ddy)
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    p = np.column_stack([np.asarray(x), np.asarray(y)])
    return p, arclength(p), kap


def dubins_path(R=RMIN):
    """Replace every corner by the largest admissible circular fillet."""
    pts, kap = [WP[0].copy()], [0.0]
    for i in range(1, len(WP) - 1):
        p0, p1, p2 = WP[i - 1], WP[i], WP[i + 1]
        v1, v2 = p1 - p0, p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        u1, u2 = v1 / l1, v2 / l2
        turn = float(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        if turn < 1e-6:
            continue
        T = min(R * np.tan(turn / 2.0), 0.45 * l1, 0.45 * l2)
        Re = T / np.tan(turn / 2.0)
        A = p1 - u1 * T
        pts.append(A)
        kap.append(0.0)
        sgn = 1.0 if (u1[0] * u2[1] - u1[1] * u2[0]) > 0 else -1.0
        C = A + np.array([-u1[1], u1[0]]) * sgn * Re
        a0 = np.arctan2(A[1] - C[1], A[0] - C[0])
        for th in np.linspace(0.0, sgn * turn, 56)[1:]:
            pts.append(C + Re * np.array([np.cos(a0 + th), np.sin(a0 + th)]))
            kap.append(1.0 / Re)
        pts.append(pts[-1].copy())     # zero-length step back to a straight
        kap.append(0.0)
    pts.append(WP[-1].copy())
    kap.append(0.0)
    p = np.array(pts)
    return p, arclength(p), np.array(kap)


def clearance(p):
    return min(float(np.min(np.hypot(p[:, 0] - cx, p[:, 1] - cy))) - r
               for cx, cy, r in OBST)


def deviation(p):
    ref = resample(WP, 0.002)
    d = [np.min(np.hypot(ref[:, 0] - q[0], ref[:, 1] - q[1]))
         for q in p[::3]]
    return float(np.max(d))


def main():
    set_pub_style(base_font=8)
    geo = {"Raw waypoint polyline": raw_path()[:3],
           "Cubic B-spline": bspline_path(),
           "Dubins arc fillets (ISOA)": dubins_path()}
    corners = raw_path()[3]
    Lref = geo["Raw waypoint polyline"][1][-1]
    smax = max(g[1][-1] for g in geo.values())

    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.878,
                          bottom=0.155, hspace=0.44, wspace=0.24,
                          height_ratios=[1.32, 1.0])

    for j, name in enumerate(METHODS):
        p, s, kap = geo[name]
        col = MCOL[name]
        isoa = name.endswith("(ISOA)")

        ax = fig.add_subplot(gs[0, j])
        for cx, cy, r in OBST:
            ax.add_patch(Circle((cx, cy), r, facecolor="#b23a48", alpha=0.16,
                                edgecolor="#b23a48", lw=0.8, ls=(0, (3, 1.6)),
                                zorder=2))
        ax.plot(WP[:, 0], WP[:, 1], color="#b8b8b8", lw=0.7, ls=(0, (3, 2)),
                zorder=3)
        ax.plot(WP[:, 0], WP[:, 1], marker="o", ms=3.0, mfc="white",
                mec="#666666", mew=0.7, ls="none", zorder=4)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=MLW[name], zorder=6,
                solid_capstyle="round")
        ax.plot(*WP[0], marker="^", ms=7, mfc="#2e7d5b", mec="#123d12",
                mew=0.8, ls="none", zorder=8)
        ax.plot(*WP[-1], marker="*", ms=12, mfc="#b23a48", mec="#5c1113",
                mew=0.8, ls="none", zorder=8)

        L = s[-1]
        kline = ("$\\kappa_{max}\\rightarrow\\infty$" if j == 0
                 else f"$\\kappa_{{max}}$ = {kap.max():.1f} 1/km")
        ax.text(0.035, 0.965,
                f"$L$ = {L:.2f} km   $\\Delta L$ = {100*(L-Lref)/Lref:+.1f} %\n"
                f"{kline}\nmax offset = {deviation(p)*1000:.0f} m\n"
                f"clearance = {clearance(p)*1000:.0f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.1,
                linespacing=1.42, zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#bbbbbb", lw=0.6, alpha=0.94))
        axi = ax.inset_axes([0.585, 0.038, 0.40, 0.40])
        axi.plot(WP[:, 0], WP[:, 1], color="#b8b8b8", lw=0.7, ls=(0, (3, 2)))
        axi.plot(p[:, 0], p[:, 1], color=col, lw=1.9,
                 solid_capstyle="round")
        axi.plot([WP[2, 0]], [WP[2, 1]], marker="o", ms=3.4, mfc="white",
                 mec="#666666", mew=0.7, ls="none")
        rtxt = ("corner: $R$ = 0" if j == 0
                else f"tightest $R$ = {1000.0 / kap.max():.0f} m")
        axi.text(0.5, 0.035, rtxt, transform=axi.transAxes, ha="center",
                 va="bottom", fontsize=6.0, color=col,
                 fontweight="bold" if isoa else "normal")
        axi.set_xlim(2.03, 2.79)
        axi.set_ylim(1.82, 2.58)
        axi.set_aspect("equal")
        axi.set_xticks([])
        axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
            sp.set_color("#888888")
        ax.indicate_inset_zoom(axi, edgecolor="#999999", lw=0.6, alpha=0.85)

        ax.set_xlim(-0.15, 7.35)
        ax.set_ylim(-0.35, 6.9)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)", labelpad=1)
        if j == 0:
            ax.set_ylabel("y (km)", labelpad=1)
        ax.set_title(name, fontsize=8.2, color=col if isoa else "#1a1a1a",
                     fontweight="bold" if isoa else "normal")
        ax.grid(lw=0.3, alpha=0.4)
        style_axes(ax)
        add_panel_label(ax, f"({'abc'[j]})", x=-0.10, y=1.02, size=10)

        axk = fig.add_subplot(gs[1, j])
        axk.axhspan(KMAX, 16.0, color="#b23a48", alpha=0.07, lw=0, zorder=0)
        axk.axhline(KMAX, color="#a11417", lw=1.0, ls=(0, (4, 2)), zorder=4)
        if j == 0:
            for sc, ang in corners:
                axk.annotate("", xy=(sc, 13.2), xytext=(sc, 0.0), zorder=5,
                             arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                             color=col))
                axk.text(sc, 13.5, f"{ang:.0f}\u00b0", ha="center",
                         va="bottom", fontsize=6.0, color=col)
            axk.text(0.52, 0.115, "unbounded at every waypoint",
                     transform=axk.transAxes, ha="center", va="center",
                     fontsize=6.2, color=col,
                     bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                               edgecolor="#cccccc", lw=0.6, alpha=0.93))
        else:
            axk.plot(s, kap, color=col, lw=1.4, zorder=5)
            axk.fill_between(s, 0, kap, color=col, alpha=0.14, lw=0, zorder=3)
            over = kap > KMAX
            ds = np.gradient(s)
            if over.any():
                axk.fill_between(s, KMAX, kap, where=over, color="#b23a48",
                                 alpha=0.35, lw=0, zorder=6)
                frac = 100.0 * float(np.sum(ds[over])) / s[-1]
                axk.text(0.975, 0.93, f"{frac:.0f} % of the route violates "
                         f"$\\kappa_{{max}}$", transform=axk.transAxes,
                         ha="right", va="top", fontsize=6.0, color="#a11417",
                         bbox=dict(boxstyle="round,pad=0.22",
                                   facecolor="white", edgecolor="none",
                                   alpha=0.85))
            else:
                axk.text(0.975, 0.93, "feasible everywhere",
                         transform=axk.transAxes, ha="right", va="top",
                         fontsize=6.4, color="#1f6b45", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.22",
                                   facecolor="white", edgecolor="none",
                                   alpha=0.85))
        axk.text(0.015, KMAX + 0.5, f"$\\kappa_{{max}}$ = {KMAX:.0f} 1/km",
                 transform=axk.get_yaxis_transform(), fontsize=6.0,
                 color="#a11417", va="bottom")
        axk.set_xlim(0, smax)
        axk.set_ylim(0, 16.0)
        axk.set_xlabel("Arc length  $s$ (km)", labelpad=1)
        if j == 0:
            axk.set_ylabel("Curvature  $\\kappa$ (1/km)", labelpad=1)
        axk.grid(lw=0.3, alpha=0.4)
        style_axes(axk)
        add_panel_label(axk, f"({'def'[j]})", x=-0.10, y=1.03, size=10)

    fig.legend(handles=[
        Line2D([], [], color="#b8b8b8", lw=0.8, ls=(0, (3, 2)),
               label="Reference waypoint sequence"),
        Line2D([], [], color="#666666", marker="o", ls="none", ms=4,
               mfc="white", mew=0.7, label="Waypoint"),
        Patch(facecolor="#b23a48", alpha=0.2, edgecolor="#b23a48", lw=0.8,
              label="Threat zone"),
        Line2D([], [], color="#a11417", lw=1.0, ls=(0, (4, 2)),
               label=f"Airframe limit $\\kappa_{{max}}$ = {KMAX:.0f} 1/km"),
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Start"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Goal")],
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.012),
        frameon=False, fontsize=6.8, handlelength=1.7, columnspacing=1.3)
    # Broken over two lines: as one line this banner was wider than the
    # canvas, so the bbox_inches="tight" export expanded instead of clipping
    # (DESIGN_SYSTEM 12.11).  The break is manual rather than textwrap so the
    # mathtext spans are never split.

    for name in METHODS:
        p, s, kap = geo[name]
        print(f"{name:28s} L={s[-1]:6.3f} km  dL="
              f"{100*(s[-1]-Lref)/Lref:+5.1f}%  kmax={kap.max():6.2f}  "
              f"offset={deviation(p)*1000:5.0f} m  "
              f"clearance={clearance(p)*1000:5.0f} m")
    savefig(fig, "161_uav_smoothing_compare.png")


if __name__ == "__main__":
    main()
