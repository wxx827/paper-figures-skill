"""Isochrone accessibility from the city centre, with cumulative opportunity.

Travel times come from a Dijkstra search over a 144 m speed raster in
which radial corridors and an orbital route carry roughly twice the speed
of the background street grid, which is what gives the isochrones their
lobed, star-like shape rather than clean circles.  The map carries the
15 / 30 / 45 / 60-minute bands, the corridors and three classes of
opportunity POI; the narrower right-hand panel converts the same surface
into cumulative reachable population and jobs, and the strip below it
doubles as the shared key.
"""
import heapq

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt, gaussian_filter
from pubstyle import set_pub_style, savefig

BANDS = [15, 30, 45, 60]
BCOL = ["#204864", "#3f6d90", "#7099b6", "#a9c4d4"]
BEYOND = "#eef0ef"
POI = [("hospital", "#b06a52", "s"), ("campus", "#6f9464", "^"),
       ("job centre", "#d3a04c", "o")]
INK = "#2b2b2b"
GREY = "#8c8c8c"

EXT = 26.0
N = 181
CX = CY = 13.0
V_BASE, V_CORR = 9.0, 19.0        # km/h


def corridors():
    lines = []
    for ang in np.deg2rad([8, 62, 126, 183, 242, 305]):
        r = np.linspace(0.0, 15.0, 200)
        w = 0.9 * np.sin(r / 5.0)
        lines.append(np.column_stack([
            CX + r * np.cos(ang) - w * np.sin(ang),
            CY + r * np.sin(ang) + w * np.cos(ang)]))
    th = np.linspace(0, 2 * np.pi, 300)
    rr = 5.6 + 0.45 * np.cos(3 * th + 0.7)
    lines.append(np.column_stack([CX + rr * np.cos(th), CY + rr * np.sin(th)]))
    return lines


def speed_field(lines):
    mark = np.zeros((N, N), bool)
    for ln in lines:
        ii = np.clip((ln[:, 1] / EXT * (N - 1)).astype(int), 0, N - 1)
        jj = np.clip((ln[:, 0] / EXT * (N - 1)).astype(int), 0, N - 1)
        mark[ii, jj] = True
    d = distance_transform_edt(~mark) * (EXT / (N - 1))
    return V_BASE + (V_CORR - V_BASE) * np.exp(-(d / 0.85) ** 2)


def travel_time(v):
    """Dijkstra over the 8-connected speed raster; minutes."""
    h = EXT / (N - 1)
    T = np.full(N * N, np.inf)
    src = int(round(CY / EXT * (N - 1))) * N + int(round(CX / EXT * (N - 1)))
    T[src] = 0.0
    vf = v.ravel()
    steps = [(-1, 0, h), (1, 0, h), (0, -1, h), (0, 1, h),
             (-1, -1, h * 1.4142), (-1, 1, h * 1.4142),
             (1, -1, h * 1.4142), (1, 1, h * 1.4142)]
    done = np.zeros(N * N, bool)
    pq = [(0.0, src)]
    while pq:
        t, u = heapq.heappop(pq)
        if done[u]:
            continue
        done[u] = True
        ui, uj = divmod(u, N)
        for di, dj, L in steps:
            i, j = ui + di, uj + dj
            if not (0 <= i < N and 0 <= j < N):
                continue
            w = i * N + j
            if done[w]:
                continue
            nt = t + L / (0.5 * (vf[u] + vf[w])) * 60.0
            if nt < T[w]:
                T[w] = nt
                heapq.heappush(pq, (nt, w))
    return T.reshape(N, N)


def main():
    set_pub_style()
    rng = np.random.default_rng(88)
    lines = corridors()
    v = speed_field(lines)
    vn = gaussian_filter(rng.normal(size=(N, N)), 7.0)
    v = np.clip(v * (1.0 + 0.22 * vn / vn.std()), 4.0, None)
    T = gaussian_filter(travel_time(v), 1.2)

    g = np.linspace(0, EXT, N)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X - CX, Y - CY)
    nz = gaussian_filter(rng.normal(size=(N, N)), 6.0)
    nz /= nz.std()
    pop = (r / 3.2) * np.exp(-(r / 4.6) ** 1.45) * (1 + 0.30 * nz)
    jobs = np.exp(-(r / 2.7) ** 1.55) * (1 + 0.25 * nz)
    for cx, cy, s, a in [(17.8, 16.6, 1.7, 0.55), (8.9, 17.4, 1.5, 0.42),
                         (16.2, 8.4, 1.5, 0.38)]:
        jobs += a * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2)))
    pop = np.clip(pop, 0, None); jobs = np.clip(jobs, 0, None)
    pop *= 2.35e6 / pop.sum(); jobs *= 1.15e6 / jobs.sum()

    fig = plt.figure(figsize=(10.2, 5.95))
    axm = fig.add_axes([0.010, 0.048, 0.548, 0.878])
    axc = fig.add_axes([0.655, 0.340, 0.318, 0.548])
    axk = fig.add_axes([0.655, 0.048, 0.318, 0.225])

    # -------------------------------------------------------- (a) isochrones
    axm.add_patch(Rectangle((0, 0), EXT, EXT, facecolor=BEYOND,
                            edgecolor="none", zorder=0))
    axm.contourf(X, Y, T, levels=[0] + BANDS, colors=BCOL, zorder=1)
    axm.contour(X, Y, T, levels=BANDS, colors="white", linewidths=0.9,
                zorder=3)
    for ln in lines:
        axm.plot(ln[:, 0], ln[:, 1], color="white", lw=1.0, alpha=0.55,
                 zorder=4)

    # POIs, drawn where the opportunity surface is dense
    w = (jobs / jobs.sum()).ravel()
    pick = rng.choice(w.size, size=21, replace=False, p=w)
    pts = np.column_stack([X.ravel()[pick], Y.ravel()[pick]])
    for k, (name, col, mk) in enumerate(POI):
        sub = pts[k::3]
        axm.plot(sub[:, 0], sub[:, 1], ls="none", marker=mk, ms=4.0,
                 mfc=col, mec="white", mew=0.8, zorder=6)
    axm.plot([CX], [CY], marker="*", ms=13.0, mfc="#f2e6d8", mec="#7f3524",
             mew=1.1, zorder=8)
    axm.text(CX + 0.55, CY + 0.55, "city centre", fontsize=6.8,
             fontweight="bold", color="#7f3524", ha="left", va="bottom",
             zorder=8, bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                 ec="none", alpha=0.82))

    # inline band labels along an empty bearing
    rr = np.linspace(0, 12.4, 400)
    for lev, col, deg in zip(BANDS, BCOL, [99, 87, 104, 78]):
        ang = np.deg2rad(deg)
        tray = np.array([T[int(np.clip((CY + s * np.sin(ang)) / EXT * (N - 1),
                                       0, N - 1)),
                           int(np.clip((CX + s * np.cos(ang)) / EXT * (N - 1),
                                       0, N - 1))] for s in rr])
        k = np.argmin(np.abs(tray - lev))
        px, py = CX + rr[k] * np.cos(ang), CY + rr[k] * np.sin(ang)
        axm.text(px, py, f"{lev} min", fontsize=6.5, fontweight="bold",
                 color=col, ha="center", va="center", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                           alpha=0.90))

    axm.plot([1.0, 6.0], [1.1, 1.1], color=INK, lw=2.2, solid_capstyle="butt",
             zorder=9)
    axm.plot([1.0, 3.5], [1.1, 1.1], color="white", lw=1.0,
             solid_capstyle="butt", zorder=10)
    axm.text(3.5, 1.55, "5 km", fontsize=6.4, ha="center", va="bottom",
             color=INK, zorder=10)
    axm.annotate("N", xy=(24.5, 24.6), xytext=(24.5, 22.6), ha="center",
                 fontsize=7.0, color=INK, fontweight="bold", zorder=10,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))

    axm.set_xlim(0, EXT); axm.set_ylim(0, EXT)
    axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
        sp.set_edgecolor("#bdb9b1")

    # ------------------------------------------------ (b) cumulative curves
    tt = np.linspace(0, 72, 145)
    cp = np.array([pop[T <= s].sum() for s in tt]) / pop.sum() * 100
    cj = np.array([jobs[T <= s].sum() for s in tt]) / jobs.sum() * 100
    for lo, hi, col in zip([0] + BANDS[:-1], BANDS, BCOL):
        axc.axvspan(lo, hi, color=col, alpha=0.13, lw=0, zorder=0)
    axc.plot(tt, cp, color="#4a4a4a", lw=1.8, zorder=4, label="population")
    axc.plot(tt, cj, color="#b06a52", lw=1.8, ls=(0, (4.5, 2.0)), zorder=4,
             label="jobs")
    for lev in BANDS:
        axc.axvline(lev, color="white", lw=1.0, zorder=2)
        for arr, col in [(cp, "#4a4a4a"), (cj, "#b06a52")]:
            val = np.interp(lev, tt, arr)
            axc.plot([lev], [val], marker="o", ms=3.4, mfc="white", mec=col,
                     mew=1.0, zorder=5)
    for arr, col, ty, lab in [(cj, "#b06a52", 52.0, "jobs"),
                              (cp, "#4a4a4a", 28.0, "population")]:
        val = np.interp(30, tt, arr)
        axc.annotate(f"{lab}\n{val:.0f}% within 30 min", xy=(30, val),
                     xytext=(38, ty), fontsize=6.8, color=col,
                     fontweight="bold", ha="left", va="center",
                     linespacing=1.25, zorder=6,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=col,
                                     shrinkA=2, shrinkB=5, alpha=0.55))
    axc.set_xlim(0, 72)
    axc.set_ylim(0, 104)
    axc.set_xticks([0, 15, 30, 45, 60, 72])
    axc.set_xlabel("travel time from centre  (min)", labelpad=2)
    axc.set_ylabel("cumulative share reached  (%)", labelpad=3)
    axc.tick_params(length=2.6)
    axc.grid(axis="y", color="#eeeeee", lw=0.55, zorder=0)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)

    # ------------------------------------------------------- shared key
    axk.set_xlim(0, 1); axk.set_ylim(0, 1)
    axk.axis("off")
    axk.text(0.0, 0.95, "isochrone band", fontsize=7.0, fontweight="bold",
             color="#33474f", va="top")
    for i, (lev, col) in enumerate(zip(BANDS, BCOL)):
        x = 0.005 + i * 0.170
        axk.add_patch(Rectangle((x, 0.50), 0.135, 0.17, facecolor=col,
                                edgecolor="white", lw=0.6))
        lo = 0 if i == 0 else BANDS[i - 1]
        axk.text(x + 0.0675, 0.44, f"{lo}-{lev}", fontsize=6.4, ha="center",
                 va="top", color="#4a4a4a")
    axk.add_patch(Rectangle((0.005 + 4 * 0.170, 0.50), 0.135, 0.17,
                            facecolor=BEYOND, edgecolor="#d8d8d4", lw=0.6))
    axk.text(0.005 + 4 * 0.170 + 0.0675, 0.44, "> 60", fontsize=6.4,
             ha="center", va="top", color="#4a4a4a")
    hs = [Line2D([], [], ls="none", marker=mk, ms=4.0, mfc=c, mec="white",
                 mew=0.8, label=nm) for nm, c, mk in POI]
    hs.append(Line2D([], [], ls="none", marker="*", ms=8.0, mfc="#f2e6d8",
                     mec="#7f3524", mew=1.0, label="centre"))
    leg = axk.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.02, -0.06),
                     ncol=4, frameon=False, fontsize=6.5, handletextpad=0.4,
                     columnspacing=0.9)
    leg.set_clip_on(False)

    for ax, tag, ttl, dx in [(axm, "(a)", "Isochrones from the city centre",
                              0.28),
                             (axc, "(b)", "Cumulative opportunity", -0.44)]:
        pos = ax.get_position()
        fig.text(pos.x0 + dx / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), tag, fontsize=10.5,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pos.x0 + (dx + 0.36) / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), ttl, fontsize=8.8,
                 fontweight="bold", ha="left", va="baseline", color="#33474f")

    savefig(fig, "211_accessibility_isochrone.png")


if __name__ == "__main__":
    main()
