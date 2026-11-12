"""Earth to Mars launch windows, from a grid of Lambert solutions.

Every point on the departure / arrival plane is a two-body Lambert problem
solved with the universal-variable formulation, bisected on z over the whole
grid at once.  The filled contours are the departure energy C3 that those
solutions ask for, the dashed contours are the hyperbolic excess speed left
at arrival, and the diagonals are the flight time implied by the two dates,
so nothing on the map is drawn independently of the solver.  The upper side
panel draws the winning trajectory in the ecliptic plane at true scale by
integrating the departure state forward, and the lower one collapses the map
onto the best departure date.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
DEEP = "#22406e"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
SLATE = "#4a4f5c"
RUST = "#a8523a"

MU_SUN = 1.32712440018e11      # km3/s2
AU = 1.495978707e8             # km
R_EARTH_ORB = 1.00000 * AU
R_MARS_ORB = 1.52371 * AU
INC_MARS = 1.850               # deg
NODE_MARS = 49.56              # deg
U0_MARS = 63.90                # deg, argument of latitude at the epoch
TH0_EARTH = 0.0                # deg, ecliptic longitude at the epoch
DAY = 86400.0

DEP = np.arange(0.0, 401.0, 3.0)         # days from the epoch
ARR = np.arange(200.0, 721.0, 4.0)       # days from the epoch
TOF_MIN, TOF_MAX = 120.0, 420.0          # days
C3_CAP = 45.0                            # km2/s2, clip for the colour scale

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.92, 1.02, 4.58, 4.48)
AUX_T = (7.52, 3.63, 2.66, 1.87)
AUX_B = (7.52, 1.02, 2.66, 1.80)


def stumpff_c(z):
    out = np.empty_like(z)
    pos, neg = z > 1e-6, z < -1e-6
    sz = np.sqrt(np.abs(z))
    out[pos] = (1.0 - np.cos(sz[pos])) / z[pos]
    out[neg] = (np.cosh(sz[neg]) - 1.0) / (-z[neg])
    mid = ~(pos | neg)
    out[mid] = 0.5 - z[mid] / 24.0
    return out


def stumpff_s(z):
    out = np.empty_like(z)
    pos, neg = z > 1e-6, z < -1e-6
    sz = np.sqrt(np.abs(z))
    out[pos] = (sz[pos] - np.sin(sz[pos])) / sz[pos] ** 3
    out[neg] = (np.sinh(sz[neg]) - sz[neg]) / sz[neg] ** 3
    mid = ~(pos | neg)
    out[mid] = 1.0 / 6.0 - z[mid] / 120.0
    return out


def earth_state(t_days):
    """Heliocentric position and velocity (km, km/s) of a circular Earth."""
    n = np.sqrt(MU_SUN / R_EARTH_ORB ** 3)
    th = np.radians(TH0_EARTH) + n * t_days * DAY
    r = R_EARTH_ORB * np.array([np.cos(th), np.sin(th), np.zeros_like(th)])
    v = R_EARTH_ORB * n * np.array([-np.sin(th), np.cos(th),
                                    np.zeros_like(th)])
    return r, v


def mars_state(t_days):
    """Same for a circular Mars orbit inclined to the ecliptic."""
    n = np.sqrt(MU_SUN / R_MARS_ORB ** 3)
    u = np.radians(U0_MARS) + n * t_days * DAY
    i, om = np.radians(INC_MARS), np.radians(NODE_MARS)
    xp, yp = np.cos(u), np.sin(u)
    x = np.cos(om) * xp - np.sin(om) * np.cos(i) * yp
    y = np.sin(om) * xp + np.cos(om) * np.cos(i) * yp
    z = np.sin(i) * yp
    r = R_MARS_ORB * np.array([x, y, z])
    dxp, dyp = -np.sin(u), np.cos(u)
    vx = np.cos(om) * dxp - np.sin(om) * np.cos(i) * dyp
    vy = np.sin(om) * dxp + np.cos(om) * np.cos(i) * dyp
    vz = np.sin(i) * dyp
    v = R_MARS_ORB * n * np.array([vx, vy, vz])
    return r, v


def lambert_grid(r1, r2, dt):
    """Vectorised universal-variable Lambert solve; returns v1, v2 (km/s)."""
    n1 = np.linalg.norm(r1, axis=0)
    n2 = np.linalg.norm(r2, axis=0)
    cross_z = r1[0] * r2[1] - r1[1] * r2[0]
    cosd = np.clip(np.sum(r1 * r2, axis=0) / (n1 * n2), -1.0, 1.0)
    dth = np.arccos(cosd)
    dth = np.where(cross_z <= 0.0, 2.0 * np.pi - dth, dth)
    a_lam = np.sin(dth) * np.sqrt(n1 * n2 / (1.0 - np.cos(dth)))

    def yz(z):
        c, s = stumpff_c(z), stumpff_s(z)
        return n1 + n2 + a_lam * (z * s - 1.0) / np.sqrt(c), c, s

    def fz(z):
        y, c, s = yz(z)
        y = np.maximum(y, 1.0)
        return (y / c) ** 1.5 * s + a_lam * np.sqrt(y) - np.sqrt(MU_SUN) * dt

    lo = np.full(dt.shape, -40.0)
    hi = np.full(dt.shape, 4.0 * np.pi ** 2 - 1e-6)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        f = fz(mid)
        neg = f < 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    z = 0.5 * (lo + hi)
    y, _c, _s = yz(z)
    y = np.maximum(y, 1.0)
    f = 1.0 - y / n1
    g = a_lam * np.sqrt(y / MU_SUN)
    gdot = 1.0 - y / n2
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


def propagate(r0, v0, tof, steps=2000):
    """Runge-Kutta 4 two-body propagation, used to draw the transfer arc."""
    def acc(r):
        return -MU_SUN * r / np.linalg.norm(r) ** 3

    h = tof / steps
    r, v = np.array(r0, dtype=float), np.array(v0, dtype=float)
    out = np.empty((steps + 1, 3))
    out[0] = r
    for k in range(steps):
        k1v, k1a = v, acc(r)
        k2v, k2a = v + 0.5 * h * k1a, acc(r + 0.5 * h * k1v)
        k3v, k3a = v + 0.5 * h * k2a, acc(r + 0.5 * h * k2v)
        k4v, k4a = v + h * k3a, acc(r + h * k3v)
        r = r + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        v = v + h / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
        out[k + 1] = r
    return out, v


def solve_grid():
    d, a = np.meshgrid(DEP, ARR)
    tof = (a - d) * DAY
    ok = (a - d >= TOF_MIN) & (a - d <= TOF_MAX)
    re, ve = earth_state(d)
    rm, vm = mars_state(a)
    safe = np.where(ok, tof, 200.0 * DAY)
    v1, v2 = lambert_grid(re, rm, safe)
    c3 = np.sum((v1 - ve) ** 2, axis=0)
    vinf = np.linalg.norm(v2 - vm, axis=0)
    c3 = np.where(ok, c3, np.nan)
    vinf = np.where(ok, vinf, np.nan)
    return d, a, c3, vinf


def main_panel(fig, ax, d, a, c3, vinf, best):
    cmap = LinearSegmentedColormap.from_list(
        "c3", ["#26456f", "#3b5b92", "#5578a6", "#87a3c6", "#b9c9de",
               "#dde5ee", "#f4f7fa"]).with_extremes(over="#fbfcfd")
    lo = np.nanmin(c3)
    levels = np.r_[np.arange(np.floor(lo), np.floor(lo) + 12.0, 1.5),
                   np.arange(np.floor(lo) + 12.0, C3_CAP + 1.0, 4.0)]
    field = np.clip(c3, None, C3_CAP + 1e-9)
    cf = ax.contourf(d, a, field, levels=levels, cmap=cmap, extend="max",
                     zorder=1)
    cs = ax.contour(d, a, field, levels=levels[::2], colors=["#6f7f95"],
                    linewidths=0.55, alpha=0.9, zorder=3)
    ax.clabel(cs, levels[::2][1::2], fmt="%.0f", fontsize=6.0, inline=True,
              inline_spacing=4, colors=SLATE)

    vl = np.arange(3.0, 6.6, 0.5)
    cv = ax.contour(d, a, np.where(c3 <= C3_CAP, vinf, np.nan), levels=vl,
                    colors=[RUST], linewidths=0.8,
                    linestyles=[(0, (4.0, 2.0))], zorder=4)
    ax.clabel(cv, vl[::2], fmt="%.1f", fontsize=6.0, inline=True,
              inline_spacing=3, colors=RUST)

    for tof in (150.0, 200.0, 250.0, 300.0, 350.0):
        x = np.array([DEP[0], DEP[-1]])
        ax.plot(x, x + tof, color=SLATE, lw=0.6, ls=(0, (1.6, 2.2)),
                alpha=0.8, zorder=5)
        xa = DEP[0] + 0.055 * (DEP[-1] - DEP[0])
        if ARR[0] <= xa + tof <= ARR[-1]:
            ax.text(xa, xa + tof + 4.0, "{:.0f} d".format(tof), fontsize=6,
                    color=SLATE, rotation=34.0, ha="left", va="bottom",
                    rotation_mode="anchor", zorder=6)

    ax.plot([best["dep"]], [best["arr"]], marker="*", ms=11.0, mfc=GOLD,
            mec=INK, mew=0.7, zorder=9)
    ax.annotate("cheapest departure\nday {:.0f}, arrive day {:.0f}\n"
                "C3 {:.1f} km$^2$/s$^2$, {:.0f} d cruise"
                .format(best["dep"], best["arr"], best["c3"], best["tof"]),
                xy=(best["dep"], best["arr"]),
                xytext=(best["dep"] - 118.0, best["arr"] + 96.0),
                fontsize=6.6, color=INK, ha="left", va="center",
                linespacing=1.35, fontweight="bold", zorder=10,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                                shrinkA=3, shrinkB=5))

    cax = ax.inset_axes([1.028, 0.0, 0.030, 1.0])
    cb = fig.colorbar(cf, cax=cax, extend="max")
    cb.set_label("departure energy C3 (km$^2$/s$^2$)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#666666")

    ax.set_xlim(DEP[0], DEP[-1])
    ax.set_ylim(ARR[0], ARR[-1])
    ax.set_xlabel("departure date (days from epoch)")
    ax.set_ylabel("arrival date (days from epoch)")
    ax.tick_params(labelsize=6.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Every date pair costs what its Lambert solution costs",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)

    handles = [Line2D([], [], color=DEEP, lw=0.8,
                      label="C3 contour (km$^2$/s$^2$)"),
               Line2D([], [], color=RUST, lw=0.9, ls=(0, (4.0, 2.0)),
                      label="arrival excess speed (km/s)"),
               Line2D([], [], color=SLATE, lw=0.7, ls=(0, (1.6, 2.2)),
                      label="constant flight time (days)")]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=2.0, borderaxespad=0.35,
                    labelspacing=0.26)
    leg.set_zorder(11)


def geometry_panel(ax, best, arc):
    th = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=0.9, zorder=3)
    tm = np.linspace(0.0, 687.0, 721)
    rm, _v = mars_state(tm)
    ax.plot(rm[0] / AU, rm[1] / AU, color=RUST, lw=0.9, zorder=3)

    ax.plot([0.0], [0.0], marker="o", ms=6.5, mfc=GOLD, mec="#8a6d10",
            mew=0.7, zorder=6)
    ax.text(0.10, -0.16, "Sun", fontsize=6.2, color=SUB, ha="left",
            va="center")

    ax.plot(arc[:, 0] / AU, arc[:, 1] / AU, color=DEEP, lw=1.8, zorder=5)
    re_d, _ = earth_state(best["dep"])
    rm_a, _ = mars_state(best["arr"])
    rm_d, _ = mars_state(best["dep"])
    re_a, _ = earth_state(best["arr"])
    ax.plot([re_d[0] / AU], [re_d[1] / AU], marker="o", ms=5.0, mfc=BLUE,
            mec="white", mew=0.8, zorder=7)
    ax.plot([rm_a[0] / AU], [rm_a[1] / AU], marker="o", ms=5.6, mfc=RUST,
            mec="white", mew=0.8, zorder=7)
    ax.plot([rm_d[0] / AU], [rm_d[1] / AU], marker="o", ms=4.2, mfc="white",
            mec=RUST, mew=0.9, zorder=7)
    ax.plot([re_a[0] / AU], [re_a[1] / AU], marker="o", ms=3.8, mfc="white",
            mec=BLUE, mew=0.9, zorder=7)
    ax.text(re_d[0] / AU, re_d[1] / AU - 0.20, "launch", fontsize=6.0,
            color=BLUE, ha="center", va="top", fontweight="bold")
    ax.text(rm_a[0] / AU + 0.12, rm_a[1] / AU + 0.10, "arrival",
            fontsize=6.0, color=RUST, ha="left", va="bottom",
            fontweight="bold")

    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("ecliptic x (AU)", fontsize=6.8)
    ax.set_ylabel("ecliptic y (AU)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=0.9, label="Earth orbit"),
                       Line2D([], [], color=RUST, lw=0.9, label="Mars orbit"),
                       Line2D([], [], color=DEEP, lw=1.8,
                              label="transfer arc")],
              loc="lower left", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.9,
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)
    ax.set_title("The winning arc, to scale", fontsize=8.0,
                 fontweight="bold", color=INK)


def window_panel(ax, c3, vinf, best, band):
    curve = np.nanmin(c3, axis=0)
    varr = np.array([np.nan if np.all(np.isnan(c3[:, j]))
                     else vinf[np.nanargmin(c3[:, j]), j]
                     for j in range(DEP.size)])
    ax.plot(DEP, curve, color=DEEP, lw=1.6, zorder=6)
    ax.axhline(band["level"], color=SUB, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)
    ax.axvspan(band["t0"], band["t1"], color="#e6ecf4", zorder=1)
    ax.plot([best["dep"]], [best["c3"]], marker="*", ms=9.0, mfc=GOLD,
            mec=INK, mew=0.6, zorder=8)
    y0 = float(np.floor(np.nanmin(curve))) - 1.0
    y1 = float(np.nanmin(curve)) + 26.0
    ax.annotate("{:.0f} day window within\n{:.0f} km$^2$/s$^2$ of the best"
                .format(band["t1"] - band["t0"], band["margin"]),
                xy=(band["t1"], band["level"]),
                xytext=(DEP[-1] - 12.0, y1 - 3.2),
                fontsize=6.1, color=INK, ha="right", va="top",
                linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(DEP[0], DEP[-1])
    ax.set_ylim(y0, y1)
    ax.set_xlabel("departure date (days from epoch)", fontsize=6.8)
    ax.set_ylabel("best C3 (km$^2$/s$^2$)", fontsize=6.8, color=DEEP)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top",):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(DEP, varr, color=RUST, lw=1.3, ls=(0, (4.0, 2.0)), zorder=5)
    ax2.set_ylabel("arrival excess speed (km/s)", fontsize=6.8, color=RUST)
    ax2.tick_params(labelsize=6.4, colors=RUST)
    ax2.set_ylim(np.nanmin(varr) - 0.2, np.nanmin(varr) + 3.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(RUST)
    ax.set_title("Cost against departure date", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    d, a, c3, vinf = solve_grid()
    k = np.unravel_index(np.nanargmin(c3), c3.shape)
    best = dict(dep=float(d[k]), arr=float(a[k]), c3=float(c3[k]),
                vinf=float(vinf[k]), tof=float(a[k] - d[k]))

    r1, v_earth = earth_state(np.array([best["dep"]]))
    r2, v_mars = mars_state(np.array([best["arr"]]))
    v1, v2 = lambert_grid(r1, r2, np.array([best["tof"] * DAY]))
    arc, v_end = propagate(r1[:, 0], v1[:, 0], best["tof"] * DAY)

    miss = np.linalg.norm(arc[-1] - r2[:, 0]) / AU
    assert miss < 1e-5, "the drawn arc must land on Mars: {:.2e} AU".format(
        miss)
    assert abs(np.sum((v1[:, 0] - v_earth[:, 0]) ** 2) - best["c3"]) < 1e-6
    assert abs(np.linalg.norm(v_end - v_mars[:, 0]) - best["vinf"]) < 1e-3

    curve = np.nanmin(c3, axis=0)
    margin = 4.0
    level = best["c3"] + margin
    inside = np.where(curve <= level)[0]
    band = dict(t0=float(DEP[inside[0]]), t1=float(DEP[inside[-1]]),
                level=level, margin=margin)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_geo = axes_inches(fig, AUX_T)
    ax_win = axes_inches(fig, AUX_B)

    main_panel(fig, ax_main, d, a, c3, vinf, best)
    geometry_panel(ax_geo, best, arc)
    window_panel(ax_win, c3, vinf, best, band)



    panel_labels_inches(fig, [ax_main, ax_geo, ax_win], dx=-0.66, dy=0.14)
    savefig(fig, "1021_orb_porkchop_window.png")


if __name__ == "__main__":
    main()
