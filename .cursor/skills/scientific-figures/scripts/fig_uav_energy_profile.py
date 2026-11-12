"""Energy profile of a terrain-following UAV mission.

(a) altitude profile against the terrain section and the admissible band,
(b) commanded speed with the accelerating / decelerating legs marked,
(c) shaft power and battery state of charge,
right: total mission energy of the five planners, from the same model.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import splprep, splev, RegularGridInterpolator

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

XMAX = YMAX = 10.0
PEAKS = [(2.2, 7.4, 1.30, 1.10, 980.0), (6.8, 6.2, 1.60, 1.30, 1250.0),
         (4.5, 2.6, 1.20, 1.50, 860.0), (8.4, 2.2, 1.00, 1.00, 700.0),
         (1.4, 3.0, 0.90, 1.20, 620.0), (7.0, 9.0, 1.10, 0.90, 540.0)]
WAYPOINTS = {
    "ISOA": [(0.6, 0.8), (2.3, 2.2), (4.6, 4.3), (6.6, 7.1), (8.3, 8.5),
             (9.4, 9.2)],
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
        "RRT*": "poly", "A*": "poly"}
# vertical-profile smoothing: ISOA optimises the climb schedule jointly with
# the route, while grid / sampling planners follow the terrain bump by bump.
ALT_SIGMA = {"ISOA": 19.0, "PSO": 7.0, "GWO": 7.0, "RRT*": 4.5, "A*": 5.5}
ACOL = {"ISOA": "#b23a48", "PSO": "#1f6fb4", "GWO": "#2e8b57",
        "RRT*": "#7b3294", "A*": "#e07b1f"}

HMIN, HMAX = 130.0, 900.0          # m, admissible AGL band
VREF, VMIN = 24.0, 14.0            # m/s, commanded cruise / floor
ALAT, VZMAX = 3.0, 5.0             # m/s^2 lateral limit, m/s climb limit
BATTERY_WH = 110.0                 # Wh usable
RESERVE = 30.0                     # % state of charge reserve

# airframe / propulsion constants
MASS, SREF, CD0, KIND_ = 6.0, 0.35, 0.032, 0.055
RHO, ETA, PAVIONICS = 1.12, 0.62, 25.0


def terrain_grid(n=201, seed=7):
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
    f = RegularGridInterpolator((y, x), np.clip(Z, 0, None),
                                bounds_error=False, fill_value=None)
    return lambda px, py: f(np.column_stack([py, px]))


def track(wp, kind, n=600):
    wp = np.asarray(wp, float)
    if kind == "spline":
        tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0, k=3)
        u = np.linspace(0, 1, n)
        xy = splev(u, tck)
        return np.asarray(xy[0]), np.asarray(xy[1])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(wp[:, 0]), np.diff(wp[:, 1])))]
    t = np.linspace(0, d[-1], n)
    x = gaussian_filter1d(np.interp(t, d, wp[:, 0]), 3.0, mode="nearest")
    y = gaussian_filter1d(np.interp(t, d, wp[:, 1]), 3.0, mode="nearest")
    x[0], y[0], x[-1], y[-1] = wp[0, 0], wp[0, 1], wp[-1, 0], wp[-1, 1]
    return x, y


def altitude(x, y, interp, clearance=HMIN, grad=300.0, sigma=7.0):
    """Climb-gradient limited terrain-following altitude.

    The profile is tapered back onto the launch and recovery altitudes so
    the aircraft actually leaves from, and returns to, low level.
    """
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    terr = interp(x, y)
    req = terr + clearance
    z = np.max(req[None, :] - grad * np.abs(s[:, None] - s[None, :]), axis=1)
    for _ in range(80):
        z = np.maximum(gaussian_filter1d(z, sigma, mode="nearest"), req)
    cap = np.minimum(req[0] + grad * s, req[-1] + grad * (s[-1] - s))
    return s, terr, np.minimum(z, np.maximum(cap, req))


def curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx ** 2 + dy ** 2) ** 1.5, 1e-9)


def profile(x, y, s, z):
    """Speed, climb rate, power and cumulative energy along the track.

    Every planner is flown with the same commanded cruise speed; the speed
    can only be *reduced*, either by the lateral-acceleration limit in a
    turn or by the maximum climb rate.  Re-accelerating afterwards costs
    kinetic energy, so a jagged route is penalised twice: longer time and
    repeated speed cycles.
    """
    grad = np.gradient(z, s)                                     # m/km
    kap = curvature(x, y) / np.maximum(np.gradient(s), 1e-9)     # 1/km
    kap = gaussian_filter1d(kap, 6, mode="nearest")
    v_turn = np.sqrt(ALAT / np.maximum(kap / 1000.0, 1e-9))
    v_clmb = np.where(grad > 1.0, VZMAX / (grad / 1000.0), 1e6)
    v = np.minimum(np.minimum(v_turn, v_clmb), VREF)
    v = np.clip(gaussian_filter1d(v, 7, mode="nearest"), VMIN, VREF)
    vz = grad * v / 1000.0                                        # m/s
    W = MASS * 9.81
    drag = (0.5 * RHO * SREF * CD0 * v ** 2
            + 2.0 * KIND_ * W ** 2 / (RHO * SREF * v ** 2))
    lift_pw = np.where(vz >= 0, W * vz, 0.35 * W * vz)
    dvdt = v * np.gradient(v, s * 1000.0)                         # m/s^2
    P = np.maximum((drag * v + lift_pw + np.clip(MASS * v * dvdt, 0, None))
                   / ETA + PAVIONICS, 60.0)
    ds = np.gradient(s) * 1000.0
    dt = ds / v
    E = np.cumsum(P * dt)                                         # J
    return v, vz, kap, P, E, np.cumsum(dt)


def main():
    set_pub_style(base_font=8)
    interp = terrain_grid()

    data = {}
    for name, wp in WAYPOINTS.items():
        x, y = track(wp, KIND[name])
        s, terr, z = altitude(x, y, interp, sigma=ALT_SIGMA[name])
        v, vz, kap, P, E, t = profile(x, y, s, z)
        data[name] = dict(s=s, terr=terr, z=z, v=v, vz=vz, kap=kap, P=P,
                          E=E, t=t, climb=float(np.sum(np.clip(np.diff(z),
                                                               0, None))))

    d = data["ISOA"]
    s, terr, z, v, vz, P, E, t = (d["s"], d["terr"], d["z"], d["v"], d["vz"],
                                  d["P"], d["E"], d["t"])
    soc = 100.0 * (1.0 - E / (BATTERY_WH * 3600.0))

    fig = plt.figure(figsize=(7.4, 5.5))
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.975, top=0.905,
                          bottom=0.145, hspace=0.18, wspace=0.32,
                          width_ratios=[2.85, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    axb = fig.add_subplot(gs[:, 1])

    # ---- (a) altitude -----------------------------------------------------
    ax1.fill_between(s, terr, terr + HMIN, color="#b23a48", alpha=0.20, lw=0,
                     zorder=2, label=f"Minimum clearance ({HMIN:.0f} m AGL)")
    ax1.fill_between(s, 0, terr, color="#c9a06a", alpha=0.9, lw=0,
                     zorder=3, label="Terrain section")
    ax1.plot(s, terr, color="#8e6f5e", lw=0.9, zorder=4)
    ax1.plot(s, z, color="#b23a48", lw=2.0, zorder=6, label="ISOA altitude")
    climb = np.gradient(z, s) > 80.0
    ax1.fill_between(s, 0, 1, where=climb, transform=ax1.get_xaxis_transform(),
                     color="#b23a48", alpha=0.055, lw=0, zorder=0)
    agl = z - terr
    ax1.set_ylim(0, 2050)
    ax1.set_ylabel("Altitude MSL (m)")
    ax1.set_title("Energy profile of the ISOA mission "
                  "(terrain-following cruise)", fontsize=8.5)
    ax1.text(0.985, 0.94, f"AGL {agl.min():.0f}\u2013{agl.max():.0f} m "
             f"(limit {HMIN:.0f}\u2013{HMAX:.0f} m)   total climb "
             f"{np.sum(np.clip(np.diff(z), 0, None)):.0f} m",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.3,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#bbbbbb", lw=0.6, alpha=0.92))
    add_panel_label(ax1, "(a)", x=-0.085, y=1.0, size=10)

    # ---- (b) speed --------------------------------------------------------
    dv = np.gradient(v, t)
    vlo = 13.0
    ax2.fill_between(s, vlo, v, where=dv < -0.008, color="#1f6fb4",
                     alpha=0.14, lw=0, zorder=1)
    ax2.fill_between(s, vlo, v, where=dv > 0.008, color="#2e8b57",
                     alpha=0.14, lw=0, zorder=1)
    ax2.plot(s, v, color="#b23a48", lw=1.8, zorder=4)
    ax2.axhline(VREF, color="#666666", lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax2.text(s[-1] * 0.995, VREF + 0.25, f"commanded cruise "
             f"$v_{{ref}}$ = {VREF:.0f} m/s", ha="right", va="bottom",
             fontsize=6.2, color="#555555")
    imin = int(np.argmin(v))
    ax2.annotate(f"climb- and turn-rate limited\n{v[imin]:.1f} m/s",
                 xy=(s[imin], v[imin]), xytext=(s[imin] + 0.9, vlo + 0.35),
                 fontsize=6.2, ha="left", va="bottom", color="#1a4d80",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#1a4d80"))
    ax2.set_ylim(vlo, VREF + 1.5)
    ax2.set_ylabel("Ground speed (m/s)")
    add_panel_label(ax2, "(b)", x=-0.085, y=1.0, size=10)

    # ---- (c) power and state of charge ------------------------------------
    ax3.plot(s, P, color="#7b3294", lw=1.5, zorder=4, label="Shaft power")
    ax3.fill_between(s, 60, P, color="#7b3294", alpha=0.12, lw=0, zorder=1)
    ipk = int(np.argmax(P))
    ax3.annotate(f"launch climb peak {P[ipk]:.0f} W", xy=(s[ipk], P[ipk]),
                 xytext=(0.20, 0.90), textcoords="axes fraction",
                 fontsize=6.2, ha="left", va="top", color="#5a2170",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#5a2170"))
    ax3.set_ylabel("Power  $P$ (W)", color="#5a2170")
    ax3.tick_params(axis="y", colors="#5a2170")
    ax3.set_ylim(0, max(P) * 1.42)
    ax3.set_xlabel("Range along track  $s$ (km)")
    ax3r = ax3.twinx()
    ax3r.plot(s, soc, color="#2e8b57", lw=1.6, zorder=5)
    ax3r.axhline(RESERVE, color="#a11417", lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax3r.text(0.02, RESERVE + 3, f"{RESERVE:.0f} % reserve",
              transform=ax3r.get_yaxis_transform(), fontsize=6.2,
              color="#a11417", va="bottom")
    ax3r.set_ylim(0, 108)
    ax3r.set_ylabel("Battery SoC (%)", color="#1f6b45")
    ax3r.tick_params(axis="y", colors="#1f6b45")
    ax3r.spines["top"].set_visible(False)
    ax3r.spines["right"].set_color("#1f6b45")
    ax3r.spines["right"].set_linewidth(0.8)
    add_panel_label(ax3, "(c)", x=-0.085, y=1.0, size=10)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, s[-1])
        ax.grid(lw=0.35, alpha=0.45)
        style_axes(ax)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.spines["right"].set_visible(True)

    # ---- right: total energy per planner ----------------------------------
    names = list(WAYPOINTS)
    etot = np.array([data[n]["E"][-1] / 1000.0 for n in names])     # kJ
    ypos = np.arange(len(names))[::-1]
    axb.barh(ypos, etot, height=0.62,
             color=[ACOL[n] for n in names], edgecolor="white", lw=0.5,
             zorder=3)
    for yp, val, n in zip(ypos, etot, names):
        axb.text(val + etot.max() * 0.025, yp, f"{val:.0f}", va="center",
                 fontsize=6.6, color=ACOL[n],
                 fontweight="bold" if n == "ISOA" else "normal")
        axb.text(val * 0.5, yp, f"{data[n]['t'][-1]/60.0:.1f} min",
                 va="center", ha="center", fontsize=6.0, color="white",
                 fontweight="bold")
    axb.set_yticks(ypos)
    axb.set_yticklabels(names, fontsize=7.2)
    for lab in axb.get_yticklabels():
        if lab.get_text() == "ISOA":
            lab.set_fontweight("bold")
            lab.set_color("#b23a48")
    axb.set_xlim(0, etot.max() * 1.26)
    axb.set_xlabel("Mission energy (kJ)")
    axb.set_title("Total energy", fontsize=8.5)
    axb.grid(axis="x", lw=0.35, alpha=0.45)
    style_axes(axb)
    tim = np.array([data[n]["t"][-1] for n in names])
    k = names.index("ISOA")
    oe, ot = np.delete(etot, k), np.delete(tim, k)
    dt = 100.0 * (tim[k] - ot.min()) / ot.min()
    de_best = 100.0 * (etot[k] - oe.min()) / oe.min()
    de_worst = 100.0 * (etot[k] - oe.max()) / oe.max()
    axb.text(0.975, 0.978, f"ISOA: {dt:+.0f} % flight time\n"
             f"{de_best:+.0f} % energy vs. PSO\n"
             f"{de_worst:+.0f} % energy vs. RRT*",
             transform=axb.transAxes, ha="right", va="top", fontsize=6.1,
             color="#a11417", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#e0b3b3", lw=0.6, alpha=0.93))
    add_panel_label(axb, "(d)", x=-0.34, y=1.0, size=10)

    fig.legend(handles=[
        Line2D([], [], color="#b23a48", lw=2.0, label="ISOA altitude"),
        Patch(facecolor="#c9a06a", alpha=0.9, label="Terrain section"),
        Patch(facecolor="#b23a48", alpha=0.25,
              label=f"Minimum clearance ({HMIN:.0f} m AGL)"),
        Patch(facecolor="#b23a48", alpha=0.12, label="Climb segment"),
        Patch(facecolor="#2e8b57", alpha=0.2, label="Accelerating leg"),
        Patch(facecolor="#1f6fb4", alpha=0.2, label="Decelerating leg"),
        Line2D([], [], color="#7b3294", lw=1.5, label="Shaft power $P$"),
        Line2D([], [], color="#2e8b57", lw=1.6, label="Battery SoC")],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015),
        frameon=False, fontsize=6.8, handlelength=1.7, columnspacing=1.5)

    print(f"ISOA: L={s[-1]:.2f} km  T={t[-1]:.0f} s  "
          f"mean v={np.mean(v):.1f} m/s  E={E[-1]/1000:.1f} kJ "
          f"({E[-1]/3600:.1f} Wh)  final SoC={soc[-1]:.1f} %")
    print("AGL min/max = "
          f"{(z-terr).min():.0f}/{(z-terr).max():.0f} m (band {HMIN:.0f}"
          f"\u2013{HMAX:.0f})")
    for n in names:
        print(f"  {n:5s} E={data[n]['E'][-1]/1000:6.1f} kJ  "
              f"L={data[n]['s'][-1]:5.2f} km  "
              f"T={data[n]['t'][-1]:5.0f} s  "
              f"climb={data[n]['climb']:5.0f} m  "
              f"AGLmax={(data[n]['z']-data[n]['terr']).max():4.0f} m")
    savefig(fig, "159_uav_energy_profile.png")


if __name__ == "__main__":
    main()
