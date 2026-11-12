"""Ground track of a 705 km sun-synchronous orbit over five consecutive revs.

The track is propagated from the classical elements rather than drawn: the
argument of latitude advances at the Keplerian mean motion, the right
ascension of the node is walked forward by the J2 secular rate, and the map
longitude is that node minus the Earth's rotation, so the westward step
between successive equator crossings in the lower side panel is measured off
the same array that draws the curve.  The swath edges are placed at the
Earth-central angle that belongs to the sensor's nadir half-angle, and a pass
is shaded where the central angle to a station falls inside the mask circle;
the upper side panel replays those passes as elevation against time.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
DEEP = "#22406e"
PALE = "#a8bcd8"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
SEA = "#e7eef3"
LAND = "#d3d8ce"
COAST = "#9aa294"
GRID = "#c9cfd4"

MU = 398600.4418          # km3/s2
R_E = 6378.14             # km
J2 = 1.08263e-3
OMEGA_E = 7.2921159e-5    # rad/s
H_SAT = 705.0             # km
INC_DEG = 98.19           # deg
LON_AN0 = -12.0           # deg, longitude of the first ascending node
N_REV = 5
FOV_HALF = 15.0           # deg, sensor nadir half-angle
MASK_DEG = 10.0           # deg, ground-station elevation mask

STATIONS = [("Fairbanks", -147.50, 64.86, TEAL),
            ("Hobart", 147.30, -42.90, BLUE),
            ("Punta Arenas", -70.90, -53.00, ORANGE)]

FIG_W, FIG_H = 11.0, 5.4
MAIN = (0.86, 1.28, 5.96, 2.98)
AUX_T = (7.36, 3.04, 3.46, 1.22)
AUX_B = (7.36, 1.28, 3.46, 1.22)

LANDS = [
    [(-168, 66), (-166, 70), (-156, 71), (-140, 70), (-128, 70), (-112, 69),
     (-100, 70), (-85, 70), (-80, 73), (-64, 60), (-56, 52), (-66, 45),
     (-70, 42), (-75, 36), (-81, 26), (-84, 30), (-90, 29), (-97, 26),
     (-105, 21), (-110, 24), (-114, 29), (-120, 34), (-124, 42), (-124, 48),
     (-132, 55), (-145, 60), (-155, 58), (-162, 63)],
    [(-45, 60), (-52, 66), (-56, 71), (-52, 76), (-40, 82), (-25, 82),
     (-20, 76), (-24, 70), (-32, 65), (-40, 61)],
    [(-80, -2), (-79, 4), (-72, 11), (-62, 10), (-52, 5), (-44, -2),
     (-35, -6), (-38, -13), (-48, -24), (-54, -34), (-58, -38), (-62, -41),
     (-66, -47), (-70, -54), (-75, -52), (-73, -44), (-72, -35), (-71, -24),
     (-75, -15), (-79, -6)],
    [(-17, 15), (-16, 21), (-10, 27), (0, 32), (10, 34), (20, 32), (30, 31),
     (33, 28), (37, 22), (43, 12), (51, 11), (48, 3), (41, -2), (40, -10),
     (35, -18), (32, -26), (28, -33), (20, -34), (15, -27), (13, -18),
     (12, -6), (9, 0), (5, 4), (-4, 5), (-12, 8)],
    [(-9, 43), (-9, 38), (-1, 36), (6, 43), (14, 41), (18, 40), (24, 38),
     (28, 41), (35, 36), (35, 32), (43, 30), (48, 29), (57, 25), (60, 25),
     (67, 25), (70, 20), (73, 16), (77, 8), (80, 13), (87, 21), (92, 22),
     (95, 16), (99, 10), (104, 1), (107, 10), (109, 15), (112, 22),
     (117, 24), (122, 31), (122, 39), (126, 40), (130, 43), (135, 49),
     (141, 53), (148, 59), (158, 62), (170, 65), (179, 66), (179, 70),
     (170, 71), (150, 74), (130, 74), (110, 77), (90, 76), (75, 73),
     (68, 72), (58, 70), (45, 68), (35, 70), (30, 70), (22, 70), (12, 65),
     (10, 58), (5, 53), (0, 50), (-3, 48), (-5, 45)],
    [(114, -22), (122, -18), (130, -12), (136, -12), (142, -11), (146, -19),
     (151, -24), (153, -28), (150, -37), (145, -38), (139, -36), (132, -32),
     (125, -33), (118, -35), (114, -30)],
    [(-5, 50), (-6, 53), (-3, 58), (0, 57), (1, 53), (-2, 51)],
    [(44, -16), (48, -14), (50, -20), (47, -25), (44, -21)],
]


def semi_major():
    return R_E + H_SAT


def mean_motion():
    return np.sqrt(MU / semi_major() ** 3)


def node_rate():
    """Secular J2 drift of the right ascension of the node (rad/s)."""
    a = semi_major()
    return (-1.5 * J2 * np.sqrt(MU) * R_E ** 2 / a ** 3.5
            * np.cos(np.radians(INC_DEG)))


def earth_angle(mask_deg):
    """Earth-central angle (deg) reached at a given elevation mask."""
    r = semi_major()
    eta = np.degrees(np.arcsin(R_E / r * np.cos(np.radians(mask_deg))))
    return 90.0 - mask_deg - eta


def swath_angle(fov_deg):
    """Earth-central angle (deg) of a sensor half-angle measured from nadir."""
    r = semi_major()
    eps = np.degrees(np.arccos(np.sin(np.radians(fov_deg)) * r / R_E))
    return 90.0 - fov_deg - eps


def ground_track(t):
    """Sub-satellite latitude and longitude (deg) at times ``t`` (s)."""
    n, om = mean_motion(), node_rate()
    inc = np.radians(INC_DEG)
    u = n * t
    lat = np.arcsin(np.sin(inc) * np.sin(u))
    lon = (np.radians(LON_AN0) + np.arctan2(np.cos(inc) * np.sin(u),
                                            np.cos(u))
           + om * t - OMEGA_E * t)
    return np.degrees(lat), wrap_lon(np.degrees(lon))


def wrap_lon(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def central_angle(lat1, lon1, lat2, lon2):
    """Great-circle angle (deg) between two points given in degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    c = (np.sin(p1) * np.sin(p2)
         + np.cos(p1) * np.cos(p2) * np.cos(dl))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def elevation(gamma_deg):
    """Elevation (deg) of the spacecraft seen at Earth-central angle gamma."""
    r = semi_major()
    g = np.radians(gamma_deg)
    return np.degrees(np.arctan2(np.cos(g) - R_E / r, np.sin(g)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.arctan2(y, x)


def destination(lat1, lon1, brg, ang_deg):
    p1 = np.radians(lat1)
    l1 = np.radians(lon1)
    d = np.radians(ang_deg)
    p2 = np.arcsin(np.sin(p1) * np.cos(d)
                   + np.cos(p1) * np.sin(d) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), wrap_lon(np.degrees(l2))


def split_at_dateline(*series, gap=180.0):
    """Yield index slices between longitude jumps, for every series given."""
    breaks = np.zeros(series[0].size - 1, dtype=bool)
    for s in series:
        breaks |= np.abs(np.diff(s)) > gap
    idx = np.r_[0, np.where(breaks)[0] + 1, series[0].size]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            yield slice(a, b)


def track_swath(lat, lon, half_deg):
    """Left and right swath edges (deg) either side of the sub-satellite path."""
    brg = np.empty_like(lat)
    brg[:-1] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    brg[-1] = brg[-2]
    la, lo = destination(lat, lon, brg + np.pi / 2.0, half_deg)
    ra, ro = destination(lat, lon, brg - np.pi / 2.0, half_deg)
    return la, lo, ra, ro


def contacts(t, lat, lon, lam):
    """Contact windows per station: duration, peak elevation, elevation trace."""
    out = []
    for name, slon, slat, col in STATIONS:
        gam = central_angle(slat, slon, lat, lon)
        vis = gam <= lam
        edges = np.diff(vis.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        stops = list(np.where(edges == -1)[0] + 1)
        if vis[0]:
            starts = [0] + starts
        if vis[-1]:
            stops = stops + [vis.size]
        for a, b in zip(starts, stops):
            if b - a < 3:
                continue
            el = elevation(gam[a:b])
            k = int(np.argmax(el))
            out.append(dict(name=name, color=col, i0=a, i1=b,
                            t=(t[a:b] - t[a + k]) / 60.0, el=el,
                            peak=float(el[k]),
                            minutes=float((t[b - 1] - t[a]) / 60.0)))
    return out


def draw_world(ax):
    ax.add_patch(plt.Rectangle((-180, -90), 360, 180, facecolor=SEA,
                               edgecolor="none", zorder=0))
    for poly in LANDS:
        p = np.asarray(poly, dtype=float)
        ax.fill(p[:, 0], p[:, 1], facecolor=LAND, edgecolor=COAST,
                linewidth=0.6, zorder=1)
    lon = np.linspace(-180.0, 180.0, 361)
    top = -68.0 + 4.0 * np.sin(np.radians(3.0 * lon)) \
        + 2.0 * np.sin(np.radians(7.0 * lon + 40.0))
    ax.fill_between(lon, -90.0, top, facecolor=LAND, edgecolor=COAST,
                    linewidth=0.6, zorder=1)
    for y in range(-60, 90, 30):
        ax.axhline(y, color=GRID, lw=0.5, zorder=2)
    for x in range(-120, 180, 60):
        ax.axvline(x, color=GRID, lw=0.5, zorder=2)
    ax.axhline(0.0, color="#9aa3ab", lw=0.8, zorder=2)


def main_panel(ax, t, lat, lon, lam, lam_swath, cons, rev_lon):
    draw_world(ax)
    per = 2.0 * np.pi / mean_motion()

    la, lo, ra, ro = track_swath(lat, lon, lam_swath)
    for sl in split_at_dateline(lon, lo, ro):
        x = np.r_[lo[sl], ro[sl][::-1]]
        y = np.r_[la[sl], ra[sl][::-1]]
        ax.fill(x, y, facecolor=PALE, edgecolor="none", alpha=0.55, zorder=3)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "revs", [PALE, DEEP])
    for k in range(N_REV):
        m = (t >= k * per) & (t <= (k + 1) * per)
        col = cmap(0.18 + 0.82 * k / max(N_REV - 1, 1))
        for sl in split_at_dateline(lon[m]):
            ax.plot(lon[m][sl], lat[m][sl], color=col, lw=1.5, zorder=5,
                    solid_capstyle="round")

    for name, slon, slat, col in STATIONS:
        brg = np.linspace(0.0, 2.0 * np.pi, 361)
        clat, clon = destination(slat, slon, brg, lam)
        for sl in split_at_dateline(clon):
            ax.plot(clon[sl], clat[sl], color=col, lw=0.9,
                    ls=(0, (3.2, 2.0)), zorder=6)
        ax.plot([slon], [slat], marker="^", ms=5.0, mfc=col, mec="white",
                mew=0.8, zorder=8)
        ax.text(slon, slat + 4.0, name, ha="center", va="bottom",
                fontsize=6.2, color=col, fontweight="bold", zorder=9)

    for c in cons:
        sl = slice(c["i0"], c["i1"])
        for s in split_at_dateline(lon[sl]):
            ax.plot(lon[sl][s], lat[sl][s], color=RED, lw=2.6, alpha=0.95,
                    zorder=7, solid_capstyle="round")

    for k, xl in enumerate(rev_lon):
        ax.plot([xl], [0.0], marker="o", ms=3.4, mfc="white", mec=DEEP,
                mew=0.9, zorder=8)
        ax.text(xl, -7.0, "{:d}".format(k + 1), ha="center", va="top",
                fontsize=6.0, color=DEEP, zorder=9)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.tick_params(labelsize=6.6)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#8a8a8a")
    ax.set_title("Five revs of a {:.0f} km sun-synchronous orbit, "
                 "inclination {:.2f} deg".format(H_SAT, INC_DEG),
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    handles = [Line2D([], [], color=DEEP, lw=1.5, label="sub-satellite track"),
               Patch(facecolor=PALE, edgecolor="none",
                     label="{:.0f} km swath".format(
                         2.0 * np.radians(lam_swath) * R_E)),
               Line2D([], [], color=RED, lw=2.4, label="station contact arc"),
               Line2D([], [], color=SUB, lw=0.9, ls=(0, (3.2, 2.0)),
                      label="{:.0f} deg elevation mask".format(MASK_DEG))]
    leg = ax.figure.legend(handles=handles, loc="lower left",
                           bbox_to_anchor=(0.076, 0.118), ncol=4,
                           fontsize=6.3, frameon=False, handlelength=1.7,
                           columnspacing=1.6, borderaxespad=0.0)
    return leg


def pass_panel(ax, cons):
    seen = set()
    for c in cons:
        lab = c["name"] if c["name"] not in seen else None
        seen.add(c["name"])
        ax.plot(c["t"], c["el"], color=c["color"], lw=1.4, zorder=5,
                label=lab)
        ax.plot([0.0], [c["peak"]], marker="o", ms=3.0, mfc=c["color"],
                mec="white", mew=0.7, zorder=6)
    ax.axhline(MASK_DEG, color=RED, lw=1.0, ls=(0, (4.0, 2.2)), zorder=4)
    ax.text(-6.6, MASK_DEG + 1.5, "{:.0f} deg mask".format(MASK_DEG),
            ha="left", va="bottom", fontsize=6.0, color=RED)
    best = max(cons, key=lambda c: c["peak"])
    ax.annotate("best pass: {:.0f} deg peak,\n{:.1f} min above the mask"
                .format(best["peak"], best["minutes"]),
                xy=(0.0, best["peak"]), xytext=(-6.6, 70.0), fontsize=6.1,
                color=INK, ha="left", va="center", linespacing=1.30,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(0.0, 88.0)
    ax.set_xlabel("time from closest approach (min)", fontsize=6.8)
    ax.set_ylabel("elevation (deg)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.25, labelspacing=0.22)
    ax.set_title("Every contact in the window", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def node_panel(ax, rev_lon, step):
    k = np.arange(1, rev_lon.size + 1)
    ax.plot(k, rev_lon, color=DEEP, lw=1.3, marker="o", ms=4.0, mfc="white",
            mec=DEEP, mew=1.0, zorder=6)
    fit = rev_lon[0] + step * (k - 1)
    ax.plot(k, fit, color=GOLD, lw=1.1, ls=(0, (3.4, 2.2)), zorder=5,
            label="{:.2f} deg per rev".format(step))
    ax.annotate("", xy=(k[2], rev_lon[2]), xytext=(k[2], rev_lon[1]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(k[2] + 0.12, 0.5 * (rev_lon[1] + rev_lon[2]),
            "{:.2f} deg".format(abs(step)), ha="left", va="center",
            fontsize=6.1, color=ORANGE)
    ax.set_xlim(0.6, rev_lon.size + 0.6)
    span = rev_lon.max() - rev_lon.min()
    ax.set_ylim(rev_lon.min() - 0.22 * span, rev_lon.max() + 0.30 * span)
    ax.set_xticks(k)
    ax.set_xlabel("revolution number", fontsize=6.8)
    ax.set_ylabel("ascending node (deg)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.1, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, borderaxespad=0.25)
    ax.set_title("The node walks west every rev", fontsize=8.0,
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
    per = 2.0 * np.pi / mean_motion()
    t = np.linspace(0.0, N_REV * per, 4001)
    lat, lon = ground_track(t)
    lam = earth_angle(MASK_DEG)
    lam_swath = swath_angle(FOV_HALF)

    rev_t = np.arange(N_REV) * per
    rev_lon = wrap_lon(np.degrees(np.radians(LON_AN0)
                                  + (node_rate() - OMEGA_E) * rev_t))
    step = float(np.degrees((node_rate() - OMEGA_E) * per))
    cons = contacts(t, lat, lon, lam)

    assert abs(np.max(np.abs(lat)) - (180.0 - INC_DEG)) < 0.2, "inclination"
    assert 0.5 < lam_swath < 4.0, "swath angle out of range"
    assert np.all(np.abs(np.diff(rev_lon) - step) < 1e-6), "uneven node step"
    assert len(cons) >= 4, "not enough contacts to plot"
    assert {c["name"] for c in cons} == {s[0] for s in STATIONS}, \
        "every station should see at least one pass"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_map = axes_inches(fig, MAIN)
    ax_pass = axes_inches(fig, AUX_T)
    ax_node = axes_inches(fig, AUX_B)

    main_panel(ax_map, t, lat, lon, lam, lam_swath, cons, rev_lon)
    pass_panel(ax_pass, cons)
    node_panel(ax_node, rev_lon, step)

    total = sum(c["minutes"] for c in cons)


    panel_labels_inches(fig, [ax_map, ax_pass, ax_node], dx=-0.60, dy=0.12)
    savefig(fig, "1020_orb_ground_track.png")


if __name__ == "__main__":
    main()
