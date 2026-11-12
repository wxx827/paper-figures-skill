"""Instantaneous coverage of a 48-satellite Walker constellation.

The snapshot is built from the constellation definition alone: six planes of
eight satellites at 550 km and 53 deg, phased by the Walker factor, are
converted to sub-satellite points and every cell of a half-degree grid is
told how many spacecraft it can see above a 10 deg mask.  The mask fixes the
Earth-central angle of each footprint, so the shaded fold count, the coverage
against latitude in the upper side panel and the revisit gaps in the lower
one are all read out of that same geometry rather than assumed.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
COAST = "#8f978c"
GRID = "#c9cfd4"

MU = 398600.4418          # km3/s2
R_E = 6378.14             # km
OMEGA_E = 7.2921159e-5    # rad/s
H_SAT = 550.0             # km
INC_DEG = 53.0            # deg
N_PLANE = 6
N_PER_PLANE = 8
WALKER_F = 1
MASK_DEG = 10.0           # deg, user elevation mask
SIM_HOURS = 24.0
LAT_PROBE = np.arange(0.0, 66.0, 5.0)

FIG_W, FIG_H = 11.0, 5.4
MAIN = (0.86, 1.24, 5.60, 2.80)
AUX_T = (7.42, 2.82, 3.25, 1.22)
AUX_B = (7.42, 1.10, 3.25, 1.22)

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


def mean_motion():
    return np.sqrt(MU / (R_E + H_SAT) ** 3)


def footprint_angle():
    """Earth-central angle (deg) of one footprint at the elevation mask."""
    r = R_E + H_SAT
    eta = np.degrees(np.arcsin(R_E / r * np.cos(np.radians(MASK_DEG))))
    return 90.0 - MASK_DEG - eta


def subpoints(t):
    """Sub-satellite latitude and longitude (deg), shape (n_sat, n_time)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = mean_motion()
    inc = np.radians(INC_DEG)
    plane = np.arange(N_PLANE)
    slot = np.arange(N_PER_PLANE)
    raan = np.radians(plane[:, None] * 360.0 / N_PLANE) * np.ones(
        (1, N_PER_PLANE))
    u0 = np.radians(slot[None, :] * 360.0 / N_PER_PLANE
                    + WALKER_F * plane[:, None] * 360.0
                    / (N_PLANE * N_PER_PLANE))
    raan = raan.ravel()[:, None]
    u = u0.ravel()[:, None] + n * t[None, :]
    lat = np.arcsin(np.sin(inc) * np.sin(u))
    lon = (raan + np.arctan2(np.cos(inc) * np.sin(u), np.cos(u))
           - OMEGA_E * t[None, :])
    return np.degrees(lat), wrap_lon(np.degrees(lon))


def wrap_lon(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def cos_central(lat_a, lon_a, lat_b, lon_b):
    pa, pb = np.radians(lat_a), np.radians(lat_b)
    dl = np.radians(lon_b - lon_a)
    return np.sin(pa) * np.sin(pb) + np.cos(pa) * np.cos(pb) * np.cos(dl)


def fold_map(lat_s, lon_s, lam, glat, glon):
    """Number of satellites in view for every grid cell."""
    gla, glo = np.meshgrid(glat, glon, indexing="ij")
    fold = np.zeros(gla.shape, dtype=int)
    thr = np.cos(np.radians(lam))
    for k in range(lat_s.size):
        fold += (cos_central(gla, glo, lat_s[k], lon_s[k]) >= thr)
    return fold


def coverage_by_latitude(lat_s, lon_s, lam, glat, glon):
    fold = fold_map(lat_s, lon_s, lam, glat, glon)
    one = 100.0 * np.mean(fold >= 1, axis=1)
    two = 100.0 * np.mean(fold >= 2, axis=1)
    w = np.cos(np.radians(glat))
    total = 100.0 * np.sum(w * np.mean(fold >= 1, axis=1)) / np.sum(w)
    return fold, one, two, total


def revisit_gaps(lam, hours=SIM_HOURS, step=30.0):
    """Mean and worst outage (min) at a set of latitudes over one day."""
    t = np.arange(0.0, hours * 3600.0 + step, step)
    lat_s, lon_s = subpoints(t)
    thr = np.cos(np.radians(lam))
    mean_gap, worst_gap, duty = [], [], []
    for phi in LAT_PROBE:
        seen = (cos_central(phi, 0.0, lat_s, lon_s) >= thr).any(axis=0)
        duty.append(100.0 * np.mean(seen))
        runs, count = [], 0
        for flag in seen:
            if flag:
                if count:
                    runs.append(count)
                count = 0
            else:
                count += 1
        if count:
            runs.append(count)
        gaps = np.array(runs, dtype=float) * step / 60.0 if runs else \
            np.array([0.0])
        mean_gap.append(float(gaps.mean()))
        worst_gap.append(float(gaps.max()))
    return (np.array(mean_gap), np.array(worst_gap), np.array(duty))


def draw_coast(ax):
    for poly in LANDS:
        p = np.asarray(poly, dtype=float)
        ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]],
                color=COAST, lw=0.6, alpha=0.85, zorder=4)
    lon = np.linspace(-180.0, 180.0, 361)
    top = -68.0 + 4.0 * np.sin(np.radians(3.0 * lon)) \
        + 2.0 * np.sin(np.radians(7.0 * lon + 40.0))
    ax.plot(lon, top, color=COAST, lw=0.6, alpha=0.85, zorder=4)


def main_panel(fig, ax, fold, glat, glon, lat_s, lon_s, lam, total):
    colors = ["#f2f4f6", "#cbd8e8", "#9db6d5", "#7093be", "#4a70a4",
              "#2c4e80"]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) - 0.4, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    mesh = ax.pcolormesh(glon, glat, np.clip(fold, 0, len(colors) - 1),
                         cmap=cmap, norm=norm, shading="nearest", zorder=1)
    draw_coast(ax)

    for y in (-INC_DEG, INC_DEG):
        ax.axhline(y, color=ORANGE, lw=0.9, ls=(0, (4.0, 2.4)), zorder=5)
    ax.text(-176.0, INC_DEG + 3.0, "inclination limit {:.0f} deg"
            .format(INC_DEG), fontsize=6.1, color=ORANGE, ha="left",
            va="bottom", zorder=6)

    ax.plot(lon_s, lat_s, linestyle="none", marker="o", ms=2.6, mfc=GOLD,
            mec="#7a5f10", mew=0.4, zorder=7)

    band = np.abs(glat) <= 50.0
    gla, glo = np.meshgrid(glat[band], glon, indexing="ij")
    near = np.full(gla.shape, -2.0)
    for k in range(lat_s.size):
        near = np.maximum(near, cos_central(gla, glo, lat_s[k], lon_s[k]))
    w = np.unravel_index(int(np.argmin(near)), near.shape)
    wlon, wlat = float(glo[w]), float(gla[w])
    hole = np.degrees(np.arccos(np.clip(near[w], -1.0, 1.0)))
    ax.plot([wlon], [wlat], marker="s", ms=4.6, mfc="none", mec=RED,
            mew=1.1, zorder=8)
    ax.annotate("worst hole: nearest satellite\n{:.1f} deg away, {:.0f} km"
                .format(hole, np.radians(hole) * R_E), xy=(wlon, wlat),
                xytext=(wlon + 14.0, wlat + (30.0 if wlat < 0 else -30.0)),
                fontsize=6.1, color=RED, ha="left", va="center", zorder=9,
                linespacing=1.30,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=2, shrinkB=3))

    cax = ax.inset_axes([1.014, 0.0, 0.022, 1.0])
    cb = fig.colorbar(mesh, cax=cax, ticks=range(len(colors)))
    cb.set_label("satellites in view", fontsize=6.5, labelpad=2.0)
    cb.ax.set_yticklabels([str(i) for i in range(len(colors) - 1)]
                          + ["{:d}+".format(len(colors) - 1)])
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#666666")

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
    ax.set_title("Snapshot of a {:d} x {:d} Walker shell at {:.0f} km, "
                 "{:.1f} per cent of the globe covered"
                 .format(N_PLANE, N_PER_PLANE, H_SAT, total),
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    handles = [Line2D([], [], color=GOLD, marker="o", ms=3.4, lw=0,
                      mec="#7a5f10", label="sub-satellite point"),
               Line2D([], [], color=ORANGE, lw=0.9, ls=(0, (4.0, 2.4)),
                      label="highest latitude reached"),
               Line2D([], [], color=COAST, lw=0.7, label="coastline"),
               Line2D([], [], color=RED, marker="s", ms=4.0, lw=0,
                      mfc="none", label="worst hole in the coverage")]
    ax.figure.legend(handles=handles, loc="lower left",
                     bbox_to_anchor=(0.076, 0.118), ncol=4, fontsize=6.3,
                     frameon=False, handlelength=1.7, columnspacing=1.6,
                     borderaxespad=0.0)


def latitude_panel(ax, glat, one, two, lam):
    ax.fill_between(glat, 0.0, one, color="#dce4ef", zorder=2)
    ax.plot(glat, one, color=DEEP, lw=1.5, zorder=5, label="at least one")
    ax.plot(glat, two, color=TEAL, lw=1.3, ls=(0, (4.0, 2.0)), zorder=5,
            label="at least two")
    for y in (-INC_DEG, INC_DEG):
        ax.axvline(y, color=ORANGE, lw=0.8, ls=(0, (3.4, 2.2)), zorder=4)
    peak = glat[int(np.argmax(one))]
    ax.annotate("best at {:.0f} deg: {:.0f} per cent"
                .format(peak, one.max()), xy=(peak, one.max()),
                xytext=(-88.0, 103.0), fontsize=6.1, color=INK, ha="left",
                va="top", zorder=7,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(0.0, 105.0)
    ax.set_xticks(range(-90, 91, 30))
    ax.set_xlabel("latitude (deg)", fontsize=6.8)
    ax.set_ylabel("covered longitudes (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, ncol=2, columnspacing=0.9,
              borderaxespad=0.2, labelspacing=0.2)
    ax.set_title("Coverage against latitude", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def gap_panel(ax, mean_gap, worst_gap, duty):
    w = 2.1
    ax.bar(LAT_PROBE - w / 2.0, worst_gap, width=w, color="#c9d5e6",
           edgecolor=DEEP, linewidth=0.5, zorder=3, label="worst outage")
    ax.bar(LAT_PROBE + w / 2.0, mean_gap, width=w, color=TEAL,
           edgecolor="white", linewidth=0.4, zorder=4, label="mean outage")
    k = int(np.argmax(worst_gap))
    ax.annotate("{:.0f} min at {:.0f} deg".format(worst_gap[k], LAT_PROBE[k]),
                xy=(LAT_PROBE[k], worst_gap[k]),
                xytext=(LAT_PROBE[k] + 20.0, worst_gap[k] * 0.72),
                fontsize=6.1, color=INK, ha="left", va="bottom", zorder=7,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(-4.0, 69.0)
    ax.set_ylim(0.0, worst_gap.max() * 1.30)
    ax.set_xticks(range(0, 66, 15))
    ax.set_xlabel("latitude (deg)", fontsize=6.8)
    ax.set_ylabel("outage (min)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(LAT_PROBE, duty, color=ORANGE, lw=1.3, marker="o", ms=2.6,
             zorder=6)
    ax2.set_ylabel("time in view (%)", fontsize=6.8, color=ORANGE)
    ax2.tick_params(labelsize=6.4, colors=ORANGE)
    ax2.set_ylim(0.0, 105.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.2, labelspacing=0.2)
    ax.set_title("Outages over one day", fontsize=8.0, fontweight="bold",
                 color=INK, pad=3.0)


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
    lam = footprint_angle()
    lat_s, lon_s = subpoints([0.0])
    lat_s, lon_s = lat_s[:, 0], lon_s[:, 0]

    glat = np.arange(-89.75, 90.0, 0.5)
    glon = np.arange(-179.5, 180.0, 1.0)
    fold, one, two, total = coverage_by_latitude(lat_s, lon_s, lam, glat,
                                                 glon)
    mean_gap, worst_gap, duty = revisit_gaps(lam)

    n_sat = N_PLANE * N_PER_PLANE
    cap = 100.0 * n_sat * (1.0 - np.cos(np.radians(lam))) / 2.0
    assert lat_s.size == n_sat, "constellation size"
    assert total < cap + 1e-6, "coverage cannot beat the no-overlap bound"
    assert np.max(np.abs(lat_s)) <= INC_DEG + 1e-6, "inclination bound"
    assert np.all(worst_gap >= mean_gap - 1e-9), "worst gap below the mean"
    assert worst_gap[0] > mean_gap[0], "the equator should see uneven gaps"
    assert np.all(duty > 0.0), "every probed latitude must be reachable"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_map = axes_inches(fig, MAIN)
    ax_lat = axes_inches(fig, AUX_T)
    ax_gap = axes_inches(fig, AUX_B)

    main_panel(fig, ax_map, fold, glat, glon, lat_s, lon_s, lam, total)
    latitude_panel(ax_lat, glat, one, two, lam)
    gap_panel(ax_gap, mean_gap, worst_gap, duty)



    panel_labels_inches(fig, [ax_map, ax_lat, ax_gap], dx=-0.46, dy=0.12)
    savefig(fig, "1022_orb_constellation_coverage.png")


if __name__ == "__main__":
    main()
