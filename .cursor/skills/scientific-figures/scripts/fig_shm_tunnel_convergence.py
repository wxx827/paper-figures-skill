"""Convergence monitoring of a horseshoe tunnel section as the face advances.

Main panel: the excavated profile drawn to scale with equal axes, the deformed
profile at forty times exaggeration, the radial movement vectors at twelve
points and the three convergence chords the tape extensometer actually reads.
Auxiliary (both at 58 % of the main panel width): the chord and crown time
histories against the day the face passed, and the same readings replotted
against distance to the face, where they collapse onto one longitudinal
displacement profile.

The section is the only input.  Radial movement varies as A + B cos 2 phi
about the profile centroid, the deformed outline is the profile plus that
movement, and each chord convergence is the length change between the two
deformed target coordinates - not a number typed into the time series.  The
time axis is the face advancing at a fixed rate, so distance to face and
elapsed time are the same variable read twice, and the fraction of the total
that occurs before the face arrives is read off the profile.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GOLD = "#c9a227"
ROCK = lighten("#8c6d4f", 0.70)

R_ARCH = 5.0            # m, crown radius
H_WALL = 3.5            # m, straight sidewall
RISE_INV = 1.0          # m, invert rise
CENTRE = (0.0, 0.0)     # reference point for the radial field
U_MEAN, U_OVAL = 18.0, -8.0     # mm, A and B of A + B cos 2 phi
EXAGG = 25.0
ADVANCE = 2.4           # m/day
INSTALL = -1.5          # m, target installed this far behind the face
NOISE = 0.28            # mm, reading scatter

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def profile(n=361):
    """Closed horseshoe outline, counter-clockwise from the right wall foot."""
    th = np.linspace(0.0, np.pi, n)
    arch = np.column_stack([R_ARCH * np.cos(th), R_ARCH * np.sin(th)])
    wall_l = np.column_stack([np.full(40, -R_ARCH),
                              np.linspace(0.0, -H_WALL, 40)])
    t = np.linspace(0.0, 1.0, 90)
    inv = np.column_stack([-R_ARCH + 2.0 * R_ARCH * t,
                           -H_WALL + RISE_INV * np.sin(np.pi * t)])
    wall_r = np.column_stack([np.full(40, R_ARCH),
                              np.linspace(-H_WALL, 0.0, 40)])
    return np.vstack([arch, wall_l, inv, wall_r])


def radial_field(pts):
    """Inward radial movement in mm at each point, and its unit direction."""
    d = pts - np.array(CENTRE)
    r = np.hypot(d[:, 0], d[:, 1])
    phi = np.arctan2(d[:, 1], d[:, 0])
    mag = U_MEAN + U_OVAL * np.cos(2.0 * phi)
    unit = -d / r[:, None]
    return mag, unit


def deformed(pts, scale=1.0):
    """Profile plus the radial field, in metres, at ``scale`` exaggeration."""
    mag, unit = radial_field(pts)
    return pts + scale * (mag[:, None] / 1000.0) * unit


def targets():
    """Monitoring points on the profile, as (name, x, y)."""
    a = np.deg2rad(135.0)
    b = np.deg2rad(45.0)
    return [("crown", 0.0, R_ARCH),
            ("SL left", -R_ARCH, 0.0), ("SL right", R_ARCH, 0.0),
            ("SH left", R_ARCH * np.cos(a), R_ARCH * np.sin(a)),
            ("SH right", R_ARCH * np.cos(b), R_ARCH * np.sin(b)),
            ("foot left", -R_ARCH, -H_WALL),
            ("foot right", R_ARCH, -H_WALL)]


def chords():
    """Tape extensometer lines, as (label, end A, end B)."""
    t = {n: np.array([x, y]) for n, x, y in targets()}
    return [("C1 springline", t["SL left"], t["SL right"]),
            ("C2 crown - left foot", t["crown"], t["foot left"]),
            ("C3 crown - right foot", t["crown"], t["foot right"])]


def chord_convergence():
    """Final convergence of each chord in mm, from the deformed geometry."""
    out = []
    for name, a, b in chords():
        pts = np.vstack([a, b])
        dpts = deformed(pts)
        l0 = np.linalg.norm(a - b)
        l1 = np.linalg.norm(dpts[0] - dpts[1])
        out.append((name, 1000.0 * (l0 - l1)))
    return out


def crown_settlement():
    """Downward movement of the crown target, in mm."""
    p = np.array([[0.0, R_ARCH]])
    return 1000.0 * (p[0, 1] - deformed(p)[0, 1])


def panet(x):
    """Fraction of the final movement already developed at distance ``x``."""
    x = np.asarray(x, dtype=float)
    ahead = 0.28 * np.exp(x / (0.8 * R_ARCH))
    behind = 1.0 - 0.72 * (0.84 * R_ARCH / (0.84 * R_ARCH + np.maximum(
        x, 0.0))) ** 2
    return np.where(x < 0.0, ahead, behind)


def main_panel(ax, pts, dpts, finals):
    ax.add_patch(Rectangle((-11.0, -8.0), 22.0, 16.0, facecolor=ROCK,
                           alpha=0.55, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(pts, closed=True, facecolor="white",
                         edgecolor="#8a8a8a", linewidth=1.3, zorder=2))
    ax.plot(dpts[:, 0], dpts[:, 1], color=RED, lw=1.4, ls=(0, (4.0, 2.0)),
            zorder=6)

    sel = np.linspace(0, len(pts) - 1, 13).astype(int)[:-1]
    mag, unit = radial_field(pts[sel])
    for p, m, u in zip(pts[sel], mag, unit):
        tip = p + EXAGG * (m / 1000.0) * u
        ax.annotate("", xy=tip, xytext=p, zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))

    # Labels go a third of the way along each chord, not at the midpoint,
    # because all three chords pass close to the centre of the section.
    for (name, a, b), (_n, conv) in zip(chords(), finals):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=0.9,
                ls=(0, (5.0, 2.0, 1.0, 2.0)), zorder=5)
        lab = a + 0.32 * (b - a)
        ax.text(lab[0], lab[1] + 0.28,
                "{:s}\n{:.1f} mm".format(name.split()[0], conv), fontsize=6.1,
                color=BLUE, ha="center", va="bottom", zorder=9,
                linespacing=1.25)
    for name, x, y in targets():
        ax.plot([x], [y], marker="o", ms=3.6, mfc=INK, mec="white", mew=0.6,
                zorder=8)

    ax.text(-9.0, 6.4, "displacement vectors and dashed outline at x {:.0f}"
            .format(EXAGG), fontsize=6.6, color=SUB, ha="left", va="top",
            zorder=9)
    ax.text(-9.0, -5.6, "axes to scale, equal aspect", fontsize=6.4,
            color=MUTED, ha="left", va="bottom", zorder=9)
    ax.set_aspect("equal")
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-6.5, 7.0)
    ax.set_xlabel("horizontal offset from tunnel axis (m)")
    ax.set_ylabel("elevation above springline (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Section at chainage 1 + 240, final deformed shape",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    handles = [Line2D([], [], color="#8a8a8a", lw=1.3, label="excavated "
                                                             "profile"),
               Line2D([], [], color=RED, lw=1.4, ls=(0, (4.0, 2.0)),
                      label="deformed profile"),
               Line2D([], [], color=ORANGE, lw=1.0, label="radial movement"),
               Line2D([], [], color=BLUE, lw=0.9,
                      ls=(0, (5.0, 2.0, 1.0, 2.0)), label="convergence "
                                                          "chord")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.3, ncol=2,
              frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.94, handlelength=1.6, columnspacing=1.0,
              borderaxespad=0.4)


def time_panel(ax, days, series, colors, t_face):
    for (name, val, obs), c in zip(series, colors):
        ax.plot(days, obs, lw=0.0, marker="o", ms=1.5, mfc=c, mec="none",
                alpha=0.55, zorder=3)
        ax.plot(days, val, color=c, lw=1.5, zorder=5, label=name)
        ax.text(days[-1] - 0.3, val[-1], " {:.1f}".format(val[-1]),
                color=c, fontsize=6, ha="left", va="center", zorder=7)
    top = max(v.max() for _n, v, _o in series) * 1.20
    ax.axvline(t_face, color=INK, lw=0.9, ls=(0, (2.5, 2.0)), zorder=4)
    ax.text(t_face + 0.45, 0.30 * top, "face passes the section",
            fontsize=6, color=INK, ha="left", va="bottom", rotation=90,
            zorder=7)
    ax.set_xlim(days[0], days[-1] + 2.6)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("days since target installation", fontsize=6.8)
    ax.set_ylabel("convergence (mm)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.2, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Face advance at {:.1f} m/day drives it".format(ADVANCE),
                 fontsize=8.0, fontweight="bold", color=INK)


def face_panel(ax, dist, series, totals, colors):
    x_hi = 44.0
    xg = np.linspace(dist[0], x_hi, 300)
    seen = 1.0 - panet(INSTALL)
    ax.axvspan(dist[0], 0.0, facecolor=GOLD, alpha=0.18, lw=0.0, zorder=1)
    ax.plot(xg, 100.0 * (panet(xg) - panet(INSTALL)), color=INK, lw=1.5,
            zorder=5, label="longitudinal profile")
    for (_name, _val, obs), tot, c in zip(series, totals, colors):
        ax.plot(dist, 100.0 * obs / tot, lw=0.0, marker="o", ms=1.8, mfc=c,
                mec="none", alpha=0.6, zorder=4)
    ax.axhline(100.0 * seen, color=RED, lw=0.9, ls=(0, (3.0, 2.0)), zorder=6)
    ax.text(x_hi - 1.0, 100.0 * seen - 3.0,
            "the tape can only ever see {:.0f} %".format(100.0 * seen),
            fontsize=6.0, color=RED, ha="right", va="top", zorder=7)
    ax.annotate("{:.0f} % is already gone when\nthe targets are installed"
                .format(100.0 * panet(INSTALL)), xy=(0.0, 6.0),
                xytext=(6.0, 34.0), fontsize=6.0, color=SUB, ha="left",
                va="center", zorder=7, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7, shrinkA=2,
                                shrinkB=2))
    ax.set_xlim(dist[0], x_hi)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("distance behind the face (m)", fontsize=6.8)
    ax.set_ylabel("share of the total movement (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.3)
    ax.set_title("Every line collapses onto one profile", fontsize=8.0,
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
    rng = np.random.default_rng(70413)

    pts = profile()
    dpts = deformed(pts, EXAGG)
    finals = chord_convergence()
    crown = crown_settlement()

    days = np.linspace(0.0, 26.0, 210)
    dist = INSTALL + ADVANCE * days
    # A tape reads movement *since installation*, so the datum is the state
    # of the section when the targets went in, not the undisturbed ground.
    frac = panet(dist) - panet(INSTALL)

    series, totals = [], []
    for name, val in finals + [("crown settlement", crown)]:
        clean = val * frac
        series.append((name, clean, clean + rng.normal(0.0, NOISE, days.size)))
        totals.append(val)
    colors = [BLUE, TEAL, ORANGE, RED]
    t_face = -INSTALL / ADVANCE

    assert all(v > 0.0 for _n, v in finals), "chords must shorten"
    assert frac[0] == 0.0 and frac[-1] < 1.0, "readings start at the datum"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_time = axes_inches(fig, AUX_T)
    ax_face = axes_inches(fig, AUX_B)

    main_panel(ax_main, pts, dpts, finals)
    time_panel(ax_time, days, series, colors, t_face)
    face_panel(ax_face, dist, series, totals, colors)



    panel_labels_inches(fig, [ax_main, ax_time, ax_face], dx=-0.62, dy=0.14)
    savefig(fig, "987_shm_tunnel_convergence.png")


if __name__ == "__main__":
    main()
