"""A day of satellite geometry at one receiver, and what it does to accuracy.

Twenty-four satellites in six planes are propagated for a day, the line of
sight to a fixed receiver is resolved into the local horizon frame, and every
epoch with four or more satellites above the mask gets its own geometry
matrix.  The dilution factors in the main panel are the square roots of the
diagonal of that matrix inverted, the sky plot shows the arrangement that
produces the worst of them, and the scatter is the same inverse pushed
through Gaussian ranging noise, so the ellipses are the geometry rather than
a fitted cloud.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

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
PALE = "#dbe3ee"

MU = 398600.4418          # km3/s2
R_E = 6378.14             # km
OMEGA_E = 7.2921159e-5    # rad/s
A_SAT = 26559.8           # km, semi-major axis of the navigation shell
INC_DEG = 55.0            # deg
N_PLANE = 6
N_PER_PLANE = 4
LAT_RX = 39.90            # deg
LON_RX = 116.40           # deg
MASK_DEG = 10.0           # deg
UERE = 1.20               # m, one-sigma user equivalent range error
HOURS = 24.0
STEP_S = 60.0
N_DRAW = 3000
SEED = 20260813

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.92, 1.02, 4.58, 4.48)
AUX_T = (7.52, 3.63, 2.66, 1.87)
AUX_B = (7.52, 1.02, 2.66, 1.80)


def constellation(t):
    """Inertial positions (km) of every satellite, shape (n_sat, n_time, 3)."""
    n = np.sqrt(MU / A_SAT ** 3)
    inc = np.radians(INC_DEG)
    out = []
    for p in range(N_PLANE):
        raan = np.radians(p * 360.0 / N_PLANE)
        for s in range(N_PER_PLANE):
            u0 = np.radians(s * 360.0 / N_PER_PLANE
                            + p * 360.0 / (N_PLANE * N_PER_PLANE))
            u = u0 + n * t
            xp, yp = A_SAT * np.cos(u), A_SAT * np.sin(u)
            x = np.cos(raan) * xp - np.sin(raan) * np.cos(inc) * yp
            y = np.sin(raan) * xp + np.cos(raan) * np.cos(inc) * yp
            z = np.sin(inc) * yp
            out.append(np.stack([x, y, z], axis=-1))
    return np.stack(out, axis=0)


def receiver(t):
    """Receiver position and local east, north, up axes in the same frame."""
    phi = np.radians(LAT_RX)
    lam = np.radians(LON_RX) + OMEGA_E * t
    up = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam),
                   np.full_like(lam, np.sin(phi))], axis=-1)
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], axis=-1)
    north = np.cross(up, east)
    return R_E * up, east, north, up


def look_angles(t):
    """Elevation and azimuth (deg) of every satellite at every epoch."""
    sats = constellation(t)
    r_rx, east, north, up = receiver(t)
    los = sats - r_rx[None, :, :]
    rng = np.linalg.norm(los, axis=-1, keepdims=True)
    los = los / rng
    e = np.sum(los * east[None], axis=-1)
    n = np.sum(los * north[None], axis=-1)
    u = np.sum(los * up[None], axis=-1)
    elev = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
    azim = np.degrees(np.arctan2(e, n)) % 360.0
    return elev, azim


def geometry_matrix(elev, azim):
    """Rows of the design matrix in the local horizon frame."""
    el, az = np.radians(elev), np.radians(azim)
    return np.stack([-np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az),
                     -np.sin(el), np.ones_like(el)], axis=-1)


def dops(elev, azim):
    """GDOP, PDOP, HDOP, VDOP and the visible count for every epoch."""
    n_t = elev.shape[1]
    out = np.full((n_t, 4), np.nan)
    count = np.zeros(n_t, dtype=int)
    for k in range(n_t):
        vis = elev[:, k] >= MASK_DEG
        count[k] = int(np.sum(vis))
        if count[k] < 4:
            continue
        g = geometry_matrix(elev[vis, k], azim[vis, k])
        try:
            q = np.linalg.inv(g.T @ g)
        except np.linalg.LinAlgError:
            continue
        d = np.diag(q)
        out[k] = [np.sqrt(d.sum()), np.sqrt(d[:3].sum()),
                  np.sqrt(d[:2].sum()), np.sqrt(d[2])]
    return out, count


def error_cloud(elev, azim, k, rng):
    """East and north errors (m) from Gaussian ranging noise at one epoch."""
    vis = elev[:, k] >= MASK_DEG
    g = geometry_matrix(elev[vis, k], azim[vis, k])
    q = np.linalg.inv(g.T @ g)
    gain = q @ g.T
    eps = rng.normal(0.0, UERE, size=(g.shape[0], N_DRAW))
    dx = gain @ eps
    return dx[0], dx[1], q


def dop_panel(ax, t_h, dop, count, k_bad, k_good):
    names = [("GDOP", SLATE, 1.1, (0, (1.8, 1.6))),
             ("PDOP", DEEP, 1.8, "-"),
             ("HDOP", TEAL, 1.4, "-"),
             ("VDOP", ORANGE, 1.3, (0, (4.0, 2.0)))]
    for j, (lab, col, lw, ls) in enumerate(names):
        ax.plot(t_h, dop[:, j], color=col, lw=lw, ls=ls, zorder=6, label=lab)
    ax.axhline(6.0, color=RED, lw=1.0, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(0.15, 6.12, "PDOP 6, the usual acceptance line", fontsize=6.3,
            color=RED, ha="left", va="bottom")

    for k, col, tag in ((k_bad, RED, "worst"), (k_good, TEAL, "best")):
        ax.axvline(t_h[k], color=col, lw=0.8, ls=(0, (3.0, 2.0)), zorder=5)
        ax.plot([t_h[k]], [dop[k, 1]], marker="o", ms=5.0, mfc=col,
                mec="white", mew=0.9, zorder=8)
    ax.annotate("worst geometry of the day\nPDOP {:.2f} with {:d} satellites"
                .format(dop[k_bad, 1], count[k_bad]),
                xy=(t_h[k_bad], dop[k_bad, 1]),
                xytext=(t_h[k_bad] + 1.1, dop[k_bad, 1] + 1.35), fontsize=6.5,
                color=RED, ha="left", va="center", linespacing=1.32,
                fontweight="bold", zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                shrinkA=2, shrinkB=4))
    ax.annotate("best geometry\nPDOP {:.2f}".format(dop[k_good, 1]),
                xy=(t_h[k_good], dop[k_good, 1]),
                xytext=(t_h[k_good] - 0.6, dop[k_good, 1] - 1.15),
                fontsize=6.5, color=TEAL, ha="right", va="center",
                linespacing=1.32, zorder=9,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8,
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(0.0, HOURS)
    ax.set_ylim(0.0, max(7.5, np.nanmax(dop[:, 0]) * 1.12))
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("time (h)")
    ax.set_ylabel("dilution of precision (dimensionless)")
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top",):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.step(t_h, count, where="mid", color=GOLD, lw=1.1, zorder=3)
    ax2.fill_between(t_h, 0.0, count, step="mid", color="#f5edd6", zorder=1)
    ax2.set_ylim(0, 18)
    ax2.set_yticks(range(0, 17, 4))
    ax2.set_ylabel("satellites above the mask (count)", color="#8a6d10")
    ax2.tick_params(labelsize=6.8, colors="#8a6d10")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GOLD)
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    handles = [Line2D([], [], color=c, lw=w, ls=s, label=l)
               for l, c, w, s in names]
    handles.append(Line2D([], [], color=GOLD, lw=1.1,
                          label="visible satellites"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=6.4,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=2.0, ncol=2,
                    columnspacing=1.0, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(11)
    ax.set_title("Geometry over a day at {:.1f} deg north, {:.1f} deg east"
                 .format(LAT_RX, LON_RX), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def sky_panel(ax, elev, azim, k, t, dop_bad):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    half = int(round(5400.0 / STEP_S))
    lo, hi = max(k - half, 0), min(k + half, elev.shape[1] - 1)
    for i in range(elev.shape[0]):
        seg_e, seg_a = elev[i, lo:hi], azim[i, lo:hi]
        keep = seg_e >= MASK_DEG
        if not np.any(keep):
            continue
        th = np.radians(np.where(keep, seg_a, np.nan))
        r = np.where(keep, 90.0 - seg_e, np.nan)
        ax.plot(th, r, color="#c3ccd8", lw=0.7, zorder=3)
    vis = elev[:, k] >= MASK_DEG
    ax.plot(np.radians(azim[vis, k]), 90.0 - elev[vis, k], linestyle="none",
            marker="o", ms=5.0, mfc=DEEP, mec="white", mew=0.8, zorder=6)
    low = (~vis) & (elev[:, k] > -6.0)
    ax.plot(np.radians(azim[low, k]), 90.0 - elev[low, k], linestyle="none",
            marker="o", ms=3.4, mfc="none", mec="#a8b0ba", mew=0.8, zorder=5)
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(th, np.full_like(th, 90.0 - MASK_DEG), color=RED, lw=1.0,
            ls=(0, (4.0, 2.2)), zorder=7)
    ax.set_rlim(0.0, 90.0)
    ax.set_rticks([30.0, 60.0, 80.0])
    ax.set_yticklabels(["60", "30", "10"], fontsize=6, color=SUB)
    ax.set_rlabel_position(148.0)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=6.4, color=INK)
    ax.tick_params(pad=0.6)
    ax.grid(lw=0.5, alpha=0.5)
    ax.set_title("Sky at hour {:.1f}, PDOP {:.2f}".format(t[k] / 3600.0,
                                                          dop_bad),
                 fontsize=8.0, fontweight="bold", color=INK, pad=9.0)
    ax.text(np.radians(52.0), 152.0,
            "rings are elevation\n(deg); the dashed\nring is the {:.0f} deg\n"
            "mask".format(MASK_DEG), fontsize=6, color=SUB, ha="center",
            va="center", linespacing=1.35)


def scatter_panel(ax, clouds):
    semis = {}
    for tag, (de, dn, q, dop_h, col, alpha) in clouds.items():
        ax.plot(de, dn, linestyle="none", marker="o", ms=1.5, mfc=col,
                mec="none", alpha=alpha, zorder=3)
        cov = UERE ** 2 * q[:2, :2]
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        semis[tag] = np.sqrt(5.991 * vals[-1])
        ax.add_patch(Ellipse((0.0, 0.0),
                             2.0 * np.sqrt(5.991 * vals[-1]),
                             2.0 * np.sqrt(5.991 * vals[0]), angle=ang,
                             facecolor="none", edgecolor=col, lw=1.3,
                             zorder=6))
    lim = 1.55 * max(semis.values())
    for tag, (de, dn, q, dop_h, col, alpha) in clouds.items():
        y = lim * (0.94 if tag == "worst" else -0.94)
        ax.text(0.0, y, "{:s}: HDOP {:.2f}, {:.1f} m at 95 per cent"
                .format(tag, dop_h, semis[tag]), fontsize=6.1, color=col,
                ha="center", va="top" if tag == "worst" else "bottom",
                zorder=8)
    ax.plot([0.0], [0.0], marker="+", ms=7.0, mew=1.1, color=INK, zorder=7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("east error (m)", fontsize=6.8)
    ax.set_ylabel("north error (m)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("{:d} fixes with {:.1f} m ranging noise"
                 .format(N_DRAW, UERE), fontsize=8.0, fontweight="bold",
                 color=INK)


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
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, HOURS * 3600.0 + STEP_S, STEP_S)
    elev, azim = look_angles(t)
    dop, count = dops(elev, azim)
    ok = np.isfinite(dop[:, 1])
    k_bad = int(np.nanargmax(np.where(ok, dop[:, 1], np.nan)))
    k_good = int(np.nanargmin(np.where(ok, dop[:, 1], np.nan)))

    de_b, dn_b, q_b = error_cloud(elev, azim, k_bad, rng)
    de_g, dn_g, q_g = error_cloud(elev, azim, k_good, rng)
    drms_b = float(np.sqrt(np.mean(de_b ** 2 + dn_b ** 2)))
    pred_b = UERE * dop[k_bad, 2]

    assert np.all(count >= 4), "the receiver should never lose the fix"
    assert dop[k_bad, 1] > dop[k_good, 1], "worst must exceed best"
    assert np.all(dop[ok, 0] >= dop[ok, 1] - 1e-9), "GDOP bounds PDOP"
    assert np.all(dop[ok, 1] >= dop[ok, 2] - 1e-9), "PDOP bounds HDOP"
    assert abs(drms_b - pred_b) / pred_b < 0.06, \
        "the scatter must reproduce HDOP times the ranging error"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_dop = axes_inches(fig, MAIN)
    ax_sky = axes_inches(fig, AUX_T, polar=True)
    ax_sca = axes_inches(fig, AUX_B)

    dop_panel(ax_dop, t / 3600.0, dop, count, k_bad, k_good)
    sky_panel(ax_sky, elev, azim, k_bad, t, dop[k_bad, 1])
    clouds = {"worst": (de_b, dn_b, q_b, dop[k_bad, 2], RED, 0.32),
              "best": (de_g, dn_g, q_g, dop[k_good, 2], TEAL, 0.32)}
    scatter_panel(ax_sca, clouds)



    panel_labels_inches(fig, [ax_dop, ax_sky, ax_sca], dx=-0.70, dy=0.14)
    savefig(fig, "1025_orb_gnss_dop.png")


if __name__ == "__main__":
    main()
