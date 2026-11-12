"""Travel-time curves, ray paths and the shadow zone of a layered earth.

One radial velocity model drives all three panels.  For every ray parameter
the turning radius is found by root-finding on the slowness r / v(r), and the
epicentral distance and travel time are the Bullen integrals evaluated with
the substitution r = rt + s^2, which cancels the inverse square-root
singularity at the turning point instead of stepping around it.  The direct P
and S branches turn inside the mantle, PP is the same integral taken twice,
and PKP is a full mantle traverse plus a leg that turns in the liquid outer
core.  Nothing is tabulated: the shadow zone quoted in the title is simply the
gap between the largest distance the mantle branch reaches and the smallest
distance the core branch comes back at.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#22406e"
BLUE = "#3b5b92"
PALE = "#a8bcd8"
RED = "#b23a48"
ROSE = "#d79aa2"
GOLD = "#c9a227"
TEAL = "#2a7f7a"
GREY = "#8a8a8a"
HAIR = "#cccccc"
MANTLE = "#e6e0d6"
OCORE = "#dfe6ee"
ICORE = "#c6d2e2"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

R_E, R_CMB, R_ICB = 6371.0, 3480.0, 1221.0
DEG = 180.0 / np.pi
N_NODE = 700


def vp(r):
    r = np.asarray(r, dtype=float)
    man = 13.70 - 5.60 * (np.clip(r - R_CMB, 0.0, None)
                          / (R_E - R_CMB)) ** 1.60
    out = 10.36 - 2.30 * (np.clip(r - R_ICB, 0.0, None)
                          / (R_CMB - R_ICB)) ** 1.20
    inn = 11.26 - 0.23 * (np.clip(r, 0.0, None) / R_ICB) ** 2
    return np.where(r >= R_CMB, man, np.where(r >= R_ICB, out, inn))


def vs(r):
    r = np.asarray(r, dtype=float)
    man = 7.26 - 2.79 * (np.clip(r - R_CMB, 0.0, None)
                         / (R_E - R_CMB)) ** 1.60
    inn = 3.67 - 0.17 * (np.clip(r, 0.0, None) / R_ICB) ** 2
    return np.where(r >= R_CMB, man, np.where(r >= R_ICB, np.nan, inn))


def eta(r, vfun):
    """Radial slowness r / v(r), in seconds per radian."""
    return np.asarray(r, dtype=float) / vfun(r)


def turning_radius(p, r_top, r_bot, vfun):
    return brentq(lambda r: float(eta(r, vfun)) - p, r_bot, r_top, xtol=1e-6)


def leg_turning(p, r_top, vfun, r_t):
    """Ascending leg from the turning radius to ``r_top``.

    ``r = r_t + s^2`` turns the integrable inverse square-root singularity
    into a bounded integrand, so a plain trapezoid rule in ``s`` is accurate
    right up to the turning point.
    """
    s = np.linspace(0.0, np.sqrt(r_top - r_t), N_NODE)[1:]
    r = r_t + s * s
    e = eta(r, vfun)
    q = np.sqrt(np.clip((e - p) * (e + p), 1e-12, None))
    gd = 2.0 * s * p / (r * q)
    gt = 2.0 * s * e * e / (r * q)
    d = cumulative_trapezoid(gd, s, initial=0.0)
    t = cumulative_trapezoid(gt, s, initial=0.0)
    return r, d, t


def leg_through(p, r_top, r_bot, vfun):
    """Ascending leg that crosses a whole shell without turning inside it."""
    r = np.linspace(r_bot, r_top, N_NODE)
    e = eta(r, vfun)
    q = np.sqrt(np.clip((e - p) * (e + p), 1e-12, None))
    d = cumulative_trapezoid(p / (r * q), r, initial=0.0)
    t = cumulative_trapezoid(e * e / (r * q), r, initial=0.0)
    return r, d, t


def ascend(p, shells):
    """Half a ray: turning point up to the surface, through the shells given.

    ``shells`` runs downward as ``(r_top, r_bot, v)``.  Returns radius,
    cumulative distance (rad) and cumulative time (s), all measured from the
    deepest point of the ray.
    """
    k = None
    for i, (_, r_bot, vfun) in enumerate(shells):
        if float(eta(r_bot, vfun)) <= p:
            k = i
            break
    if k is None:
        return None
    r_top, r_bot, vfun = shells[k]
    r_t = turning_radius(p, r_top, r_bot, vfun)
    r, d, t = leg_turning(p, r_top, vfun, r_t)
    for j in range(k - 1, -1, -1):
        rt2, rb2, vf2 = shells[j]
        r2, d2, t2 = leg_through(p, rt2, rb2, vf2)
        r = np.concatenate([r, r2])
        d = np.concatenate([d, d2 + d[-1]])
        t = np.concatenate([t, t2 + t[-1]])
    return r, d, t


def branch(ps, shells):
    """Distance (deg) and travel time (min) for a list of ray parameters."""
    dd, tt, keep = [], [], []
    for p in ps:
        out = ascend(p, shells)
        if out is None:
            continue
        _, d, t = out
        dd.append(2.0 * d[-1] * DEG)
        tt.append(2.0 * t[-1] / 60.0)
        keep.append(p)
    return np.array(dd), np.array(tt), np.array(keep)


def ray_xy(p, shells):
    """Cartesian path of one ray, source at the top of the section."""
    out = ascend(p, shells)
    if out is None:
        return None
    r, d, _ = out
    half = d[-1]
    rr = np.concatenate([r[::-1], r])
    ang = np.concatenate([half - d[::-1], half + d])
    return rr * np.sin(ang), rr * np.cos(ang)


MANTLE_P = [(R_E, R_CMB, vp)]
MANTLE_S = [(R_E, R_CMB, vs)]
CORE_P = [(R_E, R_CMB, vp), (R_CMB, R_ICB, vp)]


def main_panel(ax, br, gap):
    d_p, t_p = br["P"][0], br["P"][1]
    d_s, t_s = br["S"][0], br["S"][1]
    d_k, t_k = br["PKP"][0], br["PKP"][1]

    ax.axvspan(gap[0], gap[1], facecolor="#ebe6df", edgecolor="none",
               zorder=1)
    ax.plot(d_p, t_p, color=DEEP, lw=1.8, zorder=6)
    ax.plot(2.0 * d_p, 2.0 * t_p, color=BLUE, lw=1.3, ls=(0, (4.0, 2.2)),
            zorder=5)
    ax.plot(d_s, t_s, color=RED, lw=1.8, zorder=6)
    ax.plot(d_k, t_k, color=GOLD, lw=1.8, zorder=6)

    for x, y, lab, col, dx, dy in [
            (d_p[0], t_p[0], "P", DEEP, 2.5, -0.7),
            (d_s[0], t_s[0], "S", RED, 2.5, 0.5),
            (2.0 * d_p[0], 2.0 * t_p[0], "PP", BLUE, -13.0, 1.2),
            (d_k[-1], t_k[-1], "PKP", GOLD, 3.0, -0.9)]:
        ax.text(min(x, 176.0) + dx, y + dy, lab, fontsize=8.5,
                fontweight="bold", color=col, ha="left", va="center",
                zorder=8)

    i60 = int(np.argmin(np.abs(d_p - 60.0)))
    slow = br["P"][2][i60] / DEG
    ax.annotate("apparent slowness dT/d(distance)\n= {:.1f} s/deg at 60 deg, "
                "turning at {:.0f} km depth"
                .format(slow, R_E - br["P"][3][i60]),
                xy=(d_p[i60], t_p[i60]), xytext=(66.0, 4.2), fontsize=6.3,
                color=INK, ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=4), zorder=9)
    ax.text(0.5 * (gap[0] + gap[1]), 29.2,
            "P shadow zone\n{:.0f} to {:.0f} deg".format(*gap),
            ha="center", va="top", fontsize=6.6, color=SUB,
            fontweight="bold", linespacing=1.4, zorder=8)

    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 30.0)
    ax.set_xticks(range(0, 181, 20))
    ax.set_xlabel("epicentral distance (deg)")
    ax.set_ylabel("travel time (min)")
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Body-wave travel times for a surface source",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    handles = [Line2D([], [], color=DEEP, lw=1.8, label="P, turns in mantle"),
               Line2D([], [], color=RED, lw=1.8, label="S, turns in mantle"),
               Line2D([], [], color=BLUE, lw=1.3, ls=(0, (4.0, 2.2)),
                      label="PP, one surface bounce"),
               Line2D([], [], color=GOLD, lw=1.8,
                      label="PKP, turns in the outer core")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.8, labelspacing=0.28, borderaxespad=0.4)


def section_panel(ax, gap):
    half_y = 6820.0
    half_x = half_y * (AUX_T[2] / AUX_T[3])
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal")
    ax.axis("off")

    for rad, fc in [(R_E, MANTLE), (R_CMB, OCORE), (R_ICB, ICORE)]:
        ax.add_patch(Circle((0.0, 0.0), rad, facecolor=fc, edgecolor=GREY,
                            linewidth=0.7, zorder=2))
    ax.add_patch(Wedge((0.0, 0.0), R_E, 90.0 - gap[1], 90.0 - gap[0],
                       facecolor="#7d7365", edgecolor="none", alpha=0.30,
                       zorder=3))

    for p in np.linspace(300.0, 770.0, 9):
        xy = ray_xy(p, MANTLE_P)
        if xy is not None:
            ax.plot(xy[0], xy[1], color=DEEP, lw=0.8, zorder=6)
    for p in np.linspace(150.0, 250.0, 5):
        xy = ray_xy(p, CORE_P)
        if xy is not None:
            ax.plot(xy[0], xy[1], color=GOLD, lw=0.9, zorder=6)

    ax.plot([0.0], [R_E], marker="*", ms=10.0, mfc=RED, mec="white",
            mew=0.8, zorder=9)
    ax.text(-300.0, R_E + 260.0, "source", ha="right", va="center",
            fontsize=6.3, color=RED, fontweight="bold", zorder=9)

    for x, y, lab in [(-4750.0, 1500.0, "mantle"),
                      (-2500.0, -1600.0, "outer core")]:
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.2,
                color="#4a4237", fontweight="bold", zorder=8)
    ax.annotate("inner core", xy=(-500.0, -900.0), xytext=(-3600.0, -3900.0),
                fontsize=6.2, color="#4a4237", fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="#6f6558", lw=0.7,
                                shrinkA=2, shrinkB=2), zorder=8)

    mid = np.radians(0.5 * (gap[0] + gap[1]))
    ax.text(6750.0 * np.sin(mid), 6750.0 * np.cos(mid),
            "P shadow\n{:.0f} to {:.0f} deg".format(*gap), ha="left",
            va="center", fontsize=6.2, color=SUB, fontweight="bold",
            linespacing=1.35, zorder=9)
    ax.text(-half_x + 300.0, -half_y + 250.0,
            "blue: P turning in the mantle\ngold: PKP turning in the core",
            ha="left", va="bottom", fontsize=6.1, color=SUB,
            linespacing=1.4, zorder=9)
    ax.set_title("Rays through the mantle and the outer core",
                 fontsize=8.0, fontweight="bold", color=INK, pad=3.0)


def profile_panel(ax):
    r = np.linspace(1.0, R_E, 1400)
    depth = R_E - r
    ax.plot(vp(r), depth, color=DEEP, lw=1.5, zorder=6, label="Vp")
    ax.plot(vs(r), depth, color=RED, lw=1.5, zorder=6, label="Vs")
    for y, lab, col in [(R_E - R_CMB, "core-mantle boundary", "#6f6558"),
                        (R_E - R_ICB, "inner-core boundary", "#6f6558")]:
        ax.axhline(y, color=col, lw=0.9, ls=(0, (3.6, 2.4)), zorder=5)
        ax.text(0.35, y - 130.0, "{:s} {:.0f} km".format(lab, y),
                ha="left", va="bottom", fontsize=6.0, color=col, zorder=7)
    ax.fill_betweenx([R_E - R_CMB, R_E - R_ICB], 0.0, 14.0,
                     facecolor="#eef2f6", edgecolor="none", zorder=1)
    ax.text(2.6, 0.5 * (2.0 * R_E - R_CMB - R_ICB),
            "liquid outer core:\nno S wave", ha="left", va="center",
            fontsize=6.1, color=SUB, linespacing=1.4, zorder=7)
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(R_E, 0.0)
    ax.set_xlabel("velocity (km/s)", fontsize=6.9)
    ax.set_ylabel("depth (km, positive down)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, labelspacing=0.22, borderaxespad=0.3)
    ax.set_title("The velocity model behind both panels", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


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

    eta_cmb_p = float(eta(R_CMB, vp))
    eta_cmb_s = float(eta(R_CMB, vs))
    eta_icb_p = float(eta(R_ICB, vp))

    br = {}
    ps = np.linspace(eta_cmb_p + 0.4, float(eta(R_E, vp)) - 0.6, 260)
    d, t, pk = branch(ps, MANTLE_P)
    rt = np.array([turning_radius(p, R_E, R_CMB, vp) for p in pk])
    br["P"] = (d, t, pk, rt)

    ps = np.linspace(eta_cmb_s + 0.4, float(eta(R_E, vs)) - 0.6, 260)
    br["S"] = branch(ps, MANTLE_S)

    ps = np.linspace(eta_icb_p + 2.0, eta_cmb_p - 0.4, 260)
    br["PKP"] = branch(ps, CORE_P)

    gap = (float(br["P"][0].max()), float(br["PKP"][0].min()))

    assert br["P"][0].max() > 80.0, "direct P should reach past 80 deg"
    assert gap[1] > gap[0] + 20.0, "no clear shadow zone in this model"
    assert br["S"][1].max() > br["P"][1].max(), "S must lag P"
    assert np.all(np.diff(br["P"][0]) < 0.0), "P branch is not monotonic"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sec = axes_inches(fig, AUX_T)
    ax_vel = axes_inches(fig, AUX_B)

    main_panel(ax_main, br, gap)
    section_panel(ax_sec, gap)
    profile_panel(ax_vel)



    panel_labels_inches(fig, [ax_main, ax_sec, ax_vel], dx=-0.60, dy=0.12)
    savefig(fig, "1052_eqs_travel_time_curves.png")


if __name__ == "__main__":
    main()
