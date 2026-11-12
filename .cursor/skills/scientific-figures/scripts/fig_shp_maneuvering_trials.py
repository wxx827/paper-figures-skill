"""Standard manoeuvring trials: turning circles, zigzag overshoots, speed loss.

The main panel is the turning-circle track for two rudder angles, drawn on an
equal-aspect grid because the whole point of the trial is a shape.  Advance,
transfer and tactical diameter are measured off that track the way a trials
team measures them - by interpolating the plotted path at the instant the
heading has swung 90 and 180 degrees - and the steady turning diameter comes
from an algebraic circle fit to the last part of the same path, so none of the
four numbers can disagree with the curve.

The upper companion panel is the 10/10 zigzag: commanded rudder and the
heading it produces on one degree axis, with the first and second overshoot
angles read off the heading trace after each reversal.  The lower panel shows
what the turn costs - the speed falls away as the yaw rate builds, and both
are the arrays the main panel was integrated from.

Illustrative synthetic data from a first-order Nomoto model.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
STEEL = "#5b6b7a"
BLUE = "#2e74a8"
LIGHT = "#8fb8d4"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.92, 1.06, 5.20, 4.34)
AUX_T = (7.52, 3.62, 3.00, 1.78)
AUX_B = (7.52, 1.06, 3.00, 1.72)

L_PP = 180.0                # length between perpendiculars, m
V0_KN = 15.5                # approach speed, knots
KN = 0.514444
V0 = V0_KN * KN             # m/s
K_NOM = 0.0321              # Nomoto gain, 1/s per rad of rudder
T_NOM = 22.0                # Nomoto time constant, s
RUDDER_RATE = 2.5           # deg/s
C_SPEED = 3.0               # speed-loss coefficient, dimensionless
C_DRIFT = 0.50              # drift-angle coefficient, dimensionless
DT = 0.20                   # integration step, s

TURNS = [(35.0, BLUE, "35 deg rudder"), (25.0, LIGHT, "25 deg rudder")]


def integrate(t_end, command):
    """Nomoto run; returns time, rudder, heading, yaw rate, speed, track."""
    n = int(t_end / DT) + 1
    t = np.arange(n) * DT
    dl = np.zeros(n)
    r = np.zeros(n)
    psi = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    v = np.full(n, V0)
    d_now = 0.0
    for i in range(1, n):
        want = command(t[i - 1], psi[i - 1])
        step = RUDDER_RATE * DT
        d_now += float(np.clip(want - d_now, -step, step))
        dl[i] = d_now
        rd = np.radians(d_now)
        r[i] = r[i - 1] + DT * (K_NOM * rd - r[i - 1]) / T_NOM
        psi[i] = psi[i - 1] + DT * r[i]
        rp = r[i] * L_PP / V0
        v[i] = V0 / np.sqrt(1.0 + (C_SPEED * rp) ** 2)
        course = psi[i] - C_DRIFT * rp
        x[i] = x[i - 1] + DT * v[i] * np.cos(course)
        y[i] = y[i - 1] + DT * v[i] * np.sin(course)
    return t, dl, np.degrees(psi), np.degrees(r), v, x, y


def fit_circle(x, y):
    """Algebraic least-squares circle fit; returns (xc, yc, radius) in m."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    xc, yc = sol[0] / 2.0, sol[1] / 2.0
    return xc, yc, float(np.sqrt(sol[2] + xc ** 2 + yc ** 2))


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)

    runs = []
    for ang, col, lab in TURNS:
        out = integrate(900.0, lambda tt, pp, a=ang: a)
        runs.append((ang, col, lab, out))

    def zz(tt, pp, amp=10.0, state=[1.0]):
        if state[0] > 0 and np.degrees(pp) >= amp:
            state[0] = -1.0
        elif state[0] < 0 and np.degrees(pp) <= -amp:
            state[0] = 1.0
        return amp * state[0] if tt > 10.0 else 0.0

    t_z, d_z, psi_z, r_z, v_z, _, _ = integrate(420.0, zz)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_z = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)
    for a in (ax, ax_z, ax_s):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the turning circles -----------------------------------------
    ax.set_aspect("equal")
    stats = []
    for ang, col, lab, (t, dl, psi, r, v, x, y) in runs:
        ax.plot(y, x, color=col, lw=2.0, zorder=6, label=lab)
        adv = float(np.interp(90.0, psi, x))
        tra = float(np.interp(90.0, psi, y))
        tac = float(np.interp(180.0, psi, y))
        last = psi > 400.0
        xc, yc, rad = fit_circle(y[last], x[last])
        stats.append(dict(ang=ang, adv=adv, tra=tra, tac=tac, dia=2.0 * rad,
                          col=col))
        ax.plot([tra], [adv], marker="o", ms=5.0, mfc="white", mec=col,
                mew=1.4, zorder=9)
        ax.plot([tac], [float(np.interp(180.0, psi, x))], marker="s", ms=4.6,
                mfc="white", mec=col, mew=1.4, zorder=9)

    s0 = stats[0]
    run0 = runs[0][3]
    ax.plot([0.0, s0["tra"]], [s0["adv"], s0["adv"]], color=RED, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=7)
    ax.plot([s0["tra"], s0["tra"]], [0.0, s0["adv"]], color=RED, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=7)
    ax.plot([0.0, s0["tac"]], [-70.0, -70.0], color=RED, lw=1.2, zorder=7)
    ax.text(8.0, -108.0,
            "tactical diameter %.0f m = %.2f L" % (s0["tac"], s0["tac"] / L_PP),
            ha="left", va="top", fontsize=6.5, color=RED, zorder=9)
    ax.text(s0["tra"] / 2.0, s0["adv"] + 16.0,
            "transfer %.0f m" % s0["tra"], ha="center", va="bottom",
            fontsize=6.5, color=RED, zorder=9)
    ax.text(-20.0, s0["adv"] / 2.0, "advance %.0f m" % s0["adv"],
            ha="right", va="center", rotation=90, fontsize=6.5, color=RED,
            zorder=9)

    ax.plot([0.0, 0.0], [-70.0, s0["adv"] * 0.30], color=INK, lw=1.0,
            zorder=5)
    ax.annotate("", xy=(0.0, s0["adv"] * 0.30), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.text(-20.0, -60.0, "approach %.1f knots\nrudder over at t = 0 s"
            % V0_KN, ha="right", va="bottom", fontsize=6.5, color=INK,
            linespacing=1.45, zorder=9)

    txt = "\n".join(
        "%2.0f deg rudder: advance %.0f m (%.2f L), transfer %.0f m, "
        "tactical %.0f m, steady %.0f m"
        % (s["ang"], s["adv"], s["adv"] / L_PP, s["tra"], s["tac"], s["dia"])
        for s in stats)
    ax.text(0.985, 0.030, txt, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.3, color=SUB, linespacing=1.55, zorder=10)

    ax.set_xlim(-250.0, 1060.0)
    ax.set_ylim(-330.0, 755.0)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("transfer to starboard (m)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("advance along the approach course (m)", fontsize=8.4)
    ax.set_title("Turning-circle trial, equal-aspect track", fontsize=9.2,
                 fontweight="bold", pad=5.0)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0,
              handlelength=1.7, labelspacing=0.32, borderaxespad=0.35)

    # ---- aux (b): the 10/10 zigzag -----------------------------------------
    ax_z.plot(t_z, d_z, color=GOLD, lw=1.5, zorder=5, label="rudder $\\delta$")
    ax_z.plot(t_z, psi_z, color=STEEL, lw=1.9, zorder=6,
              label="heading $\\psi$")
    ax_z.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    for lev in (10.0, -10.0):
        ax_z.axhline(lev, color=FAINT, lw=0.7, ls=(0, (3.0, 2.0)), zorder=1)

    i1 = int(np.argmax(psi_z >= 10.0))
    i2 = i1 + int(np.argmax(psi_z[i1:] <= -10.0))
    win = int(60.0 / DT)
    overs = []
    for i0, sgn in ((i1, 1.0), (i2, -1.0)):
        seg = psi_z[i0:i0 + win] * sgn
        j = int(np.argmax(seg))
        overs.append((t_z[i0 + j], float(psi_z[i0 + j]), float(seg[j]) - 10.0))
    for k, (tp, pp, ov) in enumerate(overs):
        ax_z.plot([tp], [pp], marker="o", ms=4.2, mfc="white", mec=RED,
                  mew=1.3, zorder=9)
        ax_z.annotate("%s overshoot %.1f deg"
                      % (("first", "second")[k], ov),
                      xy=(tp, pp), xytext=(tp + 18.0, pp + (11.0, -11.0)[k]),
                      fontsize=6.3, color=RED,
                      ha="left", va=("bottom", "top")[k], zorder=10,
                      arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                      shrinkA=1, shrinkB=3))
    ax_z.set_xlim(0.0, 420.0)
    ax_z.set_ylim(-32.0, 32.0)
    ax_z.tick_params(labelsize=6.8)
    ax_z.set_xlabel("time from rudder order (s)", fontsize=7.5, labelpad=1.5)
    ax_z.set_ylabel("angle (deg)", fontsize=7.5)
    ax_z.set_title("10/10 zigzag and its overshoots", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_z.legend(loc="upper right", frameon=False, fontsize=6.4,
                handlelength=1.5, labelspacing=0.26, borderaxespad=0.35,
                ncols=2, columnspacing=1.0)

    # ---- aux (c): speed loss and yaw rate ----------------------------------
    t, dl, psi, r, v, x, y = run0
    keep = t <= 420.0
    ax_s.plot(t[keep], v[keep] / KN, color=GREEN, lw=1.9, zorder=6)
    ax_s.set_xlim(0.0, 420.0)
    ax_s.set_ylim(0.0, V0_KN * 1.25)
    ax_s.tick_params(labelsize=6.8)
    ax_s.set_xlabel("time from rudder order (s)", fontsize=7.5, labelpad=1.5)
    ax_s.set_ylabel("speed (knots)", fontsize=7.5, color=GREEN)
    ax_s.tick_params(axis="y", labelcolor=GREEN)

    ax_s2 = ax_s.twinx()
    ax_s2.plot(t[keep], r[keep], color=RED, lw=1.9, ls=(0, (5.0, 2.2)),
               zorder=6)
    ax_s2.set_ylim(0.0, float(r.max()) * 1.30)
    ax_s2.spines["top"].set_visible(False)
    ax_s2.tick_params(labelsize=6.8, labelcolor=RED, length=3)
    ax_s2.set_ylabel("yaw rate r (deg/s)", fontsize=7.5, color=RED)
    ax_s2.grid(False)

    v_min = float(v[keep].min())
    ax_s.plot([t[keep][int(np.argmin(v[keep]))]], [v_min / KN], marker="o",
              ms=4.2, mfc="white", mec=GREEN, mew=1.3, zorder=9)
    ax_s.text(0.035, 0.075,
              "steady speed %.1f knots, %.0f %% of the approach\nsteady yaw "
              "rate %.3f deg/s at %.0f deg rudder"
              % (v_min / KN, v_min / V0 * 100.0, float(r.max()),
                 TURNS[0][0]),
              transform=ax_s.transAxes, ha="left", va="bottom", fontsize=6.3,
              color=SUB, linespacing=1.45, zorder=10)
    ax_s.set_title("A hard turn costs a third of the speed", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_z, ax_s], dx=-0.70, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 1001")
    check_layout(fig, foot, [ax, ax_z, ax_s])
    savefig(fig, "1001_shp_maneuvering_trials.png")


if __name__ == "__main__":
    main()
