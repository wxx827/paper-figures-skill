"""One joint move planned twice, and what the third derivative costs.

The two profiles are generated the only way that keeps the four rows honest:
a rectangular velocity pulse is convolved with an acceleration window, and for
the jerk-limited plan once more with a jerk window, then the position is the
running integral of that velocity.  Everything drawn above the position row is
a numerical derivative of the position curve on the page, so velocity,
acceleration and jerk cannot disagree with it; the trapezoidal plan's jerk is
impulsive by construction and is clipped at the band edge rather than quietly
rescaled.  The side panels ask what the smoothing buys: a lightly damped tool
mode is driven by the same acceleration arrays, and its residual amplitude is
measured after the axis has stopped.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3d6b8e"
STEEL = "#6b7f95"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

SEED = 1062
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)

DT = 2.0e-4                 # integration step, s
TRAVEL = 90.0               # commanded joint travel, deg
V_MAX = 120.0               # velocity limit, deg/s
T_ACC = 0.25                # acceleration window, s  -> a_max = V_MAX/T_ACC
T_JRK = 0.080               # jerk window, s          -> j_max = a_max/T_JRK
A_MAX = V_MAX / T_ACC
J_MAX = A_MAX / T_JRK

F_MODE = 12.0               # tool structural mode, Hz
ZETA = 0.025                # modal damping ratio
ARM = 0.90                  # radius from the joint to the tool, m
T_END = 2.40                # simulated horizon, s
LEAD = 0.020                # quiet lead before the move is released, s

BOX = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none",
           alpha=0.86)


def profile(t_jerk):
    """Velocity by rectangular convolution, position by running integral."""
    n_flat = int(round((TRAVEL / V_MAX) / DT))
    v = np.full(n_flat, V_MAX)
    for width in (T_ACC, t_jerk):
        k = int(round(width / DT))
        if k > 1:
            v = np.convolve(v, np.ones(k) / k, mode="full")
    n = int(round(T_END / DT))
    lead = int(round(LEAD / DT))
    vel = np.zeros(n)
    take = min(v.size, n - lead)
    vel[lead:lead + take] = v[:take]
    t = np.arange(n) * DT - LEAD
    pos = np.concatenate(([0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1])) * DT))
    return t, pos


def derivatives(t, pos):
    """Velocity, acceleration and jerk, each differentiated off the position."""
    vel = np.gradient(pos, t)
    acc = np.gradient(vel, t)
    jrk = np.gradient(acc, t)
    return vel, acc, jrk


def move_time(t, pos, tol=1e-3):
    """First instant the position is within tol of the commanded travel."""
    hit = np.nonzero(np.abs(pos - TRAVEL) <= tol)[0]
    return float(t[hit[0]]) if hit.size else float(t[-1])


def tool_deflection(t, acc_deg):
    """Deflection of a lightly damped tool mode driven by the joint, in mm."""
    w = 2.0 * np.pi * F_MODE
    a_lin = np.deg2rad(acc_deg) * ARM
    sys = signal.lti([-1.0], [1.0, 2.0 * ZETA * w, w * w])
    _, y, _ = signal.lsim(sys, a_lin, t - t[0])
    return y * 1000.0


def residual_um(t, defl, t_move):
    """Peak deflection after the axis has stopped, in micrometres."""
    after = t > t_move + 0.02
    return float(np.max(np.abs(defl[after])) * 1000.0)


def band_map(v, lo, hi, centre, half=0.40):
    y = centre + half * (2.0 * (v - lo) / (hi - lo) - 1.0)
    return np.clip(y, centre - half, centre + half)


def main_panel(ax, t, curves, t_move):
    """Four quantities as stacked strips inside one axes."""
    bands = [
        ("Position\n(deg)", 0, "pos", -8.0, 100.0, None, "{:.1f} deg"),
        ("Velocity\n(deg/s)", 1, "vel", -10.0, 138.0, V_MAX, "{:.0f} deg/s"),
        ("Acceleration\n(deg/s$^2$)", 2, "acc", -640.0, 640.0, A_MAX,
         "{:.0f} deg/s$^2$"),
        ("Jerk\n(deg/s$^3$)", 3, "jrk", -9000.0, 9000.0, J_MAX,
         "{:.0f} deg/s$^3$"),
    ]
    styles = {"scurve": dict(color=ORANGE, lw=1.7, ls="-", zorder=8),
              "trap": dict(color=BLUE, lw=1.2, ls=(0, (4.2, 2.0)), zorder=7)}

    for name, centre, key, lo, hi, lim, fmt in bands:
        ax.axhspan(centre - 0.46, centre + 0.46, color="#f4f5f7", zorder=1,
                   lw=0)
        y0 = band_map(0.0, lo, hi, centre)
        ax.plot([-LEAD, 1.60], [y0, y0], color="#c3c8cf", lw=0.7, zorder=2)
        if lim is not None:
            for s in (+1.0, -1.0):
                yl = band_map(s * lim, lo, hi, centre)
                ax.plot([-LEAD, 1.60], [yl, yl], color=STEEL, lw=0.7,
                        ls=(0, (1.4, 1.8)), zorder=3)
            ax.text(1.585, band_map(lim, lo, hi, centre) + 0.035,
                    ("limit " + fmt).format(lim), ha="right", va="bottom",
                    fontsize=6.0, color=STEEL, zorder=9, bbox=BOX)
        for tag in ("scurve", "trap"):
            ax.plot(t, band_map(curves[tag][key], lo, hi, centre),
                    **styles[tag])

    for centre, key, fmt, dy in ((1, "vel", "peak {:.0f} deg/s", 0.055),
                                 (2, "acc", "peak {:.0f} deg/s$^2$", 0.055)):
        v = curves["scurve"][key]
        i = int(np.argmax(v))
        ax.text(t[i], band_map(v[i], *_range(key), centre) + dy,
                fmt.format(v[i]), ha="center", va="bottom", fontsize=6.1,
                color=ORANGE, zorder=10, bbox=BOX)

    ax.annotate("trapezoidal jerk is impulsive:\nclipped at the band edge",
                xy=(T_ACC, 3.40), xytext=(0.44, 3.06), fontsize=6.1,
                color=BLUE, ha="left", va="center", zorder=11,
                linespacing=1.32, bbox=BOX,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8,
                                shrinkA=3, shrinkB=3))

    for k, (tag, col, lab) in enumerate((("trap", BLUE, "trapezoidal"),
                                         ("scurve", ORANGE, "jerk-limited"))):
        tm = t_move[tag]
        ax.plot([tm, tm], [-0.46, 3.46], color=col, lw=0.8,
                ls=(0, (2.4, 2.0)), zorder=4)
        ax.text(1.16, 0.10 - 0.20 * k, "{} ends at {:.3f} s".format(lab, tm),
                ha="left", va="center", fontsize=6.1, color=col, zorder=10,
                bbox=BOX)

    ax.set_xlim(-LEAD, 1.60)
    ax.set_ylim(-0.62, 4.06)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([b[0] for b in bands], fontsize=6.9)
    ax.set_xlabel("time since the move was released (s)")
    ax.set_xticks(np.arange(0.0, 1.61, 0.2))
    ax.grid(axis="x", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=ORANGE, lw=1.7,
                      label="jerk-limited (S-curve)"),
               Line2D([], [], color=BLUE, lw=1.2, ls=(0, (4.2, 2.0)),
                      label="trapezoidal velocity"),
               Line2D([], [], color=STEEL, lw=0.7, ls=(0, (1.4, 1.8)),
                      label="planner constraint")]
    ax.legend(handles=handles, loc="upper center", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.95,
              handlelength=1.9, borderaxespad=0.15, labelspacing=0.26,
              ncol=3, columnspacing=1.4)
    ax.set_title("One 90 deg joint move: each row is the derivative of the "
                 "row below it", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


_RANGES = {"pos": (-8.0, 100.0), "vel": (-10.0, 138.0),
           "acc": (-640.0, 640.0), "jrk": (-9000.0, 9000.0)}


def _range(key):
    return _RANGES[key]


def sweep_panel(ax, sweep):
    j, tmove, resid = sweep
    ax.plot(j, tmove, color=BLUE, lw=1.6, marker="o", ms=2.6, zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("jerk limit (deg/s$^3$)", fontsize=6.8)
    ax.set_ylabel("move time (s)", fontsize=6.8, color=BLUE)
    ax.tick_params(axis="y", labelsize=6.6, colors=BLUE)
    ax.tick_params(axis="x", labelsize=6.6)

    axr = ax.twinx()
    axr.plot(j, resid, color=RED, lw=1.6, marker="s", ms=2.6, zorder=6)
    axr.set_ylabel("residual amplitude ($\\mu$m)", fontsize=6.8, color=RED)
    axr.tick_params(axis="y", labelsize=6.6, colors=RED)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color(RED)
    axr.spines["left"].set_visible(False)

    ax.axvline(J_MAX, color=MUTED, lw=0.8, ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(J_MAX * 0.86, ax.get_ylim()[1], "plan used here", ha="right",
            va="top", fontsize=6.0, color=SUB, rotation=90, zorder=8,
            bbox=BOX)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Softer jerk costs time, buys quiet", fontsize=8.0,
                 fontweight="bold", color=INK)


def ring_panel(ax, t, defl, t_move, resid):
    for tag, col, lw, ls, lab in (
            ("trap", BLUE, 1.1, (0, (4.2, 2.0)), "trapezoidal"),
            ("scurve", ORANGE, 1.5, "-", "jerk-limited")):
        ax.plot(t, defl[tag], color=col, lw=lw, ls=ls, zorder=6)
        ax.axvline(t_move[tag], color=col, lw=0.7, ls=(0, (2.0, 2.0)),
                   zorder=3)
    hi = max(np.max(np.abs(defl[k])) for k in defl) * 1.30
    ax.axvspan(max(t_move.values()) + 0.02, t[-1], color="#eef1f4", zorder=1,
               lw=0)
    ax.text(0.5 * (max(t_move.values()) + t[-1]), hi * 0.94,
            "after the axis has stopped", ha="center",
            va="top", fontsize=6.0, color=SUB, zorder=8, bbox=BOX)
    ax.text(t[-1] - 0.04, -hi * 0.92,
            "residual  {:.0f} $\\mu$m  vs  {:.0f} $\\mu$m"
            .format(resid["trap"], resid["scurve"]), ha="right", va="bottom",
            fontsize=6.2, color=INK, zorder=9, bbox=BOX)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-hi, hi)
    ax.set_xlabel("time (s)", fontsize=6.8)
    ax.set_ylabel("tool deflection (mm)", fontsize=6.8)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=1.5,
                              label="jerk-limited"),
                       Line2D([], [], color=BLUE, lw=1.1, ls=(0, (4.2, 2.0)),
                              label="trapezoidal")],
              loc="upper left", fontsize=6.1, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.95, handlelength=1.8,
              borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("A {:.0f} Hz tool mode rings on after the move"
                 .format(F_MODE), fontsize=8.0, fontweight="bold", color=INK)


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
    np.random.default_rng(SEED)

    curves, t_move, defl, resid = {}, {}, {}, {}
    for tag, tj in (("trap", 0.0), ("scurve", T_JRK)):
        t, pos = profile(tj)
        vel, acc, jrk = derivatives(t, pos)
        curves[tag] = dict(pos=pos, vel=vel, acc=acc, jrk=jrk)
        t_move[tag] = move_time(t, pos)
        defl[tag] = tool_deflection(t, acc)
        resid[tag] = residual_um(t, defl[tag], t_move[tag])

    assert abs(curves["scurve"]["pos"][-1] - TRAVEL) < 1e-6, "travel"
    assert np.max(curves["scurve"]["jrk"]) < 1.15 * J_MAX, "jerk limit"

    js, tms, res = [], [], []
    for jl in np.logspace(np.log10(900.0), np.log10(40000.0), 13):
        tj = A_MAX / jl
        t_s, pos_s = profile(tj)
        _v, acc_s, _j = derivatives(t_s, pos_s)
        tm = move_time(t_s, pos_s)
        js.append(jl)
        tms.append(tm)
        res.append(residual_um(t_s, tool_deflection(t_s, acc_s), tm))
    sweep = (np.array(js), np.array(tms), np.array(res))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sw = axes_inches(fig, AUX_T)
    ax_rg = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, curves, t_move)
    sweep_panel(ax_sw, sweep)
    ring_panel(ax_rg, t, defl, t_move, resid)



    panel_labels_inches(fig, [ax_main, ax_sw, ax_rg], dx=-0.66, dy=0.14)
    savefig(fig, "1062_ric_scurve_jerk_profile.png")


if __name__ == "__main__":
    main()
