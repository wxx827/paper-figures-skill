"""Marker-based motion capture: five stick-figure poses through one stride,
with the joint angles that generated them.

The skeleton is posed by forward kinematics from the angle curves on the
right, so the two halves of the figure cannot disagree: the knee that looks
flexed at t3 is flexed in panel (b) as well.  Each pose is dropped onto the
floor by shifting it until the lowest foot point touches z = 0, which is what
keeps the walk from floating.  Poses and the vertical rules that mark them
share one time ramp.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (projection)
from pubstyle import (set_pub_style, savefig, style_axes, get_palette,
                      COLORS)
from figkit import shared_limits, panel_labels_inches

STRIDE_S = 1.10
SPEED = 1.35                    # progression speed (m/s)
SNAPS = np.linspace(0.0, 0.88, 5)
ELEV, AZIM = 15.0, -70.0

L = {"pelvis_spine": 0.30, "spine_neck": 0.17, "neck_head": 0.14,
     "shoulder": 0.185, "upper_arm": 0.29, "forearm": 0.26,
     "hip": 0.095, "thigh": 0.42, "shank": 0.41, "foot": 0.17}

SEGMENTS = [("pelvis", "spine"), ("spine", "neck"), ("neck", "head"),
            ("neck", "LS"), ("neck", "RS"), ("LS", "LE"), ("LE", "LW"),
            ("RS", "RE"), ("RE", "RW"), ("pelvis", "LH"), ("pelvis", "RH"),
            ("LH", "LK"), ("LK", "LA"), ("LA", "LT"), ("RH", "RK"),
            ("RK", "RA"), ("RA", "RT")]
LEFT = {"LS", "LE", "LW", "LH", "LK", "LA", "LT"}

LOWER = [("Hip flexion", "#b23a48"), ("Knee flexion", "#3b5b92"),
         ("Ankle dorsiflexion", "#2e7d5b")]
UPPER = [("Shoulder flexion", "#c9a227"), ("Elbow flexion", "#8e6bbf"),
         ("Trunk lean", "#555555")]


def _bump(p, centre, width, amp):
    d = (p - centre + 50.0) % 100.0 - 50.0
    return amp * np.exp(-0.5 * (d / width) ** 2)


def angles(t, side="R"):
    """Sagittal joint angles (deg) for one limb at time ``t`` (s)."""
    p = ((t / STRIDE_S) * 100.0 + (50.0 if side == "L" else 0.0)) % 100.0
    hip = 10.0 + 20.0 * np.cos(2 * np.pi * p / 100.0)
    knee = 2.0 + _bump(p, 16.0, 8.0, 16.0) + _bump(p, 72.0, 12.0, 58.0)
    ankle = (2.0 + _bump(p, 45.0, 14.0, 12.0) - _bump(p, 8.0, 7.0, 7.0)
             - _bump(p, 64.0, 9.0, 22.0))
    shoulder = -0.75 * (hip - 10.0)
    elbow = 38.0 + 16.0 * np.cos(2 * np.pi * p / 100.0 + 2.2)
    return hip, knee, ankle, shoulder, elbow


def trunk_lean(t):
    p = (t / STRIDE_S) * 100.0 % 100.0
    return 6.0 + 1.6 * np.cos(4 * np.pi * p / 100.0)


def _step(origin, length, deg, lateral=0.0):
    """Advance a segment by ``length`` at ``deg`` from vertical-down."""
    a = np.deg2rad(deg)
    return np.array([origin[0] + length * np.sin(a), origin[1] + lateral,
                     origin[2] - length * np.cos(a)])


def skeleton(t):
    """Forward kinematics for one instant; returns a joint dictionary (m)."""
    j = {}
    lean = trunk_lean(t)
    j["pelvis"] = np.array([SPEED * t, 0.0, 0.95])
    j["spine"] = _step(j["pelvis"], -L["pelvis_spine"], lean)
    j["neck"] = _step(j["spine"], -L["spine_neck"], lean)
    j["head"] = _step(j["neck"], -L["neck_head"], lean * 0.4)

    for side, sgn in (("L", 1.0), ("R", -1.0)):
        hip_a, knee_a, ank_a, sh_a, el_a = angles(t, side)
        j[side + "S"] = j["neck"] + np.array([0.0, sgn * L["shoulder"],
                                              -0.03])
        j[side + "E"] = _step(j[side + "S"], L["upper_arm"], sh_a)
        j[side + "W"] = _step(j[side + "E"], L["forearm"], sh_a - el_a)
        j[side + "H"] = j["pelvis"] + np.array([0.0, sgn * L["hip"], -0.03])
        j[side + "K"] = _step(j[side + "H"], L["thigh"], hip_a)
        j[side + "A"] = _step(j[side + "K"], L["shank"], hip_a - knee_a)
        toe_pitch = np.deg2rad(18.0 - ank_a)
        j[side + "T"] = j[side + "A"] + np.array(
            [L["foot"] * np.cos(toe_pitch), 0.0,
             -L["foot"] * np.sin(toe_pitch) * 0.55])

    drop = min(j[s + "A"][2] - 0.055 for s in "LR")
    drop = min(drop, min(j[s + "T"][2] for s in "LR"))
    for k in j:
        j[k] = j[k] - np.array([0.0, 0.0, drop])
    return j


def ground(ax, xlim, ylim):
    for x in np.arange(xlim[0], xlim[1] + 1e-6, 0.20):
        ax.plot([x, x], ylim, [0.0, 0.0], color="#e2e2e2", lw=0.5, zorder=0)
    for y in np.arange(ylim[0], ylim[1] + 1e-6, 0.20):
        ax.plot(xlim, [y, y], [0.0, 0.0], color="#e2e2e2", lw=0.5, zorder=0)


def main():
    set_pub_style()
    ramp = get_palette("cb_sequential_blue", 7)[2:]      # 5 legible steps
    poses = [skeleton(t) for t in SNAPS]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.10, hspace=0.40,
                          left=0.010, right=0.960, top=0.885, bottom=0.100)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_l = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])

    # ---- (a) stick figures on the floor ------------------------------------
    xlim = (-0.35, SPEED * SNAPS[-1] + 0.55)
    ylim = (-0.45, 0.45)
    ground(ax3, xlim, ylim)
    for j, colour, t in zip(poses, ramp, SNAPS):
        for a, b in SEGMENTS:                              # cast shadow first
            ax3.plot([j[a][0], j[b][0]], [j[a][1], j[b][1]], [0.004, 0.004],
                     color="#c2c2c2", lw=0.8, alpha=0.55, zorder=1)
        for a, b in SEGMENTS:
            far = a in LEFT or b in LEFT
            ax3.plot([j[a][0], j[b][0]], [j[a][1], j[b][1]],
                     [j[a][2], j[b][2]], color=colour,
                     lw=1.0 if far else 1.5, alpha=0.55 if far else 1.0,
                     solid_capstyle="round", zorder=3)
        pts = np.array([j[k] for k in j])
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=11, c=colour,
                    edgecolors="white", linewidths=0.35, depthshade=False,
                    zorder=4)
        ax3.text(j["head"][0], j["head"][1], j["head"][2] + 0.13,
                 f"{t:.2f} s", ha="center", va="bottom", fontsize=6.2,
                 color=colour)

    ax3.set_xlim(*xlim)
    ax3.set_ylim(*ylim)
    ax3.set_zlim(0.0, 1.80)
    ax3.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], 1.80),
                       zoom=1.22)
    ax3.view_init(elev=ELEV, azim=AZIM)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_visible(False)
    ax3.grid(False)
    ax3.set_yticks([])
    ax3.set_xticks(np.arange(0.0, xlim[1], 0.4))
    ax3.set_zticks([0.0, 0.5, 1.0, 1.5])
    ax3.tick_params(labelsize=6.4, pad=0.5)
    ax3.set_xlabel("Progression (m)", fontsize=7.5, labelpad=-4.0)
    ax3.set_zlabel("Height (m)", fontsize=7.5, labelpad=-6.0)
    ax3.set_title("Stride sampled at five instants", fontsize=8.5,
                  fontweight="bold", y=0.93)

    # ---- (b), (c) the angles that posed the skeleton -----------------------
    t = np.linspace(0.0, STRIDE_S, 300)
    hip, knee, ankle, sh, el = angles(t, "R")
    series = {"Hip flexion": hip, "Knee flexion": knee,
              "Ankle dorsiflexion": ankle, "Shoulder flexion": sh,
              "Elbow flexion": el, "Trunk lean": trunk_lean(t)}
    lo, hi = shared_limits(list(series.values()), pad=0.10)

    for ax, group, title in ((ax_l, LOWER, "Right lower limb"),
                             (ax_u, UPPER, "Right arm and trunk")):
        ax.axhline(0.0, color=COLORS["baseline"], lw=0.7, ls=(0, (3.0, 2.4)))
        for name, colour in group:
            ax.plot(t, series[name], color=colour, lw=1.4, label=name)
        for ts, colour in zip(SNAPS, ramp):
            ax.axvline(ts, color=COLORS["baseline"], lw=0.7,
                       ls=(0, (2.6, 2.2)), zorder=1)
            ax.plot([ts], [hi - 0.035 * (hi - lo)], marker="s", ms=3.4,
                    color=colour, zorder=5)
        ax.set_xlim(0.0, STRIDE_S)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Joint angle (\u00b0)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.legend(loc="upper left", frameon=False, fontsize=6.6, ncol=1,
                  handlelength=1.3, bbox_to_anchor=(0.0, 0.97))
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        style_axes(ax)
    ax_l.tick_params(labelbottom=False)
    ax_u.set_xlabel("Time (s)", fontsize=7.5)

    panel_labels_inches(fig, [ax3, ax_l, ax_u], dx=-0.06, dy=0.06)
    savefig(fig, "515_bio_motion_capture.png")


if __name__ == "__main__":
    main()
