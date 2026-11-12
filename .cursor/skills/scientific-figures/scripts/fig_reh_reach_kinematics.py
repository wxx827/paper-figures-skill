"""Upper-limb reaching kinematics: hand paths, speed profiles and smoothness.

Main panel: hand paths from a single start position to three targets in the
horizontal workspace, five trials per arm per target, drawn on an equal-aspect
axes so a curved path looks curved.  The straight line to each target is the
reference the path ratio is measured against, and the marker on every path is
the sample at which that trial reached its peak tangential speed.

Auxiliary (both at 58 % of the main panel width): tangential speed against
percentage of movement time, individual trials under the per-arm mean, where
the extra peaks of the affected arm are the submovements the main panel only
hints at; and each trial as one point in the smoothness plane of path ratio
against movement-unit count, with the two arm centroids marked.

Synthetic reaching data, illustrative values only; no real patient depicted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"
NORM = "#9a9a9a"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

S = np.linspace(0.0, 1.0, 401)            # normalised movement time
REACH = 250.0                             # mm from start to each target
TARGET_ANGLES = np.array([40.0, 90.0, 140.0])    # deg in the workspace plane
TARGET_R = 22.0                           # mm, target radius
N_TRIALS = 5
ARMS = {
    "less affected": dict(color=BLUE, mt=1.12, lateral=9.0,
                          onsets=[0.0], amps=[1.0], widths=[1.0]),
    "affected": dict(color=RUST, mt=1.96, lateral=34.0,
                     onsets=[0.0, 0.30, 0.62], amps=[0.54, 0.29, 0.24],
                     widths=[0.46, 0.42, 0.38]),
}


def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5


def progress(onsets, amps, widths):
    """Fraction of the reach completed, as a sum of overlapping submovements."""
    y = np.zeros_like(S)
    for t0, a, w in zip(onsets, amps, widths):
        y += a * min_jerk((S - t0) / w)
    return y / y[-1]


def trial(rng, arm, angle):
    """One hand path, its speed profile and the numbers read off them."""
    cfg = ARMS[arm]
    onsets = [max(0.0, t + rng.normal(0.0, 0.035)) for t in cfg["onsets"]]
    amps = [max(0.05, a * (1.0 + rng.normal(0.0, 0.12)))
            for a in cfg["amps"]]
    widths = [max(0.18, w * (1.0 + rng.normal(0.0, 0.08)))
              for w in cfg["widths"]]
    p = progress(onsets, amps, widths)

    a = np.deg2rad(angle)
    direction = np.array([np.cos(a), np.sin(a)])
    perp = np.array([-direction[1], direction[0]])
    # both harmonics are windowed by sin^2 so the path leaves and reaches the
    # target with zero lateral velocity, and the speed profile starts at zero
    win = np.sin(np.pi * S) ** 2
    bow = win * (rng.normal(0.0, cfg["lateral"])
                 + rng.normal(0.0, 0.45 * cfg["lateral"])
                 * np.sin(2.0 * np.pi * S))
    xy = (REACH * p)[:, None] * direction[None, :] + bow[:, None] * perp[None, :]

    mt = cfg["mt"] * (1.0 + rng.normal(0.0, 0.09))
    t = S * mt
    v = np.gradient(xy[:, 0], t) ** 2 + np.gradient(xy[:, 1], t) ** 2
    v = np.sqrt(v)
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    path_ratio = float(step.sum() / REACH)
    return dict(xy=xy, v=v, mt=mt, path_ratio=path_ratio,
                units=movement_units(v), peak=int(np.argmax(v)))


def movement_units(v):
    """Local maxima of the speed profile that clear 5 % of its peak."""
    thr = 0.05 * v.max()
    hits = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thr)
    return int(np.count_nonzero(hits))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(983)

    trials = {arm: [trial(rng, arm, ang) for ang in TARGET_ANGLES
                    for _ in range(N_TRIALS)] for arm in ARMS}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: hand paths in the workspace ---------------------------------
    ax.set_aspect("equal")
    for ang in TARGET_ANGLES:
        a = np.deg2rad(ang)
        tx, ty = REACH * np.cos(a), REACH * np.sin(a)
        ax.plot([0.0, tx], [0.0, ty], color=NORM, lw=0.9, ls=(0, (3.0, 2.4)),
                zorder=2)
        ax.add_patch(Circle((tx, ty), TARGET_R, facecolor="white",
                            edgecolor=SUB, lw=1.0, zorder=6))
        ax.text(tx, ty, f"{ang:.0f}$\\degree$", ha="center", va="center",
                fontsize=6.2, color=SUB, zorder=7)
    ax.plot([0.0], [0.0], marker="s", ms=6.5, mfc="white", mec=INK, mew=1.2,
            zorder=7)
    ax.text(0.0, -22.0, "start", ha="center", va="top", fontsize=6.6,
            color=INK)

    for arm, cfg in ARMS.items():
        for tr in trials[arm]:
            ax.plot(tr["xy"][:, 0], tr["xy"][:, 1], color=cfg["color"],
                    lw=0.9, alpha=0.62, zorder=4)
            i = tr["peak"]
            ax.plot([tr["xy"][i, 0]], [tr["xy"][i, 1]], marker="o", ms=2.6,
                    mfc="white", mec=cfg["color"], mew=0.7, zorder=5)

    summary = []
    for arm, cfg in ARMS.items():
        pr = np.mean([t["path_ratio"] for t in trials[arm]])
        mu = np.mean([t["units"] for t in trials[arm]])
        mt = np.mean([t["mt"] for t in trials[arm]])
        summary.append((arm, cfg["color"], pr, mu, mt))
    for k, (arm, col, pr, mu, mt) in enumerate(summary):
        ax.text(-238.0, 334.0 - 25.0 * k,
                f"{arm}: path ratio {pr:.2f}, {mu:.1f} movement units, "
                f"{mt:.2f} s per reach", ha="left", va="top", fontsize=6.6,
                color=col)
    ax.text(-238.0, 334.0 - 25.0 * len(summary),
            f"n={N_TRIALS} trials per target per arm", ha="left", va="top",
            fontsize=6.4, color=MUTED)
    ax.text(-238.0, 334.0 - 25.0 * (len(summary) + 1),
            "open markers mark peak speed", ha="left", va="top",
            fontsize=6.4, color=MUTED)

    handles = [Line2D([], [], color=c["color"], lw=1.4, label=a)
               for a, c in ARMS.items()]
    handles.append(Line2D([], [], color=NORM, lw=0.9, ls=(0, (3.0, 2.4)),
                          label="straight-line reference"))
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.6,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.8, labelspacing=0.32)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(-245.0, 245.0)
    ax.set_ylim(-60.0, 342.0)
    ax.set_xticks(np.arange(-200, 201, 100))
    ax.set_yticks(np.arange(0, 301, 100))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("workspace x (mm, lateral)", fontsize=8.2, labelpad=2.5)
    ax.set_ylabel("workspace y (mm, forward)", fontsize=8.2, labelpad=3.0)
    ax.set_title(f"Hand paths to three targets {REACH:.0f} mm from the start",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): speed profiles -------------------------------------------
    pct = 100.0 * S
    for arm, cfg in ARMS.items():
        stack = np.array([t["v"] for t in trials[arm]])
        for row in stack:
            ax_t.plot(pct, row, color=cfg["color"], lw=0.5, alpha=0.30,
                      zorder=3)
        m = stack.mean(axis=0)
        ax_t.plot(pct, m, color=cfg["color"], lw=1.7, zorder=5)
        i = int(np.argmax(m))
        ax_t.plot([pct[i]], [m[i]], marker="o", ms=3.4, mfc="white",
                  mec=cfg["color"], mew=1.0, zorder=6)
        ax_t.text(pct[i] + 2.0, m[i], f"{m[i]:.0f} mm/s", ha="left",
                  va="center", fontsize=6.2, color=cfg["color"])
    ax_t.set_xlim(0.0, 100.0)
    ax_t.set_ylim(0.0, None)
    ax_t.set_xticks(np.arange(0, 101, 20))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("% of movement time", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("tangential speed (mm/s)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("One smooth peak against several", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): the smoothness plane -------------------------------------
    for arm, cfg in ARMS.items():
        u = np.array([t["units"] for t in trials[arm]], dtype=float)
        r = np.array([t["path_ratio"] for t in trials[arm]])
        jit = rng.normal(0.0, 0.07, u.size)
        ax_b.plot(u + jit, r, ls="none", marker="o", ms=3.4,
                  mfc=lighten(cfg["color"], 0.32), mec="white", mew=0.5,
                  zorder=4)
        ax_b.plot([u.mean()], [r.mean()], marker="D", ms=6.0,
                  mfc=cfg["color"], mec="white", mew=0.9, zorder=6)
        ax_b.text(u.mean(), r.mean() + 0.024,
                  f"{arm}\n{u.mean():.1f} units, ratio {r.mean():.2f}",
                  ha="center", va="bottom", fontsize=6.1, color=cfg["color"],
                  linespacing=1.35)
    ax_b.axhline(1.0, color=NORM, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)
    all_u = np.concatenate([[t["units"] for t in trials[a]] for a in ARMS])
    all_r = np.concatenate([[t["path_ratio"] for t in trials[a]]
                            for a in ARMS])
    ax_b.text(all_u.max() + 0.8, 1.005, "straight path, ratio 1.00",
              ha="right", va="bottom", fontsize=6.1, color=SUB)
    ax_b.set_xlim(0.4, all_u.max() + 0.9)
    ax_b.set_ylim(0.985, all_r.max() * 1.12)
    ax_b.set_xticks(np.arange(1, all_u.max() + 1, 1))
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("movement units (speed peaks per reach, n)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("path ratio", fontsize=7.6, labelpad=2.0)
    ax_b.set_title(f"Every trial, n={all_u.size} reaches", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "983_reh_reach_kinematics.png")


if __name__ == "__main__":
    main()
