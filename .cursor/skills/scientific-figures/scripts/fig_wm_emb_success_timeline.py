"""Execution timeline of one episode, lane by lane.

Four swim lanes share a single time axis: the sub-goal schedule, the gripper
state, the contact events picked off the force channel, and the recovery
window the supervisor opened after a slip.  Dashed rules mark the four
moments a reader needs to find, and the strip underneath integrates the
sub-goal schedule into a cumulative progress trace - including the setback,
because the slip invalidated a sub-goal that had already been credited.

Everything printed as text is derived from the arrays that are drawn: the
episode length from the schedule, the peak contact force from the event
list, the recovery cost from the two timestamps, and the final progress from
the last sample of the progress trace.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, get_palette, style_axes

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
BLUE = "#3b5b92"
ACCENT = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

SEED = 1533  # original id; figure number is 1128

# (name, start, end, credited)  - "credited" sub-goals advance the progress
SUBGOALS = [
    ("Approach", 0.00, 2.60, True),
    ("Align", 2.60, 4.40, True),
    ("Close gripper", 4.40, 5.60, True),
    ("Lift", 5.60, 7.40, True),
    ("Transport", 7.40, 12.80, True),
    ("Re-grasp", 12.80, 15.40, False),      # recovery, not a new sub-goal
    ("Place", 15.40, 18.60, True),
    ("Retract", 18.60, 21.20, True),
]

# (state, start, end) with state in {"open", "closing", "closed"}
GRIPPER = [
    ("open", 0.00, 4.90), ("closing", 4.90, 5.60), ("closed", 5.60, 13.05),
    ("open", 13.05, 13.90), ("closing", 13.90, 14.40),
    ("closed", 14.40, 17.90), ("open", 17.90, 21.20),
]
GRIP_FILL = {"open": "#e4e8ec", "closing": "#9fb2c6", "closed": BLUE}

# (time, force in N, label or None)
CONTACTS = [(4.92, 0.9, None), (5.58, 6.4, "grasp closure"),
            (9.40, 1.2, None), (13.05, 9.7, "slip"), (17.62, 5.1, "placement"),
            (18.40, 2.3, None)]

RECOVERY = (13.05, 15.40)

LANES = ["Sub-goals", "Gripper", "Contact events", "Failure / retry"]
LANE_H = 0.56


def rounded_bar(ax, x0, x1, y, h, fc, ec="white", lw=0.7, z=3, r=0.05):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                mutation_aspect=6.0, zorder=z))


def tick_mark(ax, x, y, s, color=GREEN, lw=1.3):
    ax.plot([x - 0.42 * s, x - 0.10 * s, x + 0.46 * s],
            [y + 0.02 * s, y - 0.30 * s, y + 0.38 * s], color=color, lw=lw,
            solid_capstyle="round", solid_joinstyle="miter", zorder=7)


def main():
    set_pub_style(base_font=8)

    t_end = max(e for _, _, e, _ in SUBGOALS)
    n_goals = sum(1 for _, _, _, c in SUBGOALS if c)
    f_peak = max(f for _, f, _ in CONTACTS)
    t_slip = next(t for t, f, _ in CONTACTS if f == f_peak)
    recovery_cost = RECOVERY[1] - RECOVERY[0]

    # cumulative progress: one credit per completed sub-goal, minus the one
    # the slip invalidated
    grid = np.linspace(0.0, t_end + 0.35, 900)   # past the last step so it shows
    credit = np.zeros_like(grid)
    for _, _, end, ok in SUBGOALS:
        if ok:
            credit += (grid >= end).astype(float)
    credit -= (grid >= t_slip).astype(float) * (grid < RECOVERY[1])
    progress = 100.0 * credit / n_goals

    fig = plt.figure(figsize=(10.4, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.22], left=0.115,
                          right=0.975, top=0.800, bottom=0.115, hspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[1, 0], sharex=ax)

    ypos = {name: len(LANES) - 1 - i for i, name in enumerate(LANES)}
    for i, name in enumerate(LANES):                     # lane backgrounds
        y = ypos[name]
        ax.axhspan(y - 0.46, y + 0.46, color="#f5f5f5" if i % 2 == 0
                   else "#fafafa", zorder=0, lw=0)

    # ---- lane 1: sub-goal schedule ----------------------------------------
    y = ypos["Sub-goals"]
    cols = get_palette("cb_sequential_blue", len(SUBGOALS) + 2)[2:]
    # 6.8 pt bold on this axis fits roughly this many characters per second of
    # bar; a label wider than its own bar would run over the neighbouring ones
    CHARS_PER_SEC = 7.5
    for k, (name, s, e, ok) in enumerate(SUBGOALS):
        fc = ACCENT if not ok else cols[k]
        rounded_bar(ax, s, e, y, LANE_H, fc, z=3)
        label = (name if len(name) <= CHARS_PER_SEC * (e - s)
                 else name.replace(" ", "\n"))
        ax.text(0.5 * (s + e), y, label, ha="center", va="center",
                fontsize=6.8, color="white" if k > 2 or not ok else "#1a1a1a",
                fontweight="bold", zorder=5, linespacing=1.15)
        if ok:
            tick_mark(ax, e - 0.16, y + 0.40, 0.34)

    # ---- lane 2: gripper state --------------------------------------------
    y = ypos["Gripper"]
    for state, s, e in GRIPPER:
        rounded_bar(ax, s, e, y, LANE_H * 0.72, GRIP_FILL[state], z=3)
    widest = {}
    for state, s, e in GRIPPER:
        if e - s > widest.get(state, (0.0,))[0]:
            widest[state] = (e - s, 0.5 * (s + e))
    for state, (_, xc) in widest.items():
        above = state == "closing"
        ax.text(xc, y + (0.40 if above else -0.34), state, ha="center",
                va="bottom" if above else "top", fontsize=6.4, color=MUTED,
                zorder=5)

    # ---- lane 3: contact events -------------------------------------------
    y = ypos["Contact events"]
    ax.plot([0.0, t_end], [y - 0.26, y - 0.26], color="#b3b3b3", lw=0.8,
            zorder=2)
    for t, f, lab in CONTACTS:
        h = 0.70 * f / f_peak
        col = ACCENT if f == f_peak else BLUE
        ax.plot([t, t], [y - 0.26, y - 0.26 + h], color=col, lw=1.4,
                solid_capstyle="round", zorder=4)
        ax.plot([t], [y - 0.26 + h], marker="o", ms=3.4, mfc="white",
                mec=col, mew=1.0, zorder=5)
        if lab:
            side = "right" if f == f_peak else "center"
            ax.text(t - (0.20 if side == "right" else 0.0),
                    y - 0.26 + h + 0.055, f"{lab}  {f:.1f} N", ha=side,
                    va="bottom", fontsize=6.2, color=col, zorder=6)

    # ---- lane 4: failure and recovery -------------------------------------
    y = ypos["Failure / retry"]
    rounded_bar(ax, *RECOVERY, y, LANE_H, "#f2dcdf", ec=ACCENT, lw=0.9, z=3)
    ax.text(0.5 * sum(RECOVERY), y, "retry", ha="center", va="center",
            fontsize=6.8, color=ACCENT, fontweight="bold", zorder=5)
    ax.text(0.5 * sum(RECOVERY), y - 0.34, f"slip detected  ·  re-grasp  ·  "
            f"+{recovery_cost:.2f} s", ha="center", va="top", fontsize=6.6,
            color=ACCENT, zorder=5)

    # ---- key moments -------------------------------------------------------
    marks = [(5.60, "first stable grasp"), (t_slip, "slip"),
             (RECOVERY[1], "recovery complete"), (t_end, "task success")]
    for k, (t, lab) in enumerate(marks):
        col = ACCENT if lab == "slip" else "#8a8a8a"
        ax.axvline(t, color=col, lw=0.9, ls=(0, (3.2, 2.2)), zorder=6)
        axp.axvline(t, color=col, lw=0.9, ls=(0, (3.2, 2.2)), zorder=6)
        ax.text(t, len(LANES) - 0.42 + 0.20 * (k % 2), f" {lab}",
                ha="left", va="bottom", fontsize=6.6, color=col,
                fontweight="bold" if col == ACCENT else "normal", zorder=7)

    ax.set_yticks([ypos[n] for n in LANES])
    ax.set_yticklabels(LANES, fontsize=8.0, fontweight="bold")
    ax.set_ylim(-0.62, len(LANES) - 0.10)
    ax.set_xlim(-0.35, t_end + 0.35)
    style_axes(ax)                       # sets a tick length, so trim after
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # ---- cumulative progress ----------------------------------------------
    axp.fill_between(grid, 0, progress, color=BLUE, alpha=0.22, lw=0,
                     zorder=2)
    axp.plot(grid, progress, color=BLUE, lw=1.5, zorder=3,
             solid_joinstyle="miter")
    axp.axhline(100.0, color=GREEN, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    dip = progress[(grid > t_slip) & (grid < RECOVERY[1])]
    axp.annotate(f"-{100.0 / n_goals:.0f} pp", xy=(t_slip + 0.20, dip.max()),
                 xytext=(t_slip + 1.05, dip.max() - 26), fontsize=6.4,
                 color=ACCENT, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8,
                                 shrinkA=0, shrinkB=2))
    axp.text(t_end + 0.20, progress[-1] + 4, f"{progress[-1]:.0f} %",
             ha="right", va="bottom", fontsize=7.4, fontweight="bold",
             color=GREEN, zorder=5)
    axp.set_ylim(0, 118)
    axp.set_yticks([0, 50, 100])
    axp.set_ylabel("progress\n(%)", fontsize=7.4, labelpad=2.0,
                   linespacing=1.25)
    axp.set_xlabel("time since episode start (s)", fontsize=8.0, labelpad=2.0)
    style_axes(axp)
    axp.tick_params(labelsize=6.8, length=2.4)

    savefig(fig, "1128_wm_emb_success_timeline.png")


if __name__ == "__main__":
    main()
