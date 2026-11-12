"""Shot map on the attacking half, drawn over the expected-goals surface.

The xG surface is a closed-form function of shooting distance and of the angle
the 7.32 m goal mouth subtends from the shooting position, evaluated on a
0.35 m grid and multiplied by a technique factor.  Every marker is scored with
the *same* function, so the labelled contours behind the shots and the marker
areas can never disagree, and the cumulative curve is the running sum of those
same values ordered by minute.  Outcomes are Bernoulli draws from each shot's
own xG, which is what lets the actual-goals step wander away from the
expectation without either panel being retro-fitted to the other.

Half-pitch geometry is the FIFA standard surface seen end on: a 16.5 m penalty
area 40.32 m wide, a 5.5 m goal area 18.32 m wide, the penalty spot at 11 m, a
9.15 m penalty arc and a 7.32 m goal.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - right foot
GOLD = "#c9a227"          # house gold - left foot
GREEN = "#2e7d5b"         # house green - header
ACCENT = "#b23a48"        # house red - actual goals
GRASS = "#f4f4f2"
LINE = "#c2c2c2"

HALF_L, PITCH_W = 52.5, 68.0      # metres from halfway line, pitch width
GOAL_Y = HALF_L                   # goal line
POST = 3.66                       # half the 7.32 m goal
MATCH_MIN = 90.0
N_SHOT = 19
TEAM = "Team A"

TECH = [("right foot", HOME, 1.00),
        ("left foot", GOLD, 0.92),
        ("header", GREEN, 0.74)]
ZONES = ["six-yard box", "penalty area\ncentral", "penalty area\nwide",
         "outside the box"]


def draw_half_pitch(ax, color=LINE, lw=0.9, zorder=5):
    """Attacking half seen end on, goal at the top."""
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_W, HALF_L, facecolor=GRASS,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_W, HALF_L, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Arc((PITCH_W / 2.0, 0.0), 2 * 9.15, 2 * 9.15, theta1=0.0,
                     theta2=180.0, edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Circle((PITCH_W / 2.0, 0.0), 0.40, facecolor=color,
                        edgecolor="none", zorder=zorder))

    ax.add_patch(Rectangle((PITCH_W / 2.0 - 20.16, GOAL_Y - 16.5), 40.32, 16.5,
                           fill=False, edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Rectangle((PITCH_W / 2.0 - 9.16, GOAL_Y - 5.5), 18.32, 5.5,
                           fill=False, edgecolor=color, lw=lw, zorder=zorder))
    spot = (PITCH_W / 2.0, GOAL_Y - 11.0)
    ax.add_patch(Circle(spot, 0.35, facecolor=color, edgecolor="none",
                        zorder=zorder))
    arc_half = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    ax.add_patch(Arc(spot, 2 * 9.15, 2 * 9.15, theta1=270.0 - arc_half,
                     theta2=270.0 + arc_half, edgecolor=color, lw=lw,
                     zorder=zorder))
    ax.add_patch(Rectangle((PITCH_W / 2.0 - POST, GOAL_Y), 2 * POST, 1.9,
                           fill=False, edgecolor=color, lw=lw, zorder=zorder))
    for cx, t1, t2 in ((0.0, 270.0, 360.0), (PITCH_W, 180.0, 270.0)):
        ax.add_patch(Arc((cx, GOAL_Y), 2.0, 2.0, theta1=t1, theta2=t2,
                         edgecolor=color, lw=lw, zorder=zorder))


def goal_angle(x, y):
    """Angle in radians subtended by the goal mouth from ``(x, y)``."""
    v1x, v1y = PITCH_W / 2.0 - POST - x, GOAL_Y - y
    v2x, v2y = PITCH_W / 2.0 + POST - x, GOAL_Y - y
    n1 = np.hypot(v1x, v1y)
    n2 = np.hypot(v2x, v2y)
    cos = (v1x * v2x + v1y * v2y) / np.maximum(n1 * n2, 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def xg_surface(x, y, factor=1.0):
    """Closed-form xG from distance and goal-mouth angle; 0.01 .. 0.95."""
    d = np.hypot(x - PITCH_W / 2.0, GOAL_Y - y)
    a = goal_angle(x, y)
    raw = 1.15 * np.exp(-d / 12.0) * (a / np.pi) ** 0.42
    return np.clip(raw * factor, 0.01, 0.95)


def zone_of(x, y):
    """Index into ZONES for one shot location."""
    inside_box = (abs(x - PITCH_W / 2.0) <= 20.16) and (y >= GOAL_Y - 16.5)
    if (abs(x - PITCH_W / 2.0) <= 9.16) and (y >= GOAL_Y - 5.5):
        return 0
    if inside_box:
        return 1 if abs(x - PITCH_W / 2.0) <= 12.0 else 2
    return 3


def main():
    set_pub_style()
    rng = np.random.default_rng(895)

    # ---- shots --------------------------------------------------------------
    sx = np.clip(PITCH_W / 2.0 + rng.normal(0.0, 10.5, N_SHOT), 4.0,
                 PITCH_W - 4.0)
    sy = np.clip(GOAL_Y - np.abs(rng.gamma(3.4, 5.0, N_SHOT)), 6.0,
                 GOAL_Y - 4.5)
    tech = rng.choice(len(TECH), N_SHOT, p=[0.58, 0.24, 0.18])
    fac = np.array([TECH[t][2] for t in tech])
    xg = xg_surface(sx, sy, fac)
    minute = np.sort(rng.uniform(3.0, 89.0, N_SHOT))
    goal = rng.random(N_SHOT) < xg
    dist = np.hypot(sx - PITCH_W / 2.0, GOAL_Y - sy)
    zone = np.array([zone_of(a, b) for a, b in zip(sx, sy)])

    cum_xg = np.cumsum(xg)
    cum_goal = np.cumsum(goal.astype(float))
    delta = cum_goal[-1] - cum_xg[-1]

    # ---- xG surface ---------------------------------------------------------
    gx = np.arange(0.0, PITCH_W + 0.35, 0.35)
    gy = np.arange(0.0, HALF_L + 0.35, 0.35)
    GX, GY = np.meshgrid(gx, gy)
    Z = xg_surface(GX, GY)

    # ---- layout -------------------------------------------------------------
    FW, FH = 10.4, 6.3
    fig = plt.figure(figsize=(FW, FH))
    ax_p = axes_inches(fig, (0.42, 1.05, 5.33, 4.278))
    ax_c = axes_inches(fig, (6.62, 3.15, 3.42, 2.15))
    ax_z = axes_inches(fig, (6.62, 0.95, 3.42, 1.42))

    draw_half_pitch(ax_p, zorder=5)
    levels = [0.02, 0.04, 0.07, 0.12, 0.20, 0.32, 0.50]
    ax_p.contourf(GX, GY, Z, levels=[0.0] + levels + [1.0],
                  cmap=get_cmap("cb_sequential_blue"), alpha=0.30, zorder=1)
    cs = ax_p.contour(GX, GY, Z, levels=levels, colors="#8fa8c6",
                      linewidths=0.65, linestyles="-", zorder=3)
    ax_p.clabel(cs, levels, fmt=lambda v: f"{v:.2f}", fontsize=6.0,
                inline=True, inline_spacing=3)

    for t, (name, col, _f) in enumerate(TECH):
        for m, filled in ((~goal, False), (goal, True)):
            k = (tech == t) & m
            if not k.any():
                continue
            ax_p.scatter(sx[k], sy[k], s=14.0 + 420.0 * xg[k],
                         facecolor=col if filled else "white",
                         edgecolor=col, linewidth=1.0 if filled else 0.9,
                         alpha=0.95, zorder=8 if filled else 7)
    for k in np.flatnonzero(goal):
        ax_p.scatter([sx[k]], [sy[k]], s=(14.0 + 420.0 * xg[k]) * 2.6,
                     facecolor="none", edgecolor=ACCENT, linewidth=1.1,
                     zorder=9)

    ax_p.set_xlim(-4.0, PITCH_W + 4.0)
    ax_p.set_ylim(-2.0, HALF_L + 5.8)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax_p.set_title(f"{TEAM}: {N_SHOT} shots over the expected-goals surface",
                   fontsize=9.0, fontweight="bold", pad=5)

    annotate_box(
        ax_p,
        f"total xG  {xg.sum():.2f} from {N_SHOT} shots\n"
        f"goals  {int(goal.sum())}  ({delta:+.2f} against xG)\n"
        f"mean shot distance  {dist.mean():.1f} m | best chance "
        f"{xg.max():.2f} xG",
        loc="lower left", fontsize=6.4, margin=0.014)

    # ---- cumulative xG vs goals ---------------------------------------------
    mm = np.concatenate([[0.0], minute, [MATCH_MIN]])
    cx = np.concatenate([[0.0], cum_xg, [cum_xg[-1]]])
    cg = np.concatenate([[0.0], cum_goal, [cum_goal[-1]]])
    ax_c.fill_between(mm, cx, cg, step="post", color=ACCENT, alpha=0.12,
                      lw=0.0, zorder=2)
    ax_c.step(mm, cx, where="post", color=HOME, lw=1.5, zorder=4,
              label="cumulative xG")
    ax_c.step(mm, cg, where="post", color=ACCENT, lw=1.5, zorder=5,
              label="goals scored")
    ax_c.plot(minute[goal], cum_goal[goal], marker="o", ls="none", ms=4.4,
              mfc=ACCENT, mec="white", mew=0.8, zorder=6)
    for k in np.flatnonzero(goal):
        ax_c.annotate(f"{minute[k]:.0f}'", xy=(minute[k], cum_goal[k]),
                      xytext=(3, 3), textcoords="offset points", ha="left",
                      va="bottom", fontsize=6.2, color=ACCENT)
    ax_c.annotate("", xy=(MATCH_MIN - 3.0, cum_goal[-1]),
                  xytext=(MATCH_MIN - 3.0, cum_xg[-1]),
                  arrowprops=dict(arrowstyle="<->", color=RULE, lw=0.8))
    ax_c.text(MATCH_MIN - 5.0, 0.5 * (cum_goal[-1] + cum_xg[-1]),
              f"{delta:+.2f}", ha="right", va="center", fontsize=6.6,
              color=RULE)
    ax_c.set_xlim(0.0, MATCH_MIN)
    ax_c.set_ylim(0.0, max(cum_xg[-1], cum_goal[-1]) * 1.22)
    ax_c.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_c.set_xlabel("match time (min)", fontsize=8.0)
    ax_c.set_ylabel("cumulative xG / goals", fontsize=8.0)
    ax_c.tick_params(labelsize=7.0)
    ax_c.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.9,
                handlelength=1.9, borderaxespad=0.2)
    ax_c.set_title("Chance quality against the scoreline", fontsize=8.5,
                   fontweight="bold")

    # ---- conversion by zone --------------------------------------------------
    n_z = np.array([(zone == z).sum() for z in range(4)], dtype=float)
    xg_z = np.array([xg[zone == z].sum() for z in range(4)])
    g_z = np.array([goal[zone == z].sum() for z in range(4)], dtype=float)
    exp_pct = 100.0 * xg_z / np.maximum(n_z, 1.0)
    act_pct = 100.0 * g_z / np.maximum(n_z, 1.0)

    yy = np.arange(4)[::-1]
    ax_z.barh(yy + 0.19, exp_pct, height=0.34, color=HOME, edgecolor="white",
              lw=0.6, zorder=3, label="expected (xG per shot)")
    ax_z.barh(yy - 0.19, act_pct, height=0.34, color=ACCENT,
              edgecolor="white", lw=0.6, zorder=3, label="actual (goals)")
    hi = max(exp_pct.max(), act_pct.max())
    for y, e, a in zip(yy, exp_pct, act_pct):
        ax_z.text(e + hi * 0.02, y + 0.19, f"{e:.0f}%", ha="left",
                  va="center", fontsize=6.2, color=SUB)
        ax_z.text(a + hi * 0.02, y - 0.19, f"{a:.0f}%", ha="left",
                  va="center", fontsize=6.2, color=SUB)
    ax_z.set_yticks(yy)
    ax_z.set_yticklabels([f"{z}\n({k:.0f} shots)" for z, k in
                          zip(ZONES, n_z)], fontsize=6.3)
    ax_z.set_xlim(0.0, hi * 1.24)
    ax_z.set_ylim(-0.75, 3.75)
    ax_z.set_xlabel("conversion per shot (%)", fontsize=8.0)
    ax_z.tick_params(axis="x", labelsize=7.0)
    ax_z.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_z.spines["top"].set_visible(False)
    ax_z.spines["right"].set_visible(False)
    ax_z.legend(loc="lower right", frameon=False, fontsize=6.8,
                handlelength=1.5, borderaxespad=0.2)
    ax_z.set_title("Where the chances came from", fontsize=8.5,
                   fontweight="bold")

    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=c, markeredgecolor=c, markersize=6.0,
                      label=name) for name, c, _f in TECH]
    handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=RULE, markersize=6.0, label="no goal (hollow)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=ACCENT, markersize=8.5, markeredgewidth=1.1,
               label="goal (ringed)"),
        Line2D([0], [0], color="#8fa8c6", lw=0.65,
               label="xG contour (marker area = xG)")]
    fig.legend(handles=handles, loc="lower left",
               bbox_to_anchor=(0.030, 0.050), ncol=6, frameon=False,
               fontsize=7.0, handlelength=1.6, columnspacing=1.5)

    panel_labels_inches(fig, [ax_p, ax_c, ax_z], dx=-0.28, dy=0.12)

    verb = "over-performed" if delta > 0 else "under-performed"

    print(f"total xG {xg.sum():.2f}  goals {int(goal.sum())}  "
          f"delta {delta:+.2f}  mean dist {dist.mean():.1f} m")
    print("shots per zone:", n_z.tolist())
    print("expected %:", exp_pct.round(1).tolist())
    print("actual   %:", act_pct.round(1).tolist())
    savefig(fig, "866_tac_shot_map_xg.png")


if __name__ == "__main__":
    main()
