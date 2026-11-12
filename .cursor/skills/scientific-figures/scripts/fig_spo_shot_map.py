"""Shot map on the attacking half, over the expected-goals surface.

The xG surface is a closed-form model of shooting distance and of the angle
the goal mouth subtends from the shooting position, evaluated on a 0.5 m grid.
Every shot marker is scored with the *same* function, so the background and
the markers can never disagree.  Outcomes are then simulated from that score:
a distance-dependent block probability first, then goal / on target / off
target.  The cumulative panel, the zone conversion rates and the headline
"xG versus goals" line are all read back out of those events.

Half-pitch geometry is the FIFA standard surface seen end on: a 16.5 m
penalty area 40.32 m wide, a 5.5 m goal area 18.32 m wide, the penalty spot at
11 m and a 9.15 m penalty arc.
"""
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import panel_labels_inches, safe_colorbar

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - on target
GREEN = "#2e7d5b"         # house green - goals
GREY = "#8a8a8a"          # off target / blocked
LINE = "#b3b3b3"          # pitch markings on a white surface

PITCH_L, PITCH_W = 105.0, 68.0
Y_MIN = 52.5              # halfway line: the map shows the attacking half
GOAL_Y = PITCH_L
POST = 3.66               # half of the 7.32 m goal mouth
N_SHOTS = 19

ZONES = ["six-yard box", "penalty area", "outside the box"]


def rasterize(artist):
    """3.10 mis-reports contour rasterisation as ignored; it is not."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def xg_field(x, y):
    """Expected goals for a shot at ``(x, y)``, x across, y up the pitch."""
    dist = np.hypot(x - PITCH_W / 2.0, GOAL_Y - y)
    a = np.arctan2(PITCH_W / 2.0 + POST - x, np.maximum(GOAL_Y - y, 0.20))
    b = np.arctan2(PITCH_W / 2.0 - POST - x, np.maximum(GOAL_Y - y, 0.20))
    ang = np.abs(a - b)
    return np.clip(1.05 * np.exp(-0.11 * dist) * (ang / 0.90) ** 0.55,
                   0.005, 0.93)


def draw_half_pitch(ax, color=LINE, lw=1.0, zorder=4):
    """Attacking half seen end on, goal at the top."""
    ax.add_patch(Rectangle((0.0, Y_MIN), PITCH_W, PITCH_L - Y_MIN, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Arc((PITCH_W / 2.0, Y_MIN), 2 * 9.15, 2 * 9.15, theta1=0.0,
                     theta2=180.0, edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Circle((PITCH_W / 2.0, Y_MIN), 0.40, facecolor=color,
                        edgecolor="none", zorder=zorder))
    ax.add_patch(Rectangle((PITCH_W / 2.0 - 20.16, GOAL_Y - 16.5), 40.32,
                           16.5, fill=False, edgecolor=color, lw=lw,
                           zorder=zorder))
    ax.add_patch(Rectangle((PITCH_W / 2.0 - 9.16, GOAL_Y - 5.5), 18.32, 5.5,
                           fill=False, edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Circle((PITCH_W / 2.0, GOAL_Y - 11.0), 0.40,
                        facecolor=color, edgecolor="none", zorder=zorder))
    arc_half = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    ax.add_patch(Arc((PITCH_W / 2.0, GOAL_Y - 11.0), 2 * 9.15, 2 * 9.15,
                     theta1=180.0 + arc_half, theta2=360.0 - arc_half,
                     edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Rectangle((PITCH_W / 2.0 - POST, GOAL_Y), 2 * POST, 1.9,
                           fill=False, edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Arc((0.0, GOAL_Y), 2.0, 2.0, theta1=270.0, theta2=360.0,
                     edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Arc((PITCH_W, GOAL_Y), 2.0, 2.0, theta1=180.0, theta2=270.0,
                     edgecolor=color, lw=lw, zorder=zorder))


def main():
    set_pub_style()
    rng = np.random.default_rng(6081)

    # ---- xG surface --------------------------------------------------------
    gx = np.arange(0.0, PITCH_W + 0.5, 0.5)
    gy = np.arange(Y_MIN, GOAL_Y + 0.5, 0.5)
    GX, GY = np.meshgrid(gx, gy)
    Z = xg_field(GX, GY)

    # ---- shots -------------------------------------------------------------
    sx = np.clip(PITCH_W / 2.0 + rng.normal(0.0, 9.5, N_SHOTS), 3.0,
                 PITCH_W - 3.0)
    sy = np.clip(GOAL_Y - rng.gamma(3.6, 4.6, N_SHOTS), Y_MIN + 2.0,
                 GOAL_Y - 3.0)
    minute = np.sort(rng.uniform(2.0, 92.0, N_SHOTS))
    xg = xg_field(sx, sy)
    dist = np.hypot(sx - PITCH_W / 2.0, GOAL_Y - sy)

    p_block = np.clip(0.38 - 0.010 * dist, 0.07, 0.38)
    blocked = rng.random(N_SHOTS) < p_block
    p_goal = np.clip(xg / (1.0 - p_block), 0.0, 0.95)
    goal = (~blocked) & (rng.random(N_SHOTS) < p_goal)
    on_t = (~blocked) & (~goal) & (rng.random(N_SHOTS) < 0.46)
    off_t = ~(blocked | goal | on_t)

    tot_xg = float(xg.sum())
    n_goal = int(goal.sum())

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.72],
                          left=0.030, right=0.960, top=0.850, bottom=0.185,
                          wspace=0.30, hspace=0.62)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 1])

    levels = np.array([0.01, 0.02, 0.04, 0.07, 0.12, 0.20, 0.32, 0.50, 0.75,
                       0.93])
    cf = ax_s.contourf(GX, GY, Z, levels=levels,
                       cmap=get_cmap("cb_sequential_warm"), alpha=0.62,
                       zorder=2, extend="min")
    rasterize(cf)
    cf.set_edgecolor("face")
    cl = ax_s.contour(GX, GY, Z, levels=[0.04, 0.12, 0.32], colors="#a04c18",
                      linewidths=0.6, alpha=0.75, zorder=3)
    ax_s.clabel(cl, fmt="%.2f", fontsize=6.0, inline=True, inline_spacing=6)
    draw_half_pitch(ax_s, zorder=4)

    size = 14.0 + 300.0 * xg
    ax_s.scatter(sx[off_t], sy[off_t], s=size[off_t], marker="X",
                 c=GREY, linewidths=0.0, alpha=0.85, zorder=6,
                 label="off target")
    ax_s.scatter(sx[blocked], sy[blocked], s=size[blocked], marker="s",
                 facecolors="none", edgecolors=GREY, linewidths=1.0,
                 zorder=6, label="blocked")
    ax_s.scatter(sx[on_t], sy[on_t], s=size[on_t], marker="o",
                 facecolors="none", edgecolors=HOME, linewidths=1.3,
                 zorder=7, label="on target, saved")
    ax_s.scatter(sx[goal], sy[goal], s=size[goal], marker="o",
                 facecolors=GREEN, edgecolors="white", linewidths=0.9,
                 zorder=8, label="goal")

    k = int(np.argmax(xg))
    ax_s.annotate(f"best chance  xG {xg[k]:.2f}", xy=(sx[k], sy[k]),
                  xytext=(sx[k] + 9.0, sy[k] - 7.0), fontsize=6.4, color=INK,
                  ha="left", va="center", zorder=9,
                  arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7,
                                  shrinkA=0, shrinkB=4))

    ax_s.set_xlim(-3.0, PITCH_W + 3.0)
    ax_s.set_ylim(Y_MIN - 3.0, GOAL_Y + 8.5)
    ax_s.set_aspect("equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    for s in ax_s.spines.values():
        s.set_visible(False)
    ax_s.set_title("Shot locations on the expected-goals surface",
                   fontsize=9.0, fontweight="bold", pad=6)
    annotate_box(ax_s,
                 f"shots  {N_SHOTS} | on target  {int(goal.sum() + on_t.sum())}"
                 f"\ntotal xG  {tot_xg:.2f} | goals  {n_goal}\n"
                 f"xG per shot  {tot_xg / N_SHOTS:.3f}",
                 loc="upper left", fontsize=6.6, margin=0.014)

    safe_colorbar(fig, cf, ax_s, label="expected goals per shot (xG)",
                  side="bottom", width=0.10, pad=0.10, shrink=0.60,
                  labelsize=6.5, ticks=[0.02, 0.07, 0.20, 0.50, 0.93])

    # ---- cumulative xG versus goals ---------------------------------------
    mm = np.concatenate([[0.0], minute, [95.0]])
    cxg = np.concatenate([[0.0], np.cumsum(xg), [tot_xg]])
    cgl = np.concatenate([[0.0], np.cumsum(goal.astype(float)),
                          [float(n_goal)]])
    ax_c.step(mm, cxg, where="post", color="#cc722f", lw=1.6,
              label="cumulative xG")
    ax_c.step(mm, cgl, where="post", color=GREEN, lw=1.6,
              label="cumulative goals")
    ax_c.fill_between(mm, cgl, cxg, step="post", color="#cc722f", alpha=0.12,
                      lw=0.0)
    for t in minute[goal]:
        ax_c.axvline(t, color=GREEN, lw=0.7, ls=":", alpha=0.8, zorder=1)
    ax_c.set_xlim(0.0, 95.0)
    ax_c.set_ylim(0.0, max(tot_xg, n_goal) * 1.22)
    ax_c.set_xlabel("match time (min)", fontsize=7.4)
    ax_c.set_ylabel("cumulative total (goals, xG)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.5)
    ax_c.grid(lw=0.5, color="#ececec", zorder=0)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.6,
                handlelength=1.6, borderaxespad=0.2)
    ax_c.set_title(f"Finishing ran {n_goal - tot_xg:+.2f} goals against xG",
                   fontsize=8.5, fontweight="bold")

    # ---- conversion by zone ------------------------------------------------
    six = (sy >= GOAL_Y - 5.5) & (np.abs(sx - PITCH_W / 2.0) <= 9.16)
    box = ((sy >= GOAL_Y - 16.5) & (np.abs(sx - PITCH_W / 2.0) <= 20.16)
           & ~six)
    out = ~(six | box)
    masks = [six, box, out]
    conv = np.array([100.0 * goal[m].sum() / max(m.sum(), 1) for m in masks])
    xg_pc = np.array([100.0 * xg[m].mean() if m.sum() else 0.0
                      for m in masks])

    yy = np.arange(len(ZONES))[::-1]
    ax_z.barh(yy + 0.17, conv, height=0.32, color=GREEN, edgecolor="white",
              lw=0.6, zorder=3, label="actual conversion")
    ax_z.barh(yy - 0.17, xg_pc, height=0.32, color="#e69a54",
              edgecolor="white", lw=0.6, zorder=3, label="mean xG per shot")
    for y, c, x_, m in zip(yy, conv, xg_pc, masks):
        ax_z.text(max(c, x_) + 1.6, y, f"n = {int(m.sum())}", ha="left",
                  va="center", fontsize=6.2, color=SUB)
    ax_z.set_yticks(yy)
    ax_z.set_yticklabels(ZONES, fontsize=6.6)
    ax_z.set_xlim(0.0, max(conv.max(), xg_pc.max()) * 1.34)
    ax_z.set_xlabel("share of shots converted (%)", fontsize=7.4)
    ax_z.tick_params(axis="x", labelsize=6.5)
    ax_z.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_z.legend(loc="lower right", frameon=False, fontsize=6.6,
                handlelength=1.4, borderaxespad=0.2)
    ax_z.set_title("Conversion against expectation by zone", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
               markeredgecolor="white", markersize=6.5, label="goal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=HOME, markeredgewidth=1.2, markersize=6.5,
               label="on target, saved"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=GREY,
               markeredgecolor="none", markersize=6.5, label="off target"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor=GREY, markeredgewidth=1.0, markersize=6.0,
               label="blocked"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=RULE, markeredgewidth=0.8, markersize=8.0,
               label="marker area = xG of the shot")],
        loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=5,
        frameon=False, fontsize=7.0, handlelength=1.4, columnspacing=1.8)

    panel_labels_inches(fig, [ax_s, ax_c, ax_z], dx=-0.24, dy=0.14)

    verdict = "over" if n_goal > tot_xg else "under"

    print(f"shots {N_SHOTS}  xG {tot_xg:.2f}  goals {n_goal}  "
          f"on target {int((goal | on_t).sum())}  blocked {int(blocked.sum())}")
    print("zone conversion %:", dict(zip(ZONES, conv.round(1).tolist())))
    print("mean xG per shot by zone %:", xg_pc.round(1).tolist())
    savefig(fig, "608_spo_shot_map.png")


if __name__ == "__main__":
    main()
