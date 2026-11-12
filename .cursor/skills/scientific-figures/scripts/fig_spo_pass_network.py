"""Pass network on a full-size pitch, with the matrix and centrality behind it.

One synthetic event file drives the whole figure.  Attempted passes between
every ordered pair of team mates are drawn from a distance-decay model
weighted by role, each attempt is completed with a probability that falls with
pass length, and everything else is read back out of that matrix: node area is
the player's involvement (passes made plus received), arc width is the
completed-pass count, the matrix panel is the matrix itself, the ranking is an
eigenvector centrality computed by power iteration on its symmetrised form,
and the headline completion rate is the ratio of the two totals.

Pitch geometry is the FIFA standard 105 x 68 m surface.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import (halo_text, panel_labels_inches, safe_colorbar,
                    shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - the team under analysis
GOLD = "#c9a227"          # house gold - the most involved link
LINE = "#b3b3b3"          # pitch markings on a white surface

PITCH_L, PITCH_W = 105.0, 68.0
ARC_MIN = 5               # passes; below this the arc is not drawn
TOTAL_ATTEMPTS = 520.0

# shirt number, position, average x, average y, involvement weight
SQUAD = [
    (1,  "GK",  9.0, 34.0, 0.55),
    (2,  "RB", 27.0, 58.0, 1.05),
    (5,  "RCB", 22.0, 43.0, 1.20),
    (6,  "LCB", 22.0, 25.0, 1.20),
    (3,  "LB", 27.0, 10.0, 1.05),
    (4,  "DM", 44.0, 34.0, 1.45),
    (8,  "RCM", 57.0, 47.0, 1.25),
    (10, "LCM", 60.0, 23.0, 1.30),
    (7,  "RW", 78.0, 59.0, 0.95),
    (9,  "ST", 84.0, 35.0, 0.85),
    (11, "LW", 78.0,  9.0, 0.95),
]


def draw_pitch(ax, color=LINE, lw=1.0, zorder=3):
    """Standard 105 x 68 m markings; drawn under the network."""
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.plot([PITCH_L / 2.0, PITCH_L / 2.0], [0.0, PITCH_W], color=color,
            lw=lw, zorder=zorder, solid_capstyle="butt")
    ax.add_patch(Circle((PITCH_L / 2.0, PITCH_W / 2.0), 9.15, fill=False,
                        edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Circle((PITCH_L / 2.0, PITCH_W / 2.0), 0.40,
                        facecolor=color, edgecolor="none", zorder=zorder))

    arc_half = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    for side in (0, 1):
        x_box = 0.0 if side == 0 else PITCH_L - 16.5
        x_six = 0.0 if side == 0 else PITCH_L - 5.5
        x_spot = 11.0 if side == 0 else PITCH_L - 11.0
        ax.add_patch(Rectangle((x_box, PITCH_W / 2.0 - 20.16), 16.5, 40.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Rectangle((x_six, PITCH_W / 2.0 - 9.16), 5.5, 18.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Circle((x_spot, PITCH_W / 2.0), 0.40, facecolor=color,
                            edgecolor="none", zorder=zorder))
        t1, t2 = ((-arc_half, arc_half) if side == 0
                  else (180.0 - arc_half, 180.0 + arc_half))
        ax.add_patch(Arc((x_spot, PITCH_W / 2.0), 2 * 9.15, 2 * 9.15,
                         theta1=t1, theta2=t2, edgecolor=color, lw=lw,
                         zorder=zorder))
        x_goal = -1.9 if side == 0 else PITCH_L
        ax.add_patch(Rectangle((x_goal, PITCH_W / 2.0 - 3.66), 1.9, 7.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))

    for (cx, cy, t1, t2) in ((0.0, 0.0, 0.0, 90.0),
                             (0.0, PITCH_W, 270.0, 360.0),
                             (PITCH_L, 0.0, 90.0, 180.0),
                             (PITCH_L, PITCH_W, 180.0, 270.0)):
        ax.add_patch(Arc((cx, cy), 2.0, 2.0, theta1=t1, theta2=t2,
                         edgecolor=color, lw=lw, zorder=zorder))


def eigen_centrality(w, iters=400):
    """Power iteration on a symmetric weight matrix; returns a 0..1 score."""
    s = 0.5 * (w + w.T)
    v = np.ones(s.shape[0]) / s.shape[0]
    for _ in range(iters):
        v_new = s @ v
        n = np.linalg.norm(v_new)
        if n < 1e-12:
            break
        v = v_new / n
    v = np.abs(v)
    return v / v.max()


def main():
    set_pub_style()
    rng = np.random.default_rng(607)

    nums = np.array([p[0] for p in SQUAD])
    labs = [p[1] for p in SQUAD]
    xy = np.array([[p[2], p[3]] for p in SQUAD], dtype=float)
    invo = np.array([p[4] for p in SQUAD])
    n = len(SQUAD)

    # ---- attempted passes: distance decay x role weight x noise -----------
    d = np.hypot(xy[:, 0, None] - xy[None, :, 0],
                 xy[:, 1, None] - xy[None, :, 1])
    forward = 1.0 + 0.35 * np.tanh((xy[None, :, 0] - xy[:, 0, None]) / 22.0)
    base = (np.exp(-d / 15.0) * forward *
            invo[:, None] ** 0.9 * invo[None, :] ** 0.9)
    np.fill_diagonal(base, 0.0)
    base *= rng.gamma(7.0, 1.0 / 7.0, base.shape)
    attempts = np.round(TOTAL_ATTEMPTS * base / base.sum()).astype(int)

    # completion falls with pass length, so long diagonals are the loose ones
    p_ok = np.clip(0.965 - 0.0050 * d, 0.62, 0.96)
    completed = rng.binomial(attempts, p_ok)
    acc = 100.0 * completed.sum() / max(attempts.sum(), 1)

    involve = completed.sum(axis=1) + completed.sum(axis=0)
    cent = eigen_centrality(completed.astype(float))

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.72],
                          left=0.030, right=0.900, top=0.850, bottom=0.190,
                          wspace=0.34, hspace=0.58)
    ax_n = fig.add_subplot(gs[:, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_pitch(ax_n, zorder=3)

    # ---- arcs --------------------------------------------------------------
    rad = 1.9 + 2.3 * np.sqrt(involve / involve.max())
    shown = completed >= ARC_MIN
    wmax = float(completed[shown].max())
    top = np.unravel_index(int(np.argmax(completed)), completed.shape)
    for i in range(n):
        for j in range(n):
            if not shown[i, j]:
                continue
            w = completed[i, j] / wmax
            dx, dy = xy[j] - xy[i]
            dist = float(np.hypot(dx, dy)) or 1.0
            ux, uy = dx / dist, dy / dist
            p0 = xy[i] + np.array([ux, uy]) * (rad[i] + 0.5)
            p1 = xy[j] - np.array([ux, uy]) * (rad[j] + 1.0)
            hot = (i, j) == top
            ax_n.add_patch(FancyArrowPatch(
                tuple(p0), tuple(p1), arrowstyle="-|>",
                connectionstyle="arc3,rad=0.13",
                mutation_scale=6.0 + 6.0 * w,
                lw=0.45 + 2.15 * w, color=GOLD if hot else HOME,
                alpha=0.95 if hot else 0.28 + 0.52 * w,
                shrinkA=0, shrinkB=0, zorder=5 if not hot else 6))

    for k in range(n):
        ax_n.add_patch(Circle(tuple(xy[k]), rad[k], facecolor="white",
                              edgecolor=HOME, lw=1.4, zorder=7))
        ax_n.text(xy[k, 0], xy[k, 1], str(nums[k]), ha="center", va="center",
                  fontsize=6.6, color=HOME, fontweight="bold", zorder=8)
        halo_text(ax_n, xy[k, 0], xy[k, 1] - rad[k] - 1.0, labs[k],
                  ha="center", va="top", fontsize=6.0, color=INK,
                  halo="white", halo_lw=1.8, zorder=8)

    ax_n.annotate("", xy=(84.0, 73.5), xytext=(64.0, 73.5),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_n.text(62.0, 73.5, "attacking direction", ha="right", va="center",
              fontsize=6.4, color=MUTED)

    ax_n.set_xlim(-4.5, 109.5)
    ax_n.set_ylim(-4.0, 79.0)
    ax_n.set_aspect("equal")
    ax_n.set_xticks([])
    ax_n.set_yticks([])
    for s in ax_n.spines.values():
        s.set_visible(False)
    ax_n.set_title("Completed passes between average positions", fontsize=9.0,
                   fontweight="bold", pad=6)

    annotate_box(
        ax_n,
        f"attempted  {attempts.sum():.0f} passes\n"
        f"completed  {completed.sum():.0f} ({acc:.1f}%)\n"
        f"strongest link  #{nums[top[0]]} to #{nums[top[1]]}  "
        f"{completed[top]:.0f} passes",
        loc="upper left", fontsize=6.6, margin=0.014)

    # ---- pass matrix -------------------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    norm = shared_norm([completed], vmin=0.0)
    im = ax_m.imshow(completed, cmap=cmap, norm=norm, origin="upper",
                     interpolation="nearest")
    ax_m.set_xticks(range(n))
    ax_m.set_yticks(range(n))
    ax_m.set_xticklabels([str(v) for v in nums], fontsize=6.0)
    ax_m.set_yticklabels([str(v) for v in nums], fontsize=6.0)
    ax_m.set_xlabel("receiver (shirt no.)", fontsize=7.4)
    ax_m.set_ylabel("passer (shirt no.)", fontsize=7.4)
    ax_m.tick_params(length=1.8, width=0.6)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax_m.add_patch(Rectangle((top[1] - 0.5, top[0] - 0.5), 1.0, 1.0,
                             fill=False, edgecolor=GOLD, lw=1.3, zorder=5))
    ax_m.set_title("Pass matrix", fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, im, ax_m, label="completed passes (n)", side="right",
                  width=0.10, pad=0.06, labelsize=6.5)

    # ---- centrality ranking ------------------------------------------------
    order = np.argsort(cent)
    yy = np.arange(n)
    cols = ["#9cc3dd"] * n
    cols[-1] = HOME
    ax_c.barh(yy, cent[order], height=0.66, color=cols, edgecolor="white",
              lw=0.6, zorder=3)
    for y, v in zip(yy, cent[order]):
        ax_c.text(v + 0.015, y, f"{v:.2f}", ha="left", va="center",
                  fontsize=6.0, color=SUB)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels([f"#{nums[i]} {labs[i]}" for i in order],
                         fontsize=6.2)
    ax_c.set_xlim(0.0, 1.16)
    ax_c.set_xlabel("eigenvector centrality (-)", fontsize=7.4)
    ax_c.tick_params(axis="x", labelsize=6.5)
    ax_c.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_c.set_title("Who holds the network together", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color=HOME, lw=2.0,
               label=f"completed pass link (>= {ARC_MIN} passes)"),
        Line2D([0], [0], color=GOLD, lw=2.0, label="strongest link"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=HOME, markeredgewidth=1.2, markersize=7.0,
               label="node area = passes made + received")],
        loc="lower center", bbox_to_anchor=(0.5, 0.078), ncol=3,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=2.0)

    panel_labels_inches(fig, [ax_n, ax_m, ax_c], dx=-0.26, dy=0.14)


    print(f"attempts {attempts.sum()}  completed {completed.sum()}  "
          f"accuracy {acc:.2f}%")
    print(f"links drawn {int(shown.sum())}  max link {completed[top]} "
          f"(#{nums[top[0]]}->#{nums[top[1]]})")
    print("centrality:", dict(zip(nums.tolist(), cent.round(2).tolist())))
    savefig(fig, "607_spo_pass_network.png")


if __name__ == "__main__":
    main()
