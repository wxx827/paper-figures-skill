"""Directed pass network on a full-size pitch, ranked by PageRank.

One synthetic event file drives every panel.  Attempted passes between each
ordered pair of team mates are drawn from a distance-decay model weighted by
the two roles, each attempt is completed with a probability that falls with
pass length, and the rest is read back out of that directed matrix: node area
is the player's involvement, arc width is the completed-pass count in that
direction, the butterfly panel is the row and column sums of the matrix, and
the ranking is a PageRank computed by power iteration on the column-normalised
completed-pass matrix.

PageRank is the right centrality for a pass network because the graph is
*directed*: a player who is picked out by team mates who are themselves picked
out is a genuine circulation hub, whereas a symmetrised degree count cannot
tell "receives from everyone" from "passes to everyone".

Pitch geometry is the FIFA standard 105 x 68 m surface.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from pubstyle import annotate_box, savefig, set_pub_style
from figkit import axes_inches, halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - Team A
LIGHT = "#9cc3dd"         # the same blue, lightened for secondary bars
ACCENT = "#b23a48"        # house red - team centroid
GOLD = "#c9a227"          # house gold - formation lines
GRASS = "#f4f4f2"
LINE = "#c2c2c2"

PITCH_L, PITCH_W = 105.0, 68.0
EDGE_MIN = 6              # completed passes needed before an arc is drawn
TEAM = "Team A"

# shirt, position, unit, x (m), y (m), attempt weight
SQUAD = [
    (1,  "GK",  "keeper",  9.0, 34.0, 0.55),
    (2,  "RB",  "defence", 30.0, 58.5, 1.05),
    (5,  "RCB", "defence", 24.0, 42.5, 1.15),
    (6,  "LCB", "defence", 24.0, 25.5, 1.15),
    (3,  "LB",  "defence", 30.0, 9.5, 1.05),
    (4,  "DM",  "midfield", 44.0, 34.0, 1.45),
    (8,  "RCM", "midfield", 57.0, 47.5, 1.35),
    (10, "LCM", "midfield", 60.0, 23.0, 1.30),
    (7,  "RW",  "attack", 79.0, 58.0, 0.95),
    (9,  "ST",  "attack", 85.0, 34.0, 0.85),
    (11, "LW",  "attack", 79.0, 10.5, 0.95),
]
UNITS = ["defence", "midfield", "attack"]


def draw_pitch(ax, color=LINE, lw=0.9, zorder=5):
    """Standard 105 x 68 m markings on a near-neutral surface."""
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, facecolor=GRASS,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.plot([PITCH_L / 2.0] * 2, [0.0, PITCH_W], color=color, lw=lw,
            zorder=zorder, solid_capstyle="butt")
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
        ax.add_patch(Circle((x_spot, PITCH_W / 2.0), 0.35, facecolor=color,
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


def curved_arrow(ax, p0, p1, gap_a, gap_b, rad=0.11, **kw):
    """Arc arrow whose ends are trimmed by the two node radii, in metres."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    d = float(np.hypot(dx, dy))
    if d < 1e-6:
        return None
    ux, uy = dx / d, dy / d
    a = FancyArrowPatch((x0 + ux * gap_a, y0 + uy * gap_a),
                        (x1 - ux * gap_b, y1 - uy * gap_b),
                        arrowstyle="-|>", mutation_scale=6.5,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(a)
    return a


def pagerank(mat, damping=0.85, iters=300):
    """Power iteration on the column-normalised matrix; dangling nodes spread."""
    n = mat.shape[0]
    col = mat.sum(axis=0)
    m = np.where(col > 0, mat / np.where(col > 0, col, 1.0), 1.0 / n)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1.0 - damping) / n + damping * (m @ r)
    return r / r.sum()


def main():
    set_pub_style()
    rng = np.random.default_rng(865)

    n = len(SQUAD)
    pos = np.array([[s[3], s[4]] for s in SQUAD], dtype=float)
    weight = np.array([s[5] for s in SQUAD])
    unit = [s[2] for s in SQUAD]

    # ---- attempts, then completions ----------------------------------------
    dist = np.hypot(pos[:, 0][:, None] - pos[:, 0][None, :],
                    pos[:, 1][:, None] - pos[:, 1][None, :])
    forward = 1.0 + 0.55 * np.tanh((pos[:, 0][None, :] - pos[:, 0][:, None])
                                   / 22.0)
    lam = (26.0 * weight[:, None] * weight[None, :] * forward
           * np.exp(-dist / 21.0))
    np.fill_diagonal(lam, 0.0)
    attempt = rng.poisson(lam).astype(float)
    p_ok = np.clip(0.965 - 0.0072 * dist, 0.55, 0.99)
    done = rng.binomial(attempt.astype(int), p_ok).astype(float)

    made = attempt.sum(axis=1)
    made_ok = done.sum(axis=1)
    recv = done.sum(axis=0)
    comp = 100.0 * made_ok / np.maximum(made, 1.0)
    involve = made_ok + recv
    team_comp = 100.0 * done.sum() / attempt.sum()

    rank = pagerank(done)
    centroid = np.average(pos, axis=0, weights=involve)
    depth = pos[1:, 0].max() - pos[1:, 0].min()
    width = pos[1:, 1].max() - pos[1:, 1].min()

    # ---- layout -------------------------------------------------------------
    FW, FH = 11.0, 6.3
    fig = plt.figure(figsize=(FW, FH))
    ax_p = axes_inches(fig, (0.38, 1.05, 6.00, 4.278))
    ax_b = axes_inches(fig, (7.25, 2.92, 3.35, 2.38))
    ax_c = axes_inches(fig, (7.25, 0.95, 3.35, 1.32))

    draw_pitch(ax_p, zorder=5)

    # formation lines: the players of one unit joined across the pitch
    for u in UNITS:
        ix = [i for i in range(n) if unit[i] == u]
        ix.sort(key=lambda i: pos[i, 1])
        ax_p.plot(pos[ix, 0], pos[ix, 1], color=GOLD, lw=0.9, ls=(0, (5, 3)),
                  alpha=0.85, zorder=6)
        mx = pos[ix, 0].mean()
        halo_text(ax_p, mx, -2.6, f"{u} line  x = {mx:.0f} m", ha="center",
                  va="top", fontsize=6.2, color=SUB, halo="white",
                  halo_lw=1.8, zorder=9)

    emax = done.max()
    radius = 1.9 + 2.5 * np.sqrt(involve / involve.max())
    for i in range(n):
        for j in range(n):
            if done[i, j] < EDGE_MIN:
                continue
            f = done[i, j] / emax
            curved_arrow(ax_p, pos[i], pos[j], radius[i] + 0.5,
                         radius[j] + 1.1, rad=0.11,
                         linewidth=0.45 + 2.3 * f,
                         color=HOME, alpha=0.22 + 0.55 * f, zorder=7)

    for i, (num, role, _u, x, y, _w) in enumerate(SQUAD):
        ax_p.add_patch(Circle((x, y), radius[i], facecolor=HOME,
                              edgecolor="white", lw=1.1, zorder=9))
        ax_p.text(x, y, str(num), ha="center", va="center", fontsize=6.6,
                  color="white", zorder=10)
        halo_text(ax_p, x, y - radius[i] - 0.8, role, ha="center", va="top",
                  fontsize=6.0, color=INK, halo="white", halo_lw=1.9,
                  zorder=10)

    ax_p.plot([centroid[0]], [centroid[1]], marker="X", ms=8.5, mfc=ACCENT,
              mec="white", mew=1.0, zorder=11)

    ax_p.annotate("", xy=(92.0, 72.4), xytext=(72.0, 72.4),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_p.text(70.0, 72.4, "attacking direction", ha="right", va="center",
              fontsize=6.4, color=MUTED)

    ax_p.set_xlim(-6.0, 111.0)
    ax_p.set_ylim(-9.0, 77.0)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax_p.set_title(f"{TEAM} passing structure, first 60 min", fontsize=9.0,
                   fontweight="bold", pad=5)

    annotate_box(
        ax_p,
        f"{attempt.sum():.0f} passes attempted, {done.sum():.0f} completed "
        f"({team_comp:.1f}%)\n"
        f"{int((done >= EDGE_MIN).sum())} of {n * (n - 1)} links drawn "
        f"(>= {EDGE_MIN} passes)\n"
        f"block depth {depth:.0f} m | block width {width:.0f} m\n"
        f"team centroid ({centroid[0]:.0f}, {centroid[1]:.0f}) m",
        loc="upper left", fontsize=6.4, margin=0.016)

    # ---- butterfly: receptions | completion rate ----------------------------
    order = np.argsort(involve)
    yy = np.arange(n)
    rmax = float(np.ceil(recv.max() / 10.0) * 10.0)
    for k, i in enumerate(order):
        ax_b.barh(k, -recv[i] / rmax, height=0.62, color=LIGHT,
                  edgecolor="white", lw=0.6, zorder=3)
        ax_b.barh(k, comp[i] / 100.0, height=0.62, color=HOME,
                  edgecolor="white", lw=0.6, zorder=3)
        ax_b.text(-recv[i] / rmax - 0.03, k, f"{recv[i]:.0f}", ha="right",
                  va="center", fontsize=6.2, color=SUB)
        ax_b.text(comp[i] / 100.0 + 0.03, k, f"{comp[i]:.0f}%", ha="left",
                  va="center", fontsize=6.2, color=SUB)
    ax_b.axvline(0.0, color=RULE, lw=0.8, zorder=5)
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels([f"#{SQUAD[i][0]} {SQUAD[i][1]}" for i in order],
                         fontsize=6.4)
    ax_b.set_xlim(-1.32, 1.30)
    ax_b.set_ylim(-0.75, n - 0.25)
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax_b.set_xticks(ticks)
    ax_b.set_xticklabels([f"{rmax:.0f}", f"{rmax / 2:.0f}", "0", "50", "100"],
                         fontsize=7.0)
    ax_b.text(-0.5, -1.85, "passes received (count)", ha="center", va="top",
              fontsize=7.4, color=SUB)
    ax_b.text(0.5, -1.85, "pass completion (%)", ha="center", va="top",
              fontsize=7.4, color=SUB)
    ax_b.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_title("Volume received against reliability given",
                   fontsize=8.5, fontweight="bold")

    # ---- PageRank ranking ----------------------------------------------------
    top = np.argsort(rank)[::-1][:6][::-1]
    yy2 = np.arange(len(top))
    ax_c.hlines(yy2, 0.0, 100.0 * rank[top], color=LIGHT, lw=1.4, zorder=3)
    ax_c.plot(100.0 * rank[top], yy2, marker="o", ls="none", ms=5.0,
              mfc=HOME, mec="white", mew=0.8, zorder=4)
    ax_c.plot([100.0 * rank[top][-1]], [yy2[-1]], marker="o", ls="none",
              ms=5.6, mfc=ACCENT, mec="white", mew=0.8, zorder=5)
    for y, i in zip(yy2, top):
        ax_c.text(100.0 * rank[i] + 0.35, y, f"{100.0 * rank[i]:.1f}",
                  ha="left", va="center", fontsize=6.2, color=SUB)
    ax_c.axvline(100.0 / n, color="#808080", lw=0.9, ls="--", zorder=2)
    ax_c.text(100.0 / n - 0.30, len(top) - 0.45, "even share", ha="right",
              va="top", fontsize=6.2, color=MUTED)
    ax_c.set_yticks(yy2)
    ax_c.set_yticklabels([f"#{SQUAD[i][0]} {SQUAD[i][1]}" for i in top],
                         fontsize=6.4)
    ax_c.set_xlim(0.0, 100.0 * rank.max() * 1.30)
    ax_c.set_ylim(-0.85, len(top) - 0.4)
    ax_c.set_xlabel("PageRank share of the network (%)", fontsize=8.0)
    ax_c.tick_params(axis="x", labelsize=7.0)
    ax_c.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_title("Circulation hubs (top 6)", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME,
               markeredgecolor="white", markersize=6.5,
               label="player node, area = passes made + received"),
        Line2D([0], [0], color=HOME, lw=2.0, alpha=0.7,
               label=f"completed passes, width = volume (>= {EDGE_MIN})"),
        Line2D([0], [0], color=GOLD, lw=0.9, ls=(0, (5, 3)),
               label="unit line"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=7,
               label="team centroid")],
        loc="lower left", bbox_to_anchor=(0.028, 0.050), ncol=4,
        frameon=False, fontsize=7.0, handlelength=1.9, columnspacing=1.7)

    panel_labels_inches(fig, [ax_p, ax_b, ax_c], dx=-0.30, dy=0.12)

    hub = int(np.argmax(rank))

    print(f"attempts {attempt.sum():.0f}  completed {done.sum():.0f}  "
          f"{team_comp:.1f}%  links {int((done >= EDGE_MIN).sum())}")
    print("completion %:", comp.round(1).tolist())
    print("pagerank %:", (100.0 * rank).round(2).tolist())
    savefig(fig, "865_tac_pass_network.png")


if __name__ == "__main__":
    main()
