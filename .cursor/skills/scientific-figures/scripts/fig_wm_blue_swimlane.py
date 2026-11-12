"""Wm blue swimlane: who does what in a world-model training loop.

Three tinted lanes -- simulator, world model, policy -- with the step blocks
in the flat blue dialect (solid fill, white bold type).  Every hand-off
between lanes is drawn as a right-angle connector rather than a diagonal, so
the reader can follow ownership by eye: a vertical segment always means the
work changed hands.  The dashed warm line under the bottom lane is the outer
loop; the branch into "held-out eval" is the only path that leaves the loop.

Layout is a grid of six columns on an inch canvas; blocks in different lanes
may share a column, which is what keeps the verticals honest.  Each lane is
deliberately spread across the grid rather than packed to one side: a lane
that fills three neighbouring columns and then runs empty for half the canvas
reads as unfinished, not as breathing room.  Where that forces a step to skip
a column, the resulting long arrow is labelled so its length looks chosen.
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas, anchors

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
LANE_A, LANE_B = "#f2f7fb", "#e6eff7"   # alternating lane wash
STEP = BLUE[3]
FLOW = BLUE[2]

W, H = 10.4, 5.2
X_LANE0, X_LANE1 = 0.15, 10.25
TAB_X, TAB_W = 0.20, 1.12
COLS = [2.20, 3.63, 5.06, 6.49, 7.92, 9.35]   # uniform 1.43 in pitch
BW, BH = 1.22, 0.66
LANE_TOP, LANE_H, LANE_GAP = 4.20, 1.12, 0.12

LANES = [("Simulator", BLUE[0], LANE_A),
         ("World model", BLUE[1], LANE_B),
         ("Policy", BLUE[2], LANE_A)]


def lane_y(i):
    return LANE_TOP - LANE_H / 2 - i * (LANE_H + LANE_GAP)


def lane_band(ax, i):
    name, tab_col, wash = LANES[i]
    y = lane_y(i)
    ax.add_patch(Rectangle((X_LANE0, y - LANE_H / 2), X_LANE1 - X_LANE0,
                           LANE_H, facecolor=wash, edgecolor="none",
                           zorder=1))
    ax.add_patch(FancyBboxPatch(
        (TAB_X, y - 0.31), TAB_W, 0.62,
        boxstyle="round,pad=0,rounding_size=0.09", facecolor=tab_col,
        edgecolor="none", zorder=3))
    ax.text(TAB_X + TAB_W / 2, y, name, ha="center", va="center",
            color="white", fontsize=8.0, fontweight="bold", zorder=5)


def step(ax, col, lane, label, color=STEP):
    x, y = COLS[col], lane_y(lane)
    ax.add_patch(FancyBboxPatch(
        (x - BW / 2, y - BH / 2), BW, BH,
        boxstyle="round,pad=0,rounding_size=0.10", facecolor=color,
        edgecolor="none", zorder=4))
    ax.text(x, y, label, ha="center", va="center", color="white",
            fontsize=7.6, fontweight="bold", linespacing=1.3, zorder=6)
    return anchors(x, y, BW, BH)


def arrow(ax, p0, p1, style="arc3,rad=0", color=FLOW, lw=1.5, ms=13,
          ls="-", zorder=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=style, shrinkA=0, shrinkB=0,
                                 zorder=zorder))


def elbow(ax, p0, p1, **kw):
    """Leave horizontally, arrive vertically -- the swimlane hand-off."""
    arrow(ax, p0, p1, style="angle,angleA=0,angleB=90,rad=0", **kw)


def polyline_arrow(ax, pts, color=FLOW, lw=1.5, ls="-", ms=13, zorder=7):
    """Explicit right-angle route; the last leg carries the head."""
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color,
            lw=lw, ls=ls, solid_capstyle="butt", solid_joinstyle="miter",
            zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
          zorder=zorder)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)

    ax.text(0.15, 5.00, "Who does what in a world-model training loop",
            fontsize=10.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(0.15, 4.74, "Lane changes are drawn as right angles: a vertical "
            "segment always means the work changed hands",
            fontsize=7.2, color="#555555", ha="left", va="center")

    for i in range(3):
        lane_band(ax, i)

    # Columns 0-5; each lane is spread over the grid rather than clustered.
    sim_rand = step(ax, 0, 0, "Domain\nrandomiser")
    sim_roll = step(ax, 1, 0, "Physics\nrollout")
    sim_scene = step(ax, 3, 0, "Scene\nresampler")
    sim_eval = step(ax, 5, 0, "Held-out\neval suite", color=BLUE[2])
    wm_buf = step(ax, 1, 1, "Replay\nbuffer")
    wm_enc = step(ax, 2, 1, "Observation\nencoder")
    wm_fit = step(ax, 3, 1, "Latent\ndynamics fit")
    pol_start = step(ax, 0, 2, "Current\npolicy")
    pol_expl = step(ax, 2, 2, "Exploration\nschedule")
    pol_img = step(ax, 4, 2, "Imagined\nrollout")
    pol_upd = step(ax, 5, 2, "Actor-critic\nupdate")

    arrow(ax, pol_start["t"], sim_rand["b"])
    arrow(ax, sim_rand["r"], sim_roll["l"])
    arrow(ax, sim_roll["r"], sim_scene["l"])
    arrow(ax, sim_scene["r"], sim_eval["l"])
    arrow(ax, sim_roll["b"], wm_buf["t"])
    arrow(ax, wm_buf["r"], wm_enc["l"])
    arrow(ax, wm_enc["r"], wm_fit["l"])
    elbow(ax, wm_fit["r"], pol_img["t"])
    arrow(ax, pol_start["r"], pol_expl["l"])
    arrow(ax, pol_expl["r"], pol_img["l"])
    arrow(ax, pol_img["r"], pol_upd["l"])
    arrow(ax, pol_upd["t"], sim_eval["b"])

    y_sim, y_pol = lane_y(0), lane_y(2)
    for x, y, txt, ha in [
            (COLS[0] + 0.10, 2.40, "acts in the sim", "left"),
            (COLS[1] + 0.10, 3.04, "logged episodes", "left"),
            (COLS[5] - 0.10, 2.40, "frozen checkpoint", "right"),
            (COLS[3] + BW / 2 + 0.06, 2.56, "learned model", "left"),
            ((COLS[1] + COLS[3]) / 2, y_sim + 0.15, "fresh scene set",
             "center"),
            ((COLS[3] + COLS[5]) / 2, y_sim + 0.15, "held-out split",
             "center"),
            ((COLS[0] + COLS[2]) / 2, y_pol + 0.15, "current weights",
             "center"),
            ((COLS[2] + COLS[4]) / 2, y_pol + 0.15, "exploration noise",
             "center")]:
        ax.text(x, y, txt, ha=ha, va="center", fontsize=6.3,
                color="#5c6670", zorder=8)

    # Outer loop: the one warm element, kept to a hairline under the lanes.
    y_ret = 0.38
    polyline_arrow(ax, [pol_upd["b"], (COLS[5], y_ret), (COLS[0], y_ret),
                        (COLS[0], pol_start["b"][1])],
                   color=ACCENT, lw=1.2, ls=(0, (5, 3)), ms=12)
    ax.text((COLS[0] + COLS[5]) / 2, y_ret - 0.20,
            "next outer iteration: the updated policy collects fresh episodes",
            ha="center", va="center", fontsize=6.8, color=ACCENT,
            style="italic")

    savefig(fig, "1102_wm_blue_swimlane.png")


if __name__ == "__main__":
    main()
