"""Wm blue control loop: the textbook control diagram, world-model flavoured.

The IEEE engineering dialect at its purest: light-grey blocks with hairline
black edges, two summing junctions drawn as true circles, transfer functions
set in mathtext, and one thick cyan-blue arrow class for the signal path.
Everything sits on an inch canvas with ``set_aspect("equal")``, which is the
only way a ``Circle`` stays circular (DESIGN_SYSTEM.md section 12.8).

The dashed warm rectangle marks the single learned component; the rest of the
loop is classical, and that contrast is the point of the figure.
"""
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas, anchors

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
CYAN = "#1c86b8"
INK = "#22272b"
GREY = "#e8eaec"
ACCENT = "#b5544a"

W, H = 11.0, 4.6
Y_MAIN, Y_FB = 2.45, 1.05
R_SUM = 0.22
LW_EDGE = 0.9
LW_FLOW = 2.6
MS_FLOW = 20


def block(ax, x0, x1, y, h, name, tf, fs=8.6, zorder=4):
    w = x1 - x0
    ax.add_patch(Rectangle((x0, y - h / 2), w, h, facecolor=GREY,
                           edgecolor=INK, linewidth=LW_EDGE, zorder=zorder))
    ax.text((x0 + x1) / 2, y + 0.15, name, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=zorder + 2)
    ax.text((x0 + x1) / 2, y - 0.19, tf, ha="center", va="center",
            fontsize=8.2, color=BLUE[1], zorder=zorder + 2)
    return anchors((x0 + x1) / 2, y, w, h)


def summing(ax, x, y, zorder=5):
    ax.add_patch(Circle((x, y), R_SUM, facecolor="white", edgecolor=INK,
                        linewidth=LW_EDGE, zorder=zorder))
    d = 0.085
    ax.plot([x - d, x + d], [y, y], color=INK, lw=0.9, zorder=zorder + 1)
    ax.plot([x, x], [y - d, y + d], color=INK, lw=0.9, zorder=zorder + 1)
    return anchors(x, y, 2 * R_SUM, 2 * R_SUM)


def flow(ax, p0, p1, lw=LW_FLOW, ms=MS_FLOW, color=CYAN, zorder=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=zorder))


def route(ax, pts, lw=LW_FLOW, ms=MS_FLOW, color=CYAN, zorder=6):
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color,
            lw=lw, solid_capstyle="projecting", solid_joinstyle="miter",
            zorder=zorder)
    flow(ax, pts[-2], pts[-1], lw=lw, ms=ms, color=color, zorder=zorder)


def signal(ax, x, y, text, dy=0.18):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=8.0,
            color=INK)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)

    ax.text(0.15, 4.32, "Closed-loop control with one learned block",
            fontsize=10.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(0.15, 4.06, "A world-model predictive controller dropped into an "
            "otherwise classical loop; everything outside the dashed box is "
            "hand-designed",
            fontsize=7.2, color="#555555", ha="left", va="center")

    sum1 = summing(ax, 1.52, Y_MAIN)
    ctrl = block(ax, 2.30, 3.90, Y_MAIN, 0.86, "World-model MPC",
                 r"$C(s) = K_p + K_i/s$", fs=8.2)
    act = block(ax, 4.50, 5.90, Y_MAIN, 0.86, "Actuator",
                r"$A(s) = 1/(\tau s + 1)$", fs=8.6)
    sum2 = summing(ax, 6.70, Y_MAIN)
    plant = block(ax, 7.50, 9.10, Y_MAIN, 0.86, "Embodied plant",
                  r"$G(s) = K/[s(Ts+1)]$", fs=8.6)
    sens = block(ax, 5.95, 7.45, Y_FB, 0.72, "Sensor", r"$H(s)$", fs=8.6)

    # Forward path.
    flow(ax, (0.45, Y_MAIN), sum1["l"])
    flow(ax, sum1["r"], ctrl["l"])
    flow(ax, ctrl["r"], act["l"])
    flow(ax, act["r"], sum2["l"])
    flow(ax, sum2["r"], plant["l"])
    flow(ax, plant["r"], (10.52, Y_MAIN))
    flow(ax, (6.70, Y_MAIN + 1.17), (6.70, sum2["t"][1]))

    # Feedback path: tap the output, through the sensor, back to the junction.
    x_tap = 9.85
    ax.plot([x_tap], [Y_MAIN], marker="o", ms=5.0, color=CYAN, zorder=7)
    route(ax, [(x_tap, Y_MAIN), (x_tap, Y_FB), sens["r"]])
    route(ax, [sens["l"], (1.52, Y_FB), sum1["b"]])

    # Signal names.
    signal(ax, 0.68, Y_MAIN, r"$r(t)$")
    signal(ax, 2.02, Y_MAIN, r"$e(t)$")
    signal(ax, 4.24, Y_MAIN, r"$u(t)$")
    signal(ax, 10.30, Y_MAIN, r"$y(t)$")
    ax.text(6.86, Y_MAIN + 1.05, r"$d(t)$", ha="left", va="center",
            fontsize=8.0, color=INK)
    ax.text(3.70, Y_FB + 0.16, r"measured state $\hat{y}(t)$", ha="center",
            va="bottom", fontsize=7.6, color=INK)

    # Junction signs.
    for x, y, s in [(1.30, Y_MAIN + 0.21, "+"), (1.30, Y_MAIN - 0.21, r"$-$"),
                    (6.48, Y_MAIN + 0.21, "+"), (6.94, Y_MAIN + 0.27, "+")]:
        ax.text(x, y, s, ha="center", va="center", fontsize=9.0, color=INK,
                zorder=8)

    # The one warm element: which block is learned rather than designed.
    ax.add_patch(Rectangle((2.20, Y_MAIN - 0.60), 1.80, 1.20,
                           facecolor="none", edgecolor=ACCENT, linewidth=1.1,
                           linestyle=(0, (4, 2.6)), zorder=3))
    ax.text(3.10, Y_MAIN + 0.66, "learned from data", ha="center",
            va="bottom", fontsize=7.0, color=ACCENT, style="italic")

    ax.text(0.15, 0.30, "Transfer functions are illustrative; the loop "
            "topology and the placement of the learned block are what the "
            "figure asserts.", fontsize=6.8, color="#777777", ha="left",
            va="center")

    savefig(fig, "1103_wm_blue_control_loop.png")


if __name__ == "__main__":
    main()
