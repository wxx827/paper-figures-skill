"""Wm blue infoflow: modular versus end-to-end world-model information flow.

The IEEE-Access "engineering flow" dialect: grey blocks with hairline black
edges, stacked cylinders for anything that stores or accumulates state, and
one very thick cyan-blue arrow class carrying the signal path.  Row (a) opens
the pipeline into its modules inside a plain black frame; row (b) collapses
the same span into a single end-to-end store, so the two rows can be read as
a before/after by eye alone.  Both rows close over the top of the canvas.

Everything is drawn on an inch canvas: the two rows reuse the same x anchors,
which is what makes the collapse legible.
"""
from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas, anchors

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
CYAN = "#1c86b8"            # the thick signal arrow, 3-4x a normal arrow
INK = "#22272b"
GREY = "#e8eaec"
ACCENT = "#b5544a"

W, H = 10.4, 5.2
LW_EDGE = 0.9
LW_FLOW = 3.0
MS_FLOW = 22

# Shared column anchors -- row (a) and row (b) must line up exactly.
X_ENV, W_ENV = 0.900, 1.62
X_SEN, W_SEN = 2.470, 0.95
X_FRAME0, X_FRAME1 = 3.22, 7.80
X_ACT, W_ACT = 8.58, 1.00
X_VEH, W_VEH = 9.835, 0.95


def rect_block(ax, x, y, w, h, label, fs=7.6, fc=GREY, zorder=4):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                           edgecolor=INK, linewidth=LW_EDGE, zorder=zorder))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold", linespacing=1.25, zorder=zorder + 2)
    return anchors(x, y, w, h)


def pill_block(ax, x, y, w, h, label, fs=7.6, fc=GREY, zorder=4):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + h / 2, y - h / 2), w - h, h,
        boxstyle=f"round,pad=0,rounding_size={h / 2}", facecolor=fc,
        edgecolor=INK, linewidth=LW_EDGE, zorder=zorder))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold", linespacing=1.25, zorder=zorder + 2)
    return anchors(x, y, w, h)


def cylinder(ax, x, y, w, h, segments, fs=7.0, fc=GREY, cap=0.09, zorder=4):
    """Rectangle plus two ellipse caps; extra ellipses divide the segments.

    The bottom cap is drawn first and then covered by the opaque body, which
    is what leaves only its lower arc visible -- the standard way to fake a
    3D store without importing a 3D axes.

    Sizing rule learned the hard way: a divider ellipse bulges ``cap`` inches
    into the slab on *each* side of the boundary, so segment text collides
    with it unless ``cap <= 0.09`` and every slab is at least ~0.5 in tall
    (i.e. ``h >= 2*cap + 0.5*len(segments)`` for two-line labels at 7 pt).
    At ``cap = 0.11`` with 0.39 in slabs the arc cuts straight through the
    second line of text, and nothing warns you -- it just renders wrong.
    """
    top_y, bot_y = y + h / 2 - cap, y - h / 2 + cap
    ax.add_patch(Ellipse((x, bot_y), w, 2 * cap, facecolor=fc,
                         edgecolor=INK, linewidth=LW_EDGE, zorder=zorder))
    ax.add_patch(Rectangle((x - w / 2, bot_y), w, top_y - bot_y,
                           facecolor=fc, edgecolor="none", zorder=zorder + 1))
    for sx in (x - w / 2, x + w / 2):
        ax.plot([sx, sx], [bot_y, top_y], color=INK, lw=LW_EDGE,
                zorder=zorder + 2)
    n = len(segments)
    slab = (top_y - bot_y) / n
    for k in range(1, n):
        ax.add_patch(Ellipse((x, top_y - k * slab), w, 2 * cap,
                             facecolor="none", edgecolor=INK,
                             linewidth=LW_EDGE, zorder=zorder + 2))
    ax.add_patch(Ellipse((x, top_y), w, 2 * cap, facecolor=fc, edgecolor=INK,
                         linewidth=LW_EDGE, zorder=zorder + 2))
    for k, seg in enumerate(segments):
        ax.text(x, top_y - (k + 0.5) * slab, seg, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.2, zorder=zorder + 4)
    return anchors(x, y, w, h)


def flow(ax, p0, p1, color=CYAN, lw=LW_FLOW, ms=MS_FLOW, zorder=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=zorder))


def bar(ax, pts, color=CYAN, lw=LW_FLOW, zorder=6):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            solid_capstyle="projecting", solid_joinstyle="miter",
            zorder=zorder)


def chain(ax, y, middle_left, middle_right):
    """The five outer stations shared by both rows; returns the two ends."""
    env = pill_block(ax, X_ENV, y, W_ENV, 0.62, "Environmental\nstates",
                     fs=7.2)
    sen = rect_block(ax, X_SEN, y, W_SEN, 0.55, "Sensors")
    act = rect_block(ax, X_ACT, y, W_ACT, 0.55, "Actuators")
    veh = rect_block(ax, X_VEH, y, W_VEH, 0.55, "Vehicle\nstates")
    flow(ax, env["r"], sen["l"])
    flow(ax, sen["r"], (middle_left, y))
    flow(ax, (middle_right, y), act["l"])
    flow(ax, act["r"], veh["l"])
    return env, sen, veh


def feedback(ax, env, sen, veh, y_bar, note):
    bar(ax, [veh["t"], (X_VEH, y_bar), (X_ENV, y_bar)])
    flow(ax, (X_SEN, y_bar), (X_SEN, sen["t"][1]))
    flow(ax, (X_ENV, y_bar), (X_ENV, env["t"][1]))
    ax.text(X_VEH - 0.20, y_bar + 0.11, note, ha="right", va="bottom",
            fontsize=6.8, color=INK)


def row_modular(ax):
    y = 3.72
    ax.add_patch(Rectangle((X_FRAME0, 3.02), X_FRAME1 - X_FRAME0, 1.56,
                           facecolor="white", edgecolor=INK, linewidth=1.0,
                           zorder=2))
    ax.text((X_FRAME0 + X_FRAME1) / 2, 4.44, "A generic modular pipeline",
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color=INK, zorder=3)
    env, sen, veh = chain(ax, y, X_FRAME0 - 0.005, X_FRAME1 + 0.005)

    c1 = cylinder(ax, 3.98, y, 1.28, 1.18,
                  ["Object detection\n/ tracking", "Localization"])
    c2 = cylinder(ax, 5.38, y, 1.16, 1.18,
                  ["Dynamics model", "Behaviour\nprediction"])
    c3 = cylinder(ax, 6.50, y, 0.72, 0.58, ["Planning"])
    c4 = cylinder(ax, 7.36, y, 0.64, 0.58, ["Control"])
    for a, b in zip([c1, c2, c3], [c2, c3, c4]):
        flow(ax, a["r"], b["l"], lw=2.2, ms=17)

    feedback(ax, env, sen, veh, 5.03,
             "closed loop: vehicle state re-enters the world")
    ax.text((X_FRAME0 + X_FRAME1) / 2, 2.80, "(a)", ha="center", va="center",
            fontsize=9.5, color=INK)


def row_end2end(ax):
    y = 1.20
    env, sen, veh = chain(ax, y, X_FRAME0 - 0.005, X_FRAME1 + 0.005)
    cylinder(ax, (X_FRAME0 + X_FRAME1) / 2, y, X_FRAME1 - X_FRAME0, 0.82,
             ["End-to-end world model"], fs=8.4, fc=BLUE[6])
    feedback(ax, env, sen, veh, 2.35,
             "one differentiable path from pixels to torques")
    ax.text((X_FRAME0 + X_FRAME1) / 2, 0.58,
            "no module boundary left to inspect", ha="center", va="center",
            fontsize=6.8, color=ACCENT, style="italic")
    ax.text((X_FRAME0 + X_FRAME1) / 2, 0.28, "(b)", ha="center", va="center",
            fontsize=9.5, color=INK)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)
    row_modular(ax)
    row_end2end(ax)
    savefig(fig, "1100_wm_blue_infoflow.png")


if __name__ == "__main__":
    main()
