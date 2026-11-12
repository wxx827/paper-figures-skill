"""Horizontal five-stage method pipeline.

Each stage is a rounded colour block carrying a geometric icon, a spec card
with input/output shapes and key parameters, and a chevron arrow to the next
stage. A Sankey-style ribbon underneath tracks how the data volume shrinks
along the pipeline (band width scales with the square root of the volume).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Polygon, Rectangle)
from pubstyle import set_pub_style, savefig, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#7d9a72"
SAND = "#cf9f43"
RUST = "#b0563d"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"
CARD = "#f3f1ea"
STEP = [DEEP, TEAL, SAGE, SAND, RUST]

STAGES = [
    ("Acquisition", "multi-sensor ingest",
     "4 sensors x 6 bands", "18.4 TB scenes", "10 m GSD, 5-day revisit"),
    ("Harmonisation", "radiometric + geometric",
     "18.4 TB scenes", "6.2 TB cubes", "BRDF, cloud mask, reproject"),
    ("Representation", "convolutional encoder",
     "6.2 TB cubes", "940 GB features", "64 filters, 3x3, stride 2"),
    ("Inference", "temporal transformer",
     "940 GB features", "120 GB logits", "12 blocks, d = 256, h = 8"),
    ("Delivery", "gridding + uncertainty",
     "120 GB logits", "22 GB products", "1 km grid, 95% intervals"),
]
VOLUMES = [18400.0, 6200.0, 940.0, 120.0, 22.0]
STREAMS = [("Imagery", [15800, 5200, 720, 84, 12]),
           ("Ancillary", [2100, 780, 165, 27, 7]),
           ("Labels", [500, 220, 55, 9, 3])]


def square_units(fig, ax, ymax=1.0):
    fig.canvas.draw()
    bb = ax.get_position()
    w, h = fig.get_size_inches()
    xmax = ymax * (bb.width * w) / (bb.height * h)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    return xmax


def icon_tiles(ax, cx, cy, s):
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((cx - 1.45 * s + j * s, cy - 1.45 * s + i * s),
                                   0.82 * s, 0.82 * s, facecolor="white",
                                   alpha=0.35 + 0.16 * ((i + j) % 3),
                                   edgecolor="none", zorder=5))


def icon_funnel(ax, cx, cy, s):
    ax.add_patch(Polygon([(cx - 1.7 * s, cy + 1.5 * s),
                          (cx + 1.7 * s, cy + 1.5 * s),
                          (cx + 0.34 * s, cy - 0.2 * s),
                          (cx + 0.34 * s, cy - 1.1 * s),
                          (cx - 0.34 * s, cy - 1.1 * s),
                          (cx - 0.34 * s, cy - 0.2 * s)],
                         closed=True, facecolor="white", alpha=0.75,
                         edgecolor="none", zorder=5))
    for k, dy in enumerate([-1.55, -2.0]):
        ax.add_patch(Circle((cx, cy + dy * s), 0.19 * s, facecolor="white",
                            alpha=0.55 - 0.15 * k, edgecolor="none", zorder=5))


def icon_stack(ax, cx, cy, s):
    for k in range(4):
        o = (1.5 - k) * 0.42 * s
        ax.add_patch(Rectangle((cx - 1.25 * s + o, cy - 1.05 * s - o),
                               2.0 * s, 1.85 * s, facecolor="white",
                               alpha=0.28 + 0.16 * k, edgecolor="white",
                               lw=0.6, zorder=5 + k))


def icon_net(ax, cx, cy, s):
    layers = [3, 4, 3]
    xs = np.linspace(cx - 1.55 * s, cx + 1.55 * s, 3)
    pos = [[(x, cy + (i - (n - 1) / 2) * 0.95 * s) for i in range(n)]
           for x, n in zip(xs, layers)]
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="white", lw=0.45,
                        alpha=0.45, zorder=5)
    for lay in pos:
        for p in lay:
            ax.add_patch(Circle(p, 0.26 * s, facecolor="white", alpha=0.9,
                                edgecolor="none", zorder=6))


def icon_map(ax, cx, cy, s):
    ax.add_patch(FancyBboxPatch((cx - 1.7 * s, cy - 1.25 * s), 3.4 * s,
                                2.5 * s, boxstyle="round,pad=0,rounding_size="
                                f"{0.22 * s}", facecolor="white", alpha=0.28,
                                edgecolor="white", lw=0.7, zorder=5))
    t = np.linspace(-1.5, 1.5, 60)
    for k, off in enumerate([-0.72, -0.02, 0.68]):
        ax.plot(cx + t * s, cy + (off + 0.16 * np.sin(2.4 * t)) * s,
                color="white", lw=0.9, alpha=0.55 + 0.15 * k, zorder=6)
    ax.add_patch(Circle((cx + 0.55 * s, cy + 0.15 * s), 0.20 * s,
                        facecolor="white", edgecolor="none", zorder=7))


ICONS = [icon_tiles, icon_funnel, icon_stack, icon_net, icon_map]


def pipeline(fig, ax):
    xmax = square_units(fig, ax)
    ax.axis("off")
    bw, gap = 0.176, 0.030
    ytop, bh = 0.505, 0.435
    for i, (name, sub, sin, sout, par) in enumerate(STAGES):
        x0 = xmax * i * (bw + gap)
        w = xmax * bw
        col = STEP[i]
        ax.text(x0 + 0.04 * w, 0.985, f"STAGE {i + 1}", fontsize=6.4,
                color=col, fontweight="bold", va="top", ha="left")
        ax.add_patch(FancyBboxPatch(
            (x0, ytop), w, bh, boxstyle="round,pad=0,rounding_size=0.045",
            facecolor=col, edgecolor="none", zorder=3))
        ICONS[i](ax, x0 + 0.5 * w, ytop + 0.29 * bh, 0.052)
        ax.text(x0 + 0.5 * w, ytop + 0.79 * bh, name, ha="center",
                va="center", color="white", fontsize=10, fontweight="bold",
                zorder=8)
        ax.text(x0 + 0.5 * w, ytop + 0.635 * bh, sub, ha="center",
                va="center", color="white", fontsize=6.6, alpha=0.88,
                zorder=8)

        ax.add_patch(FancyBboxPatch(
            (x0, 0.055), w, 0.385,
            boxstyle="round,pad=0,rounding_size=0.045", facecolor=CARD,
            edgecolor="#e2ded3", lw=0.8, zorder=2))
        ax.add_patch(Rectangle((x0 + 0.045 * w, 0.075), 0.012, 0.345,
                               facecolor=col, edgecolor="none", zorder=3))
        for k, (lab, val) in enumerate([("input", sin), ("output", sout),
                                        ("setting", par)]):
            yy = 0.355 - 0.115 * k
            ax.text(x0 + 0.085 * w, yy, lab, fontsize=6.0, color=MUTED,
                    va="center", ha="left", zorder=4)
            ax.text(x0 + 0.085 * w, yy - 0.048, val, fontsize=6.9, color=INK,
                    va="center", ha="left", zorder=4, fontweight="bold"
                    if k < 2 else "normal")

        if i < 4:
            xa = x0 + w
            ax.add_patch(FancyArrowPatch(
                (xa + 0.006 * xmax, ytop + bh / 2),
                (xa + (gap - 0.005) * xmax, ytop + bh / 2),
                arrowstyle="-|>", mutation_scale=15, linewidth=1.8,
                color="#95a0aa", zorder=6))
            ax.add_patch(FancyArrowPatch(
                (xa + 0.008 * xmax, 0.2475),
                (xa + (gap - 0.007) * xmax, 0.2475),
                arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                color="#c3bdb1", zorder=6))


def smoothstep(a, b, t):
    t = np.clip(t, 0, 1)
    return a + (b - a) * (t * t * (3 - 2 * t))


def sankey(ax):
    stage_x = np.linspace(0.088, 0.912, 5)
    xs = np.linspace(0.0, 1.0, 900)
    curves = []
    for _, vals in STREAMS:
        y = np.empty_like(xs)
        for j in range(5):
            if j == 0:
                m = xs <= stage_x[0]
                y[m] = np.sqrt(vals[0])
            m = xs >= stage_x[-1]
            y[m] = np.sqrt(vals[-1])
        for j in range(4):
            m = (xs > stage_x[j]) & (xs < stage_x[j + 1])
            t = (xs[m] - stage_x[j]) / (stage_x[j + 1] - stage_x[j])
            t = np.clip((t - 0.22) / 0.56, 0, 1)
            y[m] = smoothstep(np.sqrt(vals[j]), np.sqrt(vals[j + 1]), t)
        curves.append(y)

    total = np.sum(curves, axis=0)
    lower = -total / 2
    cols = [TEAL, SAGE, SAND]
    for (name, _), y, c in zip(STREAMS, curves, cols):
        ax.fill_between(xs, lower, lower + y, color=c, alpha=0.86,
                        linewidth=0.7, edgecolor="white", zorder=3)
        lower = lower + y

    for j, x in enumerate(stage_x):
        ax.plot([x, x], [-99, 99], color="#c9c2b6", lw=0.7, ls=(0, (3, 3)),
                zorder=1)
        v = VOLUMES[j]
        lab = f"{v / 1000:.1f} TB" if v >= 1000 else f"{v:.0f} GB"
        ax.text(x, 116, lab, ha="center", va="bottom", fontsize=7.6,
                fontweight="bold", color=INK, zorder=6)
        ax.text(x, 104, STAGES[j][0], ha="center", va="bottom", fontsize=6.3,
                color=MUTED, zorder=6)
    for j in range(4):
        xm = (stage_x[j] + stage_x[j + 1]) / 2
        r = VOLUMES[j + 1] / VOLUMES[j]
        ax.text(xm, -108, f"x {r:.2f}", ha="center", va="top", fontsize=6.6,
                color=RUST, fontweight="bold", zorder=6)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.86)
               for c in cols]
    ax.legend(handles, [s[0] for s in STREAMS], fontsize=6.6, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, -0.06), handlelength=1.0,
              handleheight=0.85, columnspacing=1.1, borderaxespad=0.0)
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-148, 140)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Data volume along the pipeline  "
                 "(band width $\\propto\\sqrt{\\mathrm{volume}}$)",
                 loc="left", fontsize=8.8, fontweight="bold", color=INK,
                 pad=6)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], left=0.045,
                          right=0.985, top=0.868, bottom=0.055, hspace=0.30)
    ax_p = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])
    pipeline(fig, ax_p)
    sankey(ax_s)

    add_panel_label(ax_p, "(a)", x=-0.024, y=1.00)
    add_panel_label(ax_s, "(b)", x=-0.038, y=1.025)

    savefig(fig, "91_method_pipeline.png")


if __name__ == "__main__":
    main()
