"""The rasterisation pipeline as five stages, with the data volume they carry.

The diagram is drawn on an inch canvas so the five cards, their gutters and
their corner radii are physically identical, and the three raster stages reuse
one triangle on one pixel grid: coverage, shading and the depth test are the
same nine-by-five lattice seen three times, which is what makes the growth from
vertices to fragments legible.  Every count printed between the cards comes
from one arithmetic chain seeded by the vertex count, and the depth-test
survival rate is measured on the little grid rather than asserted.

The two auxiliary panels are evidence: where the frame time actually goes for
three scene presets against the 60 fps budget, and how many invocations each
stage runs concurrently, on a log axis because the stages differ by four
orders of magnitude.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

from pubstyle import set_pub_style, savefig, get_palette
from figkit import inch_axes, axes_inches, arrow, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
HAIR = "#cccccc"
SOFT = "#f5f5f5"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

STAGE_COLORS = ["#a9cbe0", "#6fa6c9", "#3d80ac", "#255d85", "#134a76"]
HEAD_TEXT = [INK, "white", "white", "white", "white"]
STAGES = [
    ("Vertex shading", "MVP transform, per-vertex attributes"),
    ("Primitive assembly", "Index fetch, clip, backface cull"),
    ("Rasterisation", "Coverage test, attribute interpolation"),
    ("Fragment shading", "Texture fetch, lighting, alpha"),
    ("Output merging", "Depth test, blend, write to target"),
]

# Scene model.  Only the first three numbers are chosen; the rest follow.
N_VERTICES = 12_600
VERTS_PER_TRI = 3.0
AVG_TRI_AREA_PX = 35.2

# Pixel lattice shared by the raster / shade / merge cards.
NX, NY = 9, 5
TRI = np.array([[0.6, 0.4], [8.3, 1.6], [3.2, 4.6]])
OCCLUDER = np.array([[5.0, 0.25], [8.6, 2.9], [4.3, 4.2]])

PRESETS = ["Low detail", "Balanced", "Cinematic"]
STAGE_MS = np.array([                       # rows: preset, cols: stage
    [0.9, 0.6, 1.7, 4.2, 1.1],
    [1.6, 1.0, 2.9, 8.4, 1.8],
    [2.8, 1.7, 4.6, 16.1, 2.9],
])
BUDGET_FPS = 60.0
CONCURRENCY = np.array([12_600.0, 4_200.0, 64.0, 147_800.0, 96_000.0])


def barycentric(p, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    l1 = ((y2 - y3) * (p[..., 0] - x3) + (x3 - x2) * (p[..., 1] - y3)) / det
    l2 = ((y3 - y1) * (p[..., 0] - x3) + (x1 - x3) * (p[..., 1] - y3)) / det
    return np.stack([l1, l2, 1.0 - l1 - l2], axis=-1)


def cell_centres():
    gx, gy = np.meshgrid(np.arange(NX) + 0.5, np.arange(NY) + 0.5,
                         indexing="xy")
    return np.stack([gx, gy], axis=-1)


def coverage(tri):
    b = barycentric(cell_centres(), tri)
    return np.all(b >= 0.0, axis=-1), b


def rounded(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.05, z=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha,
        mutation_aspect=1.0))


def draw_cube(ax, cx, cy, s, color):
    """Projected wireframe box with its eight vertices marked."""
    front = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * s
    off = np.array([0.42 * s, 0.34 * s])
    back = front + off
    for quad in (front, back):
        ax.add_patch(Polygon(quad + [cx, cy], closed=True, facecolor="none",
                             edgecolor=color, lw=0.9, zorder=5))
    for a, b in zip(front, back):
        ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=color,
                lw=0.9, zorder=5)
    pts = np.vstack([front, back]) + [cx, cy]
    ax.scatter(pts[:, 0], pts[:, 1], s=8.0, color=color, zorder=6,
               edgecolors="white", linewidths=0.4)


def draw_fan(ax, cx, cy, w, h, color):
    """Triangle fan: the same vertices, now welded into primitives."""
    hub = np.array([cx - 0.30 * w, cy - 0.26 * h])
    ang = np.linspace(-0.10, 1.02, 5) * np.pi * 0.62
    rim = np.stack([hub[0] + 0.94 * w * np.cos(ang),
                    hub[1] + 0.94 * h * np.sin(ang)], axis=1)
    for k in range(len(rim) - 1):
        ax.add_patch(Polygon(np.vstack([hub, rim[k], rim[k + 1]]), closed=True,
                             facecolor=color, alpha=0.18, edgecolor=color,
                             lw=0.85, zorder=5))
    pts = np.vstack([hub, rim])
    ax.scatter(pts[:, 0], pts[:, 1], s=8.0, color=color, zorder=6,
               edgecolors="white", linewidths=0.4)


def draw_lattice(ax, x0, y0, cell, mode, color, ramp):
    """Nine-by-five pixel grid: coverage, shaded, or depth-tested."""
    cov, bar = coverage(TRI)
    occ, _ = coverage(OCCLUDER)
    shade = np.clip(0.24 + 0.70 * bar[..., 1] + 0.24 * bar[..., 2], 0.0, 1.0)
    for j in range(NY):
        for i in range(NX):
            fc, alpha = "white", 1.0
            if cov[j, i]:
                if mode == "coverage":
                    fc, alpha = color, 0.55
                elif mode == "shaded":
                    fc = ramp(shade[j, i])
                else:
                    if occ[j, i]:
                        fc, alpha = "#e2e2e2", 1.0
                    else:
                        fc = ramp(shade[j, i])
            ax.add_patch(Rectangle((x0 + i * cell, y0 + j * cell), cell, cell,
                                   facecolor=fc, alpha=alpha, edgecolor=HAIR,
                                   linewidth=0.35, zorder=4))
    if mode in ("coverage", "merged"):
        poly = TRI * cell + [x0, y0]
        ax.add_patch(Polygon(poly, closed=True, facecolor="none",
                             edgecolor=RULE if mode == "coverage" else GUIDE,
                             lw=0.9, zorder=6))
    if mode == "merged":
        poly = OCCLUDER * cell + [x0, y0]
        ax.add_patch(Polygon(poly, closed=True, facecolor="none",
                             edgecolor=ACCENT, lw=0.9, ls=(0, (2.2, 1.6)),
                             zorder=6))
    return cov, occ


def si(n):
    """Compact count label: 12.6 k, 148 k, 1.2 M."""
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3g} k"
    return f"{n:.0f}"


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    ramp = get_palette("cb_sequential_blue", 256)
    ramp_fn = (lambda v: ramp[int(np.clip(v, 0.0, 1.0) * 255)])

    cov, _bar = coverage(TRI)
    occ, _obar = coverage(OCCLUDER)
    occ_hit = int(np.sum(cov & occ))
    n_cov = int(np.sum(cov))
    survive = 1.0 - occ_hit / max(n_cov, 1)

    n_tri = N_VERTICES / VERTS_PER_TRI
    n_frag = n_tri * AVG_TRI_AREA_PX
    n_written = n_frag * survive
    flow = [f"{si(N_VERTICES)} vertices in", f"{si(N_VERTICES)} transformed",
            f"{si(n_tri)} triangles", f"{si(n_frag)} fragments",
            f"{si(n_frag)} shaded", f"{si(n_written)} pixels out"]

    # ---- (a) pipeline diagram on an inch canvas ---------------------------
    dx0, dy0, dw, dh = 0.26, 3.02, 9.88, 2.40
    axd = inch_axes(fig, (dx0, dy0, dw, dh))
    card_w, gap = 1.66, 0.395
    card_y0, card_h = 0.34, 1.72
    head_h = 0.245
    cell = 0.135
    lat_w, lat_h = NX * cell, NY * cell

    for k, (name, note) in enumerate(STAGES):
        x = k * (card_w + gap)
        col = STAGE_COLORS[k]
        rounded(axd, x, card_y0, card_w, card_h, SOFT, ec=HAIR, lw=0.7, r=0.07,
                z=2)
        rounded(axd, x, card_y0 + card_h - head_h, card_w, head_h, col, r=0.07,
                z=3)
        axd.text(x + card_w / 2, card_y0 + card_h - head_h / 2 - 0.004, name,
                 ha="center", va="center", fontsize=7.4, fontweight="bold",
                 color=HEAD_TEXT[k], zorder=6)

        cx = x + card_w / 2
        if k == 0:
            draw_cube(axd, cx, card_y0 + 0.98, 0.26, col)
        elif k == 1:
            draw_fan(axd, cx, card_y0 + 0.98, 0.32, 0.30, col)
        else:
            mode = {2: "coverage", 3: "shaded", 4: "merged"}[k]
            draw_lattice(axd, cx - lat_w / 2, card_y0 + 0.98 - lat_h / 2,
                         cell, mode, col, ramp_fn)

        axd.text(cx, card_y0 + 0.20, note, ha="center", va="center",
                 fontsize=6.3, color=SUB, zorder=6)

        if k < len(STAGES) - 1:
            xa = x + card_w
            arrow(axd, (xa + 0.05, card_y0 + card_h / 2 - 0.10),
                  (xa + gap - 0.05, card_y0 + card_h / 2 - 0.10),
                  color=RULE, lw=1.1, ms=9, zorder=7)

    y_flow = card_y0 + card_h + 0.11
    stops = [card_w / 2] + [k * (card_w + gap) + card_w + gap / 2
                            for k in range(4)] + \
            [4 * (card_w + gap) + card_w / 2]
    for k, (lx, txt) in enumerate(zip(stops, flow)):
        last = k == len(stops) - 1
        axd.text(lx, y_flow, txt, ha="center", va="bottom", fontsize=6.4,
                 color=ACCENT if last else MUTED, zorder=7)
    axd.text(0.0, card_y0 - 0.20,
             f"grey cells fail the depth test against the dashed occluder: "
             f"{occ_hit} of {n_cov} covered pixels, "
             f"{survive * 100:.0f} % survive to the render target",
             ha="left", va="center", fontsize=6.4, color=SUB)

    # ---- (b) where the frame time goes ------------------------------------
    axb = axes_inches(fig, (0.80, 0.86, 4.20, 1.48))
    budget = 1000.0 / BUDGET_FPS
    ypos = np.arange(len(PRESETS))[::-1]
    left = np.zeros(len(PRESETS))
    for k, (name, _n) in enumerate(STAGES):
        axb.barh(ypos, STAGE_MS[:, k], left=left, height=0.56,
                 color=STAGE_COLORS[k], edgecolor="white", linewidth=0.6,
                 label=name, zorder=3)
        left = left + STAGE_MS[:, k]
    axb.axvline(budget, color=ACCENT, lw=1.0, ls=(0, (3.2, 2.2)), zorder=5)
    axb.text(budget + 0.35, -0.45,
             f"{BUDGET_FPS:.0f} fps budget {budget:.1f} ms", ha="left",
             va="center", fontsize=6.4, color=ACCENT)
    axb.text(0.0, -0.45, "segment colours as in (a)", ha="left",
             va="center", fontsize=6.4, color=MUTED)
    for y, tot in zip(ypos, left):
        over = tot > budget
        axb.text(tot + 0.35, y, f"{tot:.1f} ms  ({1000.0 / tot:.0f} fps)",
                 ha="left", va="center", fontsize=6.4,
                 color=ACCENT if over else SUB)
    axb.set_yticks(ypos)
    axb.set_yticklabels(PRESETS, fontsize=6.8)
    axb.set_xlim(0.0, left.max() * 1.42)
    axb.set_ylim(-0.62, len(PRESETS) - 0.38)
    axb.set_xlabel("frame time (ms)", fontsize=8.0)
    axb.set_title("Fragment shading owns the frame", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)
    axb.tick_params(labelsize=6.6)
    axb.grid(True, axis="x", color=GRID, lw=0.5)

    # ---- (c) concurrency per stage ----------------------------------------
    axc = axes_inches(fig, (6.28, 0.86, 3.52, 1.48))
    ypos2 = np.arange(len(STAGES))[::-1]
    axc.barh(ypos2, CONCURRENCY, height=0.58, color=STAGE_COLORS,
             edgecolor="white", linewidth=0.6, zorder=3)
    for y, v in zip(ypos2, CONCURRENCY):
        axc.text(v * 1.25, y, si(v), ha="left", va="center", fontsize=6.4,
                 color=SUB)
    axc.set_xscale("log")
    axc.set_xlim(10.0, CONCURRENCY.max() * 14.0)
    axc.set_ylim(-0.62, len(STAGES) - 0.38)
    axc.set_yticks(ypos2)
    axc.set_yticklabels([s[0] for s in STAGES], fontsize=6.6)
    axc.set_xlabel("concurrent work items (-)", fontsize=8.0)
    axc.set_title("Four decades of parallel width", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)
    axc.tick_params(labelsize=6.6)
    axc.grid(True, axis="x", color=GRID, lw=0.5)


    panel_labels_inches(fig, [axd, axb, axc], dx=-0.04, dy=0.22)
    savefig(fig, "649_cg_rendering_pipeline.png")


if __name__ == "__main__":
    main()
