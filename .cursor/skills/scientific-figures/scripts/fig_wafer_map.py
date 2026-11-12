"""Wafer bin map with edge-ring and centre-cluster signatures, plus a reduced
statistics column: radial yield profile and failure-mode share."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
AMBER = "#c9922f"
ALERT = "#b23a48"
PASS = "#e3eaf1"
GREY = "#8b949e"
BLOCK_BG = "#f6f8fa"

GRID = 33
R_DIE = 16.1
BINS = [("Pass", PASS), ("Edge ring", ALERT), ("Centre cluster", AMBER),
        ("Scratch", BLUE), ("Parametric", NAVY)]


def build_wafer():
    rng = np.random.default_rng(24680)
    c = (GRID - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    dx, dy = jj - c, ii - c
    r = np.hypot(dx, dy) / R_DIE
    inside = np.hypot(dx, dy) <= R_DIE - 0.35

    b = np.zeros((GRID, GRID), dtype=int)
    u = rng.random((GRID, GRID))

    p_edge = np.clip((r - 0.80) / 0.20, 0, 1) ** 1.4 * 0.92
    b[(u < p_edge) & inside] = 1

    p_ctr = np.clip((0.20 - r) / 0.20, 0, 1) ** 0.8 * 0.85
    b[(u < p_ctr) & inside & (b == 0)] = 2

    # a straight handling scratch across the upper-left quadrant
    d_line = np.abs(0.80 * dx + dy + 2.5) / np.hypot(0.80, 1.0)
    scratch = (d_line < 0.85) & (dx < 4.0) & (dy > -6.0)
    b[scratch & inside & (b == 0) & (u < 0.80)] = 3

    b[(u > 0.982) & inside & (b == 0)] = 4
    return b, inside, r


def wafer_panel(ax, b, inside):
    c = (GRID - 1) / 2.0
    rects, cols = [], []
    for i in range(GRID):
        for j in range(GRID):
            if inside[i, j]:
                rects.append(Rectangle((j - c - 0.5, i - c - 0.5), 1, 1))
                cols.append(BINS[b[i, j]][1])
    ax.add_collection(PatchCollection(rects, facecolors=cols,
                                      edgecolors="white", linewidths=0.35,
                                      zorder=3))
    ax.add_patch(Circle((0, 0), R_DIE + 0.15, fill=False, edgecolor="#6c757d",
                        lw=1.2, zorder=4))
    ax.add_patch(Circle((0, -(R_DIE + 0.15)), 0.95, facecolor="white",
                        edgecolor="#6c757d", lw=1.0, zorder=5))
    ax.text(0, -(R_DIE + 2.6), "notch", ha="center", va="top", fontsize=6.0,
            color=GREY)
    lim = R_DIE + 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 1.2, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def radial_panel(ax, b, inside, r, yield_all):
    edges = np.linspace(0, 1.0, 9)
    mid, ys = [], []
    for k in range(8):
        m = inside & (r >= edges[k]) & (r < edges[k + 1])
        if m.sum():
            mid.append(0.5 * (edges[k] + edges[k + 1]))
            ys.append(100.0 * (b[m] == 0).sum() / m.sum())
    ax.axvspan(0.88, 1.0, color="#fbf1f2", zorder=0)
    ax.plot(mid, ys, "-", color=NAVY, lw=1.5, zorder=4)
    ax.plot(mid, ys, "o", ms=3.6, mfc="white", mec=NAVY, mew=1.1, zorder=5)
    ax.axhline(yield_all, color=GREY, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(0.86, yield_all - 3.0, f"wafer mean {yield_all:.1f}%", ha="right",
            va="top", fontsize=6.1, color=GREY)
    ax.text(0.94, 8, "edge\nexclusion", ha="center", va="bottom",
            fontsize=6.0, color=ALERT, linespacing=1.4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Normalised radius  $r / R$", fontsize=7.5)
    ax.set_ylabel("Zone yield (%)", fontsize=7.5)
    ax.set_title("Radial yield profile", fontsize=7.4, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e8ecf0", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def mode_panel(ax, b, inside, total):
    counts = [(name, col, int(((b == k) & inside).sum()))
              for k, (name, col) in enumerate(BINS) if k > 0]
    counts.sort(key=lambda t: t[2])
    y = np.arange(len(counts))
    ax.barh(y, [c[2] for c in counts], height=0.62,
            color=[c[1] for c in counts], edgecolor="white", linewidth=0.6,
            zorder=3)
    top = max(c[2] for c in counts)
    for yi, (name, col, n) in zip(y, counts):
        ax.text(n + top * 0.035, yi, f"{n}   {100 * n / total:.1f}%",
                ha="left", va="center", fontsize=6.3, fontweight="bold",
                color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in counts], fontsize=6.6)
    ax.set_xlim(0, top * 1.42)
    ax.set_ylim(-0.62, len(counts) - 0.38)
    ax.set_xlabel("Failing die (count)", fontsize=7.5)
    ax.set_title("Failure-mode share", fontsize=7.4, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e8ecf0", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    b, inside, r = build_wafer()
    total = int(inside.sum())
    good = int(((b == 0) & inside).sum())
    yield_all = 100.0 * good / total

    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.60], wspace=0.30,
                          left=0.015, right=0.975, top=0.800, bottom=0.118)
    ax_w = fig.add_subplot(gs[0, 0])
    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.62)
    ax_r = fig.add_subplot(sub[0])
    ax_m = fig.add_subplot(sub[1])

    # grey block groups the two subordinate statistics panels
    p_r, p_m = ax_r.get_position(), ax_m.get_position()
    fig.patches.append(FancyBboxPatch(
        (p_r.x0 - 0.115, p_m.y0 - 0.073), (p_r.x1 - p_r.x0) + 0.140,
        (p_r.y1 - p_m.y0) + 0.138, boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure, facecolor=BLOCK_BG, edgecolor="#e6eaee",
        linewidth=0.8, zorder=-1))
    for ax in (ax_r, ax_m):
        ax.set_facecolor("none")

    wafer_panel(ax_w, b, inside)
    radial_panel(ax_r, b, inside, r, yield_all)
    mode_panel(ax_m, b, inside, total)

    for ax, lab, dx in ((ax_w, "(a)", 0.010), (ax_r, "(b)", 0.100),
                        (ax_m, "(c)", 0.100)):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.016, lab, fontsize=9,
                 fontweight="bold", ha="left", va="bottom")

    handles = [Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="white",
                         linewidth=0.6, label=name) for name, col in BINS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.885),
               ncol=5, frameon=False, fontsize=6.8, handlelength=1.1,
               handleheight=1.0, columnspacing=1.3, borderaxespad=0.0)


    savefig(fig, "251_wafer_map.png")


if __name__ == "__main__":
    main()
