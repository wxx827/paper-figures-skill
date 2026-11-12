"""Method-by-criterion comparison matrix in which every cell is a small
chart rather than a number: bars, kernel densities, calibration scatters,
bullet gauges, scaling curves and rating dots.

The grid uses ``wspace = hspace = 0`` with spacer rows/columns so that the
weight strip on top and the weighted-score bars on the right stay pixel
aligned with the matrix cells.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#7d9a72"
SAND = "#cf9f43"
RUST = "#b0563d"
GREY = "#8f9ea7"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"

METHODS = [("Ridge", "linear baseline", GREY),
           ("Random forest", "500 trees", SAND),
           ("Gradient boost", "tuned, 2 000 rounds", SAGE),
           ("CNN-1D", "8 conv blocks", TEAL),
           ("Ours", "hierarchical GP-boost", DEEP)]
DIMS = ["Accuracy", "Robustness", "Calibration", "Efficiency", "Scalability",
        "Interpretability"]
WEIGHTS = np.array([0.22, 0.18, 0.15, 0.15, 0.16, 0.14])
SCORES = np.array([
    [0.38, 0.44, 0.52, 0.88, 0.34, 0.92],
    [0.58, 0.63, 0.49, 0.61, 0.55, 0.66],
    [0.71, 0.68, 0.61, 0.57, 0.62, 0.54],
    [0.79, 0.58, 0.66, 0.35, 0.78, 0.31],
    [0.91, 0.84, 0.88, 0.72, 0.86, 0.69]])


def blend(c, t):
    return tuple(1 - (1 - v) * t for v in to_rgb(c))


def frame(ax, col, score, best):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(blend(col, 0.05 + 0.15 * score))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("white")
        s.set_linewidth(2.0)
    if best:
        ax.add_patch(Rectangle((0.012, 0.012), 0.976, 0.976,
                               transform=ax.transAxes, fill=False,
                               edgecolor=col, lw=1.1, zorder=9))


def c_bar(ax, v, col, rng):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = 0.155 + np.arange(5) * 0.173
    h = np.clip(v + rng.normal(0, 0.045, 5), 0.06, 0.99)
    ax.bar(xs, 0.78 * h, width=0.115, bottom=0.10, color=col,
           edgecolor="none")
    ax.plot([0.05, 0.95], [0.10 + 0.78 * 0.75] * 2, color=RUST, lw=0.9,
            ls="--")
    ax.plot([0.05, 0.95], [0.10, 0.10], color="#c9c2b6", lw=0.7)


def c_kde(ax, v, col, rng):
    x = np.linspace(0, 1, 120)
    s = 0.20 - 0.10 * v
    y = np.exp(-((x - (0.15 + 0.72 * v)) ** 2) / (2 * s ** 2))
    ax.fill_between(x, 0, y, color=col, alpha=0.55, lw=0)
    ax.plot(x, y, color=col, lw=1.1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.55)
    ax.plot([0, 1], [0, 0], color="#c9c2b6", lw=0.7)


def c_calib(ax, v, col, rng):
    n = 26
    x = rng.uniform(0.08, 0.92, n)
    y = x + rng.normal(0, 0.28 * (1 - v) + 0.02, n)
    ax.plot([0, 1], [0, 1], color="#b9b3a6", lw=0.8, ls="--")
    ax.scatter(x, y, s=4.5, color=col, alpha=0.85, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def c_bullet(ax, v, col, rng):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.08, 0.40), 0.84, 0.20,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#e6e2d8", edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.08, 0.40), max(0.84 * v, 0.05), 0.20,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=col, edgecolor="none"))
    ax.plot([0.08 + 0.84 * 0.75] * 2, [0.31, 0.69], color=RUST, lw=1.2)


def c_scale(ax, v, col, rng):
    x = np.linspace(0, 1, 40)
    y = 0.12 + 0.80 * x ** (2.4 - 1.9 * v)
    ax.plot(x, y, color=col, lw=1.5)
    ax.fill_between(x, 0.05, y, color=col, alpha=0.14, lw=0)
    ax.plot([0, 1], [0.12, 0.92], color="#c3bdb1", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def c_dots(ax, v, col, rng):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    k = int(round(v * 5))
    for i in range(5):
        cx = 0.14 + i * 0.18
        ax.add_patch(Circle((cx, 0.5), 0.062,
                            facecolor=col if i < k else "white",
                            edgecolor=col if i < k else "#c3bdb1", lw=0.9))


CELLS = [c_bar, c_kde, c_calib, c_bullet, c_scale, c_dots]


def main():
    set_pub_style()
    rng = np.random.default_rng(43)
    totals = SCORES @ WEIGHTS
    best_row = int(np.argmax(totals))
    best_in_col = SCORES.argmax(axis=0)

    fig = plt.figure(figsize=(13.4, 7.7))
    gs = fig.add_gridspec(
        7, 9, left=0.052, right=0.982, top=0.845, bottom=0.140,
        width_ratios=[1.30] + [1.0] * 6 + [0.34, 1.65],
        height_ratios=[0.66, 0.20] + [1.0] * 5, wspace=0.0, hspace=0.0)

    ax_w = fig.add_subplot(gs[0, 1:7])
    ax_w.set_xlim(-0.5, 5.5)
    ax_w.set_ylim(-0.62, 1.0)
    ax_w.bar(np.arange(6), WEIGHTS / WEIGHTS.max() * 0.80, width=0.46,
             color="#c3ccd2", edgecolor="white", lw=0.8)
    for j, w in enumerate(WEIGHTS):
        ax_w.text(j, WEIGHTS[j] / WEIGHTS.max() * 0.80 + 0.06, f"{w:.2f}",
                  ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax_w.text(j, -0.16, DIMS[j], ha="center", va="top", fontsize=8.2,
                  fontweight="bold", color=INK)
    ax_w.axhline(0, color="#c9c2b6", lw=0.7)
    ax_w.text(-0.62, 0.40, "criterion\nweight", ha="right", va="center",
              fontsize=6.6, color=MUTED, linespacing=1.4)
    ax_w.axis("off")

    ax_n = fig.add_subplot(gs[2:7, 0])
    ax_n.set_xlim(0, 1)
    ax_n.set_ylim(5, 0)
    ax_n.axis("off")
    for i, (name, sub, col) in enumerate(METHODS):
        y = i + 0.5
        ax_n.add_patch(Rectangle((0.90, y - 0.30), 0.045, 0.60,
                                 facecolor=col, edgecolor="none"))
        ax_n.text(0.855, y - 0.08, name, ha="right", va="center",
                  fontsize=9.2, fontweight="bold",
                  color=DEEP if i == best_row else INK)
        ax_n.text(0.855, y + 0.16, sub, ha="right", va="center", fontsize=6.5,
                  color=MUTED)

    for i, (name, sub, col) in enumerate(METHODS):
        for j in range(6):
            ax = fig.add_subplot(gs[2 + i, 1 + j])
            frame(ax, col, SCORES[i, j], best_in_col[j] == i)
            CELLS[j](ax, SCORES[i, j], col, rng)
            ax.text(0.045, 0.945, f"{SCORES[i, j]:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.2, color=INK, zorder=10,
                    fontweight="bold" if best_in_col[j] == i else "normal",
                    bbox=dict(boxstyle="square,pad=0.14", facecolor="white",
                              edgecolor="none", alpha=0.62))

    ax_s = fig.add_subplot(gs[2:7, 8])
    y = np.arange(5) + 0.5
    ax_s.barh(y, totals, height=0.42, color=[m[2] for m in METHODS],
              edgecolor="white", lw=0.8)
    for yi, t, (nm, _, c) in zip(y, totals, METHODS):
        ax_s.text(t + 0.016, yi, f"{t:.3f}", va="center", fontsize=7.6,
                  color=INK, fontweight="bold" if nm == "Ours" else "normal")
    gain = totals[best_row] - np.sort(totals)[-2]
    ax_s.annotate("", xy=(totals[best_row], 4.5 - 0.30),
                  xytext=(np.sort(totals)[-2], 4.5 - 0.30),
                  arrowprops=dict(arrowstyle="<->", color=RUST, lw=0.9))
    ax_s.text((totals[best_row] + np.sort(totals)[-2]) / 2, 4.5 - 0.42,
              f"+{gain:.3f}", ha="center", va="bottom", fontsize=6.6,
              color=RUST, fontweight="bold")
    ax_s.set_ylim(5, 0)
    ax_s.set_xlim(0, 1.0)
    ax_s.set_yticks([])
    ax_s.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_s.set_xlabel("Weighted total score")
    ax_s.grid(axis="x", color=GRID, lw=0.5)
    ax_s.set_axisbelow(True)
    ax_s.spines["left"].set_visible(False)
    ax_s.set_title("Overall", loc="left", fontsize=8.2, fontweight="bold",
                   color=INK, pad=4)
    style_axes(ax_s)

    add_panel_label(ax_w, "(a)", x=-0.205, y=0.62)
    add_panel_label(ax_s, "(b)", x=-0.080, y=1.045)

    savefig(fig, "93_comparison_matrix.png")


if __name__ == "__main__":
    main()
