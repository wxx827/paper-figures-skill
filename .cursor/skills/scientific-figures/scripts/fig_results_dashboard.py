"""Academic results dashboard: a KPI strip with sparklines, an annotated
diverging heatmap next to a directly-labelled multi-line series, and a
full-width stacked bar with a total overlay.

High information density is kept legible by a strict three-band grid, one
accent colour per KPI card and a single diverging ramp for the matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#7d9a72"
SAND = "#cf9f43"
RUST = "#b0563d"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"
CARD = "#f5f3ed"

DIV = LinearSegmentedColormap.from_list(
    "div", ["#8f3f2c", RUST, "#dda88f", "#f4efe6", "#93bab4", TEAL, "#1c5a58"])

KPIS = [
    ("Overall skill  $R^2$", "0.912", +6.4, "vs. previous release", DEEP),
    ("Median error  (mm)", "11.7", -18.2, "cross-validated", TEAL),
    ("Spatial coverage  (%)", "94.3", +2.1, "of the target domain", SAGE),
    ("Compute  (GPU h)", "38", -41.0, "for a full retraining", SAND),
]
ROWS = ["Ridge", "Random forest", "Gradient boost", "CNN-1D", "Ours"]
COLS = ["Alpine", "Boreal", "Temperate", "Semi-arid", "Coastal", "Wetland"]


def square_units(fig, ax, ymax=1.0):
    fig.canvas.draw()
    bb = ax.get_position()
    w, h = fig.get_size_inches()
    xmax = ymax * (bb.width * w) / (bb.height * h)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    return xmax


def kpi_card(fig, ax, rng, spec):
    label, value, delta, note, col = spec
    xm = square_units(fig, ax)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), xm - 0.02, 0.96,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=CARD, edgecolor="#e6e2d8", lw=0.8))
    ax.add_patch(Rectangle((0.055, 0.13), 0.055, 0.74, facecolor=col,
                           edgecolor="none"))
    ax.text(0.175, 0.870, label.upper(), fontsize=6.4, color=MUTED,
            va="center", ha="left")
    ax.text(0.175, 0.575, value, fontsize=21, fontweight="bold", color=INK,
            va="center", ha="left")

    up = delta > 0
    tri = [(0.19, 0.230), (0.31, 0.230), (0.25, 0.320)] if up else \
          [(0.19, 0.320), (0.31, 0.320), (0.25, 0.230)]
    ax.add_patch(Polygon(tri, closed=True, facecolor=TEAL, edgecolor="none"))
    ax.text(0.365, 0.275, f"{abs(delta):.1f}%", fontsize=7.8,
            fontweight="bold", color=TEAL, va="center", ha="left")
    ax.text(0.175, 0.105, note, fontsize=6.2, color=MUTED, va="center",
            ha="left")

    sp = ax.inset_axes([xm - 1.05, 0.24, 0.92, 0.54], transform=ax.transData)
    n = 34
    trend = np.linspace(0, 1, n) * (1 if up else -1)
    y = trend + 0.28 * np.sin(np.linspace(0, 7, n)) + rng.normal(0, 0.10, n)
    sp.plot(y, color=col, lw=1.3)
    sp.fill_between(np.arange(n), y.min() - 0.25, y, color=col, alpha=0.12,
                    lw=0)
    sp.scatter([n - 1], [y[-1]], s=13, color=col, zorder=4,
               edgecolor="white", lw=0.6)
    sp.set_xlim(-0.5, n + 2.5)
    sp.set_ylim(y.min() - 0.28, y.max() + 0.28)
    sp.set_xticks([])
    sp.set_yticks([])
    sp.patch.set_alpha(0)
    for s in sp.spines.values():
        s.set_visible(False)


def panel_heatmap(fig, ax, rng):
    base = np.array([[-0.42, -0.31, -0.18, -0.55, -0.28, -0.36],
                     [-0.10, 0.04, 0.12, -0.22, 0.02, -0.08],
                     [0.06, 0.15, 0.24, -0.05, 0.13, 0.09],
                     [0.18, 0.27, 0.33, 0.11, 0.22, 0.19],
                     [0.41, 0.52, 0.58, 0.34, 0.47, 0.44]])
    m = base + rng.normal(0, 0.018, base.shape)
    im = ax.imshow(m, cmap=DIV, vmin=-0.62, vmax=0.62, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            c = "white" if abs(m[i, j]) > 0.34 else INK
            ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.8, color=c,
                    fontweight="bold" if i == 4 else "normal")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, fontsize=7)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    ax.add_patch(Rectangle((-0.5, 3.5), len(COLS), 1, fill=False,
                           edgecolor=INK, lw=1.4, zorder=5))
    cax = ax.inset_axes([1.022, 0.0, 0.026, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Skill gain vs. climatology", fontsize=6.5, labelpad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Skill by method and biome", loc="left", fontsize=8.8,
                 fontweight="bold", color=INK, pad=6)


def panel_lines(ax, rng):
    t = np.arange(0, 121)
    series = [("Ours", DEEP, 0.92, 0.030), ("CNN-1D", TEAL, 0.80, 0.045),
              ("Grad. boost", SAGE, 0.71, 0.052), ("Ridge", SAND, 0.58, 0.061)]
    ax.axvspan(58, 74, color="#efe7d6", zorder=0)
    ax.text(66, 0.985, "held-out\nstorm period", ha="center", va="top",
            fontsize=6.3, color="#8f6a1c", linespacing=1.35, zorder=3)
    for name, c, lvl, noise in series:
        y = lvl - 0.16 * np.exp(-((t - 66) / 9.0) ** 2) \
            + 0.035 * np.sin(t / 8.0) + rng.normal(0, noise * 0.35, t.size)
        y = np.convolve(y, np.ones(5) / 5, mode="same")
        y[:2], y[-2:] = y[2], y[-3]
        ax.plot(t, y, color=c, lw=1.5, zorder=4)
        ax.text(122, y[-1], name, fontsize=6.9, color=c, va="center",
                ha="left", fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_ylim(0.30, 1.0)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Rolling skill $R^2$")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Skill decay with forecast lead time", loc="left",
                 fontsize=8.8, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def panel_stacked(ax, rng):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
              "Oct", "Nov", "Dec"]
    cats = [("Optical", DEEP), ("Radar", TEAL), ("Thermal", SAGE),
            ("In situ", SAND), ("Reanalysis", RUST)]
    x = np.arange(12)
    seas = 1 + 0.42 * np.sin((x - 2) / 12 * 2 * np.pi)
    data = []
    for k, (_, _) in enumerate(cats):
        data.append(np.round((28 - 4.5 * k) * seas
                             * rng.uniform(0.88, 1.12, 12), 1))
    bottom = np.zeros(12)
    for (name, c), d in zip(cats, data):
        ax.bar(x, d, bottom=bottom, width=0.66, color=c, edgecolor="white",
               linewidth=0.7, label=name)
        bottom += d
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=7.5)
    ax.set_ylabel("Records ingested (millions)")
    ax.set_ylim(0, bottom.max() * 1.30)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=5, loc="upper left", handlelength=1.0,
              handleheight=0.85, columnspacing=1.2, borderaxespad=0.3)

    ax2 = ax.twinx()
    q = 0.62 + 0.30 * (bottom - bottom.min()) / np.ptp(bottom) \
        + rng.normal(0, 0.012, 12)
    ax2.plot(x, q, color=INK, lw=1.4, ls="--", marker="o", ms=3.4,
             markerfacecolor="white", markeredgewidth=1.0, zorder=6)
    ax2.set_ylim(0.35, 1.16)
    ax2.set_ylabel("Usable fraction", color=INK)
    ax2.tick_params(axis="y", colors="#333333")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#333333")
    ax2.spines["right"].set_linewidth(0.8)
    ax2.text(11.35, q[-1] + 0.055, "usable fraction", fontsize=6.6,
             color=INK, ha="right", va="bottom")
    ax.set_title("Monthly ingest volume and usable fraction", loc="left",
                 fontsize=8.8, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(31)
    fig = plt.figure(figsize=(13.2, 7.9))
    L, R = 0.048, 0.955
    gs_k = fig.add_gridspec(1, 4, left=L, right=R, top=0.885, bottom=0.748,
                            wspace=0.16)
    gs_m = fig.add_gridspec(1, 2, left=L, right=R, top=0.655, bottom=0.375,
                            wspace=0.30)
    gs_b = fig.add_gridspec(1, 1, left=L, right=R, top=0.288, bottom=0.070)

    kpi_axes = [fig.add_subplot(gs_k[0, i]) for i in range(4)]
    for ax, spec in zip(kpi_axes, KPIS):
        kpi_card(fig, ax, rng, spec)

    ax_h = fig.add_subplot(gs_m[0, 0])
    ax_l = fig.add_subplot(gs_m[0, 1])
    ax_s = fig.add_subplot(gs_b[0, 0])
    panel_heatmap(fig, ax_h, rng)
    panel_lines(ax_l, rng)
    panel_stacked(ax_s, rng)

    add_panel_label(kpi_axes[0], "(a)", x=-0.030, y=1.04)
    add_panel_label(ax_h, "(b)", x=-0.115, y=1.055)
    add_panel_label(ax_l, "(c)", x=-0.082, y=1.055)
    add_panel_label(ax_s, "(d)", x=-0.036, y=1.045)

    savefig(fig, "92_results_dashboard.png")


if __name__ == "__main__":
    main()
