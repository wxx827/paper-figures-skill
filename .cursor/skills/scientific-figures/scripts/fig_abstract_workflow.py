"""Graphical-abstract style workflow: a five-stage pipeline strip with
embedded thumbnail visualisations, above a row of three key-result panels.

Layout: gridspec with an oversized top row that hosts a coordinate-normalised
"canvas" axis; rounded boxes + arrows are drawn as patches, and each box holds
a real mini-axes created with ``inset_axes(..., transform=ax.transData)``.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"
BAND = "#f6f4ef"
STEP = ["#2f6f7e", "#3f8a78", "#7f9c6e", "#c8992f", "#b0563d"]

TITLES = ["Data", "Processing", "Model", "Validation", "Output"]
CAPTIONS = ["4 sensor networks\n2015-2024",
            "QC, gap-fill,\ndetrend",
            "Hierarchical\nGP-boosting",
            "Nested CV\n(k = 10)",
            "Gridded maps\n+ uncertainty"]


def square_units(fig, ax):
    """Set xlim so that one data unit is physically square (ylim = 0..1)."""
    fig.canvas.draw()
    bb = ax.get_position()
    w, h = fig.get_size_inches()
    ratio = (bb.width * w) / (bb.height * h)
    ax.set_xlim(0, ratio)
    ax.set_ylim(0, 1)
    return ratio


def blank(ax, fc="white"):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(fc)


def mini_scatter(ax, rng, col):
    x = rng.normal(0, 1, 90)
    y = 0.7 * x + rng.normal(0, 0.6, 90)
    ax.scatter(x, y, s=5, color=col, alpha=0.65, linewidths=0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    blank(ax)


def mini_signal(ax, rng, col):
    t = np.linspace(0, 6, 200)
    raw = np.sin(t) + 0.35 * rng.normal(0, 1, 200)
    ax.plot(t, raw, color=GRID, lw=0.8)
    ax.plot(t, np.sin(t), color=col, lw=1.4)
    ax.set_ylim(-2.2, 2.2)
    blank(ax)


def mini_network(ax, col):
    layers = [3, 4, 2]
    xs = np.linspace(0.15, 0.85, len(layers))
    pos = []
    for x, n in zip(xs, layers):
        ys = np.linspace(0.22, 0.78, n)
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.5, zorder=1)
    for lay in pos:
        for p in lay:
            ax.add_patch(Circle(p, 0.05, facecolor=col, edgecolor="white",
                                lw=0.6, zorder=3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    blank(ax)


def mini_onetoone(ax, rng, col):
    x = rng.uniform(0, 1, 70)
    y = x + rng.normal(0, 0.09, 70)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.scatter(x, y, s=5, color=col, alpha=0.7, linewidths=0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    blank(ax)


def mini_forecast(ax, rng, col):
    t = np.linspace(0, 1, 120)
    y = 0.4 + 0.45 * t + 0.06 * np.sin(9 * t)
    sd = 0.03 + 0.10 * t
    ax.fill_between(t, y - sd, y + sd, color=col, alpha=0.22, linewidth=0)
    ax.plot(t, y, color=col, lw=1.4)
    ax.set_ylim(0, 1.05)
    blank(ax)


def draw_flow(fig, ax, rng):
    ratio = square_units(fig, ax)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (-0.012 * ratio, -0.01), 1.024 * ratio, 1.03,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor=BAND,
        edgecolor="none", zorder=0, clip_on=False))

    bw, gap = 0.178, 0.0275
    y0, bh = 0.085, 0.80
    minis = [mini_scatter, mini_signal, None, mini_onetoone, mini_forecast]
    for i in range(5):
        x0 = ratio * i * (bw + gap)
        w = ratio * bw
        col = STEP[i]
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, bh, boxstyle="round,pad=0,rounding_size=0.055",
            facecolor="white", edgecolor=col, linewidth=1.2, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x0, y0 + bh - 0.155), w, 0.155,
            boxstyle="round,pad=0,rounding_size=0.055",
            facecolor=col, edgecolor=col, linewidth=1.2, zorder=3))
        ax.add_patch(plt.Rectangle((x0, y0 + bh - 0.155), w, 0.06,
                                   facecolor=col, edgecolor="none", zorder=3))
        ax.text(x0 + w / 2, y0 + bh - 0.078, f"{i + 1}.  {TITLES[i]}",
                ha="center", va="center", color="white", fontsize=9,
                fontweight="bold", zorder=4)

        mx, mw = x0 + 0.10 * w, 0.80 * w
        mh, my = 0.335, y0 + 0.235
        sub = ax.inset_axes([mx, my, mw, mh], transform=ax.transData)
        if minis[i] is None:
            mini_network(sub, col)
        elif i in (0, 1, 3):
            minis[i](sub, rng, col)
        else:
            minis[i](sub, rng, col)
        for s in sub.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
            s.set_linewidth(0.6)

        ax.text(x0 + w / 2, y0 + 0.115, CAPTIONS[i], ha="center", va="center",
                fontsize=7, color=MUTED, linespacing=1.35, zorder=4)

        if i < 4:
            xa = x0 + w
            ax.add_patch(FancyArrowPatch(
                (xa + 0.006 * ratio, y0 + bh / 2),
                (xa + (gap - 0.006) * ratio, y0 + bh / 2),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
                color="#8d97a2", zorder=5))


def panel_bars(ax, rng):
    sites = ["Alpine", "Boreal", "Temperate", "Semi-arid"]
    x = np.arange(4)
    base = np.array([0.52, 0.61, 0.68, 0.44])
    ours = base + rng.uniform(0.11, 0.19, 4)
    ax.bar(x - 0.19, base, 0.36, yerr=rng.uniform(.02, .04, 4), color="#c3ccd3",
           edgecolor="#8d97a2", linewidth=0.6, error_kw=dict(lw=0.8, capsize=2,
           ecolor=MUTED), label="Baseline")
    ax.bar(x + 0.19, ours, 0.36, yerr=rng.uniform(.02, .04, 4), color=STEP[1],
           edgecolor="#256050", linewidth=0.6, error_kw=dict(lw=0.8, capsize=2,
           ecolor=MUTED), label="This study")
    for xi, (b, o) in enumerate(zip(base, ours)):
        ax.text(xi + 0.19, o + 0.055, f"+{(o - b) * 100:.0f}%", ha="center",
                fontsize=6.5, color=STEP[1], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sites, fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Skill score $R^2$")
    ax.legend(fontsize=7, loc="upper left", ncol=2, handlelength=1.1,
              columnspacing=0.9, borderaxespad=0.2)
    ax.set_title("Cross-biome accuracy gain", fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def panel_trend(ax, rng):
    yrs = np.arange(2015, 2025)
    for i, (lab, col, drift) in enumerate([("Observed", STEP[0], 0.0),
                                           ("Reconstructed", STEP[3], 0.02),
                                           ("Projected", STEP[4], 0.05)]):
        y = 1.0 + drift * (yrs - 2015) + 0.06 * np.sin(0.9 * (yrs - 2015)) \
            + rng.normal(0, 0.02, yrs.size)
        sd = 0.03 + 0.008 * (yrs - 2015)
        ax.fill_between(yrs, y - sd, y + sd, color=col, alpha=0.16, linewidth=0)
        ax.plot(yrs, y, color=col, lw=1.5, marker="o", ms=3, label=lab)
    ax.set_xlabel("Year")
    ax.set_ylabel("Flux anomaly (norm.)")
    ax.legend(fontsize=7, loc="upper left", handlelength=1.3,
              borderaxespad=0.2)
    ax.set_title("Decadal trajectory", fontsize=8.5, fontweight="bold")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def panel_importance(ax, rng):
    names = ["Soil moisture", "VPD", "Radiation", "Leaf area", "Wind speed",
             "Air temp."]
    vals = np.array([0.31, 0.24, 0.16, 0.12, 0.09, 0.08])
    cols = [STEP[0], STEP[1], STEP[2], STEP[3], STEP[4], "#c3ccd3"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", linewidth=0.7)
    ax.errorbar(vals, y, xerr=rng.uniform(0.012, 0.03, len(vals)), fmt="none",
                ecolor=MUTED, elinewidth=0.8, capsize=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.035, yi, f"{v:.2f}", va="center", fontsize=7,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("Permutation importance")
    ax.set_title("Dominant controls", fontsize=8.5, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(12.2, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.06, 1.0],
                          left=0.055, right=0.985, top=0.855, bottom=0.088,
                          hspace=0.30, wspace=0.30)
    ax_flow = fig.add_subplot(gs[0, :])
    draw_flow(fig, ax_flow, rng)

    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    panel_bars(axes[0], rng)
    panel_trend(axes[1], rng)
    panel_importance(axes[2], rng)

    add_panel_label(ax_flow, "(a)", x=-0.028, y=1.00)
    for ax, lab in zip(axes, ["(b)", "(c)", "(d)"]):
        add_panel_label(ax, lab, x=-0.20, y=1.04)

    savefig(fig, "87_abstract_workflow.png")


if __name__ == "__main__":
    main()
