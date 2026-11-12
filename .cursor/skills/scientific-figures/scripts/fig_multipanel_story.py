"""Narrative 2x3 multipanel: six heterogeneous panel types (trend, map,
grouped boxes, correlation, mechanism schematic, radar) that read as one
story because they share a single palette, a left-aligned title block and a
one-line takeaway under every heading.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#84a07c"
SAND = "#cf9f43"
RUST = "#b0563d"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f6f1e4", "#dfe4d1", "#aec8b8", "#6ea099", "#33707a", DEEP])
SEQ_D = LinearSegmentedColormap.from_list(
    "seqd", ["#cbd7c8", "#9dbdb0", "#5b9691", "#2e7d7b", DEEP])


def head(ax, title, takeaway):
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold",
                 color=INK, pad=15)
    ax.text(0.0, 1.018, takeaway, transform=ax.transAxes, fontsize=6.7,
            color=MUTED, ha="left", va="bottom")


def p_trend(ax, rng):
    t = np.arange(1990, 2025)
    for lab, c, a in [("Reference", SAGE, 0.55), ("Managed", TEAL, 0.85),
                      ("Degraded", RUST, 1.0)]:
        y = 100 + a * (-0.85 * (t - 1990) + 6 * np.sin(0.35 * (t - 1990))) \
            + rng.normal(0, 1.6, t.size)
        sd = 2.0 + 0.09 * (t - 1990)
        ax.fill_between(t, y - sd, y + sd, color=c, alpha=0.15, lw=0)
        ax.plot(t, y, color=c, lw=1.6, label=lab)
    ax.axvspan(2008, 2013, color=SAND, alpha=0.16, lw=0)
    ax.text(2010.5, 108, "drought", ha="center", fontsize=6.4, color="#8f6a1c")
    ax.set_xlim(1990, 2024)
    ax.set_ylim(55, 118)
    ax.set_xlabel("Year")
    ax.set_ylabel("Productivity index")
    ax.legend(fontsize=6.8, loc="lower left", handlelength=1.2,
              labelspacing=0.25, borderaxespad=0.25)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def p_map(fig, ax, rng):
    n = 160
    x = np.linspace(0, 9, n)
    y = np.linspace(0, 6, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for _ in range(9):
        cx, cy, s = rng.uniform(0, 9), rng.uniform(0, 6), rng.uniform(0.8, 2.2)
        Z += rng.uniform(0.3, 1.0) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2)
                                            / (2 * s ** 2))
    Z = (Z - Z.min()) / np.ptp(Z)
    im = ax.imshow(Z, origin="lower", extent=[0, 9, 0, 6], cmap=SEQ,
                   aspect="auto", interpolation="bilinear")
    cs = ax.contour(X, Y, Z, levels=[0.35, 0.65], colors="white",
                    linewidths=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=6.0, inline=True)
    px, py = rng.uniform(0.4, 8.6, 22), rng.uniform(0.4, 5.6, 22)
    ax.scatter(px, py, s=9, marker="^", color=SAND, edgecolor="white",
               lw=0.5, zorder=4)
    cax = ax.inset_axes([1.025, 0.06, 0.030, 0.88])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Loss rate", fontsize=6.5, labelpad=3)
    ax.set_xlabel("Longitude offset (deg)")
    ax.set_ylabel("Latitude offset (deg)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)


def p_box(ax, rng):
    groups = ["Low", "Medium", "High"]
    conds = [("Control", "#b9c6cc"), ("Treated", TEAL)]
    pos, data, cols = [], [], []
    for i, g in enumerate(groups):
        for j, (cname, c) in enumerate(conds):
            base = 2.4 + 0.9 * i + (1.1 if j else 0)
            data.append(rng.normal(base, 0.45 + 0.08 * i, 40))
            pos.append(i * 1.0 + (j - 0.5) * 0.34)
            cols.append(c)
    bp = ax.boxplot(data, positions=pos, widths=0.28, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=INK, lw=1.0),
                    whiskerprops=dict(color="#6b7784", lw=0.8),
                    capprops=dict(color="#6b7784", lw=0.8),
                    boxprops=dict(lw=0.7, edgecolor="#4c5a63"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for d, p, c in zip(data, pos, cols):
        ax.scatter(p + rng.normal(0, 0.055, d.size), d, s=2.5,
                   color=INK, alpha=0.28, linewidths=0, zorder=3)
    for i in range(3):
        ax.plot([i - 0.17, i + 0.17], [6.6, 6.6], color=INK, lw=0.7)
        ax.text(i, 6.72, ["n.s.", "**", "***"][i], ha="center", fontsize=6.5,
                color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups)
    ax.set_xlabel("Baseline stress class")
    ax.set_ylabel("Recovery rate (yr$^{-1}$)")
    ax.set_ylim(0.8, 7.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.85,
                             edgecolor="#4c5a63", lw=0.7) for _, c in conds]
    ax.legend(handles, [c[0] for c in conds], fontsize=6.8, loc="upper left",
              ncol=2, handlelength=1.0, handleheight=0.85, columnspacing=0.8,
              borderaxespad=0.25)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def p_scatter(fig, ax, rng):
    n = 180
    x = rng.uniform(0.1, 1.0, n)
    z = rng.uniform(0, 1, n)
    y = 2.1 - 1.55 * x + 0.55 * z + rng.normal(0, 0.16, n)
    sc = ax.scatter(x, y, c=z, cmap=SEQ_D, s=16, edgecolor="white", lw=0.3,
                    alpha=0.9, vmin=0, vmax=1)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(0.05, 1.05, 60)
    ys = a + b * xs
    se = 0.09 * np.sqrt(1 + ((xs - x.mean()) / x.std()) ** 2)
    ax.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, color=RUST,
                    alpha=0.14, lw=0)
    ax.plot(xs, ys, color=RUST, lw=1.6)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.965, 0.94, f"$r$ = {r:.2f}\n$p$ < 0.001\n$n$ = {n}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=GRID, lw=0.6))
    cax = ax.inset_axes([1.025, 0.06, 0.030, 0.88])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Soil depth (norm.)", fontsize=6.5, labelpad=3)
    ax.set_xlabel("Canopy openness")
    ax.set_ylabel("Recovery rate (yr$^{-1}$)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def p_mechanism(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    nodes = [("Canopy\nopening", 2.0, 5.0, SAGE),
             ("Soil\nwarming", 8.0, 5.0, SAND),
             ("Moisture\nloss", 8.0, 1.5, RUST),
             ("Slower\nrecovery", 2.0, 1.5, DEEP)]
    for name, x, y, c in nodes:
        ax.add_patch(FancyBboxPatch((x - 1.35, y - 0.72), 2.7, 1.44,
                                    boxstyle="round,pad=0,rounding_size=0.28",
                                    facecolor=c, edgecolor="white", lw=1.2,
                                    alpha=0.92, zorder=3))
        ax.text(x, y, name, ha="center", va="center", color="white",
                fontsize=7.0, fontweight="bold", linespacing=1.35, zorder=4)
    arcs = [((3.45, 5.0), (6.6, 5.0), 0.0, "+", 5.0, 5.62),
            ((8.0, 4.22), (8.0, 2.28), 0.0, "+", 8.62, 3.25),
            ((6.6, 1.5), (3.45, 1.5), 0.0, "+", 5.0, 0.90),
            ((2.0, 2.28), (2.0, 4.22), 0.30, "-", 0.72, 3.25)]
    for (p0, p1, rad, sign, tx, ty) in arcs:
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=12, lw=1.3, color="#7d8892", zorder=2))
        ax.add_patch(Circle((tx, ty), 0.30, facecolor="white",
                            edgecolor="#7d8892", lw=0.8, zorder=5))
        ax.text(tx, ty + 0.02, sign, ha="center", va="center", fontsize=9,
                fontweight="bold", color=INK, zorder=6)
    ax.text(5.0, 3.25, "reinforcing\nfeedback", ha="center", va="center",
            fontsize=7, color=MUTED, style="italic", linespacing=1.4)


def p_radar(ax, rng):
    labs = ["Accuracy", "Stability", "Coverage", "Cost", "Transfer",
            "Interpret."]
    k = len(labs)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ang = np.append(ang, ang[0])
    series = [("Reference", SAGE, [0.55, 0.62, 0.48, 0.80, 0.44, 0.86]),
              ("Managed", TEAL, [0.78, 0.74, 0.71, 0.62, 0.68, 0.72]),
              ("Degraded", RUST, [0.42, 0.35, 0.66, 0.55, 0.38, 0.60])]
    for name, c, v in series:
        vv = np.array(v + [v[0]])
        ax.plot(ang, vv, color=c, lw=1.5, label=name)
        ax.fill(ang, vv, color=c, alpha=0.14)
        ax.scatter(ang[:-1], vv[:-1], color=c, s=11, zorder=5,
                   edgecolor="white", lw=0.5)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labs, fontsize=6.8)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.0,
                       color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_rlabel_position(96)
    ax.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, handlelength=1.1, columnspacing=1.0, borderaxespad=0.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(23)
    fig = plt.figure(figsize=(12.6, 7.8))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.955, top=0.845,
                          bottom=0.085, hspace=0.60, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2], projection="polar")

    p_trend(ax_a, rng)
    p_map(fig, ax_b, rng)
    p_box(ax_c, rng)
    p_scatter(fig, ax_d, rng)
    p_mechanism(ax_e)
    p_radar(ax_f, rng)

    head(ax_a, "Productivity has diverged since 2008",
         "Three management regimes, 35 years of monitoring")
    head(ax_b, "Loss concentrates in the north-west",
         "Gridded loss rate with 22 validation plots")
    head(ax_c, "Treatment only helps where stress is high",
         "40 plots per group; Wilcoxon signed-rank")
    head(ax_d, "Openness explains most of the variance",
         "Colour encodes soil depth; band is the 95% CI")
    head(ax_e, "A reinforcing moisture feedback",
         "Signs give the direction of each causal link")

    cell = ax_f.get_position()
    ax_f.set_position([cell.x0 + 0.016, cell.y0 - 0.008,
                       cell.width * 0.90, cell.height * 0.84])
    fig.text(cell.x0, cell.y1 + 0.020, "Managed sites lead on five of six "
             "axes", fontsize=8.8, fontweight="bold", color=INK, ha="left",
             va="bottom")
    fig.text(cell.x0, cell.y1 + 0.002, "Scores rescaled to the best "
             "performer", fontsize=6.7, color=MUTED, ha="left", va="bottom")
    fig.text(cell.x0 - 0.036, cell.y1 + 0.046, "(f)", fontsize=11,
             fontweight="bold", color="#000000", ha="left", va="bottom")

    for ax, lab, x, y in [(ax_a, "(a)", -0.185, 1.085),
                          (ax_b, "(b)", -0.185, 1.085),
                          (ax_c, "(c)", -0.185, 1.085),
                          (ax_d, "(d)", -0.185, 1.085),
                          (ax_e, "(e)", -0.140, 1.085)]:
        add_panel_label(ax, lab, x=x, y=y)

    savefig(fig, "90_multipanel_story.png")


if __name__ == "__main__":
    main()
