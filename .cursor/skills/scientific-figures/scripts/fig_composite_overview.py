"""Study-overview composite: one large spatial main panel spanning two rows,
two compact time-series panels, a ranked bar chart, a donut, and a full-width
project timeline ribbon underneath.

Demonstrates an irregular ``add_gridspec`` layout (row/column spans + mixed
height ratios) held together by a single five-colour palette.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Circle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#84a07c"
SAND = "#cf9f43"
RUST = "#b0563d"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"

TERR = LinearSegmentedColormap.from_list(
    "terr", ["#f4efe4", "#dfe3d2", "#b3c9bb", "#7aa79f", "#3f7c80", DEEP])
STN = LinearSegmentedColormap.from_list(
    "stn", ["#f2ddb6", SAND, RUST, "#7d3527"])


def field(rng, n=220):
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 8, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for _ in range(11):
        cx, cy = rng.uniform(0, 10), rng.uniform(0, 8)
        s = rng.uniform(0.9, 2.6)
        Z += rng.uniform(0.4, 1.0) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2)
                                            / (2 * s ** 2))
    Z += 0.22 * np.sin(0.6 * X) * np.cos(0.5 * Y)
    Z = (Z - Z.min()) / np.ptp(Z)
    return X, Y, Z


def panel_map(fig, ax, rng):
    X, Y, Z = field(rng)
    im = ax.imshow(Z, origin="lower", extent=[0, 10, 0, 8], cmap=TERR,
                   aspect="auto", interpolation="bilinear")
    ax.contour(X, Y, Z, levels=6, colors="white", linewidths=0.4, alpha=0.55)

    th = np.linspace(0, 2 * np.pi, 200)
    bx = 5 + 4.4 * np.cos(th) + 0.35 * np.sin(3 * th)
    by = 4 + 3.3 * np.sin(th) + 0.30 * np.cos(4 * th)
    ax.plot(bx, by, color="white", lw=1.6, alpha=0.9)
    ax.plot(bx, by, color=DEEP, lw=0.7)

    sx, sy = rng.uniform(1.2, 8.8, 46), rng.uniform(1.0, 7.0, 46)
    sv = np.clip(0.25 + 0.7 * rng.random(46), 0, 1)
    ax.scatter(sx, sy, s=16 + 130 * sv ** 2, facecolor="none",
               edgecolor="#f7f3e8", lw=1.0, zorder=4)
    ax.scatter(sx, sy, s=10 + 90 * sv ** 2, c=sv, cmap=STN, vmin=0,
               vmax=1.12, edgecolor=DEEP, lw=0.4, zorder=5, alpha=0.95)

    ax.annotate("", xy=(9.55, 7.6), xytext=(9.55, 6.85),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(9.55, 6.62, "N", ha="center", va="top", fontsize=7.5,
            fontweight="bold", color=INK)
    ax.plot([0.5, 2.5], [0.42, 0.42], color=INK, lw=2.2,
            solid_capstyle="butt")
    ax.text(1.5, 0.58, "200 km", ha="center", fontsize=6.8, color=INK)

    ax.add_patch(FancyBboxPatch(
        (0.28, 6.72), 3.55, 0.95, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="white", alpha=0.78, edgecolor="none", zorder=6))
    cax = ax.inset_axes([0.045, 0.855, 0.29, 0.028], zorder=8)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5, colors=INK)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Habitat integrity index", fontsize=6.5, pad=3, color=INK)

    hx = ax.inset_axes([0.685, 0.055, 0.285, 0.215], zorder=8)
    counts, _, _ = hx.hist(Z.ravel(), bins=26, color=TEAL, alpha=0.9,
                           linewidth=0)
    hx.set_ylim(0, counts.max() * 1.55)
    hx.set_yticks([])
    hx.set_xticks([0, 0.5, 1])
    hx.tick_params(labelsize=6.0, length=2, pad=1)
    hx.set_facecolor("#ffffff")
    hx.patch.set_alpha(0.82)
    for s in ("top", "right", "left"):
        hx.spines[s].set_visible(False)
    hx.spines["bottom"].set_linewidth(0.6)
    hx.text(0.5, 0.90, "pixel distribution", transform=hx.transAxes,
            ha="center", va="center", fontsize=6.0, color=INK)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 9, 2))
    ax.set_xlabel("Easting (100 km)")
    ax.set_ylabel("Northing (100 km)")
    ax.set_title("Regional habitat integrity and monitoring network",
                 fontsize=9, fontweight="bold", color=INK)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)


def panel_ts(ax, rng, title, cols, labels, kind):
    t = np.arange(2004, 2025)
    if kind == "lines":
        for c, lab, amp in zip(cols, labels, [1.0, 0.72, 0.5]):
            y = 50 + amp * (12 * np.tanh((t - 2013) / 4.5)
                            + 2.2 * np.sin(0.8 * (t - 2004))) \
                + rng.normal(0, 1.1, t.size)
            ax.plot(t, y, color=c, lw=1.5, label=lab)
            ax.fill_between(t, y - 2.4, y + 2.4, color=c, alpha=0.13, lw=0)
        ax.set_ylabel("Index (0-100)")
        ax.legend(fontsize=6.5, loc="lower right", handlelength=1.1,
                  borderaxespad=0.2, labelspacing=0.25)
    else:
        v = 3.0 * np.sin(0.55 * (t - 2004)) + rng.normal(0, 1.3, t.size) \
            + 0.13 * (t - 2014)
        ax.bar(t, v, color=np.where(v >= 0, cols[0], cols[1]), width=0.72,
               edgecolor="white", linewidth=0.4)
        ax.axhline(0, color=INK, lw=0.7)
        k = np.convolve(v, np.ones(5) / 5, mode="same")
        ax.plot(t, k, color=DEEP, lw=1.3, ls="--")
        ax.set_ylabel("Anomaly ($\\sigma$)")
    ax.set_xlim(2003, 2025)
    ax.set_xticks([2005, 2010, 2015, 2020])
    ax.set_xlabel("Year")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8.5, fontweight="bold", color=INK)
    style_axes(ax)


def panel_bar(ax, rng):
    labs = ["Wetland", "Forest", "Grassland", "Cropland", "Urban"]
    vals = np.array([28.4, 24.1, 18.7, 15.2, 8.3])
    cols = [TEAL, SAGE, SAND, RUST, "#9aa7ae"]
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, height=0.6, color=cols, edgecolor="white", linewidth=0.8)
    ax.errorbar(vals, y, xerr=rng.uniform(0.8, 2.0, len(vals)), fmt="none",
                ecolor=MUTED, elinewidth=0.8, capsize=2)
    for yi, v in zip(y, vals):
        ax.text(v + 2.4, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlim(0, 36)
    ax.set_xlabel("Area change 2004-2024 (%)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Land-cover turnover", fontsize=8.5, fontweight="bold",
                 color=INK)
    style_axes(ax)


def panel_donut(ax):
    vals = [34, 26, 19, 13, 8]
    labs = ["Climate", "Land use", "Hydrology", "Management", "Other"]
    cols = [DEEP, TEAL, SAGE, SAND, RUST]
    w, _ = ax.pie(vals, colors=cols, startangle=95, counterclock=False,
                  wedgeprops=dict(width=0.40, edgecolor="white", lw=1.1))
    for wedge, v, lab, c in zip(w, vals, labs, cols):
        a = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        r = 0.80
        ax.text(r * np.cos(a), r * np.sin(a), f"{v}%", ha="center",
                va="center", fontsize=6.8, color="white", fontweight="bold")
        ax.text(1.22 * np.cos(a), 1.22 * np.sin(a), lab, ha="center",
                va="center", fontsize=6.8, color=INK)
    ax.text(0, 0.09, "76%", ha="center", va="center", fontsize=14,
            fontweight="bold", color=DEEP)
    ax.text(0, -0.16, "variance\nexplained", ha="center", va="center",
            fontsize=6.5, color=MUTED, linespacing=1.3)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title("Attribution of change", fontsize=8.5, fontweight="bold",
                 color=INK, pad=2)


def panel_timeline(fig, ax, rng):
    ax.axis("off")
    fig.canvas.draw()
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    ratio = (bb.width * fw) / (bb.height * fh)
    ax.set_xlim(0, ratio)
    ax.set_ylim(0, 1)

    ax.add_patch(FancyBboxPatch((-0.008 * ratio, 0.02), 1.016 * ratio, 0.96,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#f6f4ef", edgecolor="none",
                                zorder=0, clip_on=False))
    phases = [("Field campaigns", 0.00, 0.26, DEEP),
              ("Satellite retrieval", 0.26, 0.24, TEAL),
              ("Model calibration", 0.50, 0.22, SAGE),
              ("Scenario runs", 0.72, 0.16, SAND),
              ("Synthesis", 0.88, 0.12, RUST)]
    ybar, hbar = 0.44, 0.21
    for name, s, w, c in phases:
        ax.add_patch(FancyBboxPatch(
            (ratio * (0.035 + s * 0.93) + 0.012, ybar),
            ratio * w * 0.93 - 0.024, hbar,
            boxstyle="round,pad=0,rounding_size=0.055", facecolor=c,
            edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(ratio * (0.035 + (s + w / 2) * 0.93), ybar + hbar / 2, name,
                ha="center", va="center", color="white", fontsize=7.2,
                fontweight="bold", zorder=4)

    ax.plot([ratio * 0.035, ratio * 0.965], [0.31, 0.31], color="#b9b3a6",
            lw=1.0, zorder=2)
    marks = [(0.00, "2016"), (0.26, "2018"), (0.50, "2020"), (0.72, "2022"),
             (0.88, "2023"), (1.00, "2025")]
    for f, lab in marks:
        x = ratio * (0.035 + f * 0.93)
        ax.add_patch(Circle((x, 0.31), 0.035, facecolor="white",
                            edgecolor=DEEP, lw=1.1, zorder=4))
        ax.text(x, 0.145, lab, ha="center", va="center", fontsize=7,
                color=INK, fontweight="bold")
    notes = [(0.13, "1 240 plots"), (0.38, "18 TB imagery"),
             (0.61, "k = 10 CV"), (0.80, "4 SSP futures"), (0.94, "Open data")]
    for f, lab in notes:
        ax.text(ratio * (0.035 + f * 0.93), 0.80, lab, ha="center",
                va="center", fontsize=6.8, color=MUTED)


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(12.4, 8.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.44],
                          width_ratios=[1.62, 1.0, 0.92],
                          left=0.055, right=0.985, top=0.885, bottom=0.055,
                          hspace=0.44, wspace=0.34)
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_t1 = fig.add_subplot(gs[0, 1])
    ax_t2 = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_tl = fig.add_subplot(gs[2, :])

    panel_map(fig, ax_map, rng)
    panel_ts(ax_t1, rng, "Integrity trajectories", [DEEP, TEAL, SAGE],
             ["Core", "Buffer", "Matrix"], "lines")
    panel_ts(ax_t2, rng, "Hydrological anomaly", [TEAL, RUST], None, "bars")
    panel_bar(ax_b, rng)
    panel_donut(ax_d)
    panel_timeline(fig, ax_tl, rng)

    for ax, lab, x in [(ax_map, "(a)", -0.095), (ax_t1, "(b)", -0.19),
                       (ax_t2, "(c)", -0.20), (ax_b, "(d)", -0.19),
                       (ax_d, "(e)", -0.13)]:
        add_panel_label(ax, lab, x=x, y=1.04)
    add_panel_label(ax_tl, "(f)", x=-0.030, y=1.00)

    savefig(fig, "88_composite_overview.png")


if __name__ == "__main__":
    main()
