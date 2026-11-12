"""Community ordination (NMDS) with fitted environmental vectors.

Main panel: sample scores for four management treatments, each wrapped in a
95 % confidence ellipse, with group centroids joined by a successional
trajectory and passively fitted environmental vectors overlaid.  Stress and
PERMANOVA statistics sit in a call-out box.  A narrow right column carries the
shared key and a small eigenvalue bar chart (~30 % of the main panel width).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, FancyArrowPatch

from pubstyle import set_pub_style, savefig, style_axes


TREATMENTS = [
    ("Reference", "#1b5e4b", (-0.95, 0.30), (0.30, 0.20, 0.35)),
    ("Restored", "#5aa17f", (-0.28, 0.50), (0.28, 0.22, -0.20)),
    ("Light grazing", "#c08f3e", (0.30, -0.08), (0.26, 0.24, 0.15)),
    ("Heavy grazing", "#a2593c", (0.92, -0.46), (0.22, 0.18, -0.30)),
]

ENV_VECTORS = [
    ("Soil moisture", -0.72, 0.11),
    ("Canopy cover", -0.46, -0.40),
    ("Soil pH", 0.50, 0.46),
    ("Bare ground", 0.70, -0.22),
    ("Available N", 0.08, 0.62),
]

VEC = "#4a4a44"
INK = "#1a1a1a"


def panel_tag(fig, ax, label, dx=-0.055, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def conf_ellipse(ax, x, y, color, n_std=2.447):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=angle,
                         facecolor=color, alpha=0.10, lw=0, zorder=2))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=angle,
                         facecolor="none", edgecolor=color, lw=0.9,
                         ls=(0, (3.5, 2.0)), alpha=0.85, zorder=3))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260225)

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.30],
                  height_ratios=[1.0, 0.60], hspace=0.62, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])

    ax.axhline(0, color="#dcdcd6", lw=0.7, zorder=1)
    ax.axvline(0, color="#dcdcd6", lw=0.7, zorder=1)

    centroids = []
    for name, col, (cx, cy), (sx, sy, r) in TREATMENTS:
        n = 15
        cov = np.array([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])
        pts = rng.multivariate_normal([cx, cy], cov, size=n)
        conf_ellipse(ax, pts[:, 0], pts[:, 1], col)
        ax.scatter(pts[:, 0], pts[:, 1], s=15, color=col, alpha=0.70,
                   edgecolor="white", linewidth=0.45, zorder=4)
        centroids.append((pts[:, 0].mean(), pts[:, 1].mean(), col, name))

    # trajectory through centroids
    cxs = [c[0] for c in centroids]
    cys = [c[1] for c in centroids]
    ax.plot(cxs, cys, color="#6f6f66", lw=0.9, ls=(0, (4, 2.2)), zorder=5)
    for cx, cy, col, name in centroids:
        ax.scatter([cx], [cy], s=52, marker="D", color=col,
                   edgecolor="white", linewidth=1.1, zorder=6)

    # ---- environmental vectors -------------------------------------------
    for label, vx, vy in ENV_VECTORS:
        ax.add_patch(FancyArrowPatch(
            (0, 0), (vx, vy), arrowstyle="-|>", mutation_scale=7.0,
            linewidth=0.95, color=VEC, alpha=0.9, zorder=7,
            shrinkA=0, shrinkB=0))
        norm = np.hypot(vx, vy)
        lx, ly = vx * (1 + 0.13 / norm), vy * (1 + 0.13 / norm)
        ax.text(lx, ly, label, fontsize=6.6, color=VEC, zorder=8,
                ha="left" if vx >= 0 else "right",
                va="bottom" if vy >= 0 else "top", fontstyle="italic",
                path_effects=[pe.withStroke(linewidth=2.2,
                                            foreground="white")])

    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
    ax.set_yticks(np.arange(-1.0, 1.1, 0.5))
    ax.set_xlabel("NMDS axis 1")
    ax.set_ylabel("NMDS axis 2")
    ax.set_title("Bray-Curtis NMDS, 60 plots $\\times$ 148 vascular species",
                 fontsize=8.5, loc="left", pad=6)
    style_axes(ax)

    ax.text(0.015, 0.028,
            "2-D stress = $\\bf{0.118}$\n"
            "PERMANOVA $R^2$ = 0.41, $p$ = 0.001\n"
            "envfit: 5 of 8 variables $p<0.05$",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            linespacing=1.35, color="#33332f", zorder=9,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#d3d3cc", linewidth=0.6, alpha=0.94))

    # ---- key -------------------------------------------------------------
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")
    ax_key.text(0.0, 0.99, "Management treatment", fontsize=7.2,
                fontweight="bold", va="top", color=INK)
    for i, (name, col, _c, _s) in enumerate(TREATMENTS):
        y = 0.845 - 0.115 * i
        ax_key.scatter([0.06], [y], s=17, color=col, alpha=0.75,
                       edgecolor="white", linewidth=0.45, clip_on=False)
        ax_key.scatter([0.155], [y], s=44, marker="D", color=col,
                       edgecolor="white", linewidth=1.0, clip_on=False)
        ax_key.text(0.26, y, name, fontsize=6.8, va="center",
                    color="#33332f")
    ax_key.text(0.0, 0.345,
                "circles = plot scores\n"
                "diamonds = group centroid\n"
                "dashed hull = 95 % ellipse\n"
                "grey arrows = env. vectors",
                fontsize=6.4, va="top", color="#6f6f66", linespacing=1.55)

    # ---- subordinate eigenvalue bars -------------------------------------
    ax_bar.set_facecolor("#f6f6f2")
    axes_lab = ["Ax 1", "Ax 2", "Ax 3", "Ax 4"]
    expl = np.array([34.2, 18.6, 9.4, 5.1])
    cols = ["#1b5e4b", "#3d8b93", "#9fbfae", "#cfd8d2"]
    ypos = np.arange(4)[::-1]
    ax_bar.barh(ypos, expl, height=0.62, color=cols, edgecolor="white",
                linewidth=0.6, zorder=3)
    for y, v in zip(ypos, expl):
        ax_bar.text(v + 1.2, y, f"{v:.1f}", va="center", ha="left",
                    fontsize=6.3, color="#4a4a44")
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(axes_lab, fontsize=6.6)
    ax_bar.set_xlim(0, 46)
    ax_bar.set_ylim(-0.62, 3.62)
    ax_bar.set_xticks([0, 20, 40])
    ax_bar.tick_params(labelsize=6.4)
    ax_bar.set_xlabel("Variance explained (%)", fontsize=6.8)
    ax_bar.set_title("PCoA eigenvalues", fontsize=7.2, loc="left", pad=4)
    ax_bar.grid(axis="x", lw=0.5, color="#e2e2dd")
    style_axes(ax_bar)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.115)

    fig.canvas.draw()   # let the equal-aspect box settle before tagging
    panel_tag(fig, ax, "(a)", dx=-0.058)
    panel_tag(fig, ax_bar, "(b)", dx=-0.058, dy=0.038)

    savefig(fig, "225_ordination_nmds.png")


if __name__ == "__main__":
    main()
