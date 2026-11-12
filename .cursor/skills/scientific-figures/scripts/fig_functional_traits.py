"""Functional trait space of 120 plant species.

(a) PCA of six standardised traits, with convex hulls per functional group and
    trait loading arrows; (b) a 2x2 block of subordinate radar plots giving the
    group trait syndromes against a common 0-1 axis and a shared grand-mean
    reference polygon; (c) the trait correlation matrix; (d) PC1 / PC2 loadings.
Everything downstream is derived from the same simulated trait table, so the
ordination, the radars and the correlation matrix stay mutually consistent.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, style_axes


GROUPS = [
    ("Tree", "#1b5e4b", (-0.85, -0.60, 1.55, 1.75, 1.05, 0.75), 28),
    ("Shrub", "#5aa17f", (-0.45, -0.25, 0.25, 0.15, 0.95, -0.15), 30),
    ("Forb", "#c08f3e", (1.05, 1.15, -0.60, -0.55, -1.05, -0.95), 34),
    ("Graminoid", "#3d8b93", (0.70, 0.15, -1.05, -0.95, -0.75, 1.15), 28),
]

TRAITS = ["SLA", "Leaf N", "Seed mass", "Height", "Wood dens.", "Root depth"]
ABBR = ["SLA", "LN", "SM", "HT", "WD", "RD"]
RUST = "#a2593c"
INK = "#1a1a1a"

CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "eco_corr", ["#8c5a2b", "#cfa877", "#f3f0e8", "#8fc0a9", "#1b5e4b"])


def panel_tag(fig, ax, label, dx=-0.050, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def psd_sqrt(c):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-6, None)
    return v @ np.diag(np.sqrt(w)) @ v.T


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260231)

    # ---- simulated trait table -------------------------------------------
    c = np.eye(6)
    for i, j, r in ((0, 1, 0.58), (0, 4, -0.46), (1, 4, -0.38),
                    (2, 3, 0.52), (3, 5, 0.08), (2, 4, 0.30),
                    (3, 4, 0.24), (0, 2, -0.12)):
        c[i, j] = c[j, i] = r
    lmat = psd_sqrt(c)

    xs, labels = [], []
    for name, col, mu, n in GROUPS:
        eps = rng.normal(0, 1, (n, 6)) @ lmat.T
        xs.append(np.array(mu) + 0.55 * eps)
        labels += [name] * n
    X = np.vstack(xs)
    labels = np.array(labels)
    Z = (X - X.mean(0)) / X.std(0)

    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    scores = u * s
    expl = s ** 2 / (s ** 2).sum() * 100.0
    load = vt[:2].T * np.sqrt(2)

    corr = np.corrcoef(Z, rowvar=False)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.58],
                  height_ratios=[1.0, 0.50], wspace=0.20, hspace=0.42)
    ax_main = fig.add_subplot(gs[0, 0])
    radar_gs = gs[0, 1].subgridspec(2, 2, hspace=0.68, wspace=0.42)
    radars = [fig.add_subplot(radar_gs[i // 2, i % 2], projection="polar")
              for i in range(4)]
    bot = gs[1, :].subgridspec(1, 3, width_ratios=[0.60, 0.98, 0.62],
                               wspace=0.42)
    ax_corr = fig.add_subplot(bot[0])
    ax_load = fig.add_subplot(bot[1])
    ax_leg = fig.add_subplot(bot[2])

    # ---- (a) ordination ---------------------------------------------------
    ax_main.axhline(0, color="#e2e2dc", lw=0.7, zorder=1)
    ax_main.axvline(0, color="#e2e2dc", lw=0.7, zorder=1)

    for name, col, _mu, _n in GROUPS:
        m = labels == name
        pts = scores[m, :2]
        hull = ConvexHull(pts)
        ax_main.add_patch(Polygon(pts[hull.vertices], closed=True,
                                  facecolor=col, alpha=0.11, lw=0, zorder=2))
        ax_main.add_patch(Polygon(pts[hull.vertices], closed=True,
                                  facecolor="none", edgecolor=col, lw=0.9,
                                  alpha=0.85, zorder=3))
        ax_main.scatter(pts[:, 0], pts[:, 1], s=14, color=col, alpha=0.72,
                        edgecolor="white", linewidth=0.4, zorder=4)

    span = np.abs(scores[:, :2]).max()
    arrow_scale = 0.78 * span / np.abs(load).max()
    # extra radial push for the two near-parallel arrows so their labels part
    label_gap = {"Seed mass": 0.34, "Wood dens.": 0.18}
    for k, tname in enumerate(TRAITS):
        vx, vy = load[k] * arrow_scale
        ax_main.add_patch(FancyArrowPatch(
            (0, 0), (vx, vy), arrowstyle="-|>", mutation_scale=7.0,
            linewidth=0.95, color="#4a4a44", alpha=0.9, zorder=6,
            shrinkA=0, shrinkB=0))
        norm = np.hypot(vx, vy)
        gap = label_gap.get(tname, 0.11)
        ax_main.text(vx * (1 + gap / norm), vy * (1 + gap / norm), tname,
                     fontsize=6.4, color="#3a3a34", fontstyle="italic",
                     ha="left" if vx >= 0 else "right",
                     va="bottom" if vy >= 0 else "top", zorder=7,
                     path_effects=[pe.withStroke(linewidth=2.2,
                                                 foreground="white")])

    lim = span * 1.28
    ax_main.set_xlim(-lim, lim)
    ax_main.set_ylim(-lim * 0.80, lim * 0.80)
    ax_main.set_xlabel(f"PC1 ({expl[0]:.1f} % of variance)")
    ax_main.set_ylabel(f"PC2 ({expl[1]:.1f} %)")
    ax_main.set_title("Trait space of 120 species, 6 standardised traits",
                      fontsize=8.4, loc="left", pad=5)
    style_axes(ax_main)

    # ---- (b) subordinate radars -------------------------------------------
    tmin, tmax = X.min(0), X.max(0)
    scaled = (X - tmin) / (tmax - tmin)
    grand = scaled.mean(0)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    for ax, (name, col, _mu, _n) in zip(radars, GROUPS):
        prof = scaled[labels == name].mean(0)
        pc = np.concatenate([prof, prof[:1]])
        gc = np.concatenate([grand, grand[:1]])
        ax.plot(ang_c, gc, color="#b4b4ac", lw=0.7, ls=(0, (2.5, 1.8)),
                zorder=3)
        ax.fill(ang_c, pc, color=col, alpha=0.26, lw=0, zorder=4)
        ax.plot(ang_c, pc, color=col, lw=1.15, zorder=5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([])
        ax.set_xticks(ang)
        ax.set_xticklabels(ABBR, fontsize=6.0, color="#5a5a54")
        # 30 deg offset keeps every label off the horizontal extremes, so
        # neighbouring radars can never collide label-to-label
        ax.set_theta_offset(np.pi / 6)
        ax.tick_params(pad=-1.0)
        ax.grid(color="#e3e3dd", lw=0.5)
        ax.spines["polar"].set_color("#d5d5cd")
        ax.spines["polar"].set_linewidth(0.7)
        ax.set_title(name, fontsize=6.8, color=col, fontweight="bold", pad=8)

    # ---- (c) correlation matrix -------------------------------------------
    nT = len(TRAITS)
    for i in range(nT):
        for j in range(nT):
            if j > i:
                continue
            v = corr[i, j]
            ax_corr.add_patch(Rectangle((j - 0.5, nT - 1 - i - 0.5), 1, 1,
                                        facecolor=CORR_CMAP((v + 1) / 2),
                                        edgecolor="white", linewidth=0.7))
            if i != j:
                ax_corr.text(j, nT - 1 - i, f"{v:+.2f}".replace("+0.", ".")
                             .replace("-0.", "\u2212."), ha="center",
                             va="center", fontsize=6.0,
                             color="white" if abs(v) > 0.45 else "#3a3a34")
    ax_corr.set_xlim(-0.55, nT - 0.45)
    ax_corr.set_ylim(-0.55, nT - 0.45)
    ax_corr.set_aspect("equal", adjustable="box")
    ax_corr.set_xticks(range(nT))
    ax_corr.set_xticklabels(ABBR, fontsize=6.0)
    ax_corr.set_yticks(range(nT))
    ax_corr.set_yticklabels(ABBR[::-1], fontsize=6.0)
    ax_corr.tick_params(length=0, pad=2)
    for sp in ax_corr.spines.values():
        sp.set_visible(False)
    ax_corr.set_title("Trait correlations", fontsize=7.4, loc="left", pad=4)

    # ---- (d) loadings -----------------------------------------------------
    ax_load.set_facecolor("#f7f7f3")
    ypos = np.arange(nT)[::-1]
    h = 0.34
    ax_load.barh(ypos + h / 2 + 0.02, vt[0], height=h, color="#1b5e4b",
                 edgecolor="white", linewidth=0.5, zorder=3, label="PC1")
    ax_load.barh(ypos - h / 2 - 0.02, vt[1], height=h, color="#3d8b93",
                 edgecolor="white", linewidth=0.5, zorder=3, label="PC2")
    ax_load.axvline(0, color="#8d8d84", lw=0.7, zorder=4)
    ax_load.set_yticks(ypos)
    ax_load.set_yticklabels(TRAITS, fontsize=6.3)
    ax_load.set_xlim(-0.72, 0.72)
    ax_load.set_ylim(-0.62, nT - 0.38)
    ax_load.set_xticks([-0.5, 0, 0.5])
    ax_load.tick_params(labelsize=6.1, length=2.0)
    ax_load.set_xlabel("Eigenvector loading", fontsize=6.8, labelpad=2)
    ax_load.set_title("PC1 / PC2 trait loadings", fontsize=7.4, loc="left",
                      pad=4)
    ax_load.grid(axis="x", lw=0.45, color="#e6e6e0")
    style_axes(ax_load)
    ll = ax_load.legend(loc="lower right", bbox_to_anchor=(1.005, 1.005),
                        fontsize=6.2, frameon=False, ncol=2,
                        handlelength=1.1, columnspacing=0.9,
                        handletextpad=0.4, borderaxespad=0.0)
    ll.set_clip_on(False)

    # ---- key --------------------------------------------------------------
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    ax_leg.text(0.0, 1.0, "Functional group", fontsize=7.2,
                fontweight="bold", va="top", color=INK)
    for i, (name, col, _mu, n) in enumerate(GROUPS):
        yy = 0.80 - 0.145 * i
        ax_leg.add_patch(Rectangle((0.0, yy - 0.048), 0.075, 0.096,
                                   facecolor=col, alpha=0.85,
                                   edgecolor="white", linewidth=0.6,
                                   clip_on=False))
        ax_leg.text(0.115, yy, f"{name}  ($n$ = {n})", fontsize=6.6,
                    va="center", color="#33332f")
    ax_leg.text(0.0, 0.185,
                f"PC1 + PC2 = {expl[0] + expl[1]:.1f} % of trait variance\n"
                "grey arrows = trait loadings\n"
                "shaded polygon = group convex hull\n"
                "dashed grey = grand-mean radar profile",
                fontsize=6.2, va="top", color="#82827a", linespacing=1.6)

    fig.subplots_adjust(left=0.072, right=0.985, top=0.898, bottom=0.075)

    fig.canvas.draw()
    panel_tag(fig, ax_main, "(a)", dx=-0.056)
    rb = radars[0].get_position()
    fig.text(rb.x0 - 0.040, rb.y1 + 0.058, "(b)", fontsize=10,
             fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(rb.x0 + 0.002, rb.y1 + 0.060, "Group trait profiles (0-1 scaled)",
             fontsize=7.4, ha="left", va="bottom", color=INK)
    panel_tag(fig, ax_corr, "(c)", dx=-0.044, dy=0.026)
    panel_tag(fig, ax_load, "(d)", dx=-0.052, dy=0.026)

    savefig(fig, "231_functional_traits.png")


if __name__ == "__main__":
    main()
