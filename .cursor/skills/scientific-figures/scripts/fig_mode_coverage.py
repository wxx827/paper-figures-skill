"""Figure 302 - Mode coverage and mode collapse.

Top: the reference eight-mode mixture, a collapsed baseline and our model,
drawn on a single shared coordinate frame so the three clouds are directly
comparable.  Dashed rings mark the true modes; rings a model never reaches
turn red.  The attached strip underneath reports per-mode recall and the
precision-recall trade-off at a reduced scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from pubstyle import set_pub_style, savefig, add_panel_label

C_REAL = "#2e7d5b"
C_OURS = "#3b5b92"
C_BAD = "#b23a48"
C_NOISE = "#8e6bbf"
C_GREY = "#8f8f8f"
C_INK = "#1a1a1a"

NMODE = 8
RAD = 2.55
SIG = 0.30

# per-mode recall of the reference mixture
REC_BASE = np.array([0.97, 0.93, 0.08, 0.62, 0.95, 0.05, 0.14, 0.11])
REC_OURS = np.array([0.98, 0.96, 0.94, 0.97, 0.95, 0.92, 0.96, 0.93])


def mode_centres():
    th = 2 * np.pi * np.arange(NMODE) / NMODE + np.pi / 8
    return np.c_[RAD * np.cos(th), RAD * np.sin(th)]


def draw_cloud(rng, weights, n_total, spread, jitter=0.0, stray=0):
    """Sample the mixture with per-mode weights and per-mode spread."""
    cen = mode_centres()
    w = np.asarray(weights, float)
    w = w / w.sum()
    counts = np.maximum((w * n_total).astype(int), 0)
    pts = []
    for k in range(NMODE):
        if counts[k] == 0:
            continue
        c = cen[k] + jitter * rng.standard_normal(2)
        pts.append(c + spread[k] * rng.standard_normal((counts[k], 2)))
    if stray:
        ang = rng.uniform(0, 2 * np.pi, stray)
        rr = rng.uniform(0.4, RAD * 1.05, stray)
        pts.append(np.c_[rr * np.cos(ang), rr * np.sin(ang)])
    return np.vstack(pts)


def style_panel(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("#444444")
    ax.tick_params(labelsize=6.4, length=2.4, width=0.7)
    ax.set_axisbelow(True)


def main():
    set_pub_style(base_font=8)
    cen = mode_centres()

    real = draw_cloud(np.random.default_rng(11), np.ones(NMODE), 2400,
                      np.full(NMODE, SIG))
    base = draw_cloud(np.random.default_rng(23),
                      [0.30, 0.25, 0.015, 0.115, 0.245, 0.010, 0.035, 0.030],
                      2100, np.array([0.20, 0.18, 0.13, 0.24, 0.19, 0.12,
                                      0.15, 0.14]),
                      jitter=0.10, stray=70)
    ours = draw_cloud(np.random.default_rng(37),
                      [0.135, 0.128, 0.118, 0.132, 0.126, 0.114, 0.128,
                       0.119], 2300,
                      np.full(NMODE, SIG * 0.97), jitter=0.02, stray=18)

    # ---- one shared frame for all three clouds ----------------------------
    allpts = np.vstack([real, base, ours])
    pad = 0.35
    lo = allpts.min(axis=0) - pad
    hi = allpts.max(axis=0) + pad
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    XLIM = (cx - span / 2, cx + span / 2)
    YLIM = (cy - span / 2, cy + span / 2)

    FW, FH = 7.4, 4.62
    fig = plt.figure(figsize=(FW, FH))

    ph = 0.435
    pw = ph * FH / FW
    y0 = 0.443
    gaps = 0.038
    x0 = (1.0 - (3 * pw + 2 * gaps)) / 2 + 0.012
    xs = [x0 + k * (pw + gaps) for k in range(3)]

    panels = [
        ("Real data", "reference mixture, 8 modes", real, C_REAL,
         np.ones(NMODE)),
        ("Baseline GAN", "mode collapse", base, C_GREY, REC_BASE),
        ("Ours", "full coverage", ours, C_OURS, REC_OURS),
    ]

    for k, (name, sub, pts, col, rec) in enumerate(panels):
        ax = fig.add_axes([xs[k], y0, pw, ph])
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        for j in range(NMODE):
            missed = rec[j] < 0.35
            ax.add_patch(Circle(cen[j], 2.2 * SIG, facecolor="none",
                                edgecolor=C_BAD if missed else "#c4c4cc",
                                linewidth=1.1 if missed else 0.7,
                                linestyle=(0, (3, 2)), zorder=4))
        ax.scatter(pts[:, 0], pts[:, 1], s=2.2, color=col, alpha=0.55,
                   edgecolor="none", zorder=3, rasterized=True)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        style_panel(ax)
        ax.grid(True, color="#f0f0f3", lw=0.5)
        if k > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("$z_2$", fontsize=7.2, labelpad=1.5)
        ax.set_xlabel("$z_1$", fontsize=7.2, labelpad=1.5)
        ax.text(0.0, 1.072, f"({'abc'[k]})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.2, fontweight="bold",
                color=C_INK)
        ax.text(0.165, 1.075, name, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8.2,
                color=col if k != 1 else "#3a3a3a",
                fontweight="bold" if k == 2 else "normal")
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6.5, color="#8a8a8a", fontstyle="italic")

        nmiss = int(np.sum(rec < 0.35))
        if nmiss:
            ax.text(0.5, 0.022, f"{nmiss} of 8 modes never sampled",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.4, color=C_BAD, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#f0c9c6", linewidth=0.6, alpha=0.94))
        else:
            ax.text(0.5, 0.022, f"mean recall {rec.mean():.2f}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.4, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#dcdce2", linewidth=0.6, alpha=0.94))

    # ---- attached strip ----------------------------------------------------
    fig.patches.append(FancyBboxPatch(
        (0.020, 0.048), 0.960, 0.318,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#f3f3f6", edgecolor="#dcdce2",
        linewidth=0.6, zorder=0))
    fig.text(0.032, 0.352, "attached diagnostics", fontsize=6.6,
             color="#9a9a9a", ha="left", va="top", fontstyle="italic")

    # (d) per-mode recall bars
    axD = fig.add_axes([0.076, 0.112, 0.455, 0.185], zorder=1)
    axD.set_facecolor("none")
    idx = np.arange(NMODE)
    bw = 0.38
    axD.bar(idx - bw / 2, 100 * REC_BASE, bw, color=C_GREY, edgecolor="white",
            linewidth=0.5, label="Baseline GAN", zorder=3)
    axD.bar(idx + bw / 2, 100 * REC_OURS, bw, color=C_OURS, edgecolor="white",
            linewidth=0.5, label="Ours", zorder=3)
    axD.axhline(35, color=C_BAD, lw=0.9, ls=(0, (3, 2)), zorder=4,
                label="35% coverage threshold")
    for j in np.where(REC_BASE < 0.35)[0]:
        axD.text(j - bw / 2, 100 * REC_BASE[j] + 2.5, "x", ha="center",
                 va="bottom", fontsize=6.2, color=C_BAD, fontweight="bold")
    axD.set_xticks(idx)
    axD.set_xticklabels([f"m{j + 1}" for j in idx], fontsize=6.3)
    axD.set_ylim(0, 138)
    axD.set_yticks([0, 50, 100])
    axD.set_ylabel("recall  (%)", fontsize=6.9, labelpad=1.8)
    axD.set_xlabel("mixture component", fontsize=6.9, labelpad=1.5)
    style_panel(axD)
    axD.grid(True, axis="y", color="#e6e6ea", lw=0.5)
    axD.legend(loc="upper center", fontsize=6.2, frameon=False, ncol=3,
               handlelength=1.1, columnspacing=1.1, borderaxespad=0.05,
               bbox_to_anchor=(0.5, 1.06))
    add_panel_label(axD, "(d)", x=-0.105, y=1.10, size=8.6)
    axD.text(0.088, 1.115, "per-mode recall", transform=axD.transAxes,
             ha="left", va="bottom", fontsize=7.0, color="#3a3a3a")

    # (e) precision-recall
    axE = fig.add_axes([0.660, 0.112, 0.250, 0.185], zorder=1)
    axE.set_facecolor("none")
    r = np.linspace(0.0, 1.0, 300)
    for nm, pmax, r50, w, c, lw in [
            ("Baseline GAN", 0.865, 0.405, 0.052, C_GREY, 1.3),
            ("Ours", 0.945, 0.790, 0.068, C_OURS, 2.0)]:
        p = pmax / (1 + np.exp((r - r50) / w))
        axE.plot(r, p, color=c, lw=lw, label=nm,
                 zorder=4 if nm == "Ours" else 3)
        k = int(np.argmin(np.abs(p - 0.5 * pmax)))
        axE.scatter([r[k]], [p[k]], s=20, facecolor="white", edgecolor=c,
                    linewidth=1.2, zorder=5)
    axE.set_xlim(0, 1.02)
    axE.set_ylim(0, 1.02)
    axE.set_xticks([0, 0.5, 1.0])
    axE.set_yticks([0, 0.5, 1.0])
    axE.set_xlabel("recall", fontsize=6.9, labelpad=1.5)
    axE.set_ylabel("precision", fontsize=6.9, labelpad=1.8)
    style_panel(axE)
    axE.grid(True, color="#e6e6ea", lw=0.5)
    axE.legend(loc="lower left", fontsize=6.2, frameon=False,
               handlelength=1.2, labelspacing=0.28, borderaxespad=0.15)
    add_panel_label(axE, "(e)", x=-0.185, y=1.10, size=8.6)
    axE.text(-0.030, 1.115, "precision-recall", transform=axE.transAxes,
             ha="left", va="bottom", fontsize=7.0, color="#3a3a3a")


    savefig(fig, "302_mode_coverage.png")


if __name__ == "__main__":
    main()
