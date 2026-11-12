"""Figure 679 - Multi-source data fusion for a city grid.

(a) is a three-tier source diagram drawn on an inch canvas, so the rounded
corners and the arrow clearances are the same physical size everywhere
(DESIGN_SYSTEM 12.8); each block carries the update interval and the native
spatial resolution that make the sources worth fusing in the first place.
(b) is the fused 500 m grid.  (c) and (d) are the evidence, at 58 % of the
main panel.

The numbers are not decoration: three latent indicator fields are observed
four times each, once per source, with that source's own blur (its
resolution) and noise (its reliability).  The fusion weights in (c) are the
inverse error variances, the fused field in (b) is that weighted mean, and
the error bars in (d) are measured against the latent field.  Change the seed
and every printed value moves with the picture (DESIGN_SYSTEM 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import (axes_inches, inch_axes, block, arrow, route,
                    safe_colorbar, panel_label_inches, norm01)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"
GOLD, PURPLE, SOLID = "#c9a227", "#8e6bbf", "#3a3a3a"

FW, FH = 12.6, 7.6
N = 24                      # grid cells per side
CELL_KM = 0.5               # 500 m cells -> 12 x 12 km footprint

# name, colour, blur in cells (coarse resolution), noise sd, update, resolution
SOURCES = [
    ("Remote sensing", BLUE, 2.1, 0.09, "1 d", "30 m"),
    ("IoT sensors", GREEN, 0.7, 0.20, "5 min", "200 m"),
    ("Social media", GOLD, 1.3, 0.32, "1 min", "500 m"),
    ("Government records", PURPLE, 3.0, 0.06, "7 d", "district"),
]
INDICATORS = ["Population", "Traffic load", "Air quality"]


def latent(rng, n, smooth):
    f = gaussian_filter(rng.normal(0.0, 1.0, (n, n)), smooth, mode="wrap")
    return norm01(f)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(679)
    fig = plt.figure(figsize=(FW, FH))

    # ---- synthetic observation model -------------------------------------
    truth = np.stack([latent(rng, N, s) for s in (2.6, 2.0, 3.2)])
    obs = np.zeros((len(INDICATORS), len(SOURCES), N, N))
    for i in range(len(INDICATORS)):
        for j, (_, _, blur, sd, _, _) in enumerate(SOURCES):
            obs[i, j] = gaussian_filter(truth[i], blur, mode="wrap") + \
                rng.normal(0.0, sd, (N, N))

    resid = obs - truth[:, None, :, :]
    w = 1.0 / (resid ** 2).mean(axis=(2, 3))
    w = w / w.sum(axis=1, keepdims=True)
    fused = (w[:, :, None, None] * obs).sum(axis=1)

    rng_span = np.ptp(truth, axis=(1, 2))
    mae_src = 100.0 * np.abs(resid).mean(axis=(2, 3)) / rng_span[:, None]
    mae_fus = 100.0 * np.abs(fused - truth).mean(axis=(1, 2)) / rng_span
    best_src = mae_src.min(axis=1)
    gain = 100.0 * (1.0 - mae_fus / best_src)

    w_mean = 100.0 * w.mean(axis=0)
    w_sd = 100.0 * w.std(axis=0)

    # =====================================================================
    # (a) three-tier source diagram on an inch canvas
    # =====================================================================
    axd = inch_axes(fig, (0.56, 0.85, 3.06, 5.60))
    axd.text(0.21, 5.44, "Tier 1   heterogeneous sources", ha="left",
             va="bottom", fontsize=6.8, color=SUB)
    tiers = []
    for k, (name, col, _, sd, upd, res) in enumerate(SOURCES):
        yb = 5.00 - k * 0.76
        a = block(axd, 1.42, yb, 2.42, 0.60, label=name, color=col,
                  sub="update %s   |   %s" % (upd, res), fontsize=7.6,
                  sub_color="#f2f2f2")
        tiers.append(a)
        arrow(axd, a["r"], (2.86, yb), color=BASE, lw=0.9, ms=8, gap_b=0.0,
              style="-")
    axd.plot([2.86, 2.86], [tiers[0]["y"], tiers[-1]["y"]], lw=0.9,
             color=BASE, zorder=2)

    axd.text(0.21, 2.20, "Tier 2   harmonisation", ha="left", va="bottom",
             fontsize=6.8, color=SUB)
    t2 = block(axd, 1.42, 1.78, 2.42, 0.56, label="QC and geo-registration",
               color=SOLID, sub="resample to a common 500 m grid",
               fontsize=7.4, sub_color="#e0e0e0")
    route(axd, [(2.86, tiers[-1]["y"]), (2.86, 1.78), t2["r"]], color=BASE,
          lw=0.9, ms=9, gap=0.03)

    axd.text(0.21, 1.28, "Tier 3   fusion", ha="left", va="bottom",
             fontsize=6.8, color=SUB)
    t3 = block(axd, 1.42, 0.86, 2.42, 0.56,
               label="Inverse-variance weighting", color=RED,
               sub="weights re-estimated per indicator", fontsize=7.4,
               sub_color="#f7e6e8")
    arrow(axd, t2["b"], t3["t"], color=BASE, lw=1.0, ms=9, gap=0.03)
    arrow(axd, t3["r"], (3.04, 0.86), color=RED, lw=1.1, ms=10)
    axd.text(3.04, 1.14, "to (b)", ha="right", va="bottom", fontsize=6.4,
             color=RED)

    # =====================================================================
    # (b) fused 500 m grid
    # =====================================================================
    comp = 100.0 * norm01(fused.mean(axis=0))
    axb = axes_inches(fig, (4.22, 2.22, 4.15, 4.15))
    edges = np.linspace(0.0, N * CELL_KM, N + 1)
    qm = axb.pcolormesh(edges, edges, comp,
                        cmap=get_cmap("cb_sequential_blue"),
                        edgecolors="white", linewidth=0.22, zorder=1)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    for s in axb.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")

    flat = np.dstack(np.unravel_index(np.argsort(comp, axis=None)[::-1],
                                      comp.shape))[0][:3]
    for r_i, c_i in flat:
        axb.add_patch(Rectangle((c_i * CELL_KM, r_i * CELL_KM), CELL_KM,
                                CELL_KM, facecolor="none", edgecolor=RED,
                                linewidth=1.3, zorder=5))
    axb.text(0.015, 1.014, "red outline: top 3 cells, index %.0f-%.0f"
             % (comp[flat[-1][0], flat[-1][1]], comp[flat[0][0], flat[0][1]]),
             transform=axb.transAxes, ha="left", va="bottom", fontsize=6.6,
             color=RED)
    axb.plot([0.4, 2.4], [0.35, 0.35], lw=2.4, color=INK,
             solid_capstyle="butt", zorder=6)
    axb.text(1.4, 0.55, "2 km", ha="center", va="bottom", fontsize=6.4,
             color=INK, zorder=6)
    safe_colorbar(fig, qm, axb, label="composite urban vitality index (-)",
                  width=0.11, pad=0.07, labelsize=6.8)
    axb.set_title("Fused grid, %d x %d cells of %.0f m" %
                  (N, N, CELL_KM * 1000.0), fontsize=9.0, fontweight="bold",
                  pad=13.0)

    # =====================================================================
    # (c) fusion weight per source
    # =====================================================================
    axw = axes_inches(fig, (9.70, 4.40, 2.45, 2.00))
    xs = np.arange(len(SOURCES))
    cols = [s[1] for s in SOURCES]
    axw.bar(xs, w_mean, yerr=w_sd, width=0.62, color=cols, edgecolor="white",
            linewidth=0.7, zorder=3,
            error_kw=dict(ecolor=SUB, elinewidth=0.8, capsize=2.0))
    for x, v in zip(xs, w_mean):
        axw.text(x, v + w_sd.max() + 1.6, "%.0f" % v, ha="center",
                 va="bottom", fontsize=6.5, color=INK)
    axw.set_xticks(xs)
    axw.set_xticklabels(["Remote\nsensing", "IoT", "Social", "Gov."],
                        fontsize=6.5)
    axw.set_ylim(0.0, w_mean.max() + w_sd.max() + 8.0)
    axw.set_ylabel("fusion weight (%)", fontsize=7.4)
    axw.tick_params(labelsize=6.5, length=2.2)
    axw.grid(axis="y", color=GRID, lw=0.55)
    axw.set_title("Inverse-variance weights, mean over %d indicators"
                  % len(INDICATORS), fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axw.spines[s].set_visible(False)

    # =====================================================================
    # (d) accuracy before and after fusion
    # =====================================================================
    axa = axes_inches(fig, (9.70, 1.45, 2.45, 2.00))
    xi = np.arange(len(INDICATORS))
    axa.bar(xi - 0.19, best_src, width=0.34, color=BASE, edgecolor="white",
            linewidth=0.7, zorder=3, label="best single source")
    axa.bar(xi + 0.19, mae_fus, width=0.34, color=GREEN, edgecolor="white",
            linewidth=0.7, zorder=3, label="fused")
    for x, b, f, g in zip(xi, best_src, mae_fus, gain):
        axa.text(x + 0.19, f + 0.30, "-%.0f %%" % g, ha="center", va="bottom",
                 fontsize=6.4, color=GREEN)
        axa.text(x - 0.19, b + 0.30, "%.1f" % b, ha="center", va="bottom",
                 fontsize=6.4, color=SUB)
    axa.set_xticks(xi)
    axa.set_xticklabels(INDICATORS, fontsize=6.5)
    axa.set_ylim(0.0, best_src.max() * 1.30)
    axa.set_ylabel("mean absolute error\n(% of indicator range)",
                   fontsize=7.4)
    axa.tick_params(labelsize=6.5, length=2.2)
    axa.grid(axis="y", color=GRID, lw=0.55)
    axa.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.1, labelspacing=0.25)
    axa.set_title("Error against the latent field", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    for ax, lab in ((axd, "(a)"), (axb, "(b)"), (axw, "(c)"), (axa, "(d)")):
        panel_label_inches(fig, ax, lab, dx=-0.34, dy=0.12)
    savefig(fig, "679_sct_multi_source_fusion.png")


if __name__ == "__main__":
    main()
