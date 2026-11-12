"""Figure 399 - Tissue nutrient diagnosis from an N-P-K survey.

The main panel is the N-P-K ternary of flag-leaf tissue composition, one point
per sampled plot, coloured by the grain yield that plot went on to produce, with
the agronomic sufficiency envelope drawn on top.  The two subordinate panels
carry the evidence: critical-concentration curves that define the envelope, and
the mix of visual deficiency symptoms scored in the field.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from matplotlib.path import Path

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from figkit import safe_colorbar, place_outside_labels

SEED = 5140
SQ3 = np.sqrt(3.0) / 2.0

OPT = np.array([0.620, 0.092, 0.288])       # sufficiency centroid, N-P-K
SUFF = [(0.500, 0.090, 0.410), (0.540, 0.055, 0.405),
        (0.700, 0.060, 0.240), (0.730, 0.100, 0.170),
        (0.660, 0.165, 0.175), (0.530, 0.155, 0.315)]

NUTRIENTS = [("N", 2.85, "% DM", 4.6, "#2e7d5b"),
             ("P", 0.26, "% DM", 3.1, "#8e6bbf"),
             ("K", 1.90, "% DM", 6.2, "#c9a227")]

SYMPTOMS = [("Balanced", 46.0, "#2e7d5b"),
            ("N deficiency", 21.0, "#c9a227"),
            ("K deficiency", 15.0, "#c0603f"),
            ("P deficiency", 11.0, "#8e6bbf"),
            ("Multiple", 7.0, "#8c2f2f")]


def bary(w):
    """Barycentric N-P-K -> cartesian; P at (0,0), K at (1,0), N on top."""
    w = np.atleast_2d(np.asarray(w, float))
    w = w / w.sum(axis=1, keepdims=True)
    n, p, k = w[:, 0], w[:, 1], w[:, 2]
    return np.column_stack([0.5 * n + k, SQ3 * n])


def draw_frame(ax):
    corners = bary([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.add_patch(Polygon(corners[[1, 2, 0]], closed=True, facecolor="white",
                         edgecolor="#333333", lw=1.0, zorder=1))
    for t in np.arange(0.2, 0.81, 0.2):
        seg = {
            "n": ([t, 1 - t, 0], [t, 0, 1 - t]),
            "p": ([1 - t, t, 0], [0, t, 1 - t]),
            "k": ([1 - t, 0, t], [0, 1 - t, t]),
        }
        for key, (a, b) in seg.items():
            pa, pb = bary([a])[0], bary([b])[0]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#d9d9d9", lw=0.5,
                    zorder=2)
        pn = bary([[t, 1 - t, 0]])[0]
        ax.text(pn[0] - 0.028, pn[1], f"{t * 100:.0f}", ha="right",
                va="center", fontsize=6.4, color="#777777")
        pk = bary([[1 - t, 0, t]])[0]
        ax.text(pk[0] + 0.028, pk[1], f"{t * 100:.0f}", ha="left",
                va="center", fontsize=6.4, color="#777777")
        pp = bary([[0, t, 1 - t]])[0]
        ax.text(pp[0], pp[1] - 0.030, f"{t * 100:.0f}", ha="center", va="top",
                fontsize=6.4, color="#777777")

    ax.text(0.185, 0.50, "N share of N+P+K (%)", rotation=60, ha="center",
            va="center", fontsize=8.0, color="#1a1a1a")
    ax.text(0.815, 0.50, "K share of N+P+K (%)", rotation=-60, ha="center",
            va="center", fontsize=8.0, color="#1a1a1a")
    ax.text(0.50, -0.088, "P share of N+P+K (%)", ha="center", va="center",
            fontsize=8.0, color="#1a1a1a")
    for w, lab, dx, dy, ha, va in [([1, 0, 0], "N", 0.0, 0.032, "center",
                                    "bottom"),
                                   ([0, 1, 0], "P", -0.028, -0.020, "right",
                                    "top"),
                                   ([0, 0, 1], "K", 0.028, -0.020, "left",
                                    "top")]:
        p = bary([w])[0]
        ax.text(p[0] + dx, p[1] + dy, lab, ha=ha, va=va, fontsize=9.0,
                fontweight="bold", color="#1a1a1a")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    n_pts = 96
    comp = rng.dirichlet(OPT * 8.0, n_pts)
    comp = np.clip(comp + rng.normal(0.0, 0.022, comp.shape), 0.01, None)
    comp /= comp.sum(axis=1, keepdims=True)

    xy = bary(comp)
    xy_opt = bary([OPT])[0]
    dist = np.hypot(xy[:, 0] - xy_opt[0], xy[:, 1] - xy_opt[1])
    yld = np.clip(11.4 * np.exp(-(dist / 0.235) ** 2)
                  + rng.normal(0.0, 0.42, n_pts) + 1.1, 1.4, 12.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.030, right=0.905, top=0.885, bottom=0.075,
                          wspace=0.30, hspace=0.46)
    ax_t = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a) N-P-K ternary -------------------------------------------------
    ax_t.set_aspect("equal")
    ax_t.set_axis_off()
    ax_t.set_xlim(-0.10, 1.10)
    ax_t.set_ylim(-0.14, SQ3 + 0.10)
    draw_frame(ax_t)

    suff_xy = bary(SUFF)
    ax_t.add_patch(Polygon(suff_xy, closed=True, facecolor="#2e7d5b",
                           alpha=0.14, edgecolor="none", zorder=3))
    # Outline re-drawn above the scatter so the envelope stays readable where
    # the points are dense.
    ax_t.add_patch(Polygon(suff_xy, closed=True, facecolor="none",
                           edgecolor="#1f6a47", lw=1.2, ls=(0, (4.0, 2.2)),
                           zorder=7))
    inside = Path(suff_xy).contains_points(xy)

    sc = ax_t.scatter(xy[:, 0], xy[:, 1], c=yld, s=27,
                      cmap=get_cmap("cb_sequential_green"), vmin=1.5,
                      vmax=12.0, edgecolor="#333333", linewidth=0.4,
                      zorder=5)
    ax_t.plot([xy_opt[0]], [xy_opt[1]], marker="*", ms=11, mfc="#b23a48",
              mec="white", mew=0.8, zorder=6)
    ax_t.annotate("sufficiency envelope\n"
                  f"{inside.mean():.0%} of plots inside",
                  xy=(suff_xy[:, 0].mean(), suff_xy[:, 1].max()),
                  xytext=(0.135, 0.80), textcoords="axes fraction",
                  fontsize=6.8, color="#1f6a47", ha="left", va="center",
                  linespacing=1.4,
                  arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#1f6a47",
                                  shrinkA=2, shrinkB=3))
    ax_t.text(0.015, 0.985,
              f"n = {n_pts} plots\nmean yield inside "
              f"{yld[inside].mean():.1f} t/ha\nmean yield outside "
              f"{yld[~inside].mean():.1f} t/ha",
              transform=ax_t.transAxes, ha="left", va="top",
              fontsize=6.8, color="#555555", linespacing=1.5)
    ax_t.set_title("Flag-leaf N-P-K composition, coloured by grain yield",
                   fontsize=9.0, fontweight="bold")
    safe_colorbar(fig, sc, ax_t, label="grain yield (t/ha)", width=0.13,
                  pad=0.10, shrink=0.72)

    # ---- (b) critical concentration curves --------------------------------
    x = np.linspace(0.30, 2.00, 240)
    for name, crit, unit, s, col in NUTRIENTS:
        x50 = np.exp(-np.log(9.0) / s)
        y = 100.0 / (1.0 + np.exp(-s * (np.log(x) - np.log(x50))))
        ax_c.plot(x, y, color=col, lw=1.6, zorder=4,
                  label=f"{name}  (critical {crit:g} {unit})")
        xo = np.array([0.42, 0.60, 0.78, 0.95, 1.15, 1.42, 1.75])
        yo = (100.0 / (1.0 + np.exp(-s * (np.log(xo) - np.log(x50))))
              + rng.normal(0.0, 2.6, xo.size))
        ax_c.errorbar(xo, yo, yerr=2.8, fmt="o", ms=2.8, mfc="white", mec=col,
                      mew=0.8, ecolor=col, elinewidth=0.6, capsize=1.6,
                      ls="none", zorder=5)

    ax_c.axvline(1.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_c.axhline(90.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_c.axvspan(0.30, 1.0, color="#c9a227", alpha=0.10, zorder=0)
    ax_c.text(0.63, 30.0, "deficient", fontsize=6.6, color="#8a6b1f",
              ha="center")
    ax_c.text(1.55, 30.0, "sufficient", fontsize=6.6, color="#1f6a47",
              ha="center")
    ax_c.set_xlim(0.30, 2.00)
    ax_c.set_ylim(18.0, 108.0)
    ax_c.set_xlabel("tissue concentration / critical value (-)", fontsize=7.6)
    ax_c.set_ylabel("relative yield (%)", fontsize=7.6)
    ax_c.tick_params(labelsize=6.6)
    ax_c.legend(loc="upper left", fontsize=6.4, frameon=False,
                handlelength=1.4, borderaxespad=0.3)
    ax_c.set_title("Critical-value curves", fontsize=8.5, fontweight="bold")

    # ---- (c) deficiency symptom mix ---------------------------------------
    ax_d.set_aspect("equal")
    ax_d.set_axis_off()
    ax_d.set_xlim(-2.35, 2.35)
    ax_d.set_ylim(-1.32, 1.32)
    total = sum(v for _, v, _ in SYMPTOMS)
    a0 = 90.0
    items = []
    for name, val, col in SYMPTOMS:
        span = 360.0 * val / total
        ax_d.add_patch(Wedge((0, 0), 1.0, a0 - span, a0, width=0.42,
                             facecolor=col, edgecolor="white", lw=0.8))
        items.append((a0 - span / 2.0, name, f"{val / total:.0%}", col))
        a0 -= span
    place_outside_labels(ax_d, items, r_edge=1.02, r_stub=1.10, x_knee=1.30,
                         x_text=1.40, min_gap=0.40, y_span=1.10,
                         name_size=6.8, stat_size=6.4)
    ax_d.text(0, 0.10, f"{SYMPTOMS[0][1] / total:.0%}", ha="center",
              va="center", fontsize=9.5, fontweight="bold", color="#2e7d5b")
    ax_d.text(0, -0.22, "balanced", ha="center", va="center", fontsize=6.4,
              color="#555555")
    ax_d.set_title("Visual symptom scoring", fontsize=8.5, fontweight="bold")

    add_panel_label(ax_t, "(a)", x=0.005, y=1.01)
    add_panel_label(ax_c, "(b)", x=-0.155, y=1.03)
    add_panel_label(ax_d, "(c)", x=-0.075, y=0.98)


    savefig(fig, "399_agr_nutrient_diagnosis.png")


if __name__ == "__main__":
    main()
