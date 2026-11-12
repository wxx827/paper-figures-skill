"""The h-index as a geometric construction on the citation-rank curve.

Main panel: papers of four fictional scholars ranked by citations received,
with the y = x diagonal and the shaded h x h square of the focal scholar.
The square is not drawn at a chosen size - h is found from the sorted
citation vector (the largest k with c_k >= k), so the corner of the square
always lands on the step of the curve where it crosses the diagonal, and the
two neighbouring papers that bracket the index are labelled with their own
counts.  The axes are given matching physical scales so the square is a
square on paper; papers above the top of the frame are annotated rather than
rescaled, because a log axis would destroy the construction.

Auxiliary (both at 58 % of the main panel width): the same four rank-citation
curves on log-log axes, where the heavy tail that the linear main panel
truncates becomes a straight line with a fitted slope; and h against total
citations for 120 simulated scholars from two fictional fields, with the
square-root law recovered as a least-squares fit in log space.

Everything quoted is measured from the generated arrays: h by search, the
tail slopes and the h-versus-citations exponents by ``np.polyfit``.  Fictional
scholars and fields, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.92, 0.86, 5.50, 4.62)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

# x and y limits are chosen in the ratio of the panel's own inch size, so one
# data unit is the same length on both axes and the h-square really is square.
X_MAX, Y_MAX = 76.0, 64.0

SCHOLARS = [
    ("R. Anselm", 74, 3.05, 1.10, DEEP, 2.0),
    ("T. Okabe-Ling", 52, 2.70, 1.00, lighten(BLUE, 0.30), 1.3),
    ("M. Duarte", 41, 2.35, 1.45, RUST, 1.3),
    ("K. Ivarsen", 23, 2.80, 0.85, lighten(RUST, 0.42), 1.3),
]


def citation_vector(rng, n, mu, sigma):
    """Citations per paper, sorted from most to least cited."""
    c = np.round(rng.lognormal(mu, sigma, n)).astype(int)
    return np.sort(c)[::-1]


def h_index(c_sorted):
    """Largest k such that the k-th most cited paper has at least k citations."""
    k = np.arange(1, c_sorted.size + 1)
    ok = c_sorted >= k
    return int(k[ok].max()) if ok.any() else 0


def loglog_slope(c_sorted, kmax=None):
    """Least-squares slope of log citations against log rank."""
    k = np.arange(1, c_sorted.size + 1, dtype=float)
    m = c_sorted > 0
    if kmax is not None:
        m &= k <= kmax
    return float(np.polyfit(np.log10(k[m]), np.log10(c_sorted[m]), 1)[0])


def field_population(rng, n_sch, mu, sigma, n_lo, n_hi):
    """(total citations, h) for a population of simulated scholars."""
    tot, hh = [], []
    for _ in range(n_sch):
        n = int(rng.integers(n_lo, n_hi))
        c = citation_vector(rng, n, mu, sigma)
        tot.append(int(c.sum()))
        hh.append(h_index(c))
    return np.array(tot, dtype=float), np.array(hh, dtype=float)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(990)

    curves = []
    for name, n, mu, sigma, col, lw in SCHOLARS:
        c = citation_vector(rng, n, mu, sigma)
        curves.append(dict(name=name, c=c, h=h_index(c), col=col, lw=lw,
                           total=int(c.sum())))
    focal = curves[0]
    h = focal["h"]
    c_at_h = int(focal["c"][h - 1])
    c_next = int(focal["c"][h]) if focal["c"].size > h else 0
    top_cited = int(focal["c"][0])
    above = int((focal["c"] > Y_MAX).sum())
    # share of the focal scholar's citations that sit inside the h-core
    core_share = 100.0 * focal["c"][:h].sum() / focal["c"].sum()

    tot_a, h_a = field_population(rng, 120, 2.45, 1.20, 18, 90)
    tot_b, h_b = field_population(rng, 120, 3.15, 1.05, 12, 60)
    fit_a = np.polyfit(np.log10(tot_a), np.log10(h_a), 1)
    fit_b = np.polyfit(np.log10(tot_b), np.log10(h_b), 1)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: rank-citation curves and the h-square ------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.55, zorder=0)
    ax.add_patch(Rectangle((0, 0), h, h, facecolor="#f0e6cd", alpha=0.55,
                           edgecolor=GOLD, linewidth=1.0, zorder=1))
    ax.plot([0, 52.0], [0, 52.0], color=SUB, lw=0.9, ls=(0, (3.2, 2.2)),
            zorder=3)
    ax.text(37.0, 37.8, "citations = rank", rotation=45.0,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=6.6,
            color=SUB)

    for cu in curves:
        k = np.arange(1, cu["c"].size + 1)
        ax.step(k, cu["c"], where="post", color=cu["col"], lw=cu["lw"],
                zorder=5 if cu is focal else 4)

    ax.plot([h], [h], marker="s", ms=5.6, mfc="white", mec=ACCENT, mew=1.4,
            zorder=8)
    ax.plot([h, h], [0, h], color=ACCENT, lw=0.9, ls=(0, (2.4, 1.8)), zorder=6)
    ax.plot([0, h], [h, h], color=ACCENT, lw=0.9, ls=(0, (2.4, 1.8)), zorder=6)
    ax.plot([h], [c_at_h], marker="o", ms=3.6, mfc="white", mec=DEEP, mew=1.0,
            zorder=8)
    ax.plot([h + 1], [c_next], marker="o", ms=3.6, mfc="white", mec=DEEP,
            mew=1.0, zorder=8)

    ax.annotate(f"paper {h} has {c_at_h} citations (>= {h})",
                xy=(h, c_at_h), xytext=(h - 20.0, c_at_h + 14.0),
                fontsize=6.5, color=DEEP, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                connectionstyle="arc3,rad=-0.22"))
    ax.annotate(f"paper {h + 1} has {c_next} citations (< {h + 1})",
                xy=(h + 1, c_next), xytext=(h + 11.0, c_next - 12.0),
                fontsize=6.5, color=DEEP, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                connectionstyle="arc3,rad=0.22"))
    sq = ax.text(h / 2.0, h / 2.0,
                 f"h = {h}\n{h} papers x {h} citations",
                 ha="center", va="center", fontsize=8.2, color="#8a6d1f",
                 fontweight="bold", linespacing=1.45, zorder=9)
    sq.set_path_effects([pe.withStroke(linewidth=2.6, foreground="#fbf5e8")])
    ax.text(45.0, 46.5,
            f"{above} papers of the focal scholar run off\nthe top of the "
            f"frame (most cited:\n{top_cited} citations); the h-core holds\n"
            f"{core_share:.0f} % of that scholar's citations",
            ha="left", va="top", fontsize=6.4, color=MUTED, linespacing=1.5)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(np.arange(0, 71, 10))
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_xlabel("paper rank, most cited first (papers)", fontsize=8.5)
    ax.set_ylabel("citations received (citations)", fontsize=8.5)
    ax.set_title("Ranked citation curves of four fictional scholars, "
                 "linear axes so the h-square is square",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    handles = [Line2D([], [], color=cu["col"], lw=cu["lw"],
                      label=f"{cu['name']}: {cu['c'].size} papers, "
                            f"{cu['total']} citations, h = {cu['h']}")
               for cu in curves]
    leg = ax.legend(handles=handles, loc="upper right", frameon=False,
                    fontsize=6.6, labelspacing=0.45, handlelength=2.0,
                    borderpad=0.2, bbox_to_anchor=(1.008, 1.008))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): the same curves on log-log axes ---------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for cu in curves:
        k = np.arange(1, cu["c"].size + 1)
        m = cu["c"] > 0
        ax_t.loglog(k[m], cu["c"][m], color=cu["col"], lw=cu["lw"], zorder=4)
        ax_t.plot([cu["h"]], [cu["h"]], marker="s", ms=3.4, mfc="white",
                  mec=ACCENT, mew=1.0, zorder=7)
    kk = np.logspace(0, np.log10(74), 40)
    s_focal = loglog_slope(focal["c"])
    ax_t.loglog(kk, 10.0 ** (np.polyfit(
        np.log10(np.arange(1, focal["c"].size + 1)[focal["c"] > 0]),
        np.log10(focal["c"][focal["c"] > 0]), 1)[1] + s_focal * np.log10(kk)),
        color=ACCENT, lw=1.1, ls=(0, (3.4, 2.0)), zorder=6)
    ax_t.set_xlim(0.85, 110)
    ax_t.set_ylim(0.7, 900)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("paper rank (papers, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("citations (log scale)", fontsize=7.6)
    ax_t.set_title("The tail the linear panel cuts off", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(0.030, 0.055,
              f"focal scholar, fitted slope {s_focal:.2f}\n"
              f"red squares mark each scholar's h (h-index, dimensionless)",
              transform=ax_t.transAxes, ha="left", va="bottom", fontsize=6.1,
              color=SUB, linespacing=1.5)

    # ---- aux (c): h against total citations ---------------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.scatter(tot_a, h_a, s=8.0, facecolor=lighten(BLUE, 0.42),
                 edgecolor=BLUE, linewidth=0.35, alpha=0.85, zorder=4)
    ax_b.scatter(tot_b, h_b, s=8.0, facecolor=lighten(RUST, 0.45),
                 edgecolor=RUST, linewidth=0.35, alpha=0.85, zorder=4)
    xg = np.logspace(np.log10(min(tot_a.min(), tot_b.min())),
                     np.log10(max(tot_a.max(), tot_b.max())), 60)
    ax_b.plot(xg, 10.0 ** (fit_a[1] + fit_a[0] * np.log10(xg)), color=DEEP,
              lw=1.4, zorder=6)
    ax_b.plot(xg, 10.0 ** (fit_b[1] + fit_b[0] * np.log10(xg)), color=RUST,
              lw=1.4, zorder=6)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(xg[0] * 0.75, xg[-1] * 1.5)
    ax_b.set_ylim(3.0, 90.0)
    ax_b.set_yticks([5, 10, 20, 40, 80])
    ax_b.set_yticklabels(["5", "10", "20", "40", "80"])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("career citations (citations, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("h-index (log scale)", fontsize=7.6)
    ax_b.set_title("h grows like the square root of the total",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    hb = [Line2D([], [], color=DEEP, lw=1.4,
                 label=f"field Kaleva, exponent {fit_a[0]:.2f}"),
          Line2D([], [], color=RUST, lw=1.4,
                 label=f"field Orsan, exponent {fit_b[0]:.2f}")]
    lg = ax_b.legend(handles=hb, loc="upper left", frameon=False,
                     fontsize=6.2, handlelength=1.8, labelspacing=0.32,
                     borderpad=0.15, bbox_to_anchor=(-0.012, 1.030))
    for t in lg.get_texts():
        t.set_color(SUB)
    ax_b.text(0.975, 0.055,
              "n = 120 simulated scholars per field",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.1,
              color=MUTED)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.64, dy=0.12)
    savefig(fig, "990_sci_h_index.png")


if __name__ == "__main__":
    main()
