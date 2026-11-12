"""Figure 621 - Annotated query execution plan with cardinality drift.

One plan table drives all three panels.  Panel (a) draws the operator tree
bottom-up, so rows flow from the scans at the base to the aggregate at the
root; every node carries its estimated and actual row count and its own
execution time, and the nodes whose actual cardinality misses the estimate by
more than four times are stroked in house red.  Those are the same nodes that
dominate the time-share bars in (b), and the ones outside the tolerance band
in the cost-model deviation plot in (c).

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import block, route, panel_label_inches, use_inch_units

ACCENT = "#b23a48"          # house red - misestimated nodes only
BLUE = "#3b5b92"
TOL = 4.0                   # act/est ratio that counts as a misestimate

# key: (label, level, x fraction, est rows, actual rows, self time ms)
PLAN = {
    "agg": ("HashAggregate", 4, 0.520, 1800, 2140, 18.3),
    "sort": ("Sort", 3, 0.520, 15400, 152600, 96.5),
    "nloop": ("Nested Loop", 2, 0.520, 15400, 152600, 214.8),
    "hjoin": ("Hash Join", 1, 0.285, 9200, 11040, 31.2),
    "scan_o": ("Seq Scan orders", 0, 0.150, 120000, 118420, 42.6),
    "scan_c": ("Index Scan customers", 0, 0.420, 8600, 8712, 6.9),
    "scan_i": ("Index Scan line_items", 0, 0.790, 24000, 96380, 58.4),
}
LINKS = [("scan_o", "hjoin"), ("scan_c", "hjoin"), ("hjoin", "nloop"),
         ("scan_i", "nloop"), ("nloop", "sort"), ("sort", "agg")]


def fmt_rows(v):
    return "%.1fk" % (v / 1e3) if v >= 1000 else "%d" % v


def main():
    set_pub_style(base_font=8)
    ramp = get_palette("cb_sequential_blue", 7)
    level_fc = {0: ramp[3], 1: ramp[4], 2: ramp[4], 3: ramp[5], 4: ramp[5]}

    keys = list(PLAN)
    ratio = np.array([PLAN[k][4] / PLAN[k][3] for k in keys])
    ms = np.array([PLAN[k][5] for k in keys])
    bad = ratio >= TOL
    total_ms = ms.sum()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.045, right=0.955, top=0.885, bottom=0.115,
                          wspace=0.34, hspace=0.50)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a) the plan tree -------------------------------------------------
    w, h = use_inch_units(ax)
    ys = {lv: 0.112 * h + lv * 0.192 * h for lv in range(5)}
    bw, bh = 0.245 * w, 0.118 * h
    boxes = {}
    for k, (lab, lv, fx, est, act, t) in PLAN.items():
        x, y = fx * w, ys[lv]
        flag = (act / est) >= TOL
        b = block(ax, x, y, bw, bh, color=level_fc[lv],
                  ec=ACCENT if flag else "white", lw=1.8 if flag else 0.9,
                  zorder=4, rounding=0.05)
        ax.text(x, y + bh * 0.29, lab, ha="center", va="center",
                fontsize=7.0, color="white", fontfamily="monospace",
                fontweight="bold", zorder=6)
        ax.text(x, y - bh * 0.02, "est %s / act %s rows"
                % (fmt_rows(est), fmt_rows(act)), ha="center", va="center",
                fontsize=6.0, color="#eef4f9", zorder=6)
        ax.text(x, y - bh * 0.31, "%.1f ms  (%.0f%% of plan)"
                % (t, 100.0 * t / total_ms), ha="center", va="center",
                fontsize=6.0, color="#eef4f9", zorder=6)
        if flag:
            ax.text(x + bw / 2 + 0.07, y, "%.1fx" % (act / est), ha="left",
                    va="center", fontsize=6.6, color=ACCENT,
                    fontfamily="monospace", fontweight="bold", zorder=7)
        boxes[k] = b

    for src, dst in LINKS:
        a, b = boxes[src], boxes[dst]
        rows = PLAN[src][4]
        lw = 0.55 + 0.26 * np.log10(max(rows, 10.0))
        ymid = 0.5 * (a["t"][1] + b["b"][1])
        route(ax, [a["t"], (a["x"], ymid), (b["x"], ymid), b["b"]],
              color="#6b7d8f", lw=lw, ms=9, gap=0.05, zorder=3)

    ax.text(0.018 * w, 0.012 * h,
            "rows flow upward; stroke width scales with rows passed on",
            ha="left", va="bottom", fontsize=6.4, color=COLORS["muted"],
            zorder=8)
    ax.set_title("Plan tree, %.0f ms total execution time" % total_ms,
                 fontsize=8.8, fontweight="bold", pad=4.0)

    # ---- (b) time share per operator --------------------------------------
    order = np.argsort(ms)
    names = [PLAN[keys[i]][0] for i in order]
    yy = np.arange(len(order))
    cols = [ACCENT if bad[i] else BLUE for i in order]
    ax_t.barh(yy, ms[order], height=0.62, color=cols, alpha=0.88,
              edgecolor="none", zorder=3)
    for y, v in zip(yy, ms[order]):
        ax_t.text(v + total_ms * 0.012, y, "%.0f%%" % (100.0 * v / total_ms),
                  va="center", fontsize=6.2, color=COLORS["sub"])
    ax_t.set_yticks(yy)
    ax_t.set_yticklabels(names, fontsize=6.4, fontfamily="monospace")
    ax_t.set_xlim(0, ms.max() * 1.20)
    ax_t.set_xlabel("self execution time (ms)", fontsize=7.4)
    ax_t.tick_params(labelsize=6.6)
    ax_t.grid(axis="x", lw=0.45, alpha=0.5)
    ax_t.set_axisbelow(True)
    for s in ("top", "right"):
        ax_t.spines[s].set_visible(False)
    share = ms[bad].sum() / total_ms
    ax_t.set_title("Misestimated nodes burn %.0f%% of the time"
                   % (100 * share), fontsize=8.5, fontweight="bold", pad=3.5)

    # ---- (c) cost-model deviation -----------------------------------------
    order2 = np.argsort(ratio)
    names2 = [PLAN[keys[i]][0] for i in order2]
    yy2 = np.arange(len(order2))
    ax_d.axvspan(1.0 / TOL, TOL, facecolor="#f0f0f0", edgecolor="none",
                 zorder=1)
    ax_d.axvline(1.0, color=COLORS["baseline"], lw=0.9, zorder=2)
    for y, i in zip(yy2, order2):
        col = ACCENT if bad[i] else BLUE
        ax_d.plot([1.0, ratio[i]], [y, y], color=col, lw=1.2, zorder=3)
        ax_d.plot([ratio[i]], [y], marker="o", ms=4.2, mfc=col, mec="white",
                  mew=0.7, zorder=4)
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.55, 26.0)
    ax_d.set_yticks(yy2)
    ax_d.set_yticklabels(names2, fontsize=6.4, fontfamily="monospace")
    ax_d.set_xlabel("actual / estimated rows (x)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.6)
    ax_d.set_xticks([1.0, 2.0, 5.0, 10.0, 20.0])
    ax_d.set_xticklabels(["1", "2", "5", "10", "20"])
    ax_d.grid(axis="x", lw=0.45, alpha=0.5)
    ax_d.set_axisbelow(True)
    for s in ("top", "right"):
        ax_d.spines[s].set_visible(False)
    ax_d.set_ylim(-0.65, len(order2) - 0.35)
    ax_d.text(0.72, len(order2) / 2.0 - 0.5, "tolerance %.0fx" % TOL,
              fontsize=6.2, color=COLORS["muted"], ha="center", va="center",
              rotation=90)
    ax_d.set_title("%d of %d nodes outside tolerance"
                   % (int(bad.sum()), bad.size), fontsize=8.5,
                   fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_t, "(b)"), (ax_d, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.34, dy=0.09)

    root_cause = min((k for k in keys if (PLAN[k][4] / PLAN[k][3]) >= TOL),
                     key=lambda k: PLAN[k][1])
    rc = PLAN[root_cause][4] / PLAN[root_cause][3]

    savefig(fig, "621_db_query_plan.png")


if __name__ == "__main__":
    main()
