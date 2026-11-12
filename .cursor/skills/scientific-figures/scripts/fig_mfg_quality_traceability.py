"""Figure 682 - Genealogy trace from raw lots to finished goods.

Main panel: a four-level batch genealogy on an inch canvas.  Edges are drawn
as smooth-step curves between the node edges rather than straight chords, so
a fan-out never leaves a sharp crossing at a box corner, and the levels are
ordered so the only long edges are the ones the trace itself needs.  The
recall path - the single suspect raw lot and everything downstream of it -
is the one red object in the figure.

Right column at 58 %: (b) where the defects were caught, as a Pareto with
the 80 % line, (c) the control chart for the coating step, whose out-of-limit
batch is exactly the batch the red path passes through.  The defect totals,
the cumulative shares, the control limits and the flagged batch are all
computed from the same two arrays (DESIGN_SYSTEM 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, inch_axes, block, panel_label_inches

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"
SOLID = "#3a3a3a"

LEVELS = ["Raw material lots", "Sub-assembly", "Assembly", "Finished goods"]
NODES = [["RM-A1", "RM-A2", "RM-A3", "RM-A4"],
         ["SA-01", "SA-02", "SA-03", "SA-04", "SA-05"],
         ["AS-01", "AS-02", "AS-03", "AS-04"],
         ["FG-101", "FG-102", "FG-103"]]
RESULT = [["COA pass", "COA pass", "Fe 0.42 % high", "COA pass"],
          ["IQC pass", "IQC pass", "IQC pass", "SPC alarm", "IQC pass"],
          ["FAI pass", "FAI pass", "coating out of limit", "FAI pass"],
          ["OQC pass", "OQC pass", "OQC reject"]]
EDGES = [[(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)],
         [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)],
         [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]]
TRACE = [2, 3, 2, 2]        # the suspect node index at each level

STEPS = ["Coating", "Welding", "Press", "Assembly", "Machining",
         "Inspection", "Packing"]
UCL_K = 3.0


def smoothstep(t):
    return 3.0 * t ** 2 - 2.0 * t ** 3


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(682)
    fig = plt.figure(figsize=(10.4, 6.4))

    # =====================================================================
    # (a) genealogy tree
    # =====================================================================
    axt = inch_axes(fig, (0.72, 0.86, 5.30, 4.24))
    cx = np.array([0.52, 1.94, 3.36, 4.78])
    nw, nh, y_top, y_bot = 1.00, 0.40, 3.56, 0.56

    pos = []
    for lv, names in enumerate(NODES):
        n = len(names)
        ys = (np.linspace(y_top, y_bot, n) if n > 1
              else np.array([(y_top + y_bot) / 2.0]))
        pos.append([(cx[lv], y) for y in ys])

    tt = np.linspace(0.0, 1.0, 60)
    for lv, elist in enumerate(EDGES):
        for i, j in elist:
            x0, y0 = pos[lv][i][0] + nw / 2.0, pos[lv][i][1]
            x1, y1 = pos[lv + 1][j][0] - nw / 2.0, pos[lv + 1][j][1]
            hot = (i == TRACE[lv]) and (j == TRACE[lv + 1])
            axt.plot(x0 + (x1 - x0) * tt, y0 + (y1 - y0) * smoothstep(tt),
                     lw=2.1 if hot else 0.8, color=RED if hot else HAIR,
                     zorder=4 if hot else 2, solid_capstyle="round")

    for lv, names in enumerate(NODES):
        axt.text(cx[lv], 3.98, LEVELS[lv], ha="center", va="bottom",
                 fontsize=7.0, color=SUB, fontweight="bold")
        for i, name in enumerate(names):
            hot = i == TRACE[lv]
            x, y = pos[lv][i]
            block(axt, x, y, nw, nh, label=name,
                  color=RED if hot else lighten(BLUE, 0.74),
                  ec=RED if hot else BLUE, lw=1.4 if hot else 0.8,
                  text_color="white" if hot else INK, fontsize=7.2,
                  sub=RESULT[lv][i],
                  sub_color="#f7e6e8" if hot else SUB, zorder=5)

    axt.text(0.02, 0.20, "red path: recall trace from lot %s down to %s "
             "(%d batches implicated)"
             % (NODES[0][TRACE[0]], NODES[-1][TRACE[-1]], len(TRACE)),
             ha="left", va="bottom", fontsize=6.8, color=RED)
    axt.text(0.02, 0.02, "each box carries the batch identifier and its "
             "inspection verdict", ha="left", va="bottom", fontsize=6.4,
             color=MUTED)
    axt.set_title("Batch genealogy, %d batches over four production levels"
                  % sum(len(n) for n in NODES), fontsize=9.0,
                  fontweight="bold", pad=4.0)

    # =====================================================================
    # (b) Pareto of defects by process step
    # =====================================================================
    axp = axes_inches(fig, (7.06, 3.44, 2.86, 1.76))
    counts = np.sort(rng.integers(9, 132, len(STEPS)))[::-1]
    order = np.argsort(-counts)
    labels = [STEPS[k] for k in order]
    counts = counts[order]
    cum = 100.0 * np.cumsum(counts) / counts.sum()
    k80 = int(np.searchsorted(cum, 80.0)) + 1
    bar_col = [RED if k < k80 else lighten(BLUE, 0.45)
               for k in range(len(counts))]
    axp.bar(np.arange(len(counts)), counts, width=0.66, color=bar_col,
            edgecolor="white", linewidth=0.7, zorder=3)
    axp.set_ylabel("defects recorded (count)", fontsize=7.4)
    axp.set_ylim(0.0, counts.max() * 1.22)
    axp.set_xticks(np.arange(len(counts)))
    axp.set_xticklabels(labels, fontsize=6.4, rotation=32, ha="right")
    axp.tick_params(labelsize=6.5, length=2.2)
    axp.grid(axis="y", color=GRID, lw=0.55)

    axq = axp.twinx()
    axq.plot(np.arange(len(counts)), cum, lw=1.4, color=SOLID, marker="o",
             ms=3.0, zorder=5)
    axq.axhline(80.0, color=GREEN, lw=1.0, ls=(0, (4.0, 2.2)), zorder=4)
    axq.text(len(counts) - 0.4, 81.0, "80 %", ha="right", va="bottom",
             fontsize=6.4, color=GREEN)
    axq.set_ylim(0.0, 108.0)
    axq.set_ylabel("cumulative share (%)", fontsize=7.4)
    axq.tick_params(labelsize=6.5, length=2.2)
    axq.spines["top"].set_visible(False)
    axp.set_title("%d of %d steps carry 80 %% of %d defects"
                  % (k80, len(STEPS), counts.sum()), fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top",):
        axp.spines[s].set_visible(False)

    # =====================================================================
    # (c) control chart for the coating step
    # =====================================================================
    axc = axes_inches(fig, (7.06, 0.95, 2.86, 1.76))
    n_b = 30
    bad = 18                                    # the batch on the red path
    thick = 42.0 + rng.normal(0.0, 0.55, n_b)
    thick[bad] += 2.35
    thick[bad + 1] += 0.85
    base = np.delete(thick, [bad, bad + 1])
    mu, sd = float(base.mean()), float(base.std(ddof=1))
    ucl, lcl = mu + UCL_K * sd, mu - UCL_K * sd
    out = (thick > ucl) | (thick < lcl)

    bx = np.arange(1, n_b + 1)
    axc.fill_between(bx, lcl, ucl, facecolor="#f5f5f5", edgecolor="none",
                     zorder=1)
    axc.plot(bx, thick, lw=1.2, color=BLUE, marker="o", ms=3.0, zorder=3)
    axc.plot(bx[out], thick[out], ls="none", marker="o", ms=5.4, mfc=RED,
             mec="white", mew=0.8, zorder=5)
    for y, lab, col in ((mu, "CL %.2f" % mu, BASE),
                        (ucl, "UCL %.2f" % ucl, RED),
                        (lcl, "LCL %.2f" % lcl, RED)):
        axc.axhline(y, color=col, lw=0.9,
                    ls="-" if col == BASE else (0, (4.0, 2.2)), zorder=2)
        axc.text(n_b + 0.4, y, lab, ha="left", va="center", fontsize=6.2,
                 color=col)
    axc.annotate("batch %d, on the recall path" % bx[bad],
                 xy=(bx[bad], thick[bad]), xytext=(-7.0, -3.0),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=6.4, color=RED)
    axc.set_xlim(0.0, n_b + 5.4)
    axc.set_ylim(min(lcl, thick.min()) - 0.35, thick.max() + 0.85)
    axc.set_xlabel("production batch (sequence)", fontsize=7.4)
    axc.set_ylabel(r"coating thickness ($\mu$m)", fontsize=7.4)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="y", color=GRID, lw=0.55)
    axc.set_title("Coating control chart, %d of %d batches out of limit"
                  % (int(out.sum()), n_b), fontsize=8.5, fontweight="bold",
                  pad=3.0)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    for ax, lab in ((axt, "(a)"), (axp, "(b)"), (axc, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.46, dy=0.12)
    savefig(fig, "682_mfg_quality_traceability.png")


if __name__ == "__main__":
    main()
