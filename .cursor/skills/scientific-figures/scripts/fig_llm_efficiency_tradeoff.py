"""Figure 322 - Serving throughput against benchmark quality.

Each bubble is a served model: horizontal position is the measured decode
throughput on one A100, vertical position is the average benchmark score, and
the area is the parameter count.  The dashed staircase is the Pareto frontier,
so any model below it is dominated by something both faster and better.  The
small panel on the right answers the obvious follow-up - how much of the
frontier survives quantisation - on the same score axis as the main panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

GREY = "#8a8a8a"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"

FIG_W, FIG_H = 10.4, 6.4
MAIN = [0.062, 0.130, 0.585, 0.745]
QUANT = [0.735, 0.480, 0.200, 0.395]
KEY = [0.700, 0.115, 0.300, 0.270]
YLIM = (44.0, 82.0)

FAMILIES = {
    "Distilled": GOLD,
    "Dense (open)": BLUE,
    "Sparse MoE": GREEN,
    "Ours (pruned + distilled)": ACCENT,
}
# name, throughput (tok/s), score, params (B), family, label offset (dx, dy)
MODELS = [
    ("Tiny-1.3B", 5200, 47.8, 1.3, "Distilled", (0, 19)),
    ("Mini-3B", 3400, 54.9, 3.0, "Distilled", (0, 17)),
    ("Dense-7B", 2050, 60.8, 7.0, "Dense (open)", (-6, -20)),
    ("Dense-13B", 1180, 66.2, 13.0, "Dense (open)", (-8, -20)),
    ("Dense-34B", 520, 71.4, 34.0, "Dense (open)", (-16, -20)),
    ("Dense-70B", 255, 75.1, 70.0, "Dense (open)", (2, -22)),
    ("MoE-8x7B", 880, 71.9, 47.0, "Sparse MoE", (-16, -20)),
    ("MoE-8x22B", 320, 78.0, 141.0, "Sparse MoE", (-2, 24)),
    ("Legacy-30B", 300, 63.5, 30.0, "Dense (open)", (16, -6)),
    ("Ours-7B", 1900, 68.4, 7.0, "Ours (pruned + distilled)", (0, 18)),
    ("Ours-13B", 1090, 73.2, 13.0, "Ours (pruned + distilled)", (0, 18)),
    ("Ours-34B", 470, 78.6, 34.0, "Ours (pruned + distilled)", (6, 16)),
]
BITS = np.array([3, 4, 6, 8, 16])
QUANT_CURVES = [
    ("Dense-70B", BLUE, np.array([61.2, 71.8, 74.4, 75.0, 75.1])),
    ("MoE-8x22B", GREEN, np.array([58.4, 73.1, 77.2, 77.9, 78.0])),
    ("Ours-34B", ACCENT, np.array([69.9, 77.4, 78.3, 78.6, 78.6])),
]


def bubble(p):
    return 20.0 * np.sqrt(p) + 16.0


def pareto(points):
    """Indices of models not dominated on (throughput, score)."""
    keep = []
    for i, (x, y) in enumerate(points):
        if not any(xx >= x and yy >= y and (xx > x or yy > y)
                   for xx, yy in points):
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- main trade-off scatter -----------------------------------------
    ax = fig.add_axes(MAIN)
    pts = [(m[1], m[2]) for m in MODELS]
    front = pareto(pts)
    fx = [pts[i][0] for i in front]
    fy = [pts[i][1] for i in front]
    ax.plot(fx, fy, color=ACCENT, ls=(0, (4, 2.5)), lw=1.2, zorder=3)
    ax.fill_between([180] + fx, YLIM[0], [fy[0]] + fy, color=ACCENT,
                    alpha=0.045, lw=0, zorder=1)

    for k, (name, thr, score, par, fam, off) in enumerate(MODELS):
        lead = fam.startswith("Ours")
        col = FAMILIES[fam]
        ax.scatter(thr, score, s=bubble(par), facecolor=col,
                   alpha=0.92 if lead else 0.60,
                   edgecolor="white" if lead else col, linewidth=1.0,
                   zorder=6 if lead else 4)
        ax.annotate(name, xy=(thr, score), xytext=off,
                    textcoords="offset points", fontsize=6.6,
                    fontweight="bold" if lead else "normal",
                    color=col if lead else SUB,
                    ha="center" if abs(off[0]) < 8 else
                    ("left" if off[0] > 0 else "right"),
                    va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.6,
                                    color="#b0b0b0", shrinkA=0.0,
                                    shrinkB=np.sqrt(bubble(par)) / 2.0 + 1.0))

    ax.annotate("", xy=(0.965, 0.955), xytext=(0.885, 0.885),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=INK))
    ax.text(0.955, 0.885, "better", transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", color=INK, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xlim(180, 8000)
    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["200", "500", "1000", "2000", "5000"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(*YLIM)
    ax.set_xlabel("decode throughput on one A100 (tokens/s, batch 1)",
                  fontsize=8.0, labelpad=2.5)
    ax.set_ylabel("average benchmark score (%)", fontsize=8.0, labelpad=2.5)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(True, which="major", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Quality per token of throughput", fontsize=8.5,
                 fontweight="bold", pad=4.0)
    ax.text(0.015, 0.03, "shaded region is dominated by the frontier",
            transform=ax.transAxes, fontsize=6.5, color=SUB, ha="left",
            va="bottom")

    # ---- quantisation sensitivity ---------------------------------------
    axq = fig.add_axes(QUANT)
    for (name, col, vals), dy in zip(QUANT_CURVES, (-1.0, -6.0, 6.0)):
        lead = name.startswith("Ours")
        axq.plot(BITS, vals, color=col, lw=1.8 if lead else 1.1, marker="o",
                 ms=3.4 if lead else 2.6, mfc="white", mew=0.9,
                 zorder=5 if lead else 3)
        axq.annotate(name, xy=(BITS[-1], vals[-1]), xytext=(3, dy),
                     textcoords="offset points", fontsize=6.2, color=col,
                     va="center", ha="left",
                     fontweight="bold" if lead else "normal",
                     annotation_clip=False)
    axq.axvline(4, color=SUB, ls=(0, (3, 2.2)), lw=0.8, zorder=2)
    axq.text(4.15, 46.0, "4-bit deployment point", fontsize=6.2, color=SUB,
             ha="left", va="bottom")
    axq.set_xscale("log", base=2)
    axq.set_xticks(BITS)
    axq.set_xticklabels([str(b) for b in BITS], fontsize=6.8)
    axq.set_xlim(2.6, 19)
    axq.set_ylim(*YLIM)
    axq.set_yticks([50, 60, 70, 80])
    axq.tick_params(labelsize=6.8, length=2.2)
    axq.set_xlabel("weight precision (bits)", fontsize=7.5, labelpad=1.5)
    axq.set_ylabel("score (%)", fontsize=7.5, labelpad=2.0)
    axq.grid(True, axis="y", lw=0.5, color="#ececec")
    axq.set_axisbelow(True)
    style_axes(axq)
    axq.set_title("Cost of quantisation", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    # ---- shared key ------------------------------------------------------
    axk = fig.add_axes(KEY)
    axk.set_axis_off()
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    handles = [Line2D([], [], ls="none", marker="o", ms=6.5,
                      mfc=c, mec="white" if n.startswith("Ours") else c,
                      mew=1.0, alpha=0.92 if n.startswith("Ours") else 0.6,
                      label=n) for n, c in FAMILIES.items()]
    handles.append(Line2D([], [], color=ACCENT, ls=(0, (4, 2.5)), lw=1.2,
                          label="Pareto frontier"))
    leg = axk.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.02, 1.0), frameon=False, fontsize=7.0,
                     handletextpad=0.7, labelspacing=0.42, borderpad=0.0)
    for t, n in zip(leg.get_texts(), list(FAMILIES) + ["Pareto frontier"]):
        if n.startswith("Ours"):
            t.set_fontweight("bold")

    axk.text(0.60, 0.93, "parameters", fontsize=7.0, color=INK,
             fontweight="bold", ha="left", va="top")
    for i, p in enumerate((7, 34, 141)):
        y = 0.66 - i * 0.235
        axk.scatter(0.655, y, s=bubble(p), facecolor="none", edgecolor=SUB,
                    linewidth=0.9)
        axk.text(0.78, y, f"{p} B", fontsize=6.8, color=SUB, ha="left",
                 va="center")

    for tag, rect in (("(a)", MAIN), ("(b)", QUANT)):
        fig.text(rect[0] - 0.38 / FIG_W, rect[1] + rect[3] + 0.06 / FIG_H,
                 tag, fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "322_llm_efficiency_tradeoff.png")


if __name__ == "__main__":
    main()
