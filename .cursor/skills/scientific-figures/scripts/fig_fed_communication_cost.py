"""Figure 489 - What compression costs and what it saves.

Left: the uplink payload one client sends per round, on a log axis because the
strategies span two and a half decades and a linear axis would collapse
everything except the uncompressed baseline into the origin.  Centre: the
trade-off that actually decides the deployment, with the Pareto frontier
dashed and the dominated region shaded.  Right: the same payloads integrated
over a 100-round schedule against a fixed uplink budget, which is the number
an operator is billed for.

Payloads, compression ratios and budget-crossing rounds are all derived from
one table of per-round sizes, so the three panels cannot disagree.
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
MUTED = "#777777"
GRID = "#ececec"

FIG_W, FIG_H = 10.4, 6.4
LEFT = [0.085, 0.150, 0.200, 0.435]
MAIN = [0.385, 0.130, 0.335, 0.725]
RIGHT = [0.800, 0.150, 0.185, 0.435]

FAMILIES = {"no compression": GREY, "quantisation": BLUE,
            "sparsification": GREEN, "low-rank": GOLD,
            "quantised sparse (ours)": ACCENT}

# label, payload MB per client per round, final accuracy %, family
POINTS = [("none (fp32)", 12.40, 91.8, "no compression"),
          ("8-bit", 3.10, 91.3, "quantisation"),
          ("4-bit", 1.55, 90.4, "quantisation"),
          ("top-5%", 1.24, 90.7, "sparsification"),
          ("top-1%", 0.25, 89.6, "sparsification"),
          ("top-0.1%", 0.031, 87.4, "sparsification"),
          ("rank 16", 1.05, 89.1, "low-rank"),
          ("rank 8", 0.62, 88.2, "low-rank"),
          ("top-1% + 8-bit", 0.09, 90.9, "quantised sparse (ours)")]
BARS = ["none (fp32)", "8-bit", "rank 8", "top-1%", "top-1% + 8-bit"]
CURVES = ["none (fp32)", "8-bit", "top-1%", "top-1% + 8-bit"]

ROUNDS = np.arange(0, 101)
COHORT = 10                # clients uploading per round
BUDGET_GB = 1.0
BAR_LO = 0.02


def table():
    return {name: (mb, acc, fam) for name, mb, acc, fam in POINTS}


def frontier(pts):
    """Indices minimising payload while maximising accuracy."""
    keep = [i for i, (x, y) in enumerate(pts)
            if not any(xx <= x and yy >= y and (xx < x or yy > y)
                       for xx, yy in pts)]
    return sorted(keep, key=lambda i: pts[i][0])


def main():
    set_pub_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    tab = table()

    # ---- per-round payload ------------------------------------------------
    axb = fig.add_axes(LEFT)
    y = np.arange(len(BARS))[::-1]
    mbs = np.array([tab[n][0] for n in BARS])
    cols = [FAMILIES[tab[n][2]] for n in BARS]
    axb.barh(y, mbs - BAR_LO, left=BAR_LO, height=0.62, color=cols,
             edgecolor="white", linewidth=0.6, zorder=3)
    for yy, v, n in zip(y, mbs, BARS):
        ratio = mbs[0] / v
        tag = f"{v:.2f} MB" if v < 10 else f"{v:.1f} MB"
        if n != BARS[0]:
            tag += f"  ({ratio:.0f}$\\times$)"
        axb.text(v * 1.25, yy, tag, ha="left", va="center", fontsize=6.3,
                 color=SUB, zorder=4)
    axb.set_xscale("log")
    axb.set_xlim(BAR_LO, 900.0)
    axb.set_xticks([0.1, 1.0, 10.0])
    axb.set_xticklabels(["0.1", "1", "10"])
    axb.xaxis.set_minor_formatter(plt.NullFormatter())
    axb.set_yticks(y)
    axb.set_yticklabels(BARS, fontsize=6.8)
    axb.tick_params(labelsize=6.8, length=2.2)
    axb.set_xlabel("uplink per client per round (MB)", fontsize=7.5,
                   labelpad=1.8)
    axb.grid(True, axis="x", lw=0.5, color=GRID)
    axb.set_axisbelow(True)
    style_axes(axb)
    axb.set_title("Payload per round", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    # ---- accuracy against payload ----------------------------------------
    ax = fig.add_axes(MAIN)
    pts = [(mb, acc) for _, mb, acc, _ in POINTS]
    front = frontier(pts)
    fx = [pts[i][0] for i in front]
    fy = [pts[i][1] for i in front]
    ax.plot(fx, fy, color=ACCENT, ls=(0, (4, 2.5)), lw=1.3, zorder=4)

    offsets = {"none (fp32)": (0, -15), "8-bit": (-6, 13),
               "4-bit": (7, -11), "top-5%": (0, 13), "top-1%": (-7, -12),
               "top-0.1%": (8, -3), "rank 16": (7, -10), "rank 8": (-7, 10),
               "top-1% + 8-bit": (-4, 15)}
    for name, mb, acc, fam in POINTS:
        lead = fam.endswith("(ours)")
        col = FAMILIES[fam]
        ax.scatter(mb, acc, s=64 if lead else 38, facecolor=col,
                   alpha=0.95 if lead else 0.72,
                   edgecolor="white" if lead else col, linewidth=1.0,
                   zorder=7 if lead else 5)
        dx, dy = offsets[name]
        ax.annotate(name, xy=(mb, acc), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.4,
                    fontweight="bold" if lead else "normal",
                    color=col if lead else SUB, va="center",
                    ha="center" if abs(dx) < 6 else
                    ("left" if dx > 0 else "right"), zorder=8,
                    arrowprops=dict(arrowstyle="-", lw=0.55, color="#b8b8b8",
                                    shrinkA=0.0, shrinkB=4.0))

    ax.annotate("", xy=(0.045, 0.955), xytext=(0.135, 0.885),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=INK))
    ax.text(0.055, 0.878, "better", transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", color=INK, ha="left", va="top")
    ax.text(0.015, 0.035, "every point below the dashed line is dominated",
            transform=ax.transAxes, fontsize=6.5, color=SUB, ha="left",
            va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(0.018, 30.0)
    ax.set_ylim(84.0, 93.6)
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("uplink per client per round (MB)", fontsize=8.0,
                  labelpad=2.5)
    ax.set_ylabel("accuracy after 100 rounds (%)", fontsize=8.0, labelpad=2.5)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(True, lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Two decades of bandwidth for one accuracy point",
                 fontsize=8.5, fontweight="bold", pad=4.0)

    handles = [Line2D([], [], ls="none", marker="o", ms=5.4, mfc=c,
                      mec="white" if n.endswith("(ours)") else c, mew=0.9,
                      alpha=0.95 if n.endswith("(ours)") else 0.72, label=n)
               for n, c in FAMILIES.items()]
    handles.append(Line2D([], [], color=ACCENT, ls=(0, (4, 2.5)), lw=1.3,
                          label="Pareto frontier"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.6,
              handletextpad=0.6, labelspacing=0.32, borderaxespad=0.35,
              bbox_to_anchor=(1.005, 0.075))

    # ---- cumulative uplink over the schedule -----------------------------
    axc = fig.add_axes(RIGHT)
    axc.axhline(BUDGET_GB, color="#808080", lw=0.9, ls=(0, (4, 2.4)),
                zorder=2)
    axc.text(2.0, BUDGET_GB * 1.18, f"{BUDGET_GB:.0f} GB budget",
             ha="left", va="bottom", fontsize=6.3, color=SUB)
    totals = {}
    for name in CURVES:
        mb, _, fam = tab[name]
        gb = mb * COHORT * ROUNDS / 1024.0
        totals[name] = gb
        lead = fam.endswith("(ours)")
        axc.plot(ROUNDS, np.maximum(gb, 1e-4), color=FAMILIES[fam],
                 lw=1.8 if lead else 1.25, zorder=6 if lead else 4)
        axc.annotate(name, xy=(ROUNDS[-1], gb[-1]), xytext=(-3, 5),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=6.2, color=FAMILIES[fam],
                     fontweight="bold" if lead else "normal")
    spent = {n: g[-1] for n, g in totals.items()}
    axc.text(0.26, 0.03,
             f"after {ROUNDS[-1]} rounds: {spent[CURVES[0]]:.1f} GB vs "
             f"{spent[CURVES[-1]]:.2f} GB", transform=axc.transAxes,
             fontsize=6.3, color=SUB, ha="left", va="bottom")
    axc.set_yscale("log")
    axc.set_xlim(0, 104)
    axc.set_ylim(0.004, 60.0)
    axc.set_xlabel("communication rounds", fontsize=7.5, labelpad=1.8)
    axc.set_ylabel("cumulative uplink (GB)", fontsize=7.5, labelpad=2.0)
    axc.tick_params(labelsize=6.8, length=2.2)
    axc.grid(True, axis="y", lw=0.5, color=GRID)
    axc.set_axisbelow(True)
    style_axes(axc)
    axc.set_title("Cost of a full schedule", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    for tag, rect in (("(a)", LEFT), ("(b)", MAIN), ("(c)", RIGHT)):
        fig.text(rect[0] - 0.62 / FIG_W, rect[1] + rect[3] + 0.07 / FIG_H,
                 tag, fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "489_fed_communication_cost.png")


if __name__ == "__main__":
    main()
