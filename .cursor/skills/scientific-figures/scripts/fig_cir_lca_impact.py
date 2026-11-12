"""Life-cycle impact assessment: six categories on one comparable axis.

Absolute LCA indicators span nine orders of magnitude, from megajoules to
kilograms of antimony equivalent, so plotting them side by side on a shared
axis is impossible and plotting them on six free axes invites the reader to
compare bar heights that mean nothing.  Each category is therefore normalised
by its own gross positive total, which makes the *composition* comparable
while the absolute total stays printed above every bar.  The end-of-life
recycling credit is a genuine negative and is drawn below the zero line, so
the net score is the visible distance from the credit to the top of the
stack rather than a number the reader has to trust.

Synthetic inventory; the normalisation, the net scores and the weighted
single score are all recomputed from the inventory table.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, get_palette, darken
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
CREDIT = "#2e7d5b"
STAGES = ["Raw materials", "Manufacturing", "Transport", "Use phase"]

# category, unit label, weight (%), then the four positive stages and the
# end-of-life credit (negative), in the category's own unit
CATEGORIES = [
    ("GWP",            "kg $CO_2$-eq",  32.0, [41.0, 22.5, 7.8, 96.0], -34.0),
    ("Primary energy", "MJ",            18.0, [620.0, 310.0, 95.0, 1450.0],
     -430.0),
    ("Acidification",  "kg $SO_2$-eq",  12.0, [0.190, 0.110, 0.050, 0.340],
     -0.140),
    ("Eutrophication", "kg $PO_4$-eq",  11.0, [0.052, 0.021, 0.008, 0.061],
     -0.019),
    ("Human toxicity", "kg 1,4-DCB-eq", 15.0, [12.4, 9.6, 1.4, 6.2], -5.1),
    ("Resource use",   "kg Sb-eq",      12.0, [3.10e-3, 0.60e-3, 0.20e-3,
                                               0.40e-3], -1.10e-3),
]


def fmt(v):
    """Compact absolute total, whatever the order of magnitude."""
    if v >= 100.0:
        return f"{v:,.0f}"
    if v >= 1.0:
        return f"{v:.1f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}".replace("e-0", "e-")


def main():
    set_pub_style(base_font=9)

    names = [c[0] for c in CATEGORIES]
    units = [c[1] for c in CATEGORIES]
    weights = np.array([c[2] for c in CATEGORIES])
    pos = np.array([c[3] for c in CATEGORIES])           # (cat, stage)
    credit = np.array([c[4] for c in CATEGORIES])

    gross = pos.sum(axis=1)
    share = 100.0 * pos / gross[:, None]                 # % of gross
    credit_pct = 100.0 * credit / gross
    net_pct = 100.0 + credit_pct
    score = weights * net_pct / 100.0                    # weighted, mPt
    total_score = score.sum()
    stage_cols = get_palette("cb_sequential_warm", 7)[2:6]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0],
                          left=0.062, right=0.975, top=0.845, bottom=0.170,
                          wspace=0.28, hspace=0.60)
    ax = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- (a) normalised stage contributions -------------------------------
    x = np.arange(len(CATEGORIES))
    bottom = np.zeros(len(CATEGORIES))
    for j, (stage, col) in enumerate(zip(STAGES, stage_cols)):
        ax.bar(x, share[:, j], bottom=bottom, width=0.62, color=col,
               edgecolor="white", linewidth=0.6, zorder=3, label=stage)
        for i in range(len(CATEGORIES)):
            if share[i, j] >= 9.0:
                ax.text(x[i], bottom[i] + share[i, j] / 2.0,
                        f"{share[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.4, color=darken(col, 0.55), zorder=6)
        bottom += share[:, j]

    ax.bar(x, credit_pct, width=0.62, color=CREDIT, edgecolor="white",
           linewidth=0.6, zorder=3, label="End-of-life recycling credit")
    for i in range(len(CATEGORIES)):
        ax.text(x[i], credit_pct[i] - 2.4, f"{credit_pct[i]:.0f}",
                ha="center", va="top", fontsize=6.4, color=darken(CREDIT, 0.2),
                zorder=6)
        ax.plot([x[i] - 0.36, x[i] + 0.36], [net_pct[i]] * 2, lw=1.2,
                color=RULE, zorder=7, solid_capstyle="butt")
        ax.text(x[i], 104.0, fmt(gross[i]), ha="center", va="bottom",
                fontsize=6.4, color=MUTED, zorder=6)

    ax.axhline(0.0, color=RULE, lw=1.0, zorder=5)
    ax.axhline(100.0, color="#b3b3b3", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.66, len(CATEGORIES) - 0.34)
    ax.set_ylim(-46.0, 152.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({u})" for n, u in zip(names, units)],
                       fontsize=6.8, linespacing=1.45)
    ax.set_yticks(np.arange(-40, 101, 20))
    ax.set_ylabel("Contribution to category gross total (%)")
    ax.set_xlabel("Impact category, gross total printed above each bar")
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    handles = [Patch(facecolor=c, edgecolor="white", lw=0.6, label=s)
               for c, s in zip(stage_cols, STAGES)]
    handles.append(Patch(facecolor=CREDIT, edgecolor="white", lw=0.6,
                         label="End-of-life recycling credit"))
    handles.append(plt.Line2D([0], [0], color=RULE, lw=1.2,
                              label="Net score after credit"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.008), ncol=3, frameon=False,
              fontsize=6.8, handlelength=1.5, columnspacing=1.8)
    ax.set_title("Every category normalised to its own gross total, so the "
                 "mix can be compared", fontsize=9.0, fontweight="bold",
                 loc="left", pad=5)

    # ---- (b) weighting set -------------------------------------------------
    y = np.arange(len(CATEGORIES))[::-1]
    ax_w.barh(y, weights, height=0.60, color="#9c9186", edgecolor="white",
              linewidth=0.5, zorder=3)
    for k, w in enumerate(weights):
        ax_w.text(w + 0.7, y[k], f"{w:.0f}%", va="center", ha="left",
                  fontsize=6.4, color=SUB)
    ax_w.set_xlim(0.0, weights.max() * 1.30)
    ax_w.set_ylim(-0.7, len(CATEGORIES) - 0.3)
    ax_w.set_yticks(y)
    ax_w.set_yticklabels(names, fontsize=6.4)
    ax_w.set_xlabel("Weighting factor (% of single score)", fontsize=8.0)
    ax_w.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_w)
    ax_w.tick_params(axis="x", labelsize=6.8)
    ax_w.tick_params(axis="y", length=0)
    ax_w.spines["left"].set_visible(False)
    ax_w.set_title("Weighting set applied to the normalised results",
                   fontsize=8.5, fontweight="bold", loc="left", pad=4)

    # ---- (c) single score --------------------------------------------------
    ax_s.barh(y, score, height=0.60, color=stage_cols[2], edgecolor="white",
              linewidth=0.5, zorder=3)
    for k, s in enumerate(score):
        ax_s.text(s + 0.5, y[k], f"{s:.1f}", va="center", ha="left",
                  fontsize=6.4, color=darken(stage_cols[2], 0.45))
    ax_s.set_xlim(0.0, score.max() * 1.34)
    ax_s.set_ylim(-0.7, len(CATEGORIES) - 0.3)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(names, fontsize=6.4)
    ax_s.set_xlabel("Weighted single score (mPt)", fontsize=8.0)
    ax_s.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_s)
    ax_s.tick_params(axis="x", labelsize=6.8)
    ax_s.tick_params(axis="y", length=0)
    ax_s.spines["left"].set_visible(False)
    ax_s.set_title(f"Single score, total {total_score:.1f} mPt", fontsize=8.5,
                   fontweight="bold", loc="left", pad=4)
    top = int(np.argmax(score))
    ax_s.text(0.985, 0.05,
              f"{names[top]} = {100.0 * score[top] / total_score:.0f}% "
              f"of total", transform=ax_s.transAxes, ha="right", va="bottom",
              fontsize=6.4, color=SUB)

    panel_labels_inches(fig, [ax, ax_w, ax_s], dx=-0.44, dy=0.09)

    use_lead = int(np.argmax(share[:, 3]))

    print(f"gross={np.round(gross, 3)}")
    print(f"credit%={np.round(credit_pct, 1)}  net%={np.round(net_pct, 1)}")
    print(f"score={np.round(score, 2)}  total={total_score:.2f} mPt")
    savefig(fig, "628_cir_lca_impact.png")


if __name__ == "__main__":
    main()
