"""Portfolio attrition across the development pipeline: stage bands whose
width tracks the number of live programmes and whose height tracks the
stage success rate, the cycle time and cost that each stage consumes, the
indication mix, and the cumulative probability of reaching filing."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches, place_outside_labels

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
BASE = "#808080"

# stage, programmes entering, years per programme, cost per programme (USD M)
STAGES = [("Discovery", 48, 4.5, 8.0),
          ("Preclinical", 22, 1.8, 22.0),
          ("Phase I", 12, 1.3, 48.0),
          ("Phase II", 7, 2.4, 96.0),
          ("Phase III", 3, 3.1, 250.0),
          ("Filed", 2, 1.2, 30.0)]

APPROVAL = 0.85          # probability that a filed programme is approved

INDICATIONS = [("Oncology", 16, RED), ("Immunology", 10, BLUE),
               ("Neurology", 8, GREEN), ("Metabolic", 6, GOLD),
               ("Infectious", 5, PURPLE), ("Other", 3, "#b3b3b3")]


def stage_rates():
    """Stage success rate = programmes advancing / programmes entering."""
    n = [s[1] for s in STAGES]
    rates = [n[i + 1] / n[i] for i in range(len(n) - 1)] + [APPROVAL]
    return np.array(n, dtype=float), np.array(rates)


def funnel_panel(ax, n, rates):
    w = 0.40 + 1.75 * np.log10(n) / np.log10(n[0])
    gap = 0.62
    ramp = [lighten(BLUE, 0.60 - 0.60 * i / (len(STAGES) - 1))
            for i in range(len(STAGES))]
    ramp[-1] = GREEN

    x = 0.0
    centres = []
    for i, ((name, _n, _y, _c), wi, h, col) in enumerate(
            zip(STAGES, w, rates, ramp)):
        ax.add_patch(FancyBboxPatch(
            (x, -h / 2), wi, h, boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=col, edgecolor="white", linewidth=1.0,
            mutation_aspect=1.0, zorder=3))
        ax.text(x + wi / 2, 0.0, f"{int(n[i])}", ha="center", va="center",
                fontsize=8.5, color="white" if i >= 2 else INK,
                fontweight="bold", zorder=5)
        ax.text(x + wi / 2, 0.52, name, ha="center", va="bottom",
                fontsize=7.4, color=INK, zorder=5)
        ax.text(x + wi / 2, -0.54, f"{100 * h:.0f} %\nadvance", ha="center",
                va="top", fontsize=6.3, color=SUB, zorder=5,
                linespacing=1.30)
        centres.append(x + wi / 2)
        if i < len(STAGES) - 1:
            drop = int(n[i] - n[i + 1])
            ax.annotate("", xy=(x + wi + gap - 0.03, 0.0),
                        xytext=(x + wi + 0.03, 0.0),
                        arrowprops=dict(arrowstyle="-|>", color="#9a9a9a",
                                        lw=0.9, shrinkA=0, shrinkB=0))
            ax.text(x + wi + gap / 2, 0.13, f"\u2212{drop}", ha="center",
                    va="bottom", fontsize=6.4, color=RED)
        x += wi + gap

    ax.set_xlim(-0.25, x - gap + 0.25)
    ax.set_ylim(-0.80, 0.82)
    ax.axis("off")
    ax.set_title("Portfolio attrition, stage by stage", fontsize=9.0,
                 fontweight="bold", pad=8)
    return centres


def cost_panel(ax, n):
    names = [s[0] for s in STAGES]
    years = np.array([s[2] for s in STAGES])
    cost = np.array([s[3] for s in STAGES])
    x = np.arange(len(STAGES))

    ax.plot(x, years, color=BLUE, lw=1.6, marker="o", ms=4.0, mfc="white",
            mec=BLUE, mew=1.1, zorder=5, label="cycle time")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylim(0, years.max() * 1.45)
    ax.set_ylabel("Cycle time (years)", fontsize=7.5, color=BLUE)
    ax.tick_params(labelsize=6.6)
    ax.tick_params(axis="y", colors=BLUE)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(x, cost, color=RED, lw=1.6, marker="s", ms=3.8, mfc="white",
             mec=RED, mew=1.1, zorder=5, label="cost per programme")
    ax2.set_yscale("log")
    ax2.set_ylim(4.0, cost.max() * 4.0)
    ax2.set_ylabel("Cost per programme\n(USD million, log scale)",
                   fontsize=7.5, color=RED)
    ax2.tick_params(labelsize=6.6, colors=RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(RED)
    ax2.grid(False)

    total_y = float(years.sum())
    total_c = float((cost * n).sum())
    ax.text(0.015, 0.94, f"{total_y:.1f} years and USD {total_c / 1000:.2f} "
                         f"bn of portfolio spend to reach filing",
            transform=ax.transAxes, fontsize=6.4, color=INK, ha="left",
            va="top")
    ax.set_title("Cycle time and cost by stage", fontsize=8.5,
                 fontweight="bold", pad=4)


def donut_panel(ax, n_total):
    counts = np.array([c for _n, c, _col in INDICATIONS], dtype=float)
    share = 100.0 * counts / counts.sum()
    cols = [c for _n, _c, c in INDICATIONS]
    wedges, _t = ax.pie(counts, colors=cols, startangle=90.0,
                        counterclock=False,
                        wedgeprops=dict(width=0.40, edgecolor="white",
                                        linewidth=0.9))
    items = []
    for wdg, (name, cnt, col), sh in zip(wedges, INDICATIONS, share):
        ang = 0.5 * (wdg.theta1 + wdg.theta2)
        items.append((ang, name, f"{int(cnt)} programmes, {sh:.0f} %", col))
    place_outside_labels(ax, items, min_gap=0.36, x_knee=1.14, x_text=1.20,
                         name_size=6.8, stat_size=6.2)
    ax.text(0, 0.10, f"{int(counts.sum())}", ha="center", va="center",
            fontsize=11.0, color=INK, fontweight="bold")
    ax.text(0, -0.16, "programmes", ha="center", va="center",
            fontsize=6.2, color=SUB)
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Discovery portfolio by indication", fontsize=8.5,
                 fontweight="bold", pad=6)


def pos_panel(ax, rates):
    names = [s[0] for s in STAGES]
    cum = np.array([np.prod(rates[i:]) for i in range(len(rates))])
    y = np.arange(len(names))[::-1]
    cols = [RED if i == 0 else lighten(BLUE, 0.55 - 0.09 * i)
            for i in range(len(names))]
    ax.barh(y, 100 * cum, height=0.60, color=cols, edgecolor="white",
            linewidth=0.6, zorder=3)
    for yi, v in zip(y, cum):
        ax.text(100 * v + 1.6, yi, f"{100 * v:.1f} %", ha="left",
                va="center", fontsize=6.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Probability of reaching approval (%)", fontsize=7.5)
    ax.set_title("Cumulative success from each stage", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    return cum


def main():
    set_pub_style()
    n, rates = stage_rates()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[1.0, 0.82], wspace=0.34, hspace=0.48,
                          left=0.055, right=0.905, top=0.880, bottom=0.105)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    funnel_panel(ax_f, n, rates)
    cost_panel(ax_c, n)
    donut_panel(ax_d, int(n[0]))
    cum = pos_panel(ax_p, rates)

    years = sum(s[2] for s in STAGES)

    panel_labels_inches(fig, [ax_f, ax_c, ax_d, ax_p], dx=-0.34, dy=0.10)
    savefig(fig, "521_drug_pipeline_funnel.png")


if __name__ == "__main__":
    main()
