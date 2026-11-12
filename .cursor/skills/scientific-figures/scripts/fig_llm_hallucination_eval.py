"""Figure 320 - A four-panel audit of hallucination in a 70B assistant.

Panel (a) splits the flagged-answer rate of every task into three failure
modes, so it is clear that "hallucination" is mostly factual invention outside
of summarisation.  Panel (b) asks whether the model knows when it is guessing;
it does not, until self-checking is added.  Panel (c) breaks the flagged spans
down by error type, and panel (d) shows what each mitigation actually buys.
Panels (a) and (d) share the same 0-50% rate axis so the bars are comparable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from pubstyle import (set_pub_style, savefig, style_axes, get_palette,
                      annotate_box)

RED = "#b23a48"
BLUE = "#3b5b92"
GOLD = "#c9a227"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"

FIG_W, FIG_H = 10.4, 6.4
RATE_MAX = 50.0

TASKS = [
    ("Biomed QA", 21.4, 9.1, 4.2),
    ("Multi-hop QA", 18.8, 11.6, 3.4),
    ("Open-domain QA", 16.2, 6.8, 2.9),
    ("News QA", 11.5, 7.4, 3.1),
    ("Summarisation", 6.4, 14.9, 4.8),
    ("Code docs", 5.1, 4.6, 8.7),
]
MODES = [("fabricated fact", RED), ("unsupported / inconsistent", BLUE),
         ("instruction violation", GOLD)]

CONF = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
CALIB = [
    ("Base 70B", GREY, np.array([0.42, 0.48, 0.55, 0.62, 0.71])),
    ("+ retrieval", BLUE, np.array([0.50, 0.58, 0.67, 0.76, 0.85])),
    ("+ self-check (ours)", RED, np.array([0.54, 0.63, 0.74, 0.84, 0.93])),
]

TYPES = [("fabricated entity", 34), ("wrong number or date", 24),
         ("unsupported causal claim", 18), ("outdated fact", 13),
         ("self-contradiction", 11)]

MITIGATION = [("greedy decoding", 24.6), ("+ self-consistency", 20.1),
              ("+ retrieval", 14.8), ("+ self-check", 12.3),
              ("all combined (ours)", 8.9)]


def panel_tasks(ax):
    y = np.arange(len(TASKS))[::-1]
    left = np.zeros(len(TASKS))
    for k, (name, col) in enumerate(MODES):
        vals = np.array([t[k + 1] for t in TASKS])
        ax.barh(y, vals, left=left, height=0.62, color=col, lw=0, zorder=3)
        left += vals
    for yy, tot in zip(y, left):
        ax.text(tot + 0.8, yy, f"{tot:.1f}", va="center", ha="left",
                fontsize=6.8, fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in TASKS], fontsize=7.5)
    ax.set_xlim(0, RATE_MAX)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("answers containing a flagged span (%)", fontsize=8.0,
                  labelpad=2.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.xaxis.grid(True, lw=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Failure mix by task", fontsize=8.5, fontweight="bold",
                 pad=4.0)
    ax.legend(handles=[Line2D([], [], color=c, lw=5, label=n)
                       for n, c in MODES],
              loc="lower right", frameon=False, fontsize=6.8,
              handlelength=1.1, labelspacing=0.28, borderaxespad=0.4)


def panel_calibration(ax):
    ax.plot([0.4, 1.0], [0.4, 1.0], color=FAINT, ls=(0, (4, 2.5)), lw=0.9,
            zorder=2)
    ax.text(0.415, 0.965, "dashed = perfect calibration", fontsize=6.4,
            color=FAINT, ha="left", va="top", transform=ax.transData)
    rows = []
    for name, col, acc in CALIB:
        lead = "ours" in name
        ax.plot(CONF, acc, color=col, lw=1.9 if lead else 1.2, marker="o",
                ms=3.4 if lead else 2.6, mfc="white", mew=0.9,
                zorder=5 if lead else 3, label=name)
        ece = float(np.mean(np.abs(acc - CONF)))
        rows.append(f"{name}: ECE = {ece:.3f}")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("stated confidence (-)", fontsize=8.0, labelpad=2.0)
    ax.set_ylabel("empirical accuracy (-)", fontsize=8.0, labelpad=2.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(True, lw=0.5, color="#eeeeee")
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Confidence is over-stated until self-checking",
                 fontsize=8.5, fontweight="bold", pad=4.0)
    annotate_box(ax, "\n".join(rows), loc="lower right", fontsize=6.4,
                 margin=0.035)


def panel_types(ax):
    cols = get_palette("cb_sequential_blue", 7)[::-1][:len(TYPES)]
    total = sum(v for _, v in TYPES)
    start = 90.0
    for (name, v), col in zip(TYPES, cols):
        span = 360.0 * v / total
        ax.add_patch(Wedge((0, 0), 1.0, start - span, start, width=0.42,
                           facecolor=col, edgecolor="white", linewidth=1.1))
        mid = np.radians(start - span / 2.0)
        r = 0.79
        ax.text(r * np.cos(mid), r * np.sin(mid), f"{v}%", ha="center",
                va="center", fontsize=6.6, fontweight="bold",
                color="white" if v > 20 else INK)
        lx, ly = 1.24 * np.cos(mid), 1.24 * np.sin(mid)
        ha = "left" if lx >= 0 else "right"
        ax.plot([1.02 * np.cos(mid), 1.18 * np.cos(mid)],
                [1.02 * np.sin(mid), 1.18 * np.sin(mid)], color="#bbbbbb",
                lw=0.7, zorder=2)
        ax.text(lx, ly, name, ha=ha, va="center", fontsize=6.8, color=INK)
        start -= span
    ax.text(0, 0.10, "1,842", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(0, -0.14, "flagged spans", ha="center", va="center", fontsize=6.6,
            color=SUB)
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("What the flagged spans actually are", fontsize=8.5,
                 fontweight="bold", pad=2.0)


def panel_mitigation(ax):
    x = np.arange(len(MITIGATION))
    vals = np.array([v for _, v in MITIGATION])
    cols = [RED if k == len(vals) - 1 else BLUE for k in range(len(vals))]
    cols = ["#c7cfdd" if 0 < k < len(vals) - 1 else c
            for k, c in enumerate(cols)]
    cols[0] = GREY
    ax.bar(x, vals, width=0.66, color=cols, lw=0, zorder=3)
    ax.axhline(vals[0], color=GREY, ls=(0, (4, 2.5)), lw=0.9, zorder=2)
    ax.text(len(vals) - 0.45, vals[0] + 0.8, "no mitigation", fontsize=6.4,
            color=SUB, ha="right", va="bottom")
    for xi, v in zip(x, vals):
        lead = xi == len(vals) - 1
        ax.text(xi, v + 0.9, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.6, color=RED if lead else SUB,
                fontweight="bold" if lead else "normal")
    ax.annotate("", xy=(len(vals) - 1, vals[-1] + 4.2),
                xytext=(0, vals[0] + 4.2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text((len(vals) - 1) / 2.0, vals[0] + 8.0,
            f"$-{vals[0] - vals[-1]:.1f}$ pp", ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in MITIGATION], fontsize=6.6, rotation=18,
                       ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color(RED)
    ax.set_ylim(0, RATE_MAX)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("flagged-span rate (%)", fontsize=8.0, labelpad=2.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.yaxis.grid(True, lw=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Every mitigation helps, none of them is enough",
                 fontsize=8.5, fontweight="bold", pad=4.0)


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.80],
                          height_ratios=[1.0, 0.82], left=0.075, right=0.975,
                          top=0.885, bottom=0.115, wspace=0.28, hspace=0.62)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_tasks(ax_a)
    panel_calibration(ax_b)
    panel_types(ax_c)
    panel_mitigation(ax_d)

    for tag, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        p = ax.get_position()
        fig.text(p.x0 - 0.44 / FIG_W, p.y1 + 0.07 / FIG_H, f"({tag})",
                 fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "320_llm_hallucination_eval.png")


if __name__ == "__main__":
    main()
