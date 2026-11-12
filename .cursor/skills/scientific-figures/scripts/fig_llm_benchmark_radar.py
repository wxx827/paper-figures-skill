"""Figure 315 - Multi-benchmark profile of four instruction-tuned models.

The radar carries the conclusion: a 70B open model fine-tuned with our recipe
closes most of the gap to a closed API model, and overtakes it on TruthfulQA,
while a 7B baseline stays inside everyone.  The bar panel on the right is
deliberately drawn at ~58% of the main panel: it is the evidence table behind
the radar, not a second headline.  Both panels use the same 0-100 score scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

GREY = "#808080"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"

BENCH = ["MMLU", "GSM8K", "HumanEval", "HellaSwag", "TruthfulQA", "MATH"]
MODELS = [
    ("Open-7B", GREY, [46, 34, 22, 76, 38, 8]),
    ("Open-70B", BLUE, [68, 72, 41, 86, 52, 24]),
    ("Closed API", GREEN, [84, 92, 78, 92, 62, 52]),
    ("Ours-70B", ACCENT, [79, 86, 63, 89, 66, 38]),
]
RLIM = 100.0


def radar(ax):
    ang = np.linspace(0, 2 * np.pi, len(BENCH), endpoint=False)
    closed = np.concatenate([ang, ang[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, RLIM)
    ax.set_rgrids([20, 40, 60, 80, 100],
                  labels=["20", "40", "60", "80", "100"], angle=90,
                  fontsize=6.2, color=FAINT)
    ax.set_thetagrids(np.degrees(ang), labels=BENCH, fontsize=7.5)
    for lbl in ax.get_yticklabels():
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.85))
    ax.tick_params(axis="x", pad=1.5)
    ax.grid(color="#d9d9d9", lw=0.6)
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_facecolor("white")

    for name, col, vals in MODELS:
        v = np.concatenate([vals, vals[:1]])
        lead = name.startswith("Ours")
        ax.plot(closed, v, color=col, lw=1.9 if lead else 1.2,
                zorder=5 if lead else 3)
        ax.fill(closed, v, color=col, alpha=0.13 if lead else 0.06,
                zorder=2, lw=0)
        if lead:
            ax.plot(closed, v, ls="none", marker="o", ms=3.2, color=col,
                    zorder=6)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")


def ranking(ax):
    order = list(range(len(BENCH)))[::-1]
    n_m = len(MODELS)
    step = 1.0
    h = 0.19
    yticks, ylabels = [], []

    for row, b in enumerate(order):
        base = row * step
        yticks.append(base)
        ylabels.append(BENCH[b])
        for m, (name, col, vals) in enumerate(MODELS):
            y = base + (m - (n_m - 1) / 2.0) * h
            lead = name.startswith("Ours")
            ax.barh(y, vals[b], height=h * 0.86, color=col,
                    alpha=1.0 if lead else 0.85, lw=0, zorder=3)
            if lead:
                ax.text(vals[b] + 2.0, y, f"{vals[b]}", va="center",
                        ha="left", fontsize=6.0, color=ACCENT,
                        fontweight="bold", zorder=4)

    # mean-score summary row, set apart by a hairline rule
    base = len(BENCH) * step + 0.25
    ax.axhline(base - 0.62, color="#cccccc", lw=0.7, zorder=1)
    yticks.append(base)
    ylabels.append("Average")
    for m, (name, col, vals) in enumerate(MODELS):
        y = base + (m - (n_m - 1) / 2.0) * h
        lead = name.startswith("Ours")
        mu = float(np.mean(vals))
        ax.barh(y, mu, height=h * 0.86, color=col,
                alpha=1.0 if lead else 0.85, lw=0, zorder=3)
        ax.text(mu + 2.0, y, f"{mu:.0f}", va="center", ha="left",
                fontsize=6.0, fontweight="bold" if lead else "normal",
                color=ACCENT if lead else SUB, zorder=4)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.8)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-0.55, base + 0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=6.5, length=2.5)
    ax.set_xlabel("score (%)", fontsize=7.0, labelpad=1.5)
    ax.xaxis.grid(True, lw=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], left=0.045,
                          right=0.975, top=0.780, bottom=0.105, wspace=0.34)

    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    radar(ax_r)
    ax_b = fig.add_subplot(gs[0, 1])
    ranking(ax_b)

    add_panel_label(ax_r, "(a)", x=-0.10, y=1.02)
    add_panel_label(ax_b, "(b)", x=-0.28, y=1.02)
    ax_r.set_title("Capability profile", fontsize=8.5, fontweight="bold",
                   pad=9.0)
    ax_b.set_title("Per-benchmark ranking", fontsize=8.5, fontweight="bold",
                   pad=5.0)

    handles = [Line2D([], [], color=c, lw=2.0 if n.startswith("Ours") else 1.4,
                      label=n) for n, c, _ in MODELS]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    for txt, (n, _, _) in zip(leg.get_texts(), MODELS):
        if n.startswith("Ours"):
            txt.set_fontweight("bold")

    savefig(fig, "315_llm_benchmark_radar.png")


if __name__ == "__main__":
    main()
