"""Figure 321 - Prompting strategies as an additive ablation.

The dot matrix on the left states exactly which ingredients each strategy
switches on, so the rows form a nested ladder rather than five unrelated
methods.  The centre matrix is the accuracy each combination reaches on six
reasoning benchmarks, all on one shared colour scale, and the right-hand bars
reduce every row to the average gain it buys over zero-shot prompting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap

ACCENT = "#b23a48"
BLUE = "#3b5b92"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"

FIG_W, FIG_H = 7.2, 4.5
DOTS = [0.152, 0.190, 0.150, 0.600]
HEAT = [0.345, 0.190, 0.400, 0.600]
BARS = [0.808, 0.190, 0.172, 0.600]
CBAR = [0.345, 0.098, 0.400, 0.016]

STRATEGIES = ["Zero-shot", "Few-shot", "Chain-of-thought",
              "Self-consistency", "Tree-of-thought"]
COMPONENTS = ["exemplars", "rationale", "multi-sample", "search", "verifier"]
ENABLED = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1],
], dtype=bool)
TASKS = ["GSM8K", "MATH", "CSQA", "StrategyQA", "HumanEval", "AQuA"]
SCORE = np.array([
    [31.2, 8.4, 62.1, 58.3, 34.5, 25.1],
    [44.6, 12.1, 71.4, 63.9, 41.2, 32.8],
    [63.8, 19.6, 74.8, 68.4, 45.1, 44.3],
    [72.4, 24.8, 77.2, 71.6, 49.8, 52.6],
    [75.1, 28.3, 78.0, 73.2, 53.4, 56.9],
])
N_R = len(STRATEGIES)


def row_axes(ax):
    ax.set_ylim(N_R - 0.5, -0.5)
    ax.set_yticks(range(N_R))
    ax.tick_params(length=0)


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cmap = get_cmap("cb_sequential_blue")
    norm = Normalize(vmin=SCORE.min(), vmax=SCORE.max())

    # ---- component dot matrix -------------------------------------------
    axd = fig.add_axes(DOTS)
    axd.set_facecolor("none")
    for r in range(N_R):
        if r % 2 == 0:
            axd.add_patch(Rectangle((-0.5, r - 0.5), len(COMPONENTS), 1.0,
                                    facecolor="#f5f5f5", edgecolor="none",
                                    zorder=1))
        on = np.flatnonzero(ENABLED[r])
        if on.size > 1:
            axd.plot([on.min(), on.max()], [r, r], color=SUB, lw=1.1,
                     zorder=2)
        for c in range(len(COMPONENTS)):
            lead = r == N_R - 1
            on = ENABLED[r, c]
            axd.plot(c, r, marker="o", ms=7.2, zorder=3,
                     mfc=(ACCENT if lead else SUB) if on else "white",
                     mec="none" if on else "#c2c2c2", mew=0.8)
    axd.set_xlim(-0.5, len(COMPONENTS) - 0.5)
    row_axes(axd)
    axd.set_yticklabels(STRATEGIES, fontsize=7.5)
    axd.get_yticklabels()[-1].set_fontweight("bold")
    axd.get_yticklabels()[-1].set_color(ACCENT)
    axd.set_xticks(range(len(COMPONENTS)))
    axd.set_xticklabels(COMPONENTS, fontsize=6.5, rotation=40, ha="left",
                        rotation_mode="anchor")
    axd.xaxis.set_ticks_position("top")
    for s in axd.spines.values():
        s.set_visible(False)
    axd.set_title("Components", fontsize=8.5, fontweight="bold", pad=38.0)

    # ---- score matrix ----------------------------------------------------
    axh = fig.add_axes(HEAT)
    axh.imshow(SCORE, cmap=cmap, norm=norm, aspect="auto",
               interpolation="nearest",
               extent=[-0.5, len(TASKS) - 0.5, N_R - 0.5, -0.5])
    best = SCORE.argmax(axis=0)
    for r in range(N_R):
        for c in range(len(TASKS)):
            v = SCORE[r, c]
            top = best[c] == r
            axh.text(c, r, f"{v:.1f}", ha="center", va="center",
                     fontsize=6.5, fontweight="bold" if top else "normal",
                     color="white" if norm(v) > 0.55 else INK, zorder=4)
    for g in np.arange(0.5, len(TASKS) - 1, 1.0):
        axh.axvline(g, color="white", lw=0.6)
    for g in np.arange(0.5, N_R - 1, 1.0):
        axh.axhline(g, color="white", lw=0.6)
    axh.add_patch(Rectangle((-0.5, N_R - 1.5), len(TASKS), 1.0,
                            facecolor="none", edgecolor=ACCENT,
                            linewidth=1.3, zorder=5))
    axh.set_xlim(-0.5, len(TASKS) - 0.5)
    row_axes(axh)
    axh.set_yticklabels([])
    axh.set_xticks(range(len(TASKS)))
    axh.set_xticklabels(TASKS, fontsize=6.8, rotation=40, ha="left",
                        rotation_mode="anchor")
    axh.xaxis.set_ticks_position("top")
    for s in axh.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    axh.set_title("Accuracy per benchmark (%)", fontsize=8.5,
                  fontweight="bold", pad=38.0)

    cax = fig.add_axes(CBAR)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[10, 30, 50, 70])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("accuracy (%)", fontsize=8.0, color=INK, labelpad=2.0)

    # ---- mean gain -------------------------------------------------------
    axb = fig.add_axes(BARS)
    gain = SCORE.mean(axis=1) - SCORE[0].mean()
    cols = [ACCENT if k == N_R - 1 else ("#c9c9c9" if k == 0 else BLUE)
            for k in range(N_R)]
    axb.barh(np.arange(N_R), gain, height=0.55, color=cols, lw=0, zorder=3)
    for r, v in enumerate(gain):
        axb.text(v + 0.7, r, f"+{v:.1f}" if v else "0", va="center",
                 ha="left", fontsize=6.5,
                 fontweight="bold" if r == N_R - 1 else "normal",
                 color=ACCENT if r == N_R - 1 else SUB)
    axb.set_xlim(0, gain.max() * 1.42)
    row_axes(axb)
    axb.set_yticklabels([])
    axb.set_xticks([0, 10, 20])
    axb.tick_params(axis="x", labelsize=6.8, length=2.5)
    axb.set_xlabel("mean gain (pp)", fontsize=7.2, labelpad=2.0)
    axb.xaxis.grid(True, lw=0.5, color="#e6e6e6")
    axb.set_axisbelow(True)
    style_axes(axb)
    axb.tick_params(axis="y", length=0)
    axb.set_title("vs zero-shot", fontsize=8.5, fontweight="bold", pad=38.0)

    for tag, rect in zip("abc", (DOTS, HEAT, BARS)):
        fig.text(rect[0] - 0.30 / FIG_W, rect[1] + rect[3] + 0.50 / FIG_H,
                 f"({tag})", fontsize=11, fontweight="bold", ha="left",
                 va="bottom")

    savefig(fig, "321_llm_prompt_ablation.png")


if __name__ == "__main__":
    main()
