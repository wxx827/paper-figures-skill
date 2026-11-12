"""Leaderboard: aggregate score decomposed per benchmark, plus a rank table.

The bars are sorted by the aggregate score and each bar is split into the
five benchmark contributions, so the reader sees both the ranking and where
it comes from.  The right-hand block repeats the same data as per-benchmark
ranks (number plus tint).  The dashed rule marks the previous state of the
art; only the proposed model crosses it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten

OURS = "#3b5b92"
GRID = "#eeeeee"

TASKS = ["ImageNet-1k", "COCO det.", "ADE20K seg.", "Kinetics-400", "VQAv2"]
TASK_SHORT = ["IN-1k", "COCO", "ADE", "K400", "VQA"]

SCORES = {
    "Ours":            [86.5, 60.2, 55.8, 84.1, 79.6],
    "SwinV2-L":        [84.9, 58.7, 54.1, 82.0, 76.4],
    "ConvNeXt-L":      [84.3, 57.9, 53.5, 80.6, 74.1],
    "CoAtNet-3":       [84.5, 57.4, 52.6, 81.3, 75.2],
    "ViT-L/16":        [83.1, 55.8, 51.2, 80.2, 74.6],
    "MetaFormer-B":    [83.6, 56.2, 51.8, 79.9, 73.8],
    "EffNetV2-L":      [82.8, 54.9, 49.8, 78.4, 71.2],
    "DeiT III-L":      [82.2, 54.1, 49.1, 77.6, 70.5],
    "ResNeSt-269":     [80.9, 53.2, 48.4, 76.2, 68.9],
    "RegNetZ-D":       [80.1, 52.6, 47.6, 75.4, 67.8],
}


def main():
    set_pub_style(base_font=8)

    names = list(SCORES)
    mat = np.array([SCORES[n] for n in names], dtype=float)
    total = mat.mean(axis=1)
    order = np.argsort(-total)
    names = [names[i] for i in order]
    mat, total = mat[order], total[order]
    # rank 1 = best on that benchmark
    rank = (-mat).argsort(axis=0).argsort(axis=0) + 1
    sota = total[1]

    seg = [lighten(OURS, f) for f in (0.72, 0.56, 0.40, 0.21, 0.0)]
    nm = len(names)
    ypos = np.arange(nm)[::-1]          # best on top

    fig = plt.figure(figsize=(7.5, 4.35))
    axb = fig.add_axes([0.148, 0.105, 0.525, 0.755])
    axr = fig.add_axes([0.735, 0.105, 0.215, 0.755])

    # ---- stacked bars ------------------------------------------------------
    axb.set_axisbelow(True)
    axb.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for k, n in enumerate(names):
        hot = n == "Ours"
        left = 0.0
        for j in range(len(TASKS)):
            wj = mat[k, j] / len(TASKS)
            axb.barh(ypos[k], wj, left=left, height=0.62 if hot else 0.56,
                     color=seg[j], edgecolor="white", lw=0.6,
                     zorder=4 if hot else 3)
            left += wj
        if hot:
            axb.add_patch(Rectangle((0, ypos[k] - 0.31), left, 0.62,
                                    fill=False, edgecolor=OURS, lw=1.4,
                                    zorder=6))
        axb.text(83.5, ypos[k], f"{total[k]:.2f}", ha="right",
                 va="center", fontsize=7.4 if hot else 7.0,
                 color=OURS if hot else "#4a4a4a",
                 fontweight="bold" if hot else "normal", zorder=6)

    axb.axvline(sota, color="#b23a48", lw=1.0, ls=(0, (4.0, 2.0)), zorder=7)
    axb.annotate(f"previous SOTA  {sota:.2f}", xy=(sota, ypos[0] + 0.52),
                 xytext=(sota - 1.2, ypos[0] + 1.05), fontsize=6.9,
                 color="#b23a48", ha="right", va="center",
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="#b23a48"))
    axb.text(83.5, ypos[0] + 1.05, f"$+${total[0] - sota:.2f}",
             fontsize=7.0, color=OURS, fontweight="bold", ha="right",
             va="center", clip_on=False)

    axb.set_yticks(ypos)
    axb.set_yticklabels(names, fontsize=7.8)
    axb.get_yticklabels()[0].set_fontweight("bold")
    axb.get_yticklabels()[0].set_color(OURS)
    axb.set_ylim(-0.75, nm - 0.25)
    axb.set_xlim(0, 84.0)
    axb.set_xticks(np.arange(0, 76, 15))
    axb.set_xlabel("Aggregate score (mean over the five benchmarks)",
                   fontsize=8.2)
    axb.tick_params(axis="y", length=0)
    style_axes(axb)

    handles = [Rectangle((0, 0), 1, 1, facecolor=seg[j], edgecolor="white",
                         lw=0.5, label=TASKS[j]) for j in range(len(TASKS))]
    leg = axb.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False,
                     fontsize=7.0, handlelength=1.5, handleheight=0.85,
                     columnspacing=1.3, handletextpad=0.5)
    leg.set_zorder(-5)

    # ---- rank block --------------------------------------------------------
    axr.set_xlim(-0.5, len(TASKS) - 0.5)
    axr.set_ylim(-0.75, nm - 0.25)
    axr.set_xticks(range(len(TASKS)))
    axr.set_xticklabels(TASK_SHORT, fontsize=6.9)
    axr.set_yticks([])
    axr.tick_params(length=0)
    axr.xaxis.set_ticks_position("top")
    for s in axr.spines.values():
        s.set_visible(False)
    for k in range(nm):
        for j in range(len(TASKS)):
            r = rank[k, j]
            f = 0.92 - 0.80 * (nm - r) / (nm - 1)
            axr.add_patch(Rectangle((j - 0.44, ypos[k] - 0.30), 0.88, 0.60,
                                    facecolor=lighten(OURS, f),
                                    edgecolor="white", lw=0.7, zorder=2))
            axr.text(j, ypos[k], str(r), ha="center", va="center",
                     fontsize=6.6, color="white" if r <= 3 else "#4a4a4a",
                     fontweight="bold" if r == 1 else "normal", zorder=3)


    for k, n in enumerate(names):
        print(f"  {n:14s} total {total[k]:6.2f}  ranks "
              f"{'  '.join(str(v) for v in rank[k])}")
    savefig(fig, "310_benchmark_leaderboard.png")


if __name__ == "__main__":
    main()
