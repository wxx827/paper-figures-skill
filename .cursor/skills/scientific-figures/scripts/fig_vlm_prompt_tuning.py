"""Figure 350 - Prompt tuning strategies across eight benchmarks.

Main panel: accuracy of six prompting strategies on eight datasets, printed as
raw accuracy and shaded by the gain over the hand-crafted prompt so the colour
compares methods rather than dataset difficulty.  Right: few-shot scaling.
Subordinate row: how many learnable prompt tokens are worth paying for, and
the average accuracy per strategy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap
from fig_vlm_image_text_align import panel_tag, bg_block, C_INK, C_SUB, C_FAINT

STRATEGIES = ["Hand-crafted prompt", "80-template ensemble", "CoOp (M=16)",
              "CoCoOp", "VPT-deep", "Ours: text + visual"]
DATASETS = ["ImageNet", "Caltech101", "OxfordPets", "StanfordCars",
            "Flowers102", "Food101", "Aircraft", "DTD"]
ACC = np.array([
    [66.7, 92.9, 89.1, 65.3, 71.3, 86.1, 24.8, 44.5],
    [68.3, 93.6, 89.8, 66.1, 72.5, 86.8, 26.4, 45.9],
    [71.5, 95.6, 91.9, 78.1, 94.5, 86.5, 38.4, 62.5],
    [71.0, 94.9, 93.3, 71.6, 87.8, 87.2, 33.1, 58.0],
    [70.6, 94.4, 91.2, 74.5, 90.1, 86.9, 35.8, 60.3],
    [73.4, 96.3, 94.1, 80.2, 95.6, 88.0, 41.2, 64.8]])
SHOTS = [0, 1, 2, 4, 8, 16]
CURVES = {
    "Hand-crafted": ([58.1] * 6, "#808080"),
    "CoOp": ([58.1, 59.6, 62.4, 66.8, 70.9, 74.9], "#3d6b8e"),
    "CoCoOp": ([58.1, 62.7, 65.1, 68.0, 70.4, 72.6], "#c9a227"),
    "Ours": ([58.1, 64.9, 68.3, 71.9, 75.2, 79.2], "#b23a48"),
}
NTOK = [1, 2, 4, 8, 16, 32]
TOK_ACC = [70.6, 72.8, 75.4, 77.6, 79.2, 78.4]
ACCENT = "#b23a48"


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(10.4, 6.4))
    bg_block(fig, 0.020, 0.030, 0.980, 0.335)
    gain = ACC - ACC[0]

    # ---- main matrix ------------------------------------------------------
    axm = fig.add_axes([0.105, 0.420, 0.465, 0.460])
    im = axm.imshow(gain, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                    vmax=25.0, aspect="auto", interpolation="nearest")
    for i in range(ACC.shape[0]):
        for j in range(ACC.shape[1]):
            axm.text(j, i, f"{ACC[i, j]:.1f}", ha="center", va="center",
                     fontsize=6.4,
                     color="white" if gain[i, j] > 13 else C_INK,
                     fontweight="bold" if i == 5 else "normal")
    axm.set_xticks(range(len(DATASETS)))
    axm.set_xticklabels(DATASETS, fontsize=7.0, rotation=30, ha="right")
    axm.set_yticks(range(len(STRATEGIES)))
    axm.set_yticklabels(STRATEGIES, fontsize=7.2)
    axm.get_yticklabels()[-1].set_fontweight("bold")
    axm.get_yticklabels()[-1].set_color(ACCENT)
    axm.tick_params(length=1.8, width=0.6)
    axm.add_patch(Rectangle((-0.5, 4.5), len(DATASETS), 1.0, facecolor="none",
                            edgecolor=ACCENT, linewidth=1.6, zorder=6))
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#b8b8b8")
    axm.set_title("Top-1 accuracy (%), shaded by gain over the hand-crafted "
                  "prompt", fontsize=9.0, fontweight="bold", pad=4.0)

    cax = fig.add_axes([0.583, 0.420, 0.011, 0.460])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("gain over hand-crafted prompt (pp)", fontsize=7.0,
                 color=C_INK)

    # ---- few-shot scaling -------------------------------------------------
    axs = fig.add_axes([0.690, 0.440, 0.275, 0.440])
    xs = np.arange(len(SHOTS))
    for name, (vals, col) in CURVES.items():
        ours = name == "Ours"
        axs.plot(xs, vals, color=col, lw=1.9 if ours else 1.3,
                 marker="o" if ours else "s", ms=4.0 if ours else 3.0,
                 mfc="white", mew=1.1, zorder=4 if ours else 3,
                 ls="-" if name != "Hand-crafted" else (0, (3, 2)))
        axs.text(xs[-1] + 0.12, vals[-1], name, fontsize=6.8, color=col,
                 va="center", ha="left",
                 fontweight="bold" if ours else "normal")
    axs.set_xticks(xs)
    axs.set_xticklabels([str(s) for s in SHOTS], fontsize=7.0)
    axs.set_xlim(-0.25, len(SHOTS) + 1.45)
    axs.set_ylim(54, 84)
    axs.set_xlabel("labelled shots per class (-)", fontsize=8.0)
    axs.set_ylabel("average accuracy (%)", fontsize=8.0)
    axs.tick_params(labelsize=7.0, length=2.4, width=0.6)
    axs.grid(axis="y", color="#e2e2e2", lw=0.6, zorder=0)
    axs.set_title("Few-shot scaling, mean over 8 datasets", fontsize=9.0,
                  fontweight="bold", pad=4.0)
    axs.spines["left"].set_linewidth(0.6)
    axs.spines["bottom"].set_linewidth(0.6)

    # ---- subordinate: prompt length --------------------------------------
    axt = fig.add_axes([0.105, 0.095, 0.290, 0.175])
    axt.set_facecolor("none")
    axt.plot(range(len(NTOK)), TOK_ACC, color=ACCENT, lw=1.5, marker="o",
             ms=3.6, mfc="white", mew=1.0)
    best = int(np.argmax(TOK_ACC))
    axt.plot([best], [TOK_ACC[best]], marker="o", ms=6.0, mfc=ACCENT,
             mec="white", mew=0.9, zorder=5)
    axt.annotate(f"best  M = {NTOK[best]}", (best, TOK_ACC[best]),
                 textcoords="offset points", xytext=(4, -14), fontsize=6.4,
                 color=ACCENT, ha="left")
    axt.set_xticks(range(len(NTOK)))
    axt.set_xticklabels([str(n) for n in NTOK], fontsize=6.5)
    axt.set_ylim(68, 82)
    axt.set_xlabel("learnable prompt tokens M (-)", fontsize=7.0)
    axt.set_ylabel("accuracy (%)", fontsize=7.0)
    axt.tick_params(labelsize=6.5, length=2.0, width=0.6)
    axt.set_title("Prompt length ablation", fontsize=8.5, fontweight="bold",
                  pad=3.5)
    axt.spines["left"].set_linewidth(0.6)
    axt.spines["bottom"].set_linewidth(0.6)

    # ---- subordinate: mean accuracy per strategy --------------------------
    axb = fig.add_axes([0.575, 0.095, 0.290, 0.175])
    axb.set_facecolor("none")
    means = ACC.mean(axis=1)
    ypos = np.arange(len(STRATEGIES))[::-1]
    cols = ["#b8b8b8"] * 5 + [ACCENT]
    axb.barh(ypos, means, height=0.62, color=cols, edgecolor="white",
             linewidth=0.5)
    for y, v, c in zip(ypos, means, cols):
        axb.text(v + 0.7, y, f"{v:.1f}", va="center", ha="left", fontsize=6.2,
                 color=C_SUB if c != ACCENT else ACCENT)
    axb.set_yticks(ypos)
    axb.set_yticklabels(STRATEGIES, fontsize=6.5)
    axb.get_yticklabels()[-1].set_fontweight("bold")
    axb.get_yticklabels()[-1].set_color(ACCENT)
    axb.set_xlim(0, 92)
    axb.set_xlabel("mean accuracy over 8 datasets (%)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.0, width=0.6)
    axb.set_title("Average across benchmarks", fontsize=8.5,
                  fontweight="bold", pad=3.5)
    axb.spines["left"].set_linewidth(0.6)
    axb.spines["bottom"].set_linewidth(0.6)

    for a, lab in ((axm, "(a)"), (axs, "(b)"), (axt, "(c)"), (axb, "(d)")):
        panel_tag(fig, a, lab, dx=0.075 if a is axm else 0.055)
    savefig(fig, "350_vlm_prompt_tuning.png")


if __name__ == "__main__":
    main()
