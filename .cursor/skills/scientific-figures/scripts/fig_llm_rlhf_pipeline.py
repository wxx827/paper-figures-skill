"""Figure 317 - The three stages of reinforcement learning from human feedback.

Each stage is a card that shows what the stage actually consumes and produces:
demonstration pairs for supervised fine-tuning, ranked comparisons for the
reward model, and a sampling loop with a KL leash for PPO.  The objective that
is minimised sits at the foot of every card.  The strip along the bottom keeps
the training curves at ~50% of the card height, so it reads as supporting
evidence rather than a second row of stages.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"
LINE = "#666666"
BLOCK = "#f5f5f5"

FIG_W, FIG_H = 10.4, 6.4
MAIN = [0.040, 0.425, 0.920, 0.505]
SUB_Y, SUB_H, SUB_W = 0.135, 0.185, 0.235
BAND = (0.030, 0.058, 0.940, 0.340)
CENTRES = [0.165, 0.500, 0.835]
BOX_W, BOX_Y0, BOX_H = 0.270, 0.070, 0.800
STAGES = [
    ("1  ·  Supervised fine-tuning", BLUE),
    ("2  ·  Reward model", GREEN),
    ("3  ·  PPO with a KL leash", PURPLE),
]


def pill(ax, cx, cy, w, h, text, fc="white", ec=LINE, tc=INK, fs=6.2,
         lw=0.8, weight="normal", mono=False, z=4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=z + 1, fontweight=weight,
            fontfamily="monospace" if mono else None)


def arrow(ax, p0, p1, color=LINE, lw=1.0, style="-|>", rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8.5,
                                 lw=lw, color=color, zorder=z,
                                 shrinkA=1.0, shrinkB=1.0,
                                 connectionstyle=f"arc3,rad={rad}"))


def card(ax, cx, title, color):
    x0 = cx - BOX_W / 2
    ax.add_patch(FancyBboxPatch((x0, BOX_Y0), BOX_W, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.016",
                                facecolor="white", edgecolor="#c9c9c9",
                                linewidth=0.9, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, BOX_Y0 + BOX_H - 0.105), BOX_W, 0.105,
                                boxstyle="round,pad=0,rounding_size=0.016",
                                facecolor=color, edgecolor=color,
                                linewidth=0.9, zorder=3))
    ax.add_patch(Rectangle((x0, BOX_Y0 + BOX_H - 0.115), BOX_W, 0.03,
                           facecolor=color, edgecolor="none", zorder=3))
    ax.text(cx, BOX_Y0 + BOX_H - 0.052, title, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="white", zorder=5)
    return x0


def objective(ax, cx, text, note):
    ax.add_patch(FancyBboxPatch((cx - BOX_W / 2 + 0.012, BOX_Y0 + 0.030),
                                BOX_W - 0.024, 0.115,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                facecolor=BLOCK, edgecolor="none", zorder=3))
    ax.text(cx, BOX_Y0 + 0.112, text, ha="center", va="center", fontsize=7.4,
            color=INK, zorder=5)
    ax.text(cx, BOX_Y0 + 0.058, note, ha="center", va="center", fontsize=6.4,
            color=SUB, zorder=5)


def stage_sft(ax, cx):
    for k, (row, lab) in enumerate(zip((0.690, 0.565, 0.440),
                                       ("cooking", "history", "code"))):
        pill(ax, cx - 0.070, row, 0.105, 0.070, "prompt", fc="#ececec",
             ec="#bdbdbd", tc=SUB)
        arrow(ax, (cx - 0.014, row), (cx + 0.012, row), lw=0.8)
        pill(ax, cx + 0.075, row, 0.115, 0.070, "human demo", fc="#e3ebf5",
             ec=BLUE, tc=BLUE)
        ax.text(cx - 0.130, row, lab, ha="right", va="center", fontsize=6.0,
                color=FAINT, zorder=5)
    ax.text(cx, 0.300, "12k demonstrations  ·  3 epochs", ha="center",
            va="center", fontsize=6.6, color=SUB, zorder=5)
    objective(ax, cx, r"$\max_{\theta}\ \log \pi_{\theta}(y \mid x)$",
              "next-token loss on the human answer only")


def stage_rm(ax, cx):
    pill(ax, cx, 0.710, 0.110, 0.068, "prompt", fc="#ececec", ec="#bdbdbd",
         tc=SUB)
    arrow(ax, (cx - 0.020, 0.675), (cx - 0.070, 0.625), lw=0.8, rad=0.10)
    arrow(ax, (cx + 0.020, 0.675), (cx + 0.070, 0.625), lw=0.8, rad=-0.10)
    for dx, tag, score, win in ((-0.070, "y$_w$", 0.82, True),
                                (0.070, "y$_l$", 0.31, False)):
        pill(ax, cx + dx, 0.585, 0.100, 0.070, f"answer {tag}",
             fc="#e4efe9" if win else "white",
             ec=ACCENT if win else "#bdbdbd", tc=GREEN if win else SUB,
             lw=1.2 if win else 0.8, weight="bold" if win else "normal")
        ax.add_patch(Rectangle((cx + dx - 0.048, 0.470), 0.096 * score, 0.036,
                               facecolor=ACCENT if win else "#c2c2c2",
                               edgecolor="none", zorder=4))
        ax.add_patch(Rectangle((cx + dx - 0.048, 0.470), 0.096, 0.036,
                               facecolor="none", edgecolor="#cccccc",
                               linewidth=0.6, zorder=4))
        ax.text(cx + dx, 0.440, f"$r_\\phi$ = {score:.2f}", ha="center",
                va="top", fontsize=6.2, color=INK, zorder=5,
                fontweight="bold" if win else "normal")
    ax.text(cx, 0.488, r"$\succ$", ha="center", va="center", fontsize=9.0,
            color=ACCENT, fontweight="bold", zorder=6)
    ax.text(cx, 0.300, "48k pairwise comparisons  ·  frozen after training",
            ha="center", va="center", fontsize=6.6, color=SUB, zorder=5)
    objective(ax, cx,
              r"$\min_{\phi}\ -\log \sigma\!\left(r_{\phi}(y_w)-"
              r"r_{\phi}(y_l)\right)$",
              "Bradley-Terry loss on human rankings")


def stage_ppo(ax, cx):
    xl, xr = cx - 0.068, cx + 0.070
    pill(ax, xl, 0.700, 0.100, 0.068, r"policy $\pi_{\theta}$",
         fc="#efeaf7", ec=PURPLE, tc=PURPLE, weight="bold")
    pill(ax, xr, 0.700, 0.100, 0.068, "sample $y$", fc="white",
         ec="#bdbdbd", tc=SUB)
    pill(ax, xr, 0.562, 0.100, 0.068, r"$r_{\phi}$ (frozen)",
         fc="#e4efe9", ec=GREEN, tc=GREEN)
    pill(ax, xl, 0.562, 0.100, 0.068, r"$r_{\phi}-\beta\,\mathrm{KL}$",
         fc="white", ec=ACCENT, tc=ACCENT, weight="bold")
    arrow(ax, (xl + 0.052, 0.700), (xr - 0.052, 0.700), lw=0.9)
    arrow(ax, (xr, 0.664), (xr, 0.600), lw=0.9)
    arrow(ax, (xr - 0.052, 0.562), (xl + 0.052, 0.562), lw=0.9)
    arrow(ax, (xl, 0.600), (xl, 0.664), color=PURPLE, lw=1.2)
    ax.text(cx, 0.470, "one PPO iteration", ha="center", va="center",
            fontsize=6.6, color=PURPLE, fontweight="bold", zorder=6)
    ax.text(cx, 0.412, r"KL is measured against the frozen $\pi_{\mathrm{ref}}$",
            ha="center", va="center", fontsize=6.2, color=SUB, zorder=6)
    ax.text(cx, 0.300, "64k prompts  ·  $\\beta$ = 0.02  ·  4 epochs",
            ha="center", va="center", fontsize=6.6, color=SUB, zorder=5)
    objective(ax, cx,
              r"$\max_{\theta}\ \mathbb{E}\left[r_{\phi}(y)-\beta\,"
              r"\mathrm{KL}\right]$",
              "reward minus drift from the SFT policy")


def curves(fig):
    rng = np.random.default_rng(317)
    axes = []
    x = np.linspace(0, 1, 160)

    for k, cx in enumerate(CENTRES):
        rect = [MAIN[0] + MAIN[2] * cx - SUB_W / 2, SUB_Y, SUB_W, SUB_H]
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        n = rng.standard_normal(x.size) * 0.012
        if k == 0:
            y = 1.05 + 1.35 * np.exp(-4.2 * x) + n
            ax.plot(x * 3.0, y, color=BLUE, lw=1.4)
            ax.set_ylabel("SFT loss (nats)", fontsize=6.8, labelpad=2.0)
            ax.set_xlabel("epoch", fontsize=6.8, labelpad=1.5)
            ax.set_xlim(0, 3)
            ax.set_ylim(0.9, 2.6)
            ax.set_title("Cross-entropy on held-out demos", fontsize=7.5,
                         fontweight="bold", pad=3.0)
        elif k == 1:
            y = 0.505 + 0.215 * (1 - np.exp(-5.0 * x)) + n * 1.6
            ax.plot(x * 6.0, y, color=GREEN, lw=1.4)
            ax.axhline(0.5, color=FAINT, ls=(0, (3, 2)), lw=0.8)
            ax.text(5.9, 0.508, "chance", fontsize=6.0, color=FAINT,
                    ha="right", va="bottom")
            ax.set_ylabel("pair accuracy (-)", fontsize=6.8, labelpad=2.0)
            ax.set_xlabel("gradient steps ($10^3$)", fontsize=6.8, labelpad=1.5)
            ax.set_xlim(0, 6)
            ax.set_ylim(0.46, 0.80)
            ax.set_title("Agreement with held-out humans", fontsize=7.5,
                         fontweight="bold", pad=3.0)
        else:
            rw = 0.10 + 1.28 * (1 - np.exp(-3.1 * x)) + n * 1.4
            kl = 14.0 * x ** 1.7 + n * 12
            ax.plot(x * 400, rw, color=PURPLE, lw=1.4, zorder=4)
            ax.set_ylabel("mean reward (-)", fontsize=6.8, labelpad=2.0,
                          color=PURPLE)
            ax.set_xlabel("PPO steps", fontsize=6.8, labelpad=1.5)
            ax.set_xlim(0, 400)
            ax.set_ylim(0, 1.6)
            ax.tick_params(axis="y", colors=PURPLE)
            ax2 = ax.twinx()
            ax2.plot(x * 400, kl, color=ACCENT, lw=1.1, ls=(0, (3.5, 2)),
                     zorder=3)
            ax2.set_ylabel("KL to $\\pi_{\\mathrm{ref}}$ (nats)", fontsize=6.8,
                           labelpad=2.0, color=ACCENT)
            ax2.set_ylim(0, 22)
            ax2.tick_params(axis="y", colors=ACCENT, labelsize=6.2,
                            length=2.2)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color(ACCENT)
            ax2.spines["right"].set_linewidth(0.8)
            ax2.set_facecolor("none")
            ax.set_title("Reward rises, KL must not", fontsize=7.5,
                         fontweight="bold", pad=3.0)
        ax.tick_params(labelsize=6.2, length=2.2)
        ax.grid(True, axis="y", lw=0.5, color="#e4e4e4")
        ax.set_axisbelow(True)
        style_axes(ax)
        if k == 2:
            ax.spines["right"].set_visible(True)
        axes.append(ax)
    return axes


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    fig.patches.append(Rectangle(BAND[:2], BAND[2], BAND[3],
                                 transform=fig.transFigure, facecolor=BLOCK,
                                 edgecolor="none", zorder=-1))

    ax = fig.add_axes(MAIN)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for cx, (title, color) in zip(CENTRES, STAGES):
        card(ax, cx, title, color)
    stage_sft(ax, CENTRES[0])
    stage_rm(ax, CENTRES[1])
    stage_ppo(ax, CENTRES[2])

    for a, b, lab in ((0, 1, r"$\pi^{\mathrm{SFT}}$"), (1, 2, r"$r_{\phi}$")):
        x0 = CENTRES[a] + BOX_W / 2
        x1 = CENTRES[b] - BOX_W / 2
        arrow(ax, (x0 + 0.006, 0.470), (x1 - 0.006, 0.470), color=LINE,
              lw=1.3)
        ax.text((x0 + x1) / 2, 0.492, lab, ha="center", va="bottom",
                fontsize=7.4, color=SUB, zorder=6)

    axes = curves(fig)
    fig.text(0.5, 0.368,
             "Training signal produced by each stage",
             fontsize=7.5, fontweight="bold", color=INK, ha="center",
             va="bottom")

    for tag, target in zip("abc", CENTRES):
        fig.text(MAIN[0] + MAIN[2] * (target - BOX_W / 2) - 0.30 / FIG_W,
                 MAIN[1] + MAIN[3] * (BOX_Y0 + BOX_H) + 0.06 / FIG_H,
                 f"({tag})", fontsize=11, fontweight="bold", ha="left",
                 va="bottom")

    savefig(fig, "317_llm_rlhf_pipeline.png")


if __name__ == "__main__":
    main()
