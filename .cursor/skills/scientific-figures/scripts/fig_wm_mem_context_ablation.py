"""Figure 1197 - Context-window ablation: where each task stops paying for
more tokens.

Top panel: four tasks scored over nine context windows from 256 to 65,536
tokens on a log2 axis.  The saturation point of a task is the smallest window
whose score is within one point of that task's best score; it is found by
scanning the array that is plotted, marked on the curve, and dropped to the
axis with an explicit two-point segment (never ``axvline``, which would strike
through the tick labels).

Bottom strip: effective context utilisation.  Each bar runs from the shortest
window to that task's saturation point *on the same shared log axis*, so the
end of a bar sits exactly under the marker that produced it, and the division
that turns it into a percentage is printed at the right-hand edge.  Both
panels are driven by one ``SCORES`` array - there is no second sampling.

Colour discipline: all-cool.  Tasks are separated by depth on one blue ladder
(darker = hungrier for context) plus a marker shape and a bar hatch.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes
from figkit import background_card

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
CARD = "#e6eaee"           # luminance 0.916, clear of the 0.92-0.95 band

EXP = np.arange(8, 17)                       # 256 ... 65536
WINDOW = (2 ** EXP).astype(int)
TOL = 0.01                                   # "within one point of the best"

# name, floor score, ceiling score, log2 rise width, colour, marker, hatch
TASKS = [
    ("Needle retrieval (acc)", 0.12, 0.97, 1.60, "#8fb8dd", "o", None),
    ("Multi-hop QA (EM)", 0.08, 0.78, 2.00, "#5b93c8", "s", ".."),
    ("Episodic recall (F1)", 0.15, 0.86, 2.60, "#2f6fa8", "^", "//"),
    ("Long-horizon planning (SR)", 0.05, 0.62, 3.10, "#12314f", "D", "\\\\"),
]


def scores():
    """Nine measured points per task, plus the saturation index read back."""
    out = []
    for i, (name, lo, hi, k, col, mk, hatch) in enumerate(TASKS):
        rng = np.random.default_rng(1602 + 13 * i)
        y = hi - (hi - lo) * np.exp(-((EXP - EXP[0]) / k) ** 2)
        y = y + 0.0035 * rng.standard_normal(EXP.size)
        best = float(y.max())
        idx = int(np.argmax(y >= best - TOL))
        out.append(dict(name=name, y=y, col=col, mk=mk, hatch=hatch,
                        best=best, idx=idx, sat=int(WINDOW[idx])))
    return out


def main():
    set_pub_style()
    S = scores()
    full = int(WINDOW[-1])

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.44], hspace=0.19,
                          left=0.062, right=0.985, top=0.795, bottom=0.112)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharex=ax)

    p = axb.get_position()
    background_card(fig, (p.x0 - 0.052, p.y0 - 0.072, p.width + 0.062,
                          p.height + 0.100), facecolor=CARD, zorder=-5,
                    radius=0.008)
    axb.set_facecolor(CARD)

    ax.set_xscale("log", base=2)
    ax.set_xlim(WINDOW[0] / 1.28, full * 1.28)
    ax.set_ylim(0.0, 1.04)

    for s in S:
        ax.plot(WINDOW, s["y"], color=s["col"], lw=1.9, marker=s["mk"],
                markersize=4.0, markerfacecolor="white",
                markeredgewidth=1.2, label=s["name"], zorder=4,
                solid_capstyle="round")
    for s in S:
        xs, ys = s["sat"], s["y"][s["idx"]]
        ax.plot([xs, xs], [0.0, ys], color=s["col"], lw=0.9,
                linestyle=(0, (1.6, 1.8)), alpha=0.65, zorder=2)
        ax.plot([xs], [ys], marker=s["mk"], markersize=8.0, color=s["col"],
                markeredgecolor="white", markeredgewidth=1.2, zorder=6,
                linestyle="none")

    ax.set_ylabel("Task score")
    ax.grid(True, axis="y", color=HAIR, lw=0.55, alpha=0.85)
    ax.tick_params(labelbottom=False)
    style_axes(ax)

    # ---- utilisation strip, same x axis, same saturation indices
    axb.set_ylim(len(S) - 0.45, -0.55)
    axb.set_yticks([])
    for j, s in enumerate(S):
        axb.add_patch(Rectangle((WINDOW[0], j - 0.30), s["sat"] - WINDOW[0],
                                0.60, facecolor=s["col"], edgecolor="white",
                                linewidth=0.7, hatch=s["hatch"], zorder=3))
        axb.plot([s["sat"]], [j], marker=s["mk"], markersize=6.5,
                 color=s["col"], markeredgecolor="white", markeredgewidth=1.1,
                 zorder=5, linestyle="none", clip_on=False)
        pct = f"{100.0 * s['sat'] / full:.2f}"
        assert abs(float(pct) - 100.0 * s["sat"] / full) < 5e-3, \
            f"{s['name']}: utilisation label does not reproduce sat / full"
        # a white chip, not white type: the hatch would swallow white text
        axb.text(WINDOW[0] * 1.12, j, s["name"], ha="left", va="center",
                 fontsize=7.0, color=s["col"], fontweight="bold", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                           edgecolor="none", alpha=0.94))
        axb.text(full * 1.22, j, f"{s['sat']:,} / {full:,} = {pct} %",
                 ha="right", va="center", fontsize=6.8, color=RULE, zorder=6)

    axb.set_xlabel("Context window (tokens)")
    axb.set_xticks(WINDOW)
    axb.set_xticklabels([f"{w // 1024}k" if w >= 1024 else str(w)
                         for w in WINDOW])
    axb.tick_params(axis="x", labelsize=7.0)
    axb.grid(True, axis="x", color="#c8d0d8", lw=0.5, alpha=0.9)
    for side in ("top", "right", "left"):
        axb.spines[side].set_visible(False)

    add_panel_label(ax, "(a)", x=-0.052, y=1.020)
    add_panel_label(axb, "(b)", x=-0.052, y=1.060)

    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(TASKS), "legend entries must match curves"

    span = S[-1]["sat"] // S[0]["sat"]

    savefig(fig, "1197_wm_mem_context_ablation.png")


if __name__ == "__main__":
    main()
