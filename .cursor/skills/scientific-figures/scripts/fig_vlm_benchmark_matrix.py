"""Figure 351 - Multi-model, multi-benchmark comparison drawn as a figure.

Every cell of the table is a mini bar on a light track instead of a bare
number: bar length and cell tint encode the score within that benchmark, the
printed value keeps the exact figure.  A weight strip sits above the header,
the weighted overall score forms the right-hand column and the ours row runs
on a tinted stripe.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_palette, lighten
from fig_vlm_image_text_align import (inch_canvas, panel_tag, C_INK, C_SUB,
                                      C_FAINT, MONO)

MODELS = ["CLIP", "ALIGN", "BLIP", "BLIP-2", "Flamingo", "LLaVA",
          "InstructBLIP", "Ours"]
TASKS = ["VQAv2", "GQA", "TextVQA", "COCO-Cap", "Flickr R@1", "RefCOCO",
         "POPE"]
WEIGHTS = np.array([0.18, 0.14, 0.14, 0.16, 0.13, 0.15, 0.10])
S = np.array([
    [56.3, 41.2, 22.8, 61.4, 68.9, 45.1, 72.4],
    [58.7, 43.6, 25.1, 63.8, 71.5, 47.3, 74.0],
    [72.1, 55.4, 38.6, 78.2, 79.8, 58.6, 82.3],
    [76.4, 58.9, 42.3, 82.7, 83.1, 62.9, 85.6],
    [74.8, 56.2, 40.1, 84.3, 81.4, 59.8, 83.2],
    [78.5, 62.0, 46.7, 85.6, 82.6, 65.4, 86.9],
    [79.2, 63.4, 48.2, 86.1, 84.0, 66.8, 87.5],
    [82.6, 66.9, 52.4, 89.4, 87.2, 71.3, 90.1]])
ACCENT = "#b23a48"
RAMP = get_palette("cb_sequential_blue")
FIGSIZE = (11.0, 6.6)


def ramp_color(t):
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("r", RAMP)
    return cmap(0.28 + 0.62 * float(np.clip(t, 0, 1)))


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    ax, cw, ch = inch_canvas(fig, [0.035, 0.045, 0.945, 0.845])

    total = S @ WEIGHTS
    lo = S.min(axis=0) - 0.22 * np.ptp(S, axis=0)
    hi = S.max(axis=0) + 0.04 * np.ptp(S, axis=0)
    tnorm = (S - lo) / (hi - lo)
    tlo, thi = total.min() - 0.22 * np.ptp(total), total.max() + 0.04 * \
        np.ptp(total)
    tot_n = (total - tlo) / (thi - tlo)

    name_w, cell_w, tot_w, gap = 1.34, 0.98, 1.62, 0.22
    x_task = [name_w + i * cell_w for i in range(len(TASKS))]
    x_tot = name_w + len(TASKS) * cell_w + gap
    y_w, y_head, y0, rh = 0.30, 0.86, 1.24, 0.50

    # ---- task weight strip -----------------------------------------------
    ax.text(name_w - 0.10, y_w + 0.24, "benchmark weight", ha="right",
            va="center", fontsize=6.8, color=C_FAINT)
    for i, w in enumerate(WEIGHTS):
        bw = 0.72 * w / WEIGHTS.max()
        x = x_task[i] + cell_w / 2.0
        ax.add_patch(Rectangle((x - 0.36, y_w + 0.10), 0.72, 0.15,
                               facecolor="#ececec", edgecolor="none"))
        ax.add_patch(Rectangle((x - 0.36, y_w + 0.10), bw, 0.15,
                               facecolor="#9aa6b4", edgecolor="none"))
        ax.text(x, y_w - 0.02, f"{w:.2f}", ha="center", va="bottom",
                fontsize=6.2, color=C_SUB, **MONO)

    # ---- header ----------------------------------------------------------
    for i, t in enumerate(TASKS):
        ax.text(x_task[i] + cell_w / 2.0, y_head + 0.30, t, ha="center",
                va="center", fontsize=7.6, fontweight="bold", color=C_INK)
    ax.text(x_tot + tot_w / 2.0, y_head + 0.30, "Overall", ha="center",
            va="center", fontsize=7.6, fontweight="bold", color=C_INK)
    ax.plot([0.0, x_tot + tot_w], [y0 - 0.06] * 2, color="#8a8a8a", lw=0.9,
            zorder=2)

    # ---- rows -------------------------------------------------------------
    for r, model in enumerate(MODELS):
        y = y0 + r * rh
        ours = model == "Ours"
        if ours:
            ax.add_patch(Rectangle((0.0, y), x_tot + tot_w, rh,
                                   facecolor=lighten(ACCENT, 0.93),
                                   edgecolor="none", zorder=0))
        ax.text(name_w - 0.14, y + rh / 2.0, model, ha="right", va="center",
                fontsize=8.0, fontweight="bold" if ours else "normal",
                color=ACCENT if ours else C_INK)
        for c in range(len(TASKS)):
            t = tnorm[r, c]
            col = ACCENT if ours else ramp_color(t)
            x = x_task[c]
            ax.add_patch(Rectangle((x + 0.05, y + 0.03), cell_w - 0.10,
                                   rh - 0.06, facecolor=col,
                                   alpha=0.05 + 0.16 * t, edgecolor="none",
                                   zorder=1))
            ax.add_patch(Rectangle((x + 0.12, y + rh - 0.20), cell_w - 0.24,
                                   0.11, facecolor="#e4e4e4",
                                   edgecolor="none", zorder=2))
            ax.add_patch(Rectangle((x + 0.12, y + rh - 0.20),
                                   (cell_w - 0.24) * (0.10 + 0.90 * t), 0.11,
                                   facecolor=col, edgecolor="none", zorder=3))
            ax.text(x + cell_w / 2.0, y + 0.17, f"{S[r, c]:.1f}",
                    ha="center", va="center", fontsize=6.6,
                    color=ACCENT if ours else C_SUB,
                    fontweight="bold" if ours else "normal", **MONO)
        col = ACCENT if ours else "#4b7ba8"
        ax.add_patch(Rectangle((x_tot, y + 0.11), tot_w - 0.42, rh - 0.22,
                               facecolor="#ececec", edgecolor="none",
                               zorder=2))
        ax.add_patch(Rectangle((x_tot, y + 0.11),
                               (tot_w - 0.42) * (0.08 + 0.92 * tot_n[r]),
                               rh - 0.22, facecolor=col, edgecolor="none",
                               zorder=3))
        ax.text(x_tot + tot_w - 0.36, y + rh / 2.0, f"{total[r]:.1f}",
                ha="left", va="center", fontsize=7.4,
                color=ACCENT if ours else C_INK,
                fontweight="bold" if ours else "normal", **MONO)

    yb = y0 + len(MODELS) * rh
    ax.plot([0.0, x_tot + tot_w], [yb + 0.04] * 2, color="#8a8a8a", lw=0.9)
    ax.text(0.0, yb + 0.30,
            "bar length and cell tint encode the score within each benchmark  "
            "|  overall = weighted mean of the seven benchmarks",
            ha="left", va="center", fontsize=7.0, color=C_SUB)

    panel_tag(fig, ax, "(a)", dx=0.018)
    savefig(fig, "351_vlm_benchmark_matrix.png")


if __name__ == "__main__":
    main()
