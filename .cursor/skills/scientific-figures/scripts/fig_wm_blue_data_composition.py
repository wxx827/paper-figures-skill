"""Wm blue data composition: nested donut plus stacked bars for the data mix.

Left: two concentric rings give the source mix of the two pre-training stages.
Right: the same five sources broken down across every split of the corpus as
stacked percentage bars.  Both panels read off one blue ladder in one fixed
source order, so a single legend serves the whole figure.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"

SOURCES = ["Web image-text", "Video captions", "Simulator rollouts",
           "Human teleop demos", "Synthetic QA"]
SOURCE_COLORS = [BLUE[1], BLUE[2], BLUE[3], BLUE[4], BLUE[5]]
# The pooled tail slice: the palest rung of the ladder pulled toward grey, so
# it reads as "not one of the five" without leaving the family.
OTHER = "Other sources"
OTHER_C = "#d9dee2"
POOL_PCT = 6.0          # a ring slice below this cannot carry a label
RING_LABEL_MIN = 4.0    # smallest ring slice that still gets a printed %
BAR_LABEL_MIN = 6.0     # same, for a stacked bar segment

STAGES = ["Stage 1 pretraining", "Stage 2 instruction tuning"]
SPLITS = ["Stage 1 pretrain", "Stage 2 tuning", "Task fine-tune",
          "Human-verified subset", "Held-out eval"]


def gen_data(seed=2521):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_data_composition.csv")
        return df.pivot(index="split", columns="source", values="samples")
    """
    rng = np.random.default_rng(seed)
    # Rows follow SPLITS, columns follow SOURCES; values are sample counts (k).
    mix = np.array([
        [5820.0, 4310.0, 1180.0, 210.0, 640.0],
        [ 430.0, 1260.0, 2140.0, 980.0, 1520.0],
        [  90.0,  240.0, 1460.0, 1310.0, 520.0],
        [  20.0,   80.0,  310.0,  820.0, 120.0],
        [  40.0,  110.0,  380.0,  260.0, 150.0],
    ])
    return mix * rng.normal(1.0, 0.02, mix.shape)


def _text_on(color):
    """Ink or white, whichever survives on this rung of the ladder."""
    r, g, b = mpl.colors.to_rgb(color)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.58 else INK


def pool_tail(vals):
    """Fold the smallest sources into one slice big enough to label.

    Sources are absorbed smallest-first, and only while they are individually
    under ``POOL_PCT``; the sweep stops as soon as the pooled slice clears that
    same threshold.  Survivors keep their position in the canonical source
    order, so ring and bar colours never disagree.
    """
    pct = vals / vals.sum() * 100.0
    pooled, acc = [], 0.0
    for i in np.argsort(pct):
        if pct[i] >= POOL_PCT:
            break
        pooled.append(int(i))
        acc += pct[i]
        if acc >= POOL_PCT:
            break
    if len(pooled) < 2:            # nothing worth pooling
        return list(vals), list(SOURCE_COLORS)
    keep = [i for i in range(len(vals)) if i not in pooled]
    return ([vals[i] for i in keep] + [sum(vals[i] for i in pooled)],
            [SOURCE_COLORS[i] for i in keep] + [OTHER_C])


def donut(ax, mix):
    outer, inner = mix[0], mix[1]
    for vals, radius, width in [(outer, 1.00, 0.32), (inner, 0.64, 0.28)]:
        v, cols = pool_tail(vals)
        share = np.asarray(v) / sum(v) * 100.0
        _, _, autotexts = ax.pie(
            v, radius=radius, startangle=90, counterclock=False, colors=cols,
            autopct="%1.0f%%", pctdistance=1 - width / 2 / radius,
            textprops=dict(fontsize=6.6),
            wedgeprops=dict(width=width, edgecolor="white", linewidth=0.9))
        for t, col, s in zip(autotexts, cols, share):
            t.set_color(_text_on(col))
            t.set_visible(s >= RING_LABEL_MIN)

    ax.text(0, 0, f"{mix[:2].sum() / 1000:.1f}M\nsamples", ha="center",
            va="center", fontsize=8.5, color=INK, linespacing=1.3)
    # A circle leaves its bounding box's corners empty; the stage captions go
    # there instead of above and below, which is what used to strand a band of
    # white under the rings.
    ax.text(-1.30, 1.22, f"{STAGES[0]}\nouter ring  ·  "
            f"{outer.sum() / 1000:.1f}M", ha="left", va="top", fontsize=7.5,
            color=INK, linespacing=1.45)
    ax.text(1.30, -1.22, f"{STAGES[1]}\ninner ring  ·  "
            f"{inner.sum() / 1000:.1f}M", ha="right", va="bottom",
            fontsize=7.5, color=INK, linespacing=1.45)
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    # Anchor north so the shrunken square box keeps its top edge -- otherwise
    # the two panel titles end up at different heights.
    ax.set_anchor("N")
    ax.set_title("Two-stage training mix", fontsize=9.0, fontweight="bold",
                 pad=10)


def stacked(ax, mix):
    pct = mix / mix.sum(axis=1, keepdims=True) * 100.0
    y = np.arange(len(SPLITS))
    left = np.zeros(len(SPLITS))
    for s in range(len(SOURCES)):
        ax.barh(y, pct[:, s], left=left, height=0.58,
                color=SOURCE_COLORS[s], edgecolor="white", linewidth=0.8,
                zorder=3)
        for i, (l, w) in enumerate(zip(left, pct[:, s])):
            if w >= BAR_LABEL_MIN:
                ax.text(l + w / 2, y[i], f"{w:.0f}%", ha="center",
                        va="center", fontsize=6.2,
                        color=_text_on(SOURCE_COLORS[s]), zorder=5)
        left += pct[:, s]

    for i, tot in enumerate(mix.sum(axis=1)):
        ax.text(101.4, y[i], f"{tot / 1000:.2f}M", ha="left", va="center",
                fontsize=6.5, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(SPLITS, fontsize=8.0)
    ax.set_ylim(len(SPLITS) - 0.45, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7.5)
    ax.set_xlabel("Share of the split (%)", fontsize=8.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, color="#e2e6ea", zorder=0)
    style_axes(ax)
    ax.set_title("Source breakdown per split", fontsize=9.0,
                 fontweight="bold", pad=10)


def main():
    set_pub_style()
    mix = gen_data()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.4),
                             gridspec_kw=dict(width_ratios=[1.18, 1.25]))
    donut(axes[0], mix)
    stacked(axes[1], mix)
    # tight_layout cannot reason about the equal-aspect pie, so the margins
    # are set by hand: the bottom band holds the shared legend and footnote.
    fig.subplots_adjust(left=0.035, right=0.955, top=0.935, bottom=0.175,
                        wspace=0.20)

    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.8, label=s)
               for s, c in zip(SOURCES + [OTHER], SOURCE_COLORS + [OTHER_C])]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.048), columnspacing=1.8,
               handlelength=1.6)
    add_panel_label(axes[0], "(a)", x=-0.09, y=1.015, size=11)
    add_panel_label(axes[1], "(b)", x=-0.24, y=1.015, size=11)
    savefig(fig, "1116_wm_blue_data_composition.png")


if __name__ == "__main__":
    main()
