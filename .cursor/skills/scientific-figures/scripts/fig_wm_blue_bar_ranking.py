"""Wm blue bar ranking: ranked object-category frequencies on a blue value ramp.

Top-28 object categories of a household world-model benchmark, sorted by
instance count.  Bar colour is driven by the count itself (deep blue = frequent,
pale blue = rare) so the ranking is readable even in a thumbnail, and every bar
carries its count just past its right end.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes

# Batch-wide blue ladder (BRIEF_WM_BLUE.md section 2).
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
# Ramp stops used for the bars: the two palest steps are dropped so that even
# the shortest bar keeps a visible body against white paper.
BAR_RAMP = BLUE[5::-1]

TOTAL_CLASSES = 112     # full taxonomy; only the head of it is plotted

CATEGORIES = [
    "chair", "bowl", "window", "table", "cabinet", "knife", "picture",
    "desk", "television", "sink", "shelves", "tomato", "curtain", "sofa",
    "lamp", "pan", "lettuce", "mug", "potato", "bed", "lettuce slice",
    "mirror", "pillow", "box", "bag", "door", "toilet", "counter",
]


def gen_data(seed=2516):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_bar_ranking.csv")
        return df["category"].tolist(), df["count"].to_numpy()
    """
    rng = np.random.default_rng(seed)
    rank = np.arange(1, len(CATEGORIES) + 1)
    # Zipf-like decay with a heavy head, plus mild per-category jitter.
    counts = 488.0 / rank ** 0.46 * rng.normal(1.0, 0.045, rank.size)
    counts = np.maximum(counts, 96.0)
    counts = np.sort(np.round(counts).astype(int))[::-1]
    return CATEGORIES, counts


def main():
    set_pub_style()
    names, counts = gen_data()
    n = len(names)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("wm_blue", BAR_RAMP)
    lo, hi = counts.min(), counts.max()
    # The counts follow a Zipf head, so a linear value->colour map would leave
    # two thirds of the bars in one indistinguishable pale step.  Ranking is
    # monotone in the count, so it keeps "darker = more frequent" while
    # spending the whole ladder on the visible range.
    shades = [cmap(1.0 - i / (n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(n)
    ax.barh(y, counts, height=0.72, color=shades, edgecolor="white",
            linewidth=0.45, zorder=3)

    pad = hi * 0.012
    for yi, (c, sh) in enumerate(zip(counts, shades)):
        ax.text(c + pad, yi, f"{c:,}", va="center", ha="left", fontsize=6.2,
                color=INK, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.0)
    ax.set_ylim(n - 0.45, -0.75)
    ax.set_xlim(0, hi * 1.10)
    ax.set_xlabel("Object instances in the annotated split (count)",
                  fontsize=8.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, color="#e2e6ea", zorder=0)
    style_axes(ax)

    key = ax.inset_axes([0.56, 0.055, 0.33, 0.020])
    key.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap,
               vmin=0, vmax=1)
    key.set_xticks([])
    key.set_yticks([])
    for sp in key.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#9aa3ab")
    key.set_zorder(7)
    key.text(-0.03, 0.5, "rare", transform=key.transAxes, ha="right",
             va="center", fontsize=6.2, color="#555555")
    key.text(1.03, 0.5, "frequent", transform=key.transAxes, ha="left",
             va="center", fontsize=6.2, color="#555555")

    ax.set_title(f"Object-category frequency, top {n} of {TOTAL_CLASSES} "
                 "classes", fontsize=9.0, fontweight="bold", loc="left",
                 pad=12)
    ax.text(1.0, 1.012,
            f"{n} shown  |  {counts.sum():,} instances  |  "
            f"head/tail ratio {hi / lo:.1f}x",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#555555")

    fig.tight_layout()
    savefig(fig, "1111_wm_blue_bar_ranking.png")


if __name__ == "__main__":
    main()
