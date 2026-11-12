"""Figure 318 - Needle-in-a-haystack retrieval across the context window.

The heatmap is the standard probe: a fact is planted at a given relative depth
inside a haystack of a given length, and the model is asked to recall it.  The
failure is not uniform - accuracy collapses first for needles buried in the
middle of long contexts, which is why the dark band creeps in from the right.
The curve panel condenses each column to a mean, and the ruler underneath
places the advertised window of four models on exactly the same axis, so the
gap between what is claimed and what still works is directly measurable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from pubstyle import set_pub_style, savefig, style_axes, get_cmap

GREY = "#808080"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"

FIG_W, FIG_H = 10.4, 6.4
HEAT = [0.055, 0.320, 0.505, 0.560]
RULER = [0.055, 0.150, 0.505, 0.100]
CBAR = [0.578, 0.320, 0.012, 0.560]
RIGHT = [0.715, 0.320, 0.250, 0.560]

LEN_K = np.array([2, 4, 8, 16, 32, 64, 128, 256])
DEPTH = np.arange(0, 101, 10)
BASE = np.array([99.5, 99.0, 98.2, 96.5, 92.0, 85.0, 71.0, 52.0])
DIP = np.array([1.0, 2.0, 4.0, 8.0, 15.0, 24.0, 33.0, 38.0])

MODELS = [
    ("Base-8k", GREY, 8, np.array([99, 98, 96, 61, 24, 9, 4, 2])),
    ("LongRoPE-32k", BLUE, 32, np.array([99, 99, 98, 95, 88, 47, 18, 7])),
    ("YaRN-128k", GREEN, 128, np.array([99, 99, 98, 96, 92, 84, 62, 26])),
    ("Ours-256k", ACCENT, 256, None),
]


def pos(length_k):
    """Map a context length in k tokens onto the shared column axis."""
    return np.log2(np.asarray(length_k, dtype=float) / 2.0)


def grid(rng):
    d = DEPTH[:, None]
    acc = (BASE[None, :] - DIP[None, :]
           * np.exp(-((d - 58.0) / 26.0) ** 2)
           - 0.06 * DIP[None, :] * np.abs(d - 50) / 50.0
           + rng.normal(0, 1.6, (DEPTH.size, LEN_K.size)))
    return np.clip(acc, 1.0, 100.0)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(318)
    acc = grid(rng)
    col_mean = acc.mean(axis=0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cmap = get_cmap("cb_diverging", reverse=True)
    norm = TwoSlopeNorm(vcenter=50.0, vmin=0.0, vmax=100.0)

    # ---- main heatmap ----------------------------------------------------
    ax = fig.add_axes(HEAT)
    ax.imshow(acc, cmap=cmap, norm=norm, origin="lower", aspect="auto",
              interpolation="nearest",
              extent=[-0.5, LEN_K.size - 0.5, -5.0, 105.0])
    for g in np.arange(0.5, LEN_K.size - 1, 1.0):
        ax.axvline(g, color="white", lw=0.5, alpha=0.6)
    for g in DEPTH[:-1] + 5:
        ax.axhline(g, color="white", lw=0.4, alpha=0.45)

    eff = float(np.interp(85.0, col_mean[::-1], pos(LEN_K)[::-1]))
    ax.axvline(eff, color="#1a1a1a", ls=(0, (3.5, 2)), lw=1.2, zorder=5)
    ax.text(eff - 0.10, 101.0, f"effective window {2 ** (eff + 1):.0f}k",
            fontsize=6.8, color=INK, fontweight="bold", ha="right", va="top",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cfcfcf", linewidth=0.5, alpha=0.94))

    for j in range(LEN_K.size):
        for i in range(DEPTH.size):
            if acc[i, j] < 35:
                ax.text(j, DEPTH[i], f"{acc[i, j]:.0f}", ha="center",
                        va="center", fontsize=6.0, zorder=4, color="white")

    ax.set_xticks(range(LEN_K.size))
    ax.set_xticklabels([])
    ax.set_yticks(DEPTH[::2])
    ax.set_yticklabels([f"{d}" for d in DEPTH[::2]], fontsize=7.0)
    ax.set_ylabel("needle depth in the haystack (%)", fontsize=8.0)
    ax.set_xlim(-0.5, LEN_K.size - 0.5)
    ax.set_ylim(-5.0, 105.0)
    ax.tick_params(length=2.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_title("Ours-256k  ·  retrieval accuracy per (length, depth) cell",
                 fontsize=8.5, fontweight="bold", pad=4.0)
    fig.text(HEAT[0] - 0.34 / FIG_W, HEAT[1] + HEAT[3] + 0.06 / FIG_H, "(a)",
             fontsize=11, fontweight="bold", ha="left", va="bottom")

    cax = fig.add_axes(CBAR)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("retrieval accuracy (%)", fontsize=8.0, color=INK,
                 labelpad=3.0)

    # ---- advertised-window ruler ----------------------------------------
    axr = fig.add_axes(RULER)
    axr.set_xlim(-0.5, LEN_K.size - 0.5)
    axr.set_ylim(-0.7, len(MODELS) - 0.3)
    for k, (name, col, win, _) in enumerate(MODELS):
        y = len(MODELS) - 1 - k
        x1 = pos(win)
        axr.plot([-0.5, x1], [y, y], color=col, lw=2.6,
                 solid_capstyle="butt", zorder=3)
        axr.plot([x1], [y], marker="|", ms=6.5, color=col, mew=1.4, zorder=4)
        axr.text(x1 + 0.10, y, f"{name}  ({win}k claimed)", fontsize=6.4,
                 color=col, va="center", ha="left",
                 fontweight="bold" if name.startswith("Ours") else "normal")
    axr.set_yticks([])
    axr.set_xticks(range(LEN_K.size))
    axr.set_xticklabels([f"{v}k" for v in LEN_K], fontsize=7.0)
    axr.set_xlabel("context length (tokens)", fontsize=8.0, labelpad=2.0)
    axr.tick_params(length=2.5)
    axr.spines["left"].set_visible(False)
    style_axes(axr)
    axr.text(0.0, 1.06, "Advertised context window", transform=axr.transAxes,
             fontsize=7.5, fontweight="bold", color=INK, ha="left",
             va="bottom")
    fig.text(RULER[0] - 0.34 / FIG_W, RULER[1] + RULER[3] + 0.10 / FIG_H,
             "(b)", fontsize=11, fontweight="bold", ha="left", va="bottom")

    # ---- depth-averaged curves ------------------------------------------
    ax2 = fig.add_axes(RIGHT)
    x = pos(LEN_K)
    for name, col, win, series in MODELS:
        y = col_mean if series is None else series.astype(float)
        lead = name.startswith("Ours")
        ax2.plot(x, y, color=col, lw=1.9 if lead else 1.2, marker="o",
                 ms=3.2 if lead else 2.4, mfc="white", mew=0.9,
                 zorder=5 if lead else 3, label=name)
        ax2.axvline(pos(win), color=col, ls=(0, (2, 2.2)), lw=0.8, alpha=0.55,
                    zorder=1)
    ax2.axhline(85.0, color=SUB, ls=(0, (4, 2.5)), lw=0.9, zorder=2)
    ax2.text(-0.4, 86.5, "85% usability threshold", fontsize=6.5, color=SUB,
             ha="left", va="bottom")
    ax2.set_xlim(-0.5, LEN_K.size - 0.5)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(range(LEN_K.size))
    ax2.set_xticklabels([f"{v}k" for v in LEN_K], fontsize=7.0)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.tick_params(labelsize=7.0, length=2.5)
    ax2.set_xlabel("context length (tokens)", fontsize=8.0, labelpad=2.0)
    ax2.set_ylabel("mean accuracy (%)", fontsize=8.0)
    ax2.grid(True, axis="y", lw=0.5, color="#e6e6e6")
    ax2.set_axisbelow(True)
    style_axes(ax2)
    ax2.set_title("Depth-averaged retrieval", fontsize=8.5, fontweight="bold",
                  pad=4.0)
    leg = ax2.legend(loc="lower left", frameon=False, fontsize=6.8,
                     handlelength=1.4, labelspacing=0.3,
                     borderaxespad=0.35)
    for t, (n, _, _, _) in zip(leg.get_texts(), MODELS):
        if n.startswith("Ours"):
            t.set_fontweight("bold")
    fig.text(RIGHT[0] - 0.34 / FIG_W, RIGHT[1] + RIGHT[3] + 0.06 / FIG_H,
             "(c)", fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "318_llm_context_length.png")


if __name__ == "__main__":
    main()
