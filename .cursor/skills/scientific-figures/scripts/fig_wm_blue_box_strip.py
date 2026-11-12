"""Wm blue box strip: box + jitter comparison with a pairwise significance panel.

Five rollout models scored on the same held-out clips.  The main panel pairs a
pale-blue box with the raw per-clip points jittered on top and prints each
median above its box; the narrow companion panel on the right condenses all ten
pairwise Welch tests into a lower-triangular matrix shaded by significance.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
BOX_FILL, BOX_EDGE, DOT = BLUE[6], BLUE[1], BLUE[2]
ACCENT = "#b5544a"
INK = "#22272b"

METHODS = ["PlaNet", "Dreamer-V2", "IRIS", "TD-MPC2", "Ours"]
N_CLIPS = 64
ROLLOUT_STEPS = 16      # horizon the PSNR is measured at
SIG_CAP = 5.0           # -log10 p is clipped here, for cell shading only


def gen_data(seed=2520):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_box_strip.csv")
        return [g["psnr"].to_numpy() for _, g in df.groupby("method")]
    """
    rng = np.random.default_rng(seed)
    level = [21.4, 22.6, 23.1, 23.4, 25.0]
    spread = [2.5, 2.2, 2.4, 2.0, 1.7]
    return [rng.normal(m, s, N_CLIPS) for m, s in zip(level, spread)]


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
        else "n.s."


def main_panel(ax, rng, data):
    pos = np.arange(1, len(data) + 1)
    bp = ax.boxplot(data, positions=pos, widths=0.46, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=BLUE[0], lw=1.4),
                    whiskerprops=dict(color=BOX_EDGE, lw=0.9),
                    capprops=dict(color=BOX_EDGE, lw=0.9), zorder=3)
    for i, box in enumerate(bp["boxes"]):
        ours = METHODS[i] == "Ours"
        box.set(facecolor=BOX_FILL, edgecolor=ACCENT if ours else BOX_EDGE,
                lw=1.2 if ours else 1.0)
    bp["medians"][METHODS.index("Ours")].set_color(ACCENT)

    for i, d in enumerate(data):
        ax.scatter(pos[i] + rng.uniform(-0.15, 0.15, d.size), d, s=7,
                   color=DOT, alpha=0.38, edgecolors="none", zorder=2)

    # Labels sit above the upper whisker cap, never inside the data.
    caps = []
    for d in data:
        q1, q3 = np.percentile(d, [25, 75])
        caps.append(float(d[d <= q3 + 1.5 * (q3 - q1)].max()))
    hi, lo = max(caps), min(float(d.min()) for d in data)
    span = hi - lo
    for i, (d, cap) in enumerate(zip(data, caps)):
        ax.text(pos[i], cap + span * 0.035, f"{np.median(d):.2f}",
                ha="center", va="bottom", fontsize=6.4, color=INK, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground="white")])

    ax.set_ylim(lo - span * 0.06, hi + span * 0.14)
    ax.set_xlim(0.45, len(data) + 0.55)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{i + 1}. {m}" for i, m in enumerate(METHODS)],
                       fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel(f"Rollout PSNR at {ROLLOUT_STEPS} steps (dB)", fontsize=8.5)
    n_clips = {d.size for d in data}
    ax.set_title(f"Per-clip rollout quality, {max(n_clips)} held-out clips",
                 fontsize=9.0, fontweight="bold")
    ax.grid(axis="y", lw=0.5, color="#e2e6ea", zorder=0)
    style_axes(ax)


def sig_panel(ax, data):
    n = len(data)
    score = np.full((n, n), np.nan)
    stars = {}
    for i in range(n):
        for j in range(i):
            p = stats.ttest_ind(data[i], data[j], equal_var=False).pvalue
            score[i, j] = np.clip(-np.log10(max(p, 1e-12)), 0.0, SIG_CAP)
            stars[(i, j)] = sig_stars(p)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sig_blue", [BLUE[7], BLUE[5], BLUE[3], BLUE[1], BLUE[0]])
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(score), cmap=cmap, vmin=0.0, vmax=SIG_CAP)
    ax.set_anchor("N")

    for (i, j), s in stars.items():
        dark = score[i, j] > 2.6
        ax.text(j, i, s, ha="center", va="center", fontsize=6.0,
                color="white" if dark else INK)

    ax.set_xticks(range(n - 1))
    ax.set_xticklabels(range(1, n), fontsize=6.5)
    ax.set_yticks(range(1, n))
    ax.set_yticklabels(range(2, n + 1), fontsize=6.5)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    ax.set_xlabel("Method index", fontsize=7.5)
    ax.set_title("Pairwise\nWelch test", fontsize=8.5, fontweight="bold",
                 linespacing=1.25)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)


def main():
    set_pub_style()
    rng = np.random.default_rng(4520)
    data = gen_data()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5),
                             gridspec_kw=dict(width_ratios=[1.0, 0.28],
                                              wspace=0.34))
    main_panel(axes[0], rng, data)
    sig_panel(axes[1], data)

    handles = [
        Line2D([], [], color=BLUE[0], lw=1.4, label="Median"),
        Line2D([], [], marker="s", color="none", markerfacecolor=BOX_FILL,
               markeredgecolor=BOX_EDGE, markersize=6, label="IQR box"),
        Line2D([], [], marker="o", color="none", markerfacecolor=DOT,
               markeredgecolor="none", alpha=0.5, markersize=4,
               label="Individual clip"),
        Line2D([], [], marker="s", color="none", markerfacecolor=BOX_FILL,
               markeredgecolor=ACCENT, markersize=6, label="Ours"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7.0,
                   frameon=False, handletextpad=0.4, labelspacing=0.4)
    add_panel_label(axes[0], "(a)", x=-0.10, y=1.03, size=11)
    add_panel_label(axes[1], "(b)", x=-0.44, y=1.10, size=11)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    savefig(fig, "1115_wm_blue_box_strip.png")


if __name__ == "__main__":
    main()
