"""Wm blue hist pair: paired distributions with KDE overlay and stats call-outs.

Two side-by-side distributions from a video-QA world-model benchmark: the
number of frames a question needs before it can be answered, and the spatial
distance between the two objects a question refers to.  Both panels use the
batch blue ladder -- pale blue bars, deep blue KDE -- with dashed mean and
dotted median rules and a corner box carrying n / mean / median.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label)

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
BAR_FILL, KDE_LINE = BLUE[4], BLUE[0]
ACCENT = "#b5544a"
INK = "#22272b"


def gen_data(seed=2517):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_hist_pair.csv")
        return df["n_frames"].to_numpy(), df["pair_distance"].to_numpy()
    """
    rng = np.random.default_rng(seed)
    # (a) evidence length: most questions need most of the 16-frame clip, few
    #     are answerable from a couple of frames -> left-skewed.  Rejection
    #     sampling keeps the support clean instead of piling mass on the edge.
    frames = np.empty(0)
    while frames.size < 5200:
        cand = np.round(17 - rng.gamma(2.6, 2.05, 8000))
        frames = np.concatenate([frames, cand[(cand >= 3) & (cand <= 16)]])
    frames = frames[:5200]
    # (b) referent separation: log-normal, in metres.
    dist = rng.lognormal(0.42, 0.52, 20000)
    dist = dist[(dist >= 0.15) & (dist <= 7.0)][:5200]
    return frames, dist


def panel(ax, data, bins, xlabel, title, label, xlim, ymax, ticks=None,
          counts_like=False):
    counts, edges = np.histogram(data, bins=bins)
    ax.hist(data, bins=bins, color=BAR_FILL, edgecolor="white", linewidth=0.5,
            zorder=3)

    xs = np.linspace(xlim[0], xlim[1], 400)
    width = edges[1] - edges[0]
    ax.plot(xs, gaussian_kde(data)(xs) * data.size * width, color=KDE_LINE,
            lw=1.7, zorder=5)

    mu, med = float(np.mean(data)), float(np.median(data))
    ax.axvline(mu, color=ACCENT, ls="--", lw=1.2, zorder=6)
    ax.axvline(med, color=INK, ls=":", lw=1.2, zorder=6)

    # Put the box on whichever half of the panel carries less mass.
    right_heavy = counts[len(counts) // 2:].sum() > counts[:len(counts) // 2].sum()
    xa, ha = (0.035, "left") if right_heavy else (0.965, "right")
    # The median of an integer-valued sample is an integer or a half, so print
    # it as one instead of padding it out to look like a measured quantity.
    med_txt = f"{med:g}" if counts_like else f"{med:.2f}"
    ax.text(xa, 0.955,
            f"n = {data.size:,}\nmean = {mu:.2f}\nmedian = {med_txt}",
            transform=ax.transAxes, ha=ha, va="top", fontsize=6.5,
            color=INK, linespacing=1.35, zorder=8,
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#c6cdd4",
                      lw=0.6, alpha=0.94))

    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.set_xlabel(xlabel, fontsize=8.0)
    ax.set_ylabel("Number of questions", fontsize=8.0)
    ax.set_title(title, fontsize=9.0, fontweight="bold")
    ax.grid(axis="y", lw=0.5, color="#e2e6ea", zorder=0)
    ax.tick_params(labelsize=7.2)
    style_axes(ax)
    add_panel_label(ax, label, x=-0.14, y=1.03, size=11)


def main():
    set_pub_style()
    frames, dist = gen_data()

    bins_a = np.arange(2.5, 17.5, 1.0)
    bins_b = np.linspace(0.0, 7.0, 29)
    # Both panels count questions, so the y axis is shared across them.
    ymax = 1.22 * max(np.histogram(frames, bins=bins_a)[0].max(),
                      np.histogram(dist, bins=bins_b)[0].max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5))
    panel(axes[0], frames, bins_a, "Frames required to answer (count)",
          "Evidence length per question", "(a)", (2.5, 16.5), ymax,
          ticks=np.arange(4, 17, 2), counts_like=True)
    panel(axes[1], dist, bins_b, "Referent separation (m)",
          "Spatial distance between object pairs", "(b)", (0.0, 7.0), ymax,
          ticks=np.arange(0, 8, 1))

    handles = [
        Line2D([], [], color=BAR_FILL, lw=6, label="Histogram"),
        Line2D([], [], color=KDE_LINE, lw=1.7, label="Gaussian KDE"),
        Line2D([], [], color=ACCENT, ls="--", lw=1.2, label="Mean"),
        Line2D([], [], color=INK, ls=":", lw=1.2, label="Median"),
    ]
    fig.tight_layout(rect=[0, 0.035, 1, 0.935], w_pad=2.6)
    savefig(fig, "1112_wm_blue_hist_pair.png")


if __name__ == "__main__":
    main()
