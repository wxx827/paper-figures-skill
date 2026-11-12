"""Wm blue corr dendrogram: clustered latent-factor correlation heatmap.

Seaborn-clustermap layout built by hand: average-linkage dendrograms above and
to the left of a single-hue blue correlation matrix, with the colourbar parked
in the free top-right corner.  Dendrogram strokes are hairline navy so they read
as structure, not as data.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from pubstyle import set_pub_style, savefig

# WM blue batch ladder (dark -> near white); see BRIEF_WM_BLUE.md section 2.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
TREE = "#1f4e79"
BLUE_CMAP = LinearSegmentedColormap.from_list("wm_blue", BLUE[::-1])

# Latent factors probed in the world model, grouped by the block they load on.
BLOCKS = [
    (["ego-vx", "ego-vy", "ego-yaw", "ego-acc", "cam-pitch"], 0.88),
    (["obj-mass", "obj-fric", "contact", "grasp", "slip-rate"], 0.83),
    (["depth-rms", "occl-frac", "flow-mag", "edge-crisp"], 0.80),
    (["reward", "goal-dist", "horizon", "entropy", "value-var", "td-err"],
     0.76),
]
N_OBS = 240


def gen_data(seed=1513):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_latent_probe.csv")
        return df.corr().to_numpy(), list(df.columns)
    """
    rng = np.random.default_rng(seed)
    cols, labels = [], []
    for members, load in BLOCKS:
        latent = rng.normal(0, 1, N_OBS)
        for k, name in enumerate(members):
            sign = -1.0 if k == len(members) - 1 else 1.0
            cols.append(sign * load * latent
                        + np.sqrt(1 - load ** 2) * rng.normal(0, 1, N_OBS))
            labels.append(name)
    return np.corrcoef(np.column_stack(cols), rowvar=False), labels


def main():
    set_pub_style(base_font=8)
    corr, labels = gen_data()
    n = len(labels)

    link = linkage(squareform(np.clip(1.0 - corr, 0, 2), checks=False),
                   method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    M = corr[np.ix_(order, order)]
    lab = [labels[i] for i in order]

    # Sweep the full matrix before drawing so the ramp uses the real range.
    norm = Normalize(vmin=float(M.min()), vmax=1.0)

    fig = plt.figure(figsize=(7.6, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.16, 1.0],
                          height_ratios=[0.155, 1.0],
                          left=0.055, right=0.800, top=0.855, bottom=0.115,
                          wspace=0.022, hspace=0.022)

    ax_top = fig.add_subplot(gs[0, 1])
    dendrogram(link, ax=ax_top, color_threshold=0, above_threshold_color=TREE,
               no_labels=True)
    ax_top.axis("off")

    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(link, ax=ax_left, orientation="left", color_threshold=0,
               above_threshold_color=TREE, no_labels=True)
    ax_left.invert_yaxis()
    ax_left.axis("off")
    for a in (ax_top, ax_left):
        for line in a.collections:
            line.set_linewidth(0.7)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(M, cmap=BLUE_CMAP, norm=norm, interpolation="nearest",
                   aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(lab, rotation=90, fontsize=6.4, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels(lab, fontsize=6.4, color=INK)
    ax.yaxis.tick_right()
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.55)
    ax.tick_params(which="both", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor(INK)

    cax = fig.add_axes([0.930, 0.598, 0.013, 0.145])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.8, 0.0, 0.8])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_title("Pearson r", fontsize=7.0, color=INK, pad=5)


    savefig(fig, "1108_wm_blue_corr_dendrogram.png")


if __name__ == "__main__":
    main()
