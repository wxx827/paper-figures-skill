"""Wm blue similarity matrix: annotated 10x10 benchmark similarity matrix.

Single-hue blue matrix in the house world-model palette: every cell carries its
value to three decimals, the label colour flips from ink to white as the fill
darkens, column names are rotated 45 degrees along the top edge, and a narrow
quarter-height colourbar sits clear of the grid.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pubstyle import set_pub_style, savefig

# WM blue batch ladder (dark -> near white); see BRIEF_WM_BLUE.md section 2.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
BLUE_CMAP = LinearSegmentedColormap.from_list("wm_blue", BLUE[::-1])

BENCHMARKS = ["AgentWorld", "DreamBench", "EgoSim-1.0", "GenieEval",
              "NavSuite-X", "OmniWorld", "PhyRe-Bench", "RoboDream",
              "VideoPred-B", "WM-Arena"]


def gen_data(seed=1510):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_similarity_matrix.csv", index_col=0)
        return df.to_numpy(), list(df.columns)
    """
    rng = np.random.default_rng(seed)
    n = len(BENCHMARKS)
    # Three latent capability axes -> cosine similarity between benchmarks.
    emb = rng.normal(0, 1, (n, 3))
    emb += rng.normal(0, 0.35, (n, 3))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T
    sim = 0.42 + 0.57 * (sim - sim.min()) / (sim.max() - sim.min())
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim, BENCHMARKS


def text_colour(rgba):
    """White on dark fills, ink on light ones (relative luminance < 0.55)."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else INK


def main():
    set_pub_style(base_font=8)
    sim, labels = gen_data()
    n = len(labels)
    lo = float(np.min(sim[~np.eye(n, dtype=bool)]))
    norm = Normalize(vmin=lo, vmax=1.0)

    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_axes([0.175, 0.045, 0.725, 0.815])
    im = ax.imshow(sim, cmap=BLUE_CMAP, norm=norm, interpolation="nearest",
                   aspect="equal")

    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                    color=text_colour(BLUE_CMAP(norm(v))))

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor",
                       fontsize=7.2, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, ha="right", fontsize=7.2, color=INK)
    ax.tick_params(length=0, pad=3)

    # 1 px white gutters between cells, hairline frame around the grid.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor(INK)

    cax = fig.add_axes([0.925, 0.085, 0.016, 0.245])
    cb = fig.colorbar(im, cax=cax, ticks=[round(lo, 2), 1.0])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_title("Sim.", fontsize=7.0, color=INK, pad=4)


    savefig(fig, "1105_wm_blue_similarity_matrix.png")


if __name__ == "__main__":
    main()
