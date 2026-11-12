"""Wm blue attention matrix: 4 layers x 6 heads of causal attention maps.

Twenty-four token-by-token attention matrices on one shared blue scale and one
shared colourbar, so a head's darkness is comparable across the whole sheet.
Each layer runs a different regime -- previous-token diagonal, local window,
attention-sink columns, sparse content-based -- and every panel is stamped with
its ``L{i}H{j}`` tag.

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
ATT_CMAP = LinearSegmentedColormap.from_list("wm_blue", BLUE[::-1])

N_LAYERS, N_HEADS, T = 4, 6, 28
REGIMES = ["previous-token diagonal", "local window", "attention sinks",
           "sparse content-based"]


def _causal_softmax(logits):
    """Row-wise softmax over the causal (lower-triangular) support."""
    mask = np.tril(np.ones((T, T), dtype=bool))
    z = np.where(mask, logits, -np.inf)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gen_data(seed=1515):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import numpy as np
        return np.load("data/attention.npy")      # (layer, head, T, T)
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.mgrid[0:T, 0:T]
    lag = ii - jj
    maps = np.zeros((N_LAYERS, N_HEADS, T, T))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            noise = rng.normal(0, 0.55, (T, T))
            if l == 0:                                  # previous token
                off = 1 + h % 3
                logits = 7.5 * np.exp(-0.5 * ((lag - off) / 0.75) ** 2)
            elif l == 1:                                # local window
                width = 2.0 + 0.9 * h
                logits = 5.0 * np.exp(-0.5 * (lag / width) ** 2)
            elif l == 2:                                # attention sinks
                sinks = [0, 1 + h % 4]
                logits = np.zeros((T, T))
                for s in sinks:
                    logits[:, s] += 6.0 - 0.8 * s
                logits += 2.2 * np.exp(-0.5 * (lag / 1.4) ** 2)
            else:                                       # sparse, content-based
                anchors = rng.choice(T, size=4, replace=False)
                logits = np.zeros((T, T))
                for a in anchors:
                    logits[:, a] += rng.uniform(3.0, 6.0)
                logits += 1.5 * np.exp(-0.5 * (lag / 1.1) ** 2)
                logits += 2.0 * (rng.random((T, T)) < 0.045)
            maps[l, h] = _causal_softmax(logits + noise)
    return maps


def main():
    set_pub_style(base_font=8)
    maps = gen_data()

    # Sweep all 24 panels first, then draw them on one scale (house rule).
    # A handful of near-1.0 diagonal cells would flatten everything else, so
    # the ramp is capped at the 96th percentile and the bar carries an arrow.
    vmax = float(np.quantile(maps, 0.96))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10.0, 7.0))
    gs = fig.add_gridspec(N_LAYERS, N_HEADS, left=0.070, right=0.905,
                          top=0.860, bottom=0.075, wspace=0.075, hspace=0.095)

    im = None
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            ax = fig.add_subplot(gs[l, h])
            im = ax.imshow(maps[l, h], cmap=ATT_CMAP, norm=norm,
                           interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)
                s.set_edgecolor(INK)
            ax.text(0.035, 0.955, f"L{l + 1}H{h + 1}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.0, color=INK, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.80))
            if l == 0:
                ax.set_title(f"Head {h + 1}", fontsize=7.6, color=INK, pad=3)
            if h == 0:
                ax.set_ylabel(f"Layer {l + 1}\n{REGIMES[l]}", fontsize=7.0,
                              color=INK, labelpad=4, linespacing=1.5)

    cax = fig.add_axes([0.925, 0.320, 0.013, 0.230])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, vmax / 2, vmax], extend="max")
    cb.ax.set_yticklabels(["0", f"{vmax / 2:.2f}", f"{vmax:.2f}+"])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_title("Attn.\nweight", fontsize=7.0, color=INK, pad=5)



    savefig(fig, "1110_wm_blue_attention_matrix.png")


if __name__ == "__main__":
    main()
