"""Wm blue confusion pair: interaction classes before and after adaptation.

Two row-normalised confusion matrices share one vmin/vmax and one colourbar, so
the darkening diagonal is the whole story.  Diagonal cells carry a heavy outline
and the footer compares the two overall accuracies, with the single warm accent
reserved for the gain.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig

# WM blue batch ladder (dark -> near white); see BRIEF_WM_BLUE.md section 2.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
BLUE_CMAP = LinearSegmentedColormap.from_list("wm_blue", BLUE[::-1])

CLASSES = ["push", "pull", "lift", "place", "rotate", "slide"]
SUPPORT = np.array([620, 480, 540, 700, 410, 390])
AFFINITY = np.array([
    [0.00, 0.55, 0.10, 0.12, 0.08, 0.35],
    [0.55, 0.00, 0.12, 0.10, 0.10, 0.20],
    [0.10, 0.14, 0.00, 0.48, 0.14, 0.08],
    [0.12, 0.10, 0.50, 0.00, 0.12, 0.10],
    [0.10, 0.12, 0.16, 0.14, 0.00, 0.30],
    [0.38, 0.20, 0.08, 0.10, 0.28, 0.00],
])


def gen_data(seed=1512):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import numpy as np
        return np.load("data/cm_before.npy"), np.load("data/cm_after.npy")
    """
    rng = np.random.default_rng(seed)
    ease = np.array([0.88, 0.72, 0.83, 0.90, 0.64, 0.69])

    def counts(skill):
        k = len(CLASSES)
        cm = np.zeros((k, k), dtype=int)
        for i in range(k):
            p_ii = float(np.clip(ease[i] * (0.66 + 0.34 * skill), 0.05, 0.99))
            off = AFFINITY[i] / AFFINITY[i].sum() * (1.0 - p_ii)
            p = off.copy()
            p[i] = p_ii
            cm[i] = rng.multinomial(SUPPORT[i], p / p.sum())
        return cm

    return counts(0.12), counts(0.94)


def text_colour(rgba):
    """White on dark fills, ink on light ones (relative luminance < 0.55)."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else INK


def panel(ax, cm, title, norm, show_y):
    frac = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, cmap=BLUE_CMAP, norm=norm, interpolation="nearest",
                   aspect="equal")
    k = len(CLASSES)
    for i in range(k):
        for j in range(k):
            v = frac[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color=text_colour(BLUE_CMAP(norm(v))))
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=INK, lw=1.5, zorder=5))

    ax.set_title(title, fontsize=9.0, fontweight="bold", color=INK, pad=5)
    ax.set_xticks(range(k))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=7.2, color=INK)
    ax.set_yticks(range(k))
    ax.set_yticklabels(CLASSES if show_y else [""] * k, fontsize=7.2,
                       color=INK)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Predicted interaction", fontsize=8.0, color=INK)
    if show_y:
        ax.set_ylabel("True interaction", fontsize=8.0, color=INK)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor(INK)
    return im


def main():
    set_pub_style(base_font=8)
    cm_before, cm_after = gen_data()

    # One scale for both panels: sweep the data first, then draw (house rule).
    fracs = [c / c.sum(axis=1, keepdims=True) for c in (cm_before, cm_after)]
    norm = Normalize(vmin=0.0, vmax=float(max(f.max() for f in fracs)))

    fig = plt.figure(figsize=(9.0, 4.6))
    axes = [fig.add_axes([0.075, 0.215, 0.355, 0.615]),
            fig.add_axes([0.500, 0.215, 0.355, 0.615])]
    titles = ["Before adaptation (source policy)",
              "After adaptation (target domain)"]
    im = None
    for ax, cm, title, first in zip(axes, (cm_before, cm_after), titles,
                                    (True, False)):
        im = panel(ax, cm, title, norm, show_y=first)

    cax = fig.add_axes([0.880, 0.300, 0.013, 0.260])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.4, 0.8])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_title("Row\nfrac.", fontsize=7.0, color=INK, pad=5)

    acc_b = np.trace(cm_before) / cm_before.sum()
    acc_a = np.trace(cm_after) / cm_after.sum()


    savefig(fig, "1107_wm_blue_confusion_pair.png")


if __name__ == "__main__":
    main()
