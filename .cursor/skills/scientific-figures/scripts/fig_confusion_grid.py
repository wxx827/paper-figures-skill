"""2x3 grid of row-normalised confusion matrices for six classifiers.

Every cell carries the row-normalised percentage (bold) over the raw support
count; diagonal cells are outlined so the correct-classification ridge reads
at a glance. Panel titles report accuracy and macro-F1 computed from the same
synthetic counts, and one shared colourbar serves all six panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

CMAP = LinearSegmentedColormap.from_list(
    "paper_teal",
    ["#ffffff", "#e4ece9", "#bcd6d0", "#84b5ac", "#48887f", "#1e5b55"])

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_LONG = ["N - normal beat", "S - supraventricular", "V - ventricular",
              "F - fusion", "Q - unclassifiable"]
SUPPORT = np.array([4180, 570, 1120, 190, 440])
EASE = np.array([0.985, 0.800, 0.920, 0.660, 0.860])
AFFINITY = np.array([
    [0.00, 0.45, 0.25, 0.10, 0.20],
    [0.60, 0.00, 0.20, 0.08, 0.12],
    [0.25, 0.15, 0.00, 0.45, 0.15],
    [0.20, 0.10, 0.55, 0.00, 0.15],
    [0.45, 0.15, 0.20, 0.20, 0.00],
])
MODELS = ["Logistic regression", "Random forest", "XGBoost",
          "1D-CNN", "BiLSTM", "CNN-Transformer"]
SKILL = [0.05, 0.34, 0.52, 0.70, 0.84, 1.00]


def make_counts(rng, skill):
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for i in range(k):
        p_ii = float(np.clip(EASE[i] * (0.70 + 0.30 * skill), 0.05, 0.995))
        off = AFFINITY[i] / AFFINITY[i].sum() * (1 - p_ii)
        p = off.copy()
        p[i] = p_ii
        cm[i] = rng.multinomial(SUPPORT[i], p / p.sum())
    return cm


def macro_f1(cm):
    f1 = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i, :].sum(), 1)
        f1.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1))


def panel(ax, cm, name, label, show_y, show_x):
    frac = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, cmap=CMAP, norm=Normalize(0, 1), aspect="equal")

    k = len(CLASSES)
    for i in range(k):
        for j in range(k):
            v = frac[i, j]
            if cm[i, j] == 0:
                ax.text(j, i, "-", ha="center", va="center", fontsize=6.5,
                        color="#b8b8b8")
                continue
            tc = "white" if v > 0.55 else "#2b2b2b"
            ax.text(j, i - 0.13, f"{100 * v:.1f}", ha="center", va="center",
                    fontsize=6.6, fontweight="bold", color=tc)
            ax.text(j, i + 0.24, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=6.0, color=tc, alpha=0.85)
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#1a1a1a", lw=1.25, zorder=5))

    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{name}\nAcc {acc:.3f}   Macro-F1 {macro_f1(cm):.3f}",
                 fontsize=8.2, fontweight="bold", linespacing=1.5, pad=4)
    ax.set_xticks(range(k), CLASSES, fontsize=7.4)
    ax.set_yticks(range(k), CLASSES, fontsize=7.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    if show_x:
        ax.set_xlabel("Predicted class", fontsize=7.8)
    if show_y:
        ax.set_ylabel("True class", fontsize=7.8)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    add_panel_label(ax, label, x=-0.17 if show_y else -0.08, y=1.15, size=9.5)
    return im


def main():
    set_pub_style()
    rng = np.random.default_rng(4242)
    fig = plt.figure(figsize=(10.4, 6.9), layout="constrained")
    fig.get_layout_engine().set(rect=(0.004, 0.055, 0.992, 0.925),
                                h_pad=0.10, w_pad=0.06)
    axes = fig.subplots(2, 3)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    im = None
    for k, ax in enumerate(axes.ravel()):
        cm = make_counts(rng, SKILL[k])
        im = panel(ax, cm, MODELS[k], labels[k],
                   show_y=(k % 3 == 0), show_x=(k >= 3))

    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.014, aspect=30)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("Row-normalised proportion (%)", fontsize=7.6, labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "65_confusion_grid.png")


if __name__ == "__main__":
    main()
