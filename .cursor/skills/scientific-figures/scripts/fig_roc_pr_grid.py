"""2x3 ROC + PR curve grid for model evaluation.

Each panel shows the ROC (solid) and the precision-recall curve (dashed) of one
classifier on the same imbalanced test set, with bootstrap 95% CIs on the AUC,
the chance diagonal for ROC and the prevalence baseline for PR. Scores come
from a latent Gaussian signal-detection model, so the AUCs land in a realistic
0.74-0.93 range instead of saturating at 1.00.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

MODELS = ["Logistic regression", "Random forest", "SVM (RBF)",
          "LightGBM", "MLP", "Stacked ensemble"]
TARGET_AUC = [0.745, 0.812, 0.836, 0.884, 0.867, 0.918]
ROC_C, PR_C = PALETTES["cb_safe"][0], PALETTES["cb_safe"][5]
N, PREVALENCE = 600, 0.32


def boot_ci(y, s, rng, n_boot=300):
    vals = []
    idx = np.arange(y.size)
    for _ in range(n_boot):
        b = rng.choice(idx, y.size, replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    return np.percentile(vals, [2.5, 97.5])


def panel(ax, rng, idx, name, target):
    y = (rng.random(N) < PREVALENCE).astype(int)
    d = np.sqrt(2) * norm.ppf(target)          # signal-detection separation
    s = rng.normal(d * y, 1.0)
    s = norm.cdf(s - d / 2)                    # map to a 0-1 score

    fpr, tpr, _ = roc_curve(y, s)
    a = roc_auc_score(y, s)
    lo, hi = boot_ci(y, s, rng)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)

    ax.plot([0, 1], [0, 1], color="#bbbbbb", ls=":", lw=0.9, zorder=1)
    ax.axhline(y.mean(), color="#bbbbbb", ls="--", lw=0.9, zorder=1)
    ax.fill_between(fpr, tpr, alpha=0.10, color=ROC_C, lw=0, zorder=2)
    ax.plot(fpr, tpr, color=ROC_C, lw=1.9, zorder=4,
            label=f"ROC  AUC = {a:.3f}\n[{lo:.3f}, {hi:.3f}]")
    ax.plot(rec, prec, color=PR_C, lw=1.5, ls="--", zorder=3,
            label=f"PR   AP = {ap:.3f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False-positive rate  /  Recall", fontsize=7.5)
    ax.set_ylabel("True-positive rate  /  Precision", fontsize=7.5)
    ax.set_title(name, fontsize=9, fontweight="bold")
    leg = ax.legend(fontsize=6.4, loc="lower right", frameon=True,
                    handlelength=1.4, borderpad=0.35, labelspacing=0.4,
                    facecolor="white", edgecolor="#cccccc", framealpha=0.92)
    leg.get_frame().set_linewidth(0.6)
    ax.grid(lw=0.5, alpha=0.35)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.18, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(301)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for k, (ax, name, t) in enumerate(zip(axes.ravel(), MODELS, TARGET_AUC)):
        panel(ax, rng, k, name, t)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], h_pad=2.2, w_pad=2.2)
    savefig(fig, "34_roc_pr_grid.png")


if __name__ == "__main__":
    main()
