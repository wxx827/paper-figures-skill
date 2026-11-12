"""Probability-calibration report: reliability diagram + score histogram + Brier bars.

(a) Reliability diagram for five models with binomial error bars and the
    perfect-calibration diagonal; (b) stacked histogram of the predicted
    probabilities on the shared x-axis; (c) Brier score bars with the
    expected calibration error annotated. All metrics are computed from the
    same synthetic held-out predictions, so the panels are mutually
    consistent.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

MODELS = [
    ("Logistic regression", "#3f5a8a", "o", "logistic"),
    ("Random forest (raw)", "#4a7c59", "s", "shrunk"),
    ("XGBoost (raw)", "#b0653a", "^", "sharpened"),
    ("XGBoost + Platt", "#8a5f9c", "D", "platt"),
    ("XGBoost + isotonic", "#a8493f", "v", "isotonic"),
]
N = 6000
N_BINS = 12


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def distort(kind, q, rng):
    """Map the true event probability to a model's reported probability."""
    if kind == "logistic":
        p = sigmoid(0.90 * logit(q) - 0.08 + rng.normal(0, 0.20, q.size))
    elif kind == "shrunk":                       # classic RF under-confidence
        p = sigmoid(0.46 * logit(q) - 0.10 + rng.normal(0, 0.28, q.size))
    elif kind == "sharpened":                    # boosted-tree over-confidence
        p = sigmoid(1.85 * logit(q) + 0.14 + rng.normal(0, 0.24, q.size))
    elif kind == "platt":
        p = sigmoid(1.02 * logit(q) + 0.03 + rng.normal(0, 0.18, q.size))
    else:                                        # isotonic
        p = sigmoid(1.00 * logit(q) + rng.normal(0, 0.14, q.size)) \
            + 0.012 * np.sin(7 * q)
    return np.clip(p, 0.002, 0.998)


def reliability(p, y, n_bins=N_BINS):
    """Equal-mass bins -> (mean predicted, observed frequency, SE, weight)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mp, of, se, wt = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb < 20:
            continue
        f = y[m].mean()
        mp.append(p[m].mean())
        of.append(f)
        se.append(np.sqrt(max(f * (1 - f), 1e-4) / nb))
        wt.append(nb)
    return (np.array(mp), np.array(of), np.array(se), np.array(wt))


def main():
    set_pub_style()
    rng = np.random.default_rng(31415)

    q = np.clip(rng.beta(1.7, 3.4, N), 0.002, 0.998)     # true risk
    y = (rng.random(N) < q).astype(int)
    floor = float(np.mean(q * (1 - q)))                  # oracle Brier score

    preds, stats = {}, {}
    for name, _, _, kind in MODELS:
        p = distort(kind, q, rng)
        mp, of, se, wt = reliability(p, y)
        ece = float(np.sum(wt / wt.sum() * np.abs(of - mp)))
        preds[name] = p
        stats[name] = dict(mp=mp, of=of, se=se,
                           brier=float(np.mean((p - y) ** 2)), ece=ece)

    fig = plt.figure(figsize=(10.4, 6.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0.004, 0.035, 0.992, 0.935),
                                h_pad=0.02, w_pad=0.10)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.85, 1.15],
                  height_ratios=[2.45, 1.0], hspace=0.0, wspace=0.14)
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_cal)
    ax_br = fig.add_subplot(gs[:, 1])

    # ---- (a) reliability diagram ----------------------------------------
    top = 0.88
    ax_cal.plot([0, top], [0, top], color="#8c8c8c", lw=1.0, ls=(0, (4, 3)),
                zorder=1)
    ax_cal.text(0.795, 0.807, "perfectly calibrated", rotation=45,
                rotation_mode="anchor", transform_rotates_text=True,
                fontsize=6.6, color="#8c8c8c", ha="center", va="bottom")
    ax_cal.text(0.615, 0.335, "model over-predicts risk", fontsize=7.2,
                color="#9a8a76", style="italic", ha="center")
    ax_cal.text(0.245, 0.700, "model under-predicts risk", fontsize=7.2,
                color="#7d90a6", style="italic", ha="center")

    for name, col, mk, _ in MODELS:
        s = stats[name]
        ax_cal.errorbar(s["mp"], s["of"], yerr=1.96 * s["se"], color=col,
                        marker=mk, ms=3.6, lw=1.5, elinewidth=0.8,
                        capsize=1.8, capthick=0.8, alpha=0.95,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=f"{name}   (ECE {s['ece']:.3f})", zorder=3)
    ax_cal.set_xlim(0, top)
    ax_cal.set_ylim(0, top)
    ax_cal.set_ylabel("Observed event frequency", fontsize=8.4)
    ax_cal.tick_params(labelbottom=False, labelsize=7.4)
    ax_cal.legend(loc="lower right", fontsize=7.0, frameon=False,
                  handlelength=1.6, handletextpad=0.6, labelspacing=0.42,
                  borderpad=0.2)
    style_axes(ax_cal)
    add_panel_label(ax_cal, "(a)", x=-0.075, y=1.01, size=10)
    ax_cal.set_title("Reliability diagram (12 equal-mass bins, "
                     "95% binomial CI)", fontsize=8.6, fontweight="bold",
                     loc="left", pad=4)

    # ---- (b) predicted-probability distribution -------------------------
    bins = np.linspace(0, 1, 41)
    ax_hist.hist([preds[m[0]] for m in MODELS], bins=bins, stacked=True,
                 color=[m[1] for m in MODELS], alpha=0.82,
                 edgecolor="white", linewidth=0.2)
    ax_hist.set_xlabel("Predicted probability", fontsize=8.4)
    ax_hist.set_ylabel("Count\n(stacked)", fontsize=7.6, linespacing=1.3)
    ax_hist.tick_params(labelsize=7.4)
    ax_hist.set_xlim(0, top)
    style_axes(ax_hist)
    add_panel_label(ax_hist, "(b)", x=-0.075, y=1.02, size=10)

    # ---- (c) Brier score / ECE ------------------------------------------
    names = [m[0] for m in MODELS]
    cols = [m[1] for m in MODELS]
    brier = np.array([stats[n]["brier"] for n in names])
    ece = np.array([stats[n]["ece"] for n in names])
    order = np.argsort(-brier)                    # worst at top, best at base
    ypos = np.arange(len(names))

    left = floor - 0.0025
    right = brier.max() + (brier.max() - floor) * 0.55
    ax_br.barh(ypos, brier[order] - left, left=left, height=0.55,
               color=[cols[i] for i in order], alpha=0.88,
               edgecolor="white", linewidth=0.6)
    for k, i in enumerate(order):
        ax_br.text(brier[i] + (right - left) * 0.022, k,
                   f"{brier[i]:.4f}", ha="left", va="center", fontsize=7.2,
                   fontweight="bold", color="#2b2b2b")
    ax_br.axvline(floor, color="#333333", lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax_br.text(floor + (right - left) * 0.015, len(names) - 0.42,
               f"oracle floor\n{floor:.4f}", fontsize=6.4, color="#555555",
               ha="left", va="center", linespacing=1.4)
    ax_br.set_yticks(ypos, [names[i] for i in order], fontsize=7.2)
    ax_br.set_xlim(left, right)
    ax_br.set_ylim(len(names) - 0.15, -0.6)
    ax_br.set_xlabel("Brier score  (lower is better;\naxis truncated near the "
                     "oracle floor)", fontsize=7.4, linespacing=1.4)
    ax_br.tick_params(labelsize=6.8, length=2.6)
    ax_br.xaxis.set_major_locator(plt.MaxNLocator(4))
    style_axes(ax_br)
    add_panel_label(ax_br, "(c)", x=-0.46, y=1.01, size=10)
    ax_br.set_title("Calibration quality", fontsize=8.6, fontweight="bold",
                    loc="left", pad=4)

    savefig(fig, "66_calibration_curves.png")


if __name__ == "__main__":
    main()
