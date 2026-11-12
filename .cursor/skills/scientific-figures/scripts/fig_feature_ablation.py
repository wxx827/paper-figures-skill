"""Feature-ablation study: cumulative removal, leave-one-out, subset heatmap.

(a) Test AUC as features are dropped one at a time under three removal
    orders, with 95% CI ribbons over 25 repeated splits.
(b) Leave-one-out contribution of every feature, signed and sorted, with SD
    error bars; a few uninformative features carry a negative contribution.
(c) AUC for cumulative top-k feature subsets across five model families.

All three panels are generated from one shared latent importance vector, so
the numbers are mutually consistent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

FEATURES = ["Serum creatinine", "Age", "HbA1c", "eGFR", "Systolic BP",
            "BMI", "LDL-C", "Smoking pack-yrs", "CRP", "Haemoglobin",
            "Sex", "Urban residence"]
MODELS = ["Logistic reg.", "Random forest", "MLP", "XGBoost",
          "TabTransformer"]
AUC_MAX = [0.845, 0.878, 0.869, 0.892, 0.901]
N_REP = 25
POS, NEG = "#2a7f7a", "#c0603f"
HMAP = LinearSegmentedColormap.from_list(
    "auc_seq", ["#f4f1ea", "#dfe5dd", "#b3ccc4", "#7aa8a0", "#417f78",
                "#1f574f"])


def auc_from_mass(mass, auc_max, rng=None, jitter=0.0):
    """Saturating map from retained importance mass to test AUC."""
    base = 0.5 + (auc_max - 0.5) * (1 - np.exp(-3.4 * mass)) / \
        (1 - np.exp(-3.4))
    if jitter:
        base = base + rng.normal(0, jitter, np.shape(mass))
    return base


def main():
    set_pub_style()
    rng = np.random.default_rng(90210)
    nf = len(FEATURES)

    # latent importance: decaying, with two near-useless trailing features
    w = np.exp(-0.40 * np.arange(nf)) * (1 + 0.14 * rng.normal(size=nf))
    w = np.clip(w, 0.004, None)
    w[-2:] *= 0.35
    order = np.argsort(-w)
    w, feats = w[order], [FEATURES[i] for i in order]
    total = w.sum()

    fig = plt.figure(figsize=(10.8, 7.9), layout="constrained")
    fig.get_layout_engine().set(rect=(0.004, 0.032, 0.992, 0.938),
                                h_pad=0.10, w_pad=0.12)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.62, 1.0],
                  height_ratios=[1.12, 1.0], hspace=0.09, wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ---- (a) cumulative removal -----------------------------------------
    ks = np.arange(nf + 1)
    schemes = [
        ("Least important removed first", "#3f6f9c", "o",
         lambda r: np.arange(nf - 1, nf - 1 - r, -1)),
        ("Random removal order", "#8a7f5c", "s", None),
        ("Most important removed first", "#b0452f", "^",
         lambda r: np.arange(r)),
    ]
    for name, col, mk, sel in schemes:
        curves = np.empty((N_REP, nf + 1))
        for rep in range(N_REP):
            for r in ks:
                if sel is None:
                    drop = rng.permutation(nf)[:r]
                else:
                    drop = sel(r)
                keep = np.setdiff1d(np.arange(nf), drop)
                curves[rep, r] = auc_from_mass(w[keep].sum() / total,
                                               0.892, rng, jitter=0.0055)
        mu = curves.mean(axis=0)
        lo = np.percentile(curves, 2.5, axis=0)
        hi = np.percentile(curves, 97.5, axis=0)
        ax_a.fill_between(ks, lo, hi, color=col, alpha=0.17, lw=0)
        ax_a.plot(ks, mu, color=col, lw=1.7, marker=mk, ms=3.4,
                  markeredgecolor="white", markeredgewidth=0.4, label=name)

    ax_a.axhline(0.5, color="#999999", lw=0.8, ls=(0, (4, 3)))
    ax_a.text(6.3, 0.508, "chance level", fontsize=6.6, color="#999999",
              ha="left", va="bottom")
    share3 = w[:3].sum() / total
    ax_a.annotate(f"top 3 features carry\n{100 * share3:.0f}% of the signal",
                  xy=(3, auc_from_mass(w[3:].sum() / total, 0.892)),
                  xytext=(5.1, 0.790), fontsize=7.0, color="#8c3a26",
                  linespacing=1.4, ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color="#8c3a26",
                                  shrinkA=2, shrinkB=3))
    ax_a.set_xlabel("Number of features removed", fontsize=8.2)
    ax_a.set_ylabel("Test AUC", fontsize=8.2)
    ax_a.set_xlim(-0.3, nf + 0.3)
    ax_a.set_xticks(ks[::2])
    ax_a.tick_params(labelsize=7.2)
    ax_a.legend(loc="lower left", fontsize=7.0, frameon=False,
                handlelength=1.7, labelspacing=0.35)
    style_axes(ax_a)
    add_panel_label(ax_a, "(a)", x=-0.085, y=1.02, size=10)
    ax_a.set_title("Cumulative feature removal (25 repeated 5-fold splits)",
                   fontsize=8.6, fontweight="bold", loc="left", pad=4)

    # ---- (b) leave-one-out contribution ---------------------------------
    full = auc_from_mass(1.0, 0.892)
    contrib = np.empty((N_REP, nf))
    for rep in range(N_REP):
        for j in range(nf):
            mass = (total - w[j]) / total
            contrib[rep, j] = (auc_from_mass(1.0, 0.892, rng, 0.0035)
                               - auc_from_mass(mass, 0.892, rng, 0.0035))
    # the weakest variables add variance without signal, so dropping them
    # slightly *improves* held-out AUC
    penalty = np.zeros(nf)
    penalty[-3:] = [0.0016, 0.0026, 0.0038]
    contrib -= penalty[None, :]
    mu = contrib.mean(axis=0)
    sd = contrib.std(axis=0)
    o = np.argsort(mu)
    ypos = np.arange(nf)
    cols = [POS if mu[i] >= 0 else NEG for i in o]
    ax_b.barh(ypos, mu[o], height=0.66, color=cols, alpha=0.88,
              edgecolor="white", linewidth=0.5)
    ax_b.errorbar(mu[o], ypos, xerr=sd[o], fmt="none", ecolor="#3a3a3a",
                  elinewidth=0.7, capsize=1.6, capthick=0.7)
    ax_b.axvline(0, color="#333333", lw=0.8)
    ax_b.set_yticks(ypos, [feats[i] for i in o], fontsize=7.0)
    ax_b.set_xlabel("Leave-one-out contribution  ($\\Delta$AUC)", fontsize=8.0)
    pad = (mu.max() - min(mu.min(), 0)) * 0.22
    ax_b.set_xlim(min(mu.min() - sd.max(), 0) - pad, mu.max() + sd.max() + pad)
    ax_b.set_ylim(-0.7, nf - 0.3)
    ax_b.tick_params(labelsize=7.0, length=2.6)
    ax_b.text(0.97, 0.045, f"Full-model AUC = {full:.3f}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=7.0,
              fontweight="bold", color="#2b2b2b")
    style_axes(ax_b)
    add_panel_label(ax_b, "(b)", x=-0.42, y=1.02, size=10)
    ax_b.set_title("Single-feature ablation", fontsize=8.6,
                   fontweight="bold", loc="left", pad=4)

    # ---- (c) cumulative top-k subsets x model families -------------------
    kk = np.arange(1, nf + 1)
    mass = np.array([w[:k].sum() / total for k in kk])
    grid = np.empty((len(MODELS), nf))
    for m, amax in enumerate(AUC_MAX):
        grid[m] = auc_from_mass(mass, amax, rng, jitter=0.0032)
        grid[m, -3:] -= np.linspace(0, 0.004 * (1 - m / 6), 3)  # mild overfit

    im = ax_c.imshow(grid, cmap=HMAP, aspect="auto",
                     vmin=grid.min(), vmax=grid.max())
    for m in range(len(MODELS)):
        best = int(np.argmax(grid[m]))
        for k in range(nf):
            v = grid[m, k]
            rel = (v - grid.min()) / (grid.max() - grid.min())
            ax_c.text(k, m, f"{v:.3f}", ha="center", va="center",
                      fontsize=6.1, color="white" if rel > 0.58 else "#2b2b2b",
                      fontweight="bold" if k == best else "normal")
        ax_c.add_patch(Rectangle((best - 0.5, m - 0.5), 1, 1, fill=False,
                                 edgecolor="#b0452f", lw=1.4, zorder=5))
    ax_c.set_xticks(range(nf), [f"top-{k}" for k in kk], fontsize=7.0)
    ax_c.set_yticks(range(len(MODELS)), MODELS, fontsize=7.4)
    ax_c.set_xlabel("Cumulative feature subset (features added in "
                    "descending importance)", fontsize=8.0)
    ax_c.tick_params(length=0)
    ax_c.set_xticks(np.arange(-0.5, nf, 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax_c.grid(which="minor", color="white", lw=0.9)
    ax_c.tick_params(which="minor", length=0)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    add_panel_label(ax_c, "(c)", x=-0.052, y=1.03, size=10)
    ax_c.set_title("Test AUC of cumulative top-k subsets  "
                   "(red outline = best subset per model)",
                   fontsize=8.6, fontweight="bold", loc="left", pad=4)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.020, pad=0.010, aspect=13)
    cb.set_label("Test AUC", fontsize=7.4, labelpad=3)
    cb.ax.tick_params(labelsize=6.6, length=2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "67_feature_ablation.png")


if __name__ == "__main__":
    main()
