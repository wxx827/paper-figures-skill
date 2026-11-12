"""Composite biomarker panel: distributions, ROC, correlation and risk strata."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

CASE_C = "#b23a48"
CTRL_C = "#4c78a8"
MARKERS = ["IL-6", "hs-CRP", "NT-proBNP", "sST2"]
UNITS = ["pg/mL", "mg/L", "pg/mL", "ng/mL"]
EFFECT = np.array([0.80, 0.62, 0.88, 0.58])
CURVE_C = ["#7f9db9", "#3d7ab8", "#6a51a3", "#c58a3d"]


def sig_text(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def wilson(k, n, z=1.96):
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def simulate(rng, n_case=132, n_ctrl=198):
    corr = np.array([[1.00, 0.58, 0.34, 0.29],
                     [0.58, 1.00, 0.27, 0.22],
                     [0.34, 0.27, 1.00, 0.46],
                     [0.29, 0.22, 0.46, 1.00]])
    chol = np.linalg.cholesky(corr)
    n = n_case + n_ctrl
    z = (chol @ rng.standard_normal((4, n))).T
    y = np.zeros(n, dtype=int)
    y[:n_case] = 1
    z[y == 1] += EFFECT
    return z, y


def main():
    set_pub_style()
    rng = np.random.default_rng(31)
    z, y = simulate(rng)
    n_case, n_ctrl = int(y.sum()), int((1 - y).sum())

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.3))
    (a, b), (c, d) = axes
    fig.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.085,
                        hspace=0.44, wspace=0.30)

    # ---- (a) distributions -------------------------------------------------
    top = z.max() + 0.35                       # one bracket height for all
    for i in range(4):
        for grp, off, col in ((0, -0.185, CTRL_C), (1, 0.185, CASE_C)):
            vals = z[y == grp, i]
            parts = a.violinplot(vals, positions=[i + off], widths=0.33,
                                 showextrema=False, showmedians=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.32)
                pc.set_edgecolor(col)
                pc.set_linewidth(0.7)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            a.plot([i + off, i + off], [q1, q3], lw=2.6, color=col,
                   solid_capstyle="butt", zorder=4)
            a.plot(i + off, med, marker="o", ms=3.0, mfc="white", mec=col,
                   mew=0.9, zorder=5)
        p = stats.mannwhitneyu(z[y == 1, i], z[y == 0, i]).pvalue
        a.plot([i - 0.185, i - 0.185, i + 0.185, i + 0.185],
               [top, top + 0.16, top + 0.16, top], lw=0.8, color="#444444",
               solid_joinstyle="miter", zorder=5)
        a.text(i, top + 0.24, sig_text(p), ha="center", va="bottom",
               fontsize=7.0, fontweight="bold", color="#444444")

    a.set_xticks(range(4))
    a.set_xticklabels([f"{m}\n({u})" for m, u in zip(MARKERS, UNITS)],
                      fontsize=6.6, linespacing=1.4)
    a.set_ylim(-3.6, 6.0)
    a.set_yticks([-2, 0, 2, 4])
    a.set_ylabel("Standardized log concentration\n(z-score)")
    a.grid(axis="y", color="#f0f0f0", lw=0.6)
    style_axes(a)
    # the band above the brackets is reserved for the legend
    a.legend(handles=[Patch(facecolor=CTRL_C, alpha=0.45, edgecolor=CTRL_C,
                            label=f"Controls (n = {n_ctrl})"),
                      Patch(facecolor=CASE_C, alpha=0.45, edgecolor=CASE_C,
                            label=f"Cases (n = {n_case})")],
             loc="upper center", ncol=2, frameon=False, fontsize=6.4,
             handlelength=1.2, handleheight=0.9, columnspacing=1.4)
    a.set_title("Biomarker distributions by case status", fontsize=8.2,
                loc="left", pad=4)
    add_panel_label(a, "(a)", x=-0.155, y=1.04, size=10)

    # ---- (b) ROC -----------------------------------------------------------
    model = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    prob = cross_val_predict(model, z, y, cv=cv, method="predict_proba")[:, 1]

    b.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.9, color="#b0b0b0", zorder=1)
    for i, m in enumerate(MARKERS):
        fpr, tpr, _ = roc_curve(y, z[:, i])
        b.plot(fpr, tpr, lw=1.1, color=CURVE_C[i], zorder=3,
               label=f"{m}  {roc_auc_score(y, z[:, i]):.3f}")
    fpr, tpr, _ = roc_curve(y, prob)
    auc_c = roc_auc_score(y, prob)
    b.plot(fpr, tpr, lw=1.9, color=CASE_C, zorder=4,
           label=f"4-marker panel  {auc_c:.3f}")

    b.set_xlim(-0.02, 1.02)
    b.set_ylim(-0.02, 1.02)
    b.set_aspect("equal", adjustable="box")
    b.set_xlabel("1 \u2212 specificity")
    b.set_ylabel("Sensitivity")
    b.grid(color="#f5f5f5", lw=0.6)
    style_axes(b)
    leg = b.legend(loc="lower right", frameon=True, fontsize=6.1,
                   handlelength=1.5, labelspacing=0.3, borderpad=0.45,
                   title="AUC (5-fold CV for the panel)", framealpha=0.95,
                   edgecolor="#d5d5d5")
    leg.get_title().set_fontsize(6.3)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"
    b.set_title("Discrimination", fontsize=8.2, loc="left", pad=4)
    add_panel_label(b, "(b)", x=-0.175, y=1.04, size=10)

    # ---- (c) correlation heatmap ------------------------------------------
    rho = np.array([[stats.spearmanr(z[:, i], z[:, j]).statistic
                     for j in range(4)] for i in range(4)])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rdbu", PALETTES["div_rdbu"][::-1])
    im = c.imshow(rho, cmap=cmap, vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            c.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                   fontsize=6.5, fontweight="bold" if i != j else "normal",
                   color="white" if abs(rho[i, j]) > 0.62 else "#1a1a1a")
    c.set_xticks(range(4))
    c.set_yticks(range(4))
    c.set_xticklabels(MARKERS, fontsize=6.6, rotation=30, ha="right")
    c.set_yticklabels(MARKERS, fontsize=6.6)
    c.tick_params(length=0)
    for s in c.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=c, fraction=0.045, pad=0.035, ticks=[-1, 0, 1])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2)
    cb.set_label("Spearman $\\rho$", fontsize=6.4)
    c.set_title("Between-marker correlation", fontsize=8.2, loc="left", pad=4)
    add_panel_label(c, "(c)", x=-0.155, y=1.04, size=10)

    # ---- (d) risk stratification ------------------------------------------
    cuts = np.quantile(prob, [0.25, 0.50, 0.75])
    grp = np.digitize(prob, cuts)
    shades = ["#cfdced", "#9db9d6", "#5f8ab8", "#1f4e79"]
    rates, los, his, ns = [], [], [], []
    for g in range(4):
        m = grp == g
        k, tot = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, tot)
        rates.append(100 * k / tot)
        los.append(100 * (k / tot - lo))
        his.append(100 * (hi - k / tot))
        ns.append((k, tot))

    xs = np.arange(4)
    d.bar(xs, rates, width=0.62, color=shades, edgecolor="white",
          linewidth=0.6, zorder=3)
    d.errorbar(xs, rates, yerr=[los, his], fmt="none", ecolor="#333333",
               elinewidth=0.9, capsize=2.6, zorder=4)
    for x, r, (k, tot) in zip(xs, rates, ns):
        d.text(x, 3, f"{k}/{tot}", ha="center", va="bottom", fontsize=6.2,
               color="white" if x >= 2 else "#1a1a1a", zorder=5)
        d.text(x, r + his[x] + 2.6, f"{r:.0f}%", ha="center", va="bottom",
               fontsize=6.6, fontweight="bold", color="#1a1a1a", zorder=5)

    k1, n1 = ns[0]
    k4, n4 = ns[3]
    or_ = (k4 / (n4 - k4)) / (k1 / (n1 - k1))
    se = np.sqrt(1 / k4 + 1 / (n4 - k4) + 1 / k1 + 1 / (n1 - k1))
    d.set_xticks(xs)
    d.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"], fontsize=6.6,
                      linespacing=1.4)
    d.set_ylim(0, 118)
    d.set_yticks(np.arange(0, 101, 25))
    d.set_ylabel("Observed case prevalence (%)")
    d.set_xlabel("Predicted-risk quartile (4-marker panel)", labelpad=4)
    d.grid(axis="y", color="#f0f0f0", lw=0.6)
    style_axes(d)
    d.text(0.03, 0.965,
           f"OR Q4 vs Q1 = {or_:.1f}\n(95% CI {or_ * np.exp(-1.96 * se):.1f}"
           f"\u2013{or_ * np.exp(1.96 * se):.1f})",
           transform=d.transAxes, ha="left", va="top", fontsize=6.4,
           linespacing=1.5, fontweight="bold", color="#1a1a1a")
    d.set_title("Risk stratification", fontsize=8.2, loc="left", pad=4)
    add_panel_label(d, "(d)", x=-0.155, y=1.04, size=10)


    savefig(fig, "118_biomarker_panel.png")


if __name__ == "__main__":
    main()
