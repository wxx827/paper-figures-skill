"""Figure 771 - Covariate balance before and after propensity-score matching.

Main panel: the Love plot.  Twelve covariates ordered by their unmatched
imbalance, each carrying two markers - hollow for the raw sample, filled for
the 1:1 matched sample - joined by a hairline so the reader follows the move
rather than hunting for pairs.  The 0.1 standardised-difference rule is drawn
on both sides of zero, because balance is a two-sided requirement.

Auxiliary (both at 60 % of the main panel width): the mirrored propensity
histogram, treated upward and control downward, with the common-support band
shaded; and the sample attrition through the two filtering stages.

Nothing is asserted: the propensity score is a real logistic fit, matching is
greedy nearest-neighbour on the linear predictor under a caliper, and every
standardised mean difference is recomputed from the matched rows.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

AFTER = "#b23a48"        # matched sample: the state we are arguing for
BEFORE = "#3b5b92"       # raw sample
LINK = "#c4c4c4"

N_UNITS = 2400
CALIPER = 0.20           # in standard deviations of the linear predictor
THRESH = 0.10            # conventional balance rule, in SD units

COVARIATES = [
    # name, mean, sd, binary, coefficient in the assignment rule
    ("Baseline risk score (SD)", 0.00, 1.00, False, 0.85),
    ("Prior utilisation (visits/yr)", 4.20, 2.30, False, 0.62),
    ("Log household income (log USD)", 10.30, 0.55, False, -0.58),
    ("Comorbidity index (-)", 1.90, 1.20, False, 0.44),
    ("Distance to clinic (km)", 12.0, 7.50, False, -0.40),
    ("Age (years)", 52.0, 13.0, False, 0.33),
    ("Years of schooling (years)", 12.4, 3.10, False, -0.26),
    ("Urban residence (0/1)", 0.58, 1.00, True, 0.24),
    ("Household size (persons)", 3.30, 1.40, False, 0.17),
    ("Employed (0/1)", 0.64, 1.00, True, -0.14),
    ("Insured at baseline (0/1)", 0.71, 1.00, True, 0.09),
    ("Region: north (0/1)", 0.34, 1.00, True, 0.04),
]


# ---------------------------------------------------------------------------
def make_sample(rng):
    p = len(COVARIATES)
    # Correlated latent factors: real covariates are never independent, and an
    # independent design would make matching look far easier than it is.
    load = rng.normal(0.0, 0.55, (p, 3))
    z = rng.normal(0.0, 1.0, (N_UNITS, 3)) @ load.T
    z = (z + rng.normal(0.0, 1.0, (N_UNITS, p))) / np.sqrt(1.0 + (load ** 2).sum(1))

    x = np.empty((N_UNITS, p))
    for j, (_, mu, sd, binary, _) in enumerate(COVARIATES):
        if binary:
            # Threshold the latent factor so the indicator keeps both the
            # requested prevalence and its correlation with the others.
            x[:, j] = (z[:, j] < stats.norm.ppf(mu)).astype(float)
        else:
            x[:, j] = mu + sd * z[:, j]

    beta = np.array([c[4] for c in COVARIATES])
    lin = z @ beta - 1.15 + rng.normal(0.0, 0.55, N_UNITS)
    d = (rng.random(N_UNITS) < 1.0 / (1.0 + np.exp(-lin))).astype(int)
    return x, d


def smd(x, d):
    """Standardised mean difference per covariate, in pooled SD units."""
    t, c = x[d == 1], x[d == 0]
    pooled = np.sqrt((t.var(0, ddof=1) + c.var(0, ddof=1)) / 2.0)
    return (t.mean(0) - c.mean(0)) / np.where(pooled > 0, pooled, np.nan)


def match(ps_lin, d, rng):
    """Greedy 1:1 nearest neighbour on the linear predictor, with a caliper."""
    cal = CALIPER * ps_lin.std(ddof=1)
    ti = np.where(d == 1)[0]
    ci = np.where(d == 0)[0]
    free = np.ones(ci.size, dtype=bool)
    pairs = []
    for i in rng.permutation(ti):
        gap = np.where(free, np.abs(ps_lin[ci] - ps_lin[i]), np.inf)
        j = int(np.argmin(gap))
        if gap[j] <= cal:
            pairs.append((i, ci[j]))
            free[j] = False
    return np.array(pairs), cal


# ---------------------------------------------------------------------------
def love_panel(ax, names, before, after, n_pairs):
    ys = np.arange(len(names))[::-1]
    for t in (-THRESH, THRESH):
        ax.axvline(t, color=BASE, lw=0.9, ls=(0, (3.0, 2.2)), zorder=1)
    ax.axvline(0.0, color="#3a3a3a", lw=0.8, zorder=1)
    ax.axvspan(-THRESH, THRESH, facecolor=CARD, edgecolor="none", zorder=0)

    for y, b, a in zip(ys, before, after):
        ax.plot([b, a], [y, y], lw=0.9, color=LINK, zorder=2)
    ax.scatter(before, ys, s=30, facecolor="white", edgecolor=BEFORE,
               linewidth=1.2, zorder=4, label="before matching")
    ax.scatter(after, ys, s=30, facecolor=AFTER, edgecolor=AFTER,
               linewidth=0.8, zorder=5,
               label=f"after 1:1 matching ({n_pairs} pairs)")

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.0)
    ax.set_ylim(-0.8, len(names) - 0.2)
    lim = max(np.abs(before).max(), np.abs(after).max()) * 1.20
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("standardised mean difference (SD units)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Every covariate moves inside the 0.1 rule after matching",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.0, frameon=False,
              handlelength=1.0, borderaxespad=0.4, labelspacing=0.32,
              scatterpoints=1)

    for t, ha in ((THRESH, "left"), (-THRESH, "right")):
        ax.text(t, len(names) - 0.42, f" {t:+.1f} ", ha=ha, va="center",
                fontsize=6.4, color=MUTED)


def support_panel(ax, ps, d, lo, hi):
    bins = np.linspace(0.0, 1.0, 34)
    ht, _ = np.histogram(ps[d == 1], bins=bins)
    hc, _ = np.histogram(ps[d == 0], bins=bins)
    mid = 0.5 * (bins[:-1] + bins[1:])
    w = bins[1] - bins[0]

    ax.axvspan(lo, hi, facecolor=CARD, edgecolor="none", zorder=0)
    ax.bar(mid, ht, width=w * 0.94, facecolor=lighten(AFTER, 0.30),
           edgecolor="white", linewidth=0.3, zorder=2)
    ax.bar(mid, -hc, width=w * 0.94, facecolor=lighten(BEFORE, 0.42),
           edgecolor="white", linewidth=0.3, zorder=2)
    ax.axhline(0.0, color="#3a3a3a", lw=0.8, zorder=3)
    for v in (lo, hi):
        ax.axvline(v, color=BASE, lw=0.8, ls=(0, (2.6, 2.0)), zorder=4)

    top = max(ht.max(), hc.max())
    ax.set_ylim(-top * 1.30, top * 1.30)
    ax.set_xlim(0.0, 1.0)
    ax.text(0.985, top * 1.16, f"treated (n = {(d == 1).sum()})", ha="right",
            va="top", fontsize=6.5, color=AFTER)
    ax.text(0.985, -top * 1.16, f"control (n = {(d == 0).sum()})", ha="right",
            va="bottom", fontsize=6.5, color=BEFORE)
    ax.text(0.5 * (lo + hi), -top * 1.27, "common support", ha="center",
            va="bottom", fontsize=6.3, color=MUTED)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{abs(v):.0f}"))
    ax.set_xlabel("estimated propensity score (-)", fontsize=7.4)
    ax.set_ylabel("units per bin (count)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Overlap: mirrored score histogram", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def attrition_panel(ax, stages, n_t, n_c):
    ys = np.arange(len(stages))[::-1]
    h = 0.34
    ax.barh(ys + h / 2 + 0.02, n_t, height=h, facecolor=AFTER,
            edgecolor="white", linewidth=0.5, zorder=3, label="treated")
    ax.barh(ys - h / 2 - 0.02, n_c, height=h, facecolor=BEFORE,
            edgecolor="white", linewidth=0.5, zorder=3, label="control")
    for y, v in zip(ys + h / 2 + 0.02, n_t):
        ax.text(v + max(n_t + n_c) * 0.012, y, f"{v:,}", ha="left",
                va="center", fontsize=6.3, color=AFTER)
    for y, v in zip(ys - h / 2 - 0.02, n_c):
        ax.text(v + max(n_t + n_c) * 0.012, y, f"{v:,}", ha="left",
                va="center", fontsize=6.3, color=BEFORE)

    ax.set_yticks(ys)
    ax.set_yticklabels(stages, fontsize=6.6)
    ax.set_ylim(-0.65, len(stages) - 0.35)
    ax.set_xlim(0.0, max(n_c) * 1.24)
    ax.set_xlabel("units retained (count)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Cost of balance: sample attrition", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.4, frameon=False,
              handlelength=1.0, borderaxespad=0.35, labelspacing=0.28)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(771)
    x, d = make_sample(rng)

    xs = (x - x.mean(0)) / x.std(0, ddof=1)
    model = LogisticRegression(max_iter=2000, C=1e4).fit(xs, d)
    lin = model.decision_function(xs)
    ps = 1.0 / (1.0 + np.exp(-lin))

    lo = max(ps[d == 1].min(), ps[d == 0].min())
    hi = min(ps[d == 1].max(), ps[d == 0].max())
    on = (ps >= lo) & (ps <= hi)

    pairs, cal = match(lin, np.where(on, d, -1), rng)
    keep = np.concatenate([pairs[:, 0], pairs[:, 1]])
    d_m = np.concatenate([np.ones(len(pairs), int), np.zeros(len(pairs), int)])

    before = smd(x, d)
    after = smd(x[keep], d_m)
    order = np.argsort(-np.abs(before))
    names = [COVARIATES[j][0] for j in order]

    stages = ["Full sample", "On common support", "Matched 1:1"]
    n_t = np.array([int((d == 1).sum()), int(((d == 1) & on).sum()),
                    len(pairs)])
    n_c = np.array([int((d == 0).sum()), int(((d == 0) & on).sum()),
                    len(pairs)])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (2.02, 0.82, 4.10, 4.52))
    ax_sup = axes_inches(fig, (7.42, 3.48, 2.46, 1.86))
    ax_att = axes_inches(fig, (7.42, 0.82, 2.46, 1.86))

    love_panel(ax_main, names, before[order], after[order], len(pairs))
    support_panel(ax_sup, ps, d, lo, hi)
    attrition_panel(ax_att, stages, n_t, n_c)

    worst_b = float(np.abs(before).max())
    worst_a = float(np.abs(after).max())

    for ax, lab in ((ax_main, "(a)"), (ax_sup, "(b)"), (ax_att, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "771_ci_propensity_balance.png")


if __name__ == "__main__":
    main()
