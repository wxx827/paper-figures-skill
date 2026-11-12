"""Dietary patterns: a food-group biplot, nutrient adequacy and outcome risk.

A cohort of 264 adults is drawn from three latent eating patterns described
only by their mean servings of ten food groups.  Everything else is measured
from that one draw: the biplot is a principal component analysis of the
standardised serving matrix, so the arrows are the real loadings and the
percentages in the axis labels are the real explained variance; the radar
multiplies each participant's servings by a per-serving nutrient table and
expresses the result as a percentage of the reference intake; and the forest
plot is the crude odds ratio of the Western against the Mediterranean pattern
for five outcomes, with the textbook log-odds standard error.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA

from pubstyle import savefig, set_pub_style
from figkit import halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"

FOODS = ["Vegetables", "Fruit", "Whole grains", "Refined grains", "Red meat",
         "Fish", "Dairy", "Legumes", "Sugary drinks", "Nuts"]
PATTERNS = [
    ("Mediterranean", GREEN,
     [3.4, 2.6, 2.2, 1.2, 0.5, 0.9, 1.4, 1.1, 0.3, 0.9], 92),
    ("Plant-forward", BLUE,
     [4.2, 3.0, 2.8, 1.0, 0.1, 0.1, 0.6, 2.2, 0.2, 1.2], 78),
    ("Western", RED,
     [1.4, 0.9, 0.7, 4.2, 1.8, 0.2, 1.6, 0.3, 1.9, 0.2], 94),
]
NUTRIENTS = ["Fibre", "Protein", "Calcium", "Iron", "Vitamin C", "Folate",
             "Potassium"]
# per serving: fibre g, protein g, calcium mg, iron mg, vit C mg,
#              folate ug, potassium mg
CONTENT = np.array([
    [2.5, 2.0, 40., 0.8, 25., 60., 350.],     # vegetables
    [2.2, 0.8, 20., 0.3, 35., 25., 300.],     # fruit
    [3.0, 3.5, 20., 1.4, 0.0, 30., 120.],     # whole grains
    [0.8, 2.5, 15., 1.0, 0.0, 40., 60.],      # refined grains
    [0.0, 21.0, 12., 2.4, 0.0, 8., 300.],     # red meat
    [0.0, 20.0, 25., 0.7, 0.0, 10., 350.],    # fish
    [0.0, 8.0, 300., 0.1, 2.0, 12., 380.],    # dairy
    [7.5, 8.0, 45., 2.2, 2.0, 130., 480.],    # legumes
    [0.0, 0.0, 5., 0.1, 0.0, 0., 15.],        # sugary drinks
    [3.0, 5.0, 60., 1.1, 0.0, 25., 200.],     # nuts
])
REFERENCE = np.array([28.0, 56.0, 1000.0, 14.0, 80.0, 400.0, 3500.0])
# outcome, event probability under the Mediterranean and Western patterns
OUTCOMES = [("Type 2 diabetes", 0.085, 0.255),
            ("Metabolic syndrome", 0.150, 0.395),
            ("Hypertension", 0.190, 0.360),
            ("Cardiovascular event", 0.070, 0.185),
            ("Obesity, BMI >= 30", 0.135, 0.330)]


def cov_ellipse(ax, xy, colour, n_sd=2.0):
    """Covariance ellipse holding roughly 95 % of a bivariate normal cloud."""
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_sd * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse(xy.mean(axis=0), w, h, angle=ang, facecolor=colour,
                         alpha=0.10, edgecolor=colour, lw=0.9, ls="--",
                         zorder=2))


def outside_labels(ax, items, span, x_text_f=1.18, min_gap_f=0.125,
                   fontsize=6.2):
    """Loading labels parked in two columns, ordered and never overlapping.

    Labelling each arrow at its own tip works only until two food groups load
    similarly, after which the names pile up in the middle of the biplot.  The
    two-sweep pass used here pushes every label at least ``min_gap`` below its
    predecessor and then lifts whatever overflowed the column; because both
    sweeps preserve the original vertical order the leader lines cannot cross.

    ``items`` is a sequence of ``(tip_x, tip_y, name)``.
    """
    x_text = x_text_f * span
    y_lim = 1.10 * span
    min_gap = min_gap_f * 2.0 * y_lim
    for side in (+1, -1):
        grp = sorted([it for it in items if (it[0] >= 0) == (side > 0)],
                     key=lambda it: -it[1])
        if not grp:
            continue
        ys = [it[1] for it in grp]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        ys[-1] = max(ys[-1], -y_lim)
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = max(ys[i], ys[i + 1] + min_gap)
        ys[0] = min(ys[0], y_lim)
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        for (tx, ty, name), y in zip(grp, ys):
            ax.plot([tx, side * x_text * 0.96], [ty, y], lw=0.6,
                    color="#b3b3b3", zorder=4)
            ax.text(side * x_text, y, name,
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=fontsize, color=INK, zorder=7)


def odds_ratio(a, b, c, d):
    """OR and 95 % CI from a 2x2 table with the Haldane correction."""
    a, b, c, d = (float(v) + (0.5 if min(a, b, c, d) == 0 else 0.0)
                  for v in (a, b, c, d))
    or_ = (a * d) / (b * c)
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = float(stats.norm.ppf(0.975))
    return or_, or_ * np.exp(-z * se), or_ * np.exp(z * se)


def main():
    set_pub_style()
    rng = np.random.default_rng(663)

    servings, group = [], []
    for k, (_, _, mu, n) in enumerate(PATTERNS):
        mu = np.asarray(mu, dtype=float)
        s = mu[None, :] * rng.lognormal(0.0, 0.34, (n, len(FOODS)))
        servings.append(np.clip(s, 0.0, None))
        group.append(np.full(n, k))
    servings = np.vstack(servings)
    group = np.concatenate(group)
    n_tot = servings.shape[0]

    z = (servings - servings.mean(axis=0)) / servings.std(axis=0, ddof=1)
    pca = PCA(n_components=2).fit(z)
    scores = pca.transform(z)
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    var = 100.0 * pca.explained_variance_ratio_

    intake = servings @ CONTENT
    adequacy = 100.0 * intake / REFERENCE[None, :]

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86],
                          left=0.055, right=0.965, top=0.838, bottom=0.152,
                          wspace=0.26, hspace=0.52)
    ax_p = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1], projection="polar")
    ax_f = fig.add_subplot(gs[1, 0 + 1])

    # ---- main: PCA biplot --------------------------------------------------
    ax_p.axhline(0.0, color="#cccccc", lw=0.7, zorder=1)
    ax_p.axvline(0.0, color="#cccccc", lw=0.7, zorder=1)
    span = float(np.abs(scores).max())
    centroid = {}
    for k, (name, col, _, _) in enumerate(PATTERNS):
        m = group == k
        ax_p.scatter(scores[m, 0], scores[m, 1], s=11, color=col, alpha=0.50,
                     edgecolors="none", zorder=3)
        cov_ellipse(ax_p, scores[m], col)
        c = scores[m].mean(axis=0)
        centroid[name] = c
        ax_p.plot([c[0]], [c[1]], marker="X", ms=6.5, color=col, mec="white",
                  mew=0.9, zorder=6)
        halo_text(ax_p, c[0], c[1] + 0.10 * span,
                  f"{name}\nn = {int(m.sum())}", ha="center", va="bottom",
                  fontsize=6.8, fontweight="bold", color=col, halo="white",
                  halo_lw=2.4, linespacing=1.25, zorder=8)

    k_arrow = 0.72 * span / float(np.abs(load).max())
    tips = []
    for j, name in enumerate(FOODS):
        vx, vy = load[j, 0] * k_arrow, load[j, 1] * k_arrow
        ax_p.annotate("", xy=(vx, vy), xytext=(0.0, 0.0),
                      arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK,
                                      shrinkA=0, shrinkB=0, alpha=0.85),
                      zorder=5)
        tips.append((vx, vy, name))
    outside_labels(ax_p, tips, span)

    ax_p.set_xlim(-span * 1.60, span * 1.60)
    ax_p.set_ylim(-span * 1.20, span * 1.20)
    ax_p.set_xlabel(f"PC1  ({var[0]:.1f}% of variance)", fontsize=8.0)
    ax_p.set_ylabel(f"PC2  ({var[1]:.1f}% of variance)", fontsize=8.0)
    ax_p.tick_params(labelsize=7.0)
    ax_p.grid(lw=0.5, color="#f4f4f4", zorder=0)
    ax_p.set_title("Ten food groups collapse onto two eating axes",
                   fontsize=9.0, fontweight="bold", pad=7)
    ax_p.text(0.5, 0.012, "arrows are food-group loadings; dashed ellipses "
              "hold 2 SD around each centroid", transform=ax_p.transAxes,
              ha="center", va="bottom", fontsize=6.2, color=MUTED)

    # ---- aux 1: nutrient adequacy radar ------------------------------------
    n_ax = len(NUTRIENTS)
    ang = np.linspace(0.0, 2.0 * np.pi, n_ax, endpoint=False)
    ring = np.concatenate([ang, ang[:1]])
    rmax = 0.0
    means = {}
    for k, (name, col, _, _) in enumerate(PATTERNS):
        v = adequacy[group == k].mean(axis=0)
        means[name] = v
        rmax = max(rmax, float(v.max()))
    for k, (name, col, _, _) in enumerate(PATTERNS):
        v = np.concatenate([means[name], means[name][:1]])
        ax_r.plot(ring, v, color=col, lw=1.4, zorder=4, label=name)
        ax_r.fill(ring, v, color=col, alpha=0.09, zorder=3)
    ax_r.plot(ring, np.full(n_ax + 1, 100.0), color=RULE, lw=1.0,
              ls=(0, (3, 2)), zorder=5)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels(NUTRIENTS, fontsize=6.2)
    ax_r.tick_params(axis="x", pad=-1.0)
    ax_r.set_ylim(0.0, rmax * 1.10)
    ticks = np.arange(100.0, rmax * 1.05, 100.0)
    ax_r.set_yticks(ticks)
    ax_r.set_yticklabels([f"{t:.0f}" for t in ticks], fontsize=6.0,
                         color=MUTED)
    ax_r.set_rlabel_position(100.0 / len(NUTRIENTS) * 3.0)
    ax_r.grid(lw=0.5, color="#e2e2e2")
    ax_r.spines["polar"].set_color("#cccccc")
    ax_r.spines["polar"].set_linewidth(0.7)
    ax_r.set_title("Nutrient intake, % of reference\n"
                   "(dashed ring = 100%)", fontsize=8.5, fontweight="bold",
                   pad=9, linespacing=1.35)
    ax_r.legend(loc="center left", bbox_to_anchor=(-0.34, 0.06),
                frameon=False, fontsize=6.4, handlelength=1.5,
                labelspacing=0.34)

    # ---- aux 2: outcome odds ratios ---------------------------------------
    med, wes = 0, 2
    rows = []
    for name, p_med, p_wes in OUTCOMES:
        e_m = int(rng.binomial((group == med).sum(), p_med))
        e_w = int(rng.binomial((group == wes).sum(), p_wes))
        n_m, n_w = int((group == med).sum()), int((group == wes).sum())
        or_, lo, hi = odds_ratio(e_w, n_w - e_w, e_m, n_m - e_m)
        rows.append(dict(name=name, or_=or_, lo=lo, hi=hi, e_m=e_m, e_w=e_w,
                         n_m=n_m, n_w=n_w))

    yy = np.arange(len(rows))[::-1]
    ax_f.axvline(1.0, color="#808080", lw=0.9, zorder=2)
    for y, r in zip(yy, rows):
        sig = r["lo"] > 1.0 or r["hi"] < 1.0
        col = (RED if r["or_"] > 1.0 else GREEN) if sig else MUTED
        ax_f.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.1, zorder=4)
        ax_f.plot([r["or_"]], [y], marker="s", ms=4.6, color=col, mec="white",
                  mew=0.6, zorder=5)
        ax_f.text(0.985, y, f"{r['or_']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                  transform=ax_f.get_yaxis_transform(), ha="right",
                  va="center", fontsize=6.0, color=SUB, zorder=6)
    ax_f.set_xscale("log")
    ax_f.set_xlim(0.55, max(r["hi"] for r in rows) * 5.2)
    ax_f.set_xticks([0.7, 1.0, 2.0, 4.0, 8.0])
    ax_f.set_xticklabels(["0.7", "1", "2", "4", "8"], fontsize=6.5)
    ax_f.minorticks_off()
    ax_f.set_ylim(-0.8, len(rows) - 0.2)
    ax_f.set_yticks(yy)
    ax_f.set_yticklabels([r["name"] for r in rows], fontsize=6.6)
    ax_f.tick_params(axis="y", length=0)
    ax_f.set_xlabel("odds ratio, Western versus Mediterranean (-)",
                    fontsize=7.4)
    ax_f.grid(axis="x", lw=0.5, color="#f2f2f2", zorder=0)
    n_sig = sum(1 for r in rows if r["lo"] > 1.0 or r["hi"] < 1.0)
    ax_f.set_title(f"All five odds ratios exceed one,\n{n_sig} of them "
                   "significantly", fontsize=8.5, fontweight="bold", pad=4,
                   linespacing=1.35)

    worst = max(rows, key=lambda r: r["or_"])
    gap = means["Western"] - means["Mediterranean"]
    j = int(np.argmin(gap))
    panel_labels_inches(fig, [ax_p, ax_r, ax_f], dx=-0.42, dy=0.12)

    print(f"PCA explained variance: PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%, "
          f"n = {n_tot}")
    for name, _, _, _ in PATTERNS:
        print(f"  {name:<15s} adequacy "
              + " ".join(f"{v:5.0f}" for v in means[name]))
    for r in rows:
        print(f"  {r['name']:<22s} OR {r['or_']:.2f} "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]  "
              f"{r['e_w']}/{r['n_w']} vs {r['e_m']}/{r['n_m']}")
    savefig(fig, "663_nut_dietary_pattern.png")


if __name__ == "__main__":
    main()
