"""Meta-analysis forest plot with subgroup and overall pooled effects.

Every study contributes a weight-scaled effect square with its 95% interval;
subgroups close with an inverse-variance pooled diamond and heterogeneity
statistics, and the grand total sits at the foot. Study labels run down the
left margin, numeric effect and weight columns down the right, all on a
log-scaled hazard-ratio axis anchored at the line of no effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import stats
from pubstyle import set_pub_style, savefig

SQUARE_C = "#3f5d80"
SUB_C = "#5b6e9e"
ALL_C = "#a8323f"
XTICKS = [0.25, 0.5, 1.0, 2.0, 4.0]
XLIM = (0.18, 5.2)

SUBGROUPS = [
    ("Randomised trials", -0.40, 0.24, [
        ("Alvarez 2016", 412), ("Bianchi 2018", 268), ("Chen 2019", 831),
        ("Duarte 2020", 154), ("Eriksson 2021", 605), ("Fontaine 2022", 349),
    ]),
    ("Observational cohorts", -0.16, 0.34, [
        ("Gupta 2017", 1120), ("Haruki 2019", 486), ("Ivanov 2020", 233),
        ("Jimenez 2021", 742), ("Kowalski 2023", 918),
    ]),
]


def pool(lhr, se):
    w = 1.0 / se ** 2
    mu = np.sum(w * lhr) / np.sum(w)
    se_mu = np.sqrt(1.0 / np.sum(w))
    q = np.sum(w * (lhr - mu) ** 2)
    df = max(len(lhr) - 1, 1)
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return mu, se_mu, q, df, i2, stats.chi2.sf(q, df)


def diamond(ax, y, lo, mid, hi, color, half=0.34):
    ax.add_patch(Polygon([[lo, y], [mid, y - half], [hi, y], [mid, y + half]],
                         closed=True, facecolor=color, edgecolor="#1a1a1a",
                         lw=0.6, zorder=5))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(404)

    blocks = []
    for name, mu, tau, studies in SUBGROUPS:
        lhr = rng.normal(mu, tau, len(studies))
        se = rng.uniform(0.08, 0.17, len(studies)) * \
            (800 / np.array([s[1] for s in studies])) ** 0.25
        blocks.append((name, studies, lhr, se))
    all_w = np.concatenate([1 / b[3] ** 2 for b in blocks])
    w_pct = 100 * all_w / all_w.sum()

    # ---- row scaffold ------------------------------------------------------
    rows, cursor, wi = [], 0.0, 0
    for name, studies, lhr, se in blocks:
        rows.append(("group", cursor, name)); cursor += 1
        for (sname, n), l, s in zip(studies, lhr, se):
            rows.append(("study", cursor, (sname, n, l, s, w_pct[wi])))
            cursor += 1; wi += 1
        rows.append(("subtotal", cursor, (name, pool(lhr, se),
                                          w_pct[wi - len(studies):wi].sum())))
        cursor += 2.5
    lhr_all = np.concatenate([b[2] for b in blocks])
    se_all = np.concatenate([b[3] for b in blocks])
    rows.append(("overall", cursor, pool(lhr_all, se_all)))
    total = cursor

    fig = plt.figure(figsize=(9.4, 6.4))
    ax = fig.add_axes([0.305, 0.115, 0.395, 0.760])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(total + 0.9, -1.5)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{v:g}" for v in XTICKS])
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([])
    ax.axvline(1.0, color="#333333", lw=0.8, zorder=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=3)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Hazard ratio (log scale)", labelpad=2)

    tr = ax.get_yaxis_transform()
    X_NAME, X_N, X_EFF, X_W = -0.70, -0.155, 1.055, 1.520

    # ---- header ------------------------------------------------------------
    for x, txt, ha in ((X_NAME, "Study", "left"), (X_N, "No.", "right"),
                       (X_EFF, "HR [95% CI]", "left"),
                       (X_W + 0.24, "Weight", "right")):
        ax.text(x, -1.05, txt, transform=tr, ha=ha, va="center", fontsize=7.8,
                fontweight="bold", color="#1a1a1a")
    ax.plot([X_NAME, X_W + 0.30], [-0.62, -0.62], transform=tr,
            color="#333333", lw=0.7, clip_on=False)

    for kind, y, payload in rows:
        if kind == "group":
            ax.text(X_NAME, y, payload, transform=tr, ha="left", va="center",
                    fontsize=7.8, fontweight="bold", color="#33475e")
        elif kind == "study":
            sname, n, l, se, w = payload
            hr, lo, hi = np.exp(l), np.exp(l - 1.96 * se), np.exp(l + 1.96 * se)
            ax.plot([max(lo, XLIM[0] * 1.02), min(hi, XLIM[1] * 0.98)], [y, y],
                    color="#444444", lw=0.9, solid_capstyle="butt", zorder=3)
            ax.scatter([hr], [y], s=18 + w * 12, marker="s", color=SQUARE_C,
                       edgecolor="#22344a", lw=0.5, zorder=4)
            ax.text(X_NAME + 0.055, y, sname, transform=tr, ha="left",
                    va="center", fontsize=7.4)
            ax.text(X_N, y, f"{n:,}", transform=tr, ha="right", va="center",
                    fontsize=7.2, color="#555555")
            ax.text(X_EFF, y, f"{hr:.2f} [{lo:.2f}, {hi:.2f}]", transform=tr,
                    ha="left", va="center", fontsize=7.2)
            ax.text(X_W + 0.24, y, f"{w:.1f}%", transform=tr, ha="right",
                    va="center", fontsize=7.2, color="#555555")
        elif kind == "subtotal":
            name, (mu, se_mu, q, df, i2, pq), wsum = payload
            hr, lo, hi = np.exp(mu), np.exp(mu - 1.96 * se_mu), \
                np.exp(mu + 1.96 * se_mu)
            diamond(ax, y, lo, hr, hi, SUB_C)
            ax.text(X_NAME + 0.055, y, f"Subtotal: {name}", transform=tr,
                    ha="left", va="center", fontsize=7.4, fontweight="bold")
            ax.text(X_EFF, y, f"{hr:.2f} [{lo:.2f}, {hi:.2f}]", transform=tr,
                    ha="left", va="center", fontsize=7.2, fontweight="bold")
            ax.text(X_W + 0.24, y, f"{wsum:.1f}%", transform=tr, ha="right",
                    va="center", fontsize=7.2, fontweight="bold",
                    color="#555555")
            ax.text(X_NAME + 0.055, y + 0.80,
                    f"Heterogeneity: $I^2$ = {i2:.0f}%, "
                    f"$Q$ = {q:.1f} (df = {df}), $P$ = {pq:.2f}",
                    transform=tr, ha="left", va="center", fontsize=6.6,
                    style="italic", color="#666666")
        else:
            mu, se_mu, q, df, i2, pq = payload
            hr, lo, hi = np.exp(mu), np.exp(mu - 1.96 * se_mu), \
                np.exp(mu + 1.96 * se_mu)
            ax.plot([X_NAME, X_W + 0.30], [y - 1.30, y - 1.30], transform=tr,
                    color="#333333", lw=0.7, clip_on=False)
            diamond(ax, y, lo, hr, hi, ALL_C, half=0.40)
            ax.axvline(hr, color=ALL_C, ls=(0, (4, 3)), lw=0.8, zorder=1)
            ax.text(X_NAME, y, "Overall (fixed effect)", transform=tr,
                    ha="left", va="center", fontsize=7.8, fontweight="bold",
                    color=ALL_C)
            ax.text(X_EFF, y, f"{hr:.2f} [{lo:.2f}, {hi:.2f}]", transform=tr,
                    ha="left", va="center", fontsize=7.4, fontweight="bold",
                    color=ALL_C)
            ax.text(X_W + 0.24, y, "100.0%", transform=tr, ha="right",
                    va="center", fontsize=7.4, fontweight="bold", color=ALL_C)
            z = mu / se_mu
            pz = 2 * stats.norm.sf(abs(z))
            ptxt = "$P$ < 0.0001" if pz < 1e-4 else f"$P$ = {pz:.4f}"
            ax.text(X_NAME, y + 0.85,
                    f"Test for overall effect: $Z$ = {abs(z):.2f}, "
                    f"{ptxt}    $I^2$ = {i2:.0f}%",
                    transform=tr, ha="left", va="center", fontsize=6.8,
                    style="italic", color="#666666")

    # ---- direction cue -----------------------------------------------------
    for xa, xb, txt, col in ((0.62, 0.30, "Favours treatment", SQUARE_C),
                             (1.62, 3.35, "Favours control", "#a8323f")):
        ax.annotate("", xy=(xb, total + 2.35), xytext=(xa, total + 2.35),
                    xycoords="data", textcoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col,
                                    mutation_scale=8))
        ax.text(np.sqrt(xa * xb), total + 3.0, txt, ha="center", va="center",
                fontsize=7.2, color=col, clip_on=False)

    savefig(fig, "77_forest_plot.png")


if __name__ == "__main__":
    main()
