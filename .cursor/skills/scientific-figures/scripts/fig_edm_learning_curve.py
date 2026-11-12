"""Learning curves on one skill: the group mean hides how long the tail is.

Every learner gets a power-law curve of their own, and the two things the
figure claims - the median number of attempts to mastery and the shape of the
distribution behind it - are read off those drawn curves by interpolation, not
from the closed form.  That matters because the curves are censored: the panel
only runs to thirty opportunities and some learners have not crossed the
threshold by then, so the mean curve keeps rising while a quarter of the
cohort is still below the line.  The two prior-knowledge groups share one
generative model and differ only in where they start, which is what makes the
rate-against-start panel readable rather than tautological.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
BLUE = "#3b5b92"          # low prior knowledge
ORANGE = "#c0603f"        # high prior knowledge
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
MAIN = (0.78, 1.05, 5.45, 4.35)
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

N_OPP = 30                # practice opportunities shown
N_PER = 32                # learners per prior-knowledge group
MASTERY = 80.0            # mastery threshold, percent correct
GROUPS = [("low prior knowledge", BLUE, 0.26, 0.30),
          ("high prior knowledge", ORANGE, 0.48, 0.36)]


def cohort(seed=897):
    """Power-law learning curves: accuracy (%) per learner per opportunity.

    p(n) = p_inf - (p_inf - p0) n^-r, the standard power law of practice.
    Only p0 differs between the groups; the rate and the asymptote are drawn
    from the same distributions in both.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(1, N_OPP + 1, dtype=float)
    out = []
    for name, col, p0_mu, p0_sd in GROUPS:
        p0 = np.clip(rng.normal(p0_mu, p0_sd * 0.28, N_PER), 0.05, 0.62)
        rate = np.clip(rng.lognormal(np.log(0.65), 0.35, N_PER), 0.08, 1.60)
        pinf = np.clip(rng.normal(0.945, 0.030, N_PER), 0.82, 0.995)
        curves = pinf[:, None] - (pinf - p0)[:, None] * n[None, :] ** -rate[:, None]
        out.append(dict(name=name, color=col, p0=p0 * 100.0, rate=rate,
                        curves=np.clip(curves, 0.0, 1.0) * 100.0))
    return n, out


def attempts_to_mastery(n, curve, thresh=MASTERY):
    """First opportunity at which a drawn curve reaches the threshold.

    Interpolated between the two bracketing opportunities; NaN if the curve
    never gets there inside the window (censored).
    """
    hit = np.flatnonzero(curve >= thresh)
    if hit.size == 0:
        return np.nan
    j = int(hit[0])
    if j == 0:
        return float(n[0])
    y0, y1 = curve[j - 1], curve[j]
    return float(n[j - 1] + (thresh - y0) / (y1 - y0) * (n[j] - n[j - 1]))


def main():
    set_pub_style()
    n, groups = cohort()

    for g in groups:
        g["need"] = np.array([attempts_to_mastery(n, c) for c in g["curves"]])
        g["mean"] = g["curves"].mean(axis=0)
        g["ci"] = 1.96 * g["curves"].std(axis=0, ddof=1) / np.sqrt(N_PER)
        ok = np.isfinite(g["need"])
        g["median"] = float(np.median(g["need"][ok])) if ok.any() else np.nan
        g["censored"] = int((~ok).sum())
    need_all = np.concatenate([g["need"] for g in groups])
    censored = int(np.sum(~np.isfinite(need_all)))
    finite = need_all[np.isfinite(need_all)]
    gap = groups[0]["median"] - groups[1]["median"]

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax_c = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    # ---- main: every learner, then the two group means --------------------
    ax_c.axhline(MASTERY, color=ACCENT, lw=1.0, ls=(0, (4.2, 2.0)), zorder=5)
    ax_c.text(1.15, MASTERY + 1.4,
              f"mastery threshold {MASTERY:.0f} % correct", ha="left",
              va="bottom", fontsize=6.8, color=ACCENT)
    for g in groups:
        for c in g["curves"]:
            ax_c.plot(n, c, color=g["color"], lw=0.55, alpha=0.22, zorder=2)
        ax_c.fill_between(n, g["mean"] - g["ci"], g["mean"] + g["ci"],
                          color=g["color"], alpha=0.28, lw=0.0, zorder=3)
        ax_c.plot(n, g["mean"], color=g["color"], lw=2.4, zorder=6,
                  label=f"{g['name']} (n = {N_PER} learners)")
    ax_c.plot([], [], color=GREY, lw=0.55, alpha=0.55,
              label="individual learner, fitted power law")
    ax_c.plot([], [], color=GREY, lw=5.0, alpha=0.28,
              label="95 % CI of the group mean")

    for g in groups:
        m = g["median"]
        ax_c.plot([m, m], [0.0, MASTERY], color=g["color"], lw=0.9,
                  ls=(0, (2.2, 1.8)), zorder=5)
        ax_c.plot([m], [MASTERY], marker="v", ms=5.0, color=g["color"],
                  zorder=7)
        ax_c.text(m + 0.28, 2.0,
                  f"median {m:.1f}\nopportunities", ha="left", va="bottom",
                  fontsize=6.6, color=g["color"], linespacing=1.35)
    ax_c.annotate(
        f"starting {groups[1]['p0'].mean() - groups[0]['p0'].mean():.0f} "
        f"points higher buys {gap:.1f} fewer\npractice opportunities to the "
        f"same threshold",
        xy=(groups[1]["median"], MASTERY),
        xytext=(groups[0]["median"] + 1.6, MASTERY - 21.0),
        fontsize=6.9, color=INK, ha="left", va="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=2.0,
                        shrinkB=3.0))
    ax_c.text(N_OPP - 0.3, 15.5,
              f"{censored} of {2 * N_PER} learners are still below threshold "
              f"at\nopportunity {N_OPP}; the group means never show them",
              ha="right", va="bottom", fontsize=6.7, color=MUTED,
              linespacing=1.4)

    ax_c.set_xlim(1.0, N_OPP)
    ax_c.set_ylim(0.0, 100.0)
    ax_c.set_xticks(np.arange(0, N_OPP + 1, 5)[1:])
    ax_c.set_yticks(np.arange(0, 101, 20))
    ax_c.set_xlabel("practice opportunities on the skill (attempts)",
                    fontsize=8.0)
    ax_c.set_ylabel("accuracy on the next attempt (% correct)", fontsize=8.0)
    ax_c.tick_params(labelsize=7.2)
    ax_c.grid(axis="y", lw=0.5, color="#e6e6e6")
    ax_c.set_title("Power-law practice: fast early gains, a long flat "
                   "approach to the asymptote",
                   fontsize=9.0, fontweight="bold", color=INK, pad=10.0)
    ax_c.legend(loc="lower right", frameon=False, fontsize=6.8,
                handlelength=2.0, labelspacing=0.32, borderaxespad=0.6,
                bbox_to_anchor=(1.0, 0.24))

    # ---- aux (b): how many opportunities mastery actually took ------------
    edges = np.arange(0.0, N_OPP + 2.0, 2.0)
    bottom = np.zeros(edges.size - 1)
    for g in groups:
        h, _ = np.histogram(g["need"][np.isfinite(g["need"])], bins=edges)
        ax_h.bar(edges[:-1], h, width=2.0 * 0.92, bottom=bottom, align="edge",
                 color=g["color"], alpha=0.85, edgecolor="white",
                 linewidth=0.4, zorder=3, label=g["name"])
        bottom = bottom + h
    top = float(max(bottom.max(), censored))
    for g in groups:
        ax_h.axvline(g["median"], color=g["color"], lw=1.0,
                     ls=(0, (3.2, 2.0)), zorder=4)
    ax_h.bar([N_OPP + 0.6], [censored], width=1.6, align="edge",
             color=HAIR, edgecolor=GREY, linewidth=0.5, hatch="///",
             zorder=3)
    ax_h.text(N_OPP + 1.4, censored + top * 0.04,
              f"{censored}\ncensored", ha="center", va="bottom", fontsize=6.2,
              color=GREY, linespacing=1.3)
    ax_h.text(0.46, 0.74,
              f"median {np.median(finite):.1f}, mean {finite.mean():.1f}\n"
              f"opportunities: the tail\npulls the mean to the right",
              transform=ax_h.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.4)
    ax_h.set_xlim(0.0, N_OPP + 3.0)
    ax_h.set_ylim(0.0, top * 1.40)
    ax_h.set_xticks(np.arange(0, N_OPP + 1, 10))
    ax_h.set_xlabel("opportunities to reach mastery (attempts)", fontsize=7.4)
    ax_h.set_ylabel("learners (count)", fontsize=7.4)
    ax_h.tick_params(labelsize=6.6)
    ax_h.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_h.set_title("A right-skewed cost, not an average one", fontsize=8.5,
                   fontweight="bold", color=INK)
    ax_h.legend(loc="upper left", frameon=False, fontsize=6.2,
                handlelength=1.0, labelspacing=0.24, borderaxespad=0.35)

    # ---- aux (c): learning rate against where the learner started ---------
    p0_all = np.concatenate([g["p0"] for g in groups])
    rate_all = np.concatenate([g["rate"] for g in groups])
    for g in groups:
        ax_s.scatter(g["p0"], g["rate"], s=15, facecolor=g["color"],
                     edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3)
    coef = np.polyfit(p0_all, rate_all, 1)
    xg = np.linspace(p0_all.min(), p0_all.max(), 50)
    ax_s.plot(xg, np.polyval(coef, xg), color=GREY, lw=1.1,
              ls=(0, (3.4, 2.0)), zorder=4)
    r_pearson = float(np.corrcoef(p0_all, rate_all)[0, 1])
    ax_s.text(0.975, 0.94,
              f"Pearson r = {r_pearson:+.2f} (n = {p0_all.size} learners):\n"
              f"where you start says little about how fast you climb",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=6.3,
              color=SUB, linespacing=1.4)
    ax_s.set_xlim(0.0, 70.0)
    ax_s.set_ylim(0.0, float(rate_all.max()) * 1.30)
    ax_s.set_xlabel("accuracy at the first opportunity (% correct)",
                    fontsize=7.4)
    ax_s.set_ylabel("learning rate r (-)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.6)
    ax_s.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_s.set_title("Starting level against learning rate", fontsize=8.5,
                   fontweight="bold", color=INK)


    note = (
        f"Power law of practice p(n) = p_inf - (p_inf - p0) n^-r for "
        f"{2 * N_PER} simulated learners over {N_OPP} opportunities on one "
        f"skill; groups differ only in p0 (mean "
        f"{groups[0]['p0'].mean():.0f} % against "
        f"{groups[1]['p0'].mean():.0f} % at the first attempt), while r and "
        f"p_inf come from the same distributions. Opportunities to mastery "
        f"are interpolated on each drawn curve at the "
        f"{MASTERY:.0f} % threshold, so panel (b) counts the same crossings "
        f"the main panel shows: median {groups[0]['median']:.1f} against "
        f"{groups[1]['median']:.1f} opportunities, mean "
        f"{finite.mean():.1f} over the {finite.size} learners who arrive and "
        f"{censored} still censored at opportunity {N_OPP}. Bands are 95 % "
        f"CIs of the group mean, 1.96 SD / sqrt(n).")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 195), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_labels_inches(fig, [ax_c, ax_h, ax_s],
                        labels=["(a)", "(b)", "(c)"], dx=-0.56, dy=0.12)
    savefig(fig, "897_edm_learning_curve.png")


if __name__ == "__main__":
    main()
