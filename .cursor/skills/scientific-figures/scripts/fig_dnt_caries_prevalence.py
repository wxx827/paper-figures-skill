"""Caries epidemiology: DMFT decomposed by age group, with prevalence and risk.

The main panel splits the DMFT index into its three components so that the
same total can be read as two different diseases: an untreated-decay burden
and a restored burden.  The two cohorts share one stack geometry and one y
axis, and the total lines are literally the sums of the stacks drawn beneath
them, so the line can never drift away from the bars.  The prevalence curve
carries a binomial 95 % confidence band computed from the per-age sample
sizes, and the forest panel colours a factor only when its interval actually
excludes an odds ratio of one.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.58, 4.56)
AUX_T = (7.00, 3.52, 3.12, 1.90)
AUX_B = (7.00, 0.86, 3.12, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#d5d5d5"
DECAY, MISS, FILL = "#b23a48", "#3a3a3a", "#3b5b92"
URBAN, RURAL = "#2e7d5b", "#c9a227"

AGES = ["6-11", "12-17", "18-34", "35-49", "50-64", "65+"]
AGE_MID = np.array([8.5, 14.5, 26.0, 42.0, 57.0, 72.0])

FACTORS = ["Sugary drinks > 2/day", "No fluoride dentifrice",
           "Irregular dental visits", "Lowest income quintile",
           "Visible plaque index > 2", "Current smoker",
           "Xerostomic medication"]

FOOT = ("Synthetic cross-sectional survey. DMFT counts decayed, missing and "
        "filled permanent teeth per person; solid bars are the urban cohort "
        "and hatched bars the rural cohort, and each total line is the sum of "
        "the stack beneath it. Prevalence bands are binomial 95 % intervals "
        "from the per-age sample sizes; odds ratios are adjusted for age and "
        "sex. Illustrative data only.")


def make_cohort(rng, shift, care):
    """Decayed / missing / filled teeth per person for the six age groups."""
    t = np.linspace(0.0, 1.0, len(AGES))
    total = 0.6 + 12.8 * t ** 1.55 + shift * t
    total = total + rng.normal(0.0, 0.22, total.size)
    frac_m = np.clip(0.03 + 0.42 * t ** 2.3, 0.0, 0.6)
    frac_d = np.clip(0.55 - 0.34 * care - 0.16 * t, 0.06, 0.9) * (1.0 - frac_m)
    miss = total * frac_m
    dec = total * frac_d
    fil = total - miss - dec
    return np.vstack([dec, miss, fil])


def prevalence(rng, level, n_per_age):
    """Logistic caries prevalence plus its binomial 95 % band."""
    a = np.linspace(6.0, 80.0, 160)
    p = 1.0 / (1.0 + np.exp(-(a - level) / 9.5))
    p = np.clip(0.06 + 0.92 * p, 0.0, 0.995)
    n = np.interp(a, AGE_MID, n_per_age)
    se = np.sqrt(p * (1.0 - p) / n)
    return a, p, np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1)


def main():
    set_pub_style()
    rng = np.random.default_rng(817)

    urb = make_cohort(rng, shift=0.0, care=1.0)
    rur = make_cohort(rng, shift=2.4, care=0.25)
    n_per_age = np.array([412.0, 388.0, 501.0, 466.0, 397.0, 302.0])

    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) stacked DMFT decomposition -----------------------------------
    ax = axes_inches(fig, MAIN)
    x = np.arange(len(AGES), dtype=float)
    w = 0.34
    for k, (data, dx, hatch, col, name) in enumerate(
            ((urb, -0.19, None, URBAN, "urban"),
             (rur, +0.19, "///", RURAL, "rural"))):
        bottom = np.zeros(len(AGES))
        for comp, colour in zip(data, (DECAY, MISS, FILL)):
            ax.bar(x + dx, comp, width=w, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.6, hatch=hatch, zorder=3)
            bottom = bottom + comp
        ax.plot(x + dx, bottom, color=col, lw=1.5, zorder=6,
                ls="-" if k == 0 else (0, (3.4, 2.0)),
                marker="o" if k == 0 else "s", ms=3.6, mfc="white", mew=1.2)
        for xi, tot in zip(x + dx, bottom):
            ax.text(xi, tot + 0.38, f"{tot:.1f}", ha="center", va="bottom",
                    fontsize=6.2, color=col)

    ax.grid(axis="y", color=HAIR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(AGES, fontsize=7.4)
    ax.set_xlabel("age group (years)", fontsize=8.0)
    ax.set_ylabel("DMFT index (teeth per person)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    ax.set_ylim(0, max(urb.sum(0).max(), rur.sum(0).max()) * 1.30)
    ax.set_xlim(-0.6, len(AGES) - 0.4)
    ax.set_title("DMFT decomposition by age group and cohort", fontsize=8.8,
                 fontweight="bold", color=INK, pad=4)

    # The gap that matters is untreated decay, not the total burden.
    d_gap = rur[0] - urb[0]
    j = int(np.argmax(d_gap))
    share_u = urb[0].sum() / urb.sum() * 100.0
    share_r = rur[0].sum() / rur.sum() * 100.0
    ax.text(0.025, 0.73,
            f"untreated decay is {share_r:.0f} % of the rural burden\n"
            f"against {share_u:.0f} % urban; widest gap +{d_gap[j]:.1f} D "
            f"teeth at {AGES[j]} y",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=DECAY, linespacing=1.5)

    handles = [Patch(facecolor=DECAY, label="D  decayed"),
               Patch(facecolor=MISS, label="M  missing"),
               Patch(facecolor=FILL, label="F  filled"),
               Line2D([], [], color=URBAN, lw=1.5, marker="o", ms=3.4,
                      mfc="white", label="urban total"),
               Line2D([], [], color=RURAL, lw=1.5, ls=(0, (3.4, 2.0)),
                      marker="s", ms=3.4, mfc="white", label="rural total")]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
              fontsize=6.8, handlelength=1.6, columnspacing=1.3,
              labelspacing=0.32)

    # ---- (b) prevalence with 95 % band ------------------------------------
    ax_b = axes_inches(fig, AUX_T)
    curves = []
    for level, col, name, ls in ((34.0, URBAN, "urban", "-"),
                                 (24.0, RURAL, "rural", (0, (3.4, 2.0)))):
        a, p, lo, hi = prevalence(rng, level, n_per_age)
        ax_b.fill_between(a, lo * 100, hi * 100, color=col, alpha=0.18, lw=0)
        ax_b.plot(a, p * 100, color=col, lw=1.5, ls=ls, label=name)
        curves.append((a, p, col, name))
    ax_b.axhline(50.0, color=RULE, lw=0.7, ls=(0, (2.5, 2.0)))
    for a, p, col, name in curves:
        a50 = float(np.interp(0.5, p, a))
        ax_b.plot([a50], [50.0], marker="v", ms=4.0, color=col, mec="white",
                  mew=0.6, clip_on=False)
        ax_b.text(a50, 55.0, f"{a50:.0f} y", ha="center", va="bottom",
                  fontsize=6.2, color=col)
    ax_b.set_xlabel("age (years)", fontsize=8.0)
    ax_b.set_ylabel("caries prevalence (%)", fontsize=8.0)
    ax_b.set_xlim(6, 80)
    ax_b.set_ylim(0, 105)
    ax_b.tick_params(labelsize=6.6)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.5,
                handlelength=1.6)
    ax_b.set_title(f"Prevalence with 95 % band (n={int(n_per_age.sum())})",
                   fontsize=8.5, fontweight="bold", color=INK, pad=3)

    # ---- (c) risk-factor forest -------------------------------------------
    ax_c = axes_inches(fig, AUX_B)
    log_or = rng.normal(0.42, 0.30, len(FACTORS)) + np.linspace(0.55, -0.10,
                                                                len(FACTORS))
    se = rng.uniform(0.10, 0.26, len(FACTORS))
    order = np.argsort(log_or)
    log_or, se = log_or[order], se[order]
    names = [FACTORS[i] for i in order]
    orv = np.exp(log_or)
    lo, hi = np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)
    yv = np.arange(len(names), dtype=float)
    sig = lo > 1.0

    ax_c.axvline(1.0, color=RULE, lw=0.8)
    for i in range(len(names)):
        col = DECAY if sig[i] else MUTED
        ax_c.plot([lo[i], hi[i]], [yv[i], yv[i]], color=col, lw=0.9,
                  solid_capstyle="butt")
        for e in (lo[i], hi[i]):
            ax_c.plot([e, e], [yv[i] - 0.18, yv[i] + 0.18], color=col, lw=0.9)
        ax_c.plot([orv[i]], [yv[i]], marker="s", ms=3.8, color=col,
                  mec="white", mew=0.5)
    ax_c.set_yticks([])
    ax_c.set_ylim(-1.6, len(names) - 0.25)
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.55, 5.4)
    ax_c.set_xticks([0.6, 1.0, 2.0, 4.0])
    ax_c.set_xticklabels(["0.6", "1.0", "2.0", "4.0"], fontsize=6.6)
    ax_c.minorticks_off()
    ax_c.set_xlabel("adjusted odds ratio (-), log scale", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    # Row names live inside the panel: as y tick labels they would reach a
    # full inch into the neighbouring main panel.
    for i, nm in enumerate(names):
        ax_c.text(0.575, yv[i] + 0.30, f"{nm}   OR {orv[i]:.2f}", ha="left",
                  va="center", fontsize=6.1,
                  color=INK if sig[i] else SUB)
    ax_c.text(0.575, -1.30, f"{int(sig.sum())} of {len(names)} factors "
              "exclude OR = 1", ha="left", va="center", fontsize=6.3,
              color=SUB)
    ax_c.set_title("Risk factors for untreated decay", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.62, dy=0.10)
    savefig(fig, "817_dnt_caries_prevalence.png")


if __name__ == "__main__":
    main()
