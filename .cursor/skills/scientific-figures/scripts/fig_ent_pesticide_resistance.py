"""Resistance evolution: spray pressure written into the dose-response curve.

One allele-frequency simulation drives all three panels.  Each population is
selected generation by generation at a rate set by its own application
schedule, and its LC50 is that allele frequency mapped onto the resistance
ratio - so the main panel, the probit shift in (b) and the frequency
trajectories in (c) cannot disagree with one another.  The grey background
bars are the schedule itself, which is why the high-pressure curve bends the
generation the area-wide rotation begins.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_label_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"
RED = "#b23a48"
GREY = "#808080"
BAR = "#d5d5d5"

W, H = 10.4, 6.4
N_GEN = 25
ROTATION = 18                 # generation the area-wide rotation starts
LC50_S = 0.85                 # susceptible baseline, mg/L
RR_MAX = 220.0                # resistance ratio at fixation
SEL_PER_SPRAY = 0.14          # selection coefficient per application
COST = 0.020                  # fitness cost of the resistance allele
Q0 = 0.004                    # starting allele frequency
SLOPE_S = 2.30                # probit slope, susceptible strain
SLOPE_R = 1.55                # probit slope, resistant strain
N_INSECTS = 30                # insects per dose in the bioassay

# (label, applications per generation before the rotation)
POPS = [("low pressure", 1), ("moderate pressure", 2), ("high pressure", 4)]
P_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def pop_colors():
    return [lighten(RED, 0.42), RED, darken(RED, 0.34)]


def schedule(sprays):
    """Applications per generation: full programme, then rotation to one."""
    s = np.full(N_GEN, float(sprays))
    s[ROTATION:] = 1.0
    return s


def evolve(sprays):
    """Allele frequency under selection with a constant fitness cost."""
    sched = schedule(sprays)
    q = np.empty(N_GEN)
    q[0] = Q0
    for g in range(1, N_GEN):
        s = SEL_PER_SPRAY * sched[g - 1]
        qs = q[g - 1] * (1.0 + s) / (1.0 + s * q[g - 1])
        q[g] = float(np.clip(qs - COST * qs * (1.0 - qs), 0.0, 1.0))
    return sched, q


def lc50_of(q):
    return LC50_S * np.exp(np.asarray(q) * np.log(RR_MAX))


# ------------------------------------------------------------------- panels
def main_panel(ax, gens, curves, cols):
    ax2 = ax.twinx()
    sched = curves[-1][0]
    ax2.bar(gens, sched, width=0.82, color=BAR, edgecolor="white", lw=0.4,
            zorder=0, label="applications (high-pressure site)")
    ax2.set_ylim(0.0, 13.0)
    ax2.set_yticks([0, 2, 4])
    ax2.set_ylabel("applications per generation", color=SUB, fontsize=8.0)
    ax2.tick_params(axis="y", colors=SUB, labelsize=7.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(HAIR)

    ax.axvline(ROTATION, color=RULE, lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)
    ax.text(ROTATION - 0.4, 0.62, "area-wide rotation\nfrom generation %d"
            % ROTATION, ha="right", va="bottom", fontsize=6.6, color=RULE,
            linespacing=1.45, zorder=6)
    ax.axhline(LC50_S, color=GREY, lw=0.8, ls=(0, (1.8, 1.8)), zorder=2)
    ax.text(0.2, LC50_S * 0.90, "susceptible baseline %.2f mg/L" % LC50_S,
            ha="left", va="top", fontsize=6.6, color=GREY, zorder=6)

    for (label, sprays), (sched_i, q), col in zip(POPS, curves, cols):
        lc = lc50_of(q)
        ax.plot(gens, lc, color=col, lw=1.9, zorder=5,
                label="%s (%d applications/gen)" % (label, sprays))
        ax.plot(gens[::4], lc[::4], ls="none", marker="o", ms=3.4,
                mfc="white", mew=1.0, mec=col, zorder=6)
        rr_i = lc[-1] / LC50_S
        ax.annotate(("%.0f" if rr_i >= 10.0 else "%.1f") % rr_i +
                    r"$\times$",
                    xy=(gens[-1], lc[-1]), xytext=(gens[-1] + 0.5, lc[-1]),
                    ha="left", va="center", fontsize=7.0, color=col,
                    fontweight="bold", clip_on=False)

    ax.set_yscale("log")
    ax.set_ylim(0.42, 330.0)
    ax.set_xlim(-0.8, N_GEN + 1.6)
    ax.set_xlabel("generations under selection (generations)", fontsize=8.0)
    ax.set_ylabel(r"$LC_{50}$ (mg/L)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", which="major", lw=0.5, alpha=0.6)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False, loc="upper left",
              bbox_to_anchor=(0.012, 1.015), handlelength=1.6)
    rr = [lc50_of(q)[-1] / LC50_S for _, q in curves]
    ax.set_title(r"Four applications per generation buy %.0f$\times$ "
                 r"resistance in %d generations" % (rr[-1], N_GEN - 1),
                 fontsize=9.0, fontweight="bold", color=INK)
    return rr


def probit_panel(ax, rng, lc_r, cols):
    dose = np.logspace(np.log10(0.05), np.log10(3000.0), 240)
    strains = [("susceptible strain", LC50_S, SLOPE_S, GREY),
               ("resistant strain", lc_r, SLOPE_R, cols[-1])]
    test = np.logspace(np.log10(0.10), np.log10(2000.0), 8)

    for name, lc, slope, col in strains:
        p = stats.norm.cdf(slope * (np.log10(dose) - np.log10(lc)))
        ax.plot(dose, stats.norm.ppf(np.clip(p, 1e-4, 1 - 1e-4)), color=col,
                lw=1.7, zorder=5, label=name)
        pt = stats.norm.cdf(slope * (np.log10(test) - np.log10(lc)))
        k = rng.binomial(N_INSECTS, np.clip(pt, 0.0, 1.0))
        # 0 % and 100 % have no probit; the convention is to drop them.
        ok = (k > 0) & (k < N_INSECTS)
        ax.plot(test[ok], stats.norm.ppf(k[ok] / N_INSECTS), ls="none",
                marker="o", ms=3.2, mfc="white", mew=0.9, mec=col, zorder=6)
        ax.plot([lc, lc], [stats.norm.ppf(P_LEVELS[0]), 0.0], color=col,
                lw=0.8, ls=(0, (1.8, 1.8)), zorder=4)

    ax.annotate("", xy=(lc_r, 0.42), xytext=(LC50_S, 0.42),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(np.sqrt(LC50_S * lc_r), 0.58,
            r"%.0f$\times$ shift" % (lc_r / LC50_S), ha="center",
            va="bottom", fontsize=6.8, color=INK)

    ax.set_xscale("log")
    ax.set_yticks([stats.norm.ppf(p) for p in P_LEVELS])
    ax.set_yticklabels(["%g" % (100.0 * p) for p in P_LEVELS], fontsize=6.4)
    ax.set_ylim(stats.norm.ppf(P_LEVELS[0]) - 0.15,
                stats.norm.ppf(P_LEVELS[-1]) + 0.15)
    ax.set_xlim(0.05, 3000.0)
    ax.set_xlabel("dose (mg/L)", fontsize=7.4)
    ax.set_ylabel("mortality (%, probit scale)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              bbox_to_anchor=(0.015, 1.01), handlelength=1.3)
    ax.set_title("Resistant strain is flatter and shifted right",
                 fontsize=8.4, fontweight="bold", color=INK)


def freq_panel(ax, gens, curves, cols):
    ax.axvspan(ROTATION, N_GEN - 1, facecolor=SOFT, edgecolor="none",
               zorder=0)
    for (label, _), (_, q), col in zip(POPS, curves, cols):
        ax.plot(gens, q, color=col, lw=1.7, zorder=4, label=label)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=(0, (1.8, 1.8)), zorder=2)
    ax.text(0.4, 0.52, "50 % of alleles", ha="left", va="bottom",
            fontsize=6.4, color=GREY)
    ax.text(ROTATION + 0.4, 0.055, "rotation", ha="left", va="bottom",
            fontsize=6.4, color=SUB)

    ax.set_xlim(-0.8, N_GEN - 0.2)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("generation (generations)", fontsize=7.4)
    ax.set_ylabel("resistance allele frequency (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              bbox_to_anchor=(0.02, 1.02), handlelength=1.3)
    ax.set_title("The allele fixes before the rotation starts",
                 fontsize=8.4, fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(7403)
    cols = pop_colors()
    gens = np.arange(N_GEN, dtype=float)
    curves = [evolve(sprays) for _, sprays in POPS]

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.92, 1.00, 4.88, 4.12))
    ax_pro = axes_inches(fig, (7.10, 3.52, 2.80, 1.72))
    ax_frq = axes_inches(fig, (7.10, 0.95, 2.80, 1.62))

    rr = main_panel(ax_main, gens, curves, cols)
    probit_panel(ax_pro, rng, lc50_of(curves[-1][1])[-1], cols)
    freq_panel(ax_frq, gens, curves, cols)

    fig.text(0.016, 6.14 / H,
             "Resistance is bought one application at a time: %.0f-fold "
             "under high pressure against %.1f-fold under low"
             % (rr[-1], rr[0]),
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H,
             r"$LC_{50}$ = %.2f mg/L $\times$ %.0f$^{q}$, with the allele "
             "frequency $q$ in (c) advanced generation by generation at "
             "%.2f per application minus a %.1f %% fitness cost. Probit "
             "points in (b) are %d insects per dose."
             % (LC50_S, RR_MAX, SEL_PER_SPRAY, 100.0 * COST, N_INSECTS),
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_main, "(a)", dx=-0.72, dy=0.16)
    panel_label_inches(fig, ax_pro, "(b)", dx=-0.62, dy=0.16)
    panel_label_inches(fig, ax_frq, "(c)", dx=-0.62, dy=0.16)

    savefig(fig, "740_ent_pesticide_resistance.png")


if __name__ == "__main__":
    main()
