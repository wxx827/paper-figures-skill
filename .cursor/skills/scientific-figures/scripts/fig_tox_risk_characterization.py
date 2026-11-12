"""Probabilistic risk characterisation: exposure against effect.

The main panel is the classic two-distribution overlay - a predicted exposure
distribution and a species effect distribution on one logarithmic
concentration axis - with the overlap shaded in house red because that
overlap *is* the risk.  The deterministic risk quotient is the ratio of the
upper exposure percentile to the lower effect percentile and is drawn as a
span between the two rules, so the reader can see the quotient rather than
merely read it.  The auxiliary panels put the quotient in context: how it
moves across exposure scenarios, and which assumption it is most sensitive to.

Synthetic data; the quotient, the exceedance probability and every tornado
bar are computed from the same Monte Carlo sample that is plotted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
RED, BLUE, GREEN, GREY = "#b23a48", "#3b5b92", "#2e7d5b", "#b3b3b3"

EXP_MU, EXP_SD = np.log10(5.5), 0.38       # exposure, log10 ug/L
EFF_MU, EFF_SD = np.log10(95.0), 0.45      # species effect, log10 ug/L
N_MC = 200000

SCENARIOS = [("Regional background", 0.16), ("Urban runoff", 0.52),
             ("Downstream of WWTP", 1.00), ("Combined sewer overflow", 1.90),
             ("Industrial discharge", 3.10)]
# parameter, low multiplier on the quotient, high multiplier
SENSITIVITY = [("Effluent dilution factor", 0.42, 2.35),
               ("Assessment factor on $HC_5$", 0.55, 1.90),
               ("Degradation half-life", 0.71, 1.44),
               ("Emission load", 0.80, 1.28),
               ("Suspended-solids sorption", 0.88, 1.14),
               ("Sampling uncertainty", 0.93, 1.08)]


def lognorm_pdf(t, mu, sd):
    """Density in log10-concentration space."""
    return np.exp(-0.5 * ((t - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(626)

    exp_s = rng.normal(EXP_MU, EXP_SD, N_MC)
    eff_s = rng.normal(EFF_MU, EFF_SD, N_MC)
    p_exceed = float(np.mean(exp_s > eff_s))
    pec95 = 10.0 ** float(np.percentile(exp_s, 95))
    pnec = 10.0 ** float(np.percentile(eff_s, 5))
    rq = pec95 / pnec

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0],
                          left=0.062, right=0.975, top=0.865, bottom=0.170,
                          wspace=0.30, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    ax_sc = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, 1])

    # ---- (a) exposure vs effect -------------------------------------------
    t = np.linspace(-0.9, 3.1, 900)
    f_exp = lognorm_pdf(t, EXP_MU, EXP_SD)
    f_eff = lognorm_pdf(t, EFF_MU, EFF_SD)
    overlap = np.minimum(f_exp, f_eff)
    ov_area = float(np.trapezoid(overlap, t))

    ax.fill_between(t, 0.0, f_exp, color=BLUE, alpha=0.20, lw=0, zorder=2)
    ax.fill_between(t, 0.0, f_eff, color=GREEN, alpha=0.20, lw=0, zorder=2)
    ax.plot(t, f_exp, lw=1.7, color=BLUE, zorder=5,
            label="Exposure distribution (PEC)")
    ax.plot(t, f_eff, lw=1.7, color=GREEN, zorder=5,
            label="Effect distribution (species $EC_{50}$)")
    ax.fill_between(t, 0.0, overlap, color=RED, alpha=0.55, lw=0, zorder=4,
                    label="Risk zone: exposure meets effect")

    ymax = max(f_exp.max(), f_eff.max())
    marks = ((np.log10(pnec), GREEN, "right", -0.05,
              f"$PNEC$\n{pnec:.1f} $\\mu$g/L"),
             (np.log10(pec95), BLUE, "left", 0.05,
              f"$PEC_{{95}}$\n{pec95:.1f} $\\mu$g/L"))
    for x0, col, ha, dx, lab in marks:
        ax.plot([x0, x0], [0.0, ymax * 1.02], lw=1.0, ls=(0, (4, 2)),
                color=col, zorder=6)
        ax.text(x0 + dx, ymax * 1.045, lab, ha=ha, va="bottom", fontsize=6.8,
                color=col, linespacing=1.4, zorder=7)

    y_rq = ymax * 0.88
    ax.annotate("", xy=(np.log10(pnec), y_rq), xytext=(np.log10(pec95), y_rq),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=RULE,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.text(0.5 * (np.log10(pec95) + np.log10(pnec)), y_rq * 1.02,
            f"$RQ$ = $PEC_{{95}}$ / $PNEC$ = {rq:.2f}", ha="center",
            va="bottom", fontsize=7.2, color=INK, zorder=7)

    ax.set_xlim(-0.9, 3.1)
    ax.set_ylim(0.0, ymax * 1.32)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("Concentration ($\\mu$g/L, log scale)")
    ax.set_ylabel("Probability density (per log$_{10}$ unit)")
    ax.grid(color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.435, 1.005), frameon=False,
              fontsize=6.8, handlelength=1.7, labelspacing=0.42)
    ax.set_title("Predicted exposure reaches the sensitive tail of the "
                 "effect distribution", fontsize=9.0, fontweight="bold",
                 loc="left", pad=5)
    ax.text(0.017, 0.615,
            f"P(exposure > effect) = {100.0 * p_exceed:.2f}%\n"
            f"distribution overlap = {100.0 * ov_area:.1f}%\n"
            f"Monte Carlo n = {N_MC:,} pairs",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB, linespacing=1.6)

    # ---- (b) risk quotient by scenario ------------------------------------
    labels = [s[0] for s in SCENARIOS]
    rqs = np.array([rq * s[1] for s in SCENARIOS])
    y = np.arange(len(SCENARIOS))
    bcol = [RED if v >= 1.0 else GREEN for v in rqs]
    ax_sc.barh(y, rqs, height=0.62, color=bcol, edgecolor="white",
               linewidth=0.5, zorder=3)
    for k, v in enumerate(rqs):
        ax_sc.text(v + 0.06, k, f"{v:.2f}", va="center", ha="left",
                   fontsize=6.4, color=RED if v >= 1.0 else GREEN)
    ax_sc.axvline(1.0, color=RED, lw=1.2, zorder=5)
    ax_sc.text(1.04, len(SCENARIOS) - 0.32, "$RQ$ = 1 acceptability threshold",
               fontsize=6.4, color=RED, ha="left", va="center")
    ax_sc.set_xlim(0.0, max(rqs.max() * 1.20, 1.4))
    ax_sc.set_ylim(-0.7, len(SCENARIOS) - 0.15)
    ax_sc.set_yticks(y)
    ax_sc.set_yticklabels(labels, fontsize=6.4)
    ax_sc.set_xlabel("Risk quotient $RQ$ (-)", fontsize=8.0)
    ax_sc.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_sc)
    ax_sc.tick_params(axis="x", labelsize=6.8)
    ax_sc.tick_params(axis="y", length=0)
    ax_sc.spines["left"].set_visible(False)
    ax_sc.set_title("Risk quotient across exposure scenarios", fontsize=8.5,
                    fontweight="bold", loc="left", pad=4)

    # ---- (c) tornado -------------------------------------------------------
    spans = [(n, rq * lo, rq * hi) for n, lo, hi in SENSITIVITY]
    spans.sort(key=lambda s: s[2] - s[1])
    yt = np.arange(len(spans))
    for k, (n, lo, hi) in enumerate(spans):
        ax_tr.barh(k, rq - lo, left=lo, height=0.60, color=BLUE,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax_tr.barh(k, hi - rq, left=rq, height=0.60, color=RED,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax_tr.text(lo - 0.03, k, f"{lo:.2f}", va="center", ha="right",
                   fontsize=6.0, color=BLUE)
        ax_tr.text(hi + 0.03, k, f"{hi:.2f}", va="center", ha="left",
                   fontsize=6.0, color=RED)
    ax_tr.axvline(rq, color=RULE, lw=0.9, zorder=5)
    ax_tr.axvline(1.0, color=RED, lw=1.0, ls=(0, (3, 2)), zorder=5)
    lo_all = min(s[1] for s in spans)
    hi_all = max(s[2] for s in spans)
    ax_tr.set_xlim(lo_all - 0.34, hi_all + 0.34)
    ax_tr.set_ylim(-0.7, len(spans) - 0.15)
    ax_tr.set_yticks(yt)
    ax_tr.set_yticklabels([s[0] for s in spans], fontsize=6.2)
    ax_tr.set_xlabel("Risk quotient $RQ$ (-)", fontsize=8.0)
    ax_tr.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_tr)
    ax_tr.tick_params(axis="x", labelsize=6.8)
    ax_tr.tick_params(axis="y", length=0)
    ax_tr.spines["left"].set_visible(False)
    ax_tr.set_title(f"Sensitivity of $RQ$, baseline {rq:.2f}", fontsize=8.5,
                    fontweight="bold", loc="left", pad=4)

    panel_labels_inches(fig, [ax, ax_sc, ax_tr], dx=-0.44, dy=0.09)

    n_over = int(np.sum(rqs >= 1.0))

    print(f"RQ={rq:.3f}  PEC95={pec95:.2f}  PNEC={pnec:.2f}  "
          f"P(exceed)={p_exceed:.5f}  overlap={ov_area:.4f}")
    savefig(fig, "626_tox_risk_characterization.png")


if __name__ == "__main__":
    main()
