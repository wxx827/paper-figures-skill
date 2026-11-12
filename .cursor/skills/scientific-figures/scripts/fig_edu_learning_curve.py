"""Power-law learning curves for four ability strata in an adaptive tutor.

One simulated cohort feeds every panel and every number.  The main panel is
the claim: mastery grows as a power law of practice and the four ability
strata differ mainly in the exponent, not in the asymptote.  The two smaller
panels carry the supporting evidence - how fast the same material is
forgotten under different practice spacings, and how many trials each stratum
needs to cross the mastery threshold.  Learning rates in the footnote are
re-fitted from the plotted mean curves, so changing the seed changes them.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_palette, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

GROUPS = ["high ability", "upper-mid", "lower-mid", "low ability"]
RATES = [0.75, 0.62, 0.51, 0.42]
STARTS = [0.34, 0.29, 0.24, 0.19]
ASYMPTOTE = 0.965
N_LEARNER = 45
N_TRIAL = 60
THRESHOLD = 0.80

SPACINGS = ["massed (same day)", "spaced 1 day", "spaced 3 days",
            "spaced 7 days"]
TAUS = [3.4, 6.6, 11.5, 18.0]
RETENTION_DAYS = 30.0


def simulate_cohort(seed=594):
    """Per-learner power-law curves; returns trials and one array per group."""
    rng = np.random.default_rng(seed)
    n = np.arange(1, N_TRIAL + 1, dtype=float)
    curves = {}
    for g, r, m0 in zip(GROUPS, RATES, STARTS):
        rr = rng.normal(r, 0.055, N_LEARNER)[:, None]
        mm = rng.normal(m0, 0.032, N_LEARNER)[:, None]
        m = ASYMPTOTE - (ASYMPTOTE - mm) * n[None, :] ** (-rr)
        m = m + rng.normal(0.0, 0.013, m.shape)
        curves[g] = np.clip(m, 0.02, 0.995)
    return n, curves


def fit_rate(n, mean_curve):
    """Recover the power-law exponent from the curve that is actually drawn."""
    gap = np.clip(ASYMPTOTE - mean_curve, 1e-4, None)
    slope, _ = np.polyfit(np.log(n), np.log(gap), 1)
    return -slope


def trials_to_threshold(n, mean_curve):
    """First trial count at which the mean curve reaches the threshold."""
    hit = np.nonzero(mean_curve >= THRESHOLD)[0]
    if hit.size == 0:
        return np.nan
    k = hit[0]
    if k == 0:
        return float(n[0])
    y0, y1 = mean_curve[k - 1], mean_curve[k]
    frac = (THRESHOLD - y0) / (y1 - y0)
    return float(n[k - 1] + frac * (n[k] - n[k - 1]))


def main():
    set_pub_style()
    n, curves = simulate_cohort()

    # Dark blue = strongest stratum: one hue, ordered by lightness.
    ramp = get_palette("cb_sequential_blue", 7)
    group_cols = [ramp[5], ramp[4], ramp[3], ramp[2]]
    green = get_palette("cb_sequential_green", 7)
    space_cols = [green[2], green[3], green[4], green[5]]

    means = {g: curves[g].mean(axis=0) for g in GROUPS}
    sems = {g: curves[g].std(axis=0, ddof=1) / np.sqrt(N_LEARNER)
            for g in GROUPS}
    n_star = {g: trials_to_threshold(n, means[g]) for g in GROUPS}
    r_hat = {g: fit_rate(n, means[g]) for g in GROUPS}

    # A row-spanning subplot inside a gridspec with unequal height_ratios
    # makes tight_layout bail out with a warning and silently do nothing, so
    # this figure places its margins explicitly.
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.26,
                          hspace=0.48)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_forget = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])

    # ---- main: mastery growth ------------------------------------------
    for g, c in zip(GROUPS, group_cols):
        mu, se = means[g], sems[g]
        ax_main.fill_between(n, mu - 1.96 * se, mu + 1.96 * se, color=c,
                             alpha=0.20, linewidth=0.0, zorder=2)
        ax_main.plot(n, mu, color=c, lw=1.7, zorder=3, label=g)

    ax_main.axhline(THRESHOLD, color=GREY, lw=0.9, ls=(0, (4.0, 2.2)),
                    zorder=1)
    ax_main.text(N_TRIAL, THRESHOLD + 0.012,
                 f"mastery threshold $P$ = {THRESHOLD:.2f}", ha="right",
                 va="bottom", fontsize=6.8, color=SUB)

    # Neighbouring markers (6 and 10 trials) sit almost on top of each other,
    # so the call-outs alternate between two baselines.
    for k, (g, c) in enumerate(zip(GROUPS, group_cols)):
        ns = n_star[g]
        if not np.isfinite(ns):
            continue
        y_lab = 0.075 if k % 2 == 0 else 0.135
        ax_main.plot([ns, ns], [y_lab + 0.045, THRESHOLD], color=c, lw=0.8,
                     ls=(0, (2.2, 1.8)), alpha=0.75, zorder=2)
        ax_main.plot([ns], [THRESHOLD], marker="o", ms=4.4, mfc="white",
                     mec=c, mew=1.2, zorder=5)
        ax_main.text(ns, y_lab, f"$n$*={ns:.0f}", ha="center", va="center",
                     fontsize=6.5, color=c)

    ax_main.set_xlim(0.5, N_TRIAL + 0.5)
    ax_main.set_ylim(0.03, 1.02)
    ax_main.set_xlabel("practice attempts $n$ (trials)", fontsize=8.0)
    ax_main.set_ylabel("mastery probability $P$(correct) (-)", fontsize=8.0)
    ax_main.tick_params(labelsize=7.2)
    ax_main.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_main.set_title("Mastery growth follows a power law of practice",
                      fontsize=9.0, fontweight="bold", color=INK)
    leg = ax_main.legend(loc="lower right", frameon=False, fontsize=7.2,
                         handlelength=1.6, labelspacing=0.32,
                         borderaxespad=0.8)
    for t in leg.get_texts():
        t.set_color(INK)
    ax_main.text(0.985, 0.30, r"$M(n)=A-(A-M_0)\,n^{-r}$",
                 transform=ax_main.transAxes, ha="right", va="bottom",
                 fontsize=7.4, color=SUB)

    # ---- aux 1: forgetting under different spacings ---------------------
    t = np.linspace(0.0, RETENTION_DAYS, 240)
    ret30 = [float(np.exp(-RETENTION_DAYS / tau)) for tau in TAUS]
    for lab, tau, c in zip(SPACINGS, TAUS, space_cols):
        ax_forget.plot(t, np.exp(-t / tau), color=c, lw=1.5, label=lab)

    # End-of-window call-outs are pushed apart from the bottom up so the two
    # fastest-decaying curves do not share one label slot.
    y_lab = 0.055
    for v, c in zip(ret30, space_cols):
        y_lab = max(v, y_lab)
        ax_forget.text(RETENTION_DAYS + 0.7, y_lab, f"{v * 100:.0f}%",
                       ha="left", va="center", fontsize=6.3, color=c)
        y_lab += 0.085

    ax_forget.set_xlim(0.0, RETENTION_DAYS * 1.13)
    ax_forget.set_ylim(0.0, 1.03)
    ax_forget.set_xlabel("days since last practice (days)", fontsize=7.4)
    ax_forget.set_ylabel("retention $R$ (-)", fontsize=7.4)
    ax_forget.tick_params(labelsize=6.6)
    ax_forget.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_forget.set_title("Spacing flattens forgetting", fontsize=8.5,
                        fontweight="bold", color=INK)
    ax_forget.legend(loc="upper right", frameon=False, fontsize=6.4,
                     handlelength=1.4, labelspacing=0.28,
                     borderaxespad=0.3)

    # ---- aux 2: trials needed to reach the threshold --------------------
    ypos = np.arange(len(GROUPS))[::-1]
    vals = [n_star[g] for g in GROUPS]
    ax_bar.barh(ypos, vals, height=0.62, color=group_cols,
                edgecolor="white", linewidth=0.6)
    for y, v in zip(ypos, vals):
        ax_bar.text(v + 0.7, y, f"{v:.0f}", ha="left", va="center",
                    fontsize=6.6, color=SUB)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(GROUPS, fontsize=6.6)
    ax_bar.set_xlim(0.0, max(vals) * 1.22)
    ax_bar.set_xlabel("trials to reach $P$ = 0.80 (count)", fontsize=7.4)
    ax_bar.tick_params(axis="x", labelsize=6.6)
    ax_bar.grid(axis="x", lw=0.5, color="#e4e4e4")
    ax_bar.set_title("Cost of mastery by ability stratum", fontsize=8.5,
                     fontweight="bold", color=INK)

    fig.subplots_adjust(left=0.068, right=0.985, bottom=0.170, top=0.900)

    rates_txt = " / ".join(f"{r_hat[g]:.2f}" for g in GROUPS)
    note = (f"Learning rates $r$ refitted from the plotted mean curves = "
            f"{rates_txt} (high to low), shared asymptote $A$ = "
            f"{ASYMPTOTE:.3f}. Bands are mean $\\pm$ 1.96 SE over "
            f"{N_LEARNER} simulated learners per stratum, {N_TRIAL} trials "
            f"each; retention uses $R(t)=e^{{-t/\\tau}}$ with $\\tau$ = "
            + " / ".join(f"{v:.1f}" for v in TAUS) + " days.")

    panel_labels_inches(fig, [ax_main, ax_forget, ax_bar],
                        labels=["(a)", "(b)", "(c)"], dx=-0.44, dy=0.10)
    savefig(fig, "594_edu_learning_curve.png")


if __name__ == "__main__":
    main()
