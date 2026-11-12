"""Policy persistency: how long a book stays on the books, and what moves it.

Main panel: Kaplan-Meier persistency curves by distribution channel, with
Greenwood log-log confidence bands, censoring ticks for policies still in
force at the observation date, the median duration of each channel and the
first-year lapse rate that underwriters actually get measured on.  A three
group log-rank test sits in the corner.  The two narrow right-hand panels
carry the shape and the drivers: the discrete hazard by policy year, which is
high in year one, settles, and spikes again as the premium-paying term ends,
and a Cox proportional-hazards forest of the lapse covariates.

The estimator, the Greenwood variance, the multi-group log-rank statistic and
the Cox partial-likelihood fit are all implemented on numpy here, so every
printed number comes from the plotted sample.  Values are illustrative.
"""
import textwrap

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from pubstyle import set_pub_style, savefig, style_axes, sig_stars
from figkit import axes_inches, panel_labels_inches

AGENCY = "#3b5b92"
BANCA = "#c9a227"
DIRECT = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.74, 5.24, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

HORIZON = 15.0
# annual lapse hazard by policy year; year 10 is the end of the premium term
BASE_HAZARD = np.array([0.210, 0.115, 0.088, 0.075, 0.068, 0.065, 0.062,
                        0.060, 0.068, 0.142, 0.081, 0.054, 0.047, 0.045,
                        0.041])
CHANNELS = [("Agency", 980, 0.00, AGENCY),
            ("Bancassurance", 760, 0.42, BANCA),
            ("Direct", 640, 0.20, DIRECT)]
# log hazard ratios used to generate the book; the Cox fit has to find them
BETA_TRUE = dict(issue_age=-0.22, premium=-0.15, monthly=0.35, rider=-0.18)
COVARIATES = ["Bancassurance vs agency", "Direct vs agency",
              "Issue age (+1 SD)", "Premium size (+1 SD)",
              "Monthly premium mode", "Rider attached"]


def gen_book(seed=887):
    """Policy durations, censoring flags, group index and the design matrix."""
    rng = np.random.default_rng(seed)
    t_all, e_all, g_all, x_all = [], [], [], []
    for gi, (_, n, ch_beta, _) in enumerate(CHANNELS):
        age = rng.normal(0.0, 1.0, n)
        prem = rng.normal(0.0, 1.0, n)
        monthly = (rng.random(n) < 0.55).astype(float)
        rider = (rng.random(n) < 0.40).astype(float)
        lp = (ch_beta + BETA_TRUE["issue_age"] * age
              + BETA_TRUE["premium"] * prem
              + BETA_TRUE["monthly"] * monthly
              + BETA_TRUE["rider"] * rider)
        mult = np.exp(lp)
        lapse = np.full(n, np.inf)
        alive = np.ones(n, dtype=bool)
        for j, h in enumerate(BASE_HAZARD):
            p = 1.0 - np.exp(-h * mult)
            hit = alive & (rng.random(n) < p)
            lapse[hit] = j + rng.random(int(hit.sum()))
            alive &= ~hit
        cens = 0.6 + (HORIZON - 0.6) * rng.random(n)
        t = np.minimum(lapse, cens)
        e = (lapse <= cens).astype(int)
        dummies = np.zeros((n, 2))
        if gi >= 1:
            dummies[:, gi - 1] = 1.0
        t_all.append(t)
        e_all.append(e)
        g_all.append(np.full(n, gi))
        x_all.append(np.column_stack([dummies, age, prem, monthly, rider]))
    return (np.concatenate(t_all), np.concatenate(e_all),
            np.concatenate(g_all), np.vstack(x_all))


def kaplan_meier(t, e):
    """Step curve with Greenwood log-log confidence limits."""
    times = np.unique(t[e == 1])
    s, lo, hi = [1.0], [1.0], [1.0]
    surv, acc = 1.0, 0.0
    for u in times:
        n_risk = float(np.sum(t >= u))
        d = float(np.sum((t == u) & (e == 1)))
        surv *= 1.0 - d / n_risk
        if n_risk > d:
            acc += d / (n_risk * (n_risk - d))
        if surv <= 0.0:
            s.append(0.0)
            lo.append(0.0)
            hi.append(0.0)
            continue
        se = np.sqrt(acc) / abs(np.log(surv))
        s.append(surv)
        lo.append(surv ** np.exp(1.96 * se))
        hi.append(surv ** np.exp(-1.96 * se))
    return (np.concatenate([[0.0], times]), np.array(s), np.array(lo),
            np.array(hi))


def step_at(x, y, q):
    """Value of a right-continuous step function at q."""
    k = int(np.searchsorted(x, q, side="right") - 1)
    return float(y[max(k, 0)])


def median_duration(x, y):
    below = np.where(y <= 0.5)[0]
    return float(x[below[0]]) if below.size else np.nan


def logrank(t, e, g, n_groups=3):
    """Multi-group log-rank chi-square and its p-value."""
    times = np.unique(t[e == 1])
    u = np.zeros(n_groups)
    v = np.zeros((n_groups, n_groups))
    for tt in times:
        at_risk = np.array([np.sum((t >= tt) & (g == k))
                            for k in range(n_groups)], dtype=float)
        d_g = np.array([np.sum((t == tt) & (e == 1) & (g == k))
                        for k in range(n_groups)], dtype=float)
        n_t, d_t = at_risk.sum(), d_g.sum()
        if n_t <= 1:
            continue
        u += d_g - at_risk * d_t / n_t
        f = d_t * (n_t - d_t) / (n_t - 1.0) / n_t
        v += f * (np.diag(at_risk) - np.outer(at_risk, at_risk) / n_t)
    idx = np.arange(n_groups - 1)
    chi2 = float(u[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], u[idx]))
    return chi2, float(stats.chi2.sf(chi2, n_groups - 1))


def cox_ph(x, t, e, iters=30):
    """Newton-Raphson partial-likelihood fit with Breslow ties."""
    order = np.argsort(-t)
    xo, to, eo = x[order], t[order], e[order]
    n, p = xo.shape
    # last index of each tied-time block in the descending ordering
    last = np.empty(n, dtype=int)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and to[j + 1] == to[k]:
            j += 1
        last[k:j + 1] = j
        k = j + 1
    beta = np.zeros(p)
    for _ in range(iters):
        w = np.exp(xo @ beta)
        s0 = np.cumsum(w)
        s1 = np.cumsum(w[:, None] * xo, axis=0)
        s2 = np.cumsum(w[:, None, None] * xo[:, :, None] * xo[:, None, :],
                       axis=0)
        ev = eo == 1
        a0 = s0[last[ev]]
        a1 = s1[last[ev]] / a0[:, None]
        a2 = s2[last[ev]] / a0[:, None, None]
        grad = np.sum(xo[ev] - a1, axis=0)
        hess = -np.sum(a2 - a1[:, :, None] * a1[:, None, :], axis=0)
        step = np.linalg.solve(-hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    cov = np.linalg.inv(-hess)
    return beta, np.sqrt(np.diag(cov))


def yearly_hazard(t, e, n_years=int(HORIZON)):
    """Discrete lapse hazard per policy year: lapses / policies in force."""
    h = np.full(n_years, np.nan)
    for j in range(n_years):
        at_risk = np.sum(t >= j)
        if at_risk >= 25:
            h[j] = np.sum((t >= j) & (t < j + 1) & (e == 1)) / at_risk
    return h


def main():
    set_pub_style(base_font=8)
    t, e, g, x = gen_book()
    chi2, p_val = logrank(t, e, g)
    beta, se = cox_ph(x, t, e)
    hr, hr_lo, hr_hi = (np.exp(beta), np.exp(beta - 1.96 * se),
                        np.exp(beta + 1.96 * se))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_h = axes_inches(fig, AUX_T_RECT)
    ax_f = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_h, ax_f):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: Kaplan-Meier persistency ------------------------------------
    summary = []
    for gi, (name, n, _, colour) in enumerate(CHANNELS):
        m = g == gi
        xs, s, lo, hi = kaplan_meier(t[m], e[m])
        xs = np.append(xs, HORIZON)
        s = np.append(s, s[-1])
        lo = np.append(lo, lo[-1])
        hi = np.append(hi, hi[-1])
        ax.fill_between(xs, 100 * lo, 100 * hi, step="post", color=colour,
                        alpha=0.13, lw=0.0, zorder=2)
        ax.step(xs, 100 * s, where="post", color=colour, lw=1.8, zorder=5)
        cens = np.sort(t[m & (e == 0)])[::7]
        ax.plot(cens, [100 * step_at(xs, s, c) for c in cens], ls="none",
                marker="|", ms=4.0, mew=0.8, color=colour, alpha=0.85,
                zorder=6)
        med = median_duration(xs, s)
        first = 100.0 * (1.0 - step_at(xs, s, 1.0))
        summary.append((name, n, med, first, colour))
        if np.isfinite(med):
            ax.plot([med], [50.0], marker="o", ms=4.2, mfc="white",
                    mec=colour, mew=1.3, zorder=8)

    ax.axhline(50.0, color=FAINT, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax.axvline(10.0, color=FAINT, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 104)
    ax.set_xticks(np.arange(0, HORIZON + 1, 1))
    ax.set_yticks([0, 20, 40, 50, 60, 80, 100])
    ax.set_xlabel("policy duration (years since issue)", fontsize=8.5)
    ax.set_ylabel("policies still in force (%)", fontsize=8.5)
    ax.set_title("Persistency by distribution channel", fontsize=9.0,
                 fontweight="bold", pad=12.0)

    rows = [f"{'channel':<15}{'n':>6}{'median':>9}{'yr-1 lapse':>12}"]
    for name, n, med, first, _ in summary:
        med_s = f"{med:.1f} yr" if np.isfinite(med) else "n.r."
        rows.append(f"{name:<15}{n:>6,}{med_s:>9}{first:>11.1f}%")
    ax.text(0.985, 0.985, "\n".join(rows), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=SUB,
            family="DejaVu Sans Mono", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))
    ax.text(0.985, 0.615,
            f"log-rank $\\chi^2$ = {chi2:.1f} on 2 df, "
            f"P = {p_val:.1e} {sig_stars(p_val)}\n"
            f"bands are Greenwood log-log 95% limits; ticks mark policies\n"
            f"still in force at the observation date; the vertical rule is\n"
            f"the end of the premium-paying term",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=SUB, linespacing=1.5)
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=1.8,
                                  label=f"{nm} (n = {n:,})")
                       for nm, n, _, _, c in summary],
              loc="lower left", frameon=False, fontsize=7.2,
              handlelength=2.0, labelspacing=0.42,
              bbox_to_anchor=(0.005, 0.010))

    # ---- aux (b): hazard by policy year ------------------------------------
    years = np.arange(1, int(HORIZON) + 1)
    haz = {}
    for gi, (name, _, _, colour) in enumerate(CHANNELS):
        m = g == gi
        haz[name] = 100.0 * yearly_hazard(t[m], e[m])
        ax_h.plot(years, haz[name], color=colour, lw=1.5, marker="o", ms=2.6,
                  mfc="white", mec=colour, mew=0.9, zorder=5)
    pooled = 100.0 * yearly_hazard(t, e)
    spike = int(years[int(np.nanargmax(pooled[6:12])) + 6])
    ax_h.axvline(10, color=FAINT, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax_h.annotate(f"year-{spike} spike\n{pooled[spike - 1]:.1f}% lapsing",
                  xy=(spike, pooled[spike - 1]), xytext=(spike - 3.4,
                                                         0.80 * np.nanmax(
                                                             pooled) + 6.0),
                  fontsize=6.4, color=ACCENT, linespacing=1.45,
                  ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ACCENT,
                                  shrinkA=2, shrinkB=2))
    ax_h.set_xlim(0.4, HORIZON + 0.6)
    ax_h.set_ylim(0, np.nanmax(pooled) * 1.55)
    ax_h.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax_h.tick_params(labelsize=6.8)
    ax_h.set_xlabel("policy year", fontsize=7.8, labelpad=1.5)
    ax_h.set_ylabel("lapse hazard (% of in force)", fontsize=7.8)
    ax_h.set_title("Year one dominates, the term end echoes", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): Cox forest -----------------------------------------------
    rowsy = np.arange(len(COVARIATES))
    for k in rowsy:
        colour = ACCENT if hr_lo[k] > 1.0 else (AGENCY if hr_hi[k] < 1.0
                                                else FAINT)
        ax_f.plot([hr_lo[k], hr_hi[k]], [k, k], color=colour, lw=1.2,
                  zorder=4, solid_capstyle="butt")
        ax_f.plot([hr[k]], [k], marker="s", ms=4.0, color=colour, zorder=5)
        ax_f.text(2.05, k, f"{hr[k]:.2f}", ha="right", va="center",
                  fontsize=6.2, color=SUB)
    ax_f.axvline(1.0, color=INK, lw=0.9, zorder=3)
    ax_f.set_yticks(rowsy)
    ax_f.set_yticklabels(COVARIATES, fontsize=6.4)
    ax_f.set_ylim(len(COVARIATES) + 0.05, -0.7)
    ax_f.set_xscale("log")
    ax_f.set_xlim(0.62, 2.08)
    ax_f.xaxis.set_major_locator(FixedLocator([0.7, 0.85, 1.0, 1.2, 1.5,
                                               1.8]))
    ax_f.xaxis.set_minor_locator(FixedLocator([]))
    ax_f.xaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax_f.tick_params(axis="y", length=0)
    ax_f.tick_params(labelsize=6.8)
    ax_f.set_xlabel("lapse hazard ratio (log scale, x)", fontsize=7.8,
                    labelpad=1.5)
    ax_f.set_title("Cox model: what makes a policy lapse", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_f.text(0.985, 0.03,
              f"bars are 95% Wald limits, n = {t.size:,} policies, "
              f"{int(e.sum()):,} lapses",
              transform=ax_f.transAxes, ha="right", va="bottom",
              fontsize=6.2, color=MUTED)

    panel_labels_inches(fig, [ax, ax_h, ax_f], dx=-0.60, dy=0.10)
    savefig(fig, "887_act_lapse_survival.png")


if __name__ == "__main__":
    main()
