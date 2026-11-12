"""Figure 769 - Event-study coefficients around a single treatment date.

Main panel: the dynamic effect at each relative period, normalised so that the
period before the event is exactly zero and drawn as a hollow marker labelled
as the base.  Pre-event coefficients wobble around the zero rule because the
group-level shocks are real AR(1) noise, not a straight line; post-event
coefficients climb and level off.

Auxiliary (both at 58 % of the main panel width): the null distribution of the
pre-period joint Wald statistic obtained by permuting the treatment label 600
times, with the analytic chi-square reference and the observed statistic on
top; and the average post-event effect as the estimation window widens, for
the full donor set and for a 1:1 matched control set.

Every coefficient, standard error, Wald statistic and p value comes from the
same simulated unit-level panel: the covariance of the coefficient vector is
computed from the unit-level differences, which is what makes the joint test
on six correlated pre-period coefficients legitimate.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches, shared_limits

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

EST = "#b23a48"          # the coefficient path: the single emphasis
PRE = "#3b5b92"          # pre-event coefficients, the falsification half
NULL = "#8a8a8a"
MATCH = "#3b5b92"

K = np.arange(-6, 9)     # relative event time
BASE_K = -1              # normalised period
N_T, N_C = 70, 112
PEAK = 0.082             # saturated effect in log units
N_PERM = 600


def effect_path(k):
    """Zero before the event, exponential approach to PEAK afterwards."""
    return np.where(k < 0, 0.0, PEAK * (1.0 - np.exp(-(k + 1.0) / 2.1)))


def ar1(rng, n, phi=0.5, sd=0.0042):
    e = rng.normal(0.0, sd, n)
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(1.0 - phi ** 2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out - out.mean()


def make_panel(rng):
    period = 0.014 * np.arange(K.size) + ar1(rng, K.size, sd=0.012)
    a_t = rng.normal(3.10, 0.19, N_T)
    a_c = rng.normal(3.04, 0.21, N_C)
    y_t = (a_t[:, None] + period[None, :] + ar1(rng, K.size)[None, :]
           + effect_path(K)[None, :] + rng.normal(0, 0.034, (N_T, K.size)))
    y_c = (a_c[:, None] + period[None, :] + ar1(rng, K.size)[None, :]
           + rng.normal(0, 0.036, (N_C, K.size)))
    return y_t, y_c


def event_study(y_t, y_c, base_col):
    """Coefficients, their full covariance, and the surviving column index.

    Differencing every unit against its own base period is what turns a panel
    regression into two sample means, so the covariance of the coefficient
    vector is just the sum of the two group covariances - and that matrix is
    what the joint pre-trend test needs.  The base column is dropped because
    it is identically zero and would make the matrix singular.
    """
    d_t = y_t - y_t[:, [base_col]]
    d_c = y_c - y_c[:, [base_col]]
    keep = np.array([j for j in range(y_t.shape[1]) if j != base_col])
    b = d_t[:, keep].mean(0) - d_c[:, keep].mean(0)
    v = (np.cov(d_t[:, keep], rowvar=False) / y_t.shape[0]
         + np.cov(d_c[:, keep], rowvar=False) / y_c.shape[0])
    return b, v, keep


def wald(b, v, idx):
    sub_b, sub_v = b[idx], v[np.ix_(idx, idx)]
    w = float(sub_b @ np.linalg.solve(sub_v, sub_b))
    return w, len(idx), float(stats.chi2.sf(w, len(idx)))


def window_att(b, v, keep, n_post):
    """Average post-event effect over the first ``n_post`` periods."""
    cols = [i for i, j in enumerate(keep) if 0 <= K[j] < n_post]
    a = np.zeros(b.size)
    a[cols] = 1.0 / len(cols)
    return float(a @ b), float(np.sqrt(a @ v @ a))


def match_controls(y_t, y_c, base_col, rng):
    """Greedy 1:1 nearest-neighbour match on the pre-event mean level."""
    mu_t = y_t[:, :base_col + 1].mean(1)
    mu_c = y_c[:, :base_col + 1].mean(1)
    free = np.ones(mu_c.size, dtype=bool)
    picked = []
    for m in mu_t[rng.permutation(mu_t.size)]:
        d = np.where(free, np.abs(mu_c - m), np.inf)
        j = int(np.argmin(d))
        picked.append(j)
        free[j] = False
    return y_c[np.array(picked)]


# ---------------------------------------------------------------------------
def coef_panel(ax, b, se, keep, w_stat, w_dof, w_p, att, att_se):
    ax.axhspan(0, 0, color="none")
    ax.axvspan(-0.5, K[-1] + 0.6, facecolor=CARD, edgecolor="none", zorder=0)
    ax.axhline(0.0, color=BASE, lw=0.8, ls=(0, (3.2, 2.2)), zorder=1)
    ax.axvline(-0.5, color="#3a3a3a", lw=1.0, ls=(0, (3.4, 2.2)), zorder=2)

    kk = K[keep]
    pre_m, post_m = kk < 0, kk >= 0
    for mask, col, lab in ((pre_m, PRE, "pre-event (falsification)"),
                           (post_m, EST, "post-event (effect)")):
        ax.errorbar(kk[mask], 100 * b[mask], yerr=100 * 1.96 * se[mask],
                    fmt="o", ms=4.4, color=col, mfc=col, mec=col, mew=0.9,
                    ecolor=col, elinewidth=1.1, capsize=2.4, capthick=0.9,
                    zorder=4, label=lab)
    ax.plot(kk[post_m], 100 * b[post_m], lw=1.0, color=lighten(EST, 0.35),
            zorder=3)

    ax.plot([BASE_K], [0.0], marker="o", ms=5.2, mfc="white", mec=INK,
            mew=1.2, zorder=5)
    ax.annotate("base period\n(normalised to 0)", xy=(BASE_K, 0.0),
                xytext=(BASE_K - 0.45, 3.4), ha="right", va="bottom",
                fontsize=6.6, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.7,
                                shrinkA=1, shrinkB=3))

    ax.text(-0.9, 0.965, "pre-event", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=6.6, color=MUTED)
    ax.text(-0.1, 0.965, "post-event", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=6.6, color=SUB)

    ax.set_xlim(K[0] - 0.6, K[-1] + 0.6)
    ax.set_xticks(K[::2])
    ax.set_xlabel("periods relative to the event (quarters)", fontsize=8.0)
    ax.set_ylabel("effect on log revenue (log points)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Flat before the event, then a levelling-off climb",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False, handlelength=1.4,
              borderaxespad=0.35, labelspacing=0.3, numpoints=1)

    ax.text(0.975, 0.055,
            f"joint pre-trend test: chi2({w_dof}) = {w_stat:.2f}, "
            f"p = {w_p:.2f}\naverage post-event effect "
            f"{100 * att:+.1f} +/- {100 * 1.96 * att_se:.1f} log pt "
            f"(95 % CI), n = {N_T} + {N_C}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=SUB, linespacing=1.45)


def null_panel(ax, draws, w_stat, dof, w_p):
    crit = float(stats.chi2.ppf(0.95, dof))
    hi = max(draws.max(), w_stat, crit) * 1.06
    bins = np.linspace(0.0, hi, 34)
    ax.hist(draws, bins=bins, density=True, facecolor=lighten(NULL, 0.55),
            edgecolor="white", linewidth=0.4, zorder=2,
            label="permuted labels")
    xs = np.linspace(1e-3, hi, 300)
    ax.plot(xs, stats.chi2.pdf(xs, dof), lw=1.3, color=INK, zorder=4,
            label=f"chi2({dof}) reference")
    ax.axvspan(crit, hi, facecolor=lighten(EST, 0.86), edgecolor="none",
               zorder=1)
    ax.axvline(crit, color=BASE, lw=0.8, ls=(0, (2.6, 2.0)), zorder=3)
    ax.axvline(w_stat, color=EST, lw=1.6, zorder=5)

    ax.annotate(f"observed\n{w_stat:.2f}", xy=(w_stat, 0.0),
                xytext=(w_stat, ax.get_ylim()[1] * 0.86), ha="center",
                va="top", fontsize=6.4, color=EST, linespacing=1.35)
    ax.text(crit, ax.get_ylim()[1] * 0.42, "5 % critical ", ha="center",
            va="center", rotation=90, fontsize=6.2, color=MUTED)

    frac = float((draws >= w_stat).mean())
    ax.set_xlim(0.0, hi)
    ax.set_xlabel("pre-period joint Wald statistic (-)", fontsize=7.4)
    ax.set_ylabel("density (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title(f"Permutation null, {N_PERM} draws: p = {frac:.2f}",
                 fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False,
              handlelength=1.3, borderaxespad=0.3, labelspacing=0.28)


def robust_panel(ax, widths, full, matched):
    for (est, se), col, mk, lab, off in ((full, EST, "o", "all controls", -0.10),
                                         (matched, MATCH, "s",
                                          "1:1 matched controls", 0.10)):
        ax.errorbar(np.asarray(widths) + off, 100 * est, yerr=100 * 1.96 * se,
                    fmt=mk, ms=3.8, color=col, mfc=col, mec=col, mew=0.8,
                    ecolor=col, elinewidth=1.0, capsize=2.0, capthick=0.8,
                    zorder=4, label=lab)
    ax.axhline(0.0, color=BASE, lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)

    lo, hi = shared_limits([100 * (full[0] - 2.1 * full[1]),
                            100 * (full[0] + 2.1 * full[1]),
                            100 * (matched[0] - 2.1 * matched[1]),
                            100 * (matched[0] + 2.1 * matched[1]), [0.0]],
                           pad=0.10)
    ax.set_ylim(lo, hi)
    ax.set_xticks(widths)
    ax.set_xlabel("post-event periods averaged (quarters)", fontsize=7.4)
    ax.set_ylabel("average effect (log pt)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Robust to window length and donor set", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.4, frameon=False,
              handlelength=1.3, borderaxespad=0.3, labelspacing=0.28,
              numpoints=1)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(769)
    y_t, y_c = make_panel(rng)
    base_col = int(np.where(K == BASE_K)[0][0])

    b, v, keep = event_study(y_t, y_c, base_col)
    se = np.sqrt(np.diag(v))
    pre_idx = [i for i, j in enumerate(keep) if K[j] < 0]
    w_stat, w_dof, w_p = wald(b, v, pre_idx)
    att, att_se = window_att(b, v, keep, K[-1] + 1)

    # Permutation null for the joint pre-trend test: reshuffling the label
    # cannot disturb the pre-event periods, so this is a valid null.
    pool = np.vstack([y_t, y_c])
    draws = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = rng.permutation(pool.shape[0])
        bb, vv, kk = event_study(pool[idx[:N_T]], pool[idx[N_T:]], base_col)
        pi = [j for j, c in enumerate(kk) if K[c] < 0]
        draws[i] = wald(bb, vv, pi)[0]

    y_m = match_controls(y_t, y_c, base_col, rng)
    b_m, v_m, keep_m = event_study(y_t, y_m, base_col)

    widths = list(range(1, K[-1] + 2))
    full = tuple(np.array(z) for z in
                 zip(*[window_att(b, v, keep, w) for w in widths]))
    matched = tuple(np.array(z) for z in
                    zip(*[window_att(b_m, v_m, keep_m, w) for w in widths]))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.82, 5.24, 4.52))
    ax_null = axes_inches(fig, (7.05, 3.48, 3.05, 1.86))
    ax_rob = axes_inches(fig, (7.05, 0.82, 3.05, 1.86))

    coef_panel(ax_main, b, se, keep, w_stat, w_dof, w_p, att, att_se)
    null_panel(ax_null, draws, w_stat, w_dof, w_p)
    robust_panel(ax_rob, widths, full, matched)


    for ax, lab in ((ax_main, "(a)"), (ax_null, "(b)"), (ax_rob, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "769_ci_event_study.png")


if __name__ == "__main__":
    main()
