"""Kaplan-Meier curves by vaccination stratum with a number-at-risk table.

Step estimates for four strata carry Greenwood log-log 95% bands and censoring
ticks; the log-rank test across strata and the hazard ratio for the two extreme
strata are boxed in the upper right. The number-at-risk table shares the time
axis with the curves, so every column sits exactly under its tick.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, sig_stars

STRATA = ["Unvaccinated", "One dose", "Two doses", "Boosted"]
STRAT_C = ["#b23a48", "#c9a227", "#3b5b92", "#2e7d5b"]
N_PER = [430, 305, 520, 470]
MEDIAN = [78.0, 118.0, 205.0, 340.0]          # days to severe outcome
HORIZON = 90.0
TICKS = [0, 15, 30, 45, 60, 75, 90]
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))


def simulate(rng, n, median_days):
    shape = 1.25
    scale = median_days / np.log(2) ** (1 / shape)
    t_event = scale * rng.weibull(shape, n)
    t_cens = rng.uniform(12.0, 1.6 * HORIZON, n)
    time = np.minimum(t_event, np.minimum(t_cens, HORIZON))
    event = (t_event <= np.minimum(t_cens, HORIZON)).astype(int)
    return time, event


def km(time, event):
    """Kaplan-Meier estimate with Greenwood log-log 95% bounds."""
    et = np.unique(time[event == 1])
    n_at = np.array([(time >= t).sum() for t in et], dtype=float)
    d = np.array([((time == t) & (event == 1)).sum() for t in et], dtype=float)
    surv = np.cumprod(1.0 - d / n_at)
    denom = n_at * (n_at - d)
    green = np.cumsum(np.divide(d, denom, out=np.zeros_like(d), where=denom > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(green) / np.abs(np.log(np.maximum(surv, 1e-12)))
        lo = surv ** np.exp(1.96 * se)
        hi = surv ** np.exp(-1.96 * se)
    lo = np.clip(np.nan_to_num(lo, nan=0.0), 0, 1)
    hi = np.clip(np.nan_to_num(hi, nan=1.0), 0, 1)
    return (np.concatenate([[0.0], et, [HORIZON]]),
            np.concatenate([[1.0], surv, surv[-1:]]),
            np.concatenate([[1.0], lo, lo[-1:]]),
            np.concatenate([[1.0], hi, hi[-1:]]))


def logrank(times, events):
    all_t = np.unique(np.concatenate([t[e == 1] for t, e in zip(times, events)]))
    k = len(times)
    obs, exp = np.zeros(k), np.zeros(k)
    for t in all_t:
        n_g = np.array([(tt >= t).sum() for tt in times], dtype=float)
        d_g = np.array([((tt == t) & (ee == 1)).sum()
                        for tt, ee in zip(times, events)], dtype=float)
        n, d = n_g.sum(), d_g.sum()
        if n <= 0:
            continue
        obs += d_g
        exp += d * n_g / n
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-9)))
    p = float(stats.chi2.sf(chi2, k - 1))
    hr = (obs[0] / exp[0]) / (obs[-1] / exp[-1])
    se = np.sqrt(1 / max(exp[0], 1e-9) + 1 / max(exp[-1], 1e-9))
    return p, hr, float(np.exp(np.log(hr) - 1.96 * se)), \
        float(np.exp(np.log(hr) + 1.96 * se)), chi2


def fmt_p(p):
    return "P < 0.0001" if p < 1e-4 else f"P = {p:.4f}".rstrip("0")


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(21)
    times, events = [], []
    for n, m in zip(N_PER, MEDIAN):
        t, e = simulate(rng, n, m)
        times.append(t); events.append(e)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 4.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 0.30], hspace=0.10,
                         left=0.115, right=0.985, top=0.845, bottom=0.115))

    # ---- Kaplan-Meier curves ----------------------------------------------
    for a, (name, col) in enumerate(zip(STRATA, STRAT_C)):
        t, s, lo, hi = km(times[a], events[a])
        ax.fill_between(t, lo, hi, step="post", color=col, alpha=0.13, lw=0,
                        zorder=2)
        ax.step(t, s, where="post", color=col, lw=1.7, zorder=4,
                label=f"{name}  (n = {N_PER[a]})")
        cens = np.sort(times[a][events[a] == 0])
        cens = cens[(cens > 0) & (cens < HORIZON)]
        idx = np.clip(np.searchsorted(t, cens, side="right") - 1, 0, s.size - 1)
        ax.plot(cens, s[idx], ls="", marker="|", ms=4.0, mew=0.8, color=col,
                alpha=0.8, zorder=5)
        ax.text(HORIZON + 0.8, s[-1], f"{100 * s[-1]:.0f}%", color=col,
                fontsize=6.8, va="center", ha="left", clip_on=False)

    # median survival, taken from the step curve of the worst stratum
    t0, s0, _, _ = km(times[0], events[0])
    below = np.flatnonzero(s0 <= 0.5)
    t_med = float(t0[below[0]]) if below.size else np.nan
    if np.isfinite(t_med):
        ax.plot([0, t_med], [0.5, 0.5], color=MUTED, ls=DASH, lw=0.8, zorder=1)
        ax.plot([t_med, t_med], [0.20, 0.5], color=MUTED, ls=DASH, lw=0.8,
                zorder=1)
        ax.text(t_med - 2.0, 0.46,
                f"median {t_med:.0f} d\n({STRATA[0].lower()})",
                fontsize=6.4, color=MUTED, ha="right", va="top",
                linespacing=1.25)

    p, hr, hr_lo, hr_hi, chi2 = logrank(times, events)
    ax.text(0.975, 0.035,
            f"Log-rank $\\chi^2$ = {chi2:.1f} (3 df), {fmt_p(p)} "
            f"{sig_stars(p)}\n"
            f"HR unvaccinated vs boosted = {hr:.2f} "
            f"[{hr_lo:.2f}-{hr_hi:.2f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            linespacing=1.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                      lw=0.6))

    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, HORIZON)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.2f}" for v in np.arange(0.0, 1.01, 0.25)])
    ax.set_ylabel("Probability free of severe outcome")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.legend(loc="lower left", fontsize=7.0, frameon=False, handlelength=1.6,
              labelspacing=0.32, borderaxespad=0.4,
              bbox_to_anchor=(0.0, 0.02))
    style_axes(ax)

    # ---- number at risk ----------------------------------------------------
    axr.set_ylim(-0.6, len(STRATA) - 0.4)
    axr.invert_yaxis()
    axr.set_yticks(range(len(STRATA)))
    axr.set_yticklabels(STRATA, fontsize=7.0)
    for lab, c in zip(axr.get_yticklabels(), STRAT_C):
        lab.set_color(c)
    axr.set_xticks(TICKS)
    axr.tick_params(axis="both", length=0, labelsize=7.2)
    axr.tick_params(axis="y", pad=6)
    for s in axr.spines.values():
        s.set_visible(False)
    axr.set_xlabel("Time since positive test (days)")
    axr.set_title("Number at risk", fontsize=7.5, fontweight="bold",
                  loc="left", pad=3, color="#444444")
    for a in range(len(STRATA)):
        for ti, tk in enumerate(TICKS):
            ha = "left" if ti == 0 else "right" if ti == len(TICKS) - 1 \
                else "center"
            axr.text(tk, a, str(int((times[a] >= tk).sum())), ha=ha,
                     va="center", fontsize=6.8, color=INK)

    savefig(fig, "430_epi_survival_strata.png")


if __name__ == "__main__":
    main()
