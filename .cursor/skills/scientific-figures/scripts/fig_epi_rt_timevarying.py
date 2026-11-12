"""Time-varying reproduction number with credible-interval fan.

Upper panel: the Cori posterior for Rt over a 7-day sliding window, drawn as a
median line inside nested 50% and 95% credible bands, with the critical level
Rt = 1 in red and every crossing of it flagged. Lower panel: the daily case
counts the estimate is computed from, on a shared date axis so the reader can
read cause and effect vertically.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#134a76"
BAND50 = "#9cc3dd"
BAND95 = "#cfe1ee"
BAR = "#a8bdd6"
ALERT = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))

START = "2021-09-06"
N_DAYS = 168
TAU = 7                     # estimation window (days)
SI_SHAPE, SI_SCALE = 2.9, 1.9
PRIOR_A, PRIOR_B = 1.0, 5.0

EVENTS = [(34, "Mask mandate"), (66, "Gathering limit"),
          (118, "Reopening phase 2")]


def si_weights(k=20):
    s = np.arange(1, k + 1)
    w = stats.gamma.cdf(s, SI_SHAPE, scale=SI_SCALE) - \
        stats.gamma.cdf(s - 1, SI_SHAPE, scale=SI_SCALE)
    return w / w.sum()


def rt_truth(n):
    t = np.arange(n, dtype=float)
    logistic = lambda t0, k=2.6: 1.0 / (1.0 + np.exp(-(t - t0) / k))
    return (1.62 - 0.52 * logistic(34) - 0.40 * logistic(66)
            + 0.58 * logistic(118))


def simulate(rng, n):
    w = si_weights()
    rt = rt_truth(n)
    inc = np.zeros(n)
    inc[:8] = rng.poisson(30, 8) + 20
    for t in range(8, n):
        lag = np.arange(1, min(len(w), t) + 1)
        inc[t] = rng.poisson(max(rt[t] * float(np.sum(inc[t - lag] * w[lag - 1])),
                                 0.5))
    return inc.astype(float), rt


def cori_posterior(inc, w, tau=TAU):
    """Gamma posterior for Rt on a sliding window (Cori et al. 2013)."""
    n = inc.size
    lam = np.zeros(n)
    for t in range(1, n):
        k = min(len(w), t)
        lam[t] = float(np.sum(inc[t - np.arange(1, k + 1)] * w[:k]))
    shape = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    for t in range(max(tau, 14), n):     # burn-in: the first estimates are wild
        sl = slice(t - tau + 1, t + 1)
        shape[t] = PRIOR_A + inc[sl].sum()
        scale[t] = 1.0 / (1.0 / PRIOR_B + lam[sl].sum())
    q = {}
    for p in (0.025, 0.25, 0.5, 0.75, 0.975):
        q[p] = stats.gamma.ppf(p, shape, scale=scale)
    return q


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(7)
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    inc, _ = simulate(rng, N_DAYS)
    q = cori_posterior(inc, si_weights())
    med = q[0.5]
    ok = np.isfinite(med)

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.2, 4.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 0.55], hspace=0.11,
                         left=0.082, right=0.985, top=0.845, bottom=0.115))

    # ---- (a) Rt posterior --------------------------------------------------
    ax.fill_between(dates, q[0.025], q[0.975], color=BAND95, lw=0, zorder=2,
                    label="95% credible interval")
    ax.fill_between(dates, q[0.25], q[0.75], color=BAND50, lw=0, zorder=3,
                    label="50% credible interval")
    ax.plot(dates, med, color=DEEP, lw=1.7, zorder=5, label="posterior median")
    ax.axhline(1.0, color=ALERT, ls=DASH, lw=1.0, zorder=6)

    top = 2.05
    ax.text(dates[2], 1.03, "critical level $R_t$ = 1", fontsize=6.8,
            color=ALERT, va="bottom", ha="left", zorder=7)

    # every crossing of the critical level, taken from the median itself
    sgn = np.sign(med[ok] - 1.0)
    idx = np.flatnonzero(ok)
    cross = idx[1:][np.diff(sgn) != 0]
    for c in cross:
        ax.plot([dates[c]], [1.0], marker="o", ms=4.2, mfc="white",
                mec=ALERT, mew=1.2, zorder=8)
        going = "below" if med[c] < 1.0 else "above"
        ax.annotate(f"{dates[c]:%d %b}\n{going} 1",
                    xy=(dates[c], 1.0),
                    xytext=(dates[c] + pd.Timedelta(days=3),
                            1.0 + (0.42 if going == "below" else -0.30)),
                    fontsize=6.4, color=ALERT, linespacing=1.25,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=ALERT))

    for day, label in EVENTS:
        for a in (ax, axb):
            a.axvline(dates[day], color=RULE, ls=DASH, lw=0.9, zorder=6)
        ax.text(dates[day] + pd.Timedelta(days=1.5), top * 0.985, label,
                fontsize=6.6, color=RULE, ha="left", va="top", zorder=7)

    ax.set_ylim(0.35, top)
    ax.set_ylabel("Reproduction number $R_t$ (-)")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
              fontsize=7.0, frameon=False, handlelength=1.5,
              columnspacing=1.4, borderaxespad=0.0)
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.075, y=1.10)

    # ---- (b) the cases behind the estimate ---------------------------------
    axb.bar(dates, inc, width=0.92, color=BAR, edgecolor="none", zorder=3)
    ma = pd.Series(inc).rolling(7, center=True, min_periods=4).mean().values
    axb.plot(dates, ma, color=DEEP, lw=1.4, zorder=4)
    axb.set_ylim(0, float(inc.max()) * 1.24)
    axb.set_xlim(dates[0] - pd.Timedelta(days=1),
                 dates[-1] + pd.Timedelta(days=1))
    axb.xaxis.set_major_locator(mdates.MonthLocator())
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axb.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axb.set_ylabel("Cases (per day)")
    axb.set_xlabel("Date of report (2021-2022)")
    axb.tick_params(labelsize=7.2)
    axb.text(dates[3], float(inc.max()) * 1.16,
             f"7-day mean peak {np.nanmax(ma):.0f} cases/day", fontsize=6.6,
             color=MUTED, va="top")
    style_axes(axb)
    add_panel_label(axb, "(b)", x=-0.075, y=1.02)

    savefig(fig, "427_epi_rt_timevarying.png")


if __name__ == "__main__":
    main()
