"""Value-at-Risk backtest over two years of synthetic daily P&L.

(a) daily returns against the time-varying 99% and 95% VaR thresholds, with
    every 99% breach flagged in house red; (b) the cumulative exceedance count
    against its expected path and a binomial acceptance band, sharing the x
    axis with (a); (c) the unconditional return distribution with the
    unconditional VaR and expected-shortfall levels; (d) observed versus
    expected exceedance counts with the Kupiec unconditional-coverage test.

Every number in the annotations is computed from the plotted series, so a new
seed moves the text with the data.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pubstyle import set_pub_style, savefig, style_axes, sig_stars
from figkit import panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

N_DAYS = 504          # two trading years
P99, P95 = 0.01, 0.05


def simulate_returns(rng, n=N_DAYS):
    """EWMA-volatility returns with Student-t shocks and one stress cluster."""
    nu = 4.5
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2.0))
    sigma = np.empty(n)
    sigma[0] = 0.0095
    lam = 0.94
    ret = np.empty(n)
    shock = np.zeros(n)
    shock[250:290] = np.linspace(0.0, 1.0, 40) ** 0.6 * 0.011
    shock[290:340] = np.linspace(1.0, 0.0, 50) ** 0.9 * 0.011
    for i in range(n):
        s = sigma[i] + shock[i]
        ret[i] = s * z[i] - 0.00015
        if i + 1 < n:
            sigma[i + 1] = np.sqrt(lam * sigma[i] ** 2 +
                                   (1.0 - lam) * ret[i] ** 2)
    return ret * 100.0, (sigma + shock) * 100.0


def kupiec_p(n_exc, n_obs, p):
    """Likelihood-ratio test of unconditional coverage -> p value."""
    if n_exc == 0:
        lr = -2.0 * n_obs * np.log(1.0 - p)
    else:
        pi = n_exc / n_obs
        lr = (-2.0 * ((n_obs - n_exc) * np.log(1.0 - p) + n_exc * np.log(p))
              + 2.0 * ((n_obs - n_exc) * np.log(1.0 - pi)
                       + n_exc * np.log(pi)))
    return float(stats.chi2.sf(lr, 1))


def panel_returns(ax, day, ret, var99, var95, exc99):
    ax.axhline(0.0, color="#b0b0b0", lw=0.6, zorder=1)
    ax.plot(day, ret, color=BLUE, lw=0.55, alpha=0.9, zorder=3,
            label="daily return")
    ax.plot(day, -var95, color=GOLD, lw=1.0, ls=(0, (4, 2)), zorder=4,
            label=r"$VaR_{95\%}$ (EWMA)")
    ax.plot(day, -var99, color=RED, lw=1.15, zorder=5,
            label=r"$VaR_{99\%}$ (EWMA)")
    ax.plot(day[exc99], ret[exc99], ls="none", marker="o", ms=3.6,
            mfc="none", mec=RED, mew=0.9, zorder=6,
            label=r"$VaR_{99\%}$ breach")
    ax.set_ylabel("daily return (%)")
    ax.set_xlim(day[0] - 4, day[-1] + 4)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=4, fontsize=7.0, handlelength=1.6,
              columnspacing=1.3, borderaxespad=0.25)
    lo, hi = ret.min(), ret.max()
    ax.set_ylim(lo - 0.32 * (hi - lo), hi + 0.10 * (hi - lo))
    style_axes(ax)


def panel_cumulative(ax, day, exc99, n_obs):
    cum = np.cumsum(exc99).astype(float)
    expected = P99 * (day + 1)
    lo = stats.binom.ppf(0.025, day + 1, P99)
    hi = stats.binom.ppf(0.975, day + 1, P99)
    ax.fill_between(day, lo, hi, color=BLUE, alpha=0.14, lw=0,
                    label="95% binomial band")
    ax.plot(day, expected, color=SUB, lw=0.9, ls=(0, (3, 2)),
            label="expected (1% of days)")
    ax.plot(day, cum, color=RED, lw=1.3, label="observed breaches")
    ax.set_ylabel("cumulative\nbreaches (count)", linespacing=1.3)
    ax.set_xlabel("trading day (days since backtest start)")
    ax.set_xlim(day[0] - 4, day[-1] + 4)
    ax.set_ylim(0, max(cum.max(), hi.max()) * 1.35)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6,
              borderaxespad=0.25)
    ax.tick_params(labelsize=7.0)
    style_axes(ax)


def panel_hist(ax, ret):
    q99 = float(np.quantile(ret, P99))
    q95 = float(np.quantile(ret, P95))
    es99 = float(ret[ret <= q99].mean())
    ax.hist(ret, bins=46, orientation="horizontal", color=BLUE, alpha=0.55,
            edgecolor="white", linewidth=0.35)
    xmax = ax.get_xlim()[1]
    for y, col, va, lab in (
            (q95, GOLD, "bottom", r"$VaR_{95\%}$" + " %.2f%%" % abs(q95)),
            (q99, RED, "bottom", r"$VaR_{99\%}$" + " %.2f%%" % abs(q99)),
            (es99, "#8c2f2f", "top",
             r"$ES_{99\%}$" + " %.2f%%" % abs(es99))):
        ax.axhline(y, color=col, lw=1.0,
                   ls="-" if col != GOLD else (0, (4, 2)))
        ax.text(xmax * 0.96, y + (0.09 if va == "bottom" else -0.09), lab,
                ha="right", va=va, fontsize=6.6, color=col)
    ax.text(xmax * 0.96, ret.max() * 0.92,
            "unconditional\ndistribution, n=%d days" % ret.size,
            ha="right", va="top", fontsize=6.6, color=MUTED, linespacing=1.3)
    ax.set_xlabel("frequency (days)")
    ax.set_ylabel("daily return (%)")
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="x", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    style_axes(ax)


def panel_counts(ax, n99, n95, n_obs):
    labels = [r"$VaR_{99\%}$", r"$VaR_{95\%}$"]
    obs = np.array([n99, n95], dtype=float)
    exp = np.array([P99, P95]) * n_obs
    p99 = kupiec_p(n99, n_obs, P99)
    p95 = kupiec_p(n95, n_obs, P95)
    x = np.arange(2)
    ax.bar(x - 0.19, exp, width=0.36, color="#c9cfdb", edgecolor=SUB,
           linewidth=0.5, label="expected")
    ax.bar(x + 0.19, obs, width=0.36, color=RED, alpha=0.85, linewidth=0,
           label="observed")
    for xi, (o, e, pv) in enumerate(zip(obs, exp, (p99, p95))):
        ax.text(xi + 0.19, o + 0.9, "%d" % o, ha="center", va="bottom",
                fontsize=6.6, color=INK)
        ax.text(xi - 0.19, e + 0.9, "%.1f" % e, ha="center", va="bottom",
                fontsize=6.6, color=SUB)
        ax.text(xi, max(o, e) + 4.2,
                "Kupiec $p$=%.3f %s" % (pv, sig_stars(pv)),
                ha="center", va="bottom", fontsize=6.6,
                color=GREEN if pv >= 0.05 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel("exceedances (count)")
    ax.set_ylim(0, max(obs.max(), exp.max()) * 1.55)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.3,
              borderaxespad=0.25)
    ax.tick_params(labelsize=7.0)
    style_axes(ax)
    return p99, p95


def main():
    set_pub_style()
    rng = np.random.default_rng(570)
    ret, sigma = simulate_returns(rng)
    day = np.arange(ret.size)

    z99, z95 = stats.norm.ppf(1.0 - P99), stats.norm.ppf(1.0 - P95)
    var99, var95 = z99 * sigma, z95 * sigma
    exc99 = ret < -var99
    exc95 = ret < -var95
    n99, n95 = int(exc99.sum()), int(exc95.sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], hspace=0.42,
                          wspace=0.30, left=0.065, right=0.975, top=0.905,
                          bottom=0.105)
    ax_ret = fig.add_subplot(gs[0, 0])
    ax_cum = fig.add_subplot(gs[1, 0], sharex=ax_ret)
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_ret)
    ax_cnt = fig.add_subplot(gs[1, 1])

    panel_returns(ax_ret, day, ret, var99, var95, exc99)
    panel_cumulative(ax_cum, day, exc99, ret.size)
    panel_hist(ax_hist, ret)
    p99, p95 = panel_counts(ax_cnt, n99, n95, ret.size)

    plt.setp(ax_ret.get_xticklabels(), visible=False)
    ax_ret.tick_params(axis="x", length=0)

    verdict = ("model passes" if p99 >= 0.05 else "model rejected")

    panel_labels_inches(fig, [ax_ret, ax_cum, ax_hist, ax_cnt],
                        dx=-0.42, dy=0.06)
    savefig(fig, "570_fin_var_backtest.png")


if __name__ == "__main__":
    main()
