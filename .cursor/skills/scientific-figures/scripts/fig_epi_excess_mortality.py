"""Excess mortality against a seasonal baseline fitted to pre-pandemic years.

Main panel: weekly all-cause deaths, the harmonic-regression expectation with
its 95% prediction interval, and the signed difference filled in red where
deaths run above expectation and in blue where they run below. Auxiliary
panel: the same excess aggregated to months and decomposed by age group,
positives stacked upward and negatives downward so the totals still add up.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label, \
    get_palette

INK = "#1a1a1a"
BLUE = "#3b5b92"
ALERT = "#b23a48"
BAND = "#dfe6ef"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))

START = "2017-01-01"
N_WEEK = 313
PRE = 157                    # weeks used to fit the baseline (three full years)
PERIOD = 365.25 / 7.0

AGE_GROUPS = ["0-44", "45-64", "65-74", "75+"]
AGE_SHARE = np.array([0.04, 0.16, 0.27, 0.53])


def design(t):
    ph = 2 * np.pi * t / PERIOD
    return np.column_stack([np.ones_like(t), t / 100.0,
                            np.sin(ph), np.cos(ph),
                            np.sin(2 * ph), np.cos(2 * ph)])


def synth(rng):
    t = np.arange(N_WEEK, dtype=float)
    ph = 2 * np.pi * t / PERIOD
    truth = (9800 + 3.4 * t + 620 * np.cos(ph - 0.30)
             + 130 * np.cos(2 * ph + 0.8))
    waves = (2900 * np.exp(-0.5 * ((t - 169) / 3.6) ** 2)
             + 1500 * np.exp(-0.5 * ((t - 200) / 6.5) ** 2)
             + 2300 * np.exp(-0.5 * ((t - 218) / 5.2) ** 2)
             + 1100 * np.exp(-0.5 * ((t - 257) / 7.5) ** 2))
    dip = -(430 * np.exp(-0.5 * ((t - 183) / 5.0) ** 2)      # displacement
            + 380 * np.exp(-0.5 * ((t - 234) / 7.0) ** 2))   # absent flu season
    obs = truth + waves + dip + rng.normal(0, 175, N_WEEK)
    return t, obs


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(3)
    t, obs = synth(rng)
    dates = pd.date_range(START, periods=N_WEEK, freq="7D")

    X = design(t)
    beta, *_ = np.linalg.lstsq(X[:PRE], obs[:PRE], rcond=None)
    base = X @ beta
    sd = float(np.std(obs[:PRE] - base[:PRE], ddof=X.shape[1]))
    lo, hi = base - 1.96 * sd, base + 1.96 * sd
    excess = obs - base

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.2, 4.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 0.55], hspace=0.20,
                         left=0.088, right=0.985, top=0.845, bottom=0.115))

    # ---- (a) observed vs expected ------------------------------------------
    ax.fill_between(dates, lo, hi, color=BAND, lw=0, zorder=2,
                    label="95% prediction interval")
    ax.fill_between(dates, base, obs, where=obs >= base, interpolate=True,
                    color=ALERT, alpha=0.32, lw=0, zorder=3,
                    label="excess deaths")
    ax.fill_between(dates, base, obs, where=obs < base, interpolate=True,
                    color=BLUE, alpha=0.30, lw=0, zorder=3,
                    label="deficit")
    ax.plot(dates, base, color=BLUE, lw=1.3, ls=DASH, zorder=4,
            label="expected (seasonal baseline)")
    ax.plot(dates, obs, color=INK, lw=1.3, zorder=5, label="observed deaths")

    ax.axvline(dates[PRE], color=RULE, ls=DASH, lw=0.9, zorder=6)
    ax.text(dates[PRE] - pd.Timedelta(days=8), obs.max() * 1.005,
            "baseline fitted on\nthese weeks", fontsize=6.4, color=RULE,
            ha="right", va="top", linespacing=1.25)

    k = int(np.argmax(excess))
    tot = excess[PRE:].sum()
    ax.annotate(f"peak week {excess[k]:+,.0f} deaths\n({dates[k]:%d %b %Y})",
                xy=(dates[k], obs[k]),
                xytext=(dates[k] + pd.Timedelta(days=90), obs.max() * 0.995),
                fontsize=6.6, color=ALERT, linespacing=1.3, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ALERT))
    ax.text(0.985, 0.055,
            f"cumulative excess since {dates[PRE]:%b %Y}: {tot:+,.0f} deaths",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=INK)

    ax.set_ylim(obs.min() * 0.92, obs.max() * 1.10)
    ax.set_ylabel("All-cause deaths (per week)")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=5,
              fontsize=6.8, frameon=False, handlelength=1.4,
              columnspacing=1.1, borderaxespad=0.0)
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.078, y=1.10)

    # ---- (b) monthly excess by age group -----------------------------------
    share = np.tile(AGE_SHARE, (N_WEEK, 1))
    drift = np.linspace(-0.055, 0.055, N_WEEK)[:, None] * np.array(
        [1.4, 1.0, -0.4, -2.0])                      # later waves hit younger
    share = np.clip(share + drift, 0.01, None)
    share /= share.sum(axis=1, keepdims=True)
    by_age = excess[:, None] * share

    frame = pd.DataFrame(by_age, index=dates, columns=AGE_GROUPS)
    monthly = frame.resample("MS").sum()
    mdates_x = monthly.index + pd.Timedelta(days=14)
    cols = get_palette("cb_sequential_blue", 6)[1:5]

    pos = np.zeros(len(monthly))
    neg = np.zeros(len(monthly))
    for g, c in zip(AGE_GROUPS, cols):
        v = monthly[g].values
        up = np.clip(v, 0, None)
        dn = np.clip(v, None, 0)
        axb.bar(mdates_x, up, bottom=pos, width=24, color=c, label=g,
                edgecolor="white", linewidth=0.2, zorder=3)
        axb.bar(mdates_x, dn, bottom=neg, width=24, color=c,
                edgecolor="white", linewidth=0.2, zorder=3)
        pos += up
        neg += dn
    axb.axhline(0.0, color=RULE, lw=0.8, zorder=4)

    axb.set_ylabel("Excess (per month)")
    axb.set_xlabel("Week ending (2017-2022)")
    axb.set_xlim(dates[0] - pd.Timedelta(days=10),
                 dates[-1] + pd.Timedelta(days=10))
    axb.xaxis.set_major_locator(mdates.YearLocator())
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axb.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    axb.tick_params(labelsize=7.2)
    axb.legend(loc="upper left", fontsize=6.6, frameon=False, ncol=4,
               handlelength=1.2, columnspacing=1.0, borderaxespad=0.3,
               title="Age group (years)", title_fontsize=6.6)
    style_axes(axb)
    add_panel_label(axb, "(b)", x=-0.078, y=1.07)

    savefig(fig, "429_epi_excess_mortality.png")


if __name__ == "__main__":
    main()
