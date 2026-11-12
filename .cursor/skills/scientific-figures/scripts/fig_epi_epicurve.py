"""Epidemic curve by date of onset, stacked by transmission setting.

Main panel: daily confirmed cases as stacked bars coloured by the setting in
which transmission is believed to have occurred, a 7-day centred moving
average on top, and dashed markers for the three control measures. Auxiliary
panels: the serial-interval distribution used to drive the renewal process,
and the cumulative case total by setting.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy import stats
from figkit import panel_label_inches
from pubstyle import set_pub_style, savefig, style_axes

SETTINGS = ["Household", "School / workplace", "Community", "Travel-associated"]
SET_C = ["#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))          # one dash style for every event / threshold

START = "2021-02-15"
N_DAYS = 133
SI_SHAPE, SI_SCALE = 2.9, 1.9   # serial interval ~ Gamma(mean 5.5 d)

EVENTS = [(30, "School closure"), (49, "Stay-at-home\norder"),
          (92, "Phased reopening")]


def serial_interval(max_lag=20):
    s = np.arange(1, max_lag + 1)
    w = stats.gamma.pdf(s, SI_SHAPE, scale=SI_SCALE)
    return s, w / w.sum()


def rt_profile(n):
    """Piecewise-smooth Rt: free spread, two clampdowns, then reopening."""
    t = np.arange(n, dtype=float)
    def step(t0, k=0.28):
        return 1.0 / (1.0 + np.exp(-(t - t0) / k / 3.0))
    return (2.45 - 1.05 * step(30) - 0.62 * step(49) + 0.48 * step(92))


def simulate(rng, n):
    s, w = serial_interval()
    rt = rt_profile(n)
    cases = np.zeros(n)
    cases[:6] = rng.poisson(2.2, 6) + 1
    for t in range(6, n):
        lag = np.arange(1, min(len(s), t) + 1)
        force = float(np.sum(cases[t - lag] * w[lag - 1]))
        cases[t] = rng.poisson(max(rt[t] * force, 0.05))
    # weekday reporting artefact: fewer onsets recorded at the weekend
    wk = np.array([1.06, 1.05, 1.04, 1.00, 0.99, 0.93, 0.90])
    cases = np.round(cases * wk[np.arange(n) % 7]).astype(int)
    return cases, rt


def split_settings(rng, cases, n):
    """Time-varying attribution of each case to a transmission setting."""
    ph = np.linspace(0.0, 1.0, n)[:, None]
    early = np.array([0.22, 0.30, 0.16, 0.32])
    late = np.array([0.40, 0.22, 0.35, 0.03])
    p = early * (1 - ph) + late * ph
    p /= p.sum(axis=1, keepdims=True)
    out = np.zeros((n, len(SETTINGS)), dtype=int)
    for t in range(n):
        if cases[t] > 0:
            out[t] = rng.multinomial(cases[t], p[t])
    return out


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(11)
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    cases, rt = simulate(rng, N_DAYS)
    by_set = split_settings(rng, cases, N_DAYS)
    total = by_set.sum(axis=1)
    ma = pd.Series(total).rolling(7, center=True, min_periods=4).mean().values

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.50],
                          width_ratios=[1.0, 0.62], hspace=0.52, wspace=0.26,
                          left=0.078, right=0.985, top=0.855, bottom=0.095)

    # ---- (a) epidemic curve ------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(N_DAYS)
    for j, (name, col) in enumerate(zip(SETTINGS, SET_C)):
        ax.bar(dates, by_set[:, j], bottom=bottom, width=0.92, color=col,
               edgecolor="white", linewidth=0.15, label=name, zorder=3)
        bottom += by_set[:, j]
    ax.plot(dates, ma, color=INK, lw=1.6, zorder=5,
            label="7-day moving average")

    top = float(np.nanmax(total)) * 1.36
    for day, label in EVENTS:
        ax.axvline(dates[day], color=RULE, ls=DASH, lw=0.9, zorder=6)
        ax.text(dates[day] + pd.Timedelta(days=1.5), top * 0.985, label,
                fontsize=6.6, color=RULE, ha="left", va="top",
                linespacing=1.25, zorder=7)

    k = int(np.nanargmax(ma))
    ax.annotate(f"peak 7-day mean {ma[k]:.0f} cases/day on {dates[k]:%d %b}",
                xy=(dates[k], ma[k]),
                xytext=(dates[k] + pd.Timedelta(days=20), top * 0.86),
                fontsize=6.8, color=INK, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))

    ax.set_ylim(0, top)
    ax.set_xlim(dates[0] - pd.Timedelta(days=1),
                dates[-1] + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.set_ylabel("Confirmed cases (per day)")
    ax.set_xlabel("Date of symptom onset (2021)")
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), fontsize=7.0,
              frameon=False, ncol=5, handlelength=1.4, columnspacing=1.3,
              borderaxespad=0.0)
    style_axes(ax)

    # ---- (b) serial-interval distribution ----------------------------------
    axb = fig.add_subplot(gs[1, 0])
    draws = stats.gamma.rvs(SI_SHAPE, scale=SI_SCALE, size=4000,
                            random_state=np.random.default_rng(4))
    draws = draws[draws < 20]
    axb.hist(draws, bins=np.arange(0, 20.5, 1.0), color="#9cc3dd",
             edgecolor="white", linewidth=0.4, density=True, zorder=3)
    xg = np.linspace(0, 20, 300)
    axb.plot(xg, stats.gamma.pdf(xg, SI_SHAPE, scale=SI_SCALE), color="#134a76",
             lw=1.5, zorder=4)
    m = float(draws.mean())
    axb.axvline(m, color=RULE, ls=DASH, lw=0.9, zorder=5)
    axb.text(m + 0.5, axb.get_ylim()[1] * 0.93, f"mean {m:.1f} d",
             fontsize=6.6, color=RULE, ha="left", va="top")
    axb.set_xlim(0, 20)
    axb.set_xlabel("Serial interval (days)", fontsize=7.6)
    axb.set_ylabel("Density (1/day)", fontsize=7.6)
    axb.set_title("Generation interval", fontsize=8.5, fontweight="bold",
                  loc="left", pad=3)
    axb.tick_params(labelsize=6.6)
    style_axes(axb)

    # ---- (c) cumulative total ----------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    cum = np.cumsum(by_set, axis=0)
    axc.stackplot(dates, cum.T / 1e3, colors=SET_C, edgecolor="white",
                  linewidth=0.3, zorder=3)
    grand = cum[-1].sum() / 1e3
    axc.text(dates[3], grand * 1.02, f"{grand:.1f}k cases total",
             fontsize=6.6, color=INK, va="bottom")
    axc.set_ylim(0, grand * 1.22)
    axc.set_xlim(dates[0], dates[-1])
    axc.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axc.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axc.set_ylabel("Cumulative (thousands)", fontsize=7.6)
    axc.set_xlabel("Date of onset", fontsize=7.6)
    axc.set_title("Cumulative burden by setting", fontsize=8.5,
                  fontweight="bold", loc="left", pad=3)
    axc.tick_params(labelsize=6.6)
    style_axes(axc)

    panel_label_inches(fig, ax, "(a)", dx=-0.42, dy=0.19)
    for ax_, lab in ((axb, "(b)"), (axc, "(c)")):
        panel_label_inches(fig, ax_, lab, dx=-0.42, dy=0.10)

    savefig(fig, "425_epi_epicurve.png")


if __name__ == "__main__":
    main()
