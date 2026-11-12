"""Double-plotted actogram: entrainment, then a free run that drifts later.

The main panel follows the chronobiology convention exactly - every raster row
spans 48 h, and the right half of one row is repeated as the left half of the
row below.  That redundancy is the whole point: a phase drift that is almost
invisible in a single-plot raster becomes an unmistakable diagonal, because the
reader can follow one rest onset across two adjacent rows without their eye
wrapping around the edge of the panel.  The first six days are entrained to a
12:12 light-dark cycle, then the light bar is switched off and the rest onsets
free run.  The period drawn on the regression line, the periodogram peak and
the amplitude collapse in the averaged profile all come from the same activity
array that fills the raster.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import chi2

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

BLUE = "#3b5b92"
RED = "#b23a48"
BAR = "#3a3a3a"          # activity bars; never pure black (large dark fill)
REST = lighten(BLUE, 0.72)

DT_H = 5.0 / 60.0        # 5-min epochs
N_DAYS = 15
N_ROWS = N_DAYS - 1
DAY_START_H = 12.0       # raster rows begin at noon
LD_DAYS = 6              # days 1-6 under 12:12 light-dark
TAU_TRUE = 24.55         # free-running period imposed on the simulation
LIGHTS_ON, LIGHTS_OFF = 7.0, 19.0


def actogram_series(seed=11):
    """Activity counts per 5-min epoch plus the true rest onsets, in hours."""
    rng = np.random.default_rng(seed)
    n = int(round(N_DAYS * 24.0 / DT_H))
    t = np.arange(n) * DT_H
    rest = np.zeros(n, dtype=bool)
    onsets, offsets = [], []
    for d in range(N_DAYS):
        if d < LD_DAYS:
            on = (LIGHTS_OFF - DAY_START_H + 4.0) + 24.0 * d   # 23:00
            dur = 8.0
        else:
            base = (LIGHTS_OFF - DAY_START_H + 4.0) + 24.0 * LD_DAYS
            on = base + (d - LD_DAYS) * TAU_TRUE
            dur = 8.0 * TAU_TRUE / 24.0
        onsets.append(on)
        offsets.append(on + dur)
        rest |= (t >= on) & (t < on + dur)

    env = np.where(rest, 0.05, 1.0)
    k = 7
    env = np.convolve(env, np.ones(k) / k, mode="same")
    # afternoon trough, and a weaker overall rhythm once the light cue is gone
    phase = np.zeros(n)
    for d in range(N_DAYS):
        phase += np.exp(-0.5 * ((t - (24.0 * d + 3.0)) / 1.4) ** 2)
    env = env * (1.0 - 0.34 * np.clip(phase, 0.0, 1.0))
    damp = np.where(t < 24.0 * LD_DAYS, 1.0, 0.80)
    act = 430.0 * env * damp * rng.gamma(4.0, 0.25, n)
    act += rng.gamma(1.0, 4.0, n)                    # sensor floor
    return t, act, np.array(onsets), np.array(offsets), rest


def fit_tau(onsets):
    """Free-running period as the slope of onset time against cycle number."""
    d = np.arange(LD_DAYS, N_DAYS)
    return float(np.polyfit(d - LD_DAYS, onsets[LD_DAYS:], 1)[0])


def chi_square_periodogram(y, periods):
    """Sokolove-Bushell Qp, divided by its own p = 0.001 critical value.

    A Lomb-Scargle fit is a pure sinusoid, and a rest-activity rhythm is not:
    the 16 h / 8 h duty cycle puts enough power in the second harmonic to pull
    the best-fit sinusoid a few tenths of an hour short of the true period.
    Folding the record at each trial period instead makes no waveform
    assumption.  Each curve is scaled by the chi-square critical value for its
    own degrees of freedom, so a single line at 1.0 is the significance
    threshold for both.
    """
    n = y.size
    ybar = float(y.mean())
    den = float(((y - ybar) ** 2).sum())
    q = np.zeros(periods.size)
    for i, per in enumerate(periods):
        p = int(round(per / DT_H))
        rows = n // p
        if rows < 2 or p < 2:
            continue
        mh = y[:rows * p].reshape(rows, p).mean(axis=0)
        # the leading ``rows`` converts the variance of a cycle mean back to
        # the variance of a single sample, which is what makes Qp chi-square
        qp = rows * (rows * p) * float(((mh - ybar) ** 2).sum()) / den
        q[i] = qp / chi2.ppf(0.999, p - 1)
    return q


def periodograms(t, act):
    """Normalised chi-square periodogram for the entrained and free-run blocks."""
    periods = np.arange(18.0, 30.0 + DT_H * 0.5, DT_H)
    blocks = {"entrained (days 1-%d)" % LD_DAYS: t < 24.0 * LD_DAYS,
              "free run (days %d-%d)" % (LD_DAYS + 1, N_DAYS):
                  t >= 24.0 * LD_DAYS}
    out = {name: chi_square_periodogram(act[m], periods)
           for name, m in blocks.items()}
    return periods, out, 1.0


def daily_profile(t, act):
    """Mean activity against clock hour, separately for the two blocks."""
    hour = (t + DAY_START_H) % 24.0
    edges = np.arange(0.0, 24.5, 1.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for name, m in [("entrained", t < 24.0 * LD_DAYS),
                    ("free run", t >= 24.0 * LD_DAYS)]:
        mu = np.array([act[m & (hour >= a) & (hour < b)].mean()
                       for a, b in zip(edges[:-1], edges[1:])])
        sd = np.array([act[m & (hour >= a) & (hour < b)].std() /
                       np.sqrt(max((m & (hour >= a) & (hour < b)).sum(), 1))
                       for a, b in zip(edges[:-1], edges[1:])])
        out[name] = (mu, sd)
    return ctr, out


def clock_labels(hours):
    return ["%02d:00" % int(round(h) % 24) for h in hours]


def raster_panel(ax, t, act, onsets, offsets, tau):
    a = act / np.percentile(act, 99.5)
    a = np.clip(a, 0.0, 1.0)

    for r in range(N_ROWS):
        if r >= LD_DAYS:
            ax.add_patch(Rectangle((0.0, r + 0.02), 48.0, 0.96,
                                   facecolor="#f0f0f0", edgecolor="none",
                                   zorder=0))
    for on, off in zip(onsets, offsets):
        for r in range(N_ROWS):
            x0, x1 = on - 24.0 * r, off - 24.0 * r
            if x1 <= 0.0 or x0 >= 48.0:
                continue
            ax.add_patch(Rectangle((max(x0, 0.0), r + 0.02),
                                   min(x1, 48.0) - max(x0, 0.0), 0.96,
                                   facecolor=REST, edgecolor="none",
                                   alpha=0.85, zorder=1))

    for r in range(N_ROWS):
        m = (t >= 24.0 * r) & (t < 24.0 * r + 48.0)
        x = t[m] - 24.0 * r
        base = r + 0.97
        ax.fill_between(x, base, base - 0.92 * a[m], step="mid", color=BAR,
                        linewidth=0.0, zorder=3)

    # light bar: dark from 19:00 to 07:00, repeated for the second day
    for x0, x1, dark in [(0.0, LIGHTS_OFF - DAY_START_H, False),
                         (LIGHTS_OFF - DAY_START_H,
                          LIGHTS_ON - DAY_START_H + 24.0, True),
                         (LIGHTS_ON - DAY_START_H + 24.0,
                          LIGHTS_OFF - DAY_START_H + 24.0, False),
                         (LIGHTS_OFF - DAY_START_H + 24.0, 48.0, True)]:
        ax.add_patch(Rectangle((x0, -1.06), x1 - x0, 0.52,
                               facecolor="#1a1a1a" if dark else "white",
                               edgecolor=RULE, linewidth=0.6, zorder=4,
                               clip_on=False))
    ax.text(-0.6, -0.80, "LD", ha="right", va="center", fontsize=6.6,
            color=SUB)

    rows = np.arange(LD_DAYS, N_ROWS)
    xs = onsets[LD_DAYS:N_ROWS] - 24.0 * rows
    ax.plot(xs, rows + 0.50, color=RED, lw=1.1, marker="o", ms=2.6,
            mfc="white", mew=0.8, zorder=6)
    ax.plot(xs + 24.0, rows + 0.50, color=RED, lw=1.1, marker="o", ms=2.6,
            mfc="white", mew=0.8, zorder=6)
    ax.axhline(float(LD_DAYS), color=RED, lw=0.9, ls=(0, (4, 2.4)), zorder=6)
    ax.text(0.35, float(LD_DAYS) - 0.10, "release into constant dim light",
            ha="left", va="bottom", fontsize=6.8, color=RED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.88))
    ax.text(xs[-1] + 24.0 + 0.6, N_ROWS - 1.2,
            r"free run, $\tau$ = %.2f h" % tau, ha="left", va="center",
            fontsize=7.0, color=RED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.88))

    ax.set_xlim(0.0, 48.0)
    ax.set_ylim(float(N_ROWS), -1.30)
    ticks = np.arange(0.0, 48.1, 6.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(clock_labels(DAY_START_H + ticks), fontsize=7.0)
    ax.set_yticks(np.arange(N_ROWS) + 0.5)
    ax.set_yticklabels(["%d" % (r + 1) for r in range(N_ROWS)], fontsize=6.8)
    ax.set_xlabel("clock time over two consecutive days (h:min)")
    ax.set_ylabel("day of recording (-)")
    ax.spines["left"].set_visible(True)
    ax.tick_params(length=2.5)
    ax.set_title("Double-plotted raster: each row repeats the following day "
                 "on its right half", fontsize=9.0, fontweight="bold",
                 color=INK)


def periodogram_panel(ax, periods, pg, thr):
    top = max(v.max() for v in pg.values())
    for name, col in zip(pg, [BLUE, RED]):
        ax.plot(periods, pg[name], color=col, lw=1.3, label=name)
    ax.axhline(thr, color="#808080", lw=0.8, ls=(0, (3, 2.2)))
    ax.text(29.7, thr + 0.012 * top, "p = 0.001", ha="right",
            va="bottom", fontsize=6.2, color=MUTED)
    for (name, col, ha, dx) in [(list(pg)[0], BLUE, "right", -0.35),
                                (list(pg)[1], RED, "left", 0.35)]:
        i = int(np.argmax(pg[name]))
        ax.plot([periods[i]], [pg[name][i]], marker="v", ms=3.6, color=col)
        ax.text(periods[i] + dx, pg[name][i] + 0.055 * top,
                r"$\tau$ = %.2f h" % periods[i], ha=ha, va="bottom",
                fontsize=6.4, color=col)
    ax.set_xlim(18.0, 30.0)
    ax.set_ylim(0.0, top * 1.42)
    ax.set_xlabel("period (h)", fontsize=7.4)
    ax.set_ylabel(r"$Q_p$ / $Q_{p,\,0.001}$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.3, borderaxespad=0.15)
    ax.set_title("Chi-square periodogram", fontsize=8.2, fontweight="bold",
                 color=INK)


def profile_panel(ax, ctr, prof):
    ax.axvspan(0.0, LIGHTS_ON, color="#e8e8e8", lw=0.0, zorder=0)
    ax.axvspan(LIGHTS_OFF, 24.0, color="#e8e8e8", lw=0.0, zorder=0)
    for (name, (mu, sd)), col in zip(prof.items(), [BLUE, RED]):
        ax.fill_between(ctr, mu - sd, mu + sd, color=col, alpha=0.18,
                        linewidth=0.0, zorder=2)
        ax.plot(ctr, mu, color=col, lw=1.4, zorder=3, label=name)
    amp = {k: (v[0].max() - v[0].min()) for k, v in prof.items()}
    drop = 100.0 * (1.0 - amp["free run"] / amp["entrained"])
    ax.text(0.98, 0.96, "profile amplitude down %.0f %%" % drop,
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=INK)
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0.0, 24.1, 4.0))
    ax.set_xticklabels(clock_labels(np.arange(0.0, 24.1, 4.0)), fontsize=6.5)
    ax.set_xlabel("clock time (h:min)", fontsize=7.4)
    ax.set_ylabel("activity (counts / 5 min)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.3, borderaxespad=0.15)
    ax.set_title("Averaged daily activity profile", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    t, act, onsets, offsets, rest = actogram_series()
    tau = fit_tau(onsets)
    periods, pg, thr = periodograms(t, act)
    ctr, prof = daily_profile(t, act)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.80, 0.92, 5.30, 4.45))
    ax_top = axes_inches(fig, (7.00, 3.505, 2.85, 1.865))
    ax_bot = axes_inches(fig, (7.00, 0.92, 2.85, 1.865))

    raster_panel(ax_main, t, act, onsets, offsets, tau)
    periodogram_panel(ax_top, periods, pg, thr)
    profile_panel(ax_bot, ctr, prof)

    note = ("Synthetic wrist actigraphy, 5-min epochs over %d days; bars are "
            "activity counts scaled to the 99.5th percentile, pale bands are "
            "scored rest, and the black-and-white bar gives the 12:12 "
            "light-dark schedule of the first %d days. The regression line "
            "through the free-run onsets, the periodogram peaks and the "
            "amplitude loss are all measured from the plotted counts."
            % (N_DAYS, LD_DAYS))
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.50, dy=0.13)
    savefig(fig, "787_slp_actigraphy_raster.png")


if __name__ == "__main__":
    main()
