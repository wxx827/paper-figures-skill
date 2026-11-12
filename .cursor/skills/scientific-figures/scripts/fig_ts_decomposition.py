"""STL decomposition of a monthly series into trend, season and remainder.

Four flush panels share one x axis - observed, trend, seasonal and remainder -
and each carries a narrow companion strip on the right showing how much of the
total variance that component accounts for.  The strips are deliberately a
fraction of the main panel width: they are evidence, not the message.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_label_inches

OBS_C = "#3a3a3a"      # history / observed: house dark neutral
TREND_C = "#3b5b92"    # house blue
SEAS_C = "#2e7d5b"     # house green
RESID_C = "#c9a227"    # house gold
YEAR0 = 2017
N_YEAR = 8


def synth_series(rng):
    """Additive series whose trend visibly stalls part-way through."""
    n = 12 * N_YEAR
    t = np.arange(n, dtype=float)
    ramp = 118.0 + 0.62 * t
    stall = np.clip(t - 12 * 4.5, 0.0, None)
    trend = ramp - 0.585 * stall + 3.0 * np.sin(2 * np.pi * t / 62.0)
    season = (9.4 * np.sin(2 * np.pi * t / 12.0)
              + 3.8 * np.cos(4 * np.pi * t / 12.0)
              - 2.1 * np.sin(6 * np.pi * t / 12.0))
    resid = rng.normal(0.0, 2.5, n)
    resid[[19, 52, 77]] += np.array([8.4, -9.1, 7.2])
    return t, trend, season, resid


def strip(ax, share, color, last=False):
    """One-bar variance-share strip; all strips share the 0-100 % scale."""
    ax.barh([0.0], [share], height=0.52, color=color, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(min(share + 4.0, 96.0), 0.0, f"{share:.0f}%", ha="left",
            va="center", fontsize=6.5, color="#555555")
    ax.tick_params(axis="x", labelsize=6.5, length=2.2)
    if last:
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("variance\nshare (%)", fontsize=6.5, color="#555555",
                      linespacing=1.25)
    else:
        ax.set_xticks([0, 50, 100])
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid(axis="x", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    t, trend, season, resid = synth_series(rng)
    obs = trend + season + resid

    parts = [trend, season, resid]
    variances = np.array([np.var(p) for p in parts])
    shares = 100.0 * variances / variances.sum()

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.0, 0.155],
                  hspace=0.14, wspace=0.045, left=0.095, right=0.965,
                  top=0.915, bottom=0.115)
    ax = [fig.add_subplot(gs[0, 0])]
    for r in range(1, 4):
        ax.append(fig.add_subplot(gs[r, 0], sharex=ax[0]))
    bx = [fig.add_subplot(gs[r, 1]) for r in range(4)]

    ax[0].plot(t, obs, color=OBS_C, lw=1.0)
    ax[0].set_ylabel("Observed\n(kt)", linespacing=1.25)

    ax[1].plot(t, obs, color="#cfcfcf", lw=0.7, zorder=1)
    ax[1].plot(t, trend, color=TREND_C, lw=1.7, zorder=3)
    ax[1].set_ylabel("Trend\n(kt)", linespacing=1.25)

    ax[2].plot(t, season, color=SEAS_C, lw=1.1)
    ax[2].axhline(0.0, color="#b8b8b8", lw=0.6)
    ax[2].set_ylabel("Seasonal\n(kt)", linespacing=1.25)

    ax[3].vlines(t, 0.0, resid, color=RESID_C, lw=0.9)
    ax[3].axhline(0.0, color="#808080", lw=0.7)
    ax[3].set_ylabel("Remainder\n(kt)", linespacing=1.25)

    for a in ax:
        a.axvline(12 * 4.5, color="#b0b0b0", lw=0.7, ls=(0, (2.2, 2.0)),
                  zorder=0)
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        style_axes(a)

    ax[0].annotate("trend break", xy=(12 * 4.5, obs.max()),
                   xytext=(12 * 4.5 + 4.0, obs.max() + 1.0),
                   fontsize=6.8, color="#555555", ha="left", va="top")

    for a in ax[:3]:
        plt.setp(a.get_xticklabels(), visible=False)
        a.tick_params(axis="x", length=0)
    ax[3].set_xticks(np.arange(0, 12 * N_YEAR + 1, 12))
    ax[3].set_xticklabels([str(YEAR0 + k) for k in range(N_YEAR + 1)],
                          fontsize=7.5)
    ax[3].set_xlim(-2.0, 12 * N_YEAR + 1.0)
    ax[3].set_xlabel("Calendar year (monthly observations)")

    bx[0].set_axis_off()
    bx[0].text(0.0, 0.5, "share of\ndecomposed\nvariance", fontsize=6.5,
               color="#777777", ha="left", va="center", linespacing=1.35,
               transform=bx[0].transAxes)
    for b, s, c, last in zip(bx[1:], shares, [TREND_C, SEAS_C, RESID_C],
                             [False, False, True]):
        strip(b, s, c, last=last)
        style_axes(b)

    for a, lab in zip(ax, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label_inches(fig, a, lab, dx=-0.62, dy=0.02)

    savefig(fig, "354_ts_decomposition.png")


if __name__ == "__main__":
    main()
