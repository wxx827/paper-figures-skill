"""Three-panel financial chart: OHLC candlesticks, volume and MACD.

(a) daily candlesticks with MA5 / MA20 overlays and the latest golden cross
    flagged; (b) turnover bars coloured by the day's direction; (c) the MACD
    histogram with its DIF / DEA lines.  The panels share one x axis and sit
    flush against each other, as in a trading terminal.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

UP, DOWN = "#2f7d5b", "#b8474d"
MA5_C, MA20_C = "#c9a227", "#3b5b92"
N_DAYS = 96


def simulate_ohlcv(rng, n=N_DAYS, p0=42.0):
    """Random-walk OHLCV that respects high >= max(o,c), low <= min(o,c)."""
    drift = np.linspace(-0.0012, 0.0026, n)
    vol = 0.016 * (1.0 + 0.45 * np.sin(np.linspace(0, 3.1, n)))
    ret = drift + rng.normal(0, 1, n) * vol
    close = p0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = p0
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.003, n - 1))
    hi_pad = np.abs(rng.normal(0, 1, n)) * vol * close * 0.85
    lo_pad = np.abs(rng.normal(0, 1, n)) * vol * close * 0.85
    high = np.maximum(open_, close) + hi_pad
    low = np.minimum(open_, close) - lo_pad
    body = np.abs(close - open_) / close
    volume = (7.5 + 130 * body + rng.gamma(2.0, 1.1, n)) * 1e6
    volume *= 1.0 + 0.25 * np.sin(np.linspace(0, 6.0, n))
    return open_, high, low, close, volume


def ema(x, span):
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def moving_average(x, w):
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def panel_price(ax, idx, o, h, l, c, ma5, ma20):
    up = c >= o
    for i in idx:
        col = UP if up[i] else DOWN
        ax.vlines(i, l[i], h[i], color=col, lw=0.75, zorder=3)
        lo = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), (h.max() - l.min()) * 0.0018)
        ax.add_patch(Rectangle((i - 0.30, lo), 0.60, height, facecolor=col,
                               edgecolor=col, lw=0.5, zorder=4))
    ax.plot(idx, ma5, color=MA5_C, lw=1.15, zorder=5, label="MA5")
    ax.plot(idx, ma20, color=MA20_C, lw=1.15, zorder=5, label="MA20")

    cross = np.where((ma5[1:] > ma20[1:]) & (ma5[:-1] <= ma20[:-1]))[0] + 1
    if cross.size:
        k = int(cross[-1])
        ax.annotate("golden cross\n(MA5 crosses MA20)",
                    xy=(k, ma5[k]), xytext=(k - 17, l.min() + 0.055 *
                                            (h.max() - l.min())),
                    fontsize=6.8, color="#2f5d4a", ha="left", va="bottom",
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#2f5d4a",
                                    connectionstyle="arc3,rad=-0.25"))

    ax.axhline(c[-1], color="#8a8a8a", lw=0.7, ls=(0, (1.5, 2)), zorder=2)
    ax.text(idx[-1] + 1.6, c[-1], f"{c[-1]:.2f}", fontsize=7,
            va="center", ha="left", color="#1a1a1a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="#f2f2f2", ec="#cccccc",
                      lw=0.5))

    ax.set_ylabel("Price (USD)")
    ax.grid(axis="y", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    ret = c[-1] / c[0] - 1.0
    ax.text(0.012, 0.965,
            f"SYNTH-A  daily OHLC   |   period return {ret * 100:+.1f}%   |   "
            f"realised vol {np.std(np.diff(np.log(c))) * np.sqrt(252) * 100:.1f}% p.a.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color="#333333")
    style_axes(ax)


def panel_volume(ax, idx, o, c, v):
    up = c >= o
    cols = np.where(up, UP, DOWN)
    ax.bar(idx, v / 1e6, width=0.62, color=cols, alpha=0.78, lw=0)
    vma = moving_average(v / 1e6, 5)
    ax.plot(idx, vma, color="#5a5a5a", lw=0.9, label="5-day mean")
    ax.set_ylabel("Volume\n(M shares)", linespacing=1.3)
    ax.set_ylim(0, (v / 1e6).max() * 1.28)
    ax.grid(axis="y", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.8, handlelength=1.3,
              borderaxespad=0.2)
    style_axes(ax)


def panel_macd(ax, idx, dif, dea, hist):
    cols = np.where(hist >= 0, UP, DOWN)
    ax.bar(idx, hist, width=0.62, color=cols, alpha=0.8, lw=0)
    ax.plot(idx, dif, color="#3b5b92", lw=1.0, label="DIF (12,26)")
    ax.plot(idx, dea, color="#c9a227", lw=1.0, label="DEA (9)")
    ax.axhline(0, color="#666666", lw=0.7)
    ax.set_ylabel("MACD\n(USD)", linespacing=1.3)
    ax.legend(loc="upper left", fontsize=6.8, ncol=2, handlelength=1.3,
              columnspacing=1.0, borderaxespad=0.2)
    stack = np.concatenate([dif, dea, hist])
    ax.set_ylim(np.nanmin(stack) * 1.25, np.nanmax(stack) * 2.3)
    ax.grid(axis="y", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    o, h, l, c, v = simulate_ohlcv(rng)
    idx = np.arange(c.size)
    dates = pd.bdate_range("2024-01-02", periods=c.size)

    ma5, ma20 = moving_average(c, 5), moving_average(c, 20)
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    hist = 2.0 * (dif - dea)

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3.0, 1.0, 1.25],
                  hspace=0.08, left=0.095, right=0.935, top=0.955,
                  bottom=0.085)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    panel_price(ax1, idx, o, h, l, c, ma5, ma20)
    panel_volume(ax2, idx, o, c, v)
    panel_macd(ax3, idx, dif, dea, hist)

    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
    ticks = idx[::16]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([d.strftime("%d %b") for d in dates[::16]],
                        fontsize=7.5)
    ax3.set_xlim(-1.5, idx[-1] + 4.5)
    ax3.set_xlabel("Trading day, 2024")

    handles = [
        Line2D([], [], color=UP, lw=4, label="Close >= open (up day)"),
        Line2D([], [], color=DOWN, lw=4, label="Close < open (down day)"),
        Line2D([], [], color=MA5_C, lw=1.3, label="MA5"),
        Line2D([], [], color=MA20_C, lw=1.3, label="MA20"),
    ]
    ax1.legend(handles=handles, loc="lower left", fontsize=7,
               bbox_to_anchor=(0.005, 0.01), ncol=2, handlelength=1.4,
               columnspacing=1.4, frameon=True, facecolor="white",
               edgecolor="#e0e0e0", framealpha=0.95, borderpad=0.45)

    add_panel_label(ax1, "(a)", x=-0.085, y=1.005)
    add_panel_label(ax2, "(b)", x=-0.085, y=0.98)
    add_panel_label(ax3, "(c)", x=-0.085, y=0.98)

    savefig(fig, "105_candlestick_panel.png")


if __name__ == "__main__":
    main()
