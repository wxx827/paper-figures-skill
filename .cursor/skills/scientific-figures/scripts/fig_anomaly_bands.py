"""Stacked anomaly-band time series for three coupled climate indicators.

Each panel fills the departure from a 1961-1990 baseline in red (positive) and
blue (negative), overlays a 13-month running mean and shares one x-axis. ENSO
warm and cold episodes, diagnosed from the Nino 3.4 index in panel (a), are
drawn as vertical background bands across all three panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from pubstyle import set_pub_style, savefig, style_axes

WARM, COOL = "#b23a48", "#2d6ca2"
WARM_BG, COOL_BG = "#f7dcd2", "#dbe6f0"
BASE = (1961.0, 1991.0)

EL_NINO = [(1957.9, 1.7), (1965.9, 1.4), (1972.9, 2.0), (1982.9, 2.6),
           (1986.9, 1.3), (1991.9, 1.7), (1997.9, 2.7), (2002.9, 1.3),
           (2009.9, 1.5), (2015.9, 2.5), (2018.9, 0.9), (2023.7, 1.9)]
LA_NINA = [(1955.9, -1.9), (1970.9, -1.4), (1973.9, -2.0), (1975.9, -1.6),
           (1988.9, -1.9), (1998.9, -1.6), (1999.9, -1.7), (2007.9, -1.6),
           (2010.9, -1.7), (2017.9, -1.0), (2020.9, -1.3), (2021.9, -1.2)]
BIG_EVENTS = [(1982.9, "1982/83"), (1997.9, "1997/98"), (2015.9, "2015/16")]


def running(x, k):
    return uniform_filter1d(x, k, mode="nearest")


def nino34(t, rng):
    x = 0.65 * gaussian_filter1d(rng.normal(size=t.size), 7.0)
    x /= x.std() / 0.55
    for t0, a in EL_NINO + LA_NINA:
        x += a * np.exp(-((t - t0) / 0.62) ** 2)
    return x


def episodes(t, idx, sign, min_len=5):
    """Contiguous runs where the 5-month mean index passes +-0.5 degC."""
    m = running(idx, 5)
    hit = (m >= 0.5) if sign > 0 else (m <= -0.5)
    out, start = [], None
    for i, h in enumerate(hit):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= min_len:
                out.append((t[start], t[i - 1]))
            start = None
    if start is not None and hit.size - start >= min_len:
        out.append((t[start], t[-1]))
    return out


def deseason(t, x):
    """Remove the 1961-1990 mean so anomalies are baseline-relative."""
    m = (t >= BASE[0]) & (t < BASE[1])
    return x - x[m].mean()


def main():
    set_pub_style()
    rng = np.random.default_rng(81)

    t = 1950.0 + np.arange(900) / 12.0
    nino = nino34(t, rng)

    # OLR responds with opposite sign and a short lag; land air temperature
    # follows ENSO with a ~5 month lag on top of a warming trend.
    lag = 3
    nlag = np.concatenate([np.full(lag, nino[0]), nino[:-lag]])
    olr = (-5.4 * nlag
           + 3.2 * gaussian_filter1d(rng.normal(size=t.size), 4.0)
           - 0.022 * (t - t[0]))
    lag2 = 5
    n2 = np.concatenate([np.full(lag2, nino[0]), nino[:-lag2]])
    tas = (0.11 * n2
           + 0.20 * gaussian_filter1d(rng.normal(size=t.size), 3.0)
           + 0.0175 * (t - t[0]) + 0.0001 * (t - t[0]) ** 2)

    series = [
        (deseason(t, nino), "Nino 3.4 SST", "$\\degree$C", "(a)"),
        (deseason(t, olr), "Tropical Pacific OLR", "W m$^{-2}$", "(b)"),
        (deseason(t, tas), "Global land 2 m $T$", "$\\degree$C", "(c)"),
    ]
    warm_ep = episodes(t, nino, +1)
    cool_ep = episodes(t, nino, -1)

    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(3, 1, left=0.085, right=0.985, top=0.885,
                          bottom=0.075, hspace=0.16)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for ax, (y, name, unit, tag) in zip(axes, series):
        for a, b in warm_ep:
            ax.axvspan(a, b, color=WARM_BG, lw=0, zorder=0)
        for a, b in cool_ep:
            ax.axvspan(a, b, color=COOL_BG, lw=0, zorder=0)

        ys = running(y, 3)
        ax.fill_between(t, 0, ys, where=ys >= 0, color=WARM, alpha=0.68,
                        lw=0, interpolate=True, zorder=2)
        ax.fill_between(t, 0, ys, where=ys < 0, color=COOL, alpha=0.68,
                        lw=0, interpolate=True, zorder=2)
        ax.plot(t, running(y, 13), color="#1a1a1a", lw=1.7, zorder=3)
        ax.axhline(0, color="#555555", lw=0.8, zorder=3.5)

        lim = 1.14 * np.abs(ys).max()
        ax.set_ylim(-lim, lim)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel(f"{name} anomaly\n({unit})", fontsize=8, linespacing=1.3)
        style_axes(ax)
        ax.text(-0.068, 1.02, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")

        trend = np.polyfit(t, y, 1)[0] * 10
        note = f"trend {trend:+.2f} {unit} decade$^{{-1}}$"
        if tag != "(a)":
            rho = np.corrcoef(running(nino, 5), running(y, 5))[0, 1]
            note += f"     $r$(Nino 3.4) = {rho:+.2f}"
        ax.text(0.998, 0.045, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, color="#333333", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                          alpha=0.78))

    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(np.arange(1950, 2026, 10))

    # thresholds and event captions on the driver panel
    ax0 = axes[0]
    ax0.set_ylim(-3.2, 4.7)
    ax0.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    for lev in (0.5, -0.5):
        ax0.axhline(lev, color="#666666", lw=0.7, ls=(0, (3, 2)), zorder=3.4)
    ax0.text(0.006, 0.985, "dashed lines: $\\pm$0.5 $\\degree$C episode "
             "threshold", transform=ax0.transAxes, fontsize=6.8,
             color="#555555", va="top", ha="left", zorder=6)
    for t0, lbl in BIG_EVENTS:
        ax0.annotate(lbl, xy=(t0, 2.8), xytext=(t0, 4.0),
                     ha="center", va="bottom", fontsize=7, fontweight="bold",
                     color="#8c2f2f", zorder=6,
                     arrowprops=dict(arrowstyle="-", color="#8c2f2f", lw=0.7,
                                     shrinkA=1, shrinkB=1))

    handles = [
        Patch(facecolor=WARM, alpha=0.68, label="positive anomaly"),
        Patch(facecolor=COOL, alpha=0.68, label="negative anomaly"),
        plt.Line2D([], [], color="#1a1a1a", lw=1.7, label="13-month mean"),
        Patch(facecolor=WARM_BG, label="El Nino episode"),
        Patch(facecolor=COOL_BG, label="La Nina episode"),
    ]
    savefig(fig, "81_anomaly_bands.png")


if __name__ == "__main__":
    main()
