"""2x3 grid of mean diurnal cycles of net ecosystem CO2 exchange.

Each panel holds one month of half-hourly eddy-covariance data: the mean
diurnal course, the day-to-day interquartile range, the zero line separating
uptake from release, night-time background shading between sunset and sunrise,
and a marker on the daily extremum. Panels share both axes so the seasonal
amplification of the cycle can be read directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes

LINE, BAND = "#1f5a3d", "#7fb08e"
NIGHT = "#e6ebf1"
PEAK = "#b0432b"
NDAY = 30

#            name  sunrise sunset  reco  gppmax
MONTHS = [
    ("January", 7.9, 16.5, 0.85, 1.2),
    ("March", 6.5, 18.2, 1.70, 6.5),
    ("May", 5.1, 19.7, 3.40, 19.0),
    ("July", 4.7, 20.3, 5.60, 26.5),
    ("September", 6.4, 18.6, 4.10, 14.5),
    ("November", 7.4, 16.8, 1.55, 2.2),
]


def diurnal(rng, m):
    """Half-hourly NEE for NDAY days: respiration minus light-driven uptake."""
    _, rise, set_, reco0, gpp0 = m
    h = np.arange(48) * 0.5 + 0.25
    day = np.clip(np.sin(np.pi * (h - rise) / (set_ - rise)), 0, None)
    par = day ** 1.15

    # air temperature drives respiration through a Q10 response
    tair = np.sin(2 * np.pi * (h - 9.5) / 24.0)
    cloud = 0.20 + 0.80 * rng.beta(2.6, 1.1, (NDAY, 1))
    tanom = rng.normal(0, 3.0, (NDAY, 1))

    reco = reco0 * 2.0 ** ((3.5 * tair[None, :] + tanom) / 10.0)
    gpp = 1.10 * gpp0 * cloud * par[None, :]
    gpp *= 1.0 + 0.13 * rng.normal(size=(NDAY, 1))
    nee = reco - gpp
    nee += 0.35 * rng.normal(size=nee.shape)          # instrument noise
    return h, nee


def main():
    set_pub_style()
    rng = np.random.default_rng(86)

    fig = plt.figure(figsize=(10.6, 6.5))
    gs = fig.add_gridspec(2, 3, left=0.062, right=0.985, top=0.865,
                          bottom=0.155, wspace=0.11, hspace=0.30)

    axes, stats = [], []
    for k, m in enumerate(MONTHS):
        ax = fig.add_subplot(gs[k // 3, k % 3],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        h, nee = diurnal(rng, m)
        mean = nee.mean(0)
        q25, q75 = np.percentile(nee, [25, 75], axis=0)
        stats.append((m, h, mean, q25, q75))

    # extra headroom below keeps the peak-uptake callouts inside the frame
    ylo = min(s[3].min() for s in stats) - 4.3
    yhi = max(s[4].max() for s in stats) + 3.2

    for k, (ax, (m, h, mean, q25, q75)) in enumerate(zip(axes, stats)):
        name, rise, set_, _, _ = m
        ax.axvspan(0, rise, color=NIGHT, lw=0, zorder=0)
        ax.axvspan(set_, 24, color=NIGHT, lw=0, zorder=0)
        ax.axhline(0, color="#555555", lw=0.9, zorder=2)

        ax.fill_between(h, q25, q75, color=BAND, alpha=0.35, lw=0, zorder=3)
        ax.plot(h, mean, color=LINE, lw=2.0, zorder=5, solid_capstyle="round")

        # daily extremum: peak uptake if the canopy is active, else respiration
        if mean.min() < -1.0:
            i, tag, col = int(np.argmin(mean)), "peak uptake", PEAK
            dy, va = -1.6, "top"
        else:
            i, tag, col = int(np.argmax(mean)), "peak release", PEAK
            dy, va = 1.6, "bottom"
        ax.plot([h[i]], [mean[i]], marker="o", ms=5.0, mfc="white",
                mec=col, mew=1.4, zorder=6)
        hh, mm = divmod(int(round(h[i] * 60)), 60)
        ha = "right" if h[i] > 14 else "left"
        ax.annotate(f"{tag}\n{mean[i]:+.1f} at {hh:02d}:{mm:02d}",
                    xy=(h[i], mean[i]),
                    xytext=(h[i] + (0.6 if ha == "left" else -0.6),
                            mean[i] + dy),
                    ha=ha, va=va, fontsize=6.8, color=col, linespacing=1.3,
                    zorder=7)

        # daily carbon balance
        daily = mean.sum() * 1800.0 * 12.011e-6
        ax.text(0.03, 0.965, f"{daily:+.2f} g C m$^{{-2}}$ d$^{{-1}}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color="#333333", zorder=7,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#e0e0e0",
                          lw=0.5, alpha=0.9))
        ax.text(0.97, 0.965, f"daylength {set_ - rise:.1f} h",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                color="#666666", zorder=7)

        ax.set_xlim(0, 24)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
        style_axes(ax)
        ax.text(-0.055, 1.03, f"({chr(97 + k)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
        if k // 3 == 1:
            ax.set_xlabel("Hour of day (local standard time)", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        if k % 3 == 0:
            ax.set_ylabel("NEE ($\\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)",
                          fontsize=8)
        else:
            ax.tick_params(labelleft=False)

    handles = [
        plt.Line2D([], [], color=LINE, lw=2.0, label="Mean diurnal cycle"),
        Patch(facecolor=BAND, alpha=0.35, label="Day-to-day IQR (25-75 %)"),
        Patch(facecolor=NIGHT, label="Night"),
        plt.Line2D([], [], color=PEAK, lw=0, marker="o", ms=5.0, mfc="white",
                   mew=1.4, label="Daily extremum"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.50, 0.048),
               ncol=4, frameon=False, fontsize=8, handlelength=1.7,
               columnspacing=2.0, handletextpad=0.6)

    savefig(fig, "86_flux_diurnal.png")


if __name__ == "__main__":
    main()
