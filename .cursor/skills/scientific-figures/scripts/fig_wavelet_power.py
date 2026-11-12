"""Continuous Morlet wavelet power spectrum of a synthetic climate index.

Top: the monthly index with its linear trend. Bottom left: normalized wavelet
power on a log2 period axis with the 95 % red-noise significance contour and a
shaded cone of influence. Bottom right: the time-averaged (global) wavelet
spectrum against its red-noise confidence level.

The transform follows Torrence & Compo (1998) and is implemented directly with
numpy FFTs, so no external wavelet package is required.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2
from pubstyle import set_pub_style, savefig, style_axes

POWER_CMAP = LinearSegmentedColormap.from_list(
    "wavepow", ["#ffffff", "#e6eef4", "#c3d9e8", "#8fbcd4", "#5b98ba",
                "#3a7396", "#2b4f74", "#f0b957", "#d4703c", "#a8342a"])
W0 = 6.0
CHI2_95_DOF2 = chi2.ppf(0.95, 2)
PMIN, PMAX = 1.0, 32.0


def morlet_cwt(x, dt, dj=0.10, s0=None, w0=W0):
    """Return (power, period, scales, coi) for a real 1-D series."""
    n = x.size
    xn = x - x.mean()
    npad = int(2 ** np.ceil(np.log2(n)))
    xp = np.concatenate([xn, np.zeros(npad - n)])
    fx = np.fft.fft(xp)
    k = 2 * np.pi * np.fft.fftfreq(npad, d=dt)

    s0 = 2 * dt if s0 is None else s0
    jtot = int(np.floor(np.log2(n * dt / s0) / dj))
    scales = s0 * 2.0 ** (dj * np.arange(jtot + 1))

    kpos = k > 0
    wave = np.empty((scales.size, npad), dtype=complex)
    for i, s in enumerate(scales):
        norm = np.sqrt(2 * np.pi * s / dt) * np.pi ** -0.25
        psi = norm * np.exp(-0.5 * (s * k - w0) ** 2) * kpos
        wave[i] = np.fft.ifft(fx * psi)
    wave = wave[:, :n]

    ffac = 4 * np.pi / (w0 + np.sqrt(2 + w0 ** 2))
    period = ffac * scales
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n)) + 1e-3
    coi = ffac / np.sqrt(2) * dt * edge
    return np.abs(wave) ** 2, period, scales, coi


def red_noise_spectrum(period, dt, alpha):
    """Theoretical AR(1) background power at each wavelet period."""
    fk = dt / period
    return (1 - alpha ** 2) / (1 - 2 * alpha * np.cos(2 * np.pi * fk) + alpha ** 2)


def make_index(rng, t):
    """Monthly climate index: wandering ENSO band, transient solar-like
    oscillation, a stable multidecadal mode, AR(1) noise and a warming trend."""
    dt = t[1] - t[0]
    per = 4.5 + 1.5 * np.sin(2 * np.pi * (t - t[0]) / 58.0)
    phase = 2 * np.pi * np.cumsum(dt / per)
    amp = 0.85 + 0.75 * np.tanh((t - 1968.0) / 11.0)
    x = amp * np.sin(phase)
    x += 0.75 * np.exp(-((t - 1946.0) / 15.0) ** 2) * np.sin(2 * np.pi * t / 11.0)
    x += 0.55 * np.sin(2 * np.pi * (t - 1930.0) / 24.0)
    a, e = 0.86, rng.normal(size=t.size)
    r = np.zeros(t.size)
    for i in range(1, t.size):
        r[i] = a * r[i - 1] + e[i]
    x += 0.62 * r / r.std()
    x += 0.011 * (t - t[0])
    return x


def main():
    set_pub_style()
    rng = np.random.default_rng(80)

    dt = 1.0 / 12.0
    t = 1921.0 + np.arange(1200) * dt
    x = make_index(rng, t)
    xs = (x - x.mean()) / x.std()
    var = xs.var()
    a1 = float(np.corrcoef(xs[:-1], xs[1:])[0, 1])
    a2 = float(np.corrcoef(xs[:-2], xs[2:])[0, 1])
    alpha = 0.5 * (a1 + np.sqrt(max(a2, 0.0)))      # Torrence & Compo estimate

    power, period, scales, coi = morlet_cwt(xs, dt)
    power = power / var
    bg = red_noise_spectrum(period, dt, alpha)
    sig95 = bg[:, None] * CHI2_95_DOF2
    ratio = power / sig95

    gws = power.mean(1)
    dof = 2 * np.sqrt(1 + ((t.size * dt) / (2.32 * scales)) ** 2)
    gws_sig = bg * chi2.ppf(0.95, dof) / dof

    keep = (period >= PMIN) & (period <= PMAX)
    period, power, ratio, gws, gws_sig = (period[keep], power[keep],
                                          ratio[keep], gws[keep], gws_sig[keep])

    fig = plt.figure(figsize=(9.9, 6.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.3, 1.15],
                          height_ratios=[1.0, 2.45],
                          left=0.075, right=0.965, top=0.885, bottom=0.175,
                          wspace=0.055, hspace=0.20)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[1, 0], sharex=ax_ts)
    ax_g = fig.add_subplot(gs[1, 1])

    # ---- (a) source time series ------------------------------------------
    ax_ts.axhline(0, color="#999999", lw=0.6)
    ax_ts.fill_between(t, 0, xs, where=xs >= 0, color="#c0603f", alpha=0.30, lw=0)
    ax_ts.fill_between(t, 0, xs, where=xs < 0, color="#3b6ea5", alpha=0.30, lw=0)
    ax_ts.plot(t, xs, color="#1f2d3d", lw=0.7)
    fit = np.polyfit(t, xs, 1)
    ax_ts.plot(t, np.polyval(fit, t), color="#a8342a", lw=1.4, ls=(0, (5, 2)))
    ax_ts.set_ylabel("Index\n($\\sigma$)", fontsize=8, linespacing=1.2)
    ax_ts.set_ylim(-3.8, 3.8)
    ax_ts.set_yticks([-3, 0, 3])
    ax_ts.tick_params(labelbottom=False, length=2.5)
    style_axes(ax_ts)
    ax_ts.text(-0.072, 1.16, "(a)", transform=ax_ts.transAxes, fontsize=11,
               fontweight="bold", va="top")
    ax_ts.text(0.995, 0.95, f"AR(1) $\\alpha$ = {alpha:.2f}     "
               f"trend {fit[0] * 10:+.2f} $\\sigma$ decade$^{{-1}}$",
               transform=ax_ts.transAxes, ha="right", va="top", fontsize=7,
               color="#444444")

    # ---- (b) wavelet power ------------------------------------------------
    levels = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    cf = ax_w.contourf(t, period, power, levels=levels, cmap=POWER_CMAP,
                       extend="max")
    ax_w.contour(t, period, ratio, levels=[1.0], colors="#111111",
                 linewidths=1.0)

    coi_c = np.clip(coi, PMIN, PMAX)
    ax_w.fill_between(t, coi_c, PMAX, color="#ffffff", alpha=0.62, lw=0,
                      zorder=3)
    ax_w.fill_between(t, coi_c, PMAX, facecolor="none", edgecolor="#6b6b6b",
                      hatch="////", lw=0.0, alpha=0.45, zorder=3.1)
    ax_w.plot(t, coi_c, color="#333333", lw=1.0, ls=(0, (4, 2)), zorder=3.2)

    ax_w.set_yscale("log", base=2)
    ax_w.set_ylim(PMAX, PMIN)
    ax_w.set_yticks([1, 2, 4, 8, 16, 32])
    ax_w.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax_w.set_ylabel("Period (years)")
    ax_w.set_xlabel("Year")
    ax_w.set_xlim(t[0], t[-1])
    for s in ax_w.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax_w.tick_params(length=3, which="both")
    ax_w.text(-0.072, 1.045, "(b)", transform=ax_w.transAxes, fontsize=11,
              fontweight="bold", va="bottom")

    # ---- (c) global wavelet spectrum --------------------------------------
    ax_g.plot(gws, period, color="#1f2d3d", lw=1.5)
    ax_g.fill_betweenx(period, 1e-3, gws, color="#5b98ba", alpha=0.25, lw=0)
    ax_g.plot(gws_sig, period, color="#a8342a", lw=1.1, ls=(0, (4, 2)))
    ax_g.set_yscale("log", base=2)
    ax_g.set_ylim(PMAX, PMIN)
    ax_g.set_yticks([1, 2, 4, 8, 16, 32])
    ax_g.set_xscale("log")
    ax_g.set_xlim(0.4, 190)
    ax_g.set_xticks([1, 10, 100])
    ax_g.set_xticklabels(["1", "10", "100"])
    ax_g.tick_params(labelleft=False, length=2.5, which="both")
    ax_g.set_xlabel("Mean power\n($\\sigma^{2}$)", fontsize=8, linespacing=1.2)
    style_axes(ax_g)
    ax_g.text(0.02, 1.045, "(c)", transform=ax_g.transAxes, fontsize=11,
              fontweight="bold", va="bottom")

    band = period[(gws > gws_sig) & (period > 2) & (period < 9)]
    if band.size:
        ax_g.axhspan(band.min(), band.max(), color="#c0603f", alpha=0.13, lw=0)
        ax_g.text(0.96, np.sqrt(band.min() * band.max()),
                  f"{band.min():.1f}-{band.max():.1f} yr",
                  transform=ax_g.get_yaxis_transform(), ha="right",
                  va="center", fontsize=6.8, color="#8c2f2f",
                  fontweight="bold")

    # ---- shared colorbar + key -------------------------------------------
    cax = fig.add_axes([0.105, 0.055, 0.31, 0.024])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=levels)
    cb.set_label("Wavelet power (normalized, $\\sigma^{2}$)", fontsize=8,
                 labelpad=2)
    cb.ax.set_xticklabels(["0", "0.25", "0.5", "1", "2", "4", "8", "16"])
    cb.ax.tick_params(labelsize=6.8, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    handles = [
        plt.Line2D([], [], color="#111111", lw=1.0,
                   label="95 % significance (b)"),
        plt.Line2D([], [], color="#333333", lw=1.0, ls=(0, (4, 2)),
                   label="cone of influence"),
        plt.Line2D([], [], color="#a8342a", lw=1.1, ls=(0, (4, 2)),
                   label="95 % red-noise level (c)"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.71, 0.062),
               frameon=False, fontsize=7.2, ncol=3, handlelength=1.8,
               columnspacing=1.3, handletextpad=0.5)

    savefig(fig, "80_wavelet_power.png")


if __name__ == "__main__":
    main()
