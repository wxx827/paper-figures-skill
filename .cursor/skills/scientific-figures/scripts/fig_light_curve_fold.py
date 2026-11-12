"""Variable-star light curve and its phase-folded counterpart.

Panel (a) is the raw, irregularly sampled time series in magnitudes (axis
reversed, brighter upwards) with photometric error bars; the inset zooms on
three consecutive nights so the individual pulsation cycles become visible.
Panel (b) folds every epoch on the best-fit period and repeats the cycle twice
so the shape reads continuously across the phase-1 boundary, with a four-term
Fourier model and binned means on top.  Panel (c) is the residual strip.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

PERIOD = 0.54712          # days
EPOCH = 2455000.0
MMEAN = 12.902


def light_model(phase):
    """Four-term Fourier series shaped like an RR Lyrae ab pulsator.

    A truncated sawtooth in magnitude gives the characteristic steep rise to
    maximum light followed by a long, nearly linear decline.
    """
    ph = 2 * np.pi * (np.asarray(phase) - 0.04)
    amps = [0.340, 0.150, 0.074, 0.030]
    m = np.full(np.shape(ph), MMEAN, dtype=float)
    for k, a in enumerate(amps, start=1):
        m -= a * np.sin(k * ph)
    return m


def make_data(rng):
    """Irregular nightly sampling over two months with realistic gaps."""
    nights = np.arange(0, 63)
    nights = nights[rng.random(nights.size) > 0.22]        # weathered-out
    t = []
    for n in nights:
        k = rng.integers(4, 11)
        t.append(n + 0.62 + np.sort(rng.random(k)) * 0.30)
    t = np.concatenate(t)
    err = 0.009 + 0.012 * rng.random(t.size)
    mag = light_model((t % PERIOD) / PERIOD) + rng.normal(0, err)
    return EPOCH + t, mag, err


def binned(phase, mag, err, nbin=24):
    edges = np.linspace(0, 1, nbin + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbin - 1)
    xc, yc, ye = [], [], []
    for i in range(nbin):
        m = idx == i
        if m.sum() < 2:
            continue
        w = 1.0 / err[m] ** 2
        xc.append(0.5 * (edges[i] + edges[i + 1]))
        yc.append(float(np.sum(w * mag[m]) / np.sum(w)))
        ye.append(float(np.std(mag[m], ddof=1) / np.sqrt(m.sum())))
    return np.array(xc), np.array(yc), np.array(ye)


def frame(ax):
    ax.tick_params(labelsize=7.0, length=2.6)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(23)
    t, mag, err = make_data(rng)
    phase = ((t - EPOCH) % PERIOD) / PERIOD
    resid = mag - light_model(phase)
    rms = float(np.std(resid))

    fig = plt.figure(figsize=(9.4, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.92, 1.0, 0.30],
                          hspace=0.30, left=0.082, right=0.975,
                          top=0.885, bottom=0.070)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ---- (a) raw time series ---------------------------------------------
    td = t - EPOCH
    ax_a.errorbar(td, mag, yerr=err, ls="none", marker="o", ms=2.6,
                  mfc=NAVY, mec="none", ecolor="#9fb0c6", elinewidth=0.6,
                  capsize=0, alpha=0.85, zorder=3)
    ax_a.set_ylim(14.30, 12.26)
    ax_a.set_xlim(td.min() - 1.5, td.max() + 1.5)
    ax_a.set_xlabel("HJD $-$ 2455000 (day)", fontsize=8.4, labelpad=2)
    ax_a.set_ylabel("$V$ magnitude (mag)", fontsize=8.4, labelpad=3)
    frame(ax_a)
    ax_a.set_title("Raw photometric time series, %d epochs over %.0f nights"
                   % (t.size, td.max() - td.min()), fontsize=8.8,
                   fontweight="bold", pad=3)
    ax_a.text(0.005, 0.055, "brighter", transform=ax_a.transAxes, fontsize=6.6,
              color=GREY, rotation=90, va="bottom", ha="left")
    ax_a.annotate("", xy=(0.014, 0.42), xytext=(0.014, 0.055),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))

    # zoom inset: three consecutive nights, model curve overlaid
    t0 = td.min() + 12.0
    win = (td > t0) & (td < t0 + 3.0)
    axi = ax_a.inset_axes([0.545, 0.055, 0.435, 0.365])
    gridt = np.linspace(t0, t0 + 3.0, 900)
    axi.plot(gridt, light_model((gridt % PERIOD) / PERIOD), color=GOLD,
             lw=1.1, zorder=2)
    axi.errorbar(td[win], mag[win], yerr=err[win], ls="none", marker="o",
                 ms=3.0, mfc=NAVY, mec="white", mew=0.4, ecolor="#9fb0c6",
                 elinewidth=0.6, zorder=3)
    axi.set_xlim(t0, t0 + 3.0)
    axi.set_ylim(13.52, 12.42)
    axi.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    axi.set_facecolor("#f7f8fa")
    axi.grid(True, lw=0.35, color="#e4e7eb")
    axi.set_axisbelow(True)
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    axi.text(0.5, 0.045, "zoom  |  3 nights, %d pulsation cycles"
             % round(3.0 / PERIOD), transform=axi.transAxes, ha="center",
             va="bottom", fontsize=6.4, style="italic", color="#555555",
             zorder=6, bbox=dict(boxstyle="round,pad=0.20", fc="white",
                                 ec="none", alpha=0.85))
    ax_a.indicate_inset_zoom(axi, edgecolor="#9aa0a6", lw=0.7, alpha=0.9)

    ax_a.text(0.005, 0.965, "$P = %.5f$ d      $\\langle V\\rangle = %.3f$ "
              "mag      amplitude $= %.2f$ mag"
              % (PERIOD, MMEAN, np.ptp(light_model(np.linspace(0, 1, 400)))),
              transform=ax_a.transAxes, ha="left", va="top", fontsize=7.2,
              zorder=8, bbox=dict(boxstyle="round,pad=0.34", fc="white",
                                  ec="#cccccc", lw=0.6, alpha=0.95))

    # ---- (b) phase-folded --------------------------------------------------
    for off, alpha, col in [(0.0, 0.85, NAVY), (1.0, 0.32, NAVY)]:
        ax_b.errorbar(phase + off, mag, yerr=err, ls="none", marker="o",
                      ms=2.9, mfc=col, mec="none", ecolor="#a9b8ca",
                      elinewidth=0.55, alpha=alpha, zorder=3)
    pg = np.linspace(0, 2, 900)
    ax_b.plot(pg, light_model(pg % 1.0), color=GOLD, lw=1.9, zorder=5)
    bx, by, be = binned(phase, mag, err)
    for off in (0.0, 1.0):
        ax_b.errorbar(bx + off, by, yerr=be, ls="none", marker="s", ms=4.4,
                      mfc=RUST, mec="white", mew=0.7, ecolor=RUST,
                      elinewidth=0.9, zorder=6)
    ax_b.axvline(1.0, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax_b.text(1.0, 12.455, " cycle repeated ", ha="center", va="top",
              fontsize=6.6, color=GREY,
              bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))
    ax_b.set_xlim(0, 2)
    ax_b.set_ylim(13.58, 12.44)
    ax_b.set_xticks(np.arange(0, 2.01, 0.25))
    ax_b.set_ylabel("$V$ magnitude (mag)", fontsize=8.4, labelpad=3)
    ax_b.set_xticklabels([])
    frame(ax_b)
    ax_b.set_title("Folded on $P = %.5f$ d  —  fast rise, slow decline of an "
                   "RR Lyrae ab pulsator" % PERIOD, fontsize=8.8,
                   fontweight="bold", pad=3)

    # rise / decline call-outs
    pmin = float(pg[np.argmin(light_model(pg % 1.0))])
    ax_b.annotate("maximum light", xy=(pmin, light_model(pmin % 1.0)),
                  xytext=(pmin + 0.28, 12.60), fontsize=7.4, color=RUST,
                  fontweight="bold", ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=RUST,
                                  shrinkB=4, connectionstyle="arc3,rad=0.2"))
    ax_b.annotate("steep rise\n$\\Delta\\phi\\approx0.10$", xy=(0.985, 13.05),
                  xytext=(0.815, 12.74), fontsize=7.0, color=CYAN,
                  ha="center", va="center", linespacing=1.4,
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=CYAN,
                                  shrinkB=3, connectionstyle="arc3,rad=-0.2"))
    ax_b.annotate("slow decline", xy=(1.52, 13.10), xytext=(1.62, 12.70),
                  fontsize=7.0, color=CYAN, ha="center", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=CYAN,
                                  shrinkB=3, connectionstyle="arc3,rad=0.2"))

    # ---- (c) residual strip ------------------------------------------------
    ax_c.set_facecolor("#f4f5f7")
    ax_c.axhspan(-rms, rms, color=CYAN, alpha=0.16, lw=0)
    ax_c.axhline(0.0, color=GREY, lw=0.8)
    for off, alpha in [(0.0, 0.8), (1.0, 0.30)]:
        ax_c.plot(phase + off, resid, ls="none", marker="o", ms=2.2,
                  mfc=INDIGO, mec="none", alpha=alpha)
    ax_c.set_xlim(0, 2)
    ax_c.set_ylim(-0.085, 0.085)
    ax_c.set_xticks(np.arange(0, 2.01, 0.25))
    ax_c.set_yticks([-0.05, 0.0, 0.05])
    ax_c.set_xlabel("Phase  $\\phi = (t - T_0)/P$   (two cycles shown)",
                    fontsize=8.4, labelpad=2)
    ax_c.set_ylabel("O $-$ C (mag)", fontsize=7.4, labelpad=2)
    ax_c.tick_params(labelsize=6.6, length=2.2)
    ax_c.grid(True, axis="y", lw=0.4, color="#e3e5e8")
    ax_c.set_axisbelow(True)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_c.set_title("Supporting statistic  |  residuals about the Fourier "
                   "model,  rms $= %.3f$ mag" % rms, fontsize=7.2,
                   style="italic", color="#555555", pad=2.5)

    for ax, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        add_panel_label(ax, lab, x=-0.055, y=1.02, size=10)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4, mfc=NAVY, mec="none",
               label="single epoch  ($\\pm 1\\sigma$)"),
        Line2D([], [], color=GOLD, lw=1.9, label="4-term Fourier model"),
        Line2D([], [], marker="s", ls="none", ms=5, mfc=RUST, mec="white",
               mew=0.7, label="binned mean (24 phase bins)"),
        Line2D([], [], marker="o", ls="none", ms=4, mfc=NAVY, mec="none",
               alpha=0.32, label="repeated cycle  $\\phi + 1$")]
    savefig(fig, "234_light_curve_fold.png")


if __name__ == "__main__":
    main()
