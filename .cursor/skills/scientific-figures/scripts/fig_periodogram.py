"""Lomb-Scargle period search on an unevenly sampled series.

Panel (a) is the raw radial-velocity series: nightly sampling over seven
months hides a 2.7 d signal completely.  Panel (b) is the normalised
Lomb-Scargle periodogram with false-alarm levels obtained by shuffling the
measurements 150 times and keeping the highest peak of each trial.  Panel (c)
is the spectral window, |sum exp(-2 pi i f t)|^2 / N^2, which explains why the
periodogram also shows peaks one cycle per day away from the true frequency.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import lombscargle
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

P_TRUE = 2.7134
K_AMP = 4.2                # m/s


def sample_times(rng):
    nights = np.arange(0, 205)
    nights = nights[rng.random(nights.size) > 0.44]
    t = []
    for n in nights:
        k = rng.integers(1, 4)
        t.append(n + 0.70 + np.sort(rng.random(k)) * 0.22)
    return np.concatenate(t)


def ls_power(t, y, f):
    yy = y - y.mean()
    return lombscargle(t, yy, 2 * np.pi * f, normalize=True)


def window_power(t, f):
    ph = np.exp(-2j * np.pi * np.outer(f, t))
    return np.abs(ph.sum(axis=1)) ** 2 / t.size ** 2


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(131)

    t = sample_times(rng)
    err = 0.55 + 0.30 * rng.random(t.size)
    y = (K_AMP * np.sin(2 * np.pi * t / P_TRUE + 0.9)
         + 1.10 * np.sin(4 * np.pi * t / P_TRUE + 2.0)
         + 1.00 * np.sin(2 * np.pi * t / 118.0)
         + rng.normal(0, err))

    f = np.logspace(np.log10(0.02), np.log10(5.0), 6000)
    p = ls_power(t, y, f)
    w = window_power(t, f)

    # false-alarm levels from 150 shuffles of the same measurements
    n_boot = 150
    peaks = np.empty(n_boot)
    for i in range(n_boot):
        peaks[i] = ls_power(t, rng.permutation(y), f).max()
    fap = {"10%": np.percentile(peaks, 90),
           "1%": np.percentile(peaks, 99),
           "0.1%": peaks.max()}

    f_best = float(f[np.argmax(p)])
    aliases = [(f_best + 1.0, "$f_0 + 1$ d$^{-1}$"),
               (abs(f_best - 1.0), "$|f_0 - 1|$ d$^{-1}$"),
               (f_best + 2.0, "$f_0 + 2$ d$^{-1}$")]

    fig = plt.figure(figsize=(10.2, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.78, 1.0, 0.50],
                          hspace=0.55, left=0.070, right=0.985, top=0.878,
                          bottom=0.072)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ---- (a) time series ----------------------------------------------------
    ax_a.errorbar(t, y, yerr=err, ls="none", marker="o", ms=2.6, mfc=NAVY,
                  mec="none", ecolor="#a9b8ca", elinewidth=0.6, alpha=0.85)
    ax_a.axhline(0.0, color=GREY, lw=0.7, ls=(0, (4, 3)))
    ax_a.set_xlim(-4, 208)
    ax_a.set_ylim(-9.4, 10.4)
    ax_a.set_xlabel("time (day)", fontsize=8.2, labelpad=2)
    ax_a.set_ylabel("radial velocity (m s$^{-1}$)", fontsize=8.2, labelpad=3)
    ax_a.tick_params(labelsize=7.0, length=2.6)
    ax_a.grid(True, lw=0.4, color="#eceef1")
    ax_a.set_axisbelow(True)
    ax_a.set_title("Unevenly sampled series: %d epochs over %d nights"
                   % (t.size, round(t.max())), fontsize=8.8,
                   fontweight="bold", pad=3)
    ax_a.text(0.006, 0.045, "the %.2f d signal is invisible by eye at this "
              "sampling" % P_TRUE, transform=ax_a.transAxes, ha="left",
              va="bottom", fontsize=6.9, zorder=8,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                        lw=0.55, alpha=0.94))

    # ---- (b) Lomb-Scargle periodogram --------------------------------------
    for fa, lab in aliases:
        ax_b.axvline(fa, color=CYAN, lw=0.7, ls=(0, (2, 2.2)), zorder=1)
        ax_b.text(fa, 0.965, lab, rotation=90, ha="right", va="top",
                  fontsize=6.4, color=CYAN, zorder=6,
                  transform=ax_b.get_xaxis_transform())
    ax_b.fill_between(f, 0, p, color=NAVY, alpha=0.18, lw=0)
    ax_b.plot(f, p, color=NAVY, lw=0.9, zorder=4)
    styles = [("10%", GOLD, (0, (6, 2.6)), 0.0225),
              ("1%", RUST, (0, (4, 2.0)), 0.105),
              ("0.1%", INDIGO, (0, (1.6, 1.8)), 0.52)]
    for name, col, ls, xpos in styles:
        ax_b.axhline(fap[name], color=col, lw=0.9, ls=ls, zorder=3)
        ax_b.text(xpos, fap[name] + 0.010, "FAP %s" % name, ha="left",
                  va="bottom", fontsize=6.6, color=col, zorder=6)
    ax_b.annotate("$f_0 = %.4f$ d$^{-1}$\n$P = %.4f$ d"
                  % (f_best, 1 / f_best), xy=(f_best, p.max()),
                  xytext=(0.115, 0.80 * p.max()), fontsize=7.6, color=RUST,
                  fontweight="bold", ha="center", va="center",
                  linespacing=1.5, zorder=8,
                  bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=RUST,
                            lw=0.6, alpha=0.94),
                  arrowprops=dict(arrowstyle="->", lw=1.0, color=RUST,
                                  shrinkB=4,
                                  connectionstyle="arc3,rad=-0.20"))
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.02, 5.0)
    ax_b.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax_b.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax_b.set_ylim(0, 1.08 * p.max())
    ax_b.set_xlabel("frequency  $f$ (cycles day$^{-1}$)", fontsize=8.4,
                    labelpad=2)
    ax_b.set_ylabel("normalised LS power", fontsize=8.2, labelpad=3)
    ax_b.tick_params(labelsize=7.0, length=2.8)
    ax_b.grid(True, lw=0.4, color="#eceef1")
    ax_b.set_axisbelow(True)
    ax_b.set_title("Lomb-Scargle periodogram with bootstrap false-alarm "
                   "levels", fontsize=8.8, fontweight="bold", pad=13)
    def _inv(v):
        v = np.asarray(v, dtype=float)
        return 1.0 / np.where(v == 0, np.nan, v)

    sec = ax_b.secondary_xaxis("top", functions=(_inv, _inv))
    sec.set_xlabel("period (day)", fontsize=7.4, labelpad=1.5)
    sec.set_xticks([0.5, 1, 2, 5, 10, 25, 50])
    sec.set_xticklabels(["0.5", "1", "2", "5", "10", "25", "50"])
    sec.tick_params(labelsize=6.6, length=2.2)

    # ---- (c) spectral window ------------------------------------------------
    ax_c.set_facecolor("#f4f5f7")
    ax_c.fill_between(f, 0, w, color=INDIGO, alpha=0.22, lw=0)
    ax_c.plot(f, w, color=INDIGO, lw=0.9)
    for fa in (1.0, 2.0):
        ax_c.axvline(fa, color=CYAN, lw=0.7, ls=(0, (2, 2.2)))
        ax_c.text(fa, 0.94, "%g d$^{-1}$" % fa, rotation=90, ha="right",
                  va="top", fontsize=6.4, color=CYAN,
                  transform=ax_c.get_xaxis_transform())
    ax_c.annotate("nightly cadence puts power at 1 and 2 cycles per day;\n"
                  "every peak in (b) is duplicated at these offsets",
                  xy=(1.0, float(w[np.argmin(np.abs(f - 1.0))])),
                  xytext=(0.075, 0.72 * w.max()), fontsize=6.9, color=INDIGO,
                  ha="left", va="center", linespacing=1.5,
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=INDIGO,
                                  shrinkB=4,
                                  connectionstyle="arc3,rad=-0.18"))
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.02, 5.0)
    ax_c.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax_c.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax_c.set_ylim(0, 1.12 * w.max())
    ax_c.set_xlabel("frequency  $f$ (cycles day$^{-1}$)", fontsize=8.0,
                    labelpad=2)
    ax_c.set_ylabel("window power", fontsize=7.4, labelpad=3)
    ax_c.tick_params(labelsize=6.8, length=2.2)
    ax_c.grid(True, lw=0.4, color="#e3e5e8")
    ax_c.set_axisbelow(True)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_c.set_title("Supporting statistic  |  spectral window of the same "
                   "time stamps", fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)

    for ax_, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        p_ = ax_.get_position()
        add_panel_label(ax_, lab, x=-0.030 / p_.width,
                        y=1.0 + 0.026 / p_.height, size=10)

    handles = [Line2D([], [], color=NAVY, lw=1.2, label="LS power"),
               Line2D([], [], color=GOLD, lw=1.0, ls=(0, (6, 2.6)),
                      label="10% false-alarm level"),
               Line2D([], [], color=RUST, lw=1.0, ls=(0, (4, 2.0)),
                      label="1% false-alarm level"),
               Line2D([], [], color=INDIGO, lw=1.0, ls=(0, (1.6, 1.8)),
                      label="0.1% false-alarm level"),
               Line2D([], [], color=CYAN, lw=0.9, ls=(0, (2, 2.2)),
                      label="one-day aliases")]
    savefig(fig, "240_periodogram.png")


if __name__ == "__main__":
    main()
