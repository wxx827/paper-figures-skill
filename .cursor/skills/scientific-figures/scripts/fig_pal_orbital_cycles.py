"""Milankovitch forcing and the ice-volume record it is supposed to explain.

Panel (a) stacks the three orbital parameters on one reversed age axis; each
trace is normalised so three very different physical amplitudes can share a
frame, and the real range is printed next to the trace name.  Panel (b) is the
point of the figure: summer insolation at 65 N is dominated by precession, yet
the ice-volume proxy underneath it is dominated by the 100 ka beat that
insolation barely carries.

The two small panels on the right supply the evidence.  Zero-padded
periodograms locate the Milankovitch peaks, and band-passed cross-correlation
gives the lag at which the ice sheets actually answer each forcing.  Every
period and lag in the labels is read off those computations, not typed in.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

PURPLE = "#8e6bbf"     # eccentricity
GREEN = "#2e7d5b"      # obliquity
GOLD = "#c9a227"       # precession
RED = "#b23a48"        # insolation (warm forcing)
BLUE = "#3b5b92"       # ice volume (cold response)
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"

PERMIL = "\u2030"
AGE_MAX = 1000.0
DT = 0.5
MILANKOVITCH = [(405.0, "405"), (100.0, "100"), (41.0, "41"), (23.0, "23")]


def orbital(seed=7):
    """Astronomical parameters, insolation and a lagged ice-volume response."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, AGE_MAX + DT, DT)

    ecc = (0.0285
           + 0.0115 * np.cos(2 * np.pi * t / 405.0 + 0.4)
           + 0.0085 * np.cos(2 * np.pi * t / 95.0 + 1.1)
           + 0.0070 * np.cos(2 * np.pi * t / 124.0 - 0.7)
           + 0.0011 * rng.normal(size=t.size))
    obl = (23.30 + 1.15 * np.cos(2 * np.pi * t / 41.0 + 0.9)
           + 0.16 * np.cos(2 * np.pi * t / 54.0)
           + 0.02 * rng.normal(size=t.size))
    prec = ecc * np.sin(2 * np.pi * t / 23.0 + 0.3) * 1.9

    ins = (476.0 - 780.0 * prec + 18.0 * (obl - obl.mean()) / obl.std()
           + 1.4 * rng.normal(size=t.size))

    # Ice volume: skewed 100 ka sawtooth locked to eccentricity, with real
    # obliquity power and only weak precession power on top.
    p100 = np.cos(2 * np.pi * t / 100.0 + 0.85 * np.sin(2 * np.pi * t / 100.0))
    ice = (1.00 * p100
           + 0.38 * np.cos(2 * np.pi * t / 41.0 + 1.6)
           + 0.16 * np.cos(2 * np.pi * t / 23.0 + 1.0)
           + 0.09 * rng.normal(size=t.size))
    ice /= np.abs(ice).max()
    d18o = 3.85 - 0.82 * ice        # benthic foraminifera, permil VPDB
    return t, ecc, obl, prec, ins, d18o


def spectrum(t, y, p_lo=16.0, p_hi=560.0, pad=8):
    """Hann-windowed, zero-padded periodogram against period in ka.

    The padding matters: without it the frequency grid of a 1000 ka record
    has no bin anywhere near 405 ka, and the eccentricity peak gets reported
    at the neighbouring bin instead.
    """
    y = np.asarray(y, float)
    y = y - y.mean()
    n = y.size
    freq = np.fft.rfftfreq(pad * n, t[1] - t[0])
    power = np.abs(np.fft.rfft(y * np.hanning(n), n=pad * n)) ** 2
    keep = (freq > 1.0 / p_hi) & (freq < 1.0 / p_lo)
    per, pw = 1.0 / freq[keep], power[keep]
    return per, pw / pw.max()


def bandpass(y, p_lo, p_hi):
    """Keep only the periods between ``p_lo`` and ``p_hi`` (ka)."""
    y = np.asarray(y, float) - np.mean(y)
    freq = np.fft.rfftfreq(y.size, DT)
    spec = np.fft.rfft(y)
    spec[(freq < 1.0 / p_hi) | (freq > 1.0 / p_lo)] = 0.0
    return np.fft.irfft(spec, n=y.size)


def lag_corr(resp, force, max_lag=55.0):
    """Correlation of ``resp`` against ``force``, positive lag = resp is later.

    The age axis runs backwards, so a *larger* index means an *older* sample.
    Shifting the response to older indices therefore makes it earlier in time;
    the returned lag axis is negated so that the sign reads the way the label
    promises.
    """
    a = (resp - resp.mean()) / resp.std()
    b = (force - force.mean()) / force.std()
    n = a.size
    steps = np.arange(-int(max_lag / DT), int(max_lag / DT) + 1)
    out = np.empty(steps.size)
    for i, k in enumerate(steps):
        if k > 0:
            out[i] = np.corrcoef(a[k:], b[:n - k])[0, 1]
        elif k < 0:
            out[i] = np.corrcoef(a[:n + k], b[-k:])[0, 1]
        else:
            out[i] = np.corrcoef(a, b)[0, 1]
    return (-steps * DT)[::-1], out[::-1]


def panel_orbits(ax, t, ecc, obl, prec, periods):
    traces = [(ecc, PURPLE, "eccentricity $e$",
               "%.3f - %.3f (-)" % (ecc.min(), ecc.max()), 5.40),
              (obl, GREEN, "obliquity $\\varepsilon$",
               "%.1f - %.1f deg" % (obl.min(), obl.max()), 2.70),
              (prec, GOLD, "precession $e\\,\\sin\\varpi$",
               "%+.3f to %+.3f (-)" % (prec.min(), prec.max()), 0.0)]
    for (y, color, name, rng_txt, off), per in zip(traces, periods):
        z = (y - y.mean()) / np.abs(y - y.mean()).max()
        ax.plot(t, z + off, color=color, lw=0.85, zorder=3)
        ax.axhline(off, color=GREY, lw=0.5, ls=(0, (2.4, 2.2)), zorder=1)
        ax.text(AGE_MAX - 12.0, off + 1.12,
                "%s,  %s,  peak at %.0f ka" % (name, rng_txt, per),
                ha="left", va="bottom", fontsize=6.9, color=color)
    ax.set_xlim(0.0, AGE_MAX)
    ax.invert_xaxis()
    ax.set_ylim(-1.35, 7.05)
    ax.set_yticks([])
    ax.set_ylabel("orbital parameters\n(normalized, offset)", fontsize=8.0)
    ax.tick_params(labelbottom=False, length=0)
    ax.spines["left"].set_visible(False)


def panel_response(ax, t, ins, d18o):
    ax.plot(t, ins, color=RED, lw=0.8, zorder=3)
    ax.set_ylabel("summer insolation, 65 N\n(W m$^{-2}$)", fontsize=8.0,
                  color=RED)
    ax.tick_params(axis="y", colors=RED, labelsize=7.0)
    ax.set_ylim(ins.min() - 14, ins.max() + 30)
    ax.set_xlim(0.0, AGE_MAX)
    ax.invert_xaxis()
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_xlabel("age (ka BP), increasing to the left", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.0)

    axr = ax.twinx()
    axr.fill_between(t, d18o.max() + 0.25, d18o, color=BLUE, alpha=0.20,
                     linewidth=0.0, zorder=1)
    axr.plot(t, d18o, color=BLUE, lw=1.0, zorder=2)
    axr.set_ylabel("benthic $\\delta^{18}O$ (%s VPDB)" % PERMIL, fontsize=8.0,
                   color=BLUE)
    axr.tick_params(axis="y", colors=BLUE, labelsize=7.0)
    axr.set_ylim(d18o.max() + 0.25, d18o.min() - 0.55)
    axr.set_xlim(0.0, AGE_MAX)
    axr.invert_xaxis()
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(BLUE)

    ax.text(AGE_MAX - 12.0, ins.max() + 24.0,
            "insolation is precession-paced; the ice-volume proxy is not "
            "(axis inverted, more ice downward)",
            ha="left", va="top", fontsize=6.8, color=SUB)


def panel_spectra(ax, lanes):
    step = 1.10
    names, colors, offs = [], [], []
    for i, (per, pw, color, name) in enumerate(lanes[::-1]):
        off = i * step
        ax.fill_between(per, off, off + pw, color=color, alpha=0.28,
                        linewidth=0.0, zorder=2)
        ax.semilogx(per, off + pw, color=color, lw=0.9, zorder=3)
        ax.axhline(off, color=GREY, lw=0.5, zorder=1)
        names.append(name)
        colors.append(color)
        offs.append(off)
    top = (len(lanes) - 1) * step + 1.0
    for p, txt in MILANKOVITCH:
        ax.axvline(p, color=GREY, lw=0.6, ls=(0, (2.6, 2.2)), zorder=1)
        ax.text(p, top + 0.30, txt, ha="center", va="bottom", fontsize=6.2,
                color=SUB)
    ax.set_xlim(16.0, 560.0)
    ax.set_ylim(-0.10, top + 0.72)
    ax.set_yticks(offs)
    ax.set_yticklabels(names, fontsize=6.6)
    for lbl, c in zip(ax.get_yticklabels(), colors):
        lbl.set_color(c)
    ax.set_xticks([20, 41, 100, 405])
    ax.set_xticklabels(["20", "41", "100", "405"])
    ax.set_xlabel("period (ka)", fontsize=7.6)
    ax.set_ylabel("normalized power (-)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Milankovitch power spectra", fontsize=8.5,
                 fontweight="bold", color=INK, pad=9)


def panel_phase(ax, curves):
    """Lagged correlation per Milankovitch band; only resolved peaks marked.

    A peak that lands on the edge of the search window is not a phase, it is
    an artefact of the window, so those bands get their peak correlation
    reported in words instead of a lag marker that would look definitive.
    """
    # Markers for the short bands land within a few ka of each other, so the
    # lag captions are fanned sideways instead of stacked on top of one another.
    fan = [(0, -13), (-19, -13), (19, -13)]
    peaks, loose = [], []
    for idx, (lag, c, half, color, name) in enumerate(curves):
        ax.plot(lag, c, color=color, lw=1.1, label=name)
        inside = np.abs(lag) <= half
        k = int(np.argmax(c[inside]))
        best, r = lag[inside][k], c[inside][k]
        if abs(best) < 0.9 * half:
            peaks.append(best)
            ax.plot([best], [r], marker="o", ms=3.4, color=color, mfc="white",
                    mew=0.9, zorder=5)
            ax.annotate("%+.1f ka" % best, xy=(best, r), xytext=fan[idx],
                        textcoords="offset points", ha="center", fontsize=6.2,
                        color=color)
        else:
            peaks.append(np.nan)
            loose.append((name, float(np.max(c[inside]))))
    ax.axvline(0.0, color=GREY, lw=0.7, ls=(0, (2.4, 2.0)))
    ax.axhline(0.0, color=GREY, lw=0.6)
    ax.set_xlim(-55.0, 55.0)
    ax.set_ylim(-1.05, 2.00)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("lag of warming behind forcing (ka)", fontsize=7.6)
    ax.set_ylabel("band-passed\ncorrelation (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    for i, (name, r) in enumerate(loose):
        ax.text(-53.0, 1.36 - 0.30 * i,
                "%s: broad, no sharp phase (r = %.2f)" % (name, r),
                ha="left", va="center", fontsize=6.2, color=PURPLE)
    ax.legend(fontsize=6.0, frameon=False, loc="upper center", ncol=3,
              handlelength=1.1, columnspacing=0.8, borderaxespad=0.05)
    ax.set_title("Phase of the ice-volume response", fontsize=8.5,
                 fontweight="bold", color=INK, pad=9)
    return peaks


def main():
    set_pub_style()
    t, ecc, obl, prec, ins, d18o = orbital()

    raw = [(ecc, PURPLE, "eccentricity"), (obl, GREEN, "obliquity"),
           (prec, GOLD, "precession"), (d18o, BLUE, "ice volume")]
    lanes = [spectrum(t, y) + (c, n) for y, c, n in raw]
    periods = [p[int(np.argmax(w))] for p, w, _, _ in lanes[:3]]

    warmth = -(d18o - d18o.mean())
    bands = [(80.0, 135.0, 50.0, ecc, PURPLE, "eccentricity"),
             (34.0, 50.0, 20.5, obl, GREEN, "obliquity"),
             (19.0, 27.0, 11.5, prec, GOLD, "precession")]
    curves = [lag_corr(bandpass(warmth, lo, hi), bandpass(f, lo, hi))
              + (half, c, n) for lo, hi, half, f, c, n in bands]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_orb = axes_inches(fig, (0.92, 3.55, 5.05, 1.95))
    ax_res = axes_inches(fig, (0.92, 0.82, 5.05, 2.08))
    ax_spc = axes_inches(fig, (7.42, 3.62, 2.55, 1.88))
    ax_phs = axes_inches(fig, (7.42, 0.95, 2.55, 1.55))

    panel_orbits(ax_orb, t, ecc, obl, prec, periods)
    panel_response(ax_res, t, ins, d18o)
    panel_spectra(ax_spc, lanes)
    peaks = panel_phase(ax_phs, curves)

    ice_per = lanes[3][0][int(np.argmax(lanes[3][1]))]

    panel_labels_inches(fig, [ax_orb, ax_res, ax_spc, ax_phs], dx=-0.74,
                        dy=0.06)
    savefig(fig, "751_pal_orbital_cycles.png")


if __name__ == "__main__":
    main()
