"""A 16S qPCR standard series: curves first, then every number taken from them.

Eighteen amplification traces (six ten-fold standards in triplicate) plus a
no-template control are generated as logistic curves whose half-point is set
by the template copies and a single per-cycle amplification factor.  Nothing
after that is written by hand.

The threshold is placed at ten baseline standard deviations above the mean of
the early cycles, measured on the traces themselves; each Ct is the linear
interpolation of the trace across that threshold; the standard curve in (b)
is an ordinary least-squares fit of those Ct values against log10 copies; and
the amplification efficiency is 10^(-1/slope) - 1 of that fit.  The melt peaks
in (c) are the argmax of the numerical derivative of the melt traces, refined
by a parabola through the three points around the maximum, so the reported
melting temperature is a property of the drawn curve.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, get_palette, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
NTC_COL = "#8a8a8a"

N_CYCLES = 40
LOG_COPIES = np.array([7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
N_REP = 3
E_TRUE = 0.94                 # per-cycle amplification efficiency used to draw
BASE_CYCLES = slice(2, 10)    # cycles used for the baseline estimate
HALF_TOP = 17.5               # half-point of the most concentrated standard
THRESH_SD = 10.0

T_LO, T_HI = 65.0, 95.0
TM_AMPLICON = 84.3
TM_DIMER = 75.6

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.24, 5.55, 4.22)
AUX_T = (7.44, 3.60, 3.18, 1.86)
AUX_B = (7.44, 1.24, 3.18, 1.72)


def amplification(rng):
    """Logistic traces, one per well, plus a flat no-template control."""
    n = np.arange(1, N_CYCLES + 1, dtype=float)
    per_cycle = np.log(1.0 + E_TRUE)
    curves = np.empty((len(LOG_COPIES), N_REP, N_CYCLES))
    for i, lc in enumerate(LOG_COPIES):
        # Half-point moves by one 1/log(1+E) step per decade of template.
        half = HALF_TOP + (7.0 - lc) * np.log(10.0) / per_cycle
        for r in range(N_REP):
            rmax = 1.00 - 0.030 * (7.0 - lc) + rng.normal(0.0, 0.012)
            shift = half + rng.normal(0.0, 0.16)
            width = 1.55 + rng.normal(0.0, 0.05)
            trace = rmax / (1.0 + np.exp(-(n - shift) / width))
            trace += rng.normal(0.0, 0.0035, N_CYCLES)
            curves[i, r] = trace - trace[BASE_CYCLES].mean()
    ntc = rng.normal(0.0, 0.0035, N_CYCLES)
    ntc = ntc - ntc[BASE_CYCLES].mean() + 0.0009 * np.maximum(n - 30.0, 0.0)
    return n, curves, ntc


def threshold_from(curves):
    """Ten baseline standard deviations above the baseline mean."""
    base = curves[:, :, BASE_CYCLES].ravel()
    return float(base.mean() + THRESH_SD * base.std(ddof=1))


def crossing(n, trace, thr):
    """Fractional cycle at which a trace first rises through the threshold."""
    above = np.where(trace >= thr)[0]
    if above.size == 0 or above[0] == 0:
        return np.nan
    j = above[0]
    y0, y1 = trace[j - 1], trace[j]
    return float(n[j - 1] + (thr - y0) / (y1 - y0) * (n[j] - n[j - 1]))


def melt_traces(rng):
    """Melt fluorescence and its negative derivative, on a 0.2 C grid."""
    t = np.arange(T_LO, T_HI + 0.001, 0.2)
    out = np.empty((len(LOG_COPIES), len(t)))
    for i, lc in enumerate(LOG_COPIES):
        amp = 0.92 - 0.045 * (7.0 - lc)
        f = amp / (1.0 + np.exp((t - TM_AMPLICON) / 0.62))
        f += 0.05 / (1.0 + np.exp((t - TM_DIMER) / 0.55)) * (lc <= 3.0)
        f = f + 0.06 + rng.normal(0.0, 0.0022, t.size)
        out[i] = f
    ntc = 0.16 / (1.0 + np.exp((t - TM_DIMER) / 0.55)) + 0.06
    ntc = ntc + rng.normal(0.0, 0.0022, t.size)
    deriv = -np.gradient(out, t, axis=1)
    dntc = -np.gradient(ntc, t)
    return t, deriv, dntc


def peak_temperature(t, d):
    """Argmax of the derivative, refined by a parabola through three points."""
    k = int(np.argmax(d))
    if k <= 0 or k >= len(t) - 1:
        return float(t[k])
    y0, y1, y2 = d[k - 1], d[k], d[k + 1]
    denom = y0 - 2.0 * y1 + y2
    off = 0.0 if abs(denom) < 1.0e-12 else 0.5 * (y0 - y2) / denom
    return float(t[k] + off * (t[1] - t[0]))


def amp_panel(ax, n, curves, ntc, thr, ct, cols):
    for i, lc in enumerate(LOG_COPIES):
        for r in range(N_REP):
            ax.plot(n, curves[i, r], color=cols[i], lw=1.15, zorder=4,
                    label=("$10^{{{:.0f}}}$ copies".format(lc)
                           if r == 0 else None))
        cm = float(np.nanmean(ct[i]))
        ax.plot([cm], [thr], marker="o", ms=4.2, mfc="white", mec=cols[i],
                mew=1.2, zorder=6)
        ax.plot([cm, cm], [0.0, thr], color=cols[i], lw=0.6,
                ls=(0, (1.6, 1.8)), zorder=3)
        ax.text(cm, thr + 0.022, "{:.1f}".format(cm), ha="center",
                va="bottom", fontsize=6, color=cols[i], rotation=90,
                zorder=7)
    ax.plot(n, ntc, color=NTC_COL, lw=1.05, ls=(0, (3.2, 2.0)), zorder=4,
            label="no-template control")

    ax.axhline(thr, color="#333333", lw=0.8, ls=(0, (3.4, 2.2)), zorder=5)
    ax.text(0.985, 0.978,
            "dashed line: threshold = baseline mean + {:.0f} SD = {:.4f} "
            "$\\Delta$Rn".format(THRESH_SD, thr), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=SUB, zorder=7)
    ax.axhline(0.0, color=HAIR, lw=0.6, zorder=1)

    ax.set_xlim(0.0, N_CYCLES + 0.5)
    ax.set_ylim(-0.045, 1.13)
    ax.set_xlabel("cycle number (-)", fontsize=7.4)
    ax.set_ylabel("baseline-corrected fluorescence $\\Delta$Rn (-)",
                  fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.015, 0.978, "numbers above the threshold are the mean Ct of "
            "{:d} replicate wells".format(N_REP), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2, color=MUTED, zorder=7)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.013, 0.935),
                    fontsize=6.6, handlelength=1.5, handletextpad=0.45,
                    labelspacing=0.30, borderpad=0.40, frameon=True,
                    framealpha=0.93, edgecolor=HAIR)
    leg.get_frame().set_linewidth(0.6)
    ax.set_title("Amplification of a six-point ten-fold standard series, "
                 "{:d} wells per point".format(N_REP), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def standard_panel(ax, ct, cols):
    x = np.repeat(LOG_COPIES, N_REP)
    y = ct.ravel()
    ok = np.isfinite(y)
    fit = stats.linregress(x[ok], y[ok])
    eff = 10.0 ** (-1.0 / fit.slope) - 1.0

    grid = np.linspace(LOG_COPIES.min() - 0.35, LOG_COPIES.max() + 0.35, 50)
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#333333", lw=1.0,
            zorder=3)
    for i, lc in enumerate(LOG_COPIES):
        v = ct[i][np.isfinite(ct[i])]
        ax.errorbar(lc, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=4.6,
                    mfc=lighten(cols[i], 0.30), mec=cols[i], mew=0.9,
                    ecolor=cols[i], elinewidth=0.9, capsize=2.0, zorder=5)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_xticks(LOG_COPIES[::-1])
    ax.set_xlabel("template ($\\log_{10}$ copies per reaction)", fontsize=7.0)
    ax.set_ylabel("Ct (cycles)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.985, 0.965,
            "Ct = {:.2f} $\\log_{{10}}$N + {:.2f}\nslope {:.3f} cycles per "
            "decade\nE = $10^{{-1/slope}}-1$ = {:.1f} %\n$R^2$ = {:.4f},  "
            "{:d} wells".format(fit.slope, fit.intercept, fit.slope,
                                100.0 * eff, fit.rvalue ** 2, int(ok.sum())),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93), zorder=8)
    ax.set_title("Standard curve fitted to the Ct values read off panel (a)",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)
    return fit, eff


def melt_panel(ax, t, deriv, dntc, cols):
    for i in range(len(LOG_COPIES)):
        ax.plot(t, deriv[i], color=cols[i], lw=1.05, zorder=4)
    ax.plot(t, dntc, color=NTC_COL, lw=1.0, ls=(0, (3.2, 2.0)), zorder=4)

    tm = peak_temperature(t, deriv[0])
    tm_ntc = peak_temperature(t, dntc)
    top = float(deriv.max())
    ax.annotate("amplicon $T_m$ = {:.1f} $^\\circ$C".format(tm),
                (tm, deriv[0].max()), xytext=(7.0, 2.0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.2, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax.annotate("primer dimer in the NTC\n$T_m$ = {:.1f} $^\\circ$C"
                .format(tm_ntc), (tm_ntc, dntc.max()), xytext=(-4.0, 9.0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.1, color=SUB, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))

    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(-0.02 * top, top * 1.42)
    ax.set_xlabel("temperature ($^\\circ$C)", fontsize=7.0)
    ax.set_ylabel("$-\\mathrm{d}F/\\mathrm{d}T$ (1/$^\\circ$C)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Melt curves: one specific product per standard",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)
    return tm, tm_ntc


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9644)
    n, curves, ntc = amplification(rng)
    thr = threshold_from(curves)
    ct = np.array([[crossing(n, curves[i, r], thr) for r in range(N_REP)]
                   for i in range(len(LOG_COPIES))])
    assert np.all(np.isfinite(ct)), "every standard must cross the threshold"
    t, deriv, dntc = melt_traces(rng)

    ramp = get_palette("cb_sequential_blue", 9)[2:]
    cols = list(ramp[::-1])[:len(LOG_COPIES)]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_amp = axes_inches(fig, MAIN)
    ax_std = axes_inches(fig, AUX_T)
    ax_melt = axes_inches(fig, AUX_B)

    amp_panel(ax_amp, n, curves, ntc, thr, ct, cols)
    fit, eff = standard_panel(ax_std, ct, cols)
    tm, tm_ntc = melt_panel(ax_melt, t, deriv, dntc, cols)

    cv = 100.0 * ct.std(axis=1, ddof=1) / ct.mean(axis=1)
    panel_labels_inches(fig, [ax_amp], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_std, ax_melt], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "964_mgn_qpcr_standard.png")


if __name__ == "__main__":
    main()
