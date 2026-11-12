"""Speech intelligibility against signal-to-noise ratio, and the thresholds it yields.

Four listening conditions are put through the same sentence-in-noise test: the
points are binomial draws over a fixed number of trials per level, the curve
through them is a logistic fitted to those draws by least squares, and the
speech reception threshold is not a fitted parameter that happens to be quoted
but the level at which the *plotted* curve is interpolated to cross fifty
percent.  The slope quoted with each threshold is the derivative of that same
curve there, in percent per decibel, and the interval on each threshold comes
from the covariance the fit returns.  The audiogram panel keeps the clinical
convention that decides whether a hearing figure can be read at all: hearing
level increases downwards, so a lower trace is always the worse ear.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#8a8a8a"
HAIR = "#cccccc"

NH = "#2a7f7a"
UNAID = "#b23a48"
AID = "#134a76"
AIDNR = "#c0603f"

SEED = 1047
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

N_TRIAL = 25
LEVELS = np.arange(-18.0, 13.0, 3.0)

CONDS = (
    ("normal hearing", NH, -7.2, 2.10, "o", "-"),
    ("hearing loss, unaided", UNAID, 2.4, 3.40, "^", "-"),
    ("hearing loss, aided", AID, -0.6, 3.00, "s", "-"),
    ("aided plus noise reduction", AIDNR, -2.9, 2.70, "D", "-"),
)


def logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-(x - a) / b))


def run_condition(rng, a_true, b_true):
    """Binomial data, a least-squares logistic fit, and the curve it draws."""
    p = logistic(LEVELS, a_true, b_true)
    k = rng.binomial(N_TRIAL, p)
    obs = k / N_TRIAL
    par, cov = curve_fit(logistic, LEVELS, obs, p0=[a_true, b_true],
                         sigma=np.sqrt(np.maximum(obs * (1 - obs), 0.004)
                                       / N_TRIAL),
                         absolute_sigma=True, maxfev=20000)
    xf = np.linspace(-22.0, 16.0, 601)
    yf = logistic(xf, *par)
    srt = float(np.interp(0.5, yf, xf))            # read off the drawn curve
    i = int(np.argmin(np.abs(xf - srt)))
    slope = float((yf[i + 1] - yf[i - 1]) / (xf[i + 1] - xf[i - 1]) * 100.0)
    se = float(np.sqrt(np.diag(cov))[0])
    return dict(obs=obs, k=k, xf=xf, yf=yf, srt=srt, slope=slope, se=se,
                par=par)


def main_panel(ax, res):
    nh = res["normal hearing"]
    ax.axvspan(nh["srt"] - 1.96 * nh["se"], nh["srt"] + 1.96 * nh["se"],
               color="#e8e8e8", zorder=1)
    ax.axhline(50.0, color=GREY, lw=0.8, ls=(0, (3.4, 2.2)), zorder=3)
    ax.text(-21.4, 51.4, "50 % criterion", fontsize=6.2, color=GREY,
            ha="left", va="bottom", zorder=8)

    for name, col, _, _, mk, ls in CONDS:
        r = res[name]
        ax.plot(r["xf"], r["yf"] * 100.0, color=col, lw=1.7, ls=ls, zorder=6)
        ax.plot(LEVELS, r["obs"] * 100.0, ls="none", marker=mk, ms=4.0,
                mfc="white", mec=col, mew=1.0, zorder=7)
        ax.plot([r["srt"], r["srt"]], [0.0, 50.0], color=col, lw=0.8,
                ls=(0, (1.8, 1.8)), zorder=4)
        ax.plot([r["srt"]], [50.0], marker="o", ms=4.6, mfc="white", mec=col,
                mew=1.4, zorder=9)

    a, b = res["hearing loss, unaided"], res["aided plus noise reduction"]
    ax.annotate("", xy=(b["srt"], 92.0), xytext=(a["srt"], 92.0),
                arrowprops=dict(arrowstyle="<|-", color=AIDNR, lw=1.1,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text(0.5 * (a["srt"] + b["srt"]), 93.4,
            "%.1f dB of benefit" % (a["srt"] - b["srt"]), fontsize=6.5,
            color=AIDNR, ha="center", va="bottom", fontweight="bold",
            zorder=9)

    ax.text(-21.4, 107.0, "the impaired ear still needs %.1f dB more "
            "signal-to-noise\nthan the normal ear even with the best "
            "processing:\n%+.1f dB SNR against %+.1f dB SNR"
            % (b["srt"] - nh["srt"], b["srt"], nh["srt"]), fontsize=6.4,
            color=SUB, ha="left", va="top", linespacing=1.38, zorder=9)
    ax.text(nh["srt"] - 0.6, 15.0, "normal 95 %\ninterval",
            fontsize=6.0, color=GREY, ha="right", va="center",
            linespacing=1.30, zorder=8)

    handles = [Line2D([], [], color=c, lw=1.7, marker=m, ms=4.0, mfc="white",
                      mew=1.0, label="%s, SRT %+.1f dB SNR"
                      % (n, res[n]["srt"]))
               for n, c, _, _, m, _ in CONDS]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.2, borderaxespad=0.4,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xlim(-21.5, 15.0)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 20, 40, 50, 60, 80, 100])
    ax.set_xlabel("signal-to-noise ratio (dB SNR), speech in stationary noise")
    ax.set_ylabel("words correct (%%), %d words per level" % N_TRIAL)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Four listening conditions on one psychometric axis",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def audiogram_panel(ax, freqs, thr_nh, thr_hl):
    ax.fill_between(freqs, -10.0, 20.0, color="#ededed", zorder=1)
    ax.text(freqs[0] * 1.04, 17.0, "normal limits, 0 to 20 dB HL",
            fontsize=6.0, color=GREY, ha="left", va="bottom", zorder=8)
    ax.plot(freqs, thr_nh, color=NH, lw=1.6, marker="o", ms=3.6, mfc="white",
            mew=1.0, zorder=6, label="normal ear, PTA %.0f dB HL"
            % thr_nh[1:4].mean())
    ax.plot(freqs, thr_hl, color=UNAID, lw=1.6, marker="x", ms=4.4, mew=1.2,
            zorder=6, label="impaired ear, PTA %.0f dB HL"
            % thr_hl[1:4].mean())
    ax.annotate("", xy=(6000.0, thr_hl[-1]), xytext=(6000.0, thr_nh[-1]),
                arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(5600.0, 0.5 * (thr_hl[-1] + thr_nh[-1]), "%.0f dB\nat 6 kHz"
            % (thr_hl[-1] - thr_nh[-1]), fontsize=6.0, color=SUB, ha="right",
            va="center", linespacing=1.30, zorder=9)
    leg = ax.legend(loc="lower left", fontsize=6.0, frameon=True,
                    facecolor="white", edgecolor="#dddddd", framealpha=0.95,
                    handlelength=1.7, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(10)
    ax.set_xscale("log")
    ax.set_xlim(220.0, 9000.0)
    ax.set_ylim(85.0, -12.0)                      # hearing level runs DOWN
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["250", "500", "1k", "2k", "4k", "8k"])
    ax.minorticks_off()
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("audiometric frequency (Hz)", fontsize=6.8)
    ax.set_ylabel("hearing level (dB HL)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The ear behind the two lower curves", fontsize=8.0,
                 fontweight="bold", color=INK)


def threshold_panel(ax, res):
    names = [c[0] for c in CONDS]
    cols = [c[1] for c in CONDS]
    y = np.arange(len(names))[::-1]
    base = res["normal hearing"]["srt"]
    ax.axvline(base, color=GREY, lw=0.9, ls=(0, (3.2, 2.2)), zorder=3)
    ax.text(base - 0.4, len(names) - 0.55, "normal reference", fontsize=6.0,
            color=GREY, ha="right", va="top", rotation=90, zorder=8)
    for yi, name, col in zip(y, names, cols):
        r = res[name]
        ci = 1.96 * r["se"]
        ax.plot([r["srt"] - ci, r["srt"] + ci], [yi, yi], color=col, lw=2.6,
                solid_capstyle="round", alpha=0.42, zorder=5)
        ax.plot([base, r["srt"]], [yi, yi], color=col, lw=0.9,
                ls=(0, (1.8, 1.8)), zorder=4)
        ax.plot([r["srt"]], [yi], marker="o", ms=5.0, mfc="white", mec=col,
                mew=1.5, zorder=8)
        ax.text(r["srt"] + ci + 0.5, yi, "%+.1f dB SNR" % r["srt"],
                fontsize=6.1, color=col, ha="left", va="center", zorder=9)
    ax.set_yticks(y)
    ax.set_yticklabels(["normal", "unaided", "aided", "aided + NR"],
                       fontsize=6.6)
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-0.66, len(names) - 0.34)
    ax.set_xlabel("speech reception threshold (dB SNR), 95 % interval",
                  fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Where each curve crosses 50 %", fontsize=8.0,
                 fontweight="bold", color=INK)


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
    rng = np.random.default_rng(SEED)

    res = {}
    for name, _, a, b, _, _ in CONDS:
        res[name] = run_condition(rng, a, b)

    freqs = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0])
    thr_nh = np.array([5.0, 5.0, 10.0, 5.0, 10.0, 15.0])
    thr_hl = np.array([18.0, 22.0, 30.0, 45.0, 58.0, 65.0])

    nh, ua = res["normal hearing"], res["hearing loss, unaided"]
    ai, nr = res["hearing loss, aided"], res["aided plus noise reduction"]
    assert ua["srt"] > nh["srt"] + 5.0, \
        "the unaided impaired listener must need a better SNR"
    assert ai["srt"] < ua["srt"], "amplification has to help"
    assert nr["srt"] < ai["srt"], "noise reduction has to help further"
    assert nr["srt"] > nh["srt"], \
        "processing must not restore a normal threshold"
    assert (thr_hl > thr_nh).all(), "the impaired audiogram must be poorer"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_aud = axes_inches(fig, AUX_T)
    ax_srt = axes_inches(fig, AUX_B)

    main_panel(ax_main, res)
    audiogram_panel(ax_aud, freqs, thr_nh, thr_hl)
    threshold_panel(ax_srt, res)



    panel_labels_inches(fig, [ax_main, ax_aud, ax_srt], dx=-0.62, dy=0.14)
    savefig(fig, "1047_vox_intelligibility_snr.png")


if __name__ == "__main__":
    main()
