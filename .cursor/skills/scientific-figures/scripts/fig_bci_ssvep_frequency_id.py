"""SSVEP: four flicker frequencies, one occipital electrode, one decision.

Main panel: the amplitude spectrum of a four-second occipital epoch for each
of the four flicker targets, drawn as four offset traces on one frequency
axis, with the fundamental and its harmonics ticked and the signal-to-noise
ratio of each peak measured against its own neighbouring bins.  Auxiliary
panels (both 58 per cent of the main panel width): the canonical correlation
of one trial against all four reference templates with the winning margin
marked, and classification accuracy against epoch length with the Wolpaw
information transfer rate on a second axis.

Nothing is asserted that is not measured.  The three occipital channels are
built from harmonic series plus a 1/f background, the spectra are the FFT of
those channels, each SNR is the peak bin over the mean of the twenty
surrounding bins of the same plotted spectrum, the classifier is a canonical
correlation analysis solved by QR and SVD, accuracy is the share of the
trials it gets right, and the transfer rate is the Wolpaw formula applied to
that accuracy.  Synthetic data, no human subject.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
GREY = "#b3b3b3"
BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.86)

FS = 250.0                       # sampling rate, Hz
T_EPOCH = 4.0                    # epoch used for the spectra, s
TARGETS = (8.0, 10.0, 12.0, 15.0)   # flicker frequencies, Hz
COLS = (BLUE, TEAL, RED, GOLD)
N_HARM = 3                       # harmonics in the signal and the templates
CHANNELS = ("O1", "Oz", "O2")
CH_GAIN = (0.72, 1.0, 0.80)
SIG_UV = 1.70                    # fundamental amplitude at Oz, microvolts
NOISE_UV = 6.5                   # 1/f background level, microvolts
PINK_EXP = 0.55                  # slope of the background, 1/f^exp
F_MIN = 5.0
N_TRIAL = 25                     # trials per target
WINDOWS = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
GAZE_SHIFT = 0.5                 # dead time between selections, s
F_MAX = 46.0
SHOW_TRIAL = 2                   # index of the single trial shown in (b)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def pink(rng, n):
    """1/f background of unit standard deviation."""
    f = np.fft.rfftfreq(n, 1.0 / FS)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** -PINK_EXP
    spec = amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    x = np.fft.irfft(spec, n)
    return x / x.std()


def epoch(rng, f0, n):
    """Three occipital channels carrying an SSVEP at f0, in microvolts."""
    t = np.arange(n) / FS
    base = np.zeros(n)
    # Real steady-state responses shrink as the flicker gets faster; the
    # exponent keeps the four targets comparably visible.
    gain = SIG_UV * (10.0 / f0) ** 0.35
    for h in range(1, N_HARM + 1):
        base += (gain / h) * np.sin(2.0 * np.pi * h * f0 * t
                                    + rng.uniform(0.0, 2.0 * np.pi))
    return np.stack([g * base + NOISE_UV * pink(rng, n) for g in CH_GAIN],
                    axis=1)


def spectrum(x):
    """Single-sided amplitude spectrum in microvolts."""
    n = x.size
    win = np.hanning(n)
    sp = np.fft.rfft((x - x.mean()) * win)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    return f, 2.0 * np.abs(sp) / (win.sum())


def peak_snr(f, a, f0, guard=2, side=10):
    """Peak bin over the mean of the neighbouring bins, in dB."""
    k = int(np.argmin(np.abs(f - f0)))
    lo = np.r_[a[max(k - guard - side, 0):k - guard],
               a[k + guard + 1:k + guard + 1 + side]]
    return 20.0 * np.log10(a[k] / lo.mean()), a[k]


def reference(f0, n):
    """Sin/cos template bank for one frequency."""
    t = np.arange(n) / FS
    cols = []
    for h in range(1, N_HARM + 1):
        cols += [np.sin(2.0 * np.pi * h * f0 * t),
                 np.cos(2.0 * np.pi * h * f0 * t)]
    return np.stack(cols, axis=1)


def cca(x, y):
    """Largest canonical correlation between two centred matrices."""
    qx, _ = np.linalg.qr(x - x.mean(axis=0))
    qy, _ = np.linalg.qr(y - y.mean(axis=0))
    return float(np.linalg.svd(qx.T @ qy, compute_uv=False)[0])


def wolpaw(p, n_class, t_sel):
    """Information transfer rate in bits per minute."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    bits = (np.log2(n_class) + p * np.log2(p)
            + (1.0 - p) * np.log2((1.0 - p) / (n_class - 1)))
    return max(bits, 0.0) * 60.0 / t_sel


def spectra_panel(ax, spec, snr):
    # One row height for all four traces, taken from the tallest bin actually
    # drawn, so no trace can climb into its neighbour's row.
    tall = max(float(a[(f >= F_MIN) & (f <= F_MAX)].max()) for f, a in spec)
    step = 1.42 * tall
    for i, (f0, col) in enumerate(zip(TARGETS, COLS)):
        f, a = spec[i]
        base = (len(TARGETS) - 1 - i) * step
        m = (f >= F_MIN) & (f <= F_MAX)
        ax.fill_between(f[m], base, base + a[m], facecolor=col, alpha=0.13,
                        edgecolor="none", zorder=2)
        ax.plot(f[m], base + a[m], color=col, lw=0.8, zorder=4)
        for h in range(1, N_HARM + 1):
            fh = h * f0
            if fh > F_MAX:
                continue
            k = int(np.argmin(np.abs(f - fh)))
            ax.plot([fh], [base + a[k] + 0.055 * step], marker="v", ms=3.4,
                    color=col, zorder=6)
            ax.text(fh, base + a[k] + 0.10 * step,
                    "{:.0f} Hz".format(fh) if h == 1 else "{:d}f".format(h),
                    fontsize=6, color=col, ha="center", va="bottom",
                    zorder=7)
        ax.text(F_MAX - 0.4, base + 0.78 * step,
                "target {:.0f} Hz: fundamental {:.2f} $\\mu V$, "
                "SNR {:.1f} dB".format(f0, snr[i][1], snr[i][0]),
                fontsize=6.4, color=col, ha="right", va="bottom", bbox=BOX,
                zorder=8)
        ax.axhline(base, color=GREY, lw=0.6, zorder=1)
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(-0.06 * step, len(TARGETS) * step + 0.06 * step)
    ax.set_yticks([(len(TARGETS) - 1 - i) * step for i in range(len(TARGETS))])
    ax.set_yticklabels(["{:.0f} Hz".format(f0) for f0 in TARGETS],
                       fontsize=7.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude spectrum at Oz ($\\mu V$), one trace per flicker "
                  "target")
    ax.tick_params(width=0.8, length=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    bar = max([v for v in (0.05, 0.1, 0.2, 0.5, 1.0) if v <= step / 2.2],
              default=0.05)
    ax.plot([F_MAX + 0.5] * 2, [0.05 * step, 0.05 * step + bar], color=INK,
            lw=1.2, zorder=8, clip_on=False)
    ax.text(F_MAX + 1.0, 0.05 * step + bar / 2.0,
            "{:g} $\\mu V$".format(bar), fontsize=6.0, color=INK,
            ha="center", va="center", rotation=90, zorder=8, clip_on=False)
    ax.set_title("Each target answers at its own frequency and at its "
                 "harmonics", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def cca_panel(ax, scores, true_f):
    xs = np.arange(len(TARGETS))
    win = int(np.argmax(scores))
    bars = ax.bar(xs, scores, width=0.62,
                  color=[COLS[i] if i == win else "#c8ccd2"
                         for i in range(len(TARGETS))],
                  edgecolor="white", linewidth=0.7, zorder=4)
    order = np.argsort(scores)[::-1]
    margin = scores[order[0]] - scores[order[1]]
    for x, s in zip(xs, scores):
        ax.text(x, s + 0.012, "{:.3f}".format(s), fontsize=6, color=SUB,
                ha="center", va="bottom", zorder=6)
    ax.annotate("", xy=(order[0], scores[order[0]]),
                xytext=(order[0], scores[order[1]]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(order[0] + 0.42, 0.5 * (scores[order[0]] + scores[order[1]]),
            "margin {:.3f}".format(margin), fontsize=6, color=INK,
            ha="left", va="center", bbox=BOX, zorder=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["{:.0f}".format(f0) for f0 in TARGETS], fontsize=6.0)
    ax.set_ylim(0.0, max(scores) * 1.32)
    ax.set_xlabel("reference template (Hz), {:.0f} s of data"
                  .format(T_EPOCH), fontsize=6.8)
    ax.set_ylabel("canonical correlation", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(axis="y", lw=0.5, alpha=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("One trial, true target {:.0f} Hz, decided {:.0f} Hz"
                 .format(true_f, TARGETS[win]), fontsize=8.0,
                 fontweight="bold", color=INK)


def window_panel(ax, acc, itr, n_trials):
    ax.plot(WINDOWS, acc, color=BLUE, lw=1.5, marker="o", ms=3.4,
            mfc="white", mec=BLUE, mew=0.9, zorder=6, label="accuracy")
    ax.axhline(25.0, color=GREY, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(WINDOWS[-1], 27.0, "chance, {:d} targets".format(len(TARGETS)),
            fontsize=6, color=MUTED, ha="right", va="bottom", bbox=BOX,
            zorder=7)
    ax.set_xlim(0.0, WINDOWS[-1] + 0.25)
    ax.set_ylim(0.0, 108.0)
    ax.set_xlabel("epoch length (s), plus {:.1f} s to shift gaze"
                  .format(GAZE_SHIFT), fontsize=6.8)
    ax.set_ylabel("accuracy (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(WINDOWS, itr, color=RED, lw=1.4, ls=(0, (4.0, 2.0)), marker="s",
             ms=3.0, mfc="white", mec=RED, mew=0.9, zorder=6,
             label="transfer rate")
    ax2.set_ylim(0.0, max(itr) * 1.45)
    ax2.set_ylabel("transfer rate (bits/min)", fontsize=6.5, color=RED)
    ax2.tick_params(labelsize=6.0, width=0.7, length=2.5, colors=RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(RED)
    k = int(np.argmax(itr))
    ax2.annotate("peak {:.0f} bits/min at {:.2f} s\n({:.0f} % of {:d} trials "
                 "correct)".format(itr[k], WINDOWS[k], acc[k], n_trials),
                 xy=(WINDOWS[k], itr[k]), xytext=(6.0, -26.0),
                 textcoords="offset points", fontsize=6, color=RED,
                 ha="left", va="center", linespacing=1.4, bbox=BOX, zorder=9,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                 shrinkA=1, shrinkB=3))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Longer looks, better decisions, slower spelling",
                 fontsize=8.0, fontweight="bold", color=INK)


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
    rng = np.random.default_rng(5203)
    n = int(T_EPOCH * FS)

    trials = {f0: [epoch(rng, f0, n) for _ in range(N_TRIAL)]
              for f0 in TARGETS}

    spec, snr = [], []
    for f0 in TARGETS:
        f, a = spectrum(trials[f0][0][:, 1])
        spec.append((f, a))
        snr.append(peak_snr(f, a, f0))

    shown = trials[TARGETS[2]][SHOW_TRIAL]
    scores = np.array([cca(shown, reference(f0, n)) for f0 in TARGETS])

    acc, itr = [], []
    for w in WINDOWS:
        m = int(w * FS)
        hit = 0
        for i, f0 in enumerate(TARGETS):
            for x in trials[f0]:
                s = [cca(x[:m], reference(fc, m)) for fc in TARGETS]
                hit += int(np.argmax(s) == i)
        p = hit / float(N_TRIAL * len(TARGETS))
        acc.append(100.0 * p)
        itr.append(wolpaw(p, len(TARGETS), w + GAZE_SHIFT))
    acc, itr = np.array(acc), np.array(itr)

    for f0, (db, amp) in zip(TARGETS, snr):
        print("{:>5.1f} Hz: fundamental {:.2f} uV, SNR {:.1f} dB"
              .format(f0, amp, db))
    print("single trial scores:",
          " ".join("{:.3f}".format(s) for s in scores))
    for w, a, r in zip(WINDOWS, acc, itr):
        print("{:.2f} s window: {:5.1f} % correct, {:5.1f} bits/min"
              .format(w, a, r))
    assert int(np.argmax(scores)) == 2, "the shown trial is decoded correctly"
    assert acc[-1] >= acc[0], "longer epochs decode better"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_s = axes_inches(fig, MAIN)
    ax_c = axes_inches(fig, AUX_T)
    ax_w = axes_inches(fig, AUX_B)

    spectra_panel(ax_s, spec, snr)
    cca_panel(ax_c, scores, TARGETS[2])
    window_panel(ax_w, acc, itr, N_TRIAL * len(TARGETS))

    k = int(np.argmax(itr))


    panel_labels_inches(fig, [ax_s, ax_c, ax_w], dx=-0.62, dy=0.14)
    savefig(fig, "1059_bci_ssvep_frequency_id.png")


if __name__ == "__main__":
    main()
