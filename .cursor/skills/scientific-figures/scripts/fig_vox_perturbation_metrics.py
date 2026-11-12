"""Jitter, shimmer and harmonics-to-noise ratio as a two-group discriminant.

Every point on this figure is measured, not declared.  Each of the 84 synthetic
speakers is a half second of voice built from a resonated glottal pulse train
whose period and pulse amplitude are perturbed cycle by cycle and which is then
buried in a controlled amount of additive noise.  The three parameters are then
recovered from that waveform the way a clinical package recovers them: periods
and peak amplitudes come from peak picking, local jitter and local shimmer are
the mean absolute differences between consecutive cycles normalised by their
own means, and the harmonics-to-noise ratio is the normalised autocorrelation
peak at the fundamental lag turned into decibels.  The reference ellipse is the
95 % contour of the normal group's measured covariance, so the claim that the
dysphonic group falls outside normal limits is a geometric fact about the
plotted coordinates and is asserted before the figure is written.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import signal

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#8a8a8a"
HAIR = "#cccccc"

NORM = "#2a7f7a"
NORM_L = "#9ecfcb"
PATH = "#b23a48"
PATH_L = "#dfa9b0"
AMBER = "#c0603f"
DEEP = "#134a76"

SEED = 1044
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

FS = 22050.0
DUR = 0.50
N_NORM, N_PATH = 48, 36


def resonator(f, bw):
    """Two-pole vocal-tract resonance as second-order section coefficients."""
    r = np.exp(-np.pi * bw / FS)
    theta = 2.0 * np.pi * f / FS
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    b = np.array([1.0 - 2.0 * r * np.cos(theta) + r * r, 0.0, 0.0])
    return b, a


def synth_voice(rng, f0, jitter_pct, shimmer_pct, snr_db):
    """One sustained /a/: perturbed pulse train, one resonance, added noise."""
    n = int(DUR * FS)
    src = np.zeros(n)
    t0 = FS / f0
    # E|X_i - X_{i+1}| for i.i.d. Gaussian relative deviations of s.d. s is
    # 2 s / sqrt(pi), so the driving s.d. is scaled to hit the target local
    # perturbation; the value that gets plotted is still the measured one.
    s_j = jitter_pct / 100.0 * np.sqrt(np.pi) / 2.0
    s_s = shimmer_pct / 100.0 * np.sqrt(np.pi) / 2.0
    pos = 2.0 * t0
    while pos < n - 2.0 * t0:
        amp = 1.0 + rng.normal(0.0, s_s)
        i = int(round(pos))
        w = int(round(0.42 * t0))
        k = np.arange(w)
        src[i:i + w] += amp * (0.5 - 0.5 * np.cos(np.pi * k / w)) ** 2
        pos += t0 * (1.0 + rng.normal(0.0, s_j))
    b1, a1 = resonator(700.0, 90.0)
    b2, a2 = resonator(1180.0, 120.0)
    x = signal.lfilter(b1, a1, src)
    x = signal.lfilter(b2, a2, x)
    x = x - x.mean()
    x /= np.abs(x).max() + 1e-12
    noise = rng.normal(0.0, 1.0, n)
    noise = signal.lfilter(*signal.butter(2, 4800.0 / (FS / 2.0)), noise)
    g = np.sqrt(np.mean(x ** 2) / np.mean(noise ** 2)) * 10.0 ** (-snr_db / 20.0)
    return x + g * noise, t0


def measure(x):
    """Local jitter (%), local shimmer (%) and HNR (dB) from the waveform.

    The cycle marks come from the waveform low-passed just above its own
    estimated fundamental, because the raw signal rings once per formant
    period and naive peak picking would then read the ringing as jitter.
    """
    n = len(x)
    xw = x - x.mean()
    ac0 = np.correlate(xw, xw, mode="full")[n - 1:]
    lo0, hi0 = int(FS / 320.0), int(FS / 68.0)
    t0 = int(lo0 + np.argmax(ac0[lo0:hi0]))
    f0 = FS / t0

    b, a = signal.butter(3, min(1.7 * f0 / (FS / 2.0), 0.9))
    lp = signal.filtfilt(b, a, x)
    pk, _ = signal.find_peaks(lp, distance=int(0.72 * t0))
    pk = pk[(pk > t0) & (pk < n - t0)]
    # Parabolic refinement of every mark: whole-sample marks alone would put a
    # quantisation floor near 0.7 % on the jitter of a 150 Hz voice.
    y0, y1, y2 = lp[pk - 1], lp[pk], lp[pk + 1]
    pos = pk + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2 + 1e-12)
    per = np.diff(pos) / FS
    amp = np.array([np.ptp(lp[pk[i]:pk[i + 1]]) for i in range(len(pk) - 1)])
    jit = 100.0 * np.mean(np.abs(np.diff(per))) / np.mean(per)
    shim = 100.0 * np.mean(np.abs(np.diff(amp))) / np.mean(amp)

    w = int(4.0 * t0)
    win = np.hanning(w)
    acw = np.correlate(win, win, mode="full")[w - 1:]
    acw /= acw[0]
    lo, hi = int(0.72 * t0), int(1.40 * t0)
    vals = []
    for s0 in range(0, n - w, w // 2):
        fr = x[s0:s0 + w]
        fr = (fr - fr.mean()) * win
        ac = np.correlate(fr, fr, mode="full")[w - 1:]
        ac /= ac[0] + 1e-12
        r = float(np.clip((ac[lo:hi] / np.maximum(acw[lo:hi], 1e-3)).max(),
                          1e-4, 0.9995))
        vals.append(10.0 * np.log10(r / (1.0 - r)))
    return jit, shim, float(np.median(vals)), pk, float(np.mean(per))


def cohort(rng, n, f0_lo, f0_hi, jit_r, shim_r, snr_r):
    rows, waves = [], []
    for _ in range(n):
        f0 = rng.uniform(f0_lo, f0_hi)
        jt = rng.uniform(*jit_r)
        sh = rng.uniform(*shim_r)
        sn = rng.uniform(*snr_r)
        x, _ = synth_voice(rng, f0, jt, sh, sn)
        j, s, h, pk, mper = measure(x)
        rows.append((j, s, h, 1.0 / mper))
        waves.append((x, pk))
    return np.array(rows), waves


def ellipse_params(xy, chi2=5.991):
    """Centre, axes lengths and rotation of the 95 % covariance contour."""
    mu = xy.mean(axis=0)
    cov = np.cov(xy.T)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
    return mu, cov, 2.0 * np.sqrt(chi2 * ev), ang


def mahal(xy, mu, cov):
    d = xy - mu
    return np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)


def spectrum(x, f0):
    """Welch spectrum in dB plus the median harmonic-to-valley contrast.

    The contrast is read at known harmonic and half-harmonic frequencies
    rather than by peak finding: on a breathy spectrum a peak finder counts
    the noise ripple as harmonic structure and both voices come out alike.
    """
    f, p = signal.welch(x, fs=FS, nperseg=4096, noverlap=3072, window="hann")
    db = 10.0 * np.log10(p / p.max() + 1e-14)
    ks = [k for k in range(2, 12) if k * f0 < 3000.0]
    contrast = float(np.median([np.interp(k * f0, f, db)
                                - np.interp((k + 0.5) * f0, f, db)
                                for k in ks]))
    return f / 1e3, db, contrast


def main_panel(ax, nm, pt, mu, cov, axes_len, ang, out_frac, acc, w, b, ex):
    ax.add_patch(Ellipse(mu, axes_len[0], axes_len[1], angle=ang,
                         facecolor="#eef1f4", edgecolor=GREY, lw=0.9,
                         ls=(0, (4.4, 2.4)), zorder=2))
    mu2, _, ax2, an2 = ellipse_params(nm[:, :2], chi2=2.279)
    ax.add_patch(Ellipse(mu2, ax2[0], ax2[1], angle=an2, facecolor="none",
                         edgecolor=GREY, lw=0.7, ls=(0, (1.7, 1.9)), zorder=3))

    xs = np.linspace(0.0, 6.6, 60)
    ys = (-b - w[0] * xs) / w[1]
    ax.plot(xs, ys, color=DEEP, lw=1.15, ls=(0, (5.0, 2.4)), zorder=4)

    lo, hi = 4.0, 22.0
    sc = ax.scatter(nm[:, 0], nm[:, 1], s=27, c=nm[:, 2], cmap="viridis",
                    vmin=lo, vmax=hi, edgecolor=NORM, linewidth=0.7,
                    marker="o", zorder=6)
    ax.scatter(pt[:, 0], pt[:, 1], s=34, c=pt[:, 2], cmap="viridis",
               vmin=lo, vmax=hi, edgecolor=PATH, linewidth=0.9,
               marker="^", zorder=7)

    for xy, col, lab in ((ex["n"], NORM, "N-21"), (ex["p"], PATH, "D-08")):
        ax.plot([xy[0]], [xy[1]], marker="o", ms=11.0, mfc="none", mec=col,
                mew=1.4, zorder=9)
        ax.text(xy[0] + 0.13, xy[1] + 0.38, lab, fontsize=6.4, color=col,
                fontweight="bold", ha="left", va="bottom", zorder=10)

    ax.text(3.42, 6.15, "95 %% reference ellipse of the normal\ncohort "
            "(n = %d), from its own measured\ncovariance; %.0f %% of the "
            "dysphonic\npoints fall outside it"
            % (len(nm), 100.0 * out_frac), fontsize=6.4, color=SUB,
            ha="left", va="top", linespacing=1.38, zorder=11)
    ax.annotate("", xy=(1.62, 3.95), xytext=(3.34, 5.05),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7),
                zorder=5)
    ax.text(0.16, 13.4, "the linear discriminant fitted on the two\nplotted "
            "parameters separates the cohorts\nat %.1f %%" % (100.0 * acc),
            fontsize=6.4, color=DEEP, ha="left", va="top", linespacing=1.38,
            zorder=11)

    safe_colorbar(ax.figure, sc, ax,
                  label="harmonics-to-noise ratio (dB)", width=0.12,
                  pad=0.10, labelsize=6.3)

    handles = [
        Line2D([], [], color="none", marker="o", mfc="#dddddd", mec=NORM,
               mew=0.9, ms=5.4, label="normal voice, n = %d" % len(nm)),
        Line2D([], [], color="none", marker="^", mfc="#dddddd", mec=PATH,
               mew=1.0, ms=6.0, label="dysphonic voice, n = %d" % len(pt)),
        Line2D([], [], color=DEEP, lw=1.1, ls=(0, (5.0, 2.4)),
               label="LDA decision boundary"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (4.4, 2.4)),
               label="95 % / 68 % normal contours"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.0, borderaxespad=0.4,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xlim(0.0, 6.6)
    ax.set_ylim(0.0, 20.5)
    ax.set_xlabel("local jitter (%), measured cycle to cycle")
    ax.set_ylabel("local shimmer (%), measured cycle to cycle")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Three measures, one plane: 0.5 s of sustained phonation "
                 "per speaker", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def wave_panel(ax, wn, wp):
    t_end = 0.0
    for (w, col, off, tag) in ((wn, NORM, 1.45, "N-21   jitter %.2f %%, "
                                "shimmer %.1f %%"),
                               (wp, PATH, -1.45, "D-08   jitter %.2f %%, "
                                "shimmer %.1f %%")):
        x, pk, jit, shim = w
        marks = pk[2:8]
        pad = int(0.18 * (marks[1] - marks[0]))
        i0, i1 = int(marks[0]) - pad, int(marks[-1]) + pad
        seg = x[i0:i1]
        sc = np.abs(seg).max()
        t = np.arange(len(seg)) / FS * 1e3
        t_end = max(t_end, t[-1])
        ax.plot(t, seg / sc * 0.98 + off, color=col, lw=0.85, zorder=6)
        for p in marks:
            tt = (p - i0) / FS * 1e3
            ax.plot([tt, tt], [off - 1.02, off + 1.02], color=col, lw=0.55,
                    ls=(0, (1.5, 1.8)), alpha=0.55, zorder=4)
        ta, tb = (marks[2] - i0) / FS * 1e3, (marks[3] - i0) / FS * 1e3
        ax.annotate("", xy=(tb, off + 1.10), xytext=(ta, off + 1.10),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=0.8,
                                    shrinkA=0, shrinkB=0), zorder=8)
        ax.text(0.5 * (ta + tb), off + 1.16, "T", fontsize=6.0, color=col,
                ha="center", va="bottom", style="italic", zorder=9)
        ax.text(0.25, off + 1.30, tag % (jit, shim), fontsize=6.1, color=col,
                fontweight="bold", ha="left", va="bottom", zorder=9)

    ax.axhline(0.0, color=HAIR, lw=0.7, zorder=2)
    ax.text(0.25, -3.42, "dashed rules are the detected cycle marks: the "
            "period and amplitude\nsequences they cut out are what jitter and "
            "shimmer average over",
            fontsize=6, color=MUTED, ha="left", va="bottom",
            linespacing=1.34, zorder=9)
    ax.set_xlim(0.0, t_end * 1.02)
    ax.set_ylim(-3.50, 3.15)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)", fontsize=6.8)
    ax.set_ylabel("radiated pressure (a.u.)", fontsize=6.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Five cycles of each ringed exemplar", fontsize=8.0,
                 fontweight="bold", color=INK)


def spectrum_panel(ax, sn, sp, hnr_n, hnr_p):
    for (f, db, col, lab, hnr, ct) in (
            (sp[0], sp[1], PATH, "D-08", hnr_p, sp[2]),
            (sn[0], sn[1], NORM, "N-21", hnr_n, sn[2])):
        ax.plot(f, db, color=col, lw=0.85, zorder=6,
                label="%s, HNR %.1f dB, harmonic contrast %.0f dB"
                      % (lab, hnr, ct))
    ax.text(4.86, -4.0, "the dysphonic spectrum has its\ninter-harmonic "
            "valleys filled in:\nthat filling is what the HNR\nmeasures",
            fontsize=6.0, color=SUB, ha="right", va="top", linespacing=1.34,
            zorder=9)
    leg = ax.legend(loc="lower left", fontsize=6.0, frameon=True,
                    facecolor="white", edgecolor="#dddddd", framealpha=0.95,
                    handlelength=1.7, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(10)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-84.0, 6.0)
    ax.set_xlabel("frequency (kHz)", fontsize=6.8)
    ax.set_ylabel("spectral level (dB re peak)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Where the noise term shows up", fontsize=8.0,
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

    nm, wav_n = cohort(rng, N_NORM, 108.0, 228.0, (0.10, 0.42), (1.0, 2.8),
                       (20.0, 30.0))
    pt, wav_p = cohort(rng, N_PATH, 100.0, 214.0, (1.60, 4.10), (6.4, 13.6),
                       (6.0, 14.0))

    mu, cov, axlen, ang = ellipse_params(nm[:, :2])
    d2 = mahal(pt[:, :2], mu, cov)
    out_frac = float((d2 > 5.991).mean())

    xy = np.vstack([nm[:, :2], pt[:, :2]])
    lab = np.r_[np.zeros(len(nm)), np.ones(len(pt))]
    m0, m1 = nm[:, :2].mean(0), pt[:, :2].mean(0)
    sw = np.cov(nm[:, :2].T) * (len(nm) - 1) + np.cov(pt[:, :2].T) * (len(pt) - 1)
    sw /= len(xy) - 2
    w = np.linalg.solve(sw, m1 - m0)
    b = -0.5 * w @ (m0 + m1)
    acc = float(((xy @ w + b > 0).astype(float) == lab).mean())

    i_n = int(np.argmin(mahal(nm[:, :2], mu, cov)))
    i_p = int(np.argmax(d2))
    ex_n = (wav_n[i_n][0], wav_n[i_n][1], nm[i_n, 0], nm[i_n, 1])
    ex_p = (wav_p[i_p][0], wav_p[i_p][1], pt[i_p, 0], pt[i_p, 1])
    sn = spectrum(ex_n[0], nm[i_n, 3])
    sp = spectrum(ex_p[0], pt[i_p, 3])

    assert out_frac > 0.95, "the dysphonic cohort must sit outside 95 % limits"
    assert nm[:, 2].mean() > pt[:, 2].mean() + 6.0, \
        "normal voices must carry the higher harmonics-to-noise ratio"
    assert acc > 0.95, "the two clouds must be linearly separable here"
    assert len(ex_n[1]) > 20 and len(ex_p[1]) > 20, "peak picking failed"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_w = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    main_panel(ax_main, nm, pt, mu, cov, axlen, ang, out_frac, acc, w, b,
               dict(n=nm[i_n, :2], p=pt[i_p, :2]))
    wave_panel(ax_w, ex_n, ex_p)
    spectrum_panel(ax_s, sn, sp, nm[i_n, 2], pt[i_p, 2])



    panel_labels_inches(fig, [ax_main, ax_w, ax_s], dx=-0.62, dy=0.14)
    savefig(fig, "1044_vox_perturbation_metrics.png")


if __name__ == "__main__":
    main()
