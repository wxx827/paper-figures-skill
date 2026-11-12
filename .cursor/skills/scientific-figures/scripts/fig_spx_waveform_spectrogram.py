"""454 - Waveform with dual-resolution spectrograms of a synthetic utterance.

The signal is built with a source-filter model: a jittered glottal pulse train
is shaped by time-varying formant resonators, fricatives are band-passed noise.
That is what puts real harmonic striations and formant bands on the display.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

FS = 16000.0
FMAX = 5000.0
INK, RULE, MUTED = "#1a1a1a", "#333333", "#777777"
WAVE, ACCENT = "#3a3a3a", "#b23a48"

# ---- source-filter voice ---------------------------------------------------
VOWELS = {
    "a":  [(730, 90), (1150, 110), (2440, 150), (3450, 240)],
    "aa": [(680, 90), (1090, 110), (2400, 150), (3400, 240)],
    "e":  [(530, 80), (1840, 110), (2480, 150), (3500, 240)],
    "i":  [(280, 70), (2290, 110), (3010, 160), (3700, 250)],
    "o":  [(570, 80), (860, 100), (2410, 150), (3350, 240)],
    "u":  [(320, 70), (900, 100), (2240, 150), (3300, 240)],
    "@":  [(520, 90), (1500, 120), (2500, 160), (3450, 250)],
    "m":  [(280, 100), (1150, 180), (2300, 240), (3300, 320)],
    "n":  [(300, 100), (1650, 190), (2600, 250), (3400, 320)],
    "l":  [(380, 80), (1150, 130), (2800, 190), (3500, 270)],
    "r":  [(420, 90), (1120, 130), (1650, 150), (3350, 270)],
}
FRIC = {"s": (4000, 7400), "sh": (2100, 5200), "f": (1300, 7000),
        "th": (1500, 6600), "h": (600, 3600), "v": (900, 4200)}
BURST = {"t": (2800, 6800), "k": (1500, 4200), "p": (500, 2600),
         "d": (2100, 5200), "b": (450, 2300), "g": (1300, 3400)}
LEVEL = {"a": 1.0, "aa": 1.0, "e": 0.95, "i": 0.82, "o": 0.95, "u": 0.80,
         "@": 0.55, "m": 0.45, "n": 0.45, "l": 0.62, "r": 0.66}

# "she had a dark suit"  -- (label, kind, duration s, word index)
UTT = [
    ("sil", "sil", 0.12, -1),
    ("sh", "f", 0.145, 0), ("i", "v", 0.155, 0),
    ("h", "f", 0.055, 1), ("a", "v", 0.150, 1), ("d", "s", 0.070, 1),
    ("@", "v", 0.080, 2),
    ("d", "s", 0.060, 3), ("aa", "v", 0.185, 3), ("r", "v", 0.085, 3),
    ("k", "s", 0.080, 3),
    ("s", "f", 0.165, 4), ("u", "v", 0.170, 4), ("t", "s", 0.075, 4),
    ("sil", "sil", 0.13, -1),
]
WORDS = ["she", "had", "a", "dark", "suit"]
STRESS = {4, 8, 12}          # indices of accented vowels in UTT


def segment_table(phones):
    """(label, kind, i0, i1, word) sample bounds for every phone."""
    segs, t0 = [], 0.0
    for lab, kind, dur, w in phones:
        i0 = int(round(t0 * FS))
        i1 = int(round((t0 + dur) * FS))
        segs.append((lab, kind, i0, i1, w))
        t0 += dur
    return segs


def _bandnoise(rng, n, lo, hi, pad=256):
    sos = signal.butter(4, [lo / (FS / 2), min(hi, 7800) / (FS / 2)],
                        btype="band", output="sos")
    return signal.sosfiltfilt(sos, rng.normal(0, 1, n + 2 * pad))[pad:pad + n]


def _fade(n, k=0.012):
    e = np.ones(n)
    m = min(int(k * FS), max(n // 3, 1))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
    e[:m], e[-m:] = ramp, ramp[::-1]
    return e


def synth(phones, seed=7, f0_hi=132.0, f0_lo=96.0):
    """Return time, waveform, F0 track, voicing mask, segment table."""
    rng = np.random.default_rng(seed)
    segs = segment_table(phones)
    n = segs[-1][3]
    t = np.arange(n) / FS

    # -- formant tracks: anchor at every voiced phone centre, glide between --
    anch_t, anch_f, anch_b = [], [], []
    for lab, kind, i0, i1, _ in segs:
        if kind == "v":
            anch_t.append(0.5 * (i0 + i1))
            anch_f.append([f for f, _ in VOWELS[lab]])
            anch_b.append([b for _, b in VOWELS[lab]])
    anch_t = np.asarray(anch_t)
    anch_f = np.asarray(anch_f).T
    anch_b = np.asarray(anch_b).T
    grid = np.arange(n, dtype=float)
    ftr = np.array([np.interp(grid, anch_t, row) for row in anch_f])
    btr = np.array([np.interp(grid, anch_t, row) for row in anch_b])

    # -- voicing, amplitude, F0 ---------------------------------------------
    voiced = np.zeros(n, dtype=bool)
    amp = np.zeros(n)
    accent = np.zeros(n)
    for k, (lab, kind, i0, i1, _) in enumerate(segs):
        if kind == "v":
            voiced[i0:i1] = True
            amp[i0:i1] = LEVEL[lab]
            if k in STRESS:
                accent[i0:i1] = 1.0
    amp = gaussian_filter1d(amp, 0.014 * FS)
    vgain = gaussian_filter1d(voiced.astype(float), 0.006 * FS)
    accent = gaussian_filter1d(accent, 0.045 * FS)

    decl = f0_hi * (f0_lo / f0_hi) ** (t / t[-1])
    jitter = gaussian_filter1d(rng.normal(0, 1, n), 0.030 * FS)
    jitter /= (np.abs(jitter).max() + 1e-9)
    f0 = decl * (1.0 + 0.10 * accent + 0.016 * jitter)

    # -- glottal source: pulse train -> two-pole tilt ------------------------
    phase = np.cumsum(f0) / FS
    pulse = np.zeros(n)
    pulse[np.nonzero(np.diff(np.floor(phase)))[0] + 1] = 1.0
    a1 = np.exp(-2 * np.pi * 520.0 / FS)
    src = signal.lfilter([1 - a1], [1.0, -a1], pulse)
    src = signal.lfilter([1 - a1], [1.0, -a1], src) * vgain

    # -- time-varying formant filter by overlap-add --------------------------
    win, hop, pre = 256, 128, 768
    w = np.hanning(win + 1)[:win]
    out = np.zeros(n + win)
    for start in range(0, n, hop):
        c0 = max(0, start - pre)
        seg = src[c0:start + win]
        if seg.size == 0:
            continue
        mid = min(start + win // 2, n - 1)
        y = seg
        for k in range(ftr.shape[0]):
            r = np.exp(-np.pi * btr[k, mid] / FS)
            th = 2 * np.pi * ftr[k, mid] / FS
            y = signal.lfilter([1 - 2 * r * np.cos(th) + r * r],
                               [1.0, -2 * r * np.cos(th), r * r], y)
        y = y[start - c0:]
        y = np.pad(y, (0, max(0, win - y.size)))[:win]
        out[start:start + win] += y * w
    voice = np.diff(out[:n], prepend=0.0)          # lip radiation, +6 dB/oct
    voice *= amp / (np.abs(voice).max() + 1e-12)

    # -- obstruent noise ------------------------------------------------------
    noise = np.zeros(n)
    for lab, kind, i0, i1, _ in segs:
        L = i1 - i0
        if kind == "f":
            lo, hi = FRIC[lab]
            noise[i0:i1] += 0.34 * _bandnoise(rng, L, lo, hi) * _fade(L, 0.020)
        elif kind == "s":
            hold = int(0.55 * L)
            lo, hi = BURST[lab]
            m = L - hold
            if m > 8:
                env = np.exp(-np.arange(m) / (0.010 * FS))
                noise[i0 + hold:i1] += (0.55 * _bandnoise(rng, m, lo, hi)
                                        * env)
    x = voice + noise + 0.0008 * rng.normal(0, 1, n)
    x /= np.abs(x).max() * 1.03
    return t, x, f0, voiced, segs


# ---- analysis --------------------------------------------------------------
def spec_db(x, nperseg, nfft, hop):
    f, tt, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=nperseg,
                                  noverlap=nperseg - hop, nfft=nfft,
                                  mode="psd")
    keep = f <= FMAX
    db = 10 * np.log10(S[keep] + 1e-14)
    return f[keep], tt, db - db.max()


def formant_peaks(spec, f, k=3):
    """Frequencies of the k strongest broad peaks below 3.6 kHz."""
    sm = gaussian_filter1d(spec, 3.0)
    idx, props = signal.find_peaks(sm, distance=max(1, int(300 / (f[1] - f[0]))),
                                   prominence=1.0)
    idx = idx[f[idx] < 3600]
    if idx.size == 0:
        return np.array([])
    order = np.argsort(props["prominences"][:idx.size])[::-1]
    sel = np.sort(idx[order[:k]])
    return f[sel]


def main():
    set_pub_style(base_font=8)
    t, x, f0, voiced, segs = synth(UTT)

    f_w, t_w, db_w = spec_db(x, 128, 512, 24)        # 8 ms  -> formants
    f_n, t_n, db_n = spec_db(x, 768, 2048, 64)       # 48 ms -> harmonics
    vmin, vmax = -72.0, 0.0

    # the long stressed /aa/ of "dark" carries the annotations
    lab, kind, i0, i1, _ = segs[8]
    c0, c1 = i0 + int(0.25 * (i1 - i0)), i1 - int(0.15 * (i1 - i0))
    seg = x[c0:c1]
    fw = (t_w >= c0 / FS) & (t_w <= c1 / FS)
    fpk = formant_peaks(db_w[:, fw].mean(axis=1), f_w, 3)

    ac = np.correlate(seg - seg.mean(), seg - seg.mean(), "full")[seg.size - 1:]
    lo, hi = int(FS / 320), int(FS / 70)
    f0_meas = FS / (lo + int(np.argmax(ac[lo:hi])))

    fig = plt.figure(figsize=(7.6, 5.4))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.62, 1.30, 1.30],
                  width_ratios=[3.30, 0.86, 0.075], hspace=0.16, wspace=0.06,
                  left=0.085, right=0.945, top=0.905, bottom=0.135)

    # ---- (a) waveform + energy envelope ------------------------------------
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.plot(t, x, color=WAVE, lw=0.32)
    hop_e = 64
    env = np.sqrt(gaussian_filter1d(x ** 2, 0.012 * FS))
    env /= env.max()
    ax_w.plot(t[::hop_e], env[::hop_e], color=ACCENT, lw=1.1, zorder=4)
    ax_w.plot(t[::hop_e], -env[::hop_e], color=ACCENT, lw=1.1, zorder=4)
    ax_w.set_xlim(0, t[-1])
    ax_w.set_ylim(-1.12, 1.12)
    ax_w.set_yticks([-1, 0, 1])
    ax_w.set_ylabel("Amplitude\n(norm.)", linespacing=1.15)
    ax_w.tick_params(labelbottom=False)
    style_axes(ax_w)
    add_panel_label(ax_w, "(a)", x=-0.075, y=1.08)
    ax_w.text(0.998, 0.96, "RMS envelope, 12 ms window",
              transform=ax_w.transAxes, ha="right", va="top",
              fontsize=6.5, color=ACCENT)

    stroke = [pe.withStroke(linewidth=1.6, foreground="#00000099")]

    # ---- (b) wideband spectrogram ------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_w)
    ax_b.pcolormesh(t_w, f_w / 1000.0, db_w, cmap="magma", vmin=vmin,
                    vmax=vmax, shading="gouraud", rasterized=True)
    ax_b.set_ylim(0, FMAX / 1000.0)
    ax_b.set_ylabel("Frequency (kHz)")
    ax_b.tick_params(labelbottom=False)
    for s in ax_b.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor(RULE)
    add_panel_label(ax_b, "(b)", x=-0.075, y=1.02)
    ax_b.text(0.010, 0.955, "wideband  |  8 ms Hann, $\\Delta f$ = 200 Hz",
              transform=ax_b.transAxes, ha="left", va="top", color="white",
              fontsize=6.6, path_effects=stroke, zorder=7)
    tc = 0.5 * (c0 + c1) / FS
    for j, fp in enumerate(fpk):
        ax_b.annotate(f"$F_{j + 1}$ = {fp:.0f} Hz",
                      xy=(tc, fp / 1000.0), xytext=(tc + 0.20, fp / 1000.0),
                      color="white", fontsize=6.6, va="center",
                      path_effects=stroke, zorder=8,
                      arrowprops=dict(arrowstyle="-", color="white", lw=0.7,
                                      shrinkA=1.0, shrinkB=1.0))
        ax_b.plot([tc], [fp / 1000.0], marker="o", ms=2.6, mfc="white",
                  mec="#1a1a1a", mew=0.5, zorder=9)

    # ---- (c) narrowband spectrogram ----------------------------------------
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_w)
    mesh = ax_c.pcolormesh(t_n, f_n / 1000.0, db_n, cmap="magma", vmin=vmin,
                           vmax=vmax, shading="gouraud", rasterized=True)
    ax_c.set_ylim(0, FMAX / 1000.0)
    ax_c.set_ylabel("Frequency (kHz)")
    ax_c.set_xlabel("Time (s)")
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor(RULE)
    add_panel_label(ax_c, "(c)", x=-0.075, y=1.02)
    ax_c.text(0.010, 0.955, "narrowband  |  48 ms Hann, $\\Delta f$ = 33 Hz",
              transform=ax_c.transAxes, ha="left", va="top", color="white",
              fontsize=6.6, path_effects=stroke, zorder=7)
    for m in (1, 2, 3, 4, 5, 6):
        ax_c.plot([tc + 0.055], [m * f0_meas / 1000.0], marker="_", ms=4.0,
                  color="white", mew=0.9, zorder=8)
    ax_c.annotate(f"harmonic spacing = $F_0$ = {f0_meas:.0f} Hz",
                  xy=(tc + 0.055, 3.5 * f0_meas / 1000.0),
                  xytext=(tc + 0.19, 1.55), color="white", fontsize=6.6,
                  va="center", path_effects=stroke, zorder=8,
                  arrowprops=dict(arrowstyle="-", color="white", lw=0.7))

    # ---- word / phone ruler on the time axis -------------------------------
    for lab, kind, i0, i1, w in segs:
        if kind == "sil":
            continue
        ax_c.axvline(i0 / FS, color="white", lw=0.4, alpha=0.30, zorder=5)
        ax_b.axvline(i0 / FS, color="white", lw=0.4, alpha=0.30, zorder=5)
    for wi, word in enumerate(WORDS):
        ii = [s for s in segs if s[4] == wi]
        xm = 0.5 * (ii[0][2] + ii[-1][3]) / FS
        ax_c.text(xm, -0.235, word, transform=ax_c.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.8, color=INK)

    # ---- (d) mean voiced spectrum (subordinate) ----------------------------
    ax_p = fig.add_subplot(gs[1:, 1])
    vt = np.interp(t_n, t, voiced.astype(float)) > 0.6
    prof = db_n[:, vt].mean(axis=1)
    ax_p.plot(prof, f_n / 1000.0, color=WAVE, lw=1.0)
    ax_p.fill_betweenx(f_n / 1000.0, prof.min() - 3, prof, color="#cfe1ee",
                       alpha=0.55, lw=0)
    for j, fp in enumerate(fpk):
        yv = np.interp(fp, f_n, prof)
        ax_p.plot([yv], [fp / 1000.0], marker="o", ms=3.0, mfc="white",
                  mec=ACCENT, mew=0.9, zorder=5)
        ax_p.text(yv + 1.4, fp / 1000.0, f"$F_{j + 1}$", fontsize=6.5,
                  color=ACCENT, va="center", ha="left")
    ax_p.set_ylim(0, FMAX / 1000.0)
    ax_p.set_xlim(prof.min() - 3, prof.max() + 9)
    ax_p.tick_params(labelleft=False, labelsize=6.5)
    ax_p.set_xlabel("Mean level\n(dB re max)", fontsize=7, labelpad=2,
                    linespacing=1.15)
    ax_p.grid(axis="x", lw=0.4, alpha=0.5)
    style_axes(ax_p)
    add_panel_label(ax_p, "(d)", x=-0.16, y=1.02)
    ax_p.text(0.96, 0.02, "voiced frames only", transform=ax_p.transAxes,
              ha="right", va="bottom", fontsize=6.2, color=MUTED, rotation=90)

    cax = fig.add_subplot(gs[1:, 2])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color=RULE)
    cb.set_label("power (dB re max)", fontsize=7, labelpad=4, color=INK)


    savefig(fig, "454_spx_waveform_spectrogram.png")


if __name__ == "__main__":
    main()
