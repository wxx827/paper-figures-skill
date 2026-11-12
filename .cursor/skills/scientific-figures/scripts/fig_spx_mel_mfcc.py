"""455 - The mel-frequency feature chain, one transform per panel.

Linear spectrogram -> triangular mel filterbank -> log-mel spectrogram ->
MFCC.  Every panel keeps frequency (or its warped successor) on the vertical
axis so the reader can follow one quantity across the whole chain.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d

from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label,
                      get_cmap, get_palette)

FS = 16000.0
FMAX = 8000.0
N_MELS = 40
N_MFCC = 13
INK, RULE, MUTED = "#1a1a1a", "#333333", "#777777"

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
        "th": (1500, 6600), "h": (600, 3600)}
BURST = {"t": (2800, 6800), "k": (1500, 4200), "p": (500, 2600),
         "d": (2100, 5200), "b": (450, 2300), "g": (1300, 3400)}
LEVEL = {"a": 1.0, "aa": 1.0, "e": 0.95, "i": 0.82, "o": 0.95, "u": 0.80,
         "@": 0.55, "m": 0.45, "n": 0.45, "l": 0.62, "r": 0.66}

UTT = [
    ("sil", "sil", 0.10, -1),
    ("sh", "f", 0.140, 0), ("i", "v", 0.150, 0),
    ("h", "f", 0.055, 1), ("a", "v", 0.150, 1), ("d", "s", 0.070, 1),
    ("@", "v", 0.075, 2),
    ("d", "s", 0.060, 3), ("aa", "v", 0.180, 3), ("r", "v", 0.085, 3),
    ("k", "s", 0.080, 3),
    ("s", "f", 0.160, 4), ("u", "v", 0.165, 4), ("t", "s", 0.070, 4),
    ("sil", "sil", 0.10, -1),
]
STRESS = {4, 8, 12}


def segment_table(phones):
    segs, t0 = [], 0.0
    for lab, kind, dur, w in phones:
        segs.append((lab, kind, int(round(t0 * FS)),
                     int(round((t0 + dur) * FS)), w))
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
    rng = np.random.default_rng(seed)
    segs = segment_table(phones)
    n = segs[-1][3]
    t = np.arange(n) / FS

    anch_t, anch_f, anch_b = [], [], []
    for lab, kind, i0, i1, _ in segs:
        if kind == "v":
            anch_t.append(0.5 * (i0 + i1))
            anch_f.append([f for f, _ in VOWELS[lab]])
            anch_b.append([b for _, b in VOWELS[lab]])
    anch_t = np.asarray(anch_t)
    grid = np.arange(n, dtype=float)
    ftr = np.array([np.interp(grid, anch_t, r) for r in np.asarray(anch_f).T])
    btr = np.array([np.interp(grid, anch_t, r) for r in np.asarray(anch_b).T])

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
    jit = gaussian_filter1d(rng.normal(0, 1, n), 0.030 * FS)
    f0 = decl * (1.0 + 0.10 * accent + 0.016 * jit / (np.abs(jit).max() + 1e-9))

    phase = np.cumsum(f0) / FS
    pulse = np.zeros(n)
    pulse[np.nonzero(np.diff(np.floor(phase)))[0] + 1] = 1.0
    a1 = np.exp(-2 * np.pi * 520.0 / FS)
    src = signal.lfilter([1 - a1], [1.0, -a1], pulse)
    src = signal.lfilter([1 - a1], [1.0, -a1], src) * vgain

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
    voice = np.diff(out[:n], prepend=0.0)
    voice *= amp / (np.abs(voice).max() + 1e-12)

    noise = np.zeros(n)
    for lab, kind, i0, i1, _ in segs:
        L = i1 - i0
        if kind == "f":
            lo, hi = FRIC[lab]
            noise[i0:i1] += 0.34 * _bandnoise(rng, L, lo, hi) * _fade(L, 0.020)
        elif kind == "s":
            hold = int(0.55 * L)
            m = L - hold
            if m > 8:
                lo, hi = BURST[lab]
                noise[i0 + hold:i1] += (0.55 * _bandnoise(rng, m, lo, hi)
                                        * np.exp(-np.arange(m) / (0.010 * FS)))
    x = voice + noise + 0.0008 * rng.normal(0, 1, n)
    return t, x / (np.abs(x).max() * 1.03), segs


# ---- mel machinery ---------------------------------------------------------
def hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_bank(freqs, n_mels=N_MELS, fmin=20.0, fmax=FMAX):
    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(ctr - lo, 1e-9)
        dn = (hi - freqs) / max(hi - ctr, 1e-9)
        bank[i] = np.clip(np.minimum(up, dn), 0.0, None)
    return bank, edges[1:-1]


def main():
    set_pub_style(base_font=8)
    t, x, segs = synth(UTT)

    nper, hop = 512, 160
    f, tt, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=nper,
                                  noverlap=nper - hop, nfft=nper, mode="psd")
    keep = f <= FMAX
    f, S = f[keep], S[keep]
    lin_db = 10 * np.log10(S + 1e-14)
    lin_db -= lin_db.max()

    bank, centres = mel_bank(f)
    mel_p = bank @ S
    mel_db = 10 * np.log10(mel_p + 1e-14)
    mel_db -= mel_db.max()

    coef = dct(mel_db, type=2, axis=0, norm="ortho")[:N_MFCC + 1]
    mfcc = coef[1:]                                   # c0 = frame energy
    dbmin, dbmax = -72.0, 0.0
    mm = float(np.percentile(np.abs(mfcc), 99))

    fig = plt.figure(figsize=(10.4, 4.2))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.048],
                  width_ratios=[1.15, 0.62, 1.05, 1.05],
                  hspace=0.55, wspace=0.36,
                  left=0.055, right=0.985, top=0.820, bottom=0.150)

    # ---- (a) linear spectrogram -------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.pcolormesh(tt, f / 1000.0, lin_db, cmap="magma", vmin=dbmin,
                           vmax=dbmax, shading="gouraud", rasterized=True)
    ax_a.set_ylim(0, FMAX / 1000.0)
    ax_a.set_ylabel("Frequency (kHz)")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("Linear spectrogram", fontsize=8.5, fontweight="bold")

    # ---- (b) triangular mel filterbank ------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    cols = get_palette("cb_sequential_blue", N_MELS + 18)[18:]
    for i in range(N_MELS):
        ax_b.fill_betweenx(f / 1000.0, 0.0, bank[i], color=cols[i],
                           alpha=0.55, lw=0.0)
        ax_b.plot(bank[i], f / 1000.0, color=cols[i], lw=0.45)
    ax_b.set_ylim(0, FMAX / 1000.0)
    ax_b.set_xlim(0, 1.08)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_xlabel("Filter weight (-)")
    ax_b.tick_params(labelleft=False)
    ax_b.set_title("Mel filterbank", fontsize=8.5, fontweight="bold")
    style_axes(ax_b)

    # ---- (c) log-mel spectrogram ------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.pcolormesh(tt, np.arange(1, N_MELS + 1), mel_db, cmap="magma",
                           vmin=dbmin, vmax=dbmax, shading="gouraud",
                           rasterized=True)
    ax_c.set_ylim(1, N_MELS)
    sel = [0, 9, 19, 29, 39]
    ax_c.set_yticks([s + 1 for s in sel])
    ax_c.set_yticklabels([f"{centres[s] / 1000:.1f}" for s in sel])
    ax_c.set_ylabel("Mel band centre (kHz)")
    ax_c.set_xlabel("Time (s)")
    ax_c.set_title("Log-mel spectrogram", fontsize=8.5, fontweight="bold")

    # ---- (d) MFCC ----------------------------------------------------------
    ax_d = fig.add_subplot(gs[0, 3])
    im_d = ax_d.pcolormesh(tt, np.arange(1, N_MFCC + 1), mfcc,
                           cmap=get_cmap("cb_diverging"), vmin=-mm, vmax=mm,
                           shading="gouraud", rasterized=True)
    ax_d.set_ylim(1, N_MFCC)
    ax_d.set_yticks([1, 4, 7, 10, 13])
    ax_d.set_ylabel("Cepstral coefficient index")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("MFCC", fontsize=8.5, fontweight="bold")
    ax_d.text(0.985, 0.975, "$c_0$ dropped (frame energy)",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=6.4,
              color=INK, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                   edgecolor="#cccccc", linewidth=0.5,
                                   alpha=0.9))

    for ax in (ax_a, ax_c, ax_d):
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_edgecolor(RULE)
        ax.set_xlim(0, t[-1])

    # ---- colour bars -------------------------------------------------------
    db_ticks = [-60, -40, -20, 0]
    for col, im, lab, tk in [(0, im_a, "power (dB re max)", db_ticks),
                             (2, im_c, "power (dB re max)", db_ticks),
                             (3, im_d, "cepstral amplitude (-)", None)]:
        cax = fig.add_subplot(gs[1, col])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=tk)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor(RULE)
        cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
        cb.set_label(lab, fontsize=6.8, labelpad=2, color=INK)

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "(a) (b) (c) (d)".split()):
        add_panel_label(ax, lab, x=-0.13, y=1.03)

    # ---- transform arrows in a band above the panel titles -----------------
    steps = [(ax_a, ax_b, "mel warping"),
             (ax_b, ax_c, "band sum + log"),
             (ax_c, ax_d, "DCT-II")]
    y = ax_a.get_position().y1 + 0.078
    for left, right, lab in steps:
        b0, b1 = left.get_position(), right.get_position()
        xa, xb = b0.x1 - 0.10 * b0.width, b1.x0 + 0.10 * b1.width
        xm = 0.5 * (xa + xb)
        arr = FancyArrowPatch((xa, y), (xb, y),
                              transform=fig.transFigure, arrowstyle="-|>",
                              mutation_scale=9, lw=1.0, color=RULE,
                              shrinkA=0, shrinkB=0, zorder=5)
        fig.add_artist(arr)
        fig.text(xm, y + 0.014, lab, ha="center", va="bottom", fontsize=6.8,
                 color=INK, bbox=dict(boxstyle="round,pad=0.22",
                                      facecolor="white", edgecolor="none"))


    savefig(fig, "455_spx_mel_mfcc.png")


if __name__ == "__main__":
    main()
