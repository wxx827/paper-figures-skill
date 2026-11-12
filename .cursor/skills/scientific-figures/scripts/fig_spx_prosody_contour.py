"""460 - Prosody of one sentence: F0, energy and syllable rate on one clock.

The F0 track is *measured back* from the synthesised waveform by short-time
autocorrelation, so the unvoiced gaps and the octave-safe voicing decision are
properties of the signal rather than of the script that made it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.ndimage import binary_opening, gaussian_filter1d, median_filter

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

FS = 16000.0
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
WAVE, ACCENT = "#3a3a3a", "#b23a48"
BLUE, GREEN, GOLD = "#3b5b92", "#2e7d5b", "#c9a227"

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

# (label, kind, duration, word index, is-syllable-nucleus)
UTT = [
    ("sil", "sil", 0.12, -1, 0),
    ("th", "f", 0.070, 0, 0), ("@", "v", 0.075, 0, 1),
    ("l", "v", 0.075, 1, 0), ("o", "v", 0.175, 1, 1), ("n", "v", 0.080, 1, 0),
    ("m", "v", 0.080, 2, 0), ("a", "v", 0.160, 2, 1), ("n", "v", 0.085, 2, 0),
    ("s", "f", 0.140, 3, 0), ("o", "v", 0.185, 3, 1), ("l", "v", 0.075, 3, 0),
    ("d", "s", 0.075, 3, 0),
    ("sil", "sil", 0.185, -1, 0),
    ("th", "f", 0.065, 4, 0), ("@", "v", 0.070, 4, 1),
    ("r", "v", 0.075, 5, 0), ("e", "v", 0.155, 5, 1), ("d", "s", 0.070, 5, 0),
    ("s", "f", 0.135, 6, 0), ("i", "v", 0.150, 6, 1), ("l", "v", 0.070, 6, 0),
    ("k", "s", 0.075, 6, 0),
    ("s", "f", 0.145, 7, 0), ("u", "v", 0.185, 7, 1), ("t", "s", 0.080, 7, 0),
    ("sil", "sil", 0.14, -1, 0),
]
WORDS = ["the", "lone", "man", "sold", "the", "red", "silk", "suit"]
ACCENTED = {1, 3, 5, 7}          # word indices carrying a pitch accent
BREAK_AFTER = 3                  # phrase boundary after this word


def segment_table(phones):
    segs, t0 = [], 0.0
    for lab, kind, dur, w, nuc in phones:
        segs.append((lab, kind, int(round(t0 * FS)),
                     int(round((t0 + dur) * FS)), w, nuc))
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


def build_f0(t, segs, brk_t, end_t):
    """Two declination phrases, accent bumps, continuation rise, final fall."""
    st = np.zeros_like(t)
    base = np.where(t < brk_t,
                    148.0 * (0.80 ** (t / max(brk_t, 1e-6))),
                    138.0 * (0.72 ** ((t - brk_t) / max(end_t - brk_t, 1e-6))))
    for lab, kind, i0, i1, w, nuc in segs:
        if nuc and w in ACCENTED:
            c = 0.5 * (i0 + i1) / FS
            st += 3.4 * np.exp(-0.5 * ((t - c) / 0.075) ** 2)
    st += 4.2 * np.exp(-0.5 * ((t - (brk_t - 0.055)) / 0.075) ** 2)
    st -= 3.0 / (1.0 + np.exp(-(t - (end_t - 0.16)) / 0.045))
    return base * 2.0 ** (st / 12.0)


def synth(phones, seed=5):
    rng = np.random.default_rng(seed)
    segs = segment_table(phones)
    n = segs[-1][3]
    t = np.arange(n) / FS
    brk = [s for s in segs if s[4] == BREAK_AFTER][-1][3] / FS

    anch_t, anch_f, anch_b = [], [], []
    for lab, kind, i0, i1, _, _ in segs:
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
    for lab, kind, i0, i1, w, nuc in segs:
        if kind == "v":
            voiced[i0:i1] = True
            amp[i0:i1] = LEVEL[lab] * (1.18 if (nuc and w in ACCENTED)
                                       else 1.0)
    amp = gaussian_filter1d(amp, 0.014 * FS)
    vgain = gaussian_filter1d(voiced.astype(float), 0.006 * FS)

    f0 = build_f0(t, segs, brk, t[-1])
    jit = gaussian_filter1d(rng.normal(0, 1, n), 0.030 * FS)
    f0 = f0 * (1.0 + 0.014 * jit / (np.abs(jit).max() + 1e-9))

    phase = np.cumsum(f0) / FS
    pulse = np.zeros(n)
    pulse[np.nonzero(np.diff(np.floor(phase)))[0] + 1] = 1.0
    a1 = np.exp(-2 * np.pi * 520.0 / FS)
    src = signal.lfilter([1 - a1], [1.0, -a1], pulse)
    src = signal.lfilter([1 - a1], [1.0, -a1], src) * vgain

    win, hop, pre = 256, 128, 768
    w_ = np.hanning(win + 1)[:win]
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
        out[start:start + win] += y * w_
    voice = np.diff(out[:n], prepend=0.0)
    voice *= amp / (np.abs(voice).max() + 1e-12)

    noise = np.zeros(n)
    for lab, kind, i0, i1, _, _ in segs:
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
    x = voice + noise + 0.0006 * rng.normal(0, 1, n)
    return t, x / (np.abs(x).max() * 1.03), segs, brk


def track_f0(x, win_ms=40.0, hop_ms=8.0, fmin=70.0, fmax=260.0):
    """Short-time autocorrelation pitch tracker with a voicing decision."""
    w = int(win_ms * FS / 1000)
    h = int(hop_ms * FS / 1000)
    starts = np.arange(0, x.size - w, h)
    tc = (starts + w / 2) / FS
    f0 = np.full(starts.size, np.nan)
    stren = np.zeros(starts.size)
    rms = np.zeros(starts.size)
    lo, hi = int(FS / fmax), int(FS / fmin)
    for i, s in enumerate(starts):
        seg = x[s:s + w] * np.hanning(w)
        seg = seg - seg.mean()
        e = np.sqrt(np.mean(seg ** 2))
        rms[i] = e
        if e < 1e-4:
            continue
        ac = np.correlate(seg, seg, "full")[w - 1:]
        ac /= ac[0] + 1e-18
        k = lo + int(np.argmax(ac[lo:hi]))
        stren[i] = ac[k]
        f0[i] = FS / k
    thr = 0.34
    voiced = (stren > thr) & (rms > 0.06 * rms.max())
    # a single voiced frame in a sea of unvoiced ones is a tracker slip, not
    # phonation: require three in a row, then median-smooth the survivors
    voiced = binary_opening(voiced, structure=np.ones(3), border_value=0)
    f0[~voiced] = np.nan
    filled = np.where(voiced, f0, np.nan)
    sm = median_filter(np.nan_to_num(filled, nan=0.0), size=5, mode="nearest")
    f0 = np.where(voiced, np.where(sm > 0, sm, filled), np.nan)
    return tc, f0, voiced, rms, thr


def main():
    set_pub_style(base_font=8)
    t, x, segs, brk = synth(UTT)
    T = t[-1]

    tc, f0, voiced, rms, vthr = track_f0(x)
    med = float(np.nanmedian(f0))
    db = 20 * np.log10(rms + 1e-9)
    db -= db.max()

    nuclei = np.array([0.5 * (s[2] + s[3]) / FS for s in segs if s[5]])
    win_s = 1.0
    grid = np.linspace(0, T, 400)
    # normalise by the part of the window that actually lies inside the
    # utterance, otherwise the first and last second look artificially slow
    span = np.minimum(grid + win_s / 2, T) - np.maximum(grid - win_s / 2, 0.0)
    rate = np.array([np.sum(np.abs(nuclei - g) <= win_s / 2) for g in grid])
    rate = gaussian_filter1d(rate / span, 6.0)

    fig = plt.figure(figsize=(7.8, 5.3))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.25, 0.78, 0.72],
                  hspace=0.16, left=0.100, right=0.905, top=0.855,
                  bottom=0.145)

    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax in axs:
        ax.set_xlim(0, T)
        style_axes(ax)
        ax.grid(axis="y", lw=0.4, alpha=0.45)

    # ---- (a) F0 ------------------------------------------------------------
    ax = axs[0]
    seg_start = 0
    runs = []
    for i in range(1, voiced.size + 1):
        if i == voiced.size or voiced[i] != voiced[i - 1]:
            if voiced[i - 1]:
                runs.append((seg_start, i))
            seg_start = i
    for a, b in runs:
        ax.plot(tc[a:b], f0[a:b], color=BLUE, lw=1.7, solid_capstyle="round",
                zorder=4)
    for i, (a, b) in enumerate(runs[:-1]):
        c, d = runs[i + 1]
        if tc[c] - tc[b - 1] < 0.26:          # bridge short unvoiced gaps only
            ax.plot([tc[b - 1], tc[c]], [f0[b - 1], f0[c]], color="#b3b3b3",
                    lw=0.7, ls=(0, (1.6, 1.8)), zorder=3)
    ax.axhline(med, color="#808080", lw=0.8, ls="--", zorder=2)
    ax.text(0.012, med + 1.5, f"median {med:.0f} Hz", ha="left", va="bottom",
            fontsize=6.4, color=SUB)
    ax.set_ylabel("$F_0$ (Hz)")
    ax.set_ylim(np.nanmin(f0) - 18, np.nanmax(f0) + 26)
    ax.tick_params(labelbottom=False)
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: 12 * np.log2(np.maximum(v, 1e-6) / med),
                            lambda s: med * 2 ** (s / 12)))
    sec.set_ylabel("semitones re median", fontsize=7.5)
    sec.tick_params(labelsize=6.8)
    add_panel_label(ax, "(a)", x=-0.075, y=1.05)

    for w in ACCENTED:
        ii = [s for s in segs if s[4] == w and s[5]]
        c = 0.5 * (ii[0][2] + ii[0][3]) / FS
        near = np.where(np.abs(tc - c) < 0.09, f0, np.nan)
        if not np.any(np.isfinite(near)):
            continue
        pk = float(np.nanmax(near))
        ax.plot([c], [pk + 7], marker="v", ms=4.4, color=ACCENT, zorder=6)
        ax.text(c, pk + 11, "\u02c8", ha="center", va="bottom", fontsize=8.5,
                color=ACCENT, fontweight="bold", zorder=6)
    ax.text(0.012, 0.955, "voiced = solid, unvoiced = gap; \u02c8 marks a "
            "pitch accent", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color=SUB)

    # ---- (b) energy --------------------------------------------------------
    ax = axs[1]
    ax.fill_between(tc, -62, db, color="#cfe1ee", alpha=0.85, lw=0, zorder=2)
    ax.plot(tc, db, color=WAVE, lw=1.1, zorder=3)
    ax.set_ylabel("Energy\n(dB re max)", linespacing=1.15)
    ax.set_ylim(-62, 4)
    ax.set_yticks([-60, -40, -20, 0])
    ax.tick_params(labelbottom=False)
    add_panel_label(ax, "(b)", x=-0.075, y=1.05)

    # ---- (c) syllable rate -------------------------------------------------
    ax = axs[2]
    ax.plot(grid, rate, color=GREEN, lw=1.6, zorder=4)
    ax.fill_between(grid, 0, rate, color=GREEN, alpha=0.16, lw=0, zorder=2)
    for c in nuclei:
        ax.plot([c, c], [0.06, 0.55], color=GOLD, lw=1.2, zorder=3)
    mean_rate = len(nuclei) / T
    ax.axhline(mean_rate, color="#808080", lw=0.8, ls="--", zorder=2)
    ax.text(0.988, 0.93, f"sentence mean {mean_rate:.2f} syl s$^{{-1}}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=SUB)
    ax.set_ylabel("Speaking rate\n(syllables s$^{-1}$)", linespacing=1.15)
    ax.set_ylim(0, max(rate.max(), mean_rate) * 1.35)
    ax.set_xlabel("Time (s)", labelpad=20)
    add_panel_label(ax, "(c)", x=-0.075, y=1.05)
    ax.text(0.988, 0.06, f"{len(nuclei)} syllable nuclei (gold ticks), "
            f"{win_s:.0f} s sliding window", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, color=SUB)

    # ---- shared overlays ---------------------------------------------------
    for wi, word in enumerate(WORDS):
        ii = [s for s in segs if s[4] == wi]
        t0, t1 = ii[0][2] / FS, ii[-1][3] / FS
        for ax in axs:
            ax.axvline(t0, color="#d5d5d5", lw=0.6, zorder=1)
        axs[2].text(0.5 * (t0 + t1), -0.30, word,
                    transform=axs[2].get_xaxis_transform(), ha="center",
                    va="top", fontsize=7.2,
                    color=ACCENT if wi in ACCENTED else INK,
                    fontweight="bold" if wi in ACCENTED else "normal")
    for ax in axs:
        ax.axvline(brk, color=ACCENT, lw=1.2, ls=(0, (4, 2)), zorder=5,
                   alpha=0.9)
    axs[0].text(brk - 0.012, 0.06, "phrase boundary  ",
                transform=axs[0].get_xaxis_transform(), ha="right",
                va="bottom", fontsize=6.6, color=ACCENT, fontweight="bold")


    savefig(fig, "460_spx_prosody_contour.png")


if __name__ == "__main__":
    main()
