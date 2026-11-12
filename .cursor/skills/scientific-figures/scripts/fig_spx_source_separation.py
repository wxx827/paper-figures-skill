"""459 - Three-speaker separation: mixture, estimates, ideal masks, BSS scores.

Each estimate is the mixture multiplied by a corrupted version of its own ideal
ratio mask, and SDR / SIR / SAR come from the standard projection decomposition
of exactly those waveforms.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from pubstyle import (set_pub_style, savefig, style_axes, get_cmap, lighten)

FS = 16000.0
FMAX = 6000.0
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
NPER, HOP = 512, 128
SRC_COLOR = ["#b23a48", "#3b5b92", "#2e7d5b"]

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

UTTS = [
    [("sil", "sil", 0.06, -1), ("sh", "f", 0.13, 0), ("i", "v", 0.16, 0),
     ("l", "v", 0.08, 1), ("o", "v", 0.17, 1), ("k", "s", 0.07, 1),
     ("s", "f", 0.14, 2), ("e", "v", 0.16, 2), ("t", "s", 0.07, 2),
     ("sil", "sil", 0.06, -1)],
    [("sil", "sil", 0.06, -1), ("m", "v", 0.10, 0), ("a", "v", 0.17, 0),
     ("d", "s", 0.06, 0), ("@", "v", 0.09, 1), ("r", "v", 0.09, 1),
     ("aa", "v", 0.19, 1), ("th", "f", 0.11, 2), ("u", "v", 0.16, 2),
     ("sil", "sil", 0.06, -1)],
    [("sil", "sil", 0.06, -1), ("f", "f", 0.12, 0), ("o", "v", 0.15, 0),
     ("n", "v", 0.09, 0), ("i", "v", 0.15, 1), ("t", "s", 0.06, 1),
     ("a", "v", 0.18, 1), ("s", "f", 0.13, 2), ("@", "v", 0.10, 2),
     ("sil", "sil", 0.06, -1)],
]
SPK = [("spk 1", 205.0, 168.0, 1.14), ("spk 2", 132.0, 102.0, 1.00),
       ("spk 3", 104.0, 84.0, 0.90)]
OFFSETS = [0.06, 0.30, 0.52]
DUR = 2.05


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


def synth(phones, seed=7, f0_hi=132.0, f0_lo=96.0, vtl=1.0):
    rng = np.random.default_rng(seed)
    segs = segment_table(phones)
    n = segs[-1][3]
    t = np.arange(n) / FS

    anch_t, anch_f, anch_b = [], [], []
    for lab, kind, i0, i1, _ in segs:
        if kind == "v":
            anch_t.append(0.5 * (i0 + i1))
            anch_f.append([f * vtl for f, _ in VOWELS[lab]])
            anch_b.append([b for _, b in VOWELS[lab]])
    anch_t = np.asarray(anch_t)
    grid = np.arange(n, dtype=float)
    ftr = np.array([np.interp(grid, anch_t, r) for r in np.asarray(anch_f).T])
    btr = np.array([np.interp(grid, anch_t, r) for r in np.asarray(anch_b).T])

    voiced = np.zeros(n, dtype=bool)
    amp = np.zeros(n)
    for lab, kind, i0, i1, _ in segs:
        if kind == "v":
            voiced[i0:i1] = True
            amp[i0:i1] = LEVEL[lab]
    amp = gaussian_filter1d(amp, 0.014 * FS)
    vgain = gaussian_filter1d(voiced.astype(float), 0.006 * FS)

    decl = f0_hi * (f0_lo / f0_hi) ** (t / t[-1])
    jit = gaussian_filter1d(rng.normal(0, 1, n), 0.030 * FS)
    f0 = decl * (1.0 + 0.016 * jit / (np.abs(jit).max() + 1e-9))

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
            th = 2 * np.pi * min(ftr[k, mid], 0.45 * FS) / FS
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
    x = voice + noise
    return x / (np.abs(x).max() * 1.03)


def stft(x):
    f, tt, Z = signal.stft(x, fs=FS, window="hann", nperseg=NPER,
                           noverlap=NPER - HOP, boundary="zeros")
    return f, tt, Z


def istft(Z, n):
    _, y = signal.istft(Z, fs=FS, window="hann", nperseg=NPER,
                        noverlap=NPER - HOP, boundary=True)
    return np.pad(y, (0, max(0, n - y.size)))[:n]


def bss_scores(refs, est, k):
    """SDR / SIR / SAR from the standard orthogonal projection decomposition."""
    S = np.asarray(refs).T
    s = S[:, k]
    tgt = (est @ s) / (s @ s + 1e-18) * s
    coef, *_ = np.linalg.lstsq(S, est, rcond=None)
    proj = S @ coef
    interf = proj - tgt
    artif = est - proj
    def db(a, b):
        return 10 * np.log10((a @ a + 1e-18) / (b @ b + 1e-18))
    return (db(tgt, interf + artif), db(tgt, interf),
            db(tgt + interf, artif))


def main():
    set_pub_style(base_font=8)
    n = int(DUR * FS)
    t = np.arange(n) / FS

    refs = []
    for i, (phones, (_, hi, lo, vtl), off) in enumerate(
            zip(UTTS, SPK, OFFSETS)):
        y = synth(phones, seed=11 + 5 * i, f0_hi=hi, f0_lo=lo, vtl=vtl)
        buf = np.zeros(n)
        i0 = int(off * FS)
        m = min(y.size, n - i0)
        buf[i0:i0 + m] = y[:m] * (0.95 - 0.10 * i)
        refs.append(buf)
    refs = np.asarray(refs)
    mix = refs.sum(axis=0)

    f, tt, Zm = stft(mix)
    Zs = np.array([stft(r)[2] for r in refs])
    P = np.abs(Zs) ** 2
    irm = P / (P.sum(axis=0, keepdims=True) + 1e-14)

    rng = np.random.default_rng(3)
    ests, masks_hat = [], []
    for k in range(3):
        err = gaussian_filter(rng.normal(0, 1, irm[k].shape), (2.5, 3.0))
        err /= np.abs(err).max() + 1e-9
        mh = np.clip(gaussian_filter(irm[k], (1.0, 1.2)) + 0.16 * err, 0.0, 1.0)
        masks_hat.append(mh)
        ests.append(istft(mh * Zm, n))

    scores = [bss_scores(refs, ests[k], k) for k in range(3)]
    for k, s in enumerate(scores):
        print(f"  src {k + 1}: SDR {s[0]:5.1f}  SIR {s[1]:5.1f}  "
              f"SAR {s[2]:5.1f} dB")

    def db_img(x):
        return 20 * np.log10(np.abs(stft(x)[2]) + 1e-8)

    spec_mix = db_img(mix)
    spec_est = [db_img(e) for e in ests]
    ref_max = max([spec_mix.max()] + [s.max() for s in spec_est])
    spec_mix -= ref_max
    spec_est = [s - ref_max for s in spec_est]
    vmin, vmax = -74.0, 0.0
    keep = f <= FMAX

    fig = plt.figure(figsize=(9.0, 6.4))
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1.0, 0.88, 0.60],
                  height_ratios=[0.98, 0.80, 0.80, 0.80], hspace=0.42,
                  wspace=0.26, left=0.075, right=0.975, top=0.880,
                  bottom=0.080)

    # ---- (a) mixture -------------------------------------------------------
    ax_m = fig.add_subplot(gs[0, 0:2])
    mesh = ax_m.pcolormesh(tt, f[keep] / 1000.0, spec_mix[keep], cmap="magma",
                           vmin=vmin, vmax=vmax, shading="gouraud",
                           rasterized=True)
    ax_m.set_ylim(0, FMAX / 1000.0)
    ax_m.set_xlim(0, DUR)
    ax_m.set_ylabel("Frequency (kHz)")
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_title("Mixture of three overlapping talkers", fontsize=8.5,
                   fontweight="bold")
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor(RULE)
    ax_m.text(-0.075, 1.06, "(a)", transform=ax_m.transAxes, ha="left",
              va="bottom", fontsize=11, fontweight="bold", color=INK)
    for k, off in enumerate(OFFSETS):
        ax_m.plot([off, off], [0, FMAX / 1000.0], color=SRC_COLOR[k], lw=0.9,
                  ls=(0, (3, 2)), alpha=0.9, zorder=5)
        ax_m.text(off + 0.012, FMAX / 1000.0 * 0.955, SPK[k][0],
                  color=SRC_COLOR[k], fontsize=6.6, va="top", ha="left",
                  fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                            edgecolor="none", alpha=0.82), zorder=6)

    # ---- colour keys -------------------------------------------------------
    ax_key = fig.add_subplot(gs[0, 2])
    ax_key.axis("off")
    c1 = ax_key.inset_axes([0.10, 0.66, 0.80, 0.085])
    cb1 = fig.colorbar(mesh, cax=c1, orientation="horizontal",
                       ticks=[-60, -40, -20, 0])
    cb1.set_label("power (dB re mixture peak)", fontsize=6.6, labelpad=2,
                  color=INK)

    # ---- (b)-(d) estimates, ideal masks, scores ----------------------------
    bar_axes = []
    for k in range(3):
        ax_s = fig.add_subplot(gs[k + 1, 0])
        ax_s.pcolormesh(tt, f[keep] / 1000.0, spec_est[k][keep], cmap="magma",
                        vmin=vmin, vmax=vmax, shading="gouraud",
                        rasterized=True)
        ax_s.set_ylim(0, FMAX / 1000.0)
        ax_s.set_xlim(0, DUR)
        ax_s.set_yticks([0, 2, 4, 6])
        ax_s.set_ylabel(SPK[k][0], rotation=0, ha="right", va="center",
                        labelpad=16, fontsize=8.2, fontweight="bold",
                        color=SRC_COLOR[k])
        for s in ax_s.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.1)
            s.set_edgecolor(SRC_COLOR[k])

        ax_k = fig.add_subplot(gs[k + 1, 1])
        immask = ax_k.pcolormesh(tt, f[keep] / 1000.0, irm[k][keep],
                                 cmap=get_cmap("cb_sequential_blue"), vmin=0,
                                 vmax=1, shading="gouraud", rasterized=True)
        ax_k.set_ylim(0, FMAX / 1000.0)
        ax_k.set_xlim(0, DUR)
        ax_k.set_yticks([0, 2, 4, 6])
        ax_k.tick_params(labelleft=False)
        for s in ax_k.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_edgecolor(RULE)
        cov = float(irm[k].mean())
        ax_k.text(0.985, 0.955, f"mean mask {cov:.2f}",
                  transform=ax_k.transAxes, ha="right", va="top",
                  fontsize=6.2, color=INK,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="none", alpha=0.85))

        ax_b = fig.add_subplot(gs[k + 1, 2])
        vals = scores[k]
        for j, (name, v) in enumerate(zip(("SDR", "SIR", "SAR"), vals)):
            ax_b.barh(2 - j, v, height=0.62,
                      color=lighten(SRC_COLOR[k], 0.16 + 0.24 * j),
                      edgecolor=SRC_COLOR[k], linewidth=0.7, zorder=3)
            ax_b.text(v + 0.4, 2 - j, f"{v:.1f}", va="center", ha="left",
                      fontsize=6.3, color=SUB, zorder=4)
        ax_b.set_yticks([2, 1, 0])
        ax_b.set_yticklabels(["SDR", "SIR", "SAR"], fontsize=6.8)
        ax_b.set_ylim(-0.65, 2.65)
        ax_b.tick_params(labelsize=6.5, length=2)
        ax_b.grid(axis="x", lw=0.4, alpha=0.5)
        style_axes(ax_b)

        if k < 2:
            ax_s.tick_params(labelbottom=False)
            ax_k.tick_params(labelbottom=False)
            ax_b.tick_params(labelbottom=False)
        else:
            ax_s.set_xlabel("Time (s)")
            ax_k.set_xlabel("Time (s)")
            ax_b.set_xlabel("Score (dB)", fontsize=7.2)
        if k == 0:
            ax_s.set_title("Separated estimate", fontsize=8.5,
                           fontweight="bold")
            ax_k.set_title("Ideal ratio mask", fontsize=8.5,
                           fontweight="bold")
            ax_b.set_title("BSS scores", fontsize=8.0, fontweight="bold")
            for ax, lab, dx in ((ax_s, "(b)", -0.075), (ax_k, "(c)", -0.055),
                                (ax_b, "(d)", -0.28)):
                ax.text(dx, 1.05, lab, transform=ax.transAxes, ha="left",
                        va="bottom", fontsize=11, fontweight="bold",
                        color=INK)
        ax_s.tick_params(labelsize=6.8)
        ax_k.tick_params(labelsize=6.8)
        bar_axes.append(ax_b)

    hi = float(np.asarray(scores).max()) * 1.20
    for ax in bar_axes:
        ax.set_xlim(0, hi)

    c2 = ax_key.inset_axes([0.10, 0.24, 0.80, 0.085])
    cb2 = fig.colorbar(immask, cax=c2, orientation="horizontal",
                       ticks=[0, 0.5, 1.0])
    cb2.set_label("ideal ratio mask (-)", fontsize=6.6, labelpad=2, color=INK)
    for cb in (cb1, cb2):
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor(RULE)
        cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.4, color=RULE)


    savefig(fig, "459_spx_source_separation.png")


if __name__ == "__main__":
    main()
