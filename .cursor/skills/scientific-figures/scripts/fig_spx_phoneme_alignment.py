"""456 - Forced alignment: spectrogram over a phone tier over a word tier.

All three time axes are the same axis object range, so a boundary drawn in one
tier lands on exactly the same pixel column in the others.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes, lighten

FS = 16000.0
FMAX = 5000.0
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
ACCENT = "#b23a48"
CAT_COLOR = {"vowel": "#3b5b92", "sonorant": "#2e7d5b",
             "fricative": "#c9a227", "stop": "#8e6bbf", "silence": "#b3b3b3"}

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
SONORANT = {"m", "n", "l", "r"}

IPA = {"sh": "\u0283", "i": "i\u02d0", "h": "h", "a": "\u00e6", "d": "d",
       "@": "\u0259", "aa": "\u0251\u02d0", "r": "\u0279", "k": "k",
       "s": "s", "u": "u\u02d0", "t": "t", "sil": "sil"}

UTT = [
    ("sil", "sil", 0.11, -1),
    ("sh", "f", 0.140, 0), ("i", "v", 0.152, 0),
    ("h", "f", 0.058, 1), ("a", "v", 0.148, 1), ("d", "s", 0.072, 1),
    ("@", "v", 0.078, 2),
    ("d", "s", 0.062, 3), ("aa", "v", 0.182, 3), ("r", "v", 0.088, 3),
    ("k", "s", 0.082, 3),
    ("s", "f", 0.162, 4), ("u", "v", 0.168, 4), ("t", "s", 0.074, 4),
    ("sil", "sil", 0.11, -1),
]
WORDS = ["she", "had", "a", "dark", "suit"]
STRESS = {4, 8, 12}


def category(lab, kind):
    if kind == "sil":
        return "silence"
    if kind == "f":
        return "fricative"
    if kind == "s":
        return "stop"
    return "sonorant" if lab in SONORANT else "vowel"


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


def tier_axes(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return ax


def left_letter(fig, ax, text, dx=-0.74, dy=None):
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    if dy is None:
        y, va = bb.y0 + bb.height / 2.0, "center"
    else:
        y, va = bb.y1 + dy / fh, "bottom"
    return fig.text(bb.x0 + dx / fw, y, text, fontsize=11, fontweight="bold",
                    color=INK, ha="left", va=va)


def main():
    set_pub_style(base_font=8)
    t, x, segs = synth(UTT)
    T = t[-1]

    nper, hop = 384, 32
    f, tt, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=nper,
                                  noverlap=nper - hop, nfft=1024, mode="psd")
    keep = f <= FMAX
    db = 10 * np.log10(S[keep] + 1e-14)
    db -= db.max()

    fig = plt.figure(figsize=(8.8, 5.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.185, 0.150],
                  width_ratios=[1.0, 0.40], hspace=0.10, wspace=0.20,
                  left=0.105, right=0.975, top=0.855, bottom=0.115)

    # ---- (a) spectrogram ---------------------------------------------------
    ax_s = fig.add_subplot(gs[0, 0])
    mesh = ax_s.pcolormesh(tt, f[keep] / 1000.0, db, cmap="magma", vmin=-70,
                           vmax=0, shading="gouraud", rasterized=True)
    ax_s.set_ylim(0, FMAX / 1000.0)
    ax_s.set_xlim(0, T)
    ax_s.set_ylabel("Frequency (kHz)")
    ax_s.tick_params(labelbottom=False)
    for s in ax_s.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor(RULE)
    # inset cax: a stolen-space colorbar would narrow (a) alone and break the
    # column-for-column alignment with the tiers below
    cax = ax_s.inset_axes([1.010, 0.0, 0.014, 1.0])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color=RULE)
    cb.set_label("power (dB re max)", fontsize=7, color=INK)

    # ---- (b) phone tier, (c) word tier -------------------------------------
    ax_p = tier_axes(fig.add_subplot(gs[1, 0], sharex=ax_s))
    ax_w = tier_axes(fig.add_subplot(gs[2, 0], sharex=ax_s))
    ax_p.tick_params(labelbottom=False)
    ax_w.set_xlabel("Time (s)")

    halo = [pe.withStroke(linewidth=1.9, foreground="white")]
    word_edges = []
    for wi in range(len(WORDS)):
        ii = [s for s in segs if s[4] == wi]
        word_edges += [ii[0][2] / FS, ii[-1][3] / FS]
    word_edges = sorted(set(word_edges))

    for lab, kind, i0, i1, wi in segs:
        t0, t1 = i0 / FS, i1 / FS
        cat = category(lab, kind)
        col = CAT_COLOR[cat]
        ax_p.add_patch(FancyBboxPatch(
            (t0 + 0.002, 0.17), max(t1 - t0 - 0.004, 0.004), 0.66,
            boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=0.06,
            facecolor=lighten(col, 0.62), edgecolor=col, linewidth=0.8,
            zorder=3))
        ax_p.text(0.5 * (t0 + t1), 0.50, IPA[lab], ha="center", va="center",
                  fontsize=7.6, color=INK if cat != "silence" else MUTED,
                  zorder=5)
        ax_p.text(0.5 * (t0 + t1), 0.03, f"{1000 * (t1 - t0):.0f}",
                  ha="center", va="bottom", fontsize=6.0, color=MUTED,
                  zorder=5)
        # phone boundary, drawn identically on both upper tiers
        for ax in (ax_s, ax_p):
            ax.axvline(t0, color=ACCENT, lw=0.65, ls=(0, (2.6, 1.8)),
                       alpha=0.92, zorder=6, path_effects=halo)

    for wi, word in enumerate(WORDS):
        ii = [s for s in segs if s[4] == wi]
        t0, t1 = ii[0][2] / FS, ii[-1][3] / FS
        ax_w.add_patch(FancyBboxPatch(
            (t0 + 0.002, 0.20), max(t1 - t0 - 0.004, 0.004), 0.62,
            boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=0.06,
            facecolor="#f0f0f0", edgecolor="#9a9a9a", linewidth=0.8, zorder=3))
        ax_w.text(0.5 * (t0 + t1), 0.51, word, ha="center", va="center",
                  fontsize=7.8, color=INK, zorder=5)

    for te in word_edges:
        for ax in (ax_s, ax_p, ax_w):
            ax.axvline(te, color=ACCENT, lw=1.15, alpha=0.95, zorder=7,
                       path_effects=halo)

    ax_p.text(-0.006, 0.5, "phone", transform=ax_p.get_yaxis_transform(),
              ha="right", va="center", fontsize=7.2, color=SUB)
    ax_w.text(-0.006, 0.5, "word", transform=ax_w.get_yaxis_transform(),
              ha="right", va="center", fontsize=7.2, color=SUB)

    # ---- (d) phone duration distribution (subordinate) ---------------------
    ax_d = fig.add_subplot(gs[:, 1])
    rows = [(lab, kind, (i1 - i0) / FS * 1000.0)
            for lab, kind, i0, i1, _ in segs if kind != "sil"]
    order = np.argsort([r[2] for r in rows])
    ypos = np.arange(len(rows))
    for y, k in zip(ypos, order):
        lab, kind, dur = rows[k]
        col = CAT_COLOR[category(lab, kind)]
        ax_d.barh(y, dur, height=0.68, color=lighten(col, 0.35),
                  edgecolor=col, linewidth=0.7, zorder=3)
        ax_d.text(dur + 3, y, f"{dur:.0f}", va="center", ha="left",
                  fontsize=6.2, color=MUTED, zorder=4)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([IPA[rows[k][0]] for k in order], fontsize=7.4)
    ax_d.set_ylim(-0.7, len(rows) - 0.3)
    ax_d.set_xlim(0, max(r[2] for r in rows) * 1.26)
    ax_d.set_xlabel("Phone duration (ms)", fontsize=7.5)
    ax_d.tick_params(labelsize=6.5)
    ax_d.grid(axis="x", lw=0.4, alpha=0.55)
    style_axes(ax_d)
    med = float(np.median([r[2] for r in rows]))
    ax_d.axvline(med, color="#808080", lw=0.9, ls="--", zorder=2)
    ax_d.text(med + 3, -0.55, f"median {med:.0f} ms", fontsize=6.3,
              color=SUB, va="bottom", ha="left")

    left_letter(fig, ax_s, "(a)", dy=0.06)
    left_letter(fig, ax_p, "(b)")
    left_letter(fig, ax_w, "(c)")
    left_letter(fig, ax_d, "(d)", dx=-0.32, dy=0.06)

    handles = [Patch(facecolor=lighten(c, 0.45), edgecolor=c, linewidth=0.7,
                     label=k) for k, c in CAT_COLOR.items()]

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).

    savefig(fig, "456_spx_phoneme_alignment.png")


if __name__ == "__main__":
    main()
