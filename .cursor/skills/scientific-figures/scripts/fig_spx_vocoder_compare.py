"""463 - Five vocoders on one utterance: output spectrograms and error maps.

Top row: the five outputs, all on one shared dB scale, so a column that looks
brighter really is brighter.  Bottom row: each output minus the ground truth on
a second shared scale, symmetric about zero because a signed error needs a
signed colour map - putting a signed difference on the magnitude scale would
make -12 dB and +12 dB the same colour.  The Griffin-Lim column is a real
32-iteration reconstruction from the ground-truth magnitude, so its phase
artefacts are earned rather than drawn, and every printed distortion number is
computed from the error map above it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import shared_norm, safe_colorbar

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
OURS = "#b23a48"

FS = 16000.0
NPER, HOP, NFFT = 512, 128, 1024
FMAX = 8000.0
GL_ITERS = 10
FLOOR = -60.0                # dB re peak; below this the error is undefined

# name, MOS, MOS 95 % CI, real-time factor (compute time / audio duration)
SYSTEMS = [
    ("Griffin-Lim", 3.02, 0.09, 1.80),
    ("WaveNet", 3.98, 0.07, 42.0),
    ("HiFi-GAN", 4.21, 0.06, 0.012),
    ("Ours", 4.38, 0.06, 0.008),
    ("Ground truth", 4.52, 0.05, np.nan),
]

# (phone kind, duration s) -- "v" voiced, "f" fricative, "s" stop, "-" silence
UTT = [("-", 0.06), ("f", 0.11), ("v", 0.15), ("v", 0.13), ("s", 0.06),
       ("v", 0.17), ("f", 0.10), ("v", 0.16), ("v", 0.12), ("s", 0.05),
       ("v", 0.15), ("-", 0.06)]
FORMANTS = [(620, 1180, 2500), (450, 1900, 2600), (780, 1250, 2450),
            (320, 2200, 2950), (560, 900, 2400), (700, 1400, 2550)]


def synth_gt(rng):
    """Additive source-filter utterance: harmonics under a formant envelope."""
    n = int(round(sum(d for _, d in UTT) * FS))
    t = np.arange(n) / FS
    voiced = np.zeros(n)
    kind_at = np.empty(n, dtype=object)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    f3 = np.zeros(n)
    i0, vi = 0, 0
    for kind, dur in UTT:
        i1 = min(n, i0 + int(round(dur * FS)))
        kind_at[i0:i1] = kind
        if kind == "v":
            voiced[i0:i1] = 1.0
            a, b, c = FORMANTS[vi % len(FORMANTS)]
            f1[i0:i1], f2[i0:i1], f3[i0:i1] = a, b, c
            vi += 1
        i0 = i1
    for arr in (f1, f2, f3):
        arr[arr == 0] = np.nan
    for k, arr in enumerate((f1, f2, f3)):
        idx = np.arange(n)
        good = ~np.isnan(arr)
        arr[:] = np.interp(idx, idx[good], arr[good])
    f1 = gaussian_filter1d(f1, 0.012 * FS)
    f2 = gaussian_filter1d(f2, 0.012 * FS)
    f3 = gaussian_filter1d(f3, 0.012 * FS)

    env = gaussian_filter1d(voiced, 0.010 * FS)
    f0 = 128.0 * (0.80 / 1.0) ** (t / t[-1]) * (
        1.0 + 0.05 * np.sin(2 * np.pi * 1.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / FS

    x = np.zeros(n)
    for k in range(1, 46):
        fk = k * f0
        amp = np.zeros(n)
        for fc, bw, g in ((f1, 90.0, 1.00), (f2, 120.0, 0.62), (f3, 170.0, 0.34)):
            amp += g * np.exp(-0.5 * ((fk - fc) / bw) ** 2)
        amp += 0.035 * np.exp(-fk / 3000.0)
        x += amp * np.sin(k * phase + 0.4 * k) * (fk < 7600)
    x *= env

    # fricatives and stop bursts
    noise = rng.normal(0.0, 1.0, n)
    hp = signal.butter(4, 2200.0 / (FS / 2), btype="high", output="sos")
    hiss = signal.sosfiltfilt(hp, noise)
    gain = np.zeros(n)
    i0 = 0
    for kind, dur in UTT:
        i1 = min(n, i0 + int(round(dur * FS)))
        if kind == "f":
            gain[i0:i1] = 0.55
        elif kind == "s":
            m = i1 - i0
            gain[i0 + m // 2:i1] = 1.1 * np.exp(
                -np.arange(i1 - i0 - m // 2) / (0.008 * FS))
        i0 = i1
    x = x + hiss * gaussian_filter1d(gain, 0.004 * FS)
    x += 0.0006 * rng.normal(0.0, 1.0, n)
    return t, x / (np.abs(x).max() * 1.02)


def stft_mag(x):
    f, tt, Z = signal.stft(x, fs=FS, window="hann", nperseg=NPER,
                           noverlap=NPER - HOP, nfft=NFFT)
    return f, tt, Z


def griffin_lim(mag, n, rng, iters=GL_ITERS):
    ph = np.exp(2j * np.pi * rng.random(mag.shape))
    y = None
    for _ in range(iters):
        _, y = signal.istft(mag * ph, fs=FS, window="hann", nperseg=NPER,
                            noverlap=NPER - HOP, nfft=NFFT)
        _, _, Z = stft_mag(y)
        ph = np.exp(1j * np.angle(Z))
    y = np.asarray(y, dtype=float)
    return np.pad(y, (0, max(0, n - y.size)))[:n]


def shaped_noise(rng, n, corner, order=4, kind="high"):
    sos = signal.butter(order, corner / (FS / 2), btype=kind, output="sos")
    return signal.sosfiltfilt(sos, rng.normal(0.0, 1.0, n))


def tilt(x, db_at_nyquist, corner=4000.0):
    """Gentle first-order spectral tilt above ``corner``."""
    sos = signal.butter(2, corner / (FS / 2), btype="high", output="sos")
    hi = signal.sosfiltfilt(sos, x)
    return x + (10.0 ** (db_at_nyquist / 20.0) - 1.0) * hi


def spec_db(x):
    f, tt, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=NPER,
                                  noverlap=NPER - HOP, nfft=NFFT, mode="psd")
    keep = f <= FMAX
    return f[keep], tt, 10.0 * np.log10(S[keep] + 1e-12)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(463)

    t, gt = synth_gt(rng)
    n = gt.size

    f_, _, Z = stft_mag(gt)
    x_gl = griffin_lim(np.abs(Z), n, rng)
    # match on RMS, not on peak: a peak match would leave a constant level
    # offset that the error map would show as a uniform tint
    x_gl *= np.sqrt((gt ** 2).mean() / ((x_gl ** 2).mean() + 1e-18))

    # modelled vocoder artefacts, gated by the signal envelope: a vocoder does
    # not hiss through the silences, it errs where there is signal to get wrong
    env = gaussian_filter1d(np.abs(gt), 0.010 * FS)
    env = np.clip(env / (env.max() + 1e-12), 0.015, None)
    x_wn = tilt(gt, -3.0) + 0.038 * shaped_noise(rng, n, 500.0) * env
    x_hf = tilt(gt, 1.2, corner=5500.0) + 0.016 * shaped_noise(rng, n, 3000.0) * env
    x_ou = tilt(gt, 0.3, corner=6500.0) + 0.008 * shaped_noise(rng, n, 2000.0) * env

    waves = [x_gl, x_wn, x_hf, x_ou, gt]
    ref_db = None
    specs, tt_, ff_ = [], None, None
    for w in waves:
        ff_, tt_, db = spec_db(w)
        specs.append(db)
    ref_db = specs[-1]
    top_max = max(s.max() for s in specs)
    specs = [s - top_max for s in specs]              # dB re the loudest frame
    ref_db = specs[-1]

    # the error is only defined where the reference has energy: below the
    # floor a dB ratio is noise divided by noise, and it would swamp the map
    mask = ref_db > FLOOR
    diffs = [np.where(mask, s - ref_db, np.nan) for s in specs]

    # log-spectral distance over exactly the bins that are drawn
    lsd = [float(np.sqrt(np.nanmean(d ** 2))) for d in diffs]

    n_top = shared_norm(specs, vmin=-75.0, vmax=0.0)
    n_dif = shared_norm(diffs, robust=(0.5, 99.5), symmetric=True)
    cmap_t, cmap_d = "magma", get_cmap("cb_diverging")
    cmap_d.set_bad("#f0f0f0")

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 5, left=0.062, right=0.905, top=0.772,
                          bottom=0.095, hspace=0.30, wspace=0.075)
    stroke = [pe.withStroke(linewidth=1.6, foreground="#00000099")]
    ext = [tt_[0], tt_[-1], 0.0, FMAX / 1000.0]

    for j, (name, mos, ci, rtf) in enumerate(SYSTEMS):
        is_gt = j == len(SYSTEMS) - 1
        is_ours = name == "Ours"

        ax = fig.add_subplot(gs[0, j])
        im_t = ax.imshow(specs[j], cmap=cmap_t, norm=n_top, origin="lower",
                         extent=ext, aspect="auto", interpolation="nearest",
                         rasterized=True)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(0, FMAX / 1000.0)
        ax.tick_params(labelbottom=False, labelsize=6.6, length=2.2)
        ax.set_yticks([0, 2, 4, 6, 8])
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Frequency (kHz)", fontsize=7.6)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.4 if is_ours else 0.8)
            s.set_color(OURS if is_ours else RULE)
        ax.set_title(name, fontsize=8.6, fontweight="bold", pad=14.0,
                     color=OURS if is_ours else INK)
        stat = (f"MOS {mos:.2f} \u00b1 {ci:.2f}   \u00b7   "
                + ("RTF \u2014" if np.isnan(rtf) else f"RTF {rtf:g}"))
        ax.text(0.5, 1.035, stat, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.4, color=SUB)

        axd = fig.add_subplot(gs[1, j])
        im_d = axd.imshow(diffs[j], cmap=cmap_d, norm=n_dif, origin="lower",
                          extent=ext, aspect="auto", interpolation="nearest",
                          rasterized=True)
        axd.set_xlim(ext[0], ext[1])
        axd.set_ylim(0, FMAX / 1000.0)
        axd.set_yticks([0, 2, 4, 6, 8])
        axd.tick_params(labelsize=6.6, length=2.2)
        if j:
            axd.tick_params(labelleft=False)
        else:
            axd.set_ylabel("Frequency (kHz)", fontsize=7.6)
        axd.set_xlabel("Time (s)", fontsize=7.6, labelpad=1.5)
        for s in axd.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.4 if is_ours else 0.8)
            s.set_color(OURS if is_ours else RULE)
        if is_gt:
            axd.text(0.5, 0.5, "reference\n(zero by construction)",
                     transform=axd.transAxes, ha="center", va="center",
                     fontsize=6.6, color=SUB, linespacing=1.25)
        if name == "Griffin-Lim":
            axd.text(0.03, 0.035,
                     "fitted to the GT magnitude, so its\n"
                     "phase artefacts cannot show up here",
                     transform=axd.transAxes, ha="left", va="bottom",
                     fontsize=6.0, color=SUB, linespacing=1.25,
                     bbox=dict(boxstyle="round,pad=0.22", fc="white",
                               ec="none", alpha=0.86))
        axd.text(0.03, 0.955, f"LSD {lsd[j]:.2f} dB", transform=axd.transAxes,
                 ha="left", va="top", fontsize=6.6, color=INK,
                 fontweight="bold" if is_ours else "normal",
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                           alpha=0.86))

    # one colorbar per scale, each anchored to the last column of its row
    cb_t = safe_colorbar(fig, im_t, fig.axes[-2], label="level (dB re peak)",
                         side="right", width=0.11, pad=0.09, labelsize=6.6)
    cb_t.ax.tick_params(labelsize=6.4)
    cb_d = safe_colorbar(fig, im_d, fig.axes[-1],
                         label="output \u2212 ground truth (dB)", side="right",
                         width=0.11, pad=0.09, labelsize=6.6)
    cb_d.ax.tick_params(labelsize=6.4)

    # row captions on the far left
    for ax, lab in ((fig.axes[0], "Output"), (fig.axes[1], "Error vs GT")):
        ax.text(-0.30, 0.5, lab, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=7.6, fontweight="bold",
                color=INK)

    best = int(np.argmin(lsd[:-1]))
    savefig(fig, "463_spx_vocoder_compare.png")


if __name__ == "__main__":
    main()
