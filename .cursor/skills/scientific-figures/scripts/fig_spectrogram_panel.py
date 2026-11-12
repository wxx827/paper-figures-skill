"""Time-frequency panel: LFP trace, spectrogram, spectral profile, band power."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.ndimage import uniform_filter

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

FS = 500.0
DUR = 12.0
N_TRIALS = 40
EVENTS = [(2.5, "cue"), (5.0, "movement"), (8.5, "reward")]

BANDS = [("Delta", 1, 4, "#5a6b8c"), ("Theta", 4, 8, "#3f8fa6"),
         ("Alpha", 8, 13, "#4f9d63"), ("Beta", 13, 30, "#cf8a3c"),
         ("Gamma", 30, 100, "#b1493f")]

BASE_WIN = (0.9, 2.1)


def pink_noise(rng, n, exponent=1.05):
    """1/f^exponent background with a realistic LFP spectrum."""
    w = rng.normal(0, 1, n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    W *= f ** (-exponent / 2.0)
    x = np.fft.irfft(W, n)
    return x / x.std()


def make_trial(rng, t):
    """One trial: 1/f background plus event-locked oscillatory structure."""
    x = 34.0 * pink_noise(rng, t.size)

    # tonic alpha, strongly suppressed around movement onset
    alpha_env = 1.0 - 0.88 * np.exp(-0.5 * ((t - 5.2) / 1.0) ** 2)
    x += 30.0 * alpha_env * np.sin(2 * np.pi * 10.2 * t + rng.uniform(0, 6.3))
    x += 10.0 * alpha_env * np.sin(2 * np.pi * 11.6 * t + rng.uniform(0, 6.3))

    # ongoing theta with a cue-locked boost
    theta_env = 1.0 + 2.1 * np.exp(-0.5 * ((t - 2.75) / 0.50) ** 2)
    x += 15.0 * theta_env * np.sin(2 * np.pi * 6.3 * t + rng.uniform(0, 6.3))

    # beta desynchronisation followed by a post-movement rebound
    beta_env = (1.0 - 0.85 * np.exp(-0.5 * ((t - 5.0) / 0.7) ** 2)
                + 2.4 * np.exp(-0.5 * ((t - 6.5) / 0.45) ** 2))
    x += 11.0 * beta_env * np.sin(2 * np.pi * 21.0 * t + rng.uniform(0, 6.3))

    # induced broadband gamma bursts locked to the events (random phase)
    for centre, amp, f0 in [(2.55, 9.0, 62), (5.05, 14.0, 74),
                            (5.28, 10.0, 48), (8.55, 11.0, 66)]:
        for fm, am, wd in [(1.0, 1.0, 0.10), (1.28, 0.65, 0.07)]:
            env = np.exp(-0.5 * ((t - centre) / wd) ** 2)
            x += amp * am * env * np.sin(2 * np.pi * f0 * fm * t
                                         + rng.uniform(0, 6.3))

    # slow post-reward delta increase + recording drift
    x += 26.0 * np.exp(-0.5 * ((t - 9.4) / 0.85) ** 2) \
        * np.sin(2 * np.pi * 2.3 * t + rng.uniform(0, 6.3))
    x += 28.0 * np.sin(2 * np.pi * 0.32 * t + rng.uniform(0, 6.3))
    return x


def band_power(x, lo, hi):
    """Squared Hilbert envelope of a band-pass, smoothed to the band's cycle."""
    sos = signal.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band",
                        output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x))) ** 2
    w = np.hanning(int(float(np.clip(3.0 / lo, 0.15, 1.10)) * FS) | 1)
    return np.convolve(env, w / w.sum(), mode="same")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(3)
    t = np.arange(int(DUR * FS)) / FS
    trials = np.array([make_trial(rng, t) for _ in range(N_TRIALS)])

    nper = 256
    Sxx = None
    for x in trials:
        f, tf, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=nper,
                                      noverlap=nper - 24, nfft=1024, mode="psd")
        Sxx = S if Sxx is None else Sxx + S
    Sxx /= N_TRIALS
    keep = (f >= 1.2) & (f <= 120.0)
    f, Sxx = f[keep], Sxx[keep]
    db = uniform_filter(10 * np.log10(Sxx + 1e-20), size=(3, 3), mode="nearest")
    vmin, vmax = np.percentile(db, [4, 99.6])

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.60, 2.05, 1.00],
                  width_ratios=[3.15, 0.80, 0.075],
                  hspace=0.20, wspace=0.07,
                  left=0.088, right=0.945, top=0.930, bottom=0.150)

    # ---- (a) example single-trial trace ------------------------------------
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_sig.plot(t, trials[0], color="#2f3b45", lw=0.42)
    ax_sig.set_xlim(0, DUR)
    ax_sig.set_ylabel("LFP\n(\u03bcV)", labelpad=6, linespacing=1.2)
    ax_sig.tick_params(labelbottom=False)
    ax_sig.set_yticks([-150, 0, 150])
    style_axes(ax_sig)
    add_panel_label(ax_sig, "(a)", x=-0.085, y=1.06, size=10)
    ax_sig.text(0.998, 0.02, "single trial", transform=ax_sig.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="#777777")

    # ---- (b) trial-averaged spectrogram ------------------------------------
    ax_tf = fig.add_subplot(gs[1, 0], sharex=ax_sig)
    mesh = ax_tf.pcolormesh(tf, f, db, cmap="magma", vmin=vmin, vmax=vmax,
                            shading="gouraud", rasterized=True)
    ax_tf.set_yscale("log")
    ax_tf.set_ylim(f[0], 120)
    ax_tf.set_yticks([2, 4, 8, 13, 30, 60, 120])
    ax_tf.set_yticklabels(["2", "4", "8", "13", "30", "60", "120"])
    ax_tf.minorticks_off()
    ax_tf.set_ylabel("Frequency (Hz)")
    ax_tf.tick_params(labelbottom=False)
    for spine in ax_tf.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")
    add_panel_label(ax_tf, "(b)", x=-0.085, y=1.01, size=10)

    stroke = [pe.withStroke(linewidth=1.7, foreground="#00000099")]
    for name, lo, hi, _ in BANDS:
        ax_tf.axhline(lo, color="white", lw=0.45, alpha=0.30, zorder=3)
        ax_tf.text(11.78, np.sqrt(lo * min(hi, 122)), name, color="white",
                   fontsize=6.5, va="center", ha="right", fontweight="bold",
                   zorder=7, path_effects=stroke)
    ax_tf.text(0.012, 0.965, f"average of {N_TRIALS} trials",
               transform=ax_tf.transAxes, ha="left", va="top", color="white",
               fontsize=6.5, zorder=7, path_effects=stroke)

    # ---- (c) mean spectral profile ----------------------------------------
    ax_prof = fig.add_subplot(gs[1, 1], sharey=ax_tf)
    prof = db.mean(axis=1)
    lo_x = prof.min() - 3
    for name, blo, bhi, col in BANDS:
        m = (f >= blo) & (f <= bhi)
        ax_prof.fill_betweenx(f[m], lo_x, prof[m], color=col, alpha=0.32,
                              lw=0, zorder=2)
    ax_prof.plot(prof, f, color="#2f3b45", lw=1.2, zorder=3)
    ax_prof.set_xlim(lo_x, prof.max() + 4)
    ax_prof.tick_params(labelleft=False)
    ax_prof.set_xlabel("Mean PSD\n(dB \u03bcV$^2$ Hz$^{-1}$)", fontsize=7,
                       labelpad=2, linespacing=1.2)
    ax_prof.grid(axis="x", lw=0.4, alpha=0.5)
    style_axes(ax_prof)

    sel = (f > 7) & (f < 14)
    ipk = int(np.argmax(np.where(sel, prof, -np.inf)))
    ax_prof.plot([prof[ipk]], [f[ipk]], marker="o", ms=3.2, mfc="white",
                 mec="#1a1a1a", mew=0.8, zorder=5)
    ax_prof.text(0.96, 0.965, f"\u03b1 peak\n{f[ipk]:.1f} Hz",
                 transform=ax_prof.transAxes, ha="right", va="top",
                 fontsize=6.5, fontweight="bold", color="#1a1a1a",
                 linespacing=1.25)
    add_panel_label(ax_prof, "(c)", x=-0.14, y=1.01, size=10)

    cax = fig.add_subplot(gs[1, 2])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2, color="#333333")
    cb.set_label("Power (dB \u03bcV$^2$ Hz$^{-1}$)", fontsize=7, labelpad=4)

    # ---- (d) band-limited power time courses -------------------------------
    ax_bp = fig.add_subplot(gs[2, 0], sharex=ax_sig)
    base = (t >= BASE_WIN[0]) & (t <= BASE_WIN[1])
    edge = (t > 0.8) & (t < DUR - 0.8)
    for name, lo, hi, col in BANDS:
        p = np.array([band_power(x, lo, hi) for x in trials])
        pct = 100.0 * (p / p[:, base].mean() - 1.0)
        m = pct.mean(axis=0)
        sem = pct.std(axis=0, ddof=1) / np.sqrt(N_TRIALS)
        ax_bp.fill_between(t[edge], (m - sem)[edge], (m + sem)[edge],
                           color=col, alpha=0.25, lw=0)
        ax_bp.plot(t[edge], m[edge], color=col, lw=1.25,
                   label=f"{name} ({lo}\u2013{hi} Hz)")
    ax_bp.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax_bp.axvspan(*BASE_WIN, color="#cccccc", alpha=0.35, lw=0, zorder=0)
    ax_bp.text(np.mean(BASE_WIN), 0.96, "baseline",
               transform=ax_bp.get_xaxis_transform(), ha="center", va="top",
               fontsize=6.3, color="#6b6b6b")
    ax_bp.set_xlim(0, DUR)
    ax_bp.set_xlabel("Time (s)")
    ax_bp.set_ylabel("Band power\n(% from baseline)", linespacing=1.2)
    style_axes(ax_bp)
    add_panel_label(ax_bp, "(d)", x=-0.085, y=1.03, size=10)

    # ---- event markers -----------------------------------------------------
    for ax in (ax_sig, ax_tf, ax_bp):
        for et, _ in EVENTS:
            ax.axvline(et, color="#f2f2f2" if ax is ax_tf else "#555555",
                       lw=0.9, ls=(0, (3, 2)), zorder=6, alpha=0.9)
    for et, lab in EVENTS:
        ax_sig.text(et, 1.10, lab, transform=ax_sig.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    color="#333333")
        ax_sig.plot([et], [1.06], marker="v", ms=3.4, color="#333333",
                    transform=ax_sig.get_xaxis_transform(), clip_on=False)

    handles, labels = ax_bp.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=7, handlelength=1.3, columnspacing=1.6,
               bbox_to_anchor=(0.52, 0.048))

    savefig(fig, "137_spectrogram_panel.png")


if __name__ == "__main__":
    main()
