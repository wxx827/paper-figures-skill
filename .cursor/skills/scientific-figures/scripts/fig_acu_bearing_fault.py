"""Outer-race bearing fault seen in the waveform, the time-frequency map and
the squared envelope spectrum.

The impulse train is invisible in the raw acceleration and only vaguely
suggested by the spectrogram; it is the envelope spectrum that resolves the
outer-race passing frequency and its harmonics.  The characteristic
frequencies on the right are computed from the bearing geometry, and the
measured harmonic spacing is read back off the envelope spectrum.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import hilbert, stft
from pubstyle import set_pub_style, savefig, style_axes, COLORS
from figkit import (text_card, use_inch_units, panel_label_inches,
                    safe_colorbar)

FS = 20000.0
DUR = 0.42
RPM = 1750.0
N_BALL, D_BALL, D_PITCH = 9, 7.94, 39.04      # rolling elements, mm
F_RES, DECAY = 3200.0, 780.0                  # resonance (Hz), decay (1/s)
N_HARM = 5


def fault_frequencies():
    fr = RPM / 60.0
    ratio = D_BALL / D_PITCH
    return {
        "Shaft, $f_r$": fr,
        "BPFO": N_BALL / 2.0 * fr * (1.0 - ratio),
        "BPFI": N_BALL / 2.0 * fr * (1.0 + ratio),
        "BSF": D_PITCH / (2.0 * D_BALL) * fr * (1.0 - ratio ** 2),
        "FTF": fr / 2.0 * (1.0 - ratio),
    }


def simulate(rng, freqs):
    t = np.arange(0.0, DUR, 1.0 / FS)
    fr, bpfo = freqs["Shaft, $f_r$"], freqs["BPFO"]

    x = 0.35 * np.sin(2 * np.pi * fr * t) + 0.18 * np.sin(4 * np.pi * fr * t
                                                          + 0.7)
    n_imp = int(DUR * bpfo) + 1
    for i in range(n_imp):
        t0 = (i + rng.normal(0.0, 0.012)) / bpfo
        if not 0.0 <= t0 < DUR:
            continue
        amp = 1.5 * (1.0 + 0.25 * np.sin(2 * np.pi * fr * t0))
        m = t >= t0
        x[m] += amp * np.exp(-DECAY * (t[m] - t0)) * \
            np.sin(2 * np.pi * F_RES * (t[m] - t0))
    return t, x + rng.normal(0.0, 0.30, t.size)


def envelope_spectrum(x):
    env = np.abs(hilbert(x))
    env = env - env.mean()
    win = np.hanning(env.size)
    spec = np.abs(np.fft.rfft(env * win)) * 4.0 / env.size
    return np.fft.rfftfreq(env.size, 1.0 / FS), spec


def main():
    set_pub_style()
    rng = np.random.default_rng(421)
    freqs = fault_frequencies()
    t, x = simulate(rng, freqs)
    fe, se = envelope_spectrum(x)

    band = (fe > 40.0) & (fe < 200.0)
    f_meas = float(fe[band][np.argmax(se[band])])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.32],
                          height_ratios=[0.70, 1.0, 1.0], wspace=0.20,
                          hspace=0.40, left=0.062, right=0.985, top=0.885,
                          bottom=0.095)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_w)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    # ---- (a) raw acceleration ---------------------------------------------
    ax_w.plot(t, x, color=COLORS["rule"], lw=0.45)
    ax_w.set_ylabel("Acceleration\n(m/s$^2$)", fontsize=8.0, linespacing=1.25)
    ax_w.set_xlim(0.0, DUR)
    ax_w.tick_params(labelbottom=False, labelsize=7.0)
    ax_w.set_title("Raw vibration signal", fontsize=8.5, fontweight="bold")
    style_axes(ax_w)

    # ---- (b) short-time Fourier transform ---------------------------------
    f_st, t_st, zxx = stft(x, fs=FS, nperseg=256, noverlap=224)
    mag = 20.0 * np.log10(np.abs(zxx) + 1e-9)
    mag = mag - mag.max()
    pcm = ax_s.pcolormesh(t_st, f_st / 1000.0, mag, cmap="magma",
                          shading="gouraud", vmin=-42.0, vmax=0.0,
                          rasterized=True)
    ax_s.axhline(F_RES / 1000.0, color="white", lw=0.8, ls=(0, (4.0, 2.4)),
                 alpha=0.85)
    ax_s.text(DUR * 0.995, F_RES / 1000.0 + 0.16,
              f"excited resonance {F_RES / 1000.0:.1f} kHz", ha="right",
              va="bottom", fontsize=6.6, color="white")
    ax_s.set_ylim(0.0, 8.0)
    ax_s.set_ylabel("Frequency (kHz)", fontsize=8.0)
    ax_s.set_xlabel("Time (s)", fontsize=8.0)
    ax_s.tick_params(labelsize=7.0)
    for sp in ax_s.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color(COLORS["rule"])
    ax_s.set_title("Time-frequency map", fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, pcm, ax_s, label="level (dB re peak)", side="right",
                  width=0.10, pad=0.07, ticks=[-40, -30, -20, -10, 0])

    # ---- (c) squared envelope spectrum ------------------------------------
    keep = fe <= 560.0
    ax_e.plot(fe[keep], se[keep], color=COLORS["rule"], lw=0.9, zorder=4)
    ax_e.fill_between(fe[keep], 0.0, se[keep], color="#3b5b92", alpha=0.18,
                      lw=0, zorder=3)
    for k in range(1, N_HARM + 1):
        fk = k * freqs["BPFO"]
        ax_e.axvline(fk, color=COLORS["accent"], lw=0.9, ls=(0, (3.6, 2.2)),
                     zorder=5)
        ax_e.text(fk, 0.985, f"{k}$\\times$", transform=ax_e.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.6,
                  color=COLORS["accent"], fontweight="bold")
    for name in ("BPFI", "BSF"):
        ax_e.axvline(freqs[name], color=COLORS["baseline"], lw=0.8,
                     ls=(0, (1.6, 1.8)), zorder=4)
        ax_e.text(freqs[name] + 3.0, 0.60, name,
                  transform=ax_e.get_xaxis_transform(), ha="left", va="bottom",
                  fontsize=6.6, color=COLORS["sub"])
    ax_e.set_xlim(0.0, 560.0)
    ax_e.set_ylim(0.0, se[keep][fe[keep] > 20.0].max() * 1.30)
    ax_e.set_xlabel("Frequency (Hz)", fontsize=8.0)
    ax_e.set_ylabel("Envelope\n(m/s$^2$)", fontsize=8.0, linespacing=1.25)
    ax_e.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_e.tick_params(labelsize=7.0)
    ax_e.set_title(f"Envelope spectrum: harmonics spaced "
                   f"{f_meas:.1f} Hz against {freqs['BPFO']:.1f} Hz theory",
                   fontsize=8.5, fontweight="bold")
    style_axes(ax_e)

    # ---- (d) characteristic frequency card --------------------------------
    ax_c.set_title("Characteristic\nfrequencies", fontsize=8.5,
                   fontweight="bold", linespacing=1.3)
    cw, ch = use_inch_units(ax_c)
    rows = list(freqs.items())
    bh = (ch - 0.30) / (len(rows) + 1) - 0.08
    for i, (name, value) in enumerate(rows):
        hit = name == "BPFO"
        y0 = ch - 0.22 - (i + 1) * (bh + 0.08)
        text_card(ax_c, 0.06, y0, cw - 0.12, bh,
                  f"{value:.1f} Hz   ({value / freqs['Shaft, $f_r$']:.2f}$\\times$)",
                  title=name, fs=8.2, title_fs=7.0,
                  title_color=COLORS["accent"] if hit else COLORS["sub"],
                  color=COLORS["ink"], weight="bold", ha="left",
                  fc="#fbeff0" if hit else "#f5f5f5",
                  ec=COLORS["accent"] if hit else "#cccccc",
                  lw=1.1 if hit else 0.6, rounding=0.05, pad=0.10)
    ax_c.text(0.06, 0.05, f"{N_BALL} rolling elements\n$d$ = {D_BALL} mm, "
              f"$D$ = {D_PITCH} mm\nshaft {RPM:.0f} rpm", ha="left",
              va="bottom", fontsize=6.6, color=COLORS["muted"],
              linespacing=1.5)

    for ax, lab in zip((ax_w, ax_s, ax_e, ax_c), ("(a)", "(b)", "(c)", "(d)")):
        panel_label_inches(fig, ax, lab, dx=-0.44, dy=0.16)

    savefig(fig, "421_acu_bearing_fault.png")


if __name__ == "__main__":
    main()
