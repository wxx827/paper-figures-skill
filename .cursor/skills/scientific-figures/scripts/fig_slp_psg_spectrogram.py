"""Whole-night EEG spectrogram with the scored stages docked underneath.

The main panel is the time-frequency view a sleep laboratory actually reads:
low-frequency power blooms whenever the scoring drops into N3 and retreats to a
thin band during REM, while the spindle ridge near 13 Hz belongs to N2.  The
value of the figure comes from the alignment - the stage strip shares the time
axis exactly, so every feature in the spectrogram can be attributed to a scored
epoch instead of being described in words.  The two smaller panels turn the
same power array into its two standard summaries: the average spectrum of each
stage against the classical frequency bands, and the overnight decay of
slow-wave activity, whose time constant is fitted from the plotted points.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import (axes_inches, panel_labels_inches, shared_norm,
                    safe_colorbar, halo_text)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

BLUE = "#3b5b92"
RED = "#b23a48"
OCHRE = "#c0603f"

NAMES = ["W", "REM", "N1", "N2", "N3"]
COL = [RED, OCHRE, lighten(BLUE, 0.46), BLUE, darken(BLUE, 0.34)]

EPOCH_MIN = 0.5
EPOCH_H = EPOCH_MIN / 60.0
LIGHTS_OFF_H = 23.0
TIB_H = 8.0
FREQ = np.arange(0.5, 30.01, 0.5)
SWA_TAU_TRUE = 3.1                      # homeostatic decay imposed on delta

BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
         ("sigma", 12.0, 16.0), ("beta", 16.0, 30.0)]

# additive band power per stage, in the row order W, REM, N1, N2, N3
BUMPS = {
    "delta": ((1.8, 1.30), [10.0, 16.0, 42.0, 150.0, 290.0]),
    "theta": ((6.0, 1.35), [7.0, 30.0, 18.0, 10.0, 8.0]),
    "alpha": ((10.0, 1.05), [38.0, 7.0, 11.0, 4.0, 3.0]),
    "sigma": ((13.2, 0.90), [2.0, 2.0, 3.0, 46.0, 16.0]),
    "beta": ((20.0, 5.20), [7.0, 5.5, 3.0, 2.0, 1.6]),
}


def hypnogram(seed=9):
    """Compact epoch scoring: N3-heavy early, REM-heavy late."""
    rng = np.random.default_rng(seed)
    n = int(round(TIB_H / EPOCH_H))
    stg = np.zeros(n, dtype=int)
    i = 0

    def put(minutes, level):
        nonlocal i
        k = max(int(round(minutes / EPOCH_MIN)), 1)
        j = min(i + k, n)
        stg[i:j] = level
        i = j

    put(11.0 + rng.normal(0.0, 2.0), 0)
    n3_plan = [32.0, 24.0, 11.0, 4.0, 0.0]
    rem_plan = [10.0, 18.0, 25.0, 30.0, 32.0]
    c = 0
    while i < n - 20:
        put(rng.uniform(2.5, 5.0), 2)
        put(rng.uniform(24.0, 32.0), 3)
        if n3_plan[min(c, 4)] > 0.0:
            put(n3_plan[min(c, 4)] * rng.uniform(0.85, 1.15), 4)
            put(rng.uniform(16.0, 23.0), 3)
        else:
            put(rng.uniform(20.0, 27.0), 3)
        if c > 0 and rng.random() < 0.7:
            put(rng.uniform(1.0, 4.0), 0)
        put(rem_plan[min(c, 4)] * rng.uniform(0.85, 1.15), 1)
        c += 1
    stg[n - int(round(8.0 / EPOCH_MIN)):] = 0
    return stg


def runs(a):
    out = []
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        out.append((i, j + 1, int(a[i])))
        i = j + 1
    return out


def spectrogram(stg, seed=17):
    """Power spectral density per epoch, in microvolt squared per hertz."""
    rng = np.random.default_rng(seed)
    t = np.arange(stg.size) * EPOCH_H
    p = np.tile(26.0 * FREQ ** -1.35, (stg.size, 1))
    decay = np.exp(-t / SWA_TAU_TRUE)
    for name, ((f0, sd), amps) in BUMPS.items():
        shape = np.exp(-0.5 * ((FREQ - f0) / sd) ** 2)
        amp = np.array(amps)[stg]
        if name == "delta":
            amp = amp * decay
        p += amp[:, None] * shape[None, :]
    p *= np.exp(rng.normal(0.0, 0.16, p.shape))
    k = np.array([0.12, 0.20, 0.36, 0.20, 0.12])
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, p)
    return t, p


def stage_spectra(stg, p):
    return {NAMES[s]: p[stg == s].mean(axis=0) for s in range(5)
            if np.any(stg == s)}


def swa_series(stg, p, bin_min=20.0):
    """Binned delta power over NREM epochs, plus the fitted decay constant."""
    band = (FREQ >= 0.5) & (FREQ < 4.0)
    delta = p[:, band].mean(axis=1)
    nrem = (stg == 3) | (stg == 4)
    per = int(round(bin_min / EPOCH_MIN))
    tc, val = [], []
    for a in range(0, stg.size, per):
        m = nrem[a:a + per]
        if m.sum() < 4:
            continue
        tc.append((a + per / 2.0) * EPOCH_H)
        val.append(float(delta[a:a + per][m].mean()))
    tc, val = np.array(tc), np.array(val)
    slope, icpt = np.polyfit(tc, np.log(val), 1)
    return tc, val, -1.0 / slope, np.exp(icpt)


def clock_labels(hours):
    return ["%02d:00" % int(round(h) % 24) for h in hours]


def spectro_panel(fig, ax, t, p):
    db = 10.0 * np.log10(p.T)
    norm = shared_norm([db], robust=(1.5, 99.5))
    te = np.append(t, t[-1] + EPOCH_H)
    fe = np.append(FREQ - 0.25, FREQ[-1] + 0.25)
    im = ax.pcolormesh(te, fe, db, cmap="magma", norm=norm, shading="flat",
                       rasterized=True)
    ax.set_ylim(0.0, 30.0)
    ax.set_xlim(0.0, TIB_H)
    ax.set_yticks(np.arange(0.0, 30.1, 5.0))
    ax.set_xticks(np.arange(0.0, TIB_H + 0.1, 1.0))
    ax.set_xticklabels([])
    ax.set_ylabel("frequency (Hz)")
    for _, f0, f1 in BANDS[:-1]:
        ax.axhline(f1, color="white", lw=0.5, alpha=0.30)
    for name, f0, f1 in BANDS:
        halo_text(ax, 0.055, 0.5 * (f0 + f1), name, ha="left", va="center",
                  fontsize=6.2, color="white", halo="#1a1a1a", halo_lw=1.5)
    safe_colorbar(fig, im, ax, label=r"PSD (dB re 1 $\mu V^2/Hz$)",
                  width=0.11, pad=0.09, labelsize=6.6)
    ax.set_title("Delta power tracks the scored depth epoch by epoch",
                 fontsize=9.0, fontweight="bold", color=INK)
    return im


def strip_panel(ax, stg):
    for a, b, lv in runs(stg):
        ax.axvspan(a * EPOCH_H, b * EPOCH_H, color=COL[lv], lw=0.0)
    ax.set_xlim(0.0, TIB_H)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ticks = np.arange(0.0, TIB_H + 0.1, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(clock_labels(LIGHTS_OFF_H + ticks), fontsize=7.0)
    ax.set_xlabel("clock time (h:min)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.text(-0.012, 0.5, "stage", transform=ax.transAxes, ha="right",
            va="center", fontsize=7.0, color=INK)


def spectra_panel(ax, spec):
    for i, (name, f0, f1) in enumerate(BANDS):
        ax.axvspan(f0, f1, color="#f5f5f5" if i % 2 else "#ebebeb", lw=0.0,
                   zorder=0)
        ax.text(0.5 * (f0 + f1), 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.0, color=MUTED)
    for name, s in spec.items():
        ax.semilogy(FREQ, s, color=COL[NAMES.index(name)], lw=1.25,
                    label=name, zorder=3)
    ax.set_xlim(0.5, 30.0)
    ax.set_xlabel("frequency (Hz)", fontsize=7.4)
    ax.set_ylabel(r"PSD ($\mu V^2/Hz$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right", ncol=2,
              handlelength=1.2, borderaxespad=0.2, columnspacing=1.0)
    ax.set_title("Mean spectrum of each stage", fontsize=8.2,
                 fontweight="bold", color=INK, pad=9)


def swa_panel(ax, tc, val, tau, amp0):
    grid = np.linspace(tc[0], tc[-1], 160)
    ax.plot(grid, amp0 * np.exp(-grid / tau), color=RULE, lw=1.2,
            ls=(0, (4, 2.4)), zorder=3,
            label=r"fit, $\tau$ = %.2f h" % tau)
    ax.plot(tc, val, marker="o", ms=3.4, lw=1.4, color=darken(BLUE, 0.34),
            mfc="white", mew=0.9, zorder=4, label="NREM 20-min bins")
    drop = 100.0 * (1.0 - val[-1] / val[0])
    ax.text(0.97, 0.94, "%.0f %% decline across the night" % drop,
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=INK)
    ax.set_xlim(0.0, TIB_H)
    ticks = np.arange(0.0, TIB_H + 0.1, 2.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(clock_labels(LIGHTS_OFF_H + ticks), fontsize=6.5)
    ax.set_ylim(0.0, val.max() * 1.30)
    ax.set_xlabel("clock time (h:min)", fontsize=7.4)
    ax.set_ylabel(r"SWA 0.5-4 Hz ($\mu V^2/Hz$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left",
              handlelength=1.4, borderaxespad=0.2)
    ax.set_title("Slow-wave activity dissipates", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    stg = hypnogram()
    t, p = spectrogram(stg)
    spec = stage_spectra(stg, p)
    tc, val, tau, amp0 = swa_series(stg, p)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.82, 1.56, 5.00, 3.81))
    ax_strip = axes_inches(fig, (0.82, 1.02, 5.00, 0.32))
    ax_top = axes_inches(fig, (6.92, 3.505, 2.90, 1.865))
    ax_bot = axes_inches(fig, (6.92, 0.98, 2.90, 1.865))

    spectro_panel(fig, ax_main, t, p)
    strip_panel(ax_strip, stg)
    spectra_panel(ax_top, spec)
    swa_panel(ax_bot, tc, val, tau, amp0)

    n3 = 100.0 * float((stg == 4).sum()) / max(float((stg != 0).sum()), 1.0)
    note = ("Synthetic single-channel EEG, %.0f h from %02d:00 in 30-s epochs; "
            "power is modelled as a 1/f background plus stage-dependent band "
            "peaks, so the spectrogram, the stage-mean spectra and the "
            "slow-wave decay all come from one array. The decay constant, the "
            "overnight decline and the N3 share are fitted or counted from the "
            "plotted values." % (TIB_H, LIGHTS_OFF_H))
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.52, dy=0.13)
    savefig(fig, "789_slp_psg_spectrogram.png")


if __name__ == "__main__":
    main()
