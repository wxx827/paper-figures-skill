"""Motor-imagery ERD: mu-band envelope, time-frequency map, and laterality.

Main panel: C3 and C4 mu-band (8-12 Hz) envelopes aligned to cue onset for
a right-hand motor-imagery block, with the ERD window shaded.  Side panels:
a C3 spectrogram of the same trial average, and laterality index across 24
trials.  ERD percent and laterality are computed from the plotted envelopes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RED, BLUE, TEAL = "#b23a48", "#3b5b92", "#2a7f7a"
SEED = 1061
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def mu_env(t, rest, drop, side_gain, rng):
    env = rest * np.ones_like(t)
    erd = (t >= 0.5) & (t <= 3.5)
    env[erd] = rest * (1.0 - drop * side_gain)
    env += rng.normal(0, 0.035 * rest, t.size)
    return np.clip(env, 0.05, None)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    fs = 250.0
    t = np.arange(-1.0, 5.01, 1 / fs)
    c3 = np.mean([mu_env(t, 1.00, 0.42, 1.00, rng) for _ in range(24)], 0)
    c4 = np.mean([mu_env(t, 1.00, 0.42, 0.28, rng) for _ in range(24)], 0)
    base = (t >= -0.8) & (t < 0.0)
    erd_win = (t >= 0.8) & (t <= 3.2)
    erd3 = 100.0 * (c3[erd_win].mean() / c3[base].mean() - 1.0)
    erd4 = 100.0 * (c4[erd_win].mean() / c4[base].mean() - 1.0)
    li = (c4 - c3) / (c4 + c3)
    trial_li = rng.normal(li[erd_win].mean(), 0.07, 24)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_tf = axes_inches(fig, AUX_T)
    ax_li = axes_inches(fig, AUX_B)

    ax.axvspan(0.8, 3.2, color="#dbe8f2", lw=0, zorder=0)
    ax.axvline(0.0, color="#888888", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.plot(t, c3, color=BLUE, lw=1.6, label="C3 (contralateral)")
    ax.plot(t, c4, color=RED, lw=1.6, label="C4 (ipsilateral)")
    ax.set_xlim(-1.0, 5.0)
    ax.set_xlabel("time from cue (s)")
    ax.set_ylabel("mu-band envelope (arb.)")
    ax.set_title("Right-hand motor imagery, mu envelope", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.03, 0.06, f"ERD C3 = {erd3:.1f} %\nERD C4 = {erd4:.1f} %",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    # synthetic C3 trial for TF: rest then suppressed mu
    n = t.size
    x = rng.normal(0, 1.0, n)
    b, a = butter(2, [8 / (fs / 2), 12 / (fs / 2)], btype="band")
    mu = filtfilt(b, a, rng.normal(0, 1.0, n))
    gain = np.interp(t, [-1, 0.5, 1.0, 3.2, 3.8, 5.0],
                     [1.0, 1.0, 0.55, 0.55, 1.0, 1.0])
    sig = 0.35 * x + 1.1 * mu * gain
    f, tt, sxx = spectrogram(sig, fs=fs, nperseg=64, noverlap=48)
    m = (f >= 4) & (f <= 30)
    im = ax_tf.pcolormesh(tt - 1.0, f[m], 10 * np.log10(sxx[m] + 1e-12),
                          cmap=get_cmap("cb_sequential_blue"), shading="auto")
    ax_tf.set_ylim(4, 30)
    ax_tf.set_xlim(-1.0, 5.0)
    ax_tf.set_ylabel("Hz")
    ax_tf.set_title("C3 time-frequency", fontsize=8.5, fontweight="bold",
                    pad=5)
    ax_tf.tick_params(labelbottom=False)
    safe_colorbar(fig, im, ax_tf, label="dB", width=0.08, pad=0.04)

    ax_li.axhline(0.0, color="#888888", lw=0.7)
    ax_li.plot(np.arange(1, 25), trial_li, color=TEAL, marker="o", ms=4.2,
               lw=1.2, mfc="white", mew=1.0)
    ax_li.set_xlim(0.5, 24.5)
    ax_li.set_xlabel("trial")
    ax_li.set_ylabel("laterality (C4$-$C3)/(C4$+$C3)")
    ax_li.set_title("Trial laterality", fontsize=8.5, fontweight="bold",
                    pad=5)
    ax_li.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_li.spines[side].set_visible(False)
    ax_li.text(0.04, 0.90, f"mean LI = {trial_li.mean():.2f}",
               transform=ax_li.transAxes, fontsize=6.8, color=INK)

    panel_labels_inches(fig, [ax, ax_tf, ax_li], dx=-0.62, dy=0.14)
    savefig(fig, "1061_bci_motor_imagery_erd.png")


if __name__ == "__main__":
    main()
