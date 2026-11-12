"""F1-F2 vowel space with 95 % ellipses and one token's formant track.

Main panel: measured (F1, F2) of 180 synthetic vowels, five phonemes, each
ellipse the sample covariance of that phoneme.  Side panels: a wideband
spectrogram of one /a/ token with F1/F2 tracks overlaid, and the same tracks
as time series.  Every printed Hertz value is read off the arrays that are
drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
RED, BLUE, TEAL = "#b23a48", "#3b5b92", "#2a7f7a"
GOLD, GREY = "#c9a227", "#8a8a8a"
COLS = [BLUE, TEAL, GOLD, RED, GREY]
VOWELS = [("i", 270, 2290), ("e", 390, 1960), ("a", 750, 1180),
          ("o", 570, 840), ("u", 300, 870)]
CHI2 = 5.991
SEED = 1048
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 5.50, 4.46)
AUX_T = (7.62, 3.52, 2.78, 1.96)
AUX_B = (7.62, 1.02, 2.78, 1.92)


def ellipse_xy(mean, cov, n=80):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2 * np.pi, n)
    a = np.sqrt(CHI2 * vals[0])
    b = np.sqrt(CHI2 * vals[1])
    rot = vecs @ np.array([a * np.cos(t), b * np.sin(t)])
    return mean[0] + rot[0], mean[1] + rot[1]


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    clouds = {}
    for name, f1, f2 in VOWELS:
        cov = np.array([[18.0 ** 2, -90.0], [-90.0, 55.0 ** 2]])
        pts = rng.multivariate_normal([f1, f2], cov, 36)
        clouds[name] = pts

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_sp = axes_inches(fig, AUX_T)
    ax_tr = axes_inches(fig, AUX_B)

    for k, (name, f1, f2) in enumerate(VOWELS):
        pts = clouds[name]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=COLS[k], alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=4)
        mu = pts.mean(0)
        cov = np.cov(pts.T)
        xs, ys = ellipse_xy(mu, cov)
        ax.plot(xs, ys, color=COLS[k], lw=1.2, zorder=5)
        ax.text(mu[0] + 18, mu[1] + 40, f"/{name}/", color=COLS[k],
                fontsize=9.0, fontweight="bold", ha="left", va="bottom")

    ax.set_xlabel("F1 (Hz)")
    ax.set_ylabel("F2 (Hz)")
    ax.set_xlim(180, 920)
    ax.set_ylim(700, 2550)
    ax.set_title("Vowel space, 95 % covariance ellipses", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.grid(lw=0.5, alpha=0.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    a_pts = clouds["a"]
    box = (f"$n$ = {a_pts.shape[0] * 5}\n"
           f"/a/ centroid {a_pts.mean(0)[0]:.0f}, {a_pts.mean(0)[1]:.0f} Hz")
    ax.text(0.03, 0.97, box, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.0, color=INK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    # one /a/ spectrogram + tracks
    fs, dur = 10000.0, 0.42
    t = np.arange(int(fs * dur)) / fs
    f1t = 750 + 40 * np.sin(2 * np.pi * 3.2 * t)
    f2t = 1180 + 70 * np.sin(2 * np.pi * 2.4 * t + 0.4)
    src = np.zeros_like(t)
    for f, a in ((f1t, 1.0), (f2t, 0.55), (2400 + 0 * t, 0.18)):
        src += a * np.sin(2 * np.pi * np.cumsum(f) / fs)
    src += 0.04 * rng.normal(size=t.size)
    nfft = 256
    spec, freqs, times, _ = ax_sp.specgram(
        src, NFFT=nfft, Fs=fs, noverlap=nfft - 32, cmap="Greys",
        scale="dB", vmin=-55, vmax=-8)
    ax_sp.plot(t, f1t, color=RED, lw=1.3, label="F1")
    ax_sp.plot(t, f2t, color=BLUE, lw=1.3, label="F2")
    ax_sp.set_ylim(0, 3500)
    ax_sp.set_xlim(0, dur)
    ax_sp.set_ylabel("Hz")
    ax_sp.set_title("/a/ spectrogram + tracks", fontsize=8.5,
                    fontweight="bold", pad=5)
    ax_sp.tick_params(labelbottom=False)
    ax_sp.legend(loc="upper right", frameon=False, fontsize=6.8, ncol=2)
    for side in ("top", "right"):
        ax_sp.spines[side].set_visible(False)

    ax_tr.plot(t * 1000.0, f1t, color=RED, lw=1.5)
    ax_tr.plot(t * 1000.0, f2t, color=BLUE, lw=1.5)
    ax_tr.set_xlim(0, dur * 1000)
    ax_tr.set_xlabel("time (ms)")
    ax_tr.set_ylabel("Hz")
    ax_tr.set_title("Formant tracks", fontsize=8.5, fontweight="bold", pad=5)
    ax_tr.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_tr.spines[side].set_visible(False)
    ax_tr.text(0.04, 0.12, f"F1 mean {f1t.mean():.0f} Hz",
               transform=ax_tr.transAxes, color=RED, fontsize=6.8)
    ax_tr.text(0.04, 0.28, f"F2 mean {f2t.mean():.0f} Hz",
               transform=ax_tr.transAxes, color=BLUE, fontsize=6.8)

    panel_labels_inches(fig, [ax, ax_sp, ax_tr], dx=-0.58, dy=0.14)
    savefig(fig, "1048_vox_formant_space.png")


if __name__ == "__main__":
    main()
