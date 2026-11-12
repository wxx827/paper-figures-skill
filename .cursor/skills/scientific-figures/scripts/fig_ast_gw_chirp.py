"""Binary-black-hole chirp: waveform, time-frequency track and the recovery.

The main panel pair is the claim: whitened strain in which the last cycles
before merger stand out of the noise, and underneath the same second of data
as a spectrogram, where the signal is the sweep from the detector's low-
frequency wall up to a few hundred hertz.  The chirp mass printed in the title
is *measured*, not assumed: the ridge of the spectrogram is traced, and the
Newtonian result that f^(-8/3) falls linearly in time is fitted to that ridge.

Side panels carry the recovery: a matched-filter statistic formed by
cross-correlating the noisy data with the template, and the joint posterior on
chirp mass and effective spin.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"

RED = "#b23a48"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"

T_SUN_S = 4.925490947e-6        # G M_sun / c^3, in seconds
MC_TRUE = 30.0                  # chirp mass, solar masses
F_MAX = 280.0                   # inspiral cut, Hz
F_RING = 255.0                  # ringdown frequency, Hz
TAU_RING = 0.0042               # ringdown damping time, s
FS = 2048.0
T0, T1 = -1.00, 0.12
NOISE = 0.17                    # whitened noise, in units of peak strain

# Perceptually uniform, dark-to-bright: the standard house choice for a
# single-ended power map (DESIGN_SYSTEM section 2.4).
SPEC_CMAP = mpl.colormaps["magma"]


def chirp_frequency(tau, mc=MC_TRUE):
    """Newtonian inspiral frequency, tau seconds before coalescence."""
    m = mc * T_SUN_S
    tau = np.clip(tau, 1e-6, None)
    return (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** 0.375 * m ** (-0.625)


def waveform(seed=57):
    rng = np.random.default_rng(seed)
    t = np.arange(T0, T1, 1.0 / FS)
    insp = t < 0.0

    f = np.full(t.size, F_RING)
    f[insp] = np.minimum(chirp_frequency(-t[insp]), F_MAX)
    phase = 2.0 * np.pi * np.concatenate(
        [[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])

    env = (f / F_MAX) ** (2.0 / 3.0)
    h = env * np.cos(phase)
    ring = ~insp
    h[ring] = (np.exp(-t[ring] / TAU_RING)
               * np.cos(2.0 * np.pi * F_RING * t[ring] + phase[insp][-1]))
    env[ring] = np.exp(-t[ring] / TAU_RING)
    h = h / np.abs(h).max()
    env = env / env.max()

    data = h + rng.normal(0.0, NOISE, t.size)
    return t, h, env, data, f


def fit_chirp_mass(t_ridge, f_ridge):
    """Recover Mc from d(f^-8/3)/dt, which is constant for a Newtonian chirp."""
    slope = np.polyfit(t_ridge, f_ridge ** (-8.0 / 3.0), 1)[0]
    m = (-slope * 5.0 / (256.0 * np.pi ** (8.0 / 3.0))) ** 0.6
    return float(m / T_SUN_S)


def matched_filter(t, data, template):
    """Normalised cross-correlation, as a function of trial coalescence time.

    ``np.correlate(..., mode="same")`` puts zero lag at index ``N // 2``, not
    at index 0, so the time axis has to be re-centred or the peak lands half a
    record away from the merger it found.
    """
    tpl = template - template.mean()
    rho = np.correlate(data, tpl, mode="same")
    rho = rho / (NOISE * np.sqrt(np.sum(tpl ** 2)))
    return t - t[t.size // 2], rho


def posterior(mc_fit, seed=12, n=9000):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.35 ** 2, 0.45 * 1.35 * 0.115],
                    [0.45 * 1.35 * 0.115, 0.115 ** 2]])
    return rng.multivariate_normal([mc_fit, 0.055], cov, n)


def wave_panel(ax, t, h, env, data):
    ax.plot(t, data, color="#9a9a9a", lw=0.5, zorder=2,
            label="whitened strain data")
    ax.plot(t, h, color=RED, lw=1.05, zorder=4, label="best-fit template")
    ax.plot(t, env, color=RED, lw=0.7, ls=(0, (3.5, 2.4)), alpha=0.75,
            zorder=3, label="template envelope")
    ax.plot(t, -env, color=RED, lw=0.7, ls=(0, (3.5, 2.4)), alpha=0.75,
            zorder=3)
    ax.axvline(0.0, color=RULE, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1)
    ax.text(0.004, 1.62, "merger", ha="left", va="top", fontsize=6.8,
            color=RULE)
    ax.annotate("", xy=(0.075, 1.30), xytext=(0.005, 1.30),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9))
    ax.text(0.078, 1.30, "ringdown", ha="left", va="center", fontsize=6.8,
            color=PURPLE)
    ax.set_xlim(T0, T1)
    ax.set_ylim(-1.85, 1.85)
    ax.set_ylabel("whitened strain (-)")
    ax.tick_params(labelbottom=False, labelsize=7.2)
    ax.grid(axis="y", lw=0.45, alpha=0.5)
    ax.legend(fontsize=7.0, frameon=False, loc="lower left", ncol=3,
              handlelength=1.6, borderaxespad=0.3, columnspacing=1.2)


def trace_ridge(z_db, freqs, t_bin):
    """Per-column spectral peak, refined to sub-bin precision and gated.

    Zero padding samples the 62 ms window finely enough that a three-point
    parabolic fit in dB recovers the peak between bins; without it the ridge
    is quantised onto the FFT grid and the f^(-8/3) fit inherits a staircase.
    """
    band = (freqs >= 24.0) & (freqs <= 300.0)
    fb, pb = freqs[band], z_db[band]
    pb = np.apply_along_axis(
        lambda r: np.convolve(r, np.ones(3) / 3.0, mode="same"), 1, pb)
    cols = np.arange(pb.shape[1])
    i = np.argmax(pb, axis=0)
    lo = np.clip(i - 1, 0, pb.shape[0] - 1)
    hi = np.clip(i + 1, 0, pb.shape[0] - 1)
    y0, y1, y2 = pb[lo, cols], pb[i, cols], pb[hi, cols]
    shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2 - 1e-9)
    ridge = fb[i] + np.clip(shift, -1.0, 1.0) * (fb[1] - fb[0])

    gate = (y1 - np.median(pb, axis=0)) > 7.0
    keep = gate & (t_bin > -0.62) & (t_bin < -0.02)
    return ridge, keep


def spec_panel(fig, ax, t, data):
    pxx, freqs, bins, im = ax.specgram(
        data, NFFT=128, pad_to=512, Fs=FS, noverlap=112, cmap=SPEC_CMAP,
        xextent=(t[0], t[-1]), scale="dB", mode="magnitude")
    z_db = 20.0 * np.log10(np.maximum(pxx, 1e-12))
    im.set_clim(np.percentile(z_db, 58.0), np.percentile(z_db, 99.7))

    t_bin = bins + t[0]
    ridge, keep = trace_ridge(z_db, freqs, t_bin)
    mc_fit = fit_chirp_mass(t_bin[keep], ridge[keep])

    # A pale dashed line vanishes into the bright ridge it is tracing, so it
    # gets a dark underlay first.
    ax.plot(t_bin[keep], ridge[keep], color="#1a1a1a", lw=2.2, alpha=0.55,
            zorder=4)
    ax.plot(t_bin[keep], ridge[keep], color="#f4f1ea", lw=1.0,
            ls=(0, (3.2, 2.2)), zorder=5)
    halo_text(ax, 0.025, 0.90, "traced ridge", transform=ax.transAxes,
              ha="left", va="top", fontsize=6.6, color="#f4f1ea")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(18.0, 340.0)
    ax.set_xlabel("time from merger (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.tick_params(labelsize=7.2)
    safe_colorbar(fig, im, ax, label="power (dB)", width=0.09, pad=0.06,
                  labelsize=6.4)
    return mc_fit, t_bin[keep], ridge[keep]


def snr_panel(ax, t, rho):
    ax.plot(t, rho, color=RULE, lw=0.8, zorder=3)
    k = int(np.argmax(np.abs(rho)))
    peak = float(np.abs(rho[k]))
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=1)
    ax.plot([t[k]], [rho[k]], marker="o", ms=4.5, mfc="white", mec=RED,
            mew=1.2, zorder=5)
    ax.annotate(f"peak $\\rho$ = {peak:.1f} at {t[k] * 1e3:+.0f} ms",
                xy=(t[k], rho[k]), xytext=(t[k] - 0.09, rho[k] * 0.80),
                ha="right", va="center", fontsize=6.9, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    thresh = 8.0
    ax.axhline(thresh, color=BLUE, lw=0.8, ls=(0, (3.2, 2.2)), zorder=2)
    ax.text(t[0] + 0.02, thresh, f"  detection threshold {thresh:.0f}",
            ha="left", va="bottom", fontsize=6.4, color=BLUE)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("trial coalescence time (s)", fontsize=7.4)
    ax.set_ylabel(r"matched-filter SNR $\rho$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Matched-filter statistic against the template",
                 fontsize=8.4, fontweight="bold", color=INK)
    return peak


def posterior_panel(ax, samples, mc_fit):
    mc, chi = samples[:, 0], samples[:, 1]
    kde = gaussian_kde(np.vstack([mc, chi]))
    gx = np.linspace(mc.min(), mc.max(), 140)
    gy = np.linspace(chi.min(), chi.max(), 140)
    gxx, gyy = np.meshgrid(gx, gy)
    dens = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)

    flat = np.sort(dens.ravel())[::-1]
    cdf = np.cumsum(flat) / flat.sum()
    levels = [float(flat[np.searchsorted(cdf, q)]) for q in (0.90, 0.50)]
    shades = get_palette("cb_sequential_blue", 6)

    ax.scatter(mc[::14], chi[::14], s=1.4, color="#cccccc", linewidths=0.0,
               zorder=2)
    ax.contourf(gx, gy, dens, levels=levels + [dens.max()],
                colors=[shades[1], shades[3]], alpha=0.85, zorder=3)
    ax.contour(gx, gy, dens, levels=levels, colors=[shades[5], RED],
               linewidths=[0.9, 1.2], zorder=4)
    ax.plot([mc_fit], [np.median(chi)], marker="+", ms=7.0, mew=1.3,
            color=RED, zorder=6)

    lo, hi = np.percentile(mc, [5, 95])
    ax.text(0.03, 0.95,
            f"$\\mathcal{{M}}_c$ = {np.median(mc):.1f}$^{{+{hi - np.median(mc):.1f}}}"
            f"_{{-{np.median(mc) - lo:.1f}}}$ $M_\\odot$ (90 %)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=INK)
    ax.text(0.97, 0.06, "50 % and 90 % credible", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, color=SUB)
    ax.set_xlabel(r"chirp mass $\mathcal{M}_c$ ($M_\odot$)", fontsize=7.4)
    ax.set_ylabel(r"effective spin $\chi_{\rm eff}$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.45, alpha=0.5)
    ax.set_title("Joint posterior on mass and spin", fontsize=8.4,
                 fontweight="bold", color=INK)
    return float(np.median(chi))


def main():
    set_pub_style()
    t, h, env, data, f_true = waveform()

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    # The left column stops at 5.82 in so that the spectrogram's colorbar and
    # its tick labels clear the posterior panel's y-axis label.
    ax_wave = axes_inches(fig, (0.92, 2.98, 4.90, 2.48))
    ax_spec = axes_inches(fig, (0.92, 0.88, 4.90, 1.94))
    ax_snr = axes_inches(fig, (7.18, 3.30, 2.92, 2.16))
    ax_post = axes_inches(fig, (7.18, 0.88, 2.92, 1.76))

    wave_panel(ax_wave, t, h, env, data)
    mc_fit, t_ridge, f_ridge = spec_panel(fig, ax_spec, t, data)
    peak = snr_panel(ax_snr, *matched_filter(t, data, h))
    chi_med = posterior_panel(ax_post, posterior(mc_fit), mc_fit)

    ax_wave.set_title(
        f"Ridge-fitted chirp mass {mc_fit:.1f} $M_\\odot$, recovered at "
        f"matched-filter SNR {peak:.1f}", fontsize=9.0, fontweight="bold",
        color=INK)


    panel_labels_inches(fig, [ax_wave, ax_snr, ax_post], dx=-0.66, dy=0.30)
    savefig(fig, "802_ast_gw_chirp.png")


if __name__ == "__main__":
    main()
