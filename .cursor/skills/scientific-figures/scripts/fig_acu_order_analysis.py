"""Order-tracked map of a gearbox run-up, with per-order speed traces.

Main panel: amplitude over the (order, speed) plane after angular resampling,
so a constant engine order is a vertical stripe and a fixed-frequency
structural resonance becomes a hyperbola.  Dashed guides mark the orders that
carry the energy.  Right panel: the amplitude of those same orders read out of
the map, column by column, so the two panels share one number.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pubstyle import (set_pub_style, savefig, add_colorbar, style_axes,
                      CATEGORICAL, COLORS)
from figkit import panel_labels_inches, halo_text

RPM = np.linspace(800.0, 6000.0, 220)
ORDER = np.linspace(0.25, 12.0, 480)
# structural resonances of the housing: (centre Hz, half-width Hz, gain)
MODES = [(196.0, 20.0, 7.5), (455.0, 33.0, 5.6), (742.0, 48.0, 3.6)]
# shaft / mesh orders -> (source strength, width in orders)
SOURCES = {0.5: (0.20, 0.055), 1.0: (0.55, 0.055), 2.0: (1.00, 0.060),
           3.0: (0.32, 0.060), 4.0: (0.70, 0.065), 6.0: (0.46, 0.070),
           8.0: (0.26, 0.075), 9.0: (0.34, 0.075)}
TRACK = [2.0, 4.0, 6.0]


def transfer(f):
    h = np.ones_like(f)
    for fc, q, g in MODES:
        h = h + g / (1.0 + ((f - fc) / q) ** 2)
    return h


def order_map(rng):
    """Amplitude (m/s^2 rms) on the (order, speed) grid."""
    O, R = np.meshgrid(ORDER, RPM)
    freq = O * R / 60.0
    amp = 0.010 * np.exp(-freq / 900.0) + 0.004
    load = (R / 1000.0) ** 0.9
    for k, (a, w) in SOURCES.items():
        amp = amp + 0.030 * a * load * transfer(k * R / 60.0) * \
            np.exp(-0.5 * ((O - k) / w) ** 2)
    ker = np.exp(-0.5 * (np.arange(-6, 7) / 2.4) ** 2)
    ker /= ker.sum()
    noise = rng.lognormal(0.0, 0.13, size=amp.shape)
    for axis in (1, 0):
        noise = np.apply_along_axis(
            lambda v: np.convolve(v, ker, mode="same"), axis, noise)
    return amp * noise / noise.mean()


def main():
    set_pub_style()
    rng = np.random.default_rng(415)
    amp = order_map(rng)
    amp_db = 20.0 * np.log10(amp / 1e-6)          # dB re 1 um/s^2

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.52], wspace=0.30,
                          left=0.065, right=0.955, top=0.895, bottom=0.105)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax)

    # ---- (a) order map ----------------------------------------------------
    pm = ax.pcolormesh(ORDER, RPM, amp_db, cmap="magma", shading="gouraud",
                       vmin=np.percentile(amp_db, 1.5),
                       vmax=np.percentile(amp_db, 99.8), rasterized=True)
    for k in TRACK:
        ax.axvline(k, color="white", lw=0.9, ls=(0, (4.0, 2.4)), alpha=0.85,
                   zorder=4)
        halo_text(ax, k, RPM[-1] - 90.0, f"{k:.0f}X", ha="center", va="top",
                  fontsize=7.5, fontweight="bold", zorder=6)
    for fc, _, _ in MODES:
        r_line = 60.0 * fc / ORDER
        ok = (r_line >= RPM[0]) & (r_line <= RPM[-1])
        ax.plot(ORDER[ok], r_line[ok], color="#cccccc", lw=0.8,
                ls=(0, (1.6, 1.8)), alpha=0.8, zorder=3)
    halo_text(ax, 0.985, 0.028, "dotted hyperbolae: fixed-frequency housing "
              "modes", transform=ax.transAxes, ha="right", va="bottom",
              fontsize=6.6, zorder=6)

    ax.set_xlabel("Engine order (-)")
    ax.set_ylabel("Shaft speed (rpm)")
    ax.set_xlim(ORDER[0], ORDER[-1])
    ax.set_ylim(RPM[0], RPM[-1])
    ax.set_xticks(np.arange(0, 13, 2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["rule"])
    ax.tick_params(which="both", color=COLORS["rule"])
    ax.set_title("Order spectrum over the run-up", fontsize=9.0,
                 fontweight="bold")
    cb = add_colorbar(fig, pm, ax, label="vibration amplitude "
                      "(dB re 1 $\\mu$m/s$^2$)")
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ---- (b) tracked orders ----------------------------------------------
    peaks = {}
    for k, col in zip(TRACK, CATEGORICAL[:3]):
        j = int(np.argmin(np.abs(ORDER - k)))
        cut = amp[:, j]                            # m/s^2
        peaks[k] = (RPM[int(np.argmax(cut))], cut.max())
        ax_r.plot(cut, RPM, color=col, lw=1.6 if k == 2.0 else 1.2,
                  label=f"order {k:.0f}", zorder=5 if k == 2.0 else 4)

    k_top = max(peaks, key=lambda kk: peaks[kk][1])
    r_top, a_top = peaks[k_top]
    ax_r.plot([a_top], [r_top], marker="o", ms=5.0, mfc="white",
              mec=COLORS["accent"], mew=1.3, zorder=6)
    ax_r.text(0.97, 0.045, f"order {k_top:.0f} resonance at {r_top:.0f} rpm\n"
              f"peak {a_top:.2f} m/s$^2$ ({k_top * r_top / 60.0:.0f} Hz)",
              transform=ax_r.transAxes, ha="right", va="bottom", fontsize=6.8,
              color=COLORS["accent"], linespacing=1.35, zorder=7)

    ax_r.set_xlabel("Order amplitude (m/s$^2$ rms)")
    ax_r.set_xlim(0, a_top * 1.09)
    ax_r.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_r.tick_params(labelsize=7.0)
    ax_r.grid(axis="x", lw=0.5, alpha=0.55)
    ax_r.legend(loc="center right", bbox_to_anchor=(1.0, 0.34), fontsize=7.0,
                frameon=False)
    ax_r.set_title("Tracked order amplitudes", fontsize=8.5,
                   fontweight="bold")
    ax_r.set_ylabel("Shaft speed (rpm)", fontsize=8.0)
    style_axes(ax_r)
    panel_labels_inches(fig, [ax, ax_r], dx=-0.40, dy=0.30)

    savefig(fig, "415_acu_order_analysis.png")


if __name__ == "__main__":
    main()
