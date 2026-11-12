"""Run-up spectral waterfall of a synthetic powertrain noise measurement.

Main panel: narrow-band spectra measured at 22 steady engine speeds, stacked
along a true rpm axis (each trace is drawn as ``rpm + gain * level``), with the
2nd / 4th / 6th engine-order ridges traced by dashed lines.  Right panel: the
A-weighted overall level derived from exactly the same spectra, so the two
panels cannot disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pubstyle import (set_pub_style, savefig, get_cmap, lighten, style_axes,
                      COLORS)
from figkit import panel_labels_inches

F_MAX = 820.0
RPM = np.arange(900.0, 6001.0, 240.0)
# structural resonances of the mounted powertrain: (centre Hz, half-width, gain)
MODES = [(212.0, 22.0, 7.5), (468.0, 34.0, 5.5), (676.0, 46.0, 3.6)]
# engine order -> relative source strength
ORDERS = {1: 0.22, 2: 1.00, 3: 0.28, 4: 0.62, 6: 0.40, 8: 0.18}
RIDGE_ORDERS = [2, 4, 6]
DB_BAR = 20.0


def transfer(f):
    """Linear amplification of the mount / body path at frequency ``f``."""
    h = np.ones_like(np.asarray(f, dtype=float))
    for fc, q, g in MODES:
        h = h + g / (1.0 + ((np.asarray(f, dtype=float) - fc) / q) ** 2)
    return h


def _smooth_noise(rng, n, width=4.0):
    k = np.exp(-0.5 * (np.arange(-12, 13) / width) ** 2)
    return np.convolve(rng.normal(size=n), k / k.sum(), mode="same")


def spectrum(f, rpm, rng):
    """Narrow-band level (dB) at one steady speed."""
    f0 = rpm / 60.0
    p = 2.0e3 * np.exp(-f / 240.0) + 1.4e2
    load = (rpm / 1000.0) ** 1.7
    for k, a in ORDERS.items():
        fk = k * f0
        if fk > F_MAX * 1.06:
            continue
        w = 3.0 + 0.004 * fk
        p = p + 1.1e4 * a * load * transfer(fk) * np.exp(-0.5 * ((f - fk) / w) ** 2)
    return 10.0 * np.log10(p * 10.0 ** (0.10 * _smooth_noise(rng, f.size)))


def a_weight(f):
    """IEC 61672 A-weighting in dB."""
    f = np.asarray(f, dtype=float)
    num = 12194.0 ** 2 * f ** 4
    den = ((f ** 2 + 20.6 ** 2)
           * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
           * (f ** 2 + 12194.0 ** 2))
    return 20.0 * np.log10(num / np.maximum(den, 1e-30)) + 2.0


def main():
    set_pub_style()
    rng = np.random.default_rng(414)

    f = np.linspace(8.0, F_MAX, 900)
    stack = np.array([spectrum(f, r, rng) for r in RPM])
    aw = a_weight(f)
    overall = 10.0 * np.log10(np.sum(10.0 ** ((stack + aw) / 10.0), axis=1))

    base = float(stack.min())
    gain = 12.6                                  # rpm of paper height per dB

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.45], wspace=0.07,
                          left=0.068, right=0.975, top=0.895, bottom=0.105)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax)

    # ---- (a) waterfall ----------------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    cols = [cmap(0.30 + 0.68 * i / (len(RPM) - 1)) for i in range(len(RPM))]
    for i in range(len(RPM) - 1, -1, -1):        # far traces first
        y = RPM[i] + gain * (stack[i] - base)
        ax.fill_between(f, RPM[i], y, facecolor=lighten(cols[i], 0.58),
                        alpha=0.80, lw=0, zorder=10 + (len(RPM) - i))
        ax.plot(f, y, color=cols[i], lw=0.85,
                zorder=10 + (len(RPM) - i) + 0.5)

    for k in RIDGE_ORDERS:
        fk = k * RPM / 60.0
        lvl = np.array([np.interp(fki, f, s) for fki, s in zip(fk, stack)])
        yk = RPM + gain * (lvl - base)
        ok = fk <= F_MAX
        ax.plot(fk[ok], yk[ok], color=COLORS["accent"], lw=1.0,
                ls=(0, (4.5, 2.2)), zorder=60)
        ax.annotate(f"{k}X", xy=(fk[ok][-1], yk[ok][-1]),
                    xytext=(4.5, 5.0), textcoords="offset points",
                    fontsize=7.5, color=COLORS["accent"], fontweight="bold",
                    zorder=61)

    x_bar = 40.0
    y_bar = RPM[0] - 130.0
    ax.plot([x_bar, x_bar], [y_bar, y_bar + gain * DB_BAR], color=COLORS["rule"],
            lw=1.1, zorder=62, solid_capstyle="butt")
    for yy in (y_bar, y_bar + gain * DB_BAR):
        ax.plot([x_bar - 9, x_bar + 9], [yy, yy], color=COLORS["rule"], lw=1.1,
                zorder=62)
    ax.text(x_bar + 14, y_bar + gain * DB_BAR / 2.0, f"{DB_BAR:.0f} dB",
            fontsize=7.0, color=COLORS["rule"], va="center", zorder=62)

    ax.set_xlim(0, F_MAX)
    ax.set_ylim(RPM[0] - 280.0, RPM[-1] + gain * (stack[-1].max() - base) + 160.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Engine speed (rpm)")
    ax.set_yticks(np.arange(1000, 6001, 1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_title("Narrow-band spectra stacked over the run-up",
                 fontsize=9.0, fontweight="bold")
    style_axes(ax)

    # ---- (b) A-weighted overall level -------------------------------------
    ax_r.plot(overall, RPM, color=COLORS["rule"], lw=1.5, zorder=4)
    ax_r.fill_betweenx(RPM, overall.min() - 4.0, overall, lw=0,
                       facecolor=lighten("#3b5b92", 0.72), zorder=2)
    i_pk = int(np.argmax(overall))
    ax_r.plot([overall[i_pk]], [RPM[i_pk]], marker="o", ms=5.0,
              mfc=COLORS["accent"], mec="white", mew=0.9, zorder=6)
    ax_r.text(0.06, 0.955, f"peak {overall[i_pk]:.1f} dB(A)\n"
              f"at {RPM[i_pk]:.0f} rpm", transform=ax_r.transAxes, ha="left",
              va="top", fontsize=6.8, color=COLORS["accent"], zorder=7,
              linespacing=1.35)
    ax_r.set_xlim(overall.min() - 4.0, overall.max() + 3.0)
    ax_r.set_xlabel("Overall level (dB(A))")
    ax_r.tick_params(labelleft=False)
    ax_r.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_r.set_title("A-weighted overall", fontsize=8.5, fontweight="bold")
    ax_r.tick_params(labelsize=7.0)
    style_axes(ax_r)
    panel_labels_inches(fig, [ax, ax_r], dx=-0.40, dy=0.30)

    rise = overall[i_pk] - overall[0]
    savefig(fig, "414_acu_spectral_waterfall.png")


if __name__ == "__main__":
    main()
