"""Aerosol size spectra: the same population seen as number, surface, volume.

Three lognormal modes are drawn once and then integrated into the three
moments an aerosol scientist actually uses.  Their peaks differ by six orders
of magnitude, so a shared linear axis would flatten two of the three curves
into the baseline; each spectrum is instead divided by its own maximum and the
absolute peak is printed on the curve with its unit, which keeps the shapes
comparable without hiding the magnitudes.  The auxiliary panels split every
moment by mode and reduce the population to four characteristic diameters -
all of them integrated from the very arrays that are plotted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_labels_inches

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]

# Moment colours: house qualitative positions 2, 3, 4.
NUM_C = "#3b5b92"
SUR_C = "#2e7d5b"
VOL_C = "#c9a227"

# (geometric mean diameter um, geometric sd, number concentration cm^-3, name)
MODES = [
    (0.030, 1.62, 8200.0, "Aitken"),
    (0.220, 1.72, 1350.0, "accumulation"),
    (2.200, 1.80, 3.2, "coarse"),
]

# Where each moment's call-out sits, in micrometres: the three valleys of the
# normalised spectra, so no box ever lands on a curve.
CALLOUT_X = [0.0135, 0.135, 1.70]


def mode_number(dp, dpg, sg, n0):
    """dN/dlogDp in cm^-3 for one lognormal mode."""
    ln = np.log10(dp / dpg)
    return n0 / (np.sqrt(2.0 * np.pi) * np.log10(sg)) * \
        np.exp(-ln ** 2 / (2.0 * np.log10(sg) ** 2))


def main():
    set_pub_style()
    rng = np.random.default_rng(553)

    dp = np.logspace(np.log10(0.006), np.log10(22.0), 600)   # um
    logd = np.log10(dp)

    # Each mode carries a small campaign-to-campaign amplitude jitter, so the
    # peaks, shares and diameters printed below all move with the seed.
    amps = [n0 * float(rng.normal(1.0, 0.06)) for _, _, n0, _ in MODES]
    per_mode = np.array([mode_number(dp, dpg, sg, a)
                         for (dpg, sg, _n0, _nm), a in zip(MODES, amps)])
    dNdlog = per_mode.sum(axis=0)
    dSdlog = np.pi * dp ** 2 * dNdlog                 # um^2 cm^-3
    dVdlog = np.pi / 6.0 * dp ** 3 * dNdlog           # um^3 cm^-3
    per_S = np.pi * dp ** 2 * per_mode
    per_V = np.pi / 6.0 * dp ** 3 * per_mode

    curves = [("d$N$/dlog$D_p$", dNdlog, NUM_C, "cm$^{-3}$"),
              ("d$S$/dlog$D_p$", dSdlog, SUR_C, "$\\mu$m$^2$ cm$^{-3}$"),
              ("d$V$/dlog$D_p$", dVdlog, VOL_C, "$\\mu$m$^3$ cm$^{-3}$")]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], left=0.058, right=0.975,
                          top=0.882, bottom=0.100, hspace=0.80, wspace=0.26)
    ax = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    # ------------------------------------------- (a) the three normalised spectra
    edges = [(0.006, 0.090), (0.090, 0.900), (0.900, 22.0)]
    for i, ((lo, hi), (_dpg, _sg, _n0, name)) in enumerate(zip(edges, MODES)):
        ax.axvspan(lo, hi, color="#f5f5f5" if i % 2 == 0 else "#f0f0f0",
                   zorder=0)
        ax.text(np.sqrt(lo * hi), 1.075, f"{name} mode", ha="center",
                va="bottom", fontsize=7.0, color=SUB)

    for (label, y, col, unit), xt in zip(curves, CALLOUT_X):
        peak = float(y.max())
        ax.plot(dp, y / peak, color=col, lw=1.7, zorder=4)
        xp = float(dp[int(np.argmax(y))])
        ax.plot([xp], [1.0], marker="o", ms=4.2, mfc="white", mec=col,
                mew=1.2, zorder=6)
        ax.annotate(f"{label}\npeak {peak:.3g} {unit}\nat {xp:.3g} $\\mu$m",
                    xy=(xp, 1.0), xytext=(xt, 0.90),
                    fontsize=6.5, color=col, ha="center", va="top",
                    linespacing=1.30, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                              edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=col))

    ax.set_xscale("log")
    ax.set_xlim(0.005, 22.0)
    ax.set_ylim(0.0, 1.16)
    ax.set_xlabel("particle diameter $D_p$ ($\\mu$m)", fontsize=8.0)
    ax.set_ylabel("spectrum normalised to its own peak (-)", fontsize=8.0)
    ax.set_title("One aerosol population, three moments", fontsize=9.0,
                 fontweight="bold", color=INK)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", color="#e4e4e4", lw=0.55, zorder=1)

    # ------------------------------------------------- (b) share by mode
    mode_cols = get_palette("cb_sequential_blue", 5)[1:4]
    shares = np.array([
        [np.trapezoid(m, logd) for m in per_mode],
        [np.trapezoid(m, logd) for m in per_S],
        [np.trapezoid(m, logd) for m in per_V],
    ])
    shares = shares / shares.sum(axis=1, keepdims=True) * 100.0

    ypos = np.arange(3)[::-1]
    left = np.zeros(3)
    for k in range(3):
        ax1.barh(ypos, shares[:, k], left=left, height=0.52,
                 color=mode_cols[k], edgecolor="white", linewidth=0.7,
                 zorder=3, label=f"{MODES[k][3]}")
        for yy, x0, w in zip(ypos, left, shares[:, k]):
            if w >= 7.0:
                ax1.text(x0 + w / 2.0, yy, f"{w:.0f}%", ha="center",
                         va="center", fontsize=6.3,
                         color="white" if k >= 1 else INK, zorder=5)
        left = left + shares[:, k]

    ax1.set_yticks(ypos)
    ax1.set_yticklabels(["number", "surface", "volume"], fontsize=7.0)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.62, 2.62)
    ax1.set_xlabel("share of the moment (%)", fontsize=7.5)
    ax1.set_title("Which mode owns which moment", fontsize=8.5,
                  fontweight="bold", color=INK)
    ax1.tick_params(labelsize=6.8)
    ax1.legend(handles=[Patch(facecolor=mode_cols[k], edgecolor="white",
                              label=MODES[k][3]) for k in range(3)],
               loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
               fontsize=6.6, frameon=False, handlelength=1.2)

    # --------------------------------------- (c) characteristic diameters
    n_tot = np.trapezoid(dNdlog, logd)
    cmd = 10.0 ** (np.trapezoid(dNdlog * logd, logd) / n_tot)
    smd = 10.0 ** (np.trapezoid(dSdlog * logd, logd) /
                   np.trapezoid(dSdlog, logd))
    vmd = 10.0 ** (np.trapezoid(dVdlog * logd, logd) /
                   np.trapezoid(dVdlog, logd))
    deff = (np.trapezoid(dNdlog * dp ** 3, logd) /
            np.trapezoid(dNdlog * dp ** 2, logd))

    names = ["count mean\n$D_{CMD}$", "surface mean\n$D_{SMD}$",
             "volume mean\n$D_{VMD}$", "effective\n$D_{eff}$"]
    vals = np.array([cmd, smd, vmd, deff])
    cols = [NUM_C, SUR_C, VOL_C, "#8e6bbf"]
    xp = np.arange(4)
    ax2.bar(xp, vals, width=0.56, color=cols, alpha=0.90, edgecolor="white",
            linewidth=0.7, zorder=3)
    for x, v in zip(xp, vals):
        ax2.text(x, v * 1.16, f"{v:.3g}", ha="center", va="bottom",
                 fontsize=6.3, color=SUB)
    ax2.set_yscale("log")
    ax2.set_ylim(0.02, 30.0)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(names, fontsize=6.5, linespacing=1.25)
    ax2.set_ylabel("diameter ($\\mu$m)", fontsize=7.5)
    ax2.set_title("Characteristic diameters", fontsize=8.5, fontweight="bold",
                  color=INK)
    ax2.tick_params(labelsize=6.8)
    ax2.grid(axis="y", color="#ececec", lw=0.55, zorder=0)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.42, dy=0.14)

    savefig(fig, "553_atm_aerosol_size.png")


if __name__ == "__main__":
    main()
