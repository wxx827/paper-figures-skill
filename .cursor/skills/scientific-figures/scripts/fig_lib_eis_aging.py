"""EIS of an NMC811 pouch cell at three cycle counts, 50 % SOC, 25 °C.

(a) Nyquist plot on equal axes so the depressed charge-transfer arc keeps
    its geometry.  Markers are the measured points; dashed lines are the
    Randles-with-Warburg response that generated them plus a small noise.
(b) Bode modulus of the same three spectra.
(c) Series and charge-transfer resistances recovered from a circle fit of
    the high-frequency arc of each plotted spectrum.

Illustrative values.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SEED = 1214
FREQ = np.logspace(-2.0, 5.0, 61)

CELLS = [
    dict(name="fresh", n=0, rs=2.4, rct=8.6, y0=3.1e-3, n_cpe=0.91,
         sigma=6.5, col=BLUE),
    dict(name="200 cycles", n=200, rs=3.1, rct=14.8, y0=2.4e-3, n_cpe=0.88,
         sigma=9.2, col=GREEN),
    dict(name="500 cycles", n=500, rs=4.6, rct=27.4, y0=1.7e-3, n_cpe=0.85,
         sigma=13.8, col=GOLD),
]


def impedance(f, rs, rct, y0, n_cpe, sigma):
    w = 2.0 * np.pi * f
    z_w = sigma * (1.0 - 1j) / np.sqrt(w)
    y_cpe = y0 * (1j * w) ** n_cpe
    return rs + 1.0 / (y_cpe + 1.0 / (rct + z_w))


def arc_fit(re, im):
    a = np.column_stack([2.0 * re, 2.0 * im, np.ones_like(re)])
    sol, *_ = np.linalg.lstsq(a, re ** 2 + im ** 2, rcond=None)
    xc, yc, cc = sol
    r = np.sqrt(max(cc + xc ** 2 + yc ** 2, 1e-9))
    half = np.sqrt(max(r ** 2 - yc ** 2, 1e-9))
    return float(xc - half), float(2.0 * half)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    spectra = []
    for cell in CELLS:
        z = impedance(FREQ, cell["rs"], cell["rct"], cell["y0"],
                      cell["n_cpe"], cell["sigma"])
        z = z + rng.normal(0, 0.08, z.size) + 1j * rng.normal(0, 0.08, z.size)
        spectra.append(z)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.90, 1.02, 5.30, 4.62))
    axb = axes_inches(fig, (7.42, 3.74, 2.66, 1.90))
    axc = axes_inches(fig, (7.42, 1.02, 2.66, 1.90))

    for cell, z in zip(CELLS, spectra):
        z0 = impedance(FREQ, cell["rs"], cell["rct"], cell["y0"],
                       cell["n_cpe"], cell["sigma"])
        ax.plot(z.real, -z.imag, ls="none", marker="o", ms=3.6, mfc="white",
                mec=cell["col"], mew=0.7, zorder=4, label=cell["name"])
        ax.plot(z0.real, -z0.imag, color=cell["col"], lw=1.15,
                ls=(0, (3.4, 1.8)), zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("Z' ($\\Omega$ cm$^{2}$)")
    ax.set_ylabel("$-$Z'' ($\\Omega$ cm$^{2}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    ax.tick_params(labelsize=7.2)
    ax.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Nyquist, 50 % SOC, 25 $^\\circ$C", fontsize=9.0,
                 fontweight="bold", pad=6)

    for cell, z in zip(CELLS, spectra):
        axb.loglog(FREQ, np.abs(z), color=cell["col"], lw=1.4)
    axb.set_xlabel("Frequency (Hz)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("|$Z$| ($\\Omega$ cm$^{2}$)", fontsize=8, labelpad=1.5)
    axb.tick_params(labelsize=6.5)
    axb.grid(color="#ededed", lw=0.5, which="both", zorder=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("Bode modulus", fontsize=8.5, fontweight="bold", pad=4)

    rs_fit, rct_fit = [], []
    for z in spectra:
        hi = FREQ > 20.0
        rs_i, rct_i = arc_fit(z.real[hi], -z.imag[hi])
        rs_fit.append(rs_i)
        rct_fit.append(rct_i)
    x = np.arange(len(CELLS))
    w = 0.36
    axc.bar(x - 0.5 * w, rs_fit, w, color=BLUE, label="$R_s$")
    axc.bar(x + 0.5 * w, rct_fit, w, color=GOLD, label="$R_{ct}$")
    axc.set_xticks(x)
    axc.set_xticklabels([c["name"].replace(" ", "\n") for c in CELLS],
                        fontsize=6.4)
    axc.set_ylabel("Resistance ($\\Omega$ cm$^{2}$)", fontsize=8, labelpad=1.5)
    axc.legend(loc="upper left", frameon=False, fontsize=6.8)
    axc.tick_params(labelsize=6.5)
    axc.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Arc-fit resistances", fontsize=8.5, fontweight="bold",
                  pad=4)
    axc.text(0.97, 0.96,
             "$R_{{ct}}$(500)/$R_{{ct}}$(0) = {:.1f}".format(
                 rct_fit[-1] / rct_fit[0]),
             transform=axc.transAxes, ha="right", va="top", fontsize=6.4,
             color=INK)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1214_lib_eis_aging.png")


if __name__ == "__main__":
    main()
