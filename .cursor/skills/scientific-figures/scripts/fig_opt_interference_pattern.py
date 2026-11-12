"""Six interference and diffraction patterns on one shared intensity scale.

The grid is the claim: the same 633 nm source produces six completely
different fringe geometries, and the only reason they can be compared at all
is that every tile is rendered with one normalisation and one colour bar.
Under each tile a thin strip carries the horizontal cut through the pattern
centre, which is where the quoted pitch, lobe width and ring radius are
measured from - each number is found by peak / zero detection on the plotted
cut, never typed in.

Layout is inch-based (``figkit.axes_inches``) rather than gridspec so that the
tile / strip pairs keep the same physical gap in both rows.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j1

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, image_axes, corner_stamp, panel_labels_inches

RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
RAMP = get_cmap("cb_sequential_blue")

LAM_NM = 633.0
LAM_MM = LAM_NM * 1e-6
L_MM = 1000.0            # aperture-to-screen distance

FW, FH = 10.4, 6.4
TW, TH = 2.35, 1.55      # tile size in inches
PH = 0.42                # profile strip height in inches
ASPECT = TW / TH
NX, NY = 460, 304


def grid(xmax):
    """Screen grid in mm whose extent aspect matches the tile aspect."""
    ymax = xmax / ASPECT
    x = np.linspace(-xmax, xmax, NX)
    y = np.linspace(-ymax, ymax, NY)
    return x, y, np.meshgrid(x, y)


def sinc2(u):
    return np.sinc(u) ** 2


def pat_double(xmax=6.0, d=0.45, a=0.12):
    x, y, (X, _) = grid(xmax)
    prof = np.cos(np.pi * d * x / (LAM_MM * L_MM)) ** 2 * \
        sinc2(a * x / (LAM_MM * L_MM))
    img = np.cos(np.pi * d * X / (LAM_MM * L_MM)) ** 2 * \
        sinc2(a * X / (LAM_MM * L_MM))
    pk, _ = find_peaks(prof, height=0.05)
    pitch = float(np.median(np.diff(x[pk])))
    return img, x, y, prof, f"pitch {pitch:.2f} mm"


def pat_single(xmax=6.0, a=0.20):
    x, y, (X, _) = grid(xmax)
    prof = sinc2(a * x / (LAM_MM * L_MM))
    img = sinc2(a * X / (LAM_MM * L_MM))
    zero = LAM_MM * L_MM / a
    half = x[(x > 0) & (prof < 0.5)][0] * 2.0
    return img, x, y, prof, f"lobe {2 * zero:.2f} mm,  FWHM {half:.2f} mm"


def pat_airy(xmax=6.0, dia=0.40):
    x, y, (X, Y) = grid(xmax)
    def airy(r):
        v = np.pi * dia * np.maximum(r, 1e-9) / (LAM_MM * L_MM)
        return (2.0 * j1(v) / v) ** 2
    img = airy(np.hypot(X, Y))
    prof = airy(np.abs(x))
    r_airy = 1.22 * LAM_MM * L_MM / dia
    return img, x, y, prof, f"Airy radius {r_airy:.2f} mm"


def pat_grating(xmax=6.0, d=0.45, a=0.10, n=6):
    x, y, (X, _) = grid(xmax)
    def inten(u):
        delta = np.pi * d * u / (LAM_MM * L_MM)
        num = np.sin(n * delta)
        den = np.where(np.abs(np.sin(delta)) < 1e-9, 1e-9, np.sin(delta))
        base = np.where(np.abs(np.sin(delta)) < 1e-9, n ** 2,
                        (num / den) ** 2)
        return base / n ** 2 * sinc2(a * u / (LAM_MM * L_MM))
    img, prof = inten(X), inten(x)
    pk, _ = find_peaks(prof, height=0.15)
    pitch = float(np.median(np.diff(x[pk])))
    return img, x, y, prof, f"order pitch {pitch:.2f} mm,  $N$ = {n}"


def pat_newton(xmax=2.5, r_lens=2000.0):
    x, y, (X, Y) = grid(xmax)
    def inten(r2):
        return np.cos(np.pi * r2 / (LAM_MM * r_lens)) ** 2
    img = inten(X ** 2 + Y ** 2)
    prof = inten(x ** 2)
    pk, _ = find_peaks(prof[x > 0], height=0.9)
    r1 = float(x[x > 0][pk[0]])
    return img, x, y, prof, f"1st bright ring {r1:.2f} mm"


def pat_michelson(xmax=4.0, pitch=0.90, tilt_deg=18.0):
    x, y, (X, Y) = grid(xmax)
    t = np.deg2rad(tilt_deg)
    phase = (X * np.cos(t) + Y * np.sin(t)) / pitch
    img = np.cos(np.pi * phase) ** 2 * (1.0 - 0.10 * (X / xmax) ** 2)
    prof = np.cos(np.pi * x * np.cos(t) / pitch) ** 2 * \
        (1.0 - 0.10 * (x / xmax) ** 2)
    pk, _ = find_peaks(prof, height=0.5)
    sp = float(np.median(np.diff(x[pk])))
    return img, x, y, prof, f"cut pitch {sp:.2f} mm,  tilt {tilt_deg:.0f} deg"


PATTERNS = [
    ("Double slit", pat_double),
    ("Single slit", pat_single),
    ("Circular aperture", pat_airy),
    ("Transmission grating", pat_grating),
    ("Newton's rings", pat_newton),
    ("Michelson wedge", pat_michelson),
]


def main():
    set_pub_style()
    data = [(name, f()) for name, f in PATTERNS]

    fig = plt.figure(figsize=(FW, FH))
    col_x = [0.70, 3.90, 7.10]
    row_tile_y = [4.35, 1.53]
    row_prof_y = [3.78, 0.96]

    # One norm object for all six tiles; the power-law stretch keeps weak
    # side lobes visible without letting any tile pick its own scale.
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)

    tiles, im = [], None
    for k, (name, (img, x, y, prof, note)) in enumerate(data):
        r, c = divmod(k, 3)
        ax = axes_inches(fig, (col_x[c], row_tile_y[r], TW, TH))
        im = ax.imshow(img, origin="lower", cmap=RAMP, norm=norm,
                       extent=[x[0], x[-1], y[0], y[-1]],
                       interpolation="bilinear")
        image_axes(ax, color="#8a8a8a", lw=0.6)
        ax.set_title(name, fontsize=8.5, fontweight="bold", color=INK, pad=3.0)
        corner_stamp(ax, note, loc="lower left", fontsize=6.2)
        tiles.append(ax)

        axp = axes_inches(fig, (col_x[c], row_prof_y[r], TW, PH))
        axp.fill_between(x, 0.0, prof, color=RED, alpha=0.18, linewidth=0.0)
        axp.plot(x, prof, color=RED, lw=0.9)
        axp.set_xlim(x[0], x[-1])
        axp.set_ylim(0.0, 1.10)
        axp.set_yticks([0.0, 1.0])
        axp.tick_params(labelsize=6.0, length=2.2)
        axp.set_xlabel("screen coordinate $x$ (mm)", fontsize=6.8,
                       labelpad=1.5)
        axp.set_ylabel("$I$ (-)", fontsize=6.8, labelpad=1.5)
        axp.grid(axis="y", lw=0.4, alpha=0.6)

    cax = fig.add_axes([9.62 / FW, row_tile_y[1] / FH, 0.13 / FW,
                        (row_tile_y[0] + TH - row_tile_y[1]) / FH])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.05, 0.2, 0.5, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")
    cb.set_label("normalised intensity (-)", fontsize=6.8, color=INK)

    panel_labels_inches(fig, tiles, dx=-0.34, dy=0.02)
    savefig(fig, "697_opt_interference_pattern.png")


if __name__ == "__main__":
    main()
