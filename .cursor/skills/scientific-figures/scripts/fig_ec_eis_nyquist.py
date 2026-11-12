"""Electrochemical impedance of three electrodes, Randles equivalent circuit.

(a) Main panel: Nyquist plot on equal axes - the only way a depressed
    semicircle keeps its shape - with the measured points and the
    equivalent-circuit response drawn as a dashed line through them.  The
    circuit itself is sketched from plain line segments in the empty corner
    that every Nyquist plot leaves above the arc.
(b) Auxiliary panel: the same three data sets as a Bode plot, impedance
    modulus on the left axis and phase angle on the right.
(c) Auxiliary panel: the resistances the fit returns, series and
    charge-transfer side by side on one resistance scale.

Impedances are area-normalised.  Every resistance quoted in the figure is
recovered from the plotted complex arrays.  Illustrative values.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, inch_axes, panel_labels_inches

RNG = np.random.default_rng(76601)

FREQ = np.logspace(-2.0, 5.0, 61)        # Hz
MAIN = (0.90, 1.02, 5.30, 4.62)          # main panel rectangle, inches

CELLS = [dict(name="pristine", rs=6.0, rct=38.0, y0=2.4e-4, n=0.90,
              sigma=26.0, col="#3b5b92"),
         dict(name="aged 200 h", rs=9.0, rct=68.0, y0=1.8e-4, n=0.87,
              sigma=41.0, col="#2e7d5b"),
         dict(name="aged 600 h", rs=12.5, rct=112.0, y0=1.3e-4, n=0.84,
              sigma=58.0, col="#c9a227")]

WIRE = "#333333"
NOTE = "#777777"


def impedance(f, rs, rct, y0, n, sigma):
    """Rs + (CPE || (Rct + semi-infinite Warburg))."""
    w = 2.0 * np.pi * f
    z_w = sigma * (1.0 - 1j) / np.sqrt(w)
    y_cpe = y0 * (1j * w) ** n
    return rs + 1.0 / (y_cpe + 1.0 / (rct + z_w))


def arc_fit(re, im):
    """Kasa circle fit on the charge-transfer arc; returns (Rs, Rct).

    A depressed arc has its centre below the real axis, so the diameter is
    not the chord: the chord that the circle cuts on the axis is the
    charge-transfer resistance and its left end is the series resistance.
    """
    a = np.column_stack([2.0 * re, 2.0 * im, np.ones_like(re)])
    sol, *_ = np.linalg.lstsq(a, re ** 2 + im ** 2, rcond=None)
    xc, yc, cc = sol
    r = np.sqrt(max(cc + xc ** 2 + yc ** 2, 1e-9))
    half = np.sqrt(max(r ** 2 - yc ** 2, 1e-9))
    return float(xc - half), float(2.0 * half)


def resistor(ax, x, y, w=0.30, h=0.13, label="", fs=6.4):
    """Box-style resistor centred on (x, y) of an inch canvas."""
    ax.add_patch(Rectangle((x - w / 2.0, y - h / 2.0), w, h, facecolor="white",
                           edgecolor=WIRE, linewidth=0.8, zorder=4))
    if label:
        ax.text(x, y + h / 2.0 + 0.045, label, ha="center", va="bottom",
                fontsize=fs, color=WIRE, zorder=5)


def capacitor(ax, x, y, gap=0.055, half=0.10, label="", fs=6.4):
    """Two parallel plates, the constant-phase element symbol."""
    for s in (-1, 1):
        ax.plot([x + s * gap, x + s * gap], [y - half, y + half], color=WIRE,
                lw=0.9, zorder=4)
    if label:
        ax.text(x, y + half + 0.045, label, ha="center", va="bottom",
                fontsize=fs, color=WIRE, zorder=5)


def circuit(fig, rect):
    """Randles circuit sketch on its own transparent inch canvas."""
    ax = inch_axes(fig, rect, aspect="equal", zorder=9)
    ax.patch.set_visible(False)
    w, h = rect[2], rect[3]
    mid, top, bot = h * 0.50, h * 0.78, h * 0.22
    ax.plot([0.10, 0.60], [mid, mid], color=WIRE, lw=0.8, zorder=3)
    resistor(ax, 0.80, mid, label="$R_s$")
    ax.plot([1.00, 1.20], [mid, mid], color=WIRE, lw=0.8, zorder=3)
    ax.plot([1.20, 1.20], [bot, top], color=WIRE, lw=0.8, zorder=3)
    ax.plot([1.20, 1.60], [top, top], color=WIRE, lw=0.8, zorder=3)
    capacitor(ax, 1.75, top, label="CPE")
    ax.plot([1.90, 2.30], [top, top], color=WIRE, lw=0.8, zorder=3)
    ax.plot([1.20, 1.35], [bot, bot], color=WIRE, lw=0.8, zorder=3)
    resistor(ax, 1.55, bot, label="")
    ax.text(1.55, bot - 0.11, "$R_{ct}$", ha="center", va="top", fontsize=6.4,
            color=WIRE)
    ax.plot([1.75, 1.85], [bot, bot], color=WIRE, lw=0.8, zorder=3)
    resistor(ax, 2.05, bot, w=0.24, label="")
    ax.text(2.05, bot - 0.11, "$W$", ha="center", va="top", fontsize=6.4,
            color=WIRE)
    ax.plot([2.17, 2.30], [bot, bot], color=WIRE, lw=0.8, zorder=3)
    ax.plot([2.30, 2.30], [bot, top], color=WIRE, lw=0.8, zorder=3)
    ax.plot([2.30, 2.80], [mid, mid], color=WIRE, lw=0.8, zorder=3)
    ax.text(0.02, 0.02, "equivalent circuit", ha="left", va="bottom",
            fontsize=6.5, color=NOTE)
    return ax


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, MAIN)
    axs = axes_inches(fig, (7.42, 3.74, 2.66, 1.90))
    axb = axes_inches(fig, (7.42, 1.02, 2.66, 1.90))

    for c in CELLS:
        z = impedance(FREQ, c["rs"], c["rct"], c["y0"], c["n"], c["sigma"])
        scale = 1.0 + RNG.normal(0.0, 0.012, FREQ.size)
        c["z_fit"] = z
        c["z"] = z.real * scale + 1j * z.imag * (2.0 - scale)
        im = -c["z"].imag
        hi_band = FREQ > 1.0
        c["f_apex"] = float(FREQ[hi_band][int(np.argmax(im[hi_band]))])
        c["z_apex"] = c["z_fit"][hi_band][int(np.argmax(
            -c["z_fit"].imag[hi_band]))]
        sel = FREQ >= c["f_apex"] / 3.0
        c["rs_fit"], c["rct_fit"] = arc_fit(c["z"].real[sel], im[sel])

    # ---------------- (a) Nyquist ----------------------------------------
    for c in CELLS:
        ax.plot(c["z_fit"].real, -c["z_fit"].imag, color=c["col"], lw=1.0,
                ls=(0, (3.2, 2.2)), alpha=0.9, zorder=3)
        ax.plot(c["z"].real, -c["z"].imag, ls="none", marker="o", ms=3.6,
                mfc="white", mec=c["col"], mew=0.85, zorder=5,
                label="%s,  $R_s$ = %.1f,  $R_{ct}$ = %.0f $\\Omega$ cm$^2$"
                      % (c["name"], c["rs_fit"], c["rct_fit"]))
        ax.plot([c["rs_fit"]], [0.0], marker="s", ms=4.0, mfc="white",
                mec=c["col"], mew=1.0, zorder=7)

    worst = CELLS[-1]
    z_apex = worst["z_apex"]
    ax.annotate("apex at %.0f Hz\n$R_{ct}$ = %.0f $\\Omega$ cm$^2$"
                % (worst["f_apex"], worst["rct_fit"]),
                xy=(z_apex.real, -z_apex.imag),
                xytext=(z_apex.real - 10.0, -z_apex.imag + 58.0),
                fontsize=6.8, color=worst["col"], ha="right", va="center",
                linespacing=1.40,
                arrowprops=dict(arrowstyle="-|>", color=worst["col"], lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    tail = worst["z_fit"][0]
    ax.annotate("45$^\\circ$ Warburg tail,\ndiffusion at %.0f mHz"
                % (FREQ[0] * 1000.0), xy=(tail.real, -tail.imag),
                xytext=(tail.real - 52.0, -tail.imag + 4.0), fontsize=6.8,
                color=NOTE, ha="right", va="center", linespacing=1.40,
                arrowprops=dict(arrowstyle="-|>", color=NOTE, lw=0.8,
                                mutation_scale=8, shrinkB=3.0))

    re_all = np.concatenate([c["z"].real for c in CELLS])
    im_all = np.concatenate([-c["z"].imag for c in CELLS])
    x0, x1 = -6.0, float(re_all.max()) * 1.05
    y0, y1 = -6.0, float(im_all.max()) * 1.08
    want = MAIN[2] / MAIN[3]
    if (x1 - x0) / (y1 - y0) > want:            # widen whichever is short
        y1 = y0 + (x1 - x0) / want
    else:
        x1 = x0 + (y1 - y0) * want
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("$Z'$ ($\\Omega$ cm$^2$)")
    ax.set_ylabel("$-Z''$ ($\\Omega$ cm$^2$)")
    ax.tick_params(labelsize=7.2)
    ax.grid(color="#f2f2f2", lw=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=7.0, handlelength=1.2,
              labelspacing=0.32, borderaxespad=0.6)
    ax.text(0.975, 0.235, "circles: measurement,  dashed: circuit fit\n"
            "squares: high-frequency intercept $R_s$", transform=ax.transAxes,
            fontsize=6.5, color=NOTE, ha="right", va="top", linespacing=1.45)
    circuit(fig, (1.24, 4.34, 2.80, 0.98))

    # ---------------- (b) Bode -------------------------------------------
    axt = axs.twinx()
    axt.spines["top"].set_visible(False)
    for c in CELLS:
        axs.plot(FREQ, np.abs(c["z_fit"]), color=c["col"], lw=1.15, zorder=4)
        axt.plot(FREQ, -np.angle(c["z_fit"], deg=True), color=c["col"],
                 lw=0.9, ls=(0, (2.6, 2.0)), alpha=0.85, zorder=3)
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlim(FREQ.min(), FREQ.max())
    axs.set_xlabel("Frequency (Hz)", fontsize=8, labelpad=1.5)
    axs.set_ylabel("$|Z|$ ($\\Omega$ cm$^2$)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5, which="both")
    axs.grid(color="#ededed", lw=0.5, zorder=0)
    axt.set_ylim(0.0, 90.0)
    axt.set_yticks([0, 30, 60, 90])
    axt.set_ylabel("$-$phase (deg)", fontsize=8, labelpad=2.0, color="#8e6bbf")
    axt.tick_params(axis="y", labelsize=6.5, colors="#8e6bbf")
    axs.legend(handles=[
        Line2D([], [], color=WIRE, lw=1.15, label="$|Z|$, left axis"),
        Line2D([], [], color=WIRE, lw=0.9, ls=(0, (2.6, 2.0)),
               label="phase, right axis")],
        loc="lower left", fontsize=6.3, handlelength=1.4, labelspacing=0.28,
        borderaxespad=0.3)

    # ---------------- (c) fitted resistances ------------------------------
    xs = np.arange(len(CELLS))
    rs_v = np.array([c["rs_fit"] for c in CELLS])
    rct_v = np.array([c["rct_fit"] for c in CELLS])
    axb.bar(xs - 0.17, rs_v, width=0.30, color="#b8c4d6", edgecolor="white",
            linewidth=0.6, zorder=3, label="$R_s$")
    axb.bar(xs + 0.17, rct_v, width=0.30, color=[c["col"] for c in CELLS],
            edgecolor="white", linewidth=0.6, zorder=3,
            label="$R_{ct}$, colour as in (a)")
    for x, v in zip(xs - 0.17, rs_v):
        axb.text(x, v + 3.0, "%.1f" % v, fontsize=6.2, color="#555555",
                 ha="center", va="bottom", zorder=4)
    for x, v in zip(xs + 0.17, rct_v):
        axb.text(x, v + 3.0, "%.0f" % v, fontsize=6.2, color="#555555",
                 ha="center", va="bottom", zorder=4)
    axb.set_xticks(xs)
    axb.set_xticklabels([c["name"] for c in CELLS], fontsize=6.6)
    axb.set_ylim(0.0, rct_v.max() * 1.34)
    axb.set_xlabel("Resistances returned by the fit", fontsize=8, labelpad=1.5)
    axb.set_ylabel("$R$ ($\\Omega$ cm$^2$)", fontsize=8, labelpad=1.5)
    axb.tick_params(axis="y", labelsize=6.5)
    axb.tick_params(axis="x", length=0)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    axb.legend(loc="upper left", fontsize=6.4, handlelength=1.0,
               labelspacing=0.28, borderaxespad=0.3, ncol=2)

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.40, dy=0.10)

    savefig(fig, "766_ec_eis_nyquist.png")


if __name__ == "__main__":
    main()
