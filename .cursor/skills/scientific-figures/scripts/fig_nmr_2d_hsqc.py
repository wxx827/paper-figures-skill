"""Multiplicity-edited 1H-13C HSQC of an acylated phenylpropanoid glycoside.

One two-dimensional array carries the whole figure.  Thirteen one-bond
correlations are laid down as 2D Gaussians on a 1H x 13C grid, CH2 groups with
inverted sign so the multiplicity editing that a real gradient-selected HSQC
gives for free is visible as a colour change rather than as a caption.  White
noise is added once, and every later number - peak volume, signal-to-noise,
the digital resolution quoted in the footnote - is read back out of that same
array instead of being typed in.

Both chemical-shift axes run high-to-low.  That is not a stylistic choice: a
shift axis is a frequency axis referenced to TMS at zero, and every spectrum
ever printed in the field increases to the left, so a forward axis makes the
plot unreadable to the only people who would ever read it.

The two marginal traces are skyline projections of the same matrix, not
separately simulated 1D spectra, so a cross peak that is missing from the
contour plot is missing from the projection too.  On the right, three 1H
slices cut through the matrix at fixed 13C, and the table lists the volumes
obtained by integrating each cross peak over a fixed window.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches, text_card

RED = "#b23a48"          # house red: positive (CH / CH3) contours
BLUE = "#3b5b92"         # house blue: negative (CH2) contours
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

# 1H shift (ppm), 13C shift (ppm), amplitude, multiplicity, assignment
PEAKS = [
    (7.52, 145.2, 0.52, "CH", "H-7 / C-7"),
    (7.28, 128.9, 0.88, "CH", "H-2,6 aryl"),
    (6.86, 116.1, 0.70, "CH", "H-3,5 aryl"),
    (6.42, 121.5, 0.44, "CH", "H-8 / C-8"),
    (5.42, 101.8, 1.00, "CH", "H-1' anomeric"),
    (3.95, 73.6, 0.86, "CH", "H-2' / C-2'"),
    (3.86, 56.1, 0.80, "CH3", "Ar-OCH3"),
    (3.79, 61.5, 0.92, "CH2", "H-6' / C-6'"),
    (3.62, 76.4, 0.78, "CH", "H-3' / C-3'"),
    (3.48, 70.2, 0.74, "CH", "H-4' / C-4'"),
    (2.62, 38.4, 0.66, "CH2", "H-9 / C-9"),
    (2.05, 21.0, 0.95, "CH3", "OAc CH3"),
    (1.28, 14.2, 0.60, "CH3", "H-12 / C-12"),
]

# assignment -> (d1H offset in ppm, d13C offset in ppm, horizontal alignment).
# Hand-placed so no caption sits on another caption or on its own contour.
# Note the alignment is in *screen* space while the shift axis runs backwards,
# so ha="right" grows the caption toward higher ppm; the offsets below were
# chosen with that inversion in mind and every one stays inside the frame.
LABEL_OFF = {
    "H-7 / C-7": (0.0, -8.0, "center"),
    "H-2,6 aryl": (0.0, -8.0, "center"),
    "H-3,5 aryl": (0.0, -8.0, "center"),
    "H-8 / C-8": (0.0, 8.5, "center"),
    "H-1' anomeric": (0.0, -8.5, "center"),
    "H-2' / C-2'": (0.78, 0.0, "right"),
    "Ar-OCH3": (0.0, -8.5, "center"),
    "H-6' / C-6'": (-0.92, -6.0, "left"),
    "H-3' / C-3'": (0.0, 9.0, "center"),
    "H-4' / C-4'": (-0.92, 4.5, "left"),
    "H-9 / C-9": (0.0, -8.5, "center"),
    "OAc CH3": (0.58, 0.0, "right"),
    "H-12 / C-12": (0.0, 9.5, "center"),
}

# 13C shifts at which the right-hand panel cuts a 1H row out of the matrix
SLICES = [101.8, 61.5, 21.0]
TABLE_ROWS = ["H-1' anomeric", "H-2,6 aryl", "Ar-OCH3", "H-6' / C-6'",
              "OAc CH3", "H-12 / C-12"]

H_LO, H_HI = 0.55, 8.15        # 1H window (ppm)
C_LO, C_HI = 8.0, 152.0        # 13C window (ppm)
N_H, N_C = 560, 620
W_H, W_C = 0.068, 1.55         # Gaussian half-widths (ppm)
SIGMA = 0.0045                 # rms noise, same units as the amplitudes
SF_H, SF_C = 500.13, 125.77    # spectrometer frequencies (MHz)
BASE_LEVEL, LEVEL_STEP, N_LEVEL = 0.055, 1.80, 6

FIG_W, FIG_H = 11.0, 6.5
LEFP = (0.42, 1.30, 0.46, 3.62)
MAIN = (0.96, 1.30, 5.10, 3.62)
TOPP = (0.96, 5.04, 5.10, 0.46)
AUX_T = (7.66, 3.62, 2.92, 1.88)
AUX_B = (7.66, 1.28, 2.92, 1.72)


def build_matrix(rng):
    """The one array the whole figure reads from."""
    h = np.linspace(H_LO, H_HI, N_H)
    c = np.linspace(C_LO, C_HI, N_C)
    hh, cc = np.meshgrid(h, c)
    z = np.zeros_like(hh)
    for h0, c0, amp, mult, _lab in PEAKS:
        sign = -1.0 if mult == "CH2" else 1.0
        z += sign * amp * np.exp(-0.5 * ((hh - h0) / W_H) ** 2
                                 - 0.5 * ((cc - c0) / W_C) ** 2)
    return h, c, z + rng.normal(0.0, SIGMA, z.shape)


def volumes(h, c, z):
    """Integrate |z| over a fixed +/-2.5 half-width box around every peak."""
    dh = float(h[1] - h[0])
    dc = float(c[1] - c[0])
    out = {}
    for h0, c0, _a, _m, lab in PEAKS:
        jh = np.abs(h - h0) <= 2.5 * W_H
        jc = np.abs(c - c0) <= 2.5 * W_C
        out[lab] = float(np.abs(z[np.ix_(jc, jh)]).sum() * dh * dc)
    return out


def contour_panel(ax, h, c, z):
    pos = BASE_LEVEL * LEVEL_STEP ** np.arange(N_LEVEL)
    ax.contour(h, c, z, levels=pos, colors=[RED], linewidths=0.48,
               zorder=4)
    ax.contour(h, c, z, levels=-pos[::-1], colors=[BLUE], linewidths=0.48,
               zorder=4)

    for h0, c0, _a, mult, lab in PEAKS:
        dh, dc, ha = LABEL_OFF[lab]
        col = BLUE if mult == "CH2" else RED
        ax.plot([h0, h0 + dh * 0.80], [c0, c0 + dc * 0.62], lw=0.5,
                color=lighten(col, 0.55), zorder=2)
        ax.text(h0 + dh, c0 + dc, lab, ha=ha, va="center", fontsize=6.0,
                color=col, zorder=7)

    ax.set_xlim(H_HI, H_LO)          # 1H high-to-low: the field convention
    ax.set_ylim(C_HI, C_LO)          # 13C high-to-low, likewise
    ax.set_xlabel(r"$\delta$ $^{1}$H (ppm)")
    ax.set_ylabel(r"$\delta$ $^{13}$C (ppm)")
    ax.set_xticks(np.arange(1.0, 8.1, 1.0))
    ax.set_yticks(np.arange(20.0, 151.0, 20.0))
    ax.grid(lw=0.45, color=HAIR, alpha=0.75)
    # The 13C scale is read on the right, as the acquisition software prints
    # it, which also keeps it clear of the 13C projection down the left edge.
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    for side in ("top", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["right"].set_visible(True)

    handles = [Line2D([], [], color=RED, lw=1.1, label="CH / CH$_3$ (positive)"),
               Line2D([], [], color=BLUE, lw=1.1, label="CH$_2$ (negative)")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.4,
                    frameon=True, handlelength=1.4, borderpad=0.34,
                    borderaxespad=0.35, labelspacing=0.30)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(HAIR)
    leg.get_frame().set_linewidth(0.6)


def projection_panels(ax_top, ax_left, h, c, z):
    """Skyline projections of the same matrix, positive and negative."""
    ph, nh = z.max(axis=0), z.min(axis=0)
    pc, nc = z.max(axis=1), z.min(axis=1)

    ax_top.plot(h, ph, color=RED, lw=0.7)
    ax_top.plot(h, nh, color=BLUE, lw=0.7)
    ax_top.set_xlim(H_HI, H_LO)
    ax_top.set_ylim(-0.55, 1.18)

    ax_left.plot(pc, c, color=RED, lw=0.7)
    ax_left.plot(nc, c, color=BLUE, lw=0.7)
    ax_left.set_xlim(1.18, -0.55)     # intensity grows toward the left margin
    ax_left.set_ylim(C_HI, C_LO)

    for ax in (ax_top, ax_left):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax_top.text(H_HI - 0.05, 1.10, "skyline projections", ha="left", va="top",
                fontsize=6.2, color=MUTED)


def slice_panel(ax, h, c, z):
    step = 2.35
    for k, c0 in enumerate(SLICES):
        i = int(np.argmin(np.abs(c - c0)))
        row = z[i]
        base = (len(SLICES) - 1 - k) * step
        peak = [p for p in PEAKS if abs(p[1] - c0) < 1e-6][0]
        col = BLUE if peak[3] == "CH2" else RED
        ax.plot(h, base + row, color=col, lw=0.75, zorder=4)
        ax.axhline(base, color=HAIR, lw=0.5, zorder=2)
        ax.text(H_HI - 0.08, base + 0.10,
                r"$\delta_{\mathrm{C}}$ = " + "{:.1f} ppm".format(c0),
                ha="left", va="bottom", fontsize=6.0, color=col)
        j = int(np.argmax(np.abs(row)))
        ax.text(h[j] - 0.28, base + row[j] * 0.55,
                "{:.2f} ppm".format(h[j]), ha="right", va="center",
                fontsize=6, color=SUB)

    ax.set_xlim(H_HI, H_LO)
    ax.set_ylim(-1.25, (len(SLICES) - 1) * step + 1.45)
    ax.set_xticks(np.arange(1.0, 8.1, 2.0))
    ax.set_yticks([])
    ax.set_xlabel(r"$\delta$ $^{1}$H (ppm)", fontsize=7.4)
    ax.grid(axis="x", lw=0.45, alpha=0.55)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Row slices at fixed $^{13}$C, intensity in a.u.",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)


def table_panel(ax, w, h, vol):
    total = sum(vol[k] for k in TABLE_ROWS)
    ax.text(0.0, h, "Cross-peak assignment and integrated volume",
            ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)

    top = h - 0.30
    text_card(ax, 0.0, 0.02, w, top - 0.02, "", fc=SOFT, ec=HAIR, rounding=0.05,
              zorder=1)
    cols = (0.10, w - 1.30, w - 0.74, w - 0.10)
    head = ("assignment", r"$\delta_{\mathrm{H}}$ (ppm)",
            r"$\delta_{\mathrm{C}}$ (ppm)", "V (%)")
    for x, s, ha in zip(cols, head, ("left", "right", "right", "right")):
        ax.text(x, top - 0.16, s, ha=ha, va="center", fontsize=6.0,
                color=SUB, fontweight="bold", zorder=4)
    ax.plot([0.06, w - 0.06], [top - 0.28] * 2, color=HAIR, lw=0.7, zorder=3)

    rows = [p for p in PEAKS if p[4] in TABLE_ROWS]
    dy = (top - 0.40) / len(rows)
    for k, (h0, c0, _a, mult, lab) in enumerate(rows):
        y = top - 0.40 - (k + 0.5) * dy + dy * 0.5
        col = BLUE if mult == "CH2" else RED
        if k % 2 == 0:
            ax.add_patch(plt.Rectangle((0.05, y - dy * 0.46), w - 0.10,
                                       dy * 0.92, facecolor="white",
                                       edgecolor="none", zorder=2))
        vals = (lab, "{:.2f}".format(h0), "{:.1f}".format(c0),
                "{:.1f}".format(100.0 * vol[lab] / total))
        for x, s, ha in zip(cols, vals, ("left", "right", "right", "right")):
            ax.text(x, y, s, ha=ha, va="center", fontsize=6.1,
                    color=col if ha == "left" else INK, zorder=4)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(936)
    h, c, z = build_matrix(rng)
    vol = volumes(h, c, z)

    noise = float(np.std(z[(c > 130.0) & (c < 152.0)][:, h < 4.4]))
    weakest = min(p[2] for p in PEAKS)
    snr = weakest / max(noise, 1e-12)
    res_h = (H_HI - H_LO) * SF_H / N_H
    res_c = (C_HI - C_LO) * SF_C / N_C

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_top = axes_inches(fig, TOPP)
    ax_left = axes_inches(fig, LEFP)
    ax_sl = axes_inches(fig, AUX_T)
    ax_tb = inch_axes(fig, AUX_B, aspect=None)
    ax_tb.set_xlim(0.0, AUX_B[2])
    ax_tb.set_ylim(0.0, AUX_B[3])

    contour_panel(ax_main, h, c, z)
    projection_panels(ax_top, ax_left, h, c, z)
    slice_panel(ax_sl, h, c, z)
    table_panel(ax_tb, AUX_B[2], AUX_B[3], vol)

    n_ch2 = sum(1 for p in PEAKS if p[3] == "CH2")
    panel_labels_inches(fig, [ax_left, ax_sl], labels=["(a)", "(b)"],
                        dx=-0.10, dy=0.14)
    panel_labels_inches(fig, [ax_tb], labels=["(c)"], dx=-0.10, dy=0.02)
    savefig(fig, "936_nmr_2d_hsqc.png")


if __name__ == "__main__":
    main()
