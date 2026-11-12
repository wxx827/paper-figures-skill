"""Calibration curve, detection limits and matrix effect for a trace method.

Ten calibration levels in triplicate and ten blank replicates are drawn from
one heteroscedastic noise model, and then every number on the page is
recovered from those points rather than declared.  The regression gives the
slope, the intercept and R squared; the standard deviation of the blanks
divided by that slope gives the limit of detection at three sigma and the
limit of quantitation at ten sigma, and the vertical lines in the magnified
inset are drawn at those computed values, so the shaded bands cannot drift
away from the arithmetic in the caption.

Three spiked samples are laid over the curve and back-calculated through the
same regression, which is the only honest way to report a recovery.  The
residual panel is the evidence for using a straight line at all - structure
there, rather than scatter, is what says the range has been pushed too far -
and the lower panel regresses two further matrices to show how much signal
suppression shifts the slope, which is the reason a method is calibrated in
matrix and not in solvent.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"          # house red: the fitted line
DATA = "#3a3a3a"         # standards
PURPLE = "#8e6bbf"       # spiked recovery samples
ZOOM = "#c9a227"         # the magnifier pairing colour
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BROWN = "#c0603f"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

LEVELS = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0])
N_REP = 3
N_BLANK = 10
SLOPE_TRUE = 152.0       # a.u. per ug/L
INTERCEPT_TRUE = 240.0   # a.u.
SIGMA_BLANK = 105.0      # a.u.
REL_NOISE = 0.012        # proportional term of the noise model

# nominal spike (ug/L), true recovery applied by the simulated sample prep
SPIKES = [(15.0, 0.965), (60.0, 1.032), (120.0, 0.991)]
# Captions live in the empty triangle below the calibration line, the only
# region of the panel that the inset, the legend and the statistics card
# between them do not claim.
SPIKE_LABEL = [(38.0, 1500.0), (86.0, 6600.0), (142.0, 15200.0)]

# matrix name, colour, signal-suppression factor on the slope
MATRICES = [("neat solvent", BLUE, 1.000),
            ("river water", GREEN, 0.882),
            ("wastewater effluent", BROWN, 0.714)]

C_HI = 215.0
ZOOM_HI = 12.0

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.30, 5.55, 4.20)
AUX_T = (7.42, 3.62, 3.18, 1.88)
AUX_B = (7.42, 1.28, 3.18, 1.72)


def respond(conc, rng, slope=SLOPE_TRUE, intercept=INTERCEPT_TRUE):
    """Heteroscedastic instrument response: a floor plus a proportional term."""
    mean = intercept + slope * np.asarray(conc, dtype=float)
    sigma = np.hypot(SIGMA_BLANK, REL_NOISE * mean)
    return mean + rng.normal(0.0, 1.0, np.shape(mean)) * sigma


def regress(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return slope, intercept, 1.0 - ss_res / ss_tot, y - pred


def build(rng):
    conc = np.repeat(LEVELS, N_REP)
    resp = respond(conc, rng)
    blanks = respond(np.zeros(N_BLANK), rng)
    slope, intercept, r2, resid = regress(conc, resp)

    sd_blank = float(np.std(blanks, ddof=1))
    lod = 3.0 * sd_blank / slope
    loq = 10.0 * sd_blank / slope

    spikes = []
    for nominal, recovery in SPIKES:
        y = respond(nominal * recovery, rng)
        found = (float(y) - intercept) / slope
        spikes.append((nominal, float(y), found, 100.0 * found / nominal))

    matrices = []
    for name, col, factor in MATRICES:
        c = np.repeat(LEVELS, N_REP)
        y = respond(c, rng, slope=SLOPE_TRUE * factor)
        s, b, r, _res = regress(c, y)
        matrices.append((name, col, s, b, r))
    return (conc, resp, blanks, slope, intercept, r2, resid, sd_blank, lod,
            loq, spikes, matrices)


def main_panel(ax, conc, resp, slope, intercept, r2, lod, loq, spikes):
    grid = np.linspace(0.0, C_HI, 200)
    ax.axvspan(0.0, loq, color=lighten(RED, 0.90), lw=0.0, zorder=1)
    ax.plot(grid, intercept + slope * grid, color=RED, lw=1.3, zorder=5)
    ax.plot(conc, resp, ls="none", marker="o", ms=4.0, mfc="none", mec=DATA,
            mew=0.8, zorder=6)

    for (nominal, y, found, rec), (tx, ty) in zip(spikes, SPIKE_LABEL):
        ax.plot([nominal], [y], marker="D", ms=5.4, mfc=lighten(PURPLE, 0.45),
                mec=PURPLE, mew=1.0, zorder=8)
        ax.annotate("{:.0f} $\\mu$g/L spike, found {:.1f} $\\mu$g/L\n"
                    "recovery {:.1f} %".format(nominal, found, rec),
                    xy=(nominal, y), xytext=(tx, ty), fontsize=6.1,
                    color=PURPLE, ha="left", va="center", linespacing=1.3,
                    zorder=8,
                    arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6,
                                    shrinkA=0.0, shrinkB=3.0))

    ax.set_xlim(-6.0, C_HI)
    ax.set_ylim(-1600.0, 34500.0)
    ax.set_xlabel(r"concentration ($\mu$g/L)")
    ax.set_ylabel("peak area response (a.u.)")
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(loq + 3.0, -900.0, "below LOQ", ha="left", va="bottom",
            fontsize=6.0, color=RED, zorder=7)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc="none", mec=DATA,
               mew=0.8, label="calibration standards, n = {:d}".format(N_REP)),
        Line2D([], [], color=RED, lw=1.3, label="ordinary least squares fit"),
        Line2D([], [], ls="none", marker="D", ms=4.4,
               mfc=lighten(PURPLE, 0.45), mec=PURPLE, mew=1.0,
               label="spiked recovery samples"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.4, frameon=False,
              ncol=3, bbox_to_anchor=(0.0, 0.998), handlelength=1.5,
              columnspacing=1.6, borderaxespad=0.0)
    ax.set_title("Calibration in neat solvent with the detection limits it "
                 "implies", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def zoom_inset(ax, conc, resp, blanks, slope, intercept, sd_blank, lod, loq):
    axi = ax.inset_axes([0.075, 0.455, 0.375, 0.400])
    axi.axvspan(0.0, lod, color=lighten(RED, 0.82), lw=0.0, zorder=1)
    axi.axvspan(lod, loq, color=lighten(ZOOM, 0.82), lw=0.0, zorder=1)
    grid = np.linspace(0.0, ZOOM_HI, 120)
    axi.plot(grid, intercept + slope * grid, color=RED, lw=1.1, zorder=5)
    axi.axhline(intercept, color=MUTED, lw=0.6, ls=(0, (3.0, 2.0)), zorder=3)
    axi.plot(np.zeros(blanks.size), blanks, ls="none", marker="_", ms=7.0,
             mew=0.9, color=SUB, zorder=6)
    m = conc <= ZOOM_HI
    axi.plot(conc[m], resp[m], ls="none", marker="o", ms=3.4, mfc="none",
             mec=DATA, mew=0.8, zorder=7)

    for value, name, col in ((lod, "LOD", RED), (loq, "LOQ", ZOOM)):
        axi.axvline(value, color=col, lw=1.1, zorder=8)
        axi.text(value + 0.18, 2000.0,
                 "{:s} {:.2f}".format(name, value), ha="left", va="top",
                 fontsize=6, color=col, fontweight="bold", rotation=90.0,
                 zorder=9)
    axi.text(ZOOM_HI - 0.3, 60.0, r"blanks n = {:d}, s = {:.0f} a.u.".format(
        N_BLANK, sd_blank), ha="right", va="bottom", fontsize=6, color=SUB,
        zorder=9)

    axi.set_xlim(-0.8, ZOOM_HI)
    axi.set_ylim(-100.0, 2150.0)
    axi.set_xlabel(r"c ($\mu$g/L)", fontsize=6.2, labelpad=1.0)
    axi.set_ylabel("response (a.u.)", fontsize=6.2, labelpad=1.5)
    axi.tick_params(labelsize=6, length=2.0, pad=1.2)
    axi.set_xticks([0.0, 4.0, 8.0, 12.0])
    axi.patch.set_facecolor("white")
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.9)
        s.set_color(ZOOM)
    # A rectangle in the same colour beats connector lines here: the crop is
    # a sliver at the origin and the leaders would run under the whole curve.
    ax.add_patch(plt.Rectangle((-0.8, -100.0), ZOOM_HI + 0.8, 2250.0,
                               facecolor="none", edgecolor=ZOOM,
                               linewidth=0.9, zorder=9))
    return axi


def stats_box(ax, slope, intercept, r2, sd_blank, lod, loq):
    ax.text(0.985, 0.025,
            "y = {:.1f} x + {:.0f}   (a.u.)\n"
            "$R^2$ = {:.5f},  n = {:d} standards\n"
            "s$_{{blank}}$ = {:.0f} a.u. (n = {:d})\n"
            "LOD = 3 s / S = {:.2f} $\\mu$g/L\n"
            "LOQ = 10 s / S = {:.2f} $\\mu$g/L\n"
            "linear range {:.1f} - {:.0f} $\\mu$g/L".format(
                slope, intercept, r2, LEVELS.size * N_REP, sd_blank, N_BLANK,
                lod, loq, loq, LEVELS.max()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color=INK, linespacing=1.55, zorder=9,
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6))


def residual_panel(ax, conc, resid):
    s = float(np.std(resid, ddof=2))
    ax.axhspan(-2.0 * s, 2.0 * s, color=lighten(BLUE, 0.88), lw=0.0, zorder=1)
    ax.axhline(0.0, color=RED, lw=0.9, zorder=4)
    ax.plot(conc, resid, ls="none", marker="o", ms=3.8, mfc="none", mec=DATA,
            mew=0.8, zorder=6)
    ax.set_xlim(-6.0, C_HI)
    lim = max(float(np.max(np.abs(resid))), 2.2 * s) * 1.22
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"concentration ($\mu$g/L)", fontsize=7.2)
    ax.set_ylabel("residual (a.u.)", fontsize=7.2)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(C_HI - 4.0, 2.0 * s, r"$\pm$2 s = $\pm${:.0f} a.u.".format(2 * s),
            ha="right", va="bottom", fontsize=6.0, color=BLUE, zorder=7)
    ax.set_title("Fit residuals: scatter, not curvature, over the whole range",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)
    return s


def matrix_panel(ax, matrices):
    ref = matrices[0][2]
    pos = np.arange(len(matrices))[::-1]
    for k, (name, col, slope, _b, _r) in enumerate(matrices):
        ax.barh(pos[k], slope, height=0.54, color=col, edgecolor="white",
                linewidth=0.5, zorder=4)
        me = 100.0 * (slope / ref - 1.0)
        tag = ("reference" if k == 0
               else "ME = {:+.1f} %".format(me))
        ax.text(slope + 3.0, pos[k], "{:.1f}   {:s}".format(slope, tag),
                ha="left", va="center", fontsize=6.3, color=col,
                fontweight="bold", zorder=6)
        ax.text(0.0, pos[k] + 0.40, name, ha="left", va="bottom",
                fontsize=6.3, color=INK, zorder=6)
    ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (3.0, 2.0)), zorder=3)
    ax.set_xlim(0.0, ref * 1.62)
    ax.set_ylim(-0.62, len(matrices) - 0.12)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, ref * 1.1, 50.0))
    ax.set_xlabel(r"calibration slope (a.u. per $\mu$g/L)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Matrix effect measured as the change in slope",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)


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
    rng = np.random.default_rng(941)
    (conc, resp, blanks, slope, intercept, r2, resid, sd_blank, lod, loq,
     spikes, matrices) = build(rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_cal = axes_inches(fig, MAIN)
    ax_res = axes_inches(fig, AUX_T)
    ax_mat = axes_inches(fig, AUX_B)

    main_panel(ax_cal, conc, resp, slope, intercept, r2, lod, loq, spikes)
    zoom_inset(ax_cal, conc, resp, blanks, slope, intercept, sd_blank, lod,
               loq)
    stats_box(ax_cal, slope, intercept, r2, sd_blank, lod, loq)
    s_res = residual_panel(ax_res, conc, resid)
    matrix_panel(ax_mat, matrices)

    rec = [sp[3] for sp in spikes]
    worst = matrices[-1]
    me_worst = 100.0 * (worst[2] / matrices[0][2] - 1.0)
    panel_labels_inches(fig, [ax_cal, ax_res, ax_mat], dx=-0.66, dy=0.14)
    savefig(fig, "941_anl_calibration_lod.png")


if __name__ == "__main__":
    main()
