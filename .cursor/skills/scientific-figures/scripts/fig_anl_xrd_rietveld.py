"""Rietveld refinement of a layered cathode with two crystalline impurities.

The three stacked strips are the standard presentation and they are drawn in
the order the numbers are produced: the observed pattern and the calculated
one on top, the Bragg positions of each phase in the middle, and underneath
the difference curve, which here is literally ``obs - calc`` on the same
grid - no separate model, no rescaling, so a bump in the residual can always
be traced to a peak above it.

The observed pattern is generated from a slightly *wrong* profile (a
different mixing parameter and a few thousandths of a degree of position
error) and then given Poisson counting noise, because a refinement that fits
perfectly produces a goodness of fit of exactly one and teaches nothing.  Rwp
and the GOF quoted on the panel are computed from those two arrays with
weights 1/y_obs, the same definition the refinement software uses.

The right-hand panels close the two questions a phase analysis has to answer.
Quantities come from integrating each phase's own calculated profile and
dividing by its reference intensity ratio.  Sizes come from measuring the
full width at half maximum of six well-isolated reflections directly off the
refined profile, removing the instrumental width in quadrature, and reading
the Williamson-Hall intercept - which separates crystallite size from
microstrain instead of blaming all of the broadening on size the way a bare
Scherrer estimate does.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"          # house red: the calculated pattern
DATA = "#3a3a3a"         # observed counts
DIFF = "#5b6b7a"         # difference curve
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

LAMBDA = 0.154060        # Cu K-alpha1 (nm)
K_SCHERRER = 0.9
INST_FWHM = 0.090        # instrumental broadening (deg 2theta)
SIZE_NM = 42.0           # crystallite size put into the simulation (nm)
STRAIN = 0.00120         # microstrain put into the simulation (-)

TWO_LO, TWO_HI = 10.0, 90.0
N_PT = 4000
ETA_CALC, ETA_OBS = 0.55, 0.60
POS_JITTER = 0.0035      # deg 2theta of un-modelled peak displacement
COUNT_SCALE = 260.0      # counts per unit of relative intensity
N_PARAM = 34             # refined parameters, for the Rexp denominator

# name, short name, colour, scale factor, reference intensity ratio,
# [(2theta deg, relative height, hkl)]
PHASES = [
    ("LiNi$_{0.8}$Co$_{0.1}$Mn$_{0.1}$O$_2$ (R-3m)", "layered", RED, 1.000,
     4.50,
     [(18.72, 100.0, "003"), (36.68, 18.0, "101"), (38.34, 14.0, "006"),
      (38.72, 11.0, "012"), (44.44, 58.0, "104"), (48.62, 22.0, "015"),
      (58.56, 20.0, "107"), (64.36, 30.0, "018"), (65.30, 26.0, "110"),
      (68.02, 16.0, "113"), (77.30, 14.0, "116"), (82.30, 12.0, "024")]),
    ("NiO rock salt (Fm-3m)", "NiO", BLUE, 0.085, 4.20,
     [(37.28, 55.0, ""), (43.30, 100.0, ""), (62.90, 42.0, ""),
      (75.40, 14.0, ""), (79.40, 12.0, "")]),
    ("Li$_2$CO$_3$ (C2/c)", "Li$_2$CO$_3$", GREEN, 0.045, 1.60,
     [(21.34, 45.0, ""), (30.60, 70.0, ""), (31.78, 100.0, ""),
      (34.10, 30.0, ""), (36.94, 25.0, ""), (39.40, 18.0, "")]),
]

# hkl captions printed on the pattern, and the vertical offset in counts
HKL_SHOWN = ["003", "101", "104", "015", "107", "018", "110", "113", "116"]
# reflections clear enough of a neighbour to give an honest width
WH_PEAKS = [18.72, 48.62, 58.56, 68.02, 77.30, 82.30]

FIG_W, FIG_H = 11.0, 6.5
AX_OBS = (0.85, 2.42, 5.55, 3.08)
AX_TICK = (0.85, 1.92, 5.55, 0.44)
AX_DIFF = (0.85, 1.30, 5.55, 0.56)
AUX_T = (7.42, 3.62, 3.18, 1.88)
AUX_B = (7.42, 1.28, 3.18, 1.72)


def pseudo_voigt(x, x0, fwhm, eta):
    """Unit-height pseudo-Voigt; ``fwhm`` in the same units as ``x``."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - x0) / (0.5 * fwhm)) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def peak_fwhm(two_theta, size_nm=SIZE_NM, strain=STRAIN):
    """Size, strain and instrument broadening added in quadrature (deg)."""
    th = np.deg2rad(two_theta / 2.0)
    b_size = K_SCHERRER * LAMBDA / (size_nm * np.cos(th))
    b_strain = 4.0 * strain * np.tan(th)
    deg = np.rad2deg(np.hypot(b_size, b_strain))
    return float(np.hypot(deg, INST_FWHM))


def phase_profile(x, refl, scale, eta, jitter=0.0):
    y = np.zeros_like(x)
    for pos, rel, _hkl in refl:
        p = pos + jitter
        y += scale * rel * pseudo_voigt(x, p, peak_fwhm(p), eta)
    return y


def background(x):
    return 300.0 + 900.0 * np.exp(-(x - TWO_LO) / 18.0) + 1.6 * (x - TWO_LO)


def build(rng):
    x = np.linspace(TWO_LO, TWO_HI, N_PT)
    parts = [phase_profile(x, refl, sc, ETA_CALC)
             for _n, _s, _c, sc, _r, refl in PHASES]
    calc = background(x) + COUNT_SCALE * np.sum(parts, axis=0)

    # The "truth" the diffractometer sees differs from the model by a shape
    # parameter and a small displacement - the usual residue of a real fit.
    truth = background(x) + COUNT_SCALE * np.sum(
        [phase_profile(x, refl, sc, ETA_OBS,
                       jitter=rng.normal(0.0, POS_JITTER))
         for _n, _s, _c, sc, _r, refl in PHASES], axis=0)
    obs = rng.poisson(np.clip(truth, 1.0, None)).astype(float)
    return x, obs, calc, [COUNT_SCALE * p for p in parts]


def figures_of_merit(obs, calc):
    w = 1.0 / np.clip(obs, 1.0, None)
    num = float(np.sum(w * (obs - calc) ** 2))
    den = float(np.sum(w * obs ** 2))
    rwp = np.sqrt(num / den)
    rexp = np.sqrt((obs.size - N_PARAM) / den)
    return 100.0 * rwp, 100.0 * rexp, rwp / rexp


def quantify(x, parts):
    """Reference-intensity-ratio weight fractions from the drawn profiles."""
    area = np.array([float(np.trapezoid(p, x)) for p in parts])
    rir = np.array([ph[4] for ph in PHASES])
    w = area / rir
    return 100.0 * w / w.sum(), area


def measure_fwhm(x, y, x0, span=1.6):
    """Half-maximum width read straight off the refined phase profile."""
    m = np.abs(x - x0) <= span
    xs, ys = x[m], y[m] - np.min(y[m])
    top = ys.max()
    j = int(np.argmax(ys))
    left = np.interp(0.5 * top, ys[:j + 1], xs[:j + 1])
    right = np.interp(0.5 * top, ys[j:][::-1], xs[j:][::-1])
    return float(right - left)


def williamson_hall(x, layered):
    xs, ys, widths = [], [], []
    for pos in WH_PEAKS:
        fwhm = measure_fwhm(x, layered, pos)
        beta = np.deg2rad(np.sqrt(max(fwhm ** 2 - INST_FWHM ** 2, 1e-9)))
        th = np.deg2rad(pos / 2.0)
        xs.append(4.0 * np.sin(th))
        ys.append(beta * np.cos(th))
        widths.append(fwhm)
    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    size = K_SCHERRER * LAMBDA / intercept
    # Single-peak Scherrer estimate on the strongest reflection, for contrast
    th0 = np.deg2rad(WH_PEAKS[0] / 2.0)
    b0 = np.deg2rad(np.sqrt(max(widths[0] ** 2 - INST_FWHM ** 2, 1e-9)))
    scherrer = K_SCHERRER * LAMBDA / (b0 * np.cos(th0))
    return xs, ys, slope, intercept, size, scherrer, widths


def pattern_panel(ax, x, obs, calc, rwp, rexp, gof):
    ax.plot(x, obs / 1e3, ls="none", marker="o", ms=1.1, mfc="none",
            mec=DATA, mew=0.35, zorder=4)
    ax.plot(x, calc / 1e3, color=RED, lw=0.95, zorder=6)

    layered = PHASES[0]
    for pos, rel, hkl in layered[5]:
        if hkl not in HKL_SHOWN:
            continue
        top = float(np.interp(pos, x, calc)) / 1e3
        ax.text(pos, top + 0.75, hkl, ha="center", va="bottom", fontsize=6,
                color=RED, zorder=8)

    ax.set_xlim(TWO_LO, TWO_HI)
    ax.set_ylim(-0.9, 30.5)
    ax.set_ylabel(r"intensity (10$^3$ counts)")
    ax.tick_params(axis="x", labelbottom=False, length=2.4)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=2.6, mfc="none", mec=DATA,
               mew=0.55, label="observed"),
        Line2D([], [], color=RED, lw=1.2, label="calculated"),
        Line2D([], [], color=DIFF, lw=1.0, label="difference"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.6, frameon=False,
              handlelength=1.4, labelspacing=0.28, borderaxespad=0.4)
    ax.text(0.985, 0.735,
            "$R_{{wp}}$ = {:.2f} %\n$R_{{exp}}$ = {:.2f} %\n"
            "GOF $\\chi$ = {:.2f}".format(rwp, rexp, gof),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6))
    ax.set_title("Three-phase Rietveld refinement of a Ni-rich layered "
                 "cathode", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def tick_panel(ax):
    for k, (_n, short, col, _s, _r, refl) in enumerate(PHASES):
        row = len(PHASES) - 1 - k
        for pos, _rel, _hkl in refl:
            ax.plot([pos, pos], [row + 0.18, row + 0.82], color=col, lw=0.9,
                    solid_capstyle="butt")
    ax.set_xlim(TWO_LO, TWO_HI)
    ax.set_ylim(0.0, len(PHASES))
    ax.set_yticks(np.arange(len(PHASES)) + 0.5)
    ax.set_yticklabels([p[1] for p in PHASES][::-1], fontsize=6.5)
    for lab, ph in zip(ax.get_yticklabels(), PHASES[::-1]):
        lab.set_color(ph[2])
    ax.tick_params(axis="x", labelbottom=False, length=2.0)
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def diff_panel(ax, x, obs, calc):
    d = (obs - calc) / 1e3
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=3)
    ax.plot(x, d, color=DIFF, lw=0.55, zorder=4)
    lim = float(np.max(np.abs(d))) * 1.25
    ax.set_xlim(TWO_LO, TWO_HI)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([0.0])
    ax.set_yticklabels(["0"], fontsize=6.2)
    ax.set_ylabel("obs $-$ calc", fontsize=6.6, labelpad=2.0)
    ax.set_xlabel(r"2$\theta$ (deg), Cu K$\alpha_1$ "
                  r"($\lambda$ = 0.15406 nm)")
    ax.set_xticks(np.arange(10.0, 91.0, 10.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return float(np.max(np.abs(obs - calc)))


def quant_panel(ax, wfrac):
    pos = np.arange(len(PHASES))[::-1]
    for k, (name, _s, col, _sc, _r, _refl) in enumerate(PHASES):
        ax.barh(pos[k], wfrac[k], height=0.56, color=col, edgecolor="white",
                linewidth=0.5, zorder=4)
        ax.text(wfrac[k] + 1.6, pos[k], "{:.1f} wt%".format(wfrac[k]),
                ha="left", va="center", fontsize=6.4, color=col,
                fontweight="bold", zorder=6)
        ax.text(0.0, pos[k] + 0.42, name, ha="left", va="bottom",
                fontsize=6.2, color=INK, zorder=6)
    ax.set_xlim(0.0, 118.0)
    ax.set_ylim(-0.62, len(PHASES) - 0.10)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, 101.0, 20.0))
    ax.set_xlabel("refined weight fraction (%)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Quantitative phase analysis by integrated intensity / RIR",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)


def wh_panel(ax, xs, ys, slope, intercept, size, scherrer):
    grid = np.linspace(0.0, xs.max() * 1.12, 60)
    ax.plot(grid, (intercept + slope * grid) * 1e3, color=RED, lw=1.1,
            zorder=4)
    ax.plot(xs, ys * 1e3, ls="none", marker="o", ms=4.2, mfc=lighten(BLUE,
                                                                     0.45),
            mec=BLUE, mew=0.8, zorder=6)
    # The two highest-angle reflections sit close together on this axis, so
    # their captions alternate above and below the point instead of touching.
    for k, (x0, y0, pos) in enumerate(zip(xs, ys, WH_PEAKS)):
        up = k % 2 == 0
        ax.text(x0, y0 * 1e3 + (0.12 if up else -0.14), "{:.1f}".format(pos),
                ha="center", va="bottom" if up else "top", fontsize=6.2,
                color=SUB, zorder=7)
    ax.axhline(intercept * 1e3, color=MUTED, lw=0.6, ls=(0, (3.0, 2.0)),
               zorder=3)
    ax.set_xlim(0.0, xs.max() * 1.12)
    ax.set_ylim(0.0, ys.max() * 1e3 * 1.42)
    ax.set_xlabel(r"4 sin$\theta$ (-)", fontsize=7.2)
    ax.set_ylabel(r"$\beta$ cos$\theta$ (10$^{-3}$ rad)", fontsize=7.2)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.035, 0.965,
            "$D$ = {:.0f} nm (intercept)\n"
            "$\\varepsilon$ = {:.2f} x 10$^{{-3}}$ (slope)\n"
            "Scherrer on (003) alone: {:.0f} nm".format(
                size, slope * 1e3, scherrer),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6))
    ax.set_title("Williamson-Hall size and strain from six reflections",
                 fontsize=8.0, fontweight="bold", color=INK, pad=14.0)
    ax.text(0.0, 1.012, r"point captions are 2$\theta$ in deg",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0,
            color=SUB)


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
    rng = np.random.default_rng(938)
    x, obs, calc, parts = build(rng)
    rwp, rexp, gof = figures_of_merit(obs, calc)
    wfrac, _area = quantify(x, parts)
    xs, ys, slope, intercept, size, scherrer, widths = williamson_hall(
        x, parts[0])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_obs = axes_inches(fig, AX_OBS)
    ax_tick = axes_inches(fig, AX_TICK)
    ax_dif = axes_inches(fig, AX_DIFF)
    ax_q = axes_inches(fig, AUX_T)
    ax_wh = axes_inches(fig, AUX_B)

    pattern_panel(ax_obs, x, obs, calc, rwp, rexp, gof)
    tick_panel(ax_tick)
    diff_panel(ax_dif, x, obs, calc)
    quant_panel(ax_q, wfrac)
    wh_panel(ax_wh, xs, ys, slope, intercept, size, scherrer)

    panel_labels_inches(fig, [ax_obs, ax_q, ax_wh], dx=-0.62, dy=0.14)
    savefig(fig, "938_anl_xrd_rietveld.png")


if __name__ == "__main__":
    main()
