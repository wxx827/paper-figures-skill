"""Far-UV circular dichroism: thermal unfolding read as a change of secondary
structure rather than as a single melting number.

Four reference spectra - helix, sheet, turn and coil - are written once as
sums of Gaussian bands.  A two-state thermal transition then sets the mixing
fractions, and every drawn spectrum is that mixture plus noise.  Nothing else
is simulated: the composition bars on the right are recovered by
non-negative least squares of each drawn spectrum against the same four
reference spectra, so they are a genuine deconvolution of the curves in the
main panel and not a copy of the fractions that produced them.  The strip
underneath the spectra shows what the four-component model cannot explain.

The melting curve is the ellipticity at 222 nm - the helix marker band -
taken from those arrays at thirteen temperatures and fitted with a two-state
logistic, which yields the midpoint and, through the slope at the midpoint,
a van't Hoff enthalpy.  The isodichroic point quoted on the spectra is found
by scanning for the wavelength of least variance across the family, which is
the evidence that the transition really is two-state.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, nnls

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

# name, colour, [(centre nm, width nm, amplitude in 10^3 deg cm2 dmol-1)]
BASIS = [
    ("alpha-helix", RED,
     [(192.0, 7.0, 78.0), (208.0, 8.5, -34.0), (222.0, 9.5, -34.0)]),
    ("beta-sheet", BLUE, [(195.0, 8.0, 34.0), (217.0, 11.0, -17.0)]),
    ("turn", GREEN, [(203.0, 9.0, 14.0), (228.0, 10.0, -5.0)]),
    ("random coil", GOLD, [(198.0, 8.0, -24.0), (219.0, 12.0, 2.5)]),
]

LAM_LO, LAM_HI, N_LAM = 190.0, 260.0, 561
T_ALL = np.arange(25.0, 86.0, 5.0)      # every temperature measured
SHOW = np.arange(0, len(T_ALL), 2)      # the subset drawn as spectra
TM_TRUE, W_TRUE = 58.0, 4.5             # transition put into the simulation
F_HELIX_HI, F_HELIX_LO = 0.56, 0.06
F_SHEET, F_TURN = 0.11, 0.16
NOISE = 0.45                            # 10^3 deg cm2 dmol-1
R_GAS = 8.314                           # J / mol / K
BAND_208, BAND_222 = 208.0, 222.0

FIG_W, FIG_H = 11.0, 6.5
AX_MAIN = (0.85, 2.12, 5.55, 3.38)
AX_RES = (0.85, 1.30, 5.55, 0.66)
AUX_T = (7.42, 3.62, 3.18, 1.88)
AUX_B = (7.42, 1.28, 3.18, 1.72)


def basis_matrix(lam):
    """Columns are the four reference spectra on the drawn wavelength grid."""
    cols = []
    for _name, _col, bands in BASIS:
        y = np.zeros_like(lam)
        for c, w, a in bands:
            y += a * np.exp(-0.5 * ((lam - c) / w) ** 2)
        cols.append(y)
    return np.column_stack(cols)


def true_fractions(temp):
    helix = F_HELIX_LO + F_HELIX_HI / (1.0 + np.exp((temp - TM_TRUE) / W_TRUE))
    coil = 1.0 - helix - F_SHEET - F_TURN
    return np.array([helix, F_SHEET, F_TURN, coil])


def build(rng):
    lam = np.linspace(LAM_LO, LAM_HI, N_LAM)
    amat = basis_matrix(lam)
    spectra = np.array([amat @ true_fractions(t)
                        + rng.normal(0.0, NOISE, lam.size) for t in T_ALL])
    return lam, amat, spectra


def deconvolve(amat, spectra):
    """Non-negative least squares of each spectrum onto the four references."""
    frac, resid = [], []
    for y in spectra:
        x, _rnorm = nnls(amat, y)
        s = x.sum() if x.sum() > 0 else 1.0
        frac.append(x / s)
        resid.append(y - amat @ x)
    return np.array(frac), np.array(resid)


def two_state(temp, y_n, y_u, tm, w):
    return y_u + (y_n - y_u) / (1.0 + np.exp((temp - tm) / w))


def spectra_panel(fig, ax, lam, spectra, cmap, iso_lam):
    norm = Normalize(T_ALL.min(), T_ALL.max())
    ax.axhline(0.0, color=MUTED, lw=0.7, zorder=3)
    for k in SHOW:
        ax.plot(lam, spectra[k], color=cmap(norm(T_ALL[k])), lw=1.25,
                zorder=5)

    for band, name in ((BAND_208, "208 nm"), (BAND_222, "222 nm")):
        ax.axvline(band, color=HAIR, lw=0.7, ls=(0, (3.5, 2.2)), zorder=2)
        ax.text(band, ax.get_ylim()[0], name, ha="center", va="bottom",
                fontsize=6.0, color=SUB, zorder=6)
    ax.plot([iso_lam], [np.mean(spectra[:, np.argmin(np.abs(lam - iso_lam))])],
            marker="o", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=8)
    ax.annotate("isodichroic point {:.1f} nm".format(iso_lam),
                xy=(iso_lam, 0.0), xytext=(iso_lam + 9.0, 26.0),
                fontsize=6.4, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.7,
                                shrinkA=1.0, shrinkB=3.0,
                                connectionstyle="arc3,rad=-0.22"))

    ax.set_xlim(LAM_LO, LAM_HI)
    ax.set_ylabel(r"[$\theta$] (10$^3$ deg cm$^2$ dmol$^{-1}$)")
    ax.tick_params(axis="x", labelbottom=False, length=2.4)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    cax = ax.inset_axes([0.615, 0.845, 0.315, 0.038])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal",
                      ticks=[25.0, 45.0, 65.0, 85.0])
    cb.set_label("temperature (deg C)", fontsize=6.2, color=INK, labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.7, length=2.2, color="#333333",
                      pad=1.2)
    ax.set_title("Far-UV CD of a helical protein between 25 and 85 deg C",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def residual_panel(ax, lam, resid):
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=3)
    ax.fill_between(lam, resid.min(axis=0), resid.max(axis=0),
                    color=HAIR, alpha=0.75, lw=0.0, zorder=2)
    ax.plot(lam, resid[0], color=BLUE, lw=0.6, zorder=4)
    ax.plot(lam, resid[-1], color=RED, lw=0.6, zorder=4)
    lim = float(np.max(np.abs(resid))) * 1.25
    ax.set_xlim(LAM_LO, LAM_HI)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([0.0])
    ax.set_yticklabels(["0"], fontsize=6.2)
    ax.set_ylabel("residual", fontsize=6.6, labelpad=2.0)
    ax.set_xlabel("wavelength (nm)")
    ax.set_xticks(np.arange(190.0, 261.0, 10.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=BLUE, lw=1.0, label="25 deg C"),
               Line2D([], [], color=RED, lw=1.0, label="85 deg C")]
    ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=False,
              ncol=2, handlelength=1.2, borderaxespad=0.2,
              columnspacing=1.0)
    return lim


def composition_panel(ax, frac):
    pos = np.arange(len(SHOW), dtype=float)
    bottom = np.zeros(len(SHOW))
    for j, (name, col, _b) in enumerate(BASIS):
        vals = 100.0 * frac[SHOW, j]
        ax.bar(pos, vals, bottom=bottom, width=0.68, color=col,
               edgecolor="white", linewidth=0.5, label=name, zorder=4)
        for x, v, b in zip(pos, vals, bottom):
            if v >= 9.0:
                ax.text(x, b + v / 2.0, "{:.0f}".format(v), ha="center",
                        va="center", fontsize=6, color="white",
                        fontweight="bold", zorder=6)
        bottom = bottom + vals
    ax.set_xticks(pos)
    ax.set_xticklabels(["{:.0f}".format(T_ALL[k]) for k in SHOW],
                       fontsize=6.4)
    ax.set_ylim(0.0, 118.0)
    ax.set_yticks(np.arange(0.0, 101.0, 25.0))
    ax.set_xlabel("temperature (deg C)", fontsize=7.2)
    ax.set_ylabel("recovered content (%)", fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6.0, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.055), handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0.0)
    ax.set_title("Secondary structure by NNLS deconvolution of the spectra",
                 fontsize=8.0, fontweight="bold", color=INK, pad=15.0)


def melt_panel(ax, theta222, popt, perr):
    y_n, y_u, tm, w = popt
    grid = np.linspace(T_ALL.min() - 2.0, T_ALL.max() + 2.0, 300)
    ax.plot(grid, two_state(grid, *popt), color=RED, lw=1.3, zorder=5)
    ax.plot(T_ALL, theta222, ls="none", marker="o", ms=4.4, mfc="white",
            mec=BLUE, mew=1.0, zorder=6)
    ax.axhline(y_n, color=MUTED, lw=0.6, ls=(0, (3.0, 2.0)), zorder=3)
    ax.axhline(y_u, color=MUTED, lw=0.6, ls=(0, (3.0, 2.0)), zorder=3)
    ax.axvline(tm, color=GREEN, lw=1.0, ls=(0, (3.5, 2.0)), zorder=4)
    ax.text(tm - 1.5, 0.5 * (y_n + y_u),
            "$T_m$ = {:.1f} $\\pm$ {:.1f} deg C".format(tm, perr),
            ha="right", va="center", fontsize=6.4, color=GREEN,
            fontweight="bold", zorder=7)
    # Each baseline caption goes to the end of the axis where the curve has
    # already left that baseline, so neither sits on a data point.
    ax.text(T_ALL.min() + 1.0, y_u - 0.7, "unfolded baseline", ha="left",
            va="top", fontsize=6, color=MUTED, zorder=7)
    ax.text(T_ALL.max() - 1.0, y_n + 0.6, "folded baseline", ha="right",
            va="bottom", fontsize=6, color=MUTED, zorder=7)
    ax.set_xlim(T_ALL.min() - 2.0, T_ALL.max() + 2.0)
    ax.set_xlabel("temperature (deg C)", fontsize=7.2)
    ax.set_ylabel(r"[$\theta$]$_{222}$ (10$^3$ deg cm$^2$ dmol$^{-1}$)",
                  fontsize=7.0)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Thermal melting followed at the 222 nm helix band",
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
    rng = np.random.default_rng(939)
    lam, amat, spectra = build(rng)
    frac, resid = deconvolve(amat, spectra)

    # Least-variance wavelength inside the crossing region: the isodichroic
    # point a genuine two-state transition has to show.
    win = (lam >= 198.0) & (lam <= 212.0)
    iso_lam = float(lam[win][np.argmin(spectra[:, win].std(axis=0))])

    j222 = int(np.argmin(np.abs(lam - BAND_222)))
    theta222 = spectra[:, j222]
    popt, pcov = curve_fit(two_state, T_ALL, theta222,
                           p0=[theta222[0], theta222[-1], 55.0, 5.0])
    perr = float(np.sqrt(np.diag(pcov))[2])
    tm_k = popt[2] + 273.15
    dh_vh = R_GAS * tm_k ** 2 / abs(popt[3]) / 1000.0        # kJ/mol

    # The pale end of a sequential ramp disappears against white paper, so
    # the lowest temperature starts a third of the way up the scale.
    full = get_cmap("cb_sequential_warm")
    cmap = LinearSegmentedColormap.from_list(
        "cd_temp", full(np.linspace(0.30, 1.0, 224)))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_sp = axes_inches(fig, AX_MAIN)
    ax_rs = axes_inches(fig, AX_RES)
    ax_cp = axes_inches(fig, AUX_T)
    ax_mt = axes_inches(fig, AUX_B)

    lo = float(spectra.min()) - 6.0
    hi = float(spectra.max()) + 12.0
    ax_sp.set_ylim(lo, hi)
    spectra_panel(fig, ax_sp, lam, spectra, cmap, iso_lam)
    ax_sp.set_ylim(lo, hi)
    rlim = residual_panel(ax_rs, lam, resid)
    composition_panel(ax_cp, frac)
    melt_panel(ax_mt, theta222, popt, perr)

    h_lo, h_hi = 100.0 * frac[0, 0], 100.0 * frac[-1, 0]
    panel_labels_inches(fig, [ax_sp, ax_cp, ax_mt], dx=-0.66, dy=0.14)
    savefig(fig, "939_anl_cd_secondary_structure.png")


if __name__ == "__main__":
    main()
