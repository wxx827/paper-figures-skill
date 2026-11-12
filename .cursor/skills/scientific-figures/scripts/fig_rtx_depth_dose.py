"""Percent depth dose for photons, electrons and protons on one axis.

Every curve is generated analytically on a single depth grid and then
normalised to its own maximum, which is exactly what "percent depth dose"
means, so the five modalities remain comparable on one scale.  The depth of
maximum dose, the surface dose, the exit dose, the spread-out Bragg peak
plateau and the 80-20 penumbra are all read back off the plotted arrays
rather than quoted from a table.

The spread-out Bragg peak is not hand-weighted: a non-negative least squares
fit picks the weights of twelve pristine peaks that best flatten the dose
between 10 and 16 cm, which is how a passive scattering system is actually
designed.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.special import erf

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIGW, FIGH = 10.4, 6.4
DMAX_DEPTH = 30.0            # depth range of the main panel (cm)
EXIT_DEPTH = 25.0            # depth at which the exit dose is quoted (cm)
SOBP_LO, SOBP_HI = 10.4, 15.9

PHOTON_C6 = "#3b5b92"
PHOTON_C15 = "#2e7d5b"
ELECTRON_C = "#c9a227"
PRISTINE_C = "#8e6bbf"
SOBP_C = "#b23a48"


def photon_pdd(depth, mu, build, surface):
    """Build-up region times exponential attenuation, peaked below surface."""
    ramp = surface + (1.0 - surface) * (1.0 - np.exp(-build * depth))
    return ramp * np.exp(-mu * depth)


def electron_pdd(depth, r50, width, brems, build=2.6):
    """Flat-ish plateau, steep fall-off, small bremsstrahlung tail."""
    ramp = 0.80 + 0.20 * (1.0 - np.exp(-build * depth))
    fall = 0.5 * (1.0 - erf((depth - r50) / width))
    return ramp * fall + brems * np.exp(-0.03 * depth)


def pristine_peak(depth, rng_cm, sigma=0.22, floor=0.30):
    """Power-law Bragg peak smeared by range straggling."""
    core = np.where(depth < rng_cm,
                    np.maximum(rng_cm - depth, floor) ** -0.435, 0.0)
    dz = float(depth[1] - depth[0])
    return gaussian_filter1d(core, sigma / dz, mode="nearest")


def falloff_80_20(x, y, from_index=None):
    """80 %-20 % width (mm) of the right-hand shoulder of a profile."""
    top = float(y.max())
    i = int(np.argmax(y)) if from_index is None else from_index
    p, xx = y[i:], x[i:]
    x80 = float(np.interp(-0.80 * top, -p, xx))
    x20 = float(np.interp(-0.20 * top, -p, xx))
    return abs(x20 - x80) * 10.0, x80, x20


def off_axis_ratio(x, half, sigma, horn=0.035):
    """Cross-plane profile with flattening-filter horns."""
    prof = 0.5 * (erf((half - x) / sigma) + erf((half + x) / sigma))
    prof = prof * (1.0 + horn * np.clip(np.abs(x) / half, 0.0, 1.0) ** 2.4)
    return 100.0 * prof / prof.max()


def main():
    set_pub_style()
    d = np.linspace(0.0, DMAX_DEPTH, 1501)

    curves = {}
    curves["6 MV photon"] = photon_pdd(d, 0.0470, 3.0, 0.16)
    curves["15 MV photon"] = photon_pdd(d, 0.0355, 1.35, 0.09)
    curves["9 MeV electron"] = electron_pdd(d, 3.65, 0.62, 0.030)
    curves["proton, pristine"] = pristine_peak(d, 16.1)

    # Spread-out Bragg peak: non-negative least squares over twelve pristine
    # peaks, flattening the region [SOBP_LO, SOBP_HI].
    ranges = np.linspace(SOBP_LO + 0.1, 16.1, 12)
    basis = np.column_stack([pristine_peak(d, r) for r in ranges])
    fit = (d >= SOBP_LO) & (d <= SOBP_HI)
    w, _ = nnls(basis[fit], np.ones(int(fit.sum())))
    curves["proton, SOBP"] = basis @ w

    pdd = {k: 100.0 * v / v.max() for k, v in curves.items()}
    style = {"6 MV photon": (PHOTON_C6, 1.5),
             "15 MV photon": (PHOTON_C15, 1.5),
             "9 MeV electron": (ELECTRON_C, 1.4),
             "proton, pristine": (PRISTINE_C, 1.2),
             "proton, SOBP": (SOBP_C, 1.9)}

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.84, 1.02, 5.28, 4.34))
    ax_oar = axes_inches(fig, (7.12, 3.70, 2.88, 1.74))
    ax_of = axes_inches(fig, (7.12, 0.95, 2.88, 1.74))
    for a in (ax, ax_oar, ax_of):
        despine(a)

    # ---- main panel: percent depth dose -----------------------------------
    plateau = pdd["proton, SOBP"] >= 95.0
    p_lo, p_hi = float(d[plateau].min()), float(d[plateau].max())
    ax.axvspan(p_lo, p_hi, color=SOBP_C, alpha=0.07, zorder=0)
    ax.text(0.5 * (p_lo + p_hi), 103.0,
            f"SOBP plateau {p_lo:.1f}-{p_hi:.1f} cm", ha="center",
            va="bottom", fontsize=6.4, color=SOBP_C)

    for name, y in pdd.items():
        col, lw = style[name]
        ax.plot(d, y, color=col, lw=lw, label=name, zorder=4)

    marks = [("6 MV photon", "$d_{max}$"), ("15 MV photon", "$d_{max}$"),
             ("9 MeV electron", "$d_{max}$"),
             ("proton, pristine", "Bragg peak")]
    notes = []
    for name, tag in marks:
        y = pdd[name]
        i = int(np.argmax(y))
        col = style[name][0]
        ax.plot([d[i]], [y[i]], marker="v", ms=4.0, color=col, zorder=6,
                clip_on=False)
        notes.append((name, tag, d[i]))

    ax.annotate(f"{notes[0][1]} = {notes[0][2]:.1f} cm",
                xy=(notes[0][2], 100.0), xytext=(notes[0][2] + 1.1, 88.0),
                fontsize=6.4, color=PHOTON_C6, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PHOTON_C6))
    ax.annotate(f"{notes[1][1]} = {notes[1][2]:.1f} cm",
                xy=(notes[1][2], 100.0), xytext=(notes[1][2] + 3.3, 96.0),
                fontsize=6.4, color=PHOTON_C15, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PHOTON_C15))
    e_r50 = float(np.interp(-50.0, -pdd["9 MeV electron"][
        int(np.argmax(pdd["9 MeV electron"])):],
        d[int(np.argmax(pdd["9 MeV electron"])):]))
    ax.annotate(f"electron $R_{{50}}$ = {e_r50:.1f} cm\n"
                f"tail {pdd['9 MeV electron'][-1]:.1f} % at "
                f"{DMAX_DEPTH:.0f} cm",
                xy=(e_r50, 50.0), xytext=(6.3, 44.0), fontsize=6.4,
                color=ELECTRON_C, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ELECTRON_C))
    ax.annotate(f"pristine peak\nat {notes[3][2]:.1f} cm",
                xy=(notes[3][2], 100.0), xytext=(notes[3][2] + 2.6, 84.0),
                fontsize=6.4, color=PRISTINE_C, ha="left", va="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PRISTINE_C))

    # surface and exit dose comparison, both read off the arrays
    i_exit = int(np.argmin(np.abs(d - EXIT_DEPTH)))
    ax.axvline(EXIT_DEPTH, color=HAIR, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    surf = {k: float(v[0]) for k, v in pdd.items()}
    exitd = {k: float(v[i_exit]) for k, v in pdd.items()}
    ax.plot([0.0, 0.0], [surf["6 MV photon"], surf["15 MV photon"]],
            color="#3a3a3a", lw=0.8, zorder=5)
    for name in ("6 MV photon", "15 MV photon"):
        ax.plot([0.0], [surf[name]], marker="o", ms=3.4,
                color=style[name][0], zorder=6, clip_on=False)
    ax.text(0.55, 0.5 * (surf["6 MV photon"] + surf["15 MV photon"]),
            f"surface dose\n{surf['6 MV photon']:.0f} % vs "
            f"{surf['15 MV photon']:.0f} %", fontsize=6.4, color=SUB,
            ha="left", va="center", linespacing=1.5)
    for name in ("6 MV photon", "15 MV photon", "proton, SOBP"):
        ax.plot([EXIT_DEPTH], [exitd[name]], marker="o", ms=3.2,
                color=style[name][0], zorder=6)
    ax.text(DMAX_DEPTH - 0.4, 96.0,
            f"exit dose at {EXIT_DEPTH:.0f} cm\n"
            f"{exitd['15 MV photon']:.0f} % (15 MV)   "
            f"{exitd['6 MV photon']:.0f} % (6 MV)   "
            f"{exitd['proton, SOBP']:.1f} % (SOBP)",
            fontsize=6.4, color=SUB, ha="right", va="top",
            linespacing=1.5)

    ax.set_xlim(0.0, DMAX_DEPTH)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("depth in water (cm)", fontsize=8.0)
    ax.set_ylabel("percent depth dose (% of that beam's $D_{max}$)",
                  fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#ececec", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", ncol=5, frameon=False,
                    bbox_to_anchor=(-0.005, 1.005), fontsize=6.9,
                    handlelength=1.6, columnspacing=1.2,
                    handletextpad=0.45)
    for h in leg.legend_handles:
        h.set_linewidth(1.7)

    # ---- aux 1: off-axis ratio for two field sizes ------------------------
    x = np.linspace(-14.0, 14.0, 1401)
    oar10 = off_axis_ratio(x, 5.0, 0.34)
    oar20 = off_axis_ratio(x, 10.0, 0.46)
    pen10, x80, x20 = falloff_80_20(x, oar10, from_index=len(x) // 2)
    pen20, _, _ = falloff_80_20(x, oar20, from_index=len(x) // 2)
    ax_oar.axvspan(x80, x20, color=GREY, alpha=0.16, zorder=0)
    ax_oar.axvspan(-x20, -x80, color=GREY, alpha=0.16, zorder=0)
    ax_oar.plot(x, oar20, color=PHOTON_C15, lw=1.2,
                label=f"20 x 20 cm, {pen20:.1f} mm")
    ax_oar.plot(x, oar10, color=PHOTON_C6, lw=1.3,
                label=f"10 x 10 cm, {pen10:.1f} mm")
    ax_oar.text(x20 + 0.35, 62.0, "80-20\npenumbra", fontsize=6.2,
                color=SUB, ha="left", va="center", linespacing=1.5)
    ax_oar.set_xlim(-14.0, 14.0)
    ax_oar.set_ylim(0.0, 142.0)
    ax_oar.set_yticks(np.arange(0, 101, 25))
    ax_oar.set_xlabel("off-axis distance (cm)", fontsize=7.2)
    ax_oar.set_ylabel("off-axis ratio (%)", fontsize=7.2)
    ax_oar.tick_params(labelsize=6.5)
    ax_oar.legend(loc="upper center", frameon=False, fontsize=6.2, ncol=2,
                  handlelength=1.3, handletextpad=0.4, columnspacing=1.0,
                  borderpad=0.1, bbox_to_anchor=(0.5, 1.005))
    ax_oar.set_title("Cross-plane profiles for two field sizes",
                     fontsize=8.5, fontweight="bold", color=INK, pad=4.0)

    # ---- aux 2: output factor versus square field side --------------------
    side = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0])
    raw = 1.0 - 0.35 * np.exp(-side / 5.0)
    of = raw / float(np.interp(10.0, side, raw))
    ax_of.plot(side, of, color="#b23a48", lw=1.3, marker="o", ms=4.0,
               mfc="white", mew=1.1, zorder=4)
    ax_of.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax_of.axvline(10.0, color=HAIR, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax_of.text(10.4, 0.882, "reference\n10 x 10 cm", fontsize=6.2,
               color=MUTED, ha="left", va="bottom", linespacing=1.5)
    for s_val, o_val in zip(side[[0, -1]], of[[0, -1]]):
        ax_of.annotate(f"{o_val:.3f}", xy=(s_val, o_val),
                       xytext=(6 if o_val < 1 else 0, 4 if o_val > 1 else 0),
                       textcoords="offset points",
                       ha="left" if o_val < 1 else "center",
                       va="bottom" if o_val > 1 else "center",
                       fontsize=6.2, color=SUB)
    ax_of.set_xlim(2.0, 27.0)
    ax_of.set_ylim(0.866, 1.062)
    ax_of.set_xlabel("square field side (cm)", fontsize=7.2)
    ax_of.set_ylabel("output factor (-)", fontsize=7.2)
    ax_of.tick_params(labelsize=6.5)
    ax_of.set_title("Output factor relative to 10 x 10 cm", fontsize=8.5,
                    fontweight="bold", color=INK, pad=4.0)



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.36)
    panel_labels_inches(fig, [ax_oar, ax_of], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.22)
    savefig(fig, "872_rtx_depth_dose.png")


if __name__ == "__main__":
    main()
