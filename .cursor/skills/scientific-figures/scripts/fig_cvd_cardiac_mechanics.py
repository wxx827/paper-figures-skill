"""Myocardial strain: AHA bull's eye, segment curve family and the GLS summary.

The bull's eye is the signature plot of cardiac imaging, and it only works if
the geometry is honest: seventeen wedges on an ``set_aspect("equal")`` axes,
basal ring outside, apical cap in the centre, drawn from the AHA sector angles
rather than eyeballed.  Strain is negative everywhere it is normal, so the
colour ramp is a house sequential blue with dark = large shortening; a
diverging map would have put its neutral white somewhere inside the healthy
range and invented a boundary that does not exist.

One array of seventeen peak strains drives everything: the wedge colours, the
colour of each segment's time-strain curve, the rug of segment values under
the gauge, and the global longitudinal strain itself, which is their mean.
The impaired segments are the LAD territory, and they are late as well as
weak - which is why the curve family is worth drawing next to the map.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Wedge

from pubstyle import set_pub_style, savefig, get_palette
from figkit import panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRIDC = "#e2e2e2"
ACCENT = "#b23a48"

VMIN, VMAX = -24.0, -4.0        # strain (%) mapped onto the ramp
IMPAIRED = -12.0                # segments weaker than this get the red ring
T_AVC = 0.355                   # aortic valve closure, s
PERIOD = 60.0 / 72.0

WALLS = ["anterior", "anteroseptal", "inferoseptal", "inferior",
         "inferolateral", "anterolateral"]
APICAL = ["anterior", "septal", "inferior", "lateral"]

# AHA territory of each segment id; the LAD set is the diseased one here.
LAD_SEG = {1, 2, 7, 8, 13, 14, 17}
TARGET = {1: -14.0, 2: -13.4, 7: -11.0, 8: -10.4, 13: -8.2, 14: -7.8,
          17: -7.2}
NORMAL_STRAIN = -20.5


def segments():
    """(id, name, r_inner, r_outer, theta0, theta1) for the AHA 17 model."""
    out = []
    for k, name in enumerate(WALLS):                      # basal 1-6
        a0 = 60.0 + 60.0 * k
        out.append((k + 1, f"basal {name}", 0.74, 1.00, a0, a0 + 60.0))
    for k, name in enumerate(WALLS):                      # mid 7-12
        a0 = 60.0 + 60.0 * k
        out.append((k + 7, f"mid {name}", 0.50, 0.74, a0, a0 + 60.0))
    for k, name in enumerate(APICAL):                     # apical 13-16
        a0 = 45.0 + 90.0 * k
        out.append((k + 13, f"apical {name}", 0.24, 0.50, a0, a0 + 90.0))
    out.append((17, "apex", 0.0, 0.24, 0.0, 360.0))
    return out


def strain_curve(t, peak, t_peak):
    """Cosine shortening ramp to peak, exponential relaxation afterwards."""
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / t_peak, 0.0, 1.0))
    relax = np.exp(-np.clip(t - t_peak, 0.0, None) / 0.115)
    return peak * rise * relax


def main():
    set_pub_style()
    rng = np.random.default_rng(15503)
    segs = segments()

    peak = {}
    t_peak = {}
    for sid, *_ in segs:
        base = TARGET.get(sid, NORMAL_STRAIN)
        peak[sid] = base + rng.normal(0.0, 1.05)
        t_peak[sid] = ((0.415 if sid in LAD_SEG else 0.330) +
                       rng.normal(0.0, 0.018))
    vals = np.array([peak[s[0]] for s in segs])
    tps = np.array([t_peak[s[0]] for s in segs]) * 1000.0
    gls = float(vals.mean())
    dispersion = float(tps.std())

    cols = get_palette("cb_sequential_blue")[1:][::-1]     # dark -> light
    cmap = LinearSegmentedColormap.from_list("strain_ramp", cols)
    norm = Normalize(vmin=VMIN, vmax=VMAX)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], hspace=0.40, wspace=0.06)
    ax_be = fig.add_subplot(gs[:, 0])
    ax_cv = fig.add_subplot(gs[0, 1])
    ax_gl = fig.add_subplot(gs[1, 1])

    # ---- (a) bull's eye -----------------------------------------------------
    ax_be.set_aspect("equal")
    ax_be.set_xlim(-1.32, 1.32)
    ax_be.set_ylim(-1.46, 1.26)
    ax_be.axis("off")
    for sid, name, ri, ro, a0, a1 in segs:
        v = peak[sid]
        bad = v > IMPAIRED
        ax_be.add_patch(Wedge((0.0, 0.0), ro, a0, a1, width=ro - ri,
                              facecolor=cmap(norm(v)),
                              edgecolor=ACCENT if bad else "white",
                              linewidth=1.5 if bad else 0.9,
                              zorder=4 if bad else 3))
        am = np.deg2rad(0.5 * (a0 + a1)) if sid != 17 else np.deg2rad(90.0)
        rm = 0.5 * (ri + ro) if sid != 17 else 0.0
        halo_text(ax_be, rm * np.cos(am), rm * np.sin(am),
                  f"{sid}\n{v:.0f}", color="white", halo="#1a1a1a",
                  halo_lw=1.6, fontsize=6.4, ha="center", va="center",
                  linespacing=1.15, zorder=8)
    for ang, lab in ((90, "anterior"), (180, "septal"), (270, "inferior"),
                     (0, "lateral")):
        a = np.deg2rad(ang)
        ax_be.text(1.11 * np.cos(a), 1.11 * np.sin(a), lab, ha="center",
                   va="center", fontsize=7.2, color=SUB)
    ax_be.set_title("Peak systolic longitudinal strain by AHA segment",
                    fontsize=8.8, fontweight="bold", loc="left", pad=8)
    ax_be.text(0.0, -1.39,
               "rings outward-in: basal 1\u20136, mid 7\u201312, apical "
               "13\u201316, apex 17   |   cell text: segment id and strain (%)",
               ha="center", va="center", fontsize=6.4, color=MUTED)
    ax_be.text(0.0, -1.28, f"red outline: segment strain weaker than "
               f"{IMPAIRED:.0f}% \u2014 the LAD territory",
               ha="center", va="center", fontsize=6.6, color=ACCENT)

    # The bull's eye is equal-aspect, so its applied box fills the cell width
    # and the spare room is below it - which is where the bar has to go.
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    safe_colorbar(fig, sm, ax_be, label="peak longitudinal strain (%)",
                  side="bottom", width=0.11, pad=0.08, shrink=0.55,
                  ticks=[-24, -20, -16, -12, -8, -4])

    # ---- (b) segment curve family (auxiliary) ------------------------------
    t = np.linspace(0.0, PERIOD, 340)
    for sid, *_ in segs:
        ax_cv.plot(t * 1000.0, strain_curve(t, peak[sid], t_peak[sid]),
                   color=cmap(norm(peak[sid])), lw=1.0,
                   alpha=0.95 if peak[sid] > IMPAIRED else 0.75,
                   zorder=4 if peak[sid] > IMPAIRED else 3)
    ax_cv.axhline(0.0, color="#808080", lw=0.8, zorder=2)
    ax_cv.axvline(T_AVC * 1000.0, color="#808080", lw=0.8, ls=(0, (4, 2.4)),
                  zorder=2)
    ax_cv.text(T_AVC * 1000.0 - 10, -0.9, "aortic valve closure",
               fontsize=6.4, color=SUB, va="top", ha="right")
    ax_cv.plot(tps, vals, ls="none", marker="o", ms=3.0, mfc="white",
               mec=INK, mew=0.6, zorder=6)
    ax_cv.set_xlabel("time from R wave (ms)", fontsize=7.4)
    ax_cv.set_ylabel("segmental strain (%)", fontsize=7.4)
    ax_cv.set_xlim(0.0, PERIOD * 1000.0)
    ax_cv.set_ylim(VMIN, 1.5)
    ax_cv.set_title("Seventeen segment curves, coloured as in (a)",
                    fontsize=8.5, fontweight="bold", loc="left")
    ax_cv.grid(color=GRIDC, lw=0.5)
    ax_cv.tick_params(labelsize=6.5)
    ax_cv.text(0.985, 0.06, f"time-to-peak dispersion SD = "
               f"{dispersion:.0f} ms  (open circles)",
               transform=ax_cv.transAxes, ha="right", va="bottom",
               fontsize=6.4, color=SUB)

    # ---- (c) GLS gauge (auxiliary) -----------------------------------------
    zones = [(-26.0, -18.0, "#cfe0d6", "normal"),
             (-18.0, -16.0, "#eee0be", "borderline"),
             (-16.0, -6.0, "#f2d7da", "reduced")]
    for lo, hi, colour, lab in zones:
        ax_gl.axvspan(lo, hi, color=colour, lw=0, zorder=0)
        ax_gl.text(0.5 * (lo + hi), 0.94, lab, ha="center", va="center",
                   fontsize=6.5, color=SUB)
    jitter = rng.uniform(0.22, 0.48, vals.size)
    ax_gl.plot(vals, jitter, ls="none", marker="o", ms=3.4,
               mfc="white", mec="#3b5b92", mew=0.7, zorder=4)
    ax_gl.plot([gls, gls], [0.05, 0.62], color=ACCENT, lw=1.6, zorder=6)
    ax_gl.plot([gls], [0.62], marker="v", ms=6.0, color=ACCENT, zorder=6)
    ax_gl.text(gls, 0.68, f"GLS  {gls:.1f}%", ha="center", va="bottom",
               fontsize=8.6, fontweight="bold", color=ACCENT, zorder=7)
    ax_gl.set_xlim(-26.0, -6.0)
    ax_gl.set_ylim(0.0, 1.0)
    ax_gl.set_yticks([])
    ax_gl.set_xlabel("global longitudinal strain (%)", fontsize=7.4)
    ax_gl.set_title("GLS against the reference range", fontsize=8.5,
                    fontweight="bold", loc="left")
    ax_gl.tick_params(labelsize=6.5)
    ax_gl.spines["left"].set_visible(False)
    ax_gl.text(-25.6, 0.09, f"open circles: the {vals.size} segment values "
               f"behind the mean", fontsize=6.4, color=MUTED, va="center")

    fig.tight_layout(rect=[0.02, 0, 1, 0.95], h_pad=2.4, w_pad=2.0)
    panel_labels_inches(fig, [ax_be, ax_cv, ax_gl], dx=-0.34, dy=0.06)
    savefig(fig, "550_cvd_cardiac_mechanics.png")


if __name__ == "__main__":
    main()
