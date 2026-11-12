"""Longitudinal residual stress left in a plate by a single welding pass.

The map is the claim: the weld line is locked in tension close to yield, and
that tension is paid for by a broad compressive field on both sides, which is
why the colour scale is diverging and pinned with white exactly on zero.  The
profile panel cuts the same field at mid-length and puts the yield strength
next to it, so "close to yield" is a measurement rather than an adjective, and
the sequence panel rescales that measured peak by the reduction each welding
sequence buys.  Bars are tinted with the same colour ruler as the map.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, halo_text, panel_labels_inches, shared_norm
from figkit import safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
ACCENT = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"

DIV_CMAP = get_cmap("cb_diverging")   # blue = compression, warm = tension

X_HALF = 90.0           # mm, transverse half width of the plate
L_WELD = 150.0          # mm, length of the weld run
B_WIDTH = 18.0          # mm, tension band half width
SIG_PEAK = 318.0        # MPa, longitudinal peak on the weld line
SIG_YIELD = 355.0       # MPa
BEAD_HALF = 6.0         # mm
RAMP = 34.0             # mm, run-in and run-out length

SEQUENCES = ["Single pass", "Backstep", "Block", "Skip / wandering",
             "Balanced two-side"]
SEQ_FACTOR = [1.00, 0.85, 0.78, 0.71, 0.61]


def transverse_profile(x):
    """Masubuchi-style longitudinal stress across the weld."""
    u = (x / B_WIDTH) ** 2
    return SIG_PEAK * (1.0 - u) * np.exp(-0.5 * u)


def run_shape(s):
    """Free ends carry little longitudinal stress, so the field ramps in."""
    return np.tanh((np.minimum(s, L_WELD - s) + 9.0) / RAMP)


def stress_field(nx=421, ns=361):
    x = np.linspace(-X_HALF, X_HALF, nx)
    s = np.linspace(0.0, L_WELD, ns)
    X, S = np.meshgrid(x, s, indexing="xy")
    return x, s, transverse_profile(X) * run_shape(S)


def map_panel(fig, ax, x, s, sig, norm):
    im = ax.imshow(sig, origin="lower", cmap=DIV_CMAP, norm=norm,
                   extent=[-X_HALF, X_HALF, 0.0, L_WELD], aspect="equal",
                   interpolation="bilinear", zorder=1, rasterized=True)
    lev = [-120.0, -60.0, 60.0, 150.0, 240.0]
    cs = ax.contour(x, s, sig, levels=lev, colors=["#3a3a3a"],
                    linewidths=0.55, alpha=0.85, zorder=3)
    ax.clabel(cs, fmt="%d", fontsize=6.0, colors=["#2b2b2b"],
              inline_spacing=5)
    ax.contour(x, s, sig, levels=[0.0], colors=["#1a1a1a"], linewidths=1.2,
               zorder=4)

    for sx in (-BEAD_HALF, BEAD_HALF):
        ax.plot([sx, sx], [0.0, L_WELD], color="white", lw=0.8,
                ls=(0, (3.0, 2.0)), zorder=5)
    halo_text(ax, 0.0, L_WELD * 0.5, "weld bead", ha="center", va="center",
              rotation=90.0, fontsize=6.6, color="white", halo="#1a1a1a",
              halo_lw=2.0, zorder=6)

    mid = int(np.argmin(np.abs(s - L_WELD * 0.5)))
    prof = sig[mid]
    i_hi = int(np.argmax(prof))
    i_lo = int(np.argmax(-prof))
    zero = x[np.argmin(np.abs(np.where(x > 0.0, prof, 1e9)))]
    ax.annotate(f"peak {prof[i_hi]:.0f} MPa", xy=(x[i_hi] - 2.0, 124.0),
                xytext=(-X_HALF + 5.0, 140.0), fontsize=6.8, color=INK,
                ha="left",
                arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.7))
    ax.annotate(f"{prof[i_lo]:.0f} MPa in compression",
                xy=(x[i_lo], 42.0), xytext=(-X_HALF + 5.0, 13.0),
                fontsize=6.8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.7))
    halo_text(ax, zero + 2.0, L_WELD - 6.0, f"sign change at {zero:.0f} mm",
              ha="left", va="top", fontsize=6.4, color="#2b2b2b",
              halo="white", halo_lw=2.2, zorder=6)

    ax.set_xlabel("transverse distance from weld centreline (mm)",
                  fontsize=8.0)
    ax.set_ylabel("distance along the weld (mm)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"Tension over the bead is balanced by compression beyond "
                 f"{zero:.0f} mm", fontsize=9.0, fontweight="bold", color=INK)
    return im, prof, zero


def profile_panel(ax, x, prof, norm, seed=9):
    rng = np.random.default_rng(seed)
    ax.axhline(0.0, color=RULE, lw=0.9, zorder=3)
    ax.fill_between(x, 0.0, prof, where=prof >= 0.0, color=ACCENT, alpha=0.18,
                    linewidth=0.0, zorder=2)
    ax.fill_between(x, 0.0, prof, where=prof < 0.0, color=BLUE, alpha=0.18,
                    linewidth=0.0, zorder=2)
    ax.plot(x, prof, color=INK, lw=1.6, zorder=5)

    xm = np.arange(-84.0, 84.1, 8.0)
    ym = np.interp(xm, x, prof) + rng.normal(0.0, 11.0, xm.size)
    ax.errorbar(xm, ym, yerr=16.0, fmt="o", ms=2.8, mfc="white", mec=RULE,
                mew=0.8, ecolor=RULE, elinewidth=0.7, capsize=1.8, zorder=6)

    for sgn in (1.0, -1.0):
        ax.axhline(sgn * SIG_YIELD, color=ACCENT, lw=0.9,
                   ls=(0, (3.4, 2.0)), zorder=4)
    ratio = 100.0 * prof.max() / SIG_YIELD
    ax.text(-X_HALF + 3.0, SIG_YIELD - 18.0,
            r"$\sigma_y$ = " + f"{SIG_YIELD:.0f} MPa", fontsize=6.5,
            color=ACCENT, va="top")
    ax.text(B_WIDTH + 8.0, prof.max() * 0.90, f"{ratio:.0f} % of " +
            r"$\sigma_y$", fontsize=6.8, color=INK, ha="left", va="center")
    ax.set_xlim(-X_HALF, X_HALF)
    ax.set_ylim(-SIG_YIELD * 1.18, SIG_YIELD * 1.28)
    ax.set_xlabel("transverse distance (mm)", fontsize=7.4)
    ax.set_ylabel(r"$\sigma_{res}$ (MPa)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Profile at mid-length against yield", fontsize=8.2,
                 fontweight="bold", color=INK)
    return ratio


def sequence_panel(ax, peak, norm):
    peaks = np.array(SEQ_FACTOR) * peak
    y = np.arange(len(SEQUENCES))[::-1]
    ax.barh(y, peaks, height=0.64, color=DIV_CMAP(norm(peaks)),
            edgecolor="white", linewidth=0.6, zorder=3)
    for yi, val in zip(y, peaks):
        ax.text(val + 6.0, yi, f"{val:.0f}", va="center", ha="left",
                fontsize=6.5, color=INK)
    ax.axvline(SIG_YIELD, color=ACCENT, lw=0.9, ls=(0, (3.4, 2.0)), zorder=4)
    ax.text(SIG_YIELD - 6.0, y.max() + 0.55, r"$\sigma_y$", fontsize=6.6,
            color=ACCENT, ha="right", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(SEQUENCES, fontsize=6.8)
    ax.set_xlim(0.0, SIG_YIELD * 1.14)
    ax.set_ylim(-0.7, len(SEQUENCES) - 0.3)
    ax.set_xlabel(r"peak $\sigma_{res}$ (MPa)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    drop = 100.0 * (peaks[0] - peaks[-1]) / peaks[0]
    ax.set_title(f"Sequence buys back {drop:.0f} % of the peak",
                 fontsize=8.2, fontweight="bold", color=INK)
    return peaks


def main():
    set_pub_style()
    x, s, sig = stress_field()
    norm = shared_norm([sig], symmetric=True)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_map = axes_inches(fig, (0.70, 1.05, 5.30, 4.42))
    ax_prof = axes_inches(fig, (7.35, 3.67, 2.55, 1.80))
    ax_seq = axes_inches(fig, (7.35, 1.05, 2.55, 1.80))

    im, prof, zero = map_panel(fig, ax_map, x, s, sig, norm)
    safe_colorbar(fig, im, ax_map, label=r"longitudinal $\sigma_{res}$ (MPa)",
                  width=0.12, pad=0.09, labelsize=6.8,
                  ticks=np.arange(-300.0, 301.0, 100.0))
    ratio = profile_panel(ax_prof, x, prof, norm)
    peaks = sequence_panel(ax_seq, prof.max(), norm)

    panel_labels_inches(fig, [ax_map, ax_prof, ax_seq], dx=-0.52, dy=0.10)
    savefig(fig, "749_wld_residual_stress.png")


if __name__ == "__main__":
    main()
