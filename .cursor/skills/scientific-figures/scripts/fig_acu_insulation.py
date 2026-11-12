"""Airborne sound insulation of four building elements.

Main panel: transmission loss per third-octave band for four constructions,
with the ISO 717-1 reference contour shifted onto the best performer by the
real 32 dB unfavourable-deviation rule, and the coincidence dip of each curve
marked where it actually falls below its own mass law.  Right panel: the
weighted single-number rating that shifting procedure produces.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from pubstyle import set_pub_style, savefig, style_axes, CATEGORICAL, COLORS
from figkit import panel_labels_inches

BANDS = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000, 2500, 3150, 4000, 5000], dtype=float)
# ISO 717-1 reference contour, 100 Hz .. 3150 Hz (16 bands), Rw = 52 dB
REF = np.array([33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56,
                56], dtype=float)
N_REF = REF.size
I500 = 7                                        # index of the 500 Hz band
MAX_UNFAV = 32.0

# name -> (surface mass kg/m2, high-frequency gain dB, split Hz,
#          mass-spring dip dB, mass-spring f0 Hz, coincidence dip dB, fc Hz)
WALLS = {
    "Laminated glass, 12 mm": (30.0, 0.0, 400.0, 0.0, 60.0, 10.0, 1600.0),
    "Double glazing 6-12-6": (30.0, 10.0, 330.0, 9.0, 215.0, 8.0, 3150.0),
    "Gypsum stud wall": (46.0, 13.0, 220.0, 8.0, 105.0, 6.0, 2500.0),
    "Stud wall + mineral wool": (46.0, 19.0, 220.0, 4.0, 105.0, 5.0, 2500.0),
}
MASS_C = 32.7                                   # field-incidence offset (dB)
DIP_LABEL_MIN = 8.0                             # only label the deep dips


def mass_law(ms, f):
    """Field-incidence mass law, 5 dB per octave rather than the ideal 6."""
    return 20.0 * np.log10(ms) + 15.0 * np.log10(f) - MASS_C


def bell(f, fc, width):
    return np.exp(-0.5 * (np.log(f / fc) / width) ** 2)


def transmission_loss(spec, f):
    ms, gain, split, d_msm, f0, d_co, fc = spec
    step = gain / (1.0 + (split / f) ** 3)
    return (mass_law(ms, f) + step - d_msm * bell(f, f0, 0.30)
            - d_co * bell(f, fc, 0.26))


def weighted_rating(r):
    """ISO 717-1 Rw: shift the contour until the unfavourable sum passes."""
    best = None
    for shift in range(-40, 41):
        unfav = np.clip((REF + shift) - r[:N_REF], 0.0, None).sum()
        if unfav <= MAX_UNFAV:
            best = shift
    return REF[I500] + best, REF + best


def main():
    set_pub_style()
    fine = np.logspace(np.log10(90.0), np.log10(5600.0), 400)

    curves = {n: transmission_loss(s, BANDS) for n, s in WALLS.items()}
    ratings = {n: weighted_rating(c)[0] for n, c in curves.items()}
    best = max(ratings, key=ratings.get)
    _, contour = weighted_rating(curves[best])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.44], wspace=0.20,
                          left=0.062, right=0.975, top=0.885, bottom=0.115)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ---- (a) transmission loss --------------------------------------------
    for (name, spec), col in zip(WALLS.items(), CATEGORICAL):
        lw = 2.0 if name == best else 1.3
        ax.plot(fine, transmission_loss(spec, fine), color=col, lw=lw,
                zorder=5 if name == best else 4)
        ax.plot(BANDS, curves[name], marker="o", ms=3.2, ls="none",
                mfc="white", mec=col, mew=1.0, zorder=6, label=name)
        _, _, _, _, _, d_co, fc = spec
        if d_co >= DIP_LABEL_MIN:
            r_fc = transmission_loss(spec, np.array([fc]))[0]
            ax.annotate("", xy=(fc, r_fc - 0.7), xytext=(fc, r_fc - 7.5),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col,
                                        shrinkA=0, shrinkB=0))
            ax.text(fc, r_fc - 8.1, f"coincidence dip\n$f_c$ = {fc:.0f} Hz",
                    ha="center", va="top", fontsize=6.4, color=col,
                    linespacing=1.3)

    ax.plot(BANDS[:N_REF], contour, color=COLORS["baseline"], lw=1.3,
            ls=(0, (5.0, 2.5)), zorder=3,
            label=f"ISO 717-1 contour shifted to $R_w$ = {ratings[best]:.0f} dB")

    ax.set_xscale("log")
    ax.set_xlim(90.0, 5600.0)
    ax.set_ylim(18.0, 80.0)
    ax.xaxis.set_major_locator(FixedLocator([125, 250, 500, 1000, 2000, 4000]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000
                                 else f"{v:g}")
    ax.xaxis.set_minor_locator(FixedLocator(BANDS))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=2.0)
    ax.set_xlabel("Third-octave centre frequency (Hz)")
    ax.set_ylabel("Sound reduction index $R$ (dB)")
    ax.grid(lw=0.5, alpha=0.55, which="major")
    ax.legend(loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.8, labelspacing=0.32)
    ax.set_title("Transmission loss per third-octave band", fontsize=9.0,
                 fontweight="bold")
    style_axes(ax)

    # ---- (b) weighted rating ----------------------------------------------
    names = list(WALLS)
    y = np.arange(len(names))[::-1]
    vals = [ratings[n] for n in names]
    ax_r.barh(y, vals, height=0.62, color=CATEGORICAL[:len(names)],
              edgecolor="white", linewidth=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax_r.text(v - 0.8, yi, f"{v:.0f}", ha="right", va="center",
                  fontsize=7.6, color="white", fontweight="bold", zorder=5)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([n.replace(", ", ",\n").replace(" + ", " +\n")
                          for n in names], fontsize=6.8)
    ax_r.set_xlim(0, max(vals) * 1.14)
    ax_r.set_xlabel("Weighted rating $R_w$ (dB)", fontsize=8.0)
    ax_r.tick_params(labelsize=7.0)
    ax_r.grid(axis="x", lw=0.5, alpha=0.6)
    ax_r.set_title("Single-number rating", fontsize=8.5, fontweight="bold")
    style_axes(ax_r)
    panel_labels_inches(fig, [ax, ax_r], dx=-0.40, dy=0.30)

    worst = min(ratings, key=ratings.get)
    savefig(fig, "419_acu_insulation.png")


if __name__ == "__main__":
    main()
