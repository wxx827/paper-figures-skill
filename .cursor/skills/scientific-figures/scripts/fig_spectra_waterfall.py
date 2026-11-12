"""In-situ FTIR waterfall: skewed 3D-style stack of absorbance spectra.

Ten spectra recorded while a polyester film is cured from 25 to 300 °C are
drawn as a skewed waterfall (each trace shifted in x and y, painted back to
front with an opaque underfill).  Parallelogram bands follow the same skew and
flag four diagnostic vibrations, the right colour bar encodes the annealing
temperature, and an inset placed in the silent 2800-2050 cm-1 window magnifies
the growing carbonyl envelope.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, savefig

# centre, half-width, label, label row (0 = lower, 1 = upper)
BANDS = [(3345, 210, r"$\nu$(O–H)", 0),
         (2925, 90, r"$\nu_{as}$(C–H)", 1),
         (1718, 55, r"$\nu$(C=O)", 0),
         (1602, 40, r"$\nu$(C=C)$_{ar}$", 1),
         (1102, 75, r"$\nu$(C–O–C)", 0)]

TEMPS = np.array([25, 60, 100, 140, 175, 205, 235, 260, 280, 300])

DX = 25.0     # horizontal skew per trace (cm-1)
DY = 0.60     # vertical offset per trace


def lorentz(x, x0, w):
    return 1.0 / (1.0 + ((x - x0) / (0.5 * w)) ** 2)


def gauss(x, x0, w):
    return np.exp(-0.5 * ((x - x0) / (w / 2.3548)) ** 2)


def spectrum(x, t, rng):
    """Absorbance of the film at annealing temperature ``t`` (°C)."""
    f = (t - TEMPS[0]) / (TEMPS[-1] - TEMPS[0])           # cure progress
    y = np.zeros_like(x)
    y += (0.62 - 0.45 * f) * gauss(x, 3345, 300)          # O–H condenses away
    y += 0.30 * lorentz(x, 2925, 55) + 0.19 * lorentz(x, 2858, 45)
    y += (0.14 + 0.80 * f ** 1.4) * lorentz(x, 1718, 38)  # C=O grows
    y += 0.11 * f * lorentz(x, 1772, 28)                  # anhydride shoulder
    y += (0.34 + 0.05 * f) * lorentz(x, 1602, 26)
    y += 0.22 * lorentz(x, 1508, 24) + 0.16 * lorentz(x, 1455, 30)
    y += (0.58 - 0.24 * f) * lorentz(x, 1102, 52)         # C–O–C consumed
    y += 0.20 * lorentz(x, 1240, 40) + 0.13 * lorentz(x, 830, 34)
    y += 0.035
    y += rng.normal(0.0, 0.0035, x.size)
    return y


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(19)

    x = np.linspace(3900.0, 650.0, 2600)
    n = len(TEMPS)
    shift = DX * (n - 1)
    cmap = LinearSegmentedColormap.from_list(
        "anneal", ["#2f5d8a", "#5b93b8", "#8fb69f", "#dcb765",
                   "#cf7f3c", "#8f2f2f"])

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    x_lo, x_hi = 4180.0, 400.0
    y_band = DY * (n - 1) + 1.24
    y_rows = (y_band + 0.30, y_band + 0.74)
    y_top = y_band + 1.28

    # --- skewed highlight bands -------------------------------------------
    for xc, w, lab, row in BANDS:
        x0, x1 = xc + w, xc - w
        ax.add_patch(Polygon([(x0, -0.05), (x1, -0.05),
                              (x1 + shift, y_band), (x0 + shift, y_band)],
                             closed=True, facecolor="#4a4a4a", alpha=0.06,
                             lw=0, zorder=1))
        xl = xc + shift
        ax.plot([xl, xl], [y_band, y_rows[row] - 0.06], color="#a5a5a5",
                lw=0.5, ls=(0, (3, 2.5)), zorder=2)
        ax.text(xl, y_rows[row], lab, ha="center", va="bottom", fontsize=7.2,
                color="#262626", fontweight="bold")

    # --- waterfall traces, painted back to front --------------------------
    for i in range(n - 1, -1, -1):
        y = spectrum(x, TEMPS[i], rng)
        xs, ys = x + DX * i, y + DY * i
        col = cmap(i / (n - 1))
        z = 10 + 3 * i
        ax.fill_between(xs, DY * i - 0.03, ys, color="white", zorder=z)
        ax.fill_between(xs, DY * i - 0.03, ys, color=col, alpha=0.13, lw=0,
                        zorder=z + 1)
        ax.plot(xs, ys, color=col, lw=0.95, zorder=z + 2)
        ax.text(xs[-1] - 22, DY * i - 0.02, f"{TEMPS[i]}", ha="left",
                va="bottom", fontsize=6.6, color=col, fontweight="bold",
                zorder=z + 3)
    ax.text(x[-1] - 22, DY * (n - 1) + 0.42, "°C", ha="left", va="bottom",
            fontsize=6.6, color="#4d4d4d")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.10, y_top)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_yticks([])
    ax.set_xticks(np.arange(3500, 600, -500))
    ax.set_xticks(np.arange(3900, 600, -100), minor=True)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)

    # --- inset: carbonyl envelope, parked in the silent window ------------
    def fx(v):
        return (v - x_lo) / (x_hi - x_lo)

    ins = ax.inset_axes([fx(2790), 0.485, fx(2070) - fx(2790), 0.345])
    xz = np.linspace(1830.0, 1630.0, 500)
    for i in range(n):
        ins.plot(xz, spectrum(xz, TEMPS[i], rng), color=cmap(i / (n - 1)),
                 lw=0.8)
    ins.set_xlim(1830, 1630)
    ins.set_ylim(0.04, 1.28)
    ins.set_xticks([1800, 1720, 1660])
    ins.set_yticks([])
    ins.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    ins.set_xlabel("cm$^{-1}$", fontsize=6.0, labelpad=0.5)
    ins.set_title("C=O envelope", fontsize=6.6, pad=2.0)
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor("#6b6b6b")
    ins.set_facecolor("white")
    ins.set_zorder(80)
    ins.annotate("", xy=(1722, 1.06), xytext=(1722, 0.34),
                 arrowprops=dict(arrowstyle="-|>", color="#8f2f2f", lw=0.8))
    ins.text(1712, 0.70, "+ cure", fontsize=6.0, color="#8f2f2f",
             ha="left", va="center", fontweight="bold")

    # --- temperature colour bar -------------------------------------------
    sm = ScalarMappable(cmap=cmap, norm=Normalize(TEMPS[0], TEMPS[-1]))
    cb = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.012, aspect=25)
    cb.set_label("Annealing temperature (°C)", fontsize=7.5)
    cb.set_ticks([25, 100, 175, 235, 300])
    cb.ax.tick_params(labelsize=6.5, width=0.6, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#555555")

    ax.set_title("In-situ FTIR of a polyester film during thermal curing",
                 fontsize=10, pad=8)
    fig.subplots_adjust(left=0.045, right=0.93, top=0.92, bottom=0.105)
    savefig(fig, "120_spectra_waterfall.png")


if __name__ == "__main__":
    main()
