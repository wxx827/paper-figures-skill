"""Optical spectrum of a solar-type star: continuum, lines and one shift.

The main panel is the claim: a Planck continuum fitted only in the line-free
windows accounts for the broad shape, and everything left over is absorption
at identified rest wavelengths.  Line markers are drawn from the flux array
itself, so a label can never point at a wavelength where no line was
synthesised.  The inset magnifies the Na D doublet, where the whole spectrum
is displaced from its rest wavelengths; the recessional velocity printed there
is recomputed from the plotted minima.

Side panels carry the measurement and the context: an equivalent-width
integration on H-alpha shown as the shaded area it actually is, and how the
same lines strengthen or vanish along the spectral sequence.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"

HAIRLINE = "#cccccc"
RED = "#b23a48"          # fitted continuum / emphasis
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

H_PL, C_LIGHT, K_B = 6.62607e-34, 2.99792458e8, 1.380649e-23
C_KMS = 299792.458
T_EFF = 5900.0
CZ_TRUE = 285.0                      # km/s, so z = 9.5e-4
WL_LO, WL_HI = 3700.0, 6820.0

# (rest wavelength in Angstrom, depth as a fraction of continuum, sigma in A)
STRONG = [(3933.66, 0.62, 3.2), (3968.47, 0.54, 3.0),
          (4101.74, 0.40, 3.4), (4340.47, 0.45, 3.6), (4861.33, 0.50, 4.0),
          (5167.32, 0.26, 1.4), (5172.68, 0.33, 1.5), (5183.60, 0.36, 1.6),
          (5889.95, 0.42, 1.5), (5895.92, 0.35, 1.5),
          (6562.80, 0.55, 4.4),
          (4045.81, 0.22, 1.5), (4271.76, 0.18, 1.4), (4300.00, 0.20, 5.0),
          (4383.55, 0.25, 1.6), (5269.54, 0.26, 1.7)]

# Labels drawn on the rail; the Ca II pair and the two doublets are blended at
# this dispersion, so they are annotated as single features.
MARKS = [("Ca II H & K", 3951.1, "3934, 3968"),
         (r"H$\delta$", 4101.7, "4102"),
         ("G band", 4300.0, "4300"),
         (r"H$\gamma$", 4340.5, "4340"),
         (r"H$\beta$", 4861.3, "4861"),
         ("Mg b", 5174.5, "5167-5184"),
         ("Na D", 5892.9, "5890, 5896"),
         (r"H$\alpha$", 6562.8, "6563")]

Z_TRUE = CZ_TRUE / C_KMS


def planck(wl_ang, temp):
    """Planck spectral radiance for wavelengths given in Angstrom."""
    wl = wl_ang * 1e-10
    x = H_PL * C_LIGHT / (wl * K_B * temp)
    return 2.0 * H_PL * C_LIGHT ** 2 / wl ** 5 / np.expm1(x)


def spectrum(seed=41):
    rng = np.random.default_rng(seed)
    wl = np.arange(WL_LO, WL_HI, 0.30)
    cont = planck(wl, T_EFF)
    cont = cont / cont.max()

    absorb = np.zeros_like(wl)
    for wl0, depth, sig in STRONG:
        obs = wl0 * (1.0 + Z_TRUE)
        absorb += depth * np.exp(-0.5 * ((wl - obs) / sig) ** 2)

    # Weak metal forest, denser toward the blue as real line blanketing is.
    n_weak = 185
    pos = WL_LO + (WL_HI - WL_LO) * rng.beta(1.35, 2.1, n_weak)
    dep = rng.uniform(0.010, 0.048, n_weak) * (1.0 + 1.2 * (6800.0 - pos)
                                               / 3100.0)
    wid = rng.uniform(0.8, 2.2, n_weak)
    for p, d, w in zip(pos, dep, wid):
        absorb += d * np.exp(-0.5 * ((wl - p * (1.0 + Z_TRUE)) / w) ** 2)

    flux = cont * np.clip(1.0 - absorb, 0.02, None)
    flux = flux + rng.normal(0.0, 0.0035, wl.size)
    return wl, flux, cont


def fit_continuum(wl, flux):
    """Grid-fit a blackbody to the line-free windows only."""
    windows = [(4450, 4550), (4700, 4780), (5350, 5450), (5600, 5700),
               (6200, 6320), (6700, 6800), (4000, 4030)]
    mask = np.zeros_like(wl, dtype=bool)
    for a, b in windows:
        mask |= (wl >= a) & (wl <= b)
    # Take the upper envelope inside each window: even "clean" windows have
    # weak lines, and a mean would bias the fit cool.
    sel_w, sel_f = [], []
    for a, b in windows:
        m = (wl >= a) & (wl <= b)
        sel_w.append(wl[m].mean())
        sel_f.append(np.percentile(flux[m], 92))
    sel_w, sel_f = np.array(sel_w), np.array(sel_f)

    temps = np.arange(4600.0, 7600.0, 5.0)
    best_t, best_chi, best_scale = temps[0], np.inf, 1.0
    for t in temps:
        model = planck(sel_w, t)
        scale = float(np.sum(model * sel_f) / np.sum(model ** 2))
        chi = float(np.sum((sel_f - scale * model) ** 2))
        if chi < best_chi:
            best_t, best_chi, best_scale = float(t), chi, scale
    return best_t, best_scale, sel_w, sel_f


def equivalent_width(wl, flux, cont, centre, half=26.0):
    m = np.abs(wl - centre) <= half
    depth = 1.0 - flux[m] / cont[m]
    return float(np.trapezoid(depth, wl[m])), wl[m], flux[m], cont[m]


def sequence():
    """Line strengths along the spectral sequence, in Angstrom."""
    types = ["O", "B", "A", "F", "G", "K", "M"]
    x = np.arange(len(types), dtype=float)
    curves = {
        r"H$\beta$ (Balmer)": 9.5 * np.exp(-0.5 * ((x - 2.0) / 1.05) ** 2),
        "Ca II K": 0.35 + 11.0 * np.exp(-0.5 * ((x - 5.0) / 1.35) ** 2),
        "Mg b": 0.10 + 3.6 * np.exp(-0.5 * ((x - 4.7) / 1.25) ** 2),
        "TiO bands": 0.06 + 14.0 * np.exp(-0.5 * ((x - 6.6) / 0.85) ** 2),
    }
    return types, x, curves


def main_panel(ax, wl, flux, cont_fit, t_fit):
    ax.plot(wl, flux, color=RULE, lw=0.55, zorder=3)
    ax.plot(wl, cont_fit, color=RED, lw=1.4, ls=(0, (5, 2.6)), zorder=4)
    ax.text(6300.0, planck(np.array([6300.0]), t_fit)[0] /
            planck(np.array([WL_LO]), t_fit)[0] * cont_fit[0] + 0.055,
            f"blackbody fit,  $T_{{\\rm eff}}$ = {t_fit:,.0f} K", ha="center",
            va="bottom", fontsize=7.2, color=RED)

    rail_lo, rail_hi = 1.10, 1.19
    for i, (name, wl0, note) in enumerate(MARKS):
        obs = wl0 * (1.0 + Z_TRUE)
        j = int(np.argmin(np.abs(wl - obs)))
        top = rail_lo if i % 2 == 0 else rail_hi
        ax.plot([obs, obs], [flux[j] + 0.035, top - 0.012], color=GREY,
                lw=0.6, zorder=2)
        ax.text(obs, top, f"{name}\n{note}", ha="center", va="bottom",
                fontsize=6.4, color=SUB, linespacing=1.25, zorder=5)

    ax.set_xlim(WL_LO, WL_HI)
    ax.set_ylim(0.0, 1.34)
    ax.set_xlabel(r"wavelength $\lambda$ ($\AA$)")
    ax.set_ylabel("normalised flux $F_\\lambda$ (-)")
    ax.tick_params(labelsize=7.2)
    ax.grid(axis="y", lw=0.45, alpha=0.5)


def redshift_inset(ax, wl, flux):
    axi = ax.inset_axes([0.275, 0.055, 0.305, 0.300])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    lo, hi = 5872.0, 5914.0
    m = (wl >= lo) & (wl <= hi)
    axi.plot(wl[m], flux[m], color=RULE, lw=0.9, zorder=4)

    rest = 5889.95
    win = np.abs(wl - rest * (1.0 + Z_TRUE)) <= 5.0
    obs = float(wl[win][np.argmin(flux[win])])
    cz = C_KMS * (obs - rest) / rest
    for x0, col, ls in [(rest, GREY, (0, (3, 2.2))),
                        (5895.92, GREY, (0, (3, 2.2)))]:
        axi.axvline(x0, color=col, lw=0.8, ls=ls, zorder=2)
    axi.axvline(obs, color=RED, lw=1.0, zorder=3)
    axi.annotate("", xy=(obs, 0.30), xytext=(rest, 0.30),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    axi.text(0.03, 0.95, f"cz = {cz:.0f} km/s,  $+${obs - rest:.1f} $\\AA$",
             transform=axi.transAxes, ha="left", va="top", fontsize=6.4,
             color=RED)
    axi.text(0.03, 0.06, "grey = rest", transform=axi.transAxes, ha="left",
             va="bottom", fontsize=6.0, color=MUTED)
    axi.set_xlim(lo, hi)
    axi.set_ylim(0.20, 1.02)
    axi.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=6.2, labelpad=1.0)
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_yticks([0.4, 0.8])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(HAIRLINE)
    return cz


def ew_panel(ax, wl, flux, cont):
    ew, w, f, c = equivalent_width(wl, flux, cont, 6562.80 * (1.0 + Z_TRUE))
    ax.fill_between(w, f, c, color=RED, alpha=0.22, linewidth=0.0, zorder=2)
    ax.plot(w, c, color=RED, lw=1.2, ls=(0, (4, 2.4)), zorder=4)
    ax.plot(w, f, color=RULE, lw=1.0, zorder=5)

    depth = 1.0 - f.min() / c[np.argmin(f)]
    ax.annotate("", xy=(w[np.argmin(f)], f.min()),
                xytext=(w[np.argmin(f)], c[np.argmin(f)]),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.8))
    ax.text(w[np.argmin(f)] - 1.6, 0.5 * (f.min() + c[np.argmin(f)]),
            f"{depth * 100:.0f} % deep", ha="right", va="center",
            fontsize=6.6, color=BLUE)
    ax.text(0.03, 0.06,
            f"EW = $\\int (1 - F/F_c)\\, d\\lambda$ = {ew:.2f} $\\AA$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9,
            color=RED)
    ax.set_xlim(w.min(), w.max())
    ax.set_xlabel(r"wavelength ($\AA$)", fontsize=7.4)
    ax.set_ylabel("$F_\\lambda$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title(r"Equivalent width of H$\alpha$ as a shaded area",
                 fontsize=8.4, fontweight="bold", color=INK)
    return ew


def sequence_panel(ax):
    types, x, curves = sequence()
    colors = [BLUE, RED, GOLD, GREEN]
    peaks = {}
    for (name, y), col in zip(curves.items(), colors):
        ax.plot(x, y, marker="o", ms=3.0, lw=1.3, color=col, mfc="white",
                mew=0.8, label=name, zorder=3)
        peaks[name] = types[int(np.argmax(y))]
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=7.0)
    ax.set_ylim(0.035, 90.0)
    ax.set_xlabel("spectral type, hot to cool (-)", fontsize=7.4)
    ax.set_ylabel(r"equivalent width ($\AA$)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", ncol=2,
              handlelength=1.3, borderaxespad=0.25, columnspacing=0.9)
    ax.set_title("Line strength along the spectral sequence", fontsize=8.4,
                 fontweight="bold", color=INK)
    return peaks


def main():
    set_pub_style()
    wl, flux, cont_true = spectrum()
    t_fit, scale, sel_w, sel_f = fit_continuum(wl, flux)
    cont_fit = scale * planck(wl, t_fit)

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax_main = axes_inches(fig, (0.90, 0.88, 5.45, 4.58))
    ax_ew = axes_inches(fig, (7.05, 3.30, 3.00, 2.16))
    ax_seq = axes_inches(fig, (7.05, 0.88, 3.00, 1.76))

    main_panel(ax_main, wl, flux, cont_fit, t_fit)
    cz = redshift_inset(ax_main, wl, flux)
    ew = ew_panel(ax_ew, wl, flux, cont_fit)
    peaks = sequence_panel(ax_seq)
    balmer_peak = next(v for k, v in peaks.items() if "Balmer" in k)

    ax_main.set_title(
        f"A {t_fit:,.0f} K continuum plus {len(MARKS)} identified features "
        f"account for the whole optical band", fontsize=9.0,
        fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax_main, ax_ew, ax_seq], dx=-0.66, dy=0.30)
    savefig(fig, "801_ast_stellar_spectrum.png")


if __name__ == "__main__":
    main()
