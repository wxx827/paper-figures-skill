"""A year of chlorophyll in the water column, with the physics that sets it.

The main panel is a time-depth section of chlorophyll a: depth increases
downward, as it must for a section, and the two overlaid curves are the reason
the pigment sits where it does.  The mixed-layer depth shoals from winter
convection to a twenty-metre summer cap, which is what lets the spring bloom
happen at all, and the euphotic depth (1% of surface irradiance) is pushed up
by the bloom's own self-shading.  Once the mixed layer caps and surface
nitrate is gone, the biomass retreats to a subsurface maximum wedged between
the nutricline and the light limit.

The side panels are cut from the same field.  The size spectra are sampled on
three days located from the chlorophyll section itself - the bloom peak and
45 days either side - and their slopes are fitted, not asserted; the nutrient
panel is drawn as percent of the winter maximum so that nitrate, silicate and
phosphate, which differ by more than an order of magnitude in absolute
concentration, can honestly share one axis.
"""
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    halo_text)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
MLD_COL = "#1a1a1a"
ZEU_COL = "#c9a227"
NUTRIENTS = [("nitrate", "#3b5b92", 12.4), ("silicate", "#2a7f7a", 7.8),
             ("phosphate", "#8e6bbf", 0.82)]
SPEC_COL = ["#9cc3dd", "#b23a48", "#3d6b8e"]

FW, FH = 10.4, 6.4
MAIN = (0.82, 1.06, 4.62, 4.10)
AUX1 = (6.90, 3.10, 2.70, 2.06)
AUX2 = (6.90, 1.06, 2.70, 1.44)

ZMAX = 140.0
MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
                        334], dtype=float)
MONTH_LAB = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
SIZE_BINS = np.logspace(np.log10(2.0), np.log10(200.0), 14)


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def mld_of(t):
    """Mixed-layer depth (m); deep winter convection, shallow summer cap."""
    return 66.0 + 46.0 * np.cos(2.0 * np.pi * t / 365.0)


def build_section(rng):
    t = np.linspace(0.0, 364.0, 240)
    z = np.linspace(0.0, ZMAX, 180)
    T, Z = np.meshgrid(t, z)
    mld = mld_of(T)

    spring = (5.05 * np.exp(-0.5 * ((T - 108.0) / 14.0) ** 2)
              * np.exp(-0.5 * (Z / (0.62 * mld + 8.0)) ** 2))
    z_scm = np.clip(mld + 13.0, 28.0, 64.0)
    summer = (2.85 * np.exp(-0.5 * ((T - 212.0) / 46.0) ** 2)
              * np.exp(-0.5 * ((Z - z_scm) / 10.5) ** 2))
    autumn = (2.20 * np.exp(-0.5 * ((T - 288.0) / 17.0) ** 2)
              * np.exp(-0.5 * (Z / 30.0) ** 2))
    chl = spring + summer + autumn + 0.055 * np.exp(-Z / 55.0)
    chl = chl * np.exp(-np.clip(Z - 105.0, 0.0, None) / 12.0)
    chl = chl * np.exp(rng.normal(0.0, 0.045, chl.shape))

    surf = chl[z <= 12.0].mean(axis=0)
    # Self-shading: the euphotic depth follows the pigment above it.
    kd = 0.041 + 0.098 * surf ** 0.72
    zeu = np.log(100.0) / kd
    peak = int(np.argmax(surf))
    return dict(t=t, z=z, chl=chl, mld=mld_of(t), zeu=zeu, surf=surf,
                peak=peak, z_scm=np.clip(mld_of(t) + 13.0, 28.0, 64.0))


def nutrient_curves(sec):
    """Percent of the winter maximum left in the surface layer."""
    t = sec["t"]
    draw = np.cumsum(np.clip(sec["surf"] - 0.20, 0.0, None))
    draw = draw / draw.max()
    winter = 0.5 * (1.0 - np.cos(np.pi * np.clip((t - 248.0) / 114.0,
                                                 0.0, 1.0)))
    eff = {"nitrate": 0.92, "silicate": 0.99, "phosphate": 0.74}
    recov = {"nitrate": 1.00, "silicate": 0.74, "phosphate": 1.25}
    out = {}
    for name, col, wmax in NUTRIENTS:
        back = np.clip(winter * recov[name], 0.0, 1.0)
        left = 100.0 * (1.0 - eff[name] * draw * (1.0 - back))
        out[name] = (col, np.clip(left, 0.0, 100.0), wmax)
    return out


def size_spectra(sec, rng):
    """Biomass size spectra on three days located from the section."""
    days = [max(sec["peak"] - 30, 0), sec["peak"],
            min(sec["peak"] + 30, sec["t"].size - 1)]
    labels = ["pre-bloom", "bloom peak", "post-bloom"]
    out = []
    d = SIZE_BINS
    for k, (day, lab) in enumerate(zip(days, labels)):
        chl_s = sec["surf"][day]
        frac_large = float(np.clip((chl_s - 0.35) / 2.6, 0.0, 1.0))
        s = -1.30 + 0.86 * frac_large
        raw = d ** s
        raw = raw / np.trapezoid(raw, np.log10(d))
        biomass = 58.0 * chl_s                       # mg C per cubic metre
        b = biomass * raw * np.exp(rng.normal(0.0, 0.10, d.size))
        slope = float(np.polyfit(np.log10(d), np.log10(b), 1)[0])
        out.append(dict(day=int(sec["t"][day]), label=lab, b=b, slope=slope,
                        color=SPEC_COL[k], chl=chl_s, biomass=biomass))
    return out


def draw_section(ax, sec):
    top = np.ceil(sec["chl"].max() * 2.0) / 2.0
    levels = np.arange(0.0, top + 0.25, 0.25)
    cf = ax.contourf(sec["t"], sec["z"], sec["chl"], levels=levels,
                     cmap=get_cmap("cb_sequential_green"), extend="max",
                     zorder=1)
    rasterize(cf)
    cf.set_edgecolor("face")

    ax.plot(sec["t"], sec["mld"], color=MLD_COL, lw=1.6, zorder=5)
    ax.plot(sec["t"], sec["mld"], color="white", lw=2.9, zorder=4, alpha=0.7)
    ax.plot(sec["t"], sec["zeu"], color=ZEU_COL, lw=1.4,
            ls=(0, (5.0, 2.4)), zorder=5)

    halo_text(ax, 42.0, mld_of(42.0) + 5.0, "mixed-layer depth", ha="left",
              va="top", fontsize=6.4, color=MLD_COL, halo="white",
              halo_lw=2.4, zorder=7)
    halo_text(ax, 332.0, float(np.interp(332.0, sec["t"], sec["zeu"])) - 4.0,
              "euphotic depth (1% $E_0$)", ha="center", va="bottom",
              fontsize=6.4, color="#a04c18", halo="white", halo_lw=2.4,
              zorder=7)
    halo_text(ax, sec["t"][sec["peak"]], 4.0, "spring bloom", ha="center",
              va="top", fontsize=6.6, color="white", halo="#0b3d2a",
              halo_lw=2.2, zorder=7)
    halo_text(ax, 218.0, 40.0, "subsurface chlorophyll\nmaximum",
              ha="center", va="center", fontsize=6.4, color=INK,
              halo="white", halo_lw=2.4, zorder=7, linespacing=1.25)

    ax.set_xlim(0.0, 364.0)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_xticks(MONTH_START)
    ax.set_xticklabels(MONTH_LAB)
    ax.set_xlabel("month", fontsize=8.0)
    ax.set_ylabel("depth (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Chlorophyll a section through one seasonal cycle",
                 fontsize=8.5, fontweight="bold")
    return cf


def draw_spectra(ax, spec):
    for sp in spec:
        ax.plot(SIZE_BINS, sp["b"], color=sp["color"], lw=1.3, marker="o",
                ms=2.8, mfc="white", mew=0.7, zorder=4,
                label="{:s}  (slope {:.2f})".format(sp["label"], sp["slope"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    allb = np.concatenate([sp["b"] for sp in spec])
    ax.set_ylim(allb.min() * 0.35, allb.max() * 24.0)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left",
              handlelength=1.5, borderpad=0.15)
    ax.axvspan(20.0, 200.0, color="#f5f5f5", zorder=0)
    ax.text(0.97, 0.60, "microplankton\nand chain diatoms",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=MUTED, linespacing=1.25, zorder=2)
    ax.set_xlabel("equivalent spherical diameter ($\\mu$m)", fontsize=7.8)
    ax.set_ylabel("biomass (mg C m$^{-3}$)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Size spectrum flattens through the bloom", fontsize=8.5,
                 fontweight="bold")


def draw_nutrients(ax, sec, nut):
    for name, col, wmax in NUTRIENTS:
        _, left, _ = nut[name]
        ax.plot(sec["t"], left, color=col, lw=1.3, zorder=4,
                label="{:s} ({:.1f} $\\mu$mol/kg)".format(name, wmax))
    ax.axvline(sec["t"][sec["peak"]], color="#9a9a9a", lw=0.8,
               ls=(0, (2.6, 2.2)), zorder=2)
    ax.text(sec["t"][sec["peak"]] + 5.0, 96.0, "bloom peak", fontsize=6.2,
            color=MUTED, ha="left", va="top", zorder=5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left",
              handlelength=1.4, borderpad=0.15, labelspacing=0.25)
    ax.set_xlim(0.0, 364.0)
    ax.set_ylim(0.0, 108.0)
    ax.set_xticks(MONTH_START)
    ax.set_xticklabels(MONTH_LAB)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("month", fontsize=7.8)
    ax.set_ylabel("left in surface layer\n(% of winter maximum)",
                  fontsize=7.4)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Nutrient drawdown", fontsize=8.5, fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(693)
    sec = build_section(rng)
    nut = nutrient_curves(sec)
    spec = size_spectra(sec, rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_s = axes_inches(fig, MAIN)
    ax_p = axes_inches(fig, AUX1)
    ax_n = axes_inches(fig, AUX2)

    cf = draw_section(ax_s, sec)
    safe_colorbar(fig, cf, ax_s, label="chlorophyll a (mg m$^{-3}$)",
                  width=0.11, pad=0.09,
                  ticks=np.arange(0.0, sec["chl"].max() + 0.5, 1.0))
    draw_spectra(ax_p, spec)
    draw_nutrients(ax_n, sec, nut)

    panel_labels_inches(fig, [ax_s, ax_p, ax_n], dx=-0.46, dy=0.20)

    peak_day = int(sec["t"][sec["peak"]])
    scm_depth = float(np.mean(sec["z_scm"][(sec["t"] > 180)
                                           & (sec["t"] < 260)]))
    no3_left = nut["nitrate"][1][sec["peak"] + 12]

    print("peak day={:d} chl={:.2f} mld={:.1f} zeu={:.1f} scm={:.1f} "
          "slopes={:s}".format(peak_day, sec["surf"].max(),
                               sec["mld"][sec["peak"]],
                               sec["zeu"][sec["peak"]], scm_depth,
                               str(["{:.2f}".format(s["slope"])
                                    for s in spec])))
    savefig(fig, "693_mar_plankton_bloom.png")


if __name__ == "__main__":
    main()
