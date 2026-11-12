"""H-mode pedestal against L-mode, and the pressure gradient it produces.

Density and both temperatures are drawn from one modified-hyperbolic-tangent
family, F(rho) = offset + (height/2) * [mtanh((rho_ped - rho) / (w/2), slope)
+ 1], with the L-mode curves given the same core shape but a pedestal width
wide enough that the transport barrier disappears.  Nothing about the barrier
is asserted: the pedestal top, the barrier width and the position of the steep
gradient are all found on the sampled arrays, by locating the peak of
|dp/drho| and taking the full width at half of that peak, and the annotated
pedestal height is the pressure difference across exactly that interval.

The right column follows the same two profiles through.  Total pressure is
p = n_e (T_e + T_i) k_B on the plotted grid, its radial derivative is a
central difference of that array, and the confinement panel evaluates an
ITER98(y,2)-style power law tau ~ P^-0.69 for both regimes so that the
H-factor quoted there is the ratio of the two curves rather than a constant
typed into the caption.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"          # ion temperature
ORANGE = "#c0603f"       # electron temperature
BLUE = "#3b5b92"         # density
PURPLE = "#8e6bbf"       # pressure gradient
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

KEV = 1.602176634e-16
RHO = np.linspace(0.0, 1.0, 1201)

#  key:  (core, sep, ped_top, rho_ped, width, core_slope)
H_MODE = dict(ne=(9.2, 0.6, 6.4, 0.955, 0.055, 0.30),
              te=(6.4, 0.10, 3.3, 0.955, 0.048, 0.85),
              ti=(7.6, 0.14, 3.6, 0.955, 0.052, 0.80))
L_MODE = dict(ne=(6.1, 0.6, 2.4, 0.86, 0.34, 0.55),
              te=(3.4, 0.10, 0.9, 0.86, 0.32, 1.25),
              ti=(3.8, 0.14, 1.0, 0.86, 0.32, 1.20))

P_HEAT = np.linspace(3.0, 40.0, 200)      # MW
TAU_L0, TAU_H0 = 0.048, 0.093             # s at 1 MW
P_EXP = -0.69
P_LH = 8.5                                # MW, L-H transition

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.06, 4.86, 4.14)
AUX1 = (6.62, 3.30, 2.80, 1.90)
AUX2 = (6.62, 1.06, 2.80, 1.52)


# ------------------------------------------------------------------ profiles
def mtanh(x, slope):
    ex, emx = np.exp(x), np.exp(-x)
    return ((1.0 + slope * x) * ex - emx) / (ex + emx)


def profile(spec, rho=RHO):
    core, sep, ped, rho_ped, width, cslope = spec
    z = (rho_ped - rho) / (0.5 * width)
    edge = sep + 0.5 * (ped - sep) * (mtanh(z, 0.0) + 1.0)
    inner = (core - ped) * np.clip(
        1.0 - (rho / rho_ped) ** 2, 0.0, None) ** (1.0 + cslope)
    return edge + inner


def bundle(spec):
    ne = profile(spec["ne"])                       # 1e19 m^-3
    te = profile(spec["te"])                       # keV
    ti = profile(spec["ti"])                       # keV
    p = ne * 1e19 * (te + ti) * KEV                # Pa
    dp = np.gradient(p, RHO)
    return dict(ne=ne, te=te, ti=ti, p=p, dp=dp)


def barrier(dat):
    """Locate the transport barrier from the pressure gradient array."""
    g = -dat["dp"]
    k = int(np.argmax(g))
    half = 0.5 * g[k]
    lo = k
    while lo > 0 and g[lo] > half:
        lo -= 1
    hi = k
    while hi < g.size - 1 and g[hi] > half:
        hi += 1
    return dict(rho_max=float(RHO[k]), grad_max=float(g[k]),
                rho_lo=float(RHO[lo]), rho_hi=float(RHO[hi]),
                width=float(RHO[hi] - RHO[lo]),
                p_top=float(dat["p"][lo]), p_sep=float(dat["p"][hi]))


def stored_energy(dat):
    """3/2 * integral p dV with dV ~ rho drho on a circular cross section."""
    return 1.5 * float(np.trapezoid(dat["p"] * RHO, RHO)) * 2.0


# ------------------------------------------------------------------ panels
def main_panel(ax, h, l, bar):
    axt = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    axt.set_zorder(1)

    axt.axvspan(bar["rho_lo"], bar["rho_hi"], color=lighten(GOLD, 0.82),
                lw=0.0, zorder=1)
    axt.text(0.5 * (bar["rho_lo"] + bar["rho_hi"]), 0.30, "pedestal",
             ha="center", va="bottom", fontsize=6.2, color="#8a6d10",
             rotation=90.0, zorder=2)

    ax.plot(RHO, h["ne"], color=BLUE, lw=1.9, zorder=6)
    ax.plot(RHO, l["ne"], color=BLUE, lw=1.2, ls=(0, (4.4, 2.2)), zorder=6)
    axt.plot(RHO, h["te"], color=ORANGE, lw=1.9, zorder=5)
    axt.plot(RHO, l["te"], color=ORANGE, lw=1.2, ls=(0, (4.4, 2.2)), zorder=5)
    axt.plot(RHO, h["ti"], color=RED, lw=1.9, zorder=5)
    axt.plot(RHO, l["ti"], color=RED, lw=1.2, ls=(0, (4.4, 2.2)), zorder=5)

    ax.text(0.045, h["ne"][0] + 0.35, "n$_e$  (H-mode)", ha="left",
            va="bottom", fontsize=6.6, color=BLUE)
    axt.text(0.045, h["ti"][0] + 0.22, "T$_i$", ha="left", va="bottom",
             fontsize=6.6, color=RED)
    axt.text(0.045, h["te"][0] - 0.30, "T$_e$", ha="left", va="top",
             fontsize=6.6, color=ORANGE)
    axt.text(0.10, l["ti"][0] + 0.18, "L-mode (dashed)", ha="left",
             va="bottom", fontsize=6.4, color=SUB)

    ax.annotate("pedestal top\n$\\rho$ = {:.3f},  p = {:.1f} kPa".format(
        bar["rho_lo"], bar["p_top"] / 1e3), xy=(bar["rho_lo"], 6.9),
        xytext=(0.40, 2.55), ha="left", va="bottom", fontsize=6.3,
        color="#8a6d10", linespacing=1.3, zorder=9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.88),
        arrowprops=dict(arrowstyle="-", color="#8a6d10", lw=0.7))
    ax.annotate("steep edge:  barrier width $\\Delta\\rho$ = {:.3f},\n"
                "max |dp/d$\\rho$| = {:.0f} kPa per unit $\\rho$".format(
                    bar["width"], bar["grad_max"] / 1e3),
                xy=(bar["rho_max"], 2.2), xytext=(0.40, 0.75), ha="left",
                va="bottom", fontsize=6.3, color=PURPLE, linespacing=1.3,
                zorder=9, bbox=dict(boxstyle="round,pad=0.22",
                                    facecolor="white", edgecolor="none",
                                    alpha=0.88),
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7))

    axi = ax.inset_axes([0.575, 0.685, 0.385, 0.285])
    lo, hi = 0.84, 1.0
    m = (RHO >= lo) & (RHO <= hi)
    axi.axvspan(bar["rho_lo"], bar["rho_hi"], color=lighten(GOLD, 0.82),
                lw=0.0)
    axi.plot(RHO[m], h["p"][m] / 1e3, color=RED, lw=1.5)
    axi.plot(RHO[m], l["p"][m] / 1e3, color=RED, lw=1.0, ls=(0, (3.2, 1.8)))
    axi.set_xlim(lo, hi)
    axi.set_ylim(0.0, float(h["p"][m].max()) / 1e3 * 1.15)
    axi.tick_params(labelsize=6, length=2.0, width=0.6, pad=1.5)
    axi.set_xticks([0.86, 0.93, 1.0])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#9a9a9a")
    axi.set_title("edge pressure magnified (kPa)", fontsize=6.0, color=SUB,
                  pad=2.0)
    axi.grid(lw=0.4, alpha=0.4)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 13.4)
    axt.set_ylim(0.0, 11.28)
    ax.set_xlabel("normalised minor radius $\\rho$ (-)")
    ax.set_ylabel("electron density n$_e$  (10$^{19}$ $m^{-3}$)", color=BLUE)
    axt.set_ylabel("temperature T$_e$, T$_i$  (keV)", color=RED)
    ax.tick_params(axis="y", colors=BLUE, labelsize=6.8)
    axt.tick_params(axis="y", colors=RED, labelsize=6.8)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.spines["left"].set_color(BLUE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    axt.spines["right"].set_color(RED)
    axt.spines["top"].set_visible(False)
    axt.spines["left"].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)

    handles = [Line2D([], [], color=INK, lw=1.9, label="H-mode"),
               Line2D([], [], color=INK, lw=1.2, ls=(0, (4.4, 2.2)),
                      label="L-mode"),
               Line2D([], [], color=lighten(GOLD, 0.70), lw=6.0,
                      label="edge transport barrier")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.4,
              frameon=False, bbox_to_anchor=(0.012, 0.012), handlelength=1.8,
              labelspacing=0.30, borderaxespad=0.0)
    ax.set_title("Kinetic profiles: the pedestal is the whole difference",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)


def pressure_panel(ax, h, l, bar):
    axg = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    axg.set_zorder(1)
    axg.axvspan(bar["rho_lo"], bar["rho_hi"], color=lighten(GOLD, 0.82),
                lw=0.0, zorder=1)

    ax.plot(RHO, h["p"] / 1e3, color=RED, lw=1.7, zorder=6)
    ax.plot(RHO, l["p"] / 1e3, color=RED, lw=1.1, ls=(0, (4.0, 2.2)),
            zorder=6)
    axg.plot(RHO, -h["dp"] / 1e3, color=PURPLE, lw=1.5, zorder=5)
    axg.plot(RHO, -l["dp"] / 1e3, color=PURPLE, lw=1.0, ls=(0, (4.0, 2.2)),
             zorder=5)
    axg.plot([bar["rho_max"]], [bar["grad_max"] / 1e3], marker="v", ms=4.2,
             color=PURPLE, zorder=7)
    axg.text(bar["rho_max"] - 0.03, bar["grad_max"] / 1e3,
             "{:.0f} kPa\nper unit $\\rho$".format(bar["grad_max"] / 1e3),
             ha="right", va="top", fontsize=6.0, color=PURPLE,
             linespacing=1.25, zorder=8)
    ax.text(0.035, 0.955, "p = n$_e$(T$_e$ + T$_i$)k$_B$", ha="left",
            va="top", fontsize=6.4, color=RED, transform=ax.transAxes)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(h["p"].max()) / 1e3 * 1.18)
    axg.set_ylim(0.0, bar["grad_max"] / 1e3 * 1.30)
    ax.set_xlabel("normalised minor radius $\\rho$ (-)", fontsize=7.4)
    ax.set_ylabel("pressure p (kPa)", fontsize=7.4, color=RED)
    axg.set_ylabel("|dp/d$\\rho$| (kPa per unit $\\rho$)", fontsize=7.0,
                   color=PURPLE)
    ax.tick_params(axis="y", colors=RED, labelsize=6.6)
    axg.tick_params(axis="y", colors=PURPLE, labelsize=6.6)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.spines["left"].set_color(RED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    axg.spines["right"].set_color(PURPLE)
    axg.spines["top"].set_visible(False)
    axg.spines["left"].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("Pressure, and where its gradient spikes", fontsize=8.2,
                 fontweight="bold", color=INK)


def tau_panel(ax, hfac):
    tl = TAU_L0 * P_HEAT ** P_EXP * 1e3
    th = TAU_H0 * P_HEAT ** P_EXP * 1e3
    ax.plot(P_HEAT, tl, color=SUB, lw=1.4, ls=(0, (4.0, 2.2)), zorder=5)
    ax.plot(P_HEAT[P_HEAT >= P_LH], th[P_HEAT >= P_LH], color=GREEN, lw=1.8,
            zorder=6)
    ax.fill_between(P_HEAT[P_HEAT >= P_LH], tl[P_HEAT >= P_LH],
                    th[P_HEAT >= P_LH], color=lighten(GREEN, 0.86), lw=0.0,
                    zorder=3)
    ax.axvline(P_LH, color=GOLD, lw=1.0, ls=(0, (3.4, 2.0)), zorder=4)
    ax.text(P_LH * 1.06, 0.985, "L-H threshold\n{:.1f} MW".format(P_LH),
            transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=6.1, color="#8a6d10", linespacing=1.25)
    ax.text(30.0, float(np.interp(30.0, P_HEAT, th)) * 1.14, "H-mode",
            ha="right", va="bottom", fontsize=6.4, color=GREEN)
    ax.text(13.0, float(np.interp(13.0, P_HEAT, tl)) * 0.84, "L-mode",
            ha="right", va="top", fontsize=6.4, color=SUB)
    ax.text(0.035, 0.055, "H-factor = {:.2f}".format(hfac),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=GREEN)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.0, 42.0)
    ax.set_ylim(2.6, 62.0)
    ax.set_xlabel("heating power P$_{heat}$ (MW)", fontsize=7.4)
    ax.set_ylabel("$\\tau_E$ (ms)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.set_xticks([3, 5, 10, 20, 40])
    ax.set_xticklabels(["3", "5", "10", "20", "40"])
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.set_yticklabels(["3", "5", "10", "20", "40"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    ax.set_title("Confinement degrades with power in both regimes",
                 fontsize=8.2, fontweight="bold", color=INK)


def main():
    set_pub_style()
    np.random.default_rng(920)
    h, l = bundle(H_MODE), bundle(L_MODE)
    bar = barrier(h)
    bar_l = barrier(l)
    hfac = TAU_H0 / TAU_L0
    w_h, w_l = stored_energy(h), stored_energy(l)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN)
    ax_p = axes_inches(fig, AUX1)
    ax_t = axes_inches(fig, AUX2)
    main_panel(ax_m, h, l, bar)
    pressure_panel(ax_p, h, l, bar)
    tau_panel(ax_t, hfac)

    panel_labels_inches(fig, [ax_m, ax_p, ax_t], dx=-0.62, dy=0.13)
    savefig(fig, "920_fus_plasma_profiles.png")


if __name__ == "__main__":
    main()
