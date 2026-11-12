"""Scale-up across four bioreactor sizes under a constant P/V criterion.

Every number on this page comes out of one geometric series: four working
volumes, each vessel geometrically similar (H = T, impeller D = T/3), with the
agitator speed at each scale solved so that the specific power draw
P/V = Np rho N^3 D^5 / V matches the bench reference.  That single choice then
fixes everything else, and the main panel shows what it costs: P/V is flat by
construction, but the mixing time grows as T^(2/3), the impeller tip shear
grows as T^(1/3), and kLa drifts upward because holding the aeration rate in
vvm makes the superficial gas velocity scale with the vessel diameter.  Four
parameters, one degree of freedom - they cannot all be held.  The supporting
panels show where the pilot vessel sits on the kLa(agitation, aeration)
surface, and the mixing-time power law refitted from the four computed points.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"         # kLa
OCHRE = "#c0603f"        # specific power P/V
GREEN = "#2e7d5b"        # mixing time
PURPLE = "#8e6bbf"       # tip shear
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

RHO = 1000.0             # kg/m3
NP = 5.0                 # Rushton power number
FILL = 0.70              # working volume / vessel volume
DELTA = 1.0e-3           # m, boundary-layer thickness at the impeller tip
VVM = 0.40               # min^-1, aeration held constant in vvm
N_REF_RPM = 600.0        # rpm at the bench reference
REF = 1                  # index of the 5 L bench reference

V_WORK = np.array([5.0e-4, 5.0e-3, 0.50, 10.0])          # m3
SCALES = ["0.5 L flask", "5 L bench", "500 L pilot", r"10 m$^3$ production"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 0.95, 4.62, 4.55)
TOP = (7.82, 3.61, 2.68, 1.89)
BOT = (7.82, 0.95, 2.68, 1.89)


def geometry(v_work):
    t = (4.0 * (v_work / FILL) / np.pi) ** (1.0 / 3.0)
    return t, t / 3.0


def power_per_volume(n, d, v_work):
    return NP * RHO * n ** 3 * d ** 5 / v_work


def speed_for_pv(pv, d, v_work):
    return (pv * v_work / (NP * RHO * d ** 5)) ** (1.0 / 3.0)


def kla_hour(pv, vs):
    """Van't Riet correlation, s^-1 -> h^-1."""
    return 3600.0 * 0.026 * pv ** 0.40 * vs ** 0.50


def mixing_time(t_vessel, d, pv):
    """Nienow 95 % homogenisation time, s."""
    return (5.9 * t_vessel ** (2.0 / 3.0) * (pv / RHO) ** (-1.0 / 3.0) *
            (d / t_vessel) ** (-1.0 / 3.0))


def scale_table():
    t_v, d = geometry(V_WORK)
    n_ref = N_REF_RPM / 60.0
    pv_ref = power_per_volume(n_ref, d[REF], V_WORK[REF])
    n = speed_for_pv(pv_ref, d, V_WORK)
    pv = power_per_volume(n, d, V_WORK)
    area = np.pi / 4.0 * t_v ** 2
    vs = VVM * V_WORK / 60.0 / area
    return dict(t=t_v, d=d, n=n, rpm=n * 60.0, pv=pv, vs=vs,
                kla=kla_hour(pv, vs), theta=mixing_time(t_v, d, pv),
                shear=np.pi * n * d / DELTA, tip=np.pi * n * d)


SERIES = [("kLa", "kla", BLUE, "o", r"h$^{-1}$"),
          ("specific power P/V", "pv", OCHRE, "s", r"W/m$^3$"),
          ("mixing time " + r"$\theta_{95}$", "theta", GREEN, "^", "s"),
          ("tip shear rate", "shear", PURPLE, "D", r"s$^{-1}$")]


def main_panel(ax, tab):
    x = np.arange(len(SCALES), dtype=float)
    ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.axvspan(REF - 0.30, REF + 0.30, facecolor=lighten(GREY, 0.90),
               edgecolor="none", zorder=0)
    ax.text(REF, 13.5, "reference\nscale", ha="center", va="top",
            fontsize=6.6, color=SUB, linespacing=1.30)

    for _name, key, col, mk, unit in SERIES:
        v = tab[key]
        ratio = v[-1] / v[REF]
        ax.plot(x, v / v[REF], color=col, lw=1.7, marker=mk, ms=5.2,
                mfc="white", mec=col, mew=1.2, zorder=5)
        fmt = "{:.0f}" if v[-1] >= 100.0 else "{:.1f}"
        ax.text(x[-1] + 0.080, ratio * (1.05 if key == "pv" else 1.0),
                r"$\times$" + "{:.2f}   ".format(ratio) +
                fmt.format(v[-1]) + " " + unit, ha="left",
                va="bottom" if key == "pv" else "center", fontsize=6.6,
                color=col)

    ax.annotate("criterion: P/V held at {:.0f} ".format(tab["pv"][REF]) +
                r"W/m$^3$" + "\nagitation solved per scale, " +
                "{:.0f} -> {:.0f} rpm".format(tab["rpm"][0], tab["rpm"][-1]),
                xy=(1.62, 1.0), xytext=(0.02, 3.6), ha="left", va="bottom",
                fontsize=7.0, color=INK, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.text(0.5, 0.985,
            "holding one parameter constant forces the other three to move",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
            color=SUB)

    ax.set_yscale("log")
    ax.set_xlim(-0.42, len(SCALES) - 0.05)
    ax.set_ylim(0.055, 26.0)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALES, fontsize=7.4)
    ax.set_xlabel("working volume (geometrically similar vessels)")
    ax.set_ylabel("parameter relative to the 5 L bench reference")
    ax.grid(axis="y", lw=0.5, alpha=0.6, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Line2D([], [], color=c, lw=1.7, marker=m, ms=5.0, mfc="white",
                      mec=c, mew=1.2, label="{:s}  ({:s})".format(n, u))
               for n, _k, c, m, u in SERIES]
    handles.append(Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2.4)),
                          label="scale-up criterion, ratio = 1"))
    ax.legend(handles=handles, loc="lower left", fontsize=7.0, frameon=False,
              bbox_to_anchor=(0.015, 0.020), handlelength=1.8,
              borderaxespad=0.0, labelspacing=0.34)
    ax.set_title("Four scales, one degree of freedom", fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def contour_panel(ax, tab):
    i = 2                                   # the 500 L pilot vessel
    rpm = np.linspace(40.0, 260.0, 220)
    vvm = np.linspace(0.10, 1.00, 200)
    rr, vv = np.meshgrid(rpm, vvm)
    pv = power_per_volume(rr / 60.0, tab["d"][i], V_WORK[i])
    area = np.pi / 4.0 * tab["t"][i] ** 2
    vs = vv * V_WORK[i] / 60.0 / area
    kla = kla_hour(pv, vs)

    levels = np.array([20, 40, 60, 90, 130, 180, 240, 320])
    ax.contourf(rr, vv, kla, levels=levels,
                colors=[lighten(BLUE, f) for f in
                        np.linspace(0.94, 0.42, levels.size - 1)],
                extend="both", zorder=1)
    cs = ax.contour(rr, vv, kla, levels=levels, colors=[BLUE], linewidths=0.8,
                    zorder=3)
    ax.clabel(cs, inline=True, fmt="%.0f", fontsize=6.2, colors=[INK])

    ax.plot([tab["rpm"][i]], [VVM], marker="*", ms=11.0, color=OCHRE,
            mec="white", mew=0.7, zorder=6)
    ax.annotate("pilot set point\n{:.0f} rpm, {:.2f} vvm\n".format(
        tab["rpm"][i], VVM) + r"kLa = {:.0f} h$^{{-1}}$".format(
        tab["kla"][i]), xy=(tab["rpm"][i], VVM), xycoords="data",
        xytext=(0.035, 0.965), textcoords="axes fraction", ha="left",
        va="top", fontsize=6.4, color=OCHRE, linespacing=1.32, zorder=7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=HAIR, linewidth=0.6, alpha=0.94),
        arrowprops=dict(arrowstyle="-", color=OCHRE, lw=0.7))

    ax.set_xlim(rpm[0], rpm[-1])
    ax.set_ylim(vvm[0], vvm[-1])
    ax.set_xlabel("agitation speed (rpm)", fontsize=7.6)
    ax.set_ylabel("aeration rate (vvm)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.set_title(r"kLa (h$^{-1}$) at the 500 L pilot scale", fontsize=8.4,
                 fontweight="bold", color=INK)


def power_law_panel(ax, tab):
    v, th = V_WORK, tab["theta"]
    k, b = np.polyfit(np.log10(v), np.log10(th), 1)
    grid = np.logspace(np.log10(v[0]) - 0.35, np.log10(v[-1]) + 0.35, 120)
    ax.plot(grid, 10.0 ** b * grid ** k, color=GREEN, lw=1.5, zorder=3)
    ax.plot(v, th, ls="none", marker="^", ms=6.0, mfc="white", mec=GREEN,
            mew=1.3, zorder=5)
    for vi, ti, nm in zip(v, th, ["0.5 L", "5 L", "500 L", r"10 m$^3$"]):
        ax.text(vi, ti * 0.80, nm, ha="center", va="top", fontsize=6.2,
                color=SUB)

    resid = np.log10(th) - (k * np.log10(v) + b)
    r2 = 1.0 - resid.var() / np.log10(th).var()
    ax.text(0.035, 0.955,
            r"$\theta_{95}$ = " + "{:.2f}".format(10.0 ** b) +
            r" $V^{" + "{:.3f}".format(k) + r"}$" + "  (V in " +
            r"m$^3$, $\theta$ in s)" + "\nfitted exponent {:.3f} vs "
            "{:.3f} expected for\ngeometric similarity at constant "
            "P/V".format(k, 2.0 / 9.0) + "\n" + r"$R^2$ = " +
            "{:.4f}".format(r2), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color=INK, linespacing=1.42)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(th[0] * 0.42, th[-1] * 3.6)
    ax.set_xlabel(r"working volume V (m$^3$)", fontsize=7.6)
    ax.set_ylabel(r"$\theta_{95}$ (s)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.5, alpha=0.55, which="both")
    ax.set_title("Mixing time: a power law in volume", fontsize=8.4,
                 fontweight="bold", color=INK)
    return k, r2


def main():
    set_pub_style()
    tab = scale_table()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_cs = axes_inches(fig, TOP)
    ax_pl = axes_inches(fig, BOT)

    main_panel(ax_main, tab)
    contour_panel(ax_cs, tab)
    k, _r2 = power_law_panel(ax_pl, tab)

    panel_labels_inches(fig, [ax_main, ax_cs, ax_pl], dx=-0.62, dy=0.14)
    savefig(fig, "879_bpr_bioreactor_scaleup.png")


if __name__ == "__main__":
    main()
