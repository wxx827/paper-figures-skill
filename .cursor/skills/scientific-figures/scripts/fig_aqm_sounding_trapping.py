"""A morning sounding, the lapse rate it implies, and the pollution it traps.

The main panel is the thermodynamic profile with height on the vertical axis:
the environmental temperature, the dew point, and the path a surface parcel
takes if it is lifted - dry adiabatic until it saturates at the lifting
condensation level, moist adiabatic above it.  The two shaded areas are the
integrals that matter, the negative area the parcel has to be pushed through
and the positive area it gains once it is through, and the numbers printed in
the panel are those integrals rather than quoted values.

The companion panels are the same profile read two other ways.  The lapse rate
sorts every layer into inversion, stable or unstable air against the dry
adiabat, which is what fixes the mixing height, and the concentration panel
shows the consequence: the identical column burden of fine particles spread
through a shallow trapped layer and through a deep afternoon one, with the
surface concentrations that follow from conserving that burden.

Illustrative synthetic sounding, not a launched radiosonde.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
BROWN = "#8c5a3c"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.72, 1.02, 5.40, 4.20)
AUX_T_RECT = (7.50, 3.42, 3.14, 1.80)
AUX_B_RECT = (7.50, 1.02, 3.14, 1.80)

Z_TOP = 4.0                 # km, top of the plotted sounding
GAMMA_D = 9.8               # K/km, dry adiabatic lapse rate
GAMMA_M = 6.0               # K/km, representative moist adiabatic rate
GAMMA_TD = 1.8              # K/km, dew-point lapse of a lifted parcel
T_SFC = 9.0                 # deg C, environmental surface temperature
T_PARCEL = 13.0             # deg C, heated surface parcel
TD_PARCEL = 8.5             # deg C, its dew point
INV_TOP = 0.30              # km, top of the nocturnal inversion
INV_RATE = 6.0              # K/km, warming through the inversion
FREE_RATE = 8.2             # K/km, cooling above it
H_AFTERNOON = 1.50          # km, afternoon mixing height at the same station
BURDEN_TARGET = 24.0        # mg/m^2, column burden held fixed between cases


def env_temp(z):
    """Environmental temperature, deg C, for height in km."""
    z = np.asarray(z, dtype=float)
    return np.where(z <= INV_TOP, T_SFC + INV_RATE * z,
                    T_SFC + INV_RATE * INV_TOP - FREE_RATE * (z - INV_TOP))


def env_dew(z):
    """Environmental dew point, deg C, with a dry layer aloft."""
    z = np.asarray(z, dtype=float)
    return T_SFC - 3.0 - 3.4 * z - 4.5 * np.exp(-0.5 * ((z - 2.1) / 0.55) ** 2)


def parcel_path(z):
    """Lifted-parcel temperature, deg C, and the LCL height in km."""
    z_lcl = (T_PARCEL - TD_PARCEL) / (GAMMA_D - GAMMA_TD)
    t_lcl = T_PARCEL - GAMMA_D * z_lcl
    z = np.asarray(z, dtype=float)
    return np.where(z <= z_lcl, T_PARCEL - GAMMA_D * z,
                    t_lcl - GAMMA_M * (z - z_lcl)), z_lcl


def energies(z, tp, te):
    """CAPE and CIN in J/kg, plus the LFC height, from the plotted curves.

    The level of free convection is taken above the *highest* negatively
    buoyant level, so the shallow positive blip a heated surface parcel has
    inside the first few metres does not masquerade as the LFC.
    """
    buoy = 9.81 * (tp - te) / (te + 273.15)
    dz = np.gradient(z) * 1.0e3
    neg = np.flatnonzero(buoy < 0.0)
    k = int(neg.max()) + 1 if neg.size else 0
    z_lfc = float(z[min(k, z.size - 1)])
    cape = float((np.where(buoy > 0.0, buoy, 0.0) * dz)[z >= z_lfc].sum())
    cin = float((np.where(buoy < 0.0, buoy, 0.0) * dz)[z <= z_lfc].sum())
    return cape, cin, z_lfc


def pm_profile(z, h_mix, burden):
    """Well-mixed layer capped by a thin transition, scaled to the burden."""
    shape = 1.0 / (1.0 + np.exp((z - h_mix) / 0.045)) + 0.035
    col = np.trapezoid(shape, z * 1.0e3)          # (shape units) x m
    return shape * burden * 1.0e3 / col           # ug/m^3 for burden in mg/m^2


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1013)

    z = np.linspace(0.0, Z_TOP, 801)

    def wiggle(amp):
        """Smooth low-order perturbation, so the lapse rate stays readable."""
        out = np.zeros_like(z)
        for k in range(1, 5):
            out += (amp / k) * np.sin(2.0 * np.pi * k * z / Z_TOP
                                      + rng.uniform(0.0, 2.0 * np.pi))
        return out

    te = env_temp(z) + wiggle(0.45)
    td = env_dew(z) + wiggle(0.60)
    tp, z_lcl = parcel_path(z)
    cape, cin, z_lfc = energies(z, tp, te)

    theta = (te + 273.15) + GAMMA_D * z
    dtheta = np.gradient(theta, z)
    h_mix = float(z[np.argmax(dtheta[z > 0.05] < 3.0) + np.argmax(z > 0.05)])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_c, ax_d):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the sounding -------------------------------------------------
    ax.fill_betweenx(z, te, tp, where=(tp < te) & (z <= z_lfc), color="#f3dfe2",
                     zorder=2)
    ax.fill_betweenx(z, te, tp, where=(tp > te) & (z >= z_lfc),
                     color="#dfeae2", zorder=2)
    ax.plot(te, z, color=RED, lw=2.0, zorder=6, label="temperature")
    ax.plot(td, z, color=GREEN, lw=2.0, ls=(0, (5.0, 2.2)), zorder=6,
            label="dew point")
    ax.plot(tp, z, color=INK, lw=1.3, ls=(0, (2.4, 1.8)), zorder=7,
            label="lifted surface parcel")
    ax.axhspan(0.0, INV_TOP, color="#f6ece3", zorder=1)

    for zz, lab, col in ((z_lcl, "LCL", BROWN), (z_lfc, "LFC", INK),
                         (h_mix, "mixing height", GOLD)):
        ax.axhline(zz, color=col, lw=0.9, ls=(0, (3.4, 2.2)), zorder=4)
        ax.text(29.4, zz + 0.035, f"{lab}, {zz * 1e3:.0f} m", ha="right",
                va="bottom", fontsize=6.5, color=col)
    ax.text(26.0, INV_TOP / 2.0, "nocturnal inversion", ha="right",
            va="center", fontsize=6.6, color=BROWN)

    ax.set_xlim(-20.0, 30.0)
    ax.set_ylim(0.0, Z_TOP)
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("temperature ($^\\circ$C)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("height above ground (km)", fontsize=8.4)
    ax.set_title("Morning sounding with a surface-based inversion",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(0.985, 0.660,
            f"CIN = {cin:.0f} J/kg, the pink area the parcel\nmust be pushed "
            f"through\nCAPE = {cape:.0f} J/kg to {Z_TOP:.0f} km, the green "
            f"area\nabove the LFC\ninversion strength "
            f"{env_temp(INV_TOP) - T_SFC:+.1f} K over the lowest "
            f"{INV_TOP * 1e3:.0f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=12,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    ax.legend(loc="lower left", frameon=False, fontsize=6.6,
              handlelength=1.9, labelspacing=0.26, borderaxespad=0.40)

    # ---- aux (b): lapse rate against the dry adiabat ------------------------
    lapse = np.gradient(te, z)
    ax_c.axvspan(0.0, 30.0, color="#f7e4e6", zorder=1)
    ax_c.axvspan(-GAMMA_D, 0.0, color="#f6f0dd", zorder=1)
    ax_c.axvspan(-30.0, -GAMMA_D, color="#e4f0e8", zorder=1)
    ax_c.plot(lapse, z, color=INK, lw=1.6, zorder=6)
    ax_c.axvline(0.0, color=RED, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax_c.axvline(-GAMMA_D, color=GREEN, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax_c.axhline(h_mix, color=GOLD, lw=1.0, ls=(0, (3.4, 2.2)), zorder=5)
    ax_c.set_xlim(-16.0, 12.0)
    ax_c.set_ylim(0.0, Z_TOP)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("lapse rate dT/dz (K/km)", fontsize=7.5, labelpad=1.5)
    ax_c.set_ylabel("height (km)", fontsize=7.5)
    ax_c.set_title("Every layer, sorted by stability", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.text(1.2, 3.72, "inversion", ha="left", va="top", fontsize=6.2,
              color=RED)
    ax_c.text(-1.2, 3.72, "stable", ha="right", va="top", fontsize=6.2,
              color=BROWN)
    ax_c.text(-15.4, 3.72, "unstable", ha="left", va="top", fontsize=6.2,
              color=GREEN)
    ax_c.text(11.4, h_mix + 0.09, f"mixing height {h_mix * 1e3:.0f} m",
              ha="right", va="bottom", fontsize=6.3, color=GOLD)
    ax_c.text(11.4, 3.30,
              f"free air sits near {-FREE_RATE:.1f} K/km, short of\nthe "
              f"{-GAMMA_D:.1f} K/km dry adiabat, so nothing\nabove the "
              "inversion mixes on its own",
              ha="right", va="top", fontsize=6.2, color=SUB,
              linespacing=1.45)

    # ---- aux (c): the same burden in two mixing layers ----------------------
    zc = np.linspace(0.0, 2.0, 601)
    pm_trap = pm_profile(zc, h_mix, BURDEN_TARGET)
    pm_mix = pm_profile(zc, H_AFTERNOON, BURDEN_TARGET)
    b_trap = np.trapezoid(pm_trap, zc * 1e3) / 1e3
    b_mix = np.trapezoid(pm_mix, zc * 1e3) / 1e3
    ax_d.fill_betweenx(zc, 0.0, pm_trap, color="#f3dfe2", zorder=2)
    ax_d.plot(pm_trap, zc, color=RED, lw=1.9, zorder=6,
              label=f"trapped, {h_mix * 1e3:.0f} m")
    ax_d.plot(pm_mix, zc, color=GREEN, lw=1.9, ls=(0, (4.6, 2.0)), zorder=6,
              label=f"mixed, {H_AFTERNOON * 1e3:.0f} m")
    for arr, col, y_lab, ha, dx in ((pm_trap, RED, 0.10, "right", -2.5),
                                    (pm_mix, GREEN, 0.62, "left", 3.0)):
        ax_d.plot([arr[0]], [0.0], marker="o", ms=4.2, mfc="white", mec=col,
                  mew=1.3, zorder=8)
        ax_d.text(arr[0] + dx, y_lab, f"{arr[0]:.0f} $\\mu g/m^3$ at the "
                  "ground", ha=ha, va="bottom", fontsize=6.4, color=col)
    ax_d.set_xlim(0.0, pm_trap.max() * 1.32)
    ax_d.set_ylim(0.0, 2.0)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("$PM_{2.5}$ concentration ($\\mu g/m^3$)", fontsize=7.5,
                    labelpad=1.5)
    ax_d.set_ylabel("height (km)", fontsize=7.5)
    ax_d.set_title("One burden, two surface readings", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.text(0.975, 0.955,
              f"both columns carry {b_trap:.1f} and {b_mix:.1f} $mg/m^2$;\n"
              f"the shallow layer reads "
              f"{pm_trap[0] / pm_mix[0]:.1f} times higher at the ground",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_d.legend(handles=[Patch(facecolor=RED, edgecolor="white",
                               label=f"trapped, {h_mix * 1e3:.0f} m"),
                         Patch(facecolor=GREEN, edgecolor="white",
                               label=f"mixed, {H_AFTERNOON * 1e3:.0f} m")],
                loc="center right", frameon=False, fontsize=6.3,
                handlelength=1.1, handleheight=0.9, labelspacing=0.26,
                borderaxespad=0.55)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.56, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=1013")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "1013_aqm_sounding_trapping.png")


if __name__ == "__main__":
    main()
