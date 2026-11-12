"""Hot-channel thermal hydraulics along the active height of a PWR core.

The three stacked panels share one axial ruler so the reader can drop a
vertical line through them: coolant and clad temperature, surface heat flux
against the departure-from-nucleate-boiling limit, and the void fraction that
follows.  The minimum-DNBR elevation is found in the plotted arrays and marked
in all three, and every quoted number - the exit temperature, the minimum
DNBR, the exit void - is read back from those arrays.  The two smaller panels
on the right carry the cross-stream evidence: subchannel exit temperatures in
the bundle, and the radial temperature drop through the hottest pin.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # coolant
RED = "#b23a48"         # clad, limits, the DNBR emphasis
STEEL = "#8a8a8a"       # structure
HEAT_CMAP = get_cmap("cb_sequential_warm")

HEIGHT = 3.66           # active height, m
T_IN = 292.0            # inlet temperature, degC
T_SAT = 345.0           # saturation at 15.5 MPa, degC
PRESSURE = 15.5         # MPa
Q_AVG = 0.58            # channel-average surface heat flux, MW/m2
DNBR_LIMIT = 1.30
H_CONV = 0.019          # MW/(m2 K), forced-convection film coefficient
VOID_COEF = 0.0062      # void fraction accumulated per K m of wall superheat
T_MELT = 2840.0         # UO2 melting point, degC
C_CLAD, C_GAP, C_FUEL = 36.0, 560.0, 1400.0   # K per (MW/m2) across each layer
N_ROD = 5
PITCH = 12.6            # rod pitch, mm
R_CLAD = 4.75           # clad outer radius, mm


def axial(n=401):
    z = np.linspace(0.0, HEIGHT, n)
    u = (z - HEIGHT / 2.0) / (HEIGHT / 2.0)
    shape = np.cos(0.5 * np.pi * u / 1.15) * (1.0 + 0.22 * np.exp(
        -((u + 0.35) / 0.55) ** 2))
    shape /= shape.mean()
    q = Q_AVG * shape                                   # MW/m2

    rise = np.concatenate([[0.0], np.cumsum(np.diff(z) * 0.5 *
                                            (q[1:] + q[:-1]))])
    t_cool = T_IN + (T_SAT - 20.0 - T_IN) * rise / rise[-1]
    t_clad = t_cool + q / H_CONV
    chf = 2.45 - 0.85 * (z / HEIGHT) - 0.35 * (z / HEIGHT) ** 2   # MW/m2
    dnbr = chf / q

    sup = np.clip(t_clad - T_SAT, 0.0, None)          # wall superheat, K
    void = VOID_COEF * np.concatenate(
        [[0.0], np.cumsum(np.diff(z) * 0.5 * (sup[1:] + sup[:-1]))])
    return z, q, t_cool, t_clad, chf, dnbr, void


def subchannel_map(seed=4):
    n = N_ROD + 1
    i = np.arange(n) - (n - 1) / 2.0
    x, y = np.meshgrid(i, i)
    rng = np.random.default_rng(seed)
    r = np.hypot(x, y) / np.hypot(i[0], i[0])
    t = T_SAT - 18.0 + 16.0 * np.exp(-(r / 0.62) ** 2)
    t += 2.4 * np.exp(-(((x - 0.9) ** 2 + (y - 0.5) ** 2) / 1.6))
    t += rng.normal(0.0, 0.35, t.shape)
    return t


def pin_profile(t_clad_hot, q_hot):
    """Radial temperature through pellet, gap and clad at the hot elevation."""
    r_pel, r_gap = 4.10, 4.18
    r = np.linspace(0.0, R_CLAD, 260)
    dt_clad = q_hot * C_CLAD
    dt_gap = q_hot * C_GAP
    dt_fuel = q_hot * C_FUEL
    t = np.empty_like(r)
    inner = r <= r_pel
    t[inner] = (t_clad_hot + dt_clad + dt_gap + dt_fuel *
                (1.0 - (r[inner] / r_pel) ** 2))
    gap = (r > r_pel) & (r <= r_gap)
    t[gap] = (t_clad_hot + dt_clad + dt_gap *
              (r_gap - r[gap]) / (r_gap - r_pel))
    out = r > r_gap
    t[out] = t_clad_hot + dt_clad * (R_CLAD - r[out]) / (R_CLAD - r_gap)
    return r, t, r_pel, r_gap


def temp_panel(ax, z, t_cool, t_clad, z_min):
    ax.plot(z, t_cool, color=BLUE, lw=1.7, zorder=4, label="coolant bulk")
    ax.plot(z, t_clad, color=RED, lw=1.7, zorder=4, label="clad surface")
    ax.axhline(T_SAT, color=GREY, lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    ax.text(0.04, T_SAT + 2.0, f"saturation {T_SAT:.0f} "
            r"$^{\circ}$C at " f"{PRESSURE:.1f} MPa", ha="left", va="bottom",
            fontsize=6.5, color=SUB)
    ax.annotate(f"exit {t_cool[-1]:.0f} " r"$^{\circ}$C",
                xy=(z[-1], t_cool[-1]), xytext=(z[-1] - 0.06, t_cool[-1] - 6),
                ha="right", va="top", fontsize=6.8, color=BLUE)
    ax.set_ylabel(r"temperature ($^{\circ}$C)", fontsize=7.8)
    ax.set_ylim(T_IN - 12.0, t_clad.max() + 26.0)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=2,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_title("Coolant heats up, clad tracks the flux",
                 fontsize=8.6, fontweight="bold", color=INK)


def flux_panel(ax, z, q, chf, dnbr, z_min):
    ax.fill_between(z, q, chf, color=BLUE, alpha=0.12, linewidth=0.0, zorder=2)
    ax.plot(z, chf, color=RULE, lw=1.3, ls=(0, (5, 2.2)), zorder=4,
            label="critical heat flux")
    ax.plot(z, q, color=BLUE, lw=1.7, zorder=4, label=r"surface $q''$")
    k = int(np.argmin(dnbr))
    ax.plot([z[k]], [q[k]], marker="o", ms=4.0, mfc="white", mec=RED, mew=1.2,
            zorder=6)
    ax.annotate(f"MDNBR = {dnbr[k]:.2f} at {z[k]:.2f} m\n"
                f"(limit {DNBR_LIMIT:.2f})",
                xy=(z[k], q[k]), xytext=(z[k] + 0.16, q[k] - 0.30),
                ha="left", va="top", fontsize=6.8, color=RED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_ylabel(r"heat flux (MW/m$^2$)", fontsize=7.8)
    ax.set_ylim(0.0, chf.max() * 1.12)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=2,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_title("Margin to boiling crisis is thinnest above the flux peak",
                 fontsize=8.6, fontweight="bold", color=INK)


def void_panel(ax, z, void, z_min):
    ax.fill_between(z, 0.0, void, color=BLUE, alpha=0.22, linewidth=0.0,
                    zorder=2)
    ax.plot(z, void, color=BLUE, lw=1.7, zorder=4)
    on = z[void > 1.0e-4]
    if on.size:
        ax.axvline(on[0], color=STEEL, lw=0.8, ls=(0, (2.2, 2.0)), zorder=3)
        ax.text(on[0] - 0.06, void.max() * 0.70,
                f"boiling onset {on[0]:.2f} m", ha="right", va="center",
                fontsize=6.5, color=SUB)
    ax.text(z[-1] - 0.04, void[-1], f"exit void {void[-1]:.3f}  ", ha="right",
            va="top", fontsize=6.8, color=BLUE)
    ax.set_ylabel("void fraction (-)", fontsize=7.8)
    ax.set_xlabel("elevation above core inlet (m)", fontsize=7.8)
    ax.set_ylim(0.0, max(void.max() * 1.35, 0.02))
    ax.set_title("Subcooled void only appears in the upper channel",
                 fontsize=8.6, fontweight="bold", color=INK)


def bundle_panel(fig, ax, t):
    n = t.shape[0]
    half = PITCH / 2.0
    ext = [-half, (n - 1) * PITCH + half, -half, (n - 1) * PITCH + half]
    norm = mpl.colors.Normalize(vmin=t.min(), vmax=t.max())
    ax.imshow(t, origin="lower", extent=ext, cmap=HEAT_CMAP, norm=norm,
              interpolation="nearest", zorder=2)
    for i in range(N_ROD):
        for j in range(N_ROD):
            ax.add_patch(Circle(((i + 0.5) * PITCH, (j + 0.5) * PITCH),
                                R_CLAD, facecolor="#e2e2e2",
                                edgecolor="#555555", lw=0.6, zorder=4))
    k = np.unravel_index(np.argmax(t), t.shape)
    ax.add_patch(Rectangle((k[1] * PITCH - half, k[0] * PITCH - half),
                           PITCH, PITCH, facecolor="none", edgecolor=RED,
                           lw=1.4, zorder=6))
    ax.text(ext[0] + 0.4, ext[3] - 1.2,
            f"hot subchannel {t.max():.0f} " r"$^{\circ}$C", ha="left",
            va="top", fontsize=6.5, color=RED, zorder=7)
    ax.set_aspect("equal")
    ax.set_xlabel("bundle x (mm)", fontsize=7.4)
    ax.set_ylabel("bundle y (mm)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"Subchannel exit field, {N_ROD}x{N_ROD} rods",
                 fontsize=8.2, fontweight="bold", color=INK)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=HEAT_CMAP)
    safe_colorbar(fig, sm, ax, label=r"exit temperature ($^{\circ}$C)",
                  width=0.10, pad=0.07, labelsize=6.5)


def pin_panel(ax, r, t, r_pel, r_gap):
    ax.set_xlim(0.0, R_CLAD)
    ax.set_ylim(180.0, t[0] * 1.16)
    ax.axvspan(0.0, r_pel, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.axvspan(r_gap, R_CLAD, facecolor="#f5f5f5", edgecolor="none", zorder=0)
    ax.plot(r, t, color=RED, lw=1.7, zorder=4)

    xtr = ax.get_xaxis_transform()
    ax.text(r_pel / 2.0, 0.05, "pellet", ha="center", va="bottom",
            fontsize=6.5, color=SUB, transform=xtr)
    ax.text((r_gap + R_CLAD) / 2.0, 0.05, "clad", ha="center", va="bottom",
            fontsize=6.5, color=SUB, transform=xtr, rotation=90)
    gap_dt = float(t[np.argmin(np.abs(r - r_pel))] -
                   t[np.argmin(np.abs(r - r_gap))])
    ax.text(R_CLAD - 0.12, t[0] * 1.13,
            f"centreline {t[0]:.0f} " r"$^{\circ}$C" "\n"
            f"margin to melt {T_MELT - t[0]:.0f} K\n"
            f"gap drop {gap_dt:.0f} K", ha="right", va="top", fontsize=6.5,
            color=SUB, linespacing=1.45)
    ax.set_xlabel("radius from pin centre (mm)", fontsize=7.4)
    ax.set_ylabel(r"temperature ($^{\circ}$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Hot-pin radial profile", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    z, q, t_cool, t_clad, chf, dnbr, void = axial()
    k = int(np.argmin(dnbr))
    t_bundle = subchannel_map()
    r, t_pin, r_pel, r_gap = pin_profile(t_clad[k], q[k])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0, 1.0], left=0.072,
                          right=0.905, top=0.905, bottom=0.115, wspace=0.36,
                          hspace=0.42)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_v = fig.add_subplot(gs[2, 0], sharex=ax_t)
    ax_b = fig.add_subplot(gs[0:2, 1])
    ax_p = fig.add_subplot(gs[2, 1])

    temp_panel(ax_t, z, t_cool, t_clad, z[k])
    flux_panel(ax_q, z, q, chf, dnbr, z[k])
    void_panel(ax_v, z, void, z[k])
    for ax in (ax_t, ax_q, ax_v):
        ax.axvline(z[k], color=RED, lw=0.9, ls=(0, (1.6, 1.8)), zorder=3)
        ax.set_xlim(0.0, HEIGHT)
        ax.grid(axis="y", lw=0.5, alpha=0.7)
        ax.tick_params(labelsize=7.0)
    for ax in (ax_t, ax_q):
        ax.tick_params(labelbottom=False)

    bundle_panel(fig, ax_b, t_bundle)
    pin_panel(ax_p, r, t_pin, r_pel, r_gap)

    panel_labels_inches(fig, [ax_t, ax_q, ax_v, ax_b, ax_p], dx=-0.44,
                        dy=0.10)
    savefig(fig, "656_nuc_thermal_hydraulics.png")


if __name__ == "__main__":
    main()
