"""Smoke layer descent and thermal stratification in an open-plan compartment.

The main panel is a time-height section of the gas temperature for the design
fire, with the smoke layer interface of three growth rates drawn over it.  The
horizontal rule at 1.8 m is the tenability limit for a standing occupant, so
where each interface crosses it is the available safe egress time (ASET) - the
budget every evacuation calculation has to fit inside.  A small equal-aspect
room section in the reserved band above the ceiling says what the geometry is.

The physics is integrated, not sketched: a Zukoski plume entrains
0.071 Qc^(1/3) z^(5/3) kg/s into the upper layer, the layer volume is filled by
Euler integration, the layer temperature comes from a convective energy balance
and the visibility profile from soot yield over the layer volume.  ASET and the
detector activation times are read back out of those arrays by interpolation.
Illustrative zone-model output, not a CFD result or a test.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon

from pubstyle import set_pub_style, savefig, despine, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BASE = "#808080"
SMOKE = "#8a8a8a"

ROOM_L, ROOM_W, ROOM_H = 12.0, 9.0, 4.0     # m
FLOOR_A = ROOM_L * ROOM_W                    # m2
T_AMB = 20.0                                 # deg C
CP = 1.0                                     # kJ/(kg K)
CHI_C = 0.70                                 # convective fraction of the HRR
CHI_LOSS = 0.50                              # fraction lost to the boundaries
RHO_U = 1.10                                 # kg/m3, hot layer density
TENABLE_Z = 1.8                              # m, standing occupant head height
T_END = 420.0                                # s
Y_TOP = 5.5                                  # m; band above the ceiling
DET_T = 57.0                                 # deg C, heat detector rating
DET_R = 3.0                                  # m, radius from the plume axis
HC = 20.0                                    # MJ/kg, effective heat of combustion
SOOT_Y = 0.07                                # kg soot per kg fuel
K_M = 8700.0                                 # m2/kg, mass optical density
K_SIGN = 3.0                                 # reflecting-sign constant
# A little of the layer always mixes down, so the lower zone is not optically
# empty; without it the "clear layer" visibility is infinite and the profile
# degenerates into a line pinned against the axis limit.
MIX_FRAC = 0.02
VIS_CRIT = 10.0                              # m, tenability criterion

SCEN = [
    dict(name="slow", alpha=0.00293, color=BLUE),
    dict(name="medium (design fire)", alpha=0.01172, color=INK),
    dict(name="fast", alpha=0.04690, color=GREEN),
]


def fill_model(alpha, t):
    """Euler-integrate the two-zone filling model; returns interface and T_u.

    The layer temperature is a running energy balance over the *whole* layer
    (convective energy in, minus boundary loss, over the mass entrained so
    far).  Dividing the instantaneous HRR by the instantaneous plume flow
    instead would send T_u to infinity as the interface reaches the floor and
    the plume shortens to nothing.
    """
    z = np.empty_like(t)
    t_u = np.empty_like(t)
    zi, e_c = ROOM_H, 0.0
    for i, ti in enumerate(t):
        q_c = CHI_C * alpha * ti * ti                      # kW
        m_p = (0.071 * max(q_c, 1e-6) ** (1.0 / 3.0)
               * max(zi, 0.05) ** (5.0 / 3.0))             # kg/s
        m_layer = RHO_U * FLOOR_A * (ROOM_H - zi)          # kg
        z[i] = zi
        t_u[i] = T_AMB + (1.0 - CHI_LOSS) * e_c / (CP * max(m_layer, 5.0))
        if i + 1 < t.size:
            dt = t[i + 1] - ti
            zi = max(zi - (m_p / (RHO_U * FLOOR_A)) * dt, 0.0)
            e_c += q_c * dt
    return z, t_u


def temperature_field(t, z_grid, z_iface, t_u, delta=0.18):
    """Two-layer field smoothed across the interface, deg C."""
    frac = 1.0 / (1.0 + np.exp(-(z_grid[:, None] - z_iface[None, :]) / delta))
    return T_AMB + (t_u[None, :] - T_AMB) * frac


def aset_of(t, z):
    """Time the interface falls through the tenability height, by interp."""
    return float(np.interp(-TENABLE_Z, -z, t))


def ceiling_jet(alpha, t):
    """Alpert's ceiling-jet temperature at DET_R from the plume axis, deg C."""
    q = alpha * t * t
    return T_AMB + 5.38 * np.power(np.maximum(q, 1e-9) / DET_R,
                                   2.0 / 3.0) / ROOM_H


def visibility(alpha, t, z_iface, z_grid, i_t, delta=0.18):
    """Visibility profile at one instant, from integrated soot over the layer."""
    e_mj = np.trapezoid(alpha * t[:i_t + 1] ** 2, t[:i_t + 1]) / 1000.0
    soot = SOOT_Y * e_mj / HC                                   # kg
    v_u = max(FLOOR_A * (ROOM_H - z_iface[i_t]), 1.0)           # m3
    c_u = soot / v_u                                            # kg/m3
    frac = 1.0 / (1.0 + np.exp(-(z_grid - z_iface[i_t]) / delta))
    c = c_u * (MIX_FRAC + (1.0 - MIX_FRAC) * frac)
    return K_SIGN / (K_M * c), c_u


def room_section(ax, z_iface_final):
    """Equal-aspect schematic of the compartment in the free band up top."""
    axi = ax.inset_axes([0.615, 0.795, 0.375, 0.185])
    axi.set_xlim(-0.4, ROOM_L + 0.4)
    axi.set_ylim(-0.35, ROOM_H + 0.35)
    axi.set_aspect("equal")
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.patch.set_facecolor("white")
    axi.axis("off")

    axi.add_patch(Rectangle((0.0, 0.0), ROOM_L, ROOM_H, facecolor="white",
                            edgecolor=RULE, lw=0.9, zorder=2))
    axi.add_patch(Rectangle((0.0, z_iface_final), ROOM_L,
                            ROOM_H - z_iface_final, facecolor=SMOKE,
                            alpha=0.55, edgecolor="none", zorder=3))
    axi.plot([0.0, ROOM_L], [z_iface_final] * 2, color=RULE, lw=0.9, zorder=4)
    axi.plot([0.0, ROOM_L], [TENABLE_Z] * 2, color=RED, lw=0.9,
             ls=(0, (3.0, 1.8)), zorder=5)
    axi.add_patch(Polygon([[ROOM_L / 2 - 0.55, 0.0], [ROOM_L / 2 + 0.55, 0.0],
                           [ROOM_L / 2, 1.15]], facecolor="#cc722f",
                          edgecolor="none", zorder=4))
    axi.add_patch(Rectangle((ROOM_L - 1.3, 0.0), 0.9, 2.1, facecolor="white",
                            edgecolor=RULE, lw=0.7, zorder=5))
    axi.annotate("", xy=(1.6, z_iface_final + 0.15), xytext=(1.6, ROOM_H - 0.2),
                 arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.8,
                                 mutation_scale=7), zorder=6)
    axi.text(ROOM_L / 2, ROOM_H + 0.42, f"section {ROOM_L:.0f} m x "
                                        f"{ROOM_H:.1f} m (equal aspect)",
             ha="center", va="bottom", fontsize=6.0, color=SUB, zorder=6)
    axi.text(2.4, (z_iface_final + ROOM_H) / 2.0, "smoke layer", ha="left",
             va="center", fontsize=6.0, color=INK, zorder=6)
    axi.text(ROOM_L - 1.6, 1.05, "door", ha="right", va="center",
             fontsize=6.0, color=SUB, zorder=6)
    return axi


def main_panel(fig, ax, t, z_grid, runs):
    design = runs[1]
    field = temperature_field(t, z_grid, design["z"], design["t_u"])
    vmax = float(np.ceil(field.max() / 50.0) * 50.0)
    mesh = ax.pcolormesh(t, z_grid, field, cmap=get_cmap("cb_sequential_warm"),
                         vmin=T_AMB, vmax=vmax, shading="gouraud", zorder=1)
    safe_colorbar(fig, mesh, ax, label="gas temperature ($^{\\circ}$C)",
                  width=0.12, pad=0.09, shrink=0.72)

    ax.axhline(TENABLE_Z, color=RED, lw=1.2, ls=(0, (4.5, 2.2)), zorder=5)
    tag = ax.text(T_END - 8.0, TENABLE_Z + 0.09,
                  f"tenability limit {TENABLE_Z:.1f} m", ha="right",
                  va="bottom", fontsize=6.6, color=RED, zorder=7)
    tag.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    handles, labels = [], []
    for r in runs:
        ln, = ax.plot(t, r["z"], color=r["color"], lw=1.9, zorder=6)
        ln.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        ax.plot([r["aset"]], [TENABLE_Z], marker="o", ms=5.0, mfc="white",
                mec=RED, mew=1.3, zorder=8)
        handles.append(ln)
        labels.append(f"{r['name']} ($\\alpha$ = {r['alpha']:.4f} "
                      f"kW/s$^{{2}}$): ASET {r['aset']:.0f} s")

    # Every curve is below 1.8 m by the time the slowest one crosses, so the
    # upper-right corner of the field is guaranteed clear for this call-out.
    order = sorted(runs, key=lambda r: r["aset"])
    ax.annotate("ASET, interface at "
                + f"{TENABLE_Z:.1f} m\n"
                + " / ".join(f"{r['aset']:.0f} s" for r in order)
                + "  (" + " / ".join(r["name"].split()[0] for r in order)
                + ")",
                xy=(order[-1]["aset"], TENABLE_Z),
                xytext=(T_END * 0.62, 3.35), ha="left", va="center",
                fontsize=6.6, color=RED, linespacing=1.4, zorder=9,
                bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=3.0, shrinkB=3.0))

    ax.plot([0.0, T_END], [ROOM_H] * 2, color=RULE, lw=1.0, zorder=5)
    ax.text(6.0, ROOM_H + 0.06, f"ceiling {ROOM_H:.1f} m", ha="left",
            va="bottom", fontsize=6.5, color=SUB)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(np.arange(0.0, ROOM_H + 0.1, 1.0))
    ax.set_xticks(np.arange(0.0, T_END + 1.0, 100.0))
    ax.set_xlabel("time from ignition (s)", fontsize=8.0)
    ax.set_ylabel("height above floor (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    despine(ax)
    ax.set_title("The smoke layer reaches head height long before the fire "
                 "peaks", fontsize=9.0, fontweight="bold", color=INK)
    leg = ax.legend(handles, labels, fontsize=6.7, frameon=False, ncol=1,
                    loc="upper left", bbox_to_anchor=(0.0, 1.0),
                    handlelength=1.9, labelspacing=0.36)
    leg.set_zorder(10)
    return vmax


def jet_panel(ax, t, runs):
    acts = []
    for r in runs:
        ax.plot(t, r["jet"], color=r["color"], lw=1.6, zorder=5)
        ok = r["jet"] >= DET_T
        t_act = float(np.interp(DET_T, r["jet"], t)) if ok.any() else np.nan
        acts.append(t_act)
        if np.isfinite(t_act):
            ax.plot([t_act], [DET_T], marker="s", ms=3.8, mfc="white",
                    mec=RED, mew=1.1, zorder=7)
    ax.axhline(DET_T, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=4)
    ax.text(0.02, DET_T + 4.0, f"detector rating {DET_T:.0f} "
                               f"$^{{\\circ}}$C",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6.3, color=RED)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 320.0)
    ax.set_xticks(np.arange(0.0, T_END + 1.0, 100.0))
    ax.set_xlabel("time from ignition (s)", fontsize=7.6)
    ax.set_ylabel("ceiling jet temperature ($^{\\circ}$C)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    fin = [a for a in acts if np.isfinite(a)]
    ax.text(0.5, 0.035, f"activation {min(fin):.0f}-{max(fin):.0f} s at "
                        f"r = {DET_R:.0f} m (Alpert)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4,
            color=SUB)
    ax.set_title("Detection is fast; the layer is faster", fontsize=8.4,
                 fontweight="bold", color=INK)
    return acts


def vis_panel(ax, t, z_grid, design):
    aset = design["aset"]
    times = [0.5 * aset, aset, 1.6 * aset]
    shades = ["#9cc3dd", "#2e74a8", "#08263f"]
    conc, curves = [], []
    for tt, col in zip(times, shades):
        i_t = int(np.searchsorted(t, tt))
        vis, c_u = visibility(design["alpha"], t, design["z"], z_grid, i_t)
        conc.append(c_u)
        curves.append(vis)
        ax.plot(vis, z_grid, color=col, lw=1.6, zorder=5,
                label=f"t = {t[i_t]:.0f} s")
        z_i = design["z"][i_t]
        ax.plot([np.interp(z_i, z_grid, vis)], [z_i], marker="o", ms=3.4,
                mfc="white", mec=col, mew=1.0, zorder=6)
    ax.axvline(VIS_CRIT, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=4)
    ax.axhline(TENABLE_Z, color=RED, lw=0.9, ls=(0, (2.0, 1.8)), zorder=4)
    ax.text(VIS_CRIT * 1.16, 0.10, f"{VIS_CRIT:.0f} m criterion", ha="left",
            va="bottom", fontsize=6.3, color=RED, rotation=90)
    ax.text(0.985, TENABLE_Z + 0.06, f"{TENABLE_Z:.1f} m",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.3, color=RED)
    lo = min(float(np.min(v)) for v in curves)
    hi = max(float(np.max(v)) for v in curves)
    ax.set_xscale("log")
    ax.set_xlim(lo * 0.55, hi * 1.9)
    ax.set_ylim(0.0, ROOM_H)
    ax.set_xlabel("visibility to a reflecting sign (m)", fontsize=7.6)
    ax.set_ylabel("height above floor (m)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.6, labelspacing=0.3)
    ax.set_title("Visibility collapses across the interface", fontsize=8.4,
                 fontweight="bold", color=INK)
    return conc


def main():
    set_pub_style()
    rng = np.random.default_rng(85302)
    t = np.linspace(0.0, T_END, 1201)
    z_grid = np.linspace(0.0, ROOM_H, 181)

    runs = []
    for s in SCEN:
        # A small scenario-to-scenario spread in the growth coefficient, so
        # nothing on the figure is a round number typed by hand.
        alpha = s["alpha"] * (1.0 + 0.03 * rng.normal())
        z, t_u = fill_model(alpha, t)
        runs.append(dict(s, alpha=alpha, z=z, t_u=t_u, aset=aset_of(t, z),
                         jet=ceiling_jet(alpha, t)))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.86, 4.86, 4.52))
    ax_jet = axes_inches(fig, (7.30, 3.40, 2.72, 1.98))
    ax_vis = axes_inches(fig, (7.30, 0.86, 2.72, 1.98))

    vmax = main_panel(fig, ax_main, t, z_grid, runs)
    # Draw the section a third of the way past ASET, so the layer is visibly
    # below the tenability rule rather than sitting exactly on it.
    z_sec = float(np.interp(1.33 * runs[1]["aset"], t, runs[1]["z"]))
    room_section(ax_main, z_sec)
    acts = jet_panel(ax_jet, t, runs)
    conc = vis_panel(ax_vis, t, z_grid, runs[1])

    design = runs[1]
    margin = design["aset"] - acts[1]
    verdict = (f"{margin:.0f} s of warning before the layer arrives"
               if margin > 0 else
               f"{-margin:.0f} s after the layer has already passed head "
               f"height")
    panel_labels_inches(fig, [ax_main, ax_jet, ax_vis], dx=-0.52, dy=0.14)
    savefig(fig, "853_fir_smoke_layer.png")


if __name__ == "__main__":
    main()
