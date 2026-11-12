"""Compressor characteristic map with the engine working line laid over it.

The main panel is the classic corrected-flow / pressure-ratio map: a family of
corrected speed lines, isentropic-efficiency islands, the surge line that caps
them and the steady-state working line that must keep clear of it.  The surge
margin quoted on the page is measured from the two curves that are drawn.  The
subordinate column shows what the same engine delivers to the airframe: net
thrust and thrust-specific fuel consumption against Mach number.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from pubstyle import set_pub_style, savefig, get_cmap, add_colorbar
from figkit import panel_labels_inches, halo_text

C_WORK = "#3b5b92"     # house blue: working line, nominal operation
C_SURGE = "#b23a48"    # house red: surge boundary
C_SPEED = "#555555"    # neutral: constant-speed family
C_GREEN = "#2e7d5b"
C_GOLD = "#c9a227"
C_PURPLE = "#8e6bbf"
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

SPEEDS = np.array([0.60, 0.70, 0.78, 0.85, 0.90, 0.95, 1.00, 1.05])
M_REF = 62.0           # choked corrected mass flow at 100 % speed (kg/s)
PR_REF = 22.0          # pressure-ratio scale
F_SL = 121.0           # installed sea-level static thrust (kN)
SFC_REF = 0.0360       # reference TSFC (kg/(N.h))
M_CRUISE = 0.80
R_GAS, GAMMA = 287.053, 1.4


def isa(h_km):
    h = np.asarray(h_km, dtype=float) * 1000.0
    t = np.where(h < 11000.0, 288.15 - 0.0065 * h, 216.65)
    p = np.where(h < 11000.0,
                 101325.0 * (np.maximum(t, 1.0) / 288.15) ** 5.2559,
                 22632.1 * np.exp(-9.80665 * (h - 11000.0)
                                  / (R_GAS * 216.65)))
    return t, p, p / (R_GAS * t)


def speed_line(n, t):
    """Corrected flow (kg/s) and pressure ratio along one constant-speed line.

    ``t`` runs from 0 at the surge end to 1 at the choked end; the pressure
    ratio is held almost flat near surge and falls away steeply into choke,
    which is what gives the family its characteristic knee.
    """
    m_choke = M_REF * n ** 1.05
    m_surge = m_choke * (0.70 + 0.22 * n)
    pr_surge = 1.0 + PR_REF * n ** 2.5
    pr_choke = 1.0 + PR_REF * n ** 2.5 * 0.62
    m = m_surge + (m_choke - m_surge) * t
    pr = pr_surge - (pr_surge - pr_choke) * t ** 1.8
    return m, pr


def efficiency(m, pr):
    """Isentropic efficiency as a ridge running along the working line."""
    xn, yn = (m - 47.0) / M_REF, (pr - 14.0) / PR_REF
    phi = np.deg2rad(46.0)
    u = xn * np.cos(phi) + yn * np.sin(phi)
    v = -xn * np.sin(phi) + yn * np.cos(phi)
    return 0.893 * np.exp(-0.5 * ((u / 0.44) ** 2 + (v / 0.118) ** 2))


def rasterize(artist):
    """3.10 mis-reports ContourSet rasterization as ignored; it is not."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def main():
    set_pub_style(base_font=9)

    t_par = np.linspace(0.0, 1.0, 240)
    lines = [speed_line(n, t_par) for n in SPEEDS]
    m_surge = np.array([ln[0][0] for ln in lines])
    pr_surge = np.array([ln[1][0] for ln in lines])
    m_choke = np.array([ln[0][-1] for ln in lines])
    pr_choke = np.array([ln[1][-1] for ln in lines])

    t_work = 0.55 - 0.12 * SPEEDS
    work = np.array([speed_line(n, t) for n, t in zip(SPEEDS, t_work)])
    m_work, pr_work = work[:, 0], work[:, 1]
    pr_surge_at_work = np.interp(m_work, m_surge, pr_surge)
    margin = 100.0 * (pr_surge_at_work / pr_work - 1.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.56],
                          height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ===================== (a) compressor characteristic ===================
    boundary = np.vstack([
        np.column_stack([m_surge, pr_surge]),
        np.column_stack([lines[-1][0], lines[-1][1]]),
        np.column_stack([m_choke[::-1], pr_choke[::-1]]),
        np.column_stack([lines[0][0][::-1], lines[0][1][::-1]])])
    inside = Path(boundary)

    gm = np.linspace(15.0, 68.0, 320)
    gp = np.linspace(1.0, 26.0, 320)
    MG, PG = np.meshgrid(gm, gp)
    eta = efficiency(MG, PG)
    ok = inside.contains_points(
        np.column_stack([MG.ravel(), PG.ravel()])).reshape(MG.shape)
    eta = np.where(ok, eta, np.nan)

    levels = [0.60, 0.68, 0.74, 0.79, 0.83, 0.86, 0.878]
    cf = ax.contourf(MG, PG, eta, levels=levels + [0.90],
                     cmap=get_cmap("cb_sequential_blue"), alpha=0.72,
                     extend="min", zorder=2)
    rasterize(cf)
    cf.set_edgecolor("face")
    cs = ax.contour(MG, PG, eta, levels=levels, colors="#1f4e73",
                    linewidths=0.55, zorder=3)
    ax.clabel(cs, [0.68], inline=True, fontsize=6.0, fmt="%.2f")
    ax.plot([47.0], [14.0], marker="x", ms=5.0, mew=1.2, color="#0d3352",
            zorder=6)
    # The label sits on filled contours, so it needs a stroke to stay legible.
    halo_text(ax, 50.2, 12.2, f"peak $\\eta_c$ = {efficiency(47.0, 14.0):.3f}",
              halo="#0d3352", fontsize=6.4, ha="center", va="center",
              zorder=6)

    for n, (m, pr) in zip(SPEEDS, lines):
        ax.plot(m, pr, color=C_SPEED, lw=0.9, zorder=4)
        ax.text(m[-1] + 0.7, pr[-1] - 0.20, f"{n * 100:.0f}", fontsize=6.2,
                color=C_SPEED, ha="left", va="top", zorder=6)
    ax.text(m_choke[-1] + 0.7, pr_choke[-1] + 3.4,
            "$N/\\sqrt{\\theta}$ (%)", fontsize=6.6, color=C_SPEED,
            ha="left", va="bottom", zorder=6)

    ax.plot(m_surge, pr_surge, color=C_SURGE, lw=2.4, zorder=7)

    ax.plot(m_work, pr_work, color=C_WORK, lw=1.8, zorder=8)
    ax.plot(m_work, pr_work, marker="o", ms=4.0, ls="none", mfc="white",
            mec=C_WORK, mew=1.2, zorder=9)
    i_c = int(np.argmin(np.abs(SPEEDS - 0.95)))
    ax.plot(m_work[i_c], pr_work[i_c], marker="o", ms=7.0, mfc="white",
            mec=C_GOLD, mew=1.8, zorder=10)
    ax.annotate(f"cruise point\n{m_work[i_c]:.1f} kg/s,  $\\Pi_c$ = "
                f"{pr_work[i_c]:.1f}\nsurge margin {margin[i_c]:.1f} %",
                xy=(m_work[i_c], pr_work[i_c]),
                xytext=(m_work[i_c] - 15.5, pr_work[i_c] + 4.8), fontsize=7.0,
                color="#8a6f18", ha="left", va="center", zorder=11,
                arrowprops=dict(arrowstyle="-|>", color=C_GOLD, lw=0.9,
                                shrinkA=1.0, shrinkB=5.0,
                                connectionstyle="arc3,rad=-0.22"))
    ax.annotate("", xy=(m_work[i_c], pr_surge_at_work[i_c]),
                xytext=(m_work[i_c], pr_work[i_c]),
                arrowprops=dict(arrowstyle="<|-|>", color=C_SURGE, lw=1.0,
                                shrinkA=0.0, shrinkB=0.0))

    ax.set_xlim(15.0, 70.0)
    ax.set_ylim(1.0, 26.0)
    ax.set_xlabel("corrected mass flow $\\dot{m}\\sqrt{\\theta}/\\delta$ "
                  "(kg/s)")
    ax.set_ylabel("total pressure ratio $\\Pi_c$ (-)")
    ax.set_title("Compressor map, working line and surge margin",
                 fontsize=9.0, fontweight="bold", pad=4)
    ax.grid(True, lw=0.5, alpha=0.35)
    handles = [Line2D([], [], color=C_SURGE, lw=2.4, label="surge line"),
               Line2D([], [], color=C_WORK, lw=1.8, marker="o", ms=4.0,
                      mfc="white", mec=C_WORK, label="steady working line"),
               Line2D([], [], color=C_SPEED, lw=0.9,
                      label="constant corrected speed")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.9, frameon=False,
              borderaxespad=0.5)
    # aspect 22 on a full-height panel gives a 0.21 in bar; 30 lands inside
    # the 0.13-0.18 in house range (DESIGN_SYSTEM section 6.2).
    add_colorbar(fig, cf, ax, label="isentropic efficiency $\\eta_c$ (-)",
                 aspect=30)

    # ===================== (b) net thrust (subordinate) ====================
    mach = np.linspace(0.0, 0.92, 220)
    ram = 1.0 - 0.28 * mach + 0.30 * mach ** 2
    alts = ((0.0, C_GREEN), (6.0, C_WORK), (11.0, C_PURPLE))
    for h_i, col in alts:
        _, _, rho = isa(h_i)
        thrust = F_SL * (rho / 1.225) ** 0.85 * ram
        ax_t.plot(mach, thrust, color=col, lw=1.4, label=f"{h_i:.0f} km")
        if h_i == 11.0:
            f_cr = float(np.interp(M_CRUISE, mach, thrust))
            ax_t.plot(M_CRUISE, f_cr, marker="o", ms=5.0, mfc="white",
                      mec=C_GOLD, mew=1.5, zorder=6)
            ax_t.annotate(f"cruise {f_cr:.1f} kN", xy=(M_CRUISE, f_cr),
                          xytext=(0.60, f_cr + 26.0), fontsize=6.5,
                          color="#8a6f18", ha="center", va="bottom",
                          arrowprops=dict(arrowstyle="-|>", color=C_GOLD,
                                          lw=0.8, shrinkA=1.0, shrinkB=4.0))
    ax_t.set_xlim(0.0, 0.92)
    ax_t.set_ylim(0.0, 132.0)
    ax_t.set_xlabel("Mach number $M$ (-)", fontsize=7.6)
    ax_t.set_ylabel("net thrust $F_n$ (kN)", fontsize=7.6)
    ax_t.tick_params(labelsize=6.5)
    ax_t.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=3,
                columnspacing=1.0, handlelength=1.3, borderaxespad=0.3)
    ax_t.set_title("Thrust lapse with altitude and Mach", fontsize=8.5,
                   fontweight="bold", pad=3)
    ax_t.grid(True, lw=0.5, alpha=0.5)

    # ===================== (c) specific fuel consumption ===================
    for h_i, col in alts:
        t_i, _, _ = isa(h_i)
        sfc = SFC_REF * (1.0 + 0.62 * mach) * np.sqrt(t_i / 288.15)
        ax_s.plot(mach, sfc, color=col, lw=1.4, label=f"{h_i:.0f} km")
        if h_i == 11.0:
            s_cr = float(np.interp(M_CRUISE, mach, sfc))
            ax_s.plot(M_CRUISE, s_cr, marker="o", ms=5.0, mfc="white",
                      mec=C_GOLD, mew=1.5, zorder=6)
            ax_s.annotate(f"cruise {s_cr:.4f} kg/(N$\\cdot$h)",
                          xy=(M_CRUISE, s_cr), xytext=(0.90, 0.0305),
                          fontsize=6.5, color="#8a6f18", ha="right",
                          va="bottom",
                          arrowprops=dict(arrowstyle="-|>", color=C_GOLD,
                                          lw=0.8, shrinkA=1.0, shrinkB=4.0))
    ax_s.set_xlim(0.0, 0.92)
    ax_s.set_ylim(0.028, 0.059)
    ax_s.set_xlabel("Mach number $M$ (-)", fontsize=7.6)
    ax_s.set_ylabel("TSFC (kg/(N$\\cdot$h))", fontsize=7.6)
    ax_s.tick_params(labelsize=6.5)
    ax_s.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=3,
                columnspacing=1.0, handlelength=1.3, borderaxespad=0.2)
    ax_s.set_title("Cold, thin air buys the lowest fuel burn", fontsize=8.5,
                   fontweight="bold", pad=3)
    ax_s.grid(True, lw=0.5, alpha=0.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=2.2, w_pad=2.4)
    panel_labels_inches(fig, [ax, ax_t, ax_s], dx=-0.42, dy=0.06)
    savefig(fig, "507_aero_propulsion_map.png")


if __name__ == "__main__":
    main()
