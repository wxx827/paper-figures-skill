"""Operating envelope of a transport aircraft in the altitude-Mach plane.

One point-performance model (ISA atmosphere, a transonic drag polar and a
lapsing installed thrust) generates every curve on the page, so the shaded
usable envelope, the four limit lines that bound it, the manoeuvre V-n diagram
and the specific excess power curves cannot disagree with one another.  Each
boundary binds over a different altitude band: buffet at the left, dynamic
pressure low down, flutter in the middle, thrust at the ceiling.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

C_MAIN = "#3b5b92"     # house blue: usable envelope, thrust limit
C_LIM = "#b23a48"      # house red: structural and flutter limits
C_BUF = "#2e7d5b"      # house green: buffet / stall boundary
C_ALT = "#c9a227"      # house gold: cruise point, second load case
C_PUR = "#8e6bbf"      # house purple: third altitude in panel (d)
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

W_S = 5800.0           # wing loading (N/m2)
T_S = 1750.0           # installed sea-level thrust per wing area (N/m2)
CL_MAX = 1.55          # clean-wing buffet-onset lift coefficient
CD0 = 0.0200
K_IND = 1.0 / (np.pi * 8.4 * 0.82)
Q_MAX = 38.0e3         # design dynamic-pressure limit (Pa)
N_MAX, N_MIN = 2.5, -1.0
R_GAS, GAMMA = 287.053, 1.4


def isa(h_km):
    """ISA temperature (K), pressure (Pa), density (kg/m3), sound speed (m/s)."""
    h = np.asarray(h_km, dtype=float) * 1000.0
    t = np.where(h < 11000.0, 288.15 - 0.0065 * h, 216.65)
    p = np.where(h < 11000.0,
                 101325.0 * (np.maximum(t, 1.0) / 288.15) ** 5.2559,
                 22632.1 * np.exp(-9.80665 * (h - 11000.0)
                                  / (R_GAS * 216.65)))
    return t, p, p / (R_GAS * t), np.sqrt(GAMMA * R_GAS * t)


def cd0_of_mach(m):
    """Zero-lift drag with a transonic rise centred just below Mach 0.9."""
    return CD0 + 0.045 / (1.0 + np.exp(-(m - 0.865) / 0.022))


def excess_thrust(m, h_km):
    """Installed thrust minus drag, per unit wing area (N/m2)."""
    _, p, rho, _ = isa(h_km)
    q = 0.5 * GAMMA * p * m ** 2
    drag = q * cd0_of_mach(m) + K_IND * W_S ** 2 / np.maximum(q, 1.0)
    sigma = rho / 1.225
    thrust = T_S * sigma ** 0.85 * (1.0 - 0.28 * m + 0.30 * m ** 2)
    return thrust - drag


def main():
    set_pub_style(base_font=9)

    h = np.linspace(0.0, 16.0, 400)
    _, p_h, rho_h, a_h = isa(h)

    m_buffet = np.sqrt(2.0 * W_S / (rho_h * a_h ** 2 * CL_MAX))
    m_struct = np.sqrt(2.0 * Q_MAX / (GAMMA * p_h))
    m_flutter = 0.795 + 0.0080 * h

    m_grid = np.linspace(0.10, 1.05, 500)
    m_thrust = np.full_like(h, np.nan)
    for i, hi in enumerate(h):
        ok = excess_thrust(m_grid, hi) >= 0.0
        if ok.any():
            m_thrust[i] = m_grid[ok].max()

    # np.min, not np.nanmin: above the ceiling the thrust boundary does not
    # exist, and the envelope has to end there rather than fall back to the
    # limits that are still finite.
    m_hi = np.min(np.vstack([m_struct, m_flutter, m_thrust]), axis=0)
    m_lo = m_buffet
    usable = np.isfinite(m_hi) & (m_hi > m_lo)
    h_ceiling = float(h[usable].max())
    i_wide = int(np.argmax(np.where(usable, m_hi - m_lo, -1.0)))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.57],
                          height_ratios=[1.0, 0.88], hspace=0.40, wspace=0.24)
    ax = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ===================== (a) altitude-Mach envelope ======================
    ax.fill_betweenx(h[usable], m_lo[usable], m_hi[usable], color=C_MAIN,
                     alpha=0.16, lw=0, zorder=2)
    ax.plot(m_buffet, h, color=C_BUF, lw=1.6, zorder=5,
            label="buffet onset, $C_{L,max}$ = %.2f" % CL_MAX)
    ax.plot(m_struct, h, color=C_LIM, lw=1.6, zorder=5,
            label="structural limit, $q$ = %.0f kPa" % (Q_MAX / 1e3))
    ax.plot(m_flutter, h, color=C_LIM, lw=1.4, ls=(0, (6, 2, 1, 2)), zorder=5,
            label="flutter boundary")
    ax.plot(m_thrust, h, color=C_MAIN, lw=1.5, ls=(0, (5, 2.6)), zorder=5,
            label="thrust available = drag")

    ax.plot(0.80, 11.6, marker="o", ms=6.0, mfc="white", mec=C_LIM, mew=1.5,
            zorder=8)
    ax.annotate("design point\nMach 0.80,  11.6 km", xy=(0.80, 11.6),
                xytext=(0.455, 13.4), fontsize=7.0, color=C_LIM,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_LIM, lw=0.8,
                                shrinkA=1.0, shrinkB=4.0,
                                connectionstyle="arc3,rad=-0.20"))
    ax.plot(0.775, 10.4, marker="D", ms=5.0, mfc="white", mec=C_ALT, mew=1.5,
            zorder=8)
    ax.annotate("long-range cruise\nMach 0.78,  10.4 km", xy=(0.775, 10.4),
                xytext=(0.90, 8.4), fontsize=7.0, color="#8a6f18",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_ALT, lw=0.8,
                                shrinkA=1.0, shrinkB=4.0,
                                connectionstyle="arc3,rad=0.22"))

    ax.axhline(h_ceiling, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=3)
    ax.text(0.118, h_ceiling + 0.14,
            f"absolute ceiling {h_ceiling:.1f} km", fontsize=6.8,
            color=MUTED, ha="left", va="bottom")
    ax.annotate("", xy=(m_lo[i_wide], h[i_wide]), xytext=(m_hi[i_wide],
                                                          h[i_wide]),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.8,
                                shrinkA=0.0, shrinkB=0.0))
    ax.text(0.5 * (m_lo[i_wide] + m_hi[i_wide]), h[i_wide] - 0.42,
            f"widest speed band $\\Delta M$ = {m_hi[i_wide] - m_lo[i_wide]:.2f}"
            f" at {h[i_wide]:.1f} km", fontsize=6.8, color=SUB, ha="center",
            va="top")
    ax.text(0.61, 3.0, "usable\nenvelope", fontsize=8.2, color="#2c4570",
            fontweight="bold", ha="center", va="center", zorder=6)

    ax.set_xlim(0.10, 1.06)
    ax.set_ylim(0.0, 15.5)
    ax.set_xlabel("Mach number $M$ (-)")
    ax.set_ylabel("pressure altitude $h$ (km)")
    ax.set_title("Usable envelope and its four binding limits", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.grid(True, lw=0.5, alpha=0.5)
    # The strip left of the buffet curve is empty at every altitude, which is
    # the only place a four-entry legend fits without covering data.
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.70), fontsize=6.9,
              frameon=False, labelspacing=0.5, handlelength=2.2)

    # ===================== (b) V-n manoeuvre envelope ======================
    rho0 = 1.225
    v_s = np.sqrt(2.0 * W_S / (rho0 * CL_MAX))
    v_a = v_s * np.sqrt(N_MAX)
    v_sn = v_s * 1.20                       # inverted-flight stall speed
    v_g = v_sn * np.sqrt(abs(N_MIN))
    v_d = np.sqrt(2.0 * Q_MAX / rho0)       # dive speed set by the same q limit
    v_c = 0.82 * v_d

    vv = np.linspace(0.0, v_a, 120)
    ax_v.plot(vv, (vv / v_s) ** 2, color=C_BUF, lw=1.4)
    vn = np.linspace(0.0, v_g, 120)
    ax_v.plot(vn, -(vn / v_sn) ** 2, color=C_BUF, lw=1.4)
    ax_v.plot([v_a, v_d], [N_MAX, N_MAX], color=C_LIM, lw=1.5)
    ax_v.plot([v_g, v_c], [N_MIN, N_MIN], color=C_LIM, lw=1.5)
    ax_v.plot([v_c, v_d], [N_MIN, 0.0], color=C_LIM, lw=1.5)
    ax_v.plot([v_d, v_d], [0.0, N_MAX], color=C_LIM, lw=1.5)
    ax_v.fill(np.concatenate([vv, [v_d, v_d, v_c], vn[::-1]]),
              np.concatenate([(vv / v_s) ** 2, [N_MAX, 0.0, N_MIN],
                              -(vn[::-1] / v_sn) ** 2]),
              color=C_MAIN, alpha=0.14, lw=0)

    for ude, style in ((15.2, (0, (5, 2.2))), (7.6, (0, (1.6, 1.8)))):
        slope = ude / 2400.0
        ax_v.plot([0.0, v_d], [1.0, 1.0 + slope * v_d], color=MUTED, lw=0.8,
                  ls=style)
        ax_v.plot([0.0, v_d], [1.0, 1.0 - slope * v_d], color=MUTED, lw=0.8,
                  ls=style)
    ax_v.text(v_d * 0.16, 3.30, "gust lines, $U_{de}$ = 7.6 / 15.2 m/s",
              fontsize=6.2, color=MUTED, ha="left", va="top")

    ax_v.axhline(0.0, color=RULE, lw=0.7)
    for vx, lab in ((v_s, "$V_S$"), (v_a, "$V_A$"), (v_c, "$V_C$"),
                    (v_d, "$V_D$")):
        ax_v.plot([vx, vx], [-1.36, -1.24], color=SUB, lw=0.8)
        ax_v.text(vx, -1.80, f"{lab} {vx:.0f}", fontsize=6.2, color=SUB,
                  ha="center", va="bottom")
    ax_v.text(v_a + 6.0, N_MAX + 0.10, f"$n_{{max}}$ = +{N_MAX:.1f} g",
              fontsize=6.5, color=C_LIM, ha="left", va="bottom")
    ax_v.text(v_c - 6.0, N_MIN - 0.12, f"$n_{{min}}$ = {N_MIN:.1f} g",
              fontsize=6.5, color=C_LIM, ha="right", va="top")

    ax_v.set_xlim(0.0, v_d * 1.12)
    ax_v.set_ylim(-1.95, 3.55)
    ax_v.set_xticks([0, 50, 100, 150, 200, 250])
    ax_v.set_xlabel("equivalent airspeed (m/s)", fontsize=7.6)
    ax_v.set_ylabel("load factor $n$ (g)", fontsize=7.6)
    ax_v.tick_params(labelsize=6.5)
    ax_v.set_title("Manoeuvre V-n envelope, sea level", fontsize=8.5,
                   fontweight="bold", pad=3)

    # ===================== (c) specific excess power =======================
    m_ps = np.linspace(0.20, 1.02, 300)
    for h_i, col in ((0.0, C_BUF), (6.0, C_MAIN), (11.6, C_PUR)):
        _, _, _, a_i = isa(h_i)
        ps = m_ps * a_i * excess_thrust(m_ps, h_i) / W_S
        ax_p.plot(m_ps, ps, color=col, lw=1.4, label=f"{h_i:.1f} km")
        j = int(np.argmax(ps))
        ax_p.plot(m_ps[j], ps[j], marker="o", ms=3.4, color=col, zorder=6)
    ax_p.axhline(0.0, color=RULE, lw=0.8)
    ax_p.axvline(0.80, color=C_LIM, lw=0.9, ls=(0, (4, 2.4)))
    ax_p.text(0.785, 47.0, "design Mach", fontsize=6.4, color=C_LIM,
              ha="right", va="top")
    ax_p.set_xlim(0.20, 1.02)
    ax_p.set_ylim(-95.0, 52.0)
    ax_p.set_xlabel("Mach number $M$ (-)", fontsize=7.6)
    ax_p.set_ylabel("specific excess power (m/s)", fontsize=7.6)
    ax_p.tick_params(labelsize=6.5)
    ax_p.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=3,
                columnspacing=1.0, handlelength=1.3, borderaxespad=0.2)
    ax_p.set_title("Excess power lapses to zero at the ceiling", fontsize=8.5,
                   fontweight="bold", pad=3)
    ax_p.grid(True, lw=0.5, alpha=0.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=2.2, w_pad=2.4)
    panel_labels_inches(fig, [ax, ax_v, ax_p], dx=-0.40, dy=0.06)
    savefig(fig, "505_aero_flight_envelope.png")


if __name__ == "__main__":
    main()
